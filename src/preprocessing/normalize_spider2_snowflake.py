import argparse
import csv
import glob
import json
import os
import shutil
from typing import Any, Dict, List

import snowflake.connector
from snowflake.connector import ProgrammingError


def load_credentials(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def fetch_columns_for_db(creds: Dict[str, Any], database: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Return mapping: schema_lower -> table_lower -> list of column dicts in ordinal order.
    Each column dict has keys: table_catalog, table_schema, table_name, column_name, data_type.
    """
    conn = snowflake.connector.connect(**creds, database=database)
    try:
        cur = conn.cursor()
        try:
            # Ensure session database is set (defensive even if provided on connect)
            safe_db = database.replace('"', '""')
            cur.execute(f'USE DATABASE "{safe_db}"')
            cur.execute(
                """
                SELECT
                    TABLE_CATALOG,
                    TABLE_SCHEMA,
                    TABLE_NAME,
                    COLUMN_NAME,
                    DATA_TYPE,
                    ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS
                ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
                """
            )
            columns: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for row in cur.fetchall():
                (
                    table_catalog,
                    table_schema,
                    table_name,
                    column_name,
                    data_type,
                    _ordinal_position,
                ) = row
                schema_key = table_schema.lower()
                table_key = table_name.lower()
                table_entry = columns.setdefault(schema_key, {}).setdefault(table_key, [])
                table_entry.append(
                    {
                        "table_catalog": table_catalog,
                        "table_schema": table_schema,
                        "table_name": table_name,
                        "column_name": column_name,
                        "data_type": data_type,
                    }
                )
            return columns
        finally:
            cur.close()
    finally:
        conn.close()


def remap_by_column(
    values: List[Any], old_column_names: List[str], new_columns: List[str], default_value: Any = None
) -> List[Any]:
    """
    Reorder a list (e.g., descriptions) to align with new_columns using old_column_names for matching.
    """
    lookup = {name.lower(): idx for idx, name in enumerate(old_column_names)}
    reordered = []
    for col in new_columns:
        idx = lookup.get(col.lower())
        reordered.append(values[idx] if idx is not None and idx < len(values) else default_value)
    return reordered


def remap_sample_rows(sample_rows: List[Dict[str, Any]], new_columns: List[str]) -> List[Dict[str, Any]]:
    remapped = []
    for row in sample_rows or []:
        lower_row = {k.lower(): v for k, v in row.items()}
        remapped.append({col: lower_row.get(col.lower()) for col in new_columns})
    return remapped


def normalize_table_metadata(
    file_path: str,
    output_file_path: str,
    db_name: str,
    schema_dir: str,
    table_file: str,
    column_map: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> bool:
    table_key = table_file.lower()
    schema_key = schema_dir.lower()
    with open(file_path, "r") as f:
        metadata = json.load(f)

    schema_tables = column_map.get(schema_key, {})
    if table_key not in schema_tables:
        print(f"[WARN] Missing columns in Snowflake for {db_name}.{schema_dir}.{table_file}")
        with open(output_file_path, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        return False

    snow_cols = schema_tables[table_key]
    new_column_names = [c["column_name"] for c in snow_cols]
    new_column_types = [c["data_type"] for c in snow_cols]
    table_catalog = snow_cols[0]["table_catalog"]
    table_schema = snow_cols[0]["table_schema"]
    table_name = snow_cols[0]["table_name"]

    old_column_names = metadata.get("column_names", [])
    descriptions = metadata.get("description", [])
    metadata["description"] = remap_by_column(descriptions, old_column_names, new_column_names, None)

    metadata["column_names"] = new_column_names
    metadata["column_types"] = new_column_types
    metadata["table_name"] = f"{table_schema}.{table_name}"
    metadata["table_fullname"] = f"{table_catalog}.{table_schema}.{table_name}"
    metadata["sample_rows"] = remap_sample_rows(metadata.get("sample_rows", []), new_column_names)

    with open(output_file_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return True


def read_schema_table_names_from_ddl(ddl_path: str) -> set:
    if not os.path.isfile(ddl_path):
        return set()
    names = set()
    with open(ddl_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("table_name") or "").strip()
            if name:
                names.add(name)
    return names


def normalize_schema_ddl(
    ddl_path: str,
    out_ddl_path: str,
    db_name: str,
    schema_name: str,
    column_map: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> set:
    """
    Write DDL.csv to output with normalized table_name as fully-qualified
    <catalog>.<schema>.<table> (using Snowflake casing when available).
    Returns the set of normalized table names (lowercased) for mismatch checks.
    """
    if not os.path.isfile(ddl_path):
        return set()

    # Build lookup from table lower -> (catalog, schema, table)
    schema_key = schema_name.lower()
    table_lookup: Dict[str, str] = {}
    for table_key, cols in column_map.get(schema_key, {}).items():
        if cols:
            tcat = cols[0]["table_catalog"]
            tschema = cols[0]["table_schema"]
            tname = cols[0]["table_name"]
            table_lookup[table_key] = f"{tcat}.{tschema}.{tname}"

    normalized_names = set()
    rows_out = []
    with open(ddl_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["table_name", "description", "DDL"]
        for row in reader:
            raw_name = (row.get("table_name") or "").strip()
            table_token = raw_name.split(".")[-1] if raw_name else ""
            lookup_key = table_token.lower() if table_token else raw_name.lower()

            if lookup_key in table_lookup:
                normalized = table_lookup[lookup_key]
            else:
                # Fallback: build from folder names; keep raw table part if present
                table_part = table_token or raw_name
                normalized = f"{db_name}.{schema_name}.{table_part}"

            row["table_name"] = normalized
            normalized_names.add(normalized.lower())
            rows_out.append(row)

    os.makedirs(os.path.dirname(out_ddl_path), exist_ok=True)
    with open(out_ddl_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return normalized_names


def normalize_database(db_path: str, out_db_path: str, db_name: str, creds: Dict[str, Any]) -> None:
    print(f"[INFO] Normalizing database {db_name} -> {out_db_path}")
    try:
        column_map = fetch_columns_for_db(creds, db_name)
    except ProgrammingError as e:
        print(f"[ERROR] Skipping database {db_name}: {e}")
        if os.path.isdir(out_db_path):
            shutil.rmtree(out_db_path, ignore_errors=True)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Unexpected failure for database {db_name}: {e}")
        if os.path.isdir(out_db_path):
            shutil.rmtree(out_db_path, ignore_errors=True)
        return

    total = 0
    updated = 0
    for schema_dir in sorted(next(os.walk(db_path))[1]):
        schema_path = os.path.join(db_path, schema_dir)
        out_schema_path = os.path.join(out_db_path, schema_dir)
        os.makedirs(out_schema_path, exist_ok=True)

        ddl_path = os.path.join(schema_path, "DDL.csv")
        out_ddl_path = os.path.join(out_schema_path, "DDL.csv")
        ddl_table_names = normalize_schema_ddl(ddl_path, out_ddl_path, db_name, schema_dir, column_map)

        for json_file in glob.glob(os.path.join(schema_path, "*.json")):
            table_file = os.path.splitext(os.path.basename(json_file))[0]
            out_file = os.path.join(out_schema_path, os.path.basename(json_file))
            if ddl_table_names:
                # warn if table missing in DDL
                table_file_lower = table_file.lower()
                schema_key = schema_dir.lower()
                table_lookup = column_map.get(schema_key, {})
                if table_file_lower in table_lookup and table_lookup[table_file_lower]:
                    first_col = table_lookup[table_file_lower][0]
                    full_name = f"{first_col['table_catalog']}.{first_col['table_schema']}.{first_col['table_name']}"
                else:
                    full_name = f"{db_name}.{schema_dir}.{table_file}"
                if full_name.lower() not in ddl_table_names:
                    print(f"[WARN] DDL.csv missing table '{table_file}' in {db_name}.{schema_dir}")
            total += 1
            try:
                if normalize_table_metadata(json_file, out_file, db_name, schema_dir, table_file, column_map):
                    updated += 1
            except Exception as e:  # noqa: BLE001
                # log and remove partial output
                print(f"[ERROR] Skipping table {db_name}.{schema_dir}.{table_file}: {e}")
                if os.path.isfile(out_file):
                    os.remove(out_file)
    print(f"[INFO] {db_name}: updated {updated}/{total} tables (copied rest unchanged)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Spider2 Snowflake metadata to actual identifier casing."
    )
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument(
        "--metadata-root",
        default=os.path.join(project_root, "datasets", "Spider2", "spider2-snow", "resource", "databases"),
        help="Root folder containing database metadata directories.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(project_root, "datasets", "Spider2", "spider2-snow", "resource", "databases_normalized"),
        help="Destination root for normalized metadata (will mirror database/schema structure).",
    )
    parser.add_argument(
        "--credential-path",
        default=os.path.join(project_root, "datasets", "Spider2", "spider2-snow", "snowflake_credential.json"),
        help="Path to Snowflake credential JSON file.",
    )
    parser.add_argument(
        "--databases",
        nargs="*",
        help="Optional list of database directory names to process. Defaults to all in metadata root.",
    )
    args = parser.parse_args()

    metadata_root = os.path.abspath(args.metadata_root)
    output_root = os.path.abspath(args.output_root)
    creds = load_credentials(os.path.abspath(args.credential_path))

    if args.databases:
        db_dirs = [os.path.join(metadata_root, db) for db in args.databases]
    else:
        db_dirs = [os.path.join(metadata_root, d) for d in next(os.walk(metadata_root))[1]]

    os.makedirs(output_root, exist_ok=True)
    for db_dir in sorted(db_dirs):
        if not os.path.isdir(db_dir):
            print(f"[WARN] Skipping non-directory {db_dir}")
            continue
        db_name = os.path.basename(db_dir)
        out_db_dir = os.path.join(output_root, db_name)
        os.makedirs(out_db_dir, exist_ok=True)
        normalize_database(db_dir, out_db_dir, db_name, creds)


if __name__ == "__main__":
    main()
