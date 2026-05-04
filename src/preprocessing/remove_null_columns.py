#!/usr/bin/env python3
"""
Script to remove columns containing only null values or only "nan" string values from Snowflake database metadata.

This script:
1. Queries Snowflake to identify columns that contain only null values or only "nan" string values
2. Removes those columns from JSON metadata files
3. Removes those column definitions from DDL.csv files
4. Outputs to a new directory structure
5. Logs all column removals
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

import snowflake.connector
from sqlglot import parse_one, exp


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remove_null_columns.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_credentials(path: str) -> Dict[str, Any]:
    """Load Snowflake credentials from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def unquote_identifier(identifier: str) -> str:
    """
    Remove double quotes from identifier, handling cases where it might be quoted twice.
    Examples:
    - "column_name" -> column_name
    - ""column_name"" -> column_name
    - column_name -> column_name
    """
    if not identifier:
        return identifier
    # Remove outer quotes if present
    while identifier.startswith('"') and identifier.endswith('"'):
        identifier = identifier[1:-1]
    return identifier


def normalize_column_name(name: str) -> str:
    """Normalize column name for comparison (unquote and lowercase)."""
    return unquote_identifier(name).lower()


def safe_copy_file(src: str, dst: str) -> None:
    """
    Safely copy a file, handling the case where source and destination are the same file.
    If paths are the same, the copy operation is effectively a no-op (overwrites with itself).
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    # Ensure destination directory exists
    dst_dir = os.path.dirname(dst)
    if dst_dir:  # Only create directory if there's a directory component
        os.makedirs(dst_dir, exist_ok=True)
    
    try:
        shutil.copy2(src, dst)
    except shutil.SameFileError:
        # Source and destination are the same file - this is fine, just continue
        # Copying a file to itself is effectively a no-op (overwrites with itself)
        pass


def check_null_only_columns(
    creds: Dict[str, Any],
    table_fullname: str,
    columns: List[str]
) -> Set[str]:
    """
    Check which columns contain only null values or only "nan" string values.
    
    Returns a set of column names (normalized) that are null-only or nan-only.
    """
    null_only = set()
    
    try:
        # Parse table_fullname: database.schema.table
        parts = table_fullname.split(".")
        if len(parts) < 3:
            logger.warning(f"  [WARN] Invalid table_fullname format: {table_fullname}")
            return null_only
        
        database = parts[0]
        schema = parts[1]
        table = parts[2]
        
        conn = snowflake.connector.connect(**creds, database=database)
        try:
            cur = conn.cursor()
            try:
                # Build fully qualified table name with proper quoting
                safe_db = database.replace('"', '""')
                safe_schema = schema.replace('"', '""')
                safe_table = table.replace('"', '""')
                table_fqn = f'"{safe_db}"."{safe_schema}"."{safe_table}"'
                
                # First, check if table has any rows at all
                cur.execute(f'SELECT COUNT(*) FROM {table_fqn}')
                row_count_result = cur.fetchone()
                total_rows = row_count_result[0] if row_count_result else 0
                
                if total_rows == 0:
                    logger.warning(f"  [WARN] Table {table_fullname} is empty (0 rows), skipping null check")
                    return null_only
                
                # Check each column
                for col in columns:
                    try:
                        # Escape column name properly
                        safe_col = col.replace('"', '""')
                        
                        # First check: does column have any non-null values?
                        query = f'SELECT 1 FROM {table_fqn} WHERE "{safe_col}" IS NOT NULL LIMIT 1'
                        cur.execute(query)
                        result = cur.fetchone()
                        
                        if result is None:
                            # No non-null values means column contains only nulls
                            null_only.add(normalize_column_name(col))
                            logger.info(f"  [NULL-ONLY] {table_fullname}.{col}")
                        else:
                            # Column has non-null values, check if they're all "nan"
                            # Check for distinct non-null, non-nan values
                            # Use UPPER() for case-insensitive comparison
                            query = f'''SELECT 1 FROM {table_fqn} 
                                       WHERE "{safe_col}" IS NOT NULL 
                                       AND UPPER(TRIM(CAST("{safe_col}" AS VARCHAR))) != 'NAN'
                                       LIMIT 1'''
                            cur.execute(query)
                            result = cur.fetchone()
                            
                            if result is None:
                                # No non-null, non-nan values found - column contains only nulls and/or "nan"
                                null_only.add(normalize_column_name(col))
                                logger.info(f"  [NAN-ONLY] {table_fullname}.{col} (contains only null/nan values)")
                                
                    except Exception as e:
                        logger.warning(f"  [WARN] Could not check column {col} in {table_fullname}: {e}")
                        # If we can't check, assume it's not null-only
                        continue
                        
            finally:
                cur.close()
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"  [ERROR] Failed to check null columns for {table_fullname}: {e}")
    
    return null_only


def fetch_table_columns(
    creds: Dict[str, Any],
    table_fullname: str
) -> List[Dict[str, Any]]:
    """
    Fetch column information for a specific table from Snowflake using fully qualified table name.
    Returns list of column dicts with column_name, data_type, ordinal_position.
    """
    columns = []
    try:
        # Parse table_fullname: database.schema.table
        parts = table_fullname.split(".")
        if len(parts) < 3:
            logger.warning(f"  [WARN] Invalid table_fullname format: {table_fullname}")
            return columns
        
        database = parts[0]
        schema = parts[1]
        table = parts[2]
        
        conn = snowflake.connector.connect(**creds, database=database)
        try:
            cur = conn.cursor()
            try:
                # Use fully qualified table name directly in INFORMATION_SCHEMA query
                # Escape single quotes for string literals
                safe_schema_str = schema.replace("'", "''")
                safe_table_str = table.replace("'", "''")
                cur.execute(
                    f"""
                    SELECT
                        COLUMN_NAME,
                        DATA_TYPE,
                        ORDINAL_POSITION
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_CATALOG = '{database}' AND TABLE_SCHEMA = '{safe_schema_str}' AND TABLE_NAME = '{safe_table_str}'
                    ORDER BY ORDINAL_POSITION
                    """
                )
                
                for row in cur.fetchall():
                    column_name, data_type, ordinal_position = row
                    columns.append({
                        "column_name": column_name,
                        "data_type": data_type,
                        "ordinal_position": ordinal_position
                    })
            finally:
                cur.close()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"  [WARN] Could not fetch columns for {table_fullname}: {e}")
    
    return columns


def parse_ddl_columns(ddl_text: str) -> List[Tuple[str, str]]:
    """
    Parse DDL text to extract column definitions using sqlglot.
    Returns list of (column_name, column_definition) tuples.
    
    Handles quoted column names (double quotes, possibly double-quoted).
    Based on the approach used in src/get_ddl.py for finding column names.
    """
    columns = []
    
    try:
        # Check for double double-quotes and normalize them if present
        # sqlglot cannot parse ""identifier"" syntax, so we must normalize first
        if '""' in ddl_text:
            # Normalize double-quoted identifiers: ""identifier"" -> "identifier"
            # This handles Snowflake DDL that uses double double-quotes (common in CSV exports)
            # Strategy: Use a simple, explicit pattern that matches column identifiers
            # Pattern matches: tab/whitespace at start of line, then ""identifier"", then space before data type
            
            def normalize_quotes(match):
                prefix = match.group(1)  # whitespace/tab before
                identifier = match.group(2)  # the identifier itself
                suffix = match.group(3)  # whitespace after
                # Replace any "" inside the identifier with " (unescape quotes)
                identifier = identifier.replace('""', '"')
                return f'{prefix}"{identifier}"{suffix}'
            
            # Pattern matches column identifiers in DDL:
            # Match ""identifier"" where identifier can contain "" for escaped quotes
            # Pattern: match "" then a sequence of: (non-quote OR two quotes), then ""
            
            # First pattern: identifiers at start of line (after whitespace/tabs)
            normalized_ddl = re.sub(
                r'^(\s*)""((?:[^"]|"")+)""(\s+)',
                normalize_quotes,
                ddl_text,
                flags=re.MULTILINE
            )
            
            # Second pattern: identifiers after commas
            def normalize_quotes_after_comma(match):
                prefix = match.group(1)  # comma and whitespace before
                identifier = match.group(2)  # the identifier itself
                suffix = match.group(3)  # whitespace after
                # Replace any "" inside the identifier with " (unescape quotes)
                identifier = identifier.replace('""', '"')
                return f'{prefix}"{identifier}"{suffix}'
            
            normalized_ddl = re.sub(
                r'(,\s*)""((?:[^"]|"")+)""(\s+)',
                normalize_quotes_after_comma,
                normalized_ddl,
                flags=re.MULTILINE
            )
            
            # Parse the normalized DDL statement using sqlglot
            parsed = parse_one(normalized_ddl, read="snowflake")
        else:
            # No double double-quotes, parse the DDL as-is
            parsed = parse_one(ddl_text, read="snowflake")
        
        if parsed is None:
            logger.warning("Failed to parse DDL with sqlglot - parsed is None")
            return columns
        
        # Check if it's a CREATE TABLE statement
        if not isinstance(parsed, exp.Create):
            logger.warning(f"Not a Create statement, got: {type(parsed)}")
            return columns
        
        # Find column definitions using find_all - this searches the entire AST
        column_defs = parsed.find_all(exp.ColumnDef)
        
        if column_defs:
            # Extract column name and definition from each ColumnDef
            for expr in column_defs:
                # ColumnDef has 'this' which is the column identifier
                col_name = expr.this.sql(dialect="snowflake")
                # Get the full column definition SQL
                col_def = expr.sql(dialect="snowflake")
                columns.append((col_name, col_def))
        else:
            # Fallback: check parsed.expressions directly
            if parsed.expressions:
                for expr in parsed.expressions:
                    # Skip constraints
                    if isinstance(expr, (exp.PrimaryKey, exp.ForeignKey, exp.Unique, exp.Check, exp.Constraint)):
                        continue
                    
                    if isinstance(expr, exp.ColumnDef):
                        col_name = expr.this.sql(dialect="snowflake")
                        col_def = expr.sql(dialect="snowflake")
                        columns.append((col_name, col_def))
            
            # Also check parsed.this.expressions if it exists (for Schema expressions)
            if parsed.this and hasattr(parsed.this, 'expressions') and parsed.this.expressions:
                for expr in parsed.this.expressions:
                    if isinstance(expr, exp.ColumnDef):
                        col_name = expr.this.sql(dialect="snowflake")
                        col_def = expr.sql(dialect="snowflake")
                        columns.append((col_name, col_def))
        
    except Exception as e:
        # If sqlglot parsing fails, log and return empty list
        # Log a snippet of the DDL for debugging (first 200 chars)
        ddl_snippet = ddl_text[:200].replace('\n', '\\n')
        logger.warning(f"Error parsing DDL with sqlglot: {e}")
        logger.debug(f"DDL snippet (first 200 chars): {ddl_snippet}")
        return columns
    
    return columns


def remove_columns_from_json(
    json_path: str,
    output_path: str,
    columns_to_remove: Set[str],
    table_fullname: str
) -> bool:
    """
    Remove specified columns from JSON metadata file.
    Returns True if any columns were removed.
    """
    with open(json_path, "r") as f:
        metadata = json.load(f)
    
    original_columns = metadata.get("column_names", [])
    if not original_columns:
        # No columns to remove
        safe_copy_file(json_path, output_path)
        return False
    
    # Find indices of columns to remove
    indices_to_remove = []
    columns_removed = []
    
    for idx, col_name in enumerate(original_columns):
        if normalize_column_name(col_name) in columns_to_remove:
            indices_to_remove.append(idx)
            columns_removed.append(col_name)
            logger.info(f"    Removing column '{col_name}' from JSON")
    
    if not indices_to_remove:
        # No columns to remove
        safe_copy_file(json_path, output_path)
        return False
    
    # Remove columns from all relevant fields
    # Remove in reverse order to maintain indices
    for idx in sorted(indices_to_remove, reverse=True):
        if "column_names" in metadata:
            metadata["column_names"].pop(idx)
        if "column_types" in metadata and idx < len(metadata["column_types"]):
            metadata["column_types"].pop(idx)
        if "description" in metadata and idx < len(metadata["description"]):
            metadata["description"].pop(idx)
    
    # Update sample_rows to remove the columns
    if "sample_rows" in metadata:
        for row in metadata["sample_rows"]:
            for col_name in columns_removed:
                # Try both quoted and unquoted versions
                if col_name in row:
                    del row[col_name]
                elif unquote_identifier(col_name) in row:
                    del row[unquote_identifier(col_name)]
    
    # Write updated metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    return True


def remove_columns_from_ddl_row(
    ddl_text: str,
    columns_to_remove: Set[str]
) -> Tuple[str, List[str]]:
    """
    Remove specified columns from a DDL text.
    Returns (new_ddl_text, list_of_removed_column_names).
    """
    if not ddl_text:
        return ddl_text, []
    
    # Parse columns from DDL
    ddl_columns = parse_ddl_columns(ddl_text)
    
    if not ddl_columns:
        return ddl_text, []
    
    # Filter out columns to remove
    filtered_columns = []
    removed_names = []
    
    for col_name, col_def in ddl_columns:
        if normalize_column_name(col_name) not in columns_to_remove:
            filtered_columns.append(col_def)
        else:
            removed_names.append(unquote_identifier(col_name))
            logger.info(f"    Removing column '{col_name}' from DDL")
    
    if not removed_names:
        # No columns removed
        return ddl_text, []
    
    # Rebuild DDL
    # Extract CREATE TABLE statement prefix (everything before the opening parenthesis)
    create_match = re.match(
        r'(CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+[^(]+\()',
        ddl_text,
        re.IGNORECASE | re.DOTALL
    )
    
    if not create_match:
        logger.warning(f"    Could not parse DDL prefix, keeping original")
        return ddl_text, []
    
    prefix = create_match.group(1)
    
    # Join remaining columns with proper formatting
    # Preserve indentation from original
    indent = "    "  # Default indent
    if "\n" in ddl_text:
        # Try to detect indent from first column
        first_col_match = re.search(r'\(\s*\n(\s+)', ddl_text)
        if first_col_match:
            indent = first_col_match.group(1)
    
    # Join columns with comma and newline
    columns_text = ",\n".join([f"{indent}{col}" for col in filtered_columns])
    
    # Rebuild DDL
    new_ddl = f"{prefix}\n{columns_text}\n);"
    
    return new_ddl, removed_names


def remove_columns_from_ddl(
    ddl_path: str,
    output_path: str,
    table_to_columns: Dict[str, Set[str]]
) -> bool:
    """
    Remove specified columns from DDL.csv file.
    
    Args:
        ddl_path: Path to input DDL.csv
        output_path: Path to output DDL.csv
        table_to_columns: Dict mapping table identifiers (normalized) to sets of column names to remove
    
    Returns True if any columns were removed.
    """
    if not os.path.isfile(ddl_path):
        return False
    
    rows_out = []
    total_removed = 0
    
    with open(ddl_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["table_name", "description", "DDL"]
        
        for row in reader:
            row_table = (row.get("table_name") or "").strip()
            ddl_text = row.get("DDL", "")
            
            # Find matching table in table_to_columns
            columns_to_remove = None
            for table_key, cols in table_to_columns.items():
                # Match by table name (could be fully qualified or just table name)
                if (table_key.lower() in row_table.lower() or 
                    row_table.lower() in table_key.lower() or
                    any(part.lower() in row_table.lower() for part in table_key.split(".") if part)):
                    columns_to_remove = cols
                    break
            
            if columns_to_remove and ddl_text:
                # Remove columns from this DDL
                new_ddl, removed = remove_columns_from_ddl_row(ddl_text, columns_to_remove)
                row["DDL"] = new_ddl
                total_removed += len(removed)
            # else: keep row as-is
            
            rows_out.append(row)
    
    # Write updated DDL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    
    return total_removed > 0


def process_table(
    json_path: str,
    output_json_path: str,
    db_name: str,
    schema_dir: str,
    creds: Dict[str, Any]
) -> Tuple[str, Set[str], int, bool]:
    """
    Process a single table to remove null-only columns.
    
    Returns: (table_fullname, null_only_columns_set, columns_removed_count, was_modified)
    """
    table_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Load JSON to get table info
    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"    [ERROR] Failed to load {json_path}: {e}")
        safe_copy_file(json_path, output_json_path)
        return ("", set(), 0, False)
    
    # Get table fullname to determine database.schema.table
    table_fullname = metadata.get("table_fullname", "")
    if not table_fullname:
        # Try to construct from table_name
        table_name_field = metadata.get("table_name", "")
        if "." in table_name_field:
            table_fullname = f"{db_name}.{table_name_field}"
        else:
            table_fullname = f"{db_name}.{schema_dir}.{table_name}"
    
    # Extract table name for logging
    parts = table_fullname.split(".")
    actual_table = parts[-1] if parts else table_name
    
    logger.info(f"    [INFO] Processing table: {actual_table} ({table_fullname})")
    
    # Get columns from JSON
    json_columns = metadata.get("column_names", [])
    if not json_columns:
        logger.warning(f"    [WARN] No columns found in JSON for {table_name}")
        safe_copy_file(json_path, output_json_path)
        return (table_fullname, set(), 0, False)
    
    # Fetch actual columns from Snowflake using table_fullname
    try:
        snowflake_columns = fetch_table_columns(creds, table_fullname)
        if not snowflake_columns:
            logger.warning(f"    [WARN] Could not fetch columns from Snowflake for {table_fullname}")
            # Use JSON columns as fallback
            columns_to_check = json_columns
        else:
            # Use Snowflake column names (they're the source of truth)
            columns_to_check = [col["column_name"] for col in snowflake_columns]
    except Exception as e:
        logger.warning(f"    [WARN] Error fetching columns from Snowflake: {e}, using JSON columns")
        columns_to_check = json_columns
    
    # Check which columns are null-only using table_fullname
    null_only_columns = check_null_only_columns(
        creds, table_fullname, columns_to_check
    )
    
    if not null_only_columns:
        logger.info(f"    [INFO] No null-only columns found for {table_fullname}")
        # Copy files as-is
        safe_copy_file(json_path, output_json_path)
        return (table_fullname, set(), 0, False)
    else:
        logger.info(f"    [INFO] Found {len(null_only_columns)} null-only column(s) for {table_fullname}")
        
        # Map Snowflake column names to JSON column names
        # Create a mapping from normalized Snowflake names to actual JSON column names
        json_col_map = {normalize_column_name(c): c for c in json_columns}
        
        # Convert null_only_columns (which are normalized Snowflake names) to JSON column names
        json_columns_to_remove = set()
        for sf_col_normalized in null_only_columns:
            # Try to find matching JSON column
            if sf_col_normalized in json_col_map:
                json_columns_to_remove.add(normalize_column_name(json_col_map[sf_col_normalized]))
            else:
                # If not found, use the Snowflake name (might work if they match)
                json_columns_to_remove.add(sf_col_normalized)
        
        # Remove columns from JSON
        removed = remove_columns_from_json(json_path, output_json_path, json_columns_to_remove, table_fullname)
        return (table_fullname, null_only_columns, len(null_only_columns), removed)


def process_database(
    db_path: str,
    output_path: str,
    db_name: str,
    creds: Dict[str, Any],
    max_workers: int = 8
) -> Dict[str, int]:
    """
    Process a single database, removing null-only columns from all tables.
    Uses multiprocessing with max_workers concurrent queries.
    Returns dict with statistics: {'tables_processed': int, 'columns_removed': int}
    """
    logger.info(f"[INFO] Processing database: {db_name}")
    
    stats = {'tables_processed': 0, 'columns_removed': 0}
    
    # Get all schemas in the database
    schema_dirs = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
    
    for schema_dir in sorted(schema_dirs):
        schema_path = os.path.join(db_path, schema_dir)
        output_schema_path = os.path.join(output_path, schema_dir)
        os.makedirs(output_schema_path, exist_ok=True)
        
        logger.info(f"  [INFO] Processing schema: {schema_dir}")
        
        # Track which columns were removed for each table (for DDL processing)
        table_removed_columns: Dict[str, Set[str]] = {}
        
        # Collect all JSON files (tables) to process
        json_files = [f for f in os.listdir(schema_path) if f.endswith(".json")]
        
        # Prepare tasks for parallel processing
        tasks = []
        for json_file in sorted(json_files):
            json_path = os.path.join(schema_path, json_file)
            output_json_path = os.path.join(output_schema_path, json_file)
            tasks.append((json_path, output_json_path, db_name, schema_dir))
        
        # Process tables in parallel with max_workers limit
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_table, json_path, output_json_path, db_name, schema_dir, creds): (json_path, output_json_path, db_name, schema_dir)
                for json_path, output_json_path, db_name, schema_dir in tasks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                task_info = future_to_task[future]
                json_path = task_info[0]
                try:
                    table_fullname, null_only_columns, columns_removed, was_modified = future.result()
                    
                    stats['tables_processed'] += 1
                    stats['columns_removed'] += columns_removed
                    
                    if was_modified and null_only_columns:
                        # Store removed columns for DDL processing
                        table_removed_columns[table_fullname] = null_only_columns
                        # Also add variations for matching
                        parts = table_fullname.split(".")
                        if len(parts) >= 3:
                            table_removed_columns[parts[2]] = null_only_columns  # Just table name
                            table_removed_columns[f"{parts[1]}.{parts[2]}"] = null_only_columns  # schema.table
                        elif len(parts) == 2:
                            table_removed_columns[parts[1]] = null_only_columns
                            
                except Exception as e:
                    logger.error(f"    [ERROR] Exception processing {json_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    stats['tables_processed'] += 1  # Count failed tables too
        
        # Process DDL.csv after all tables are processed
        ddl_path = os.path.join(schema_path, "DDL.csv")
        output_ddl_path = os.path.join(output_schema_path, "DDL.csv")
        
        if os.path.isfile(ddl_path) and table_removed_columns:
            # Process DDL to remove columns
            logger.info(f"  [INFO] Processing DDL.csv for schema {schema_dir}")
            remove_columns_from_ddl(ddl_path, output_ddl_path, table_removed_columns)
        elif os.path.isfile(ddl_path):
            # No columns removed, just copy DDL as-is
            safe_copy_file(ddl_path, output_ddl_path)
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove columns containing only null values from Snowflake database metadata."
    )
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument(
        "--metadata-root",
        default=os.path.join(project_root, "datasets", "Spider2", "spider2-snow", "resource", "databases"),
        help="Root folder containing database metadata directories.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(project_root, "datasets", "Spider2", "spider2-snow", "resource", "databases_no_nulls_2"),
        help="Destination root for cleaned metadata (will mirror database/schema structure).",
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent Snowflake queries (default: 8).",
    )
    args = parser.parse_args()
    
    metadata_root = os.path.abspath(args.metadata_root)
    output_root = os.path.abspath(args.output_root)
    cred_path = os.path.abspath(args.credential_path)
    
    if not os.path.exists(cred_path):
        logger.error(f"[ERROR] Credential file not found: {cred_path}")
        return
    
    creds = load_credentials(cred_path)
    
    if args.databases:
        db_dirs = [os.path.join(metadata_root, db) for db in args.databases]
    else:
        db_dirs = [
            os.path.join(metadata_root, d)
            for d in os.listdir(metadata_root)
            if os.path.isdir(os.path.join(metadata_root, d))
        ]
    
    os.makedirs(output_root, exist_ok=True)
    
    total_stats = {'tables_processed': 0, 'columns_removed': 0}
    
    for db_dir in sorted(db_dirs):
        if not os.path.isdir(db_dir):
            logger.warning(f"[WARN] Skipping non-directory {db_dir}")
            continue
        
        db_name = os.path.basename(db_dir)
        out_db_dir = os.path.join(output_root, db_name)
        os.makedirs(out_db_dir, exist_ok=True)
        
        try:
            stats = process_database(db_dir, out_db_dir, db_name, creds, max_workers=args.max_workers)
            total_stats['tables_processed'] += stats['tables_processed']
            total_stats['columns_removed'] += stats['columns_removed']
        except Exception as e:
            logger.error(f"[ERROR] Failed to process database {db_name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"[INFO] Processing complete!")
    logger.info(f"[INFO] Summary: Processed {total_stats['tables_processed']} tables, removed {total_stats['columns_removed']} null-only columns")


if __name__ == "__main__":
    main()
