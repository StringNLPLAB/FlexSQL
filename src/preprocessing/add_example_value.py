import os
import json
import argparse
import sys

# Ensure we can import from src/ (parent dir, for get_ddl) and project root (for utils.program_frame_sf)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

try:
    from get_ddl import truncate_nested_data
    from utils.program_frame_sf import create_snowpark_session
    from snowflake.snowpark import Session
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

import concurrent.futures

# Add threading lock for session usage if needed, though Snowpark session is generally thread-safe for executing queries.
# However, to restrict concurrency to 8, we can use a semaphore or just rely on ThreadPoolExecutor.

def fetch_single_column_example(session, table_fullname, col):
    """
    Fetches examples for a single column.
    """
    col_expr = f'"{col}"'
    q = f"SELECT DISTINCT '{col}' as col_name, CAST({col_expr} AS STRING) as val FROM {table_fullname} WHERE {col_expr} IS NOT NULL LIMIT 10"
    
    try:
        rows = session.sql(q).collect()
        examples = []
        for row in rows:
            v = row['VAL']
            if isinstance(v, str):
                try:
                    if v.strip().startswith(("[", "{")):
                        v = json.loads(v)
                except Exception:
                    pass
            v = truncate_nested_data(v)
            examples.append(v)
        return col, examples
    except Exception as e:
        print(f"Error fetching examples for {table_fullname}.{col}: {e}")
        return col, []

def get_column_examples(session, table_fullname: str, columns_to_fetch: list[str], existing_examples: dict) -> dict[str, list]:
    """
    Fetches examples for specified columns using ThreadPoolExecutor to limit concurrency.
    Merges with existing_examples.
    """
    examples = existing_examples.copy()
    if not columns_to_fetch or not session:
        return examples
    
    print(f"Fetching examples for {len(columns_to_fetch)} columns in {table_fullname}...")

    # Use ThreadPoolExecutor to limit concurrent queries to 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_col = {
            executor.submit(fetch_single_column_example, session, table_fullname, col): col 
            for col in columns_to_fetch
        }
        
        for future in concurrent.futures.as_completed(future_to_col):
            col = future_to_col[future]
            try:
                c, vals = future.result()
                if vals:
                    examples[c] = vals
            except Exception as e:
                print(f"Exception for column {col}: {e}")
                
    return examples

def process_table_metadata(session: Session, metadata_path: str, table_fullname: str) -> bool:
    """
    Reads metadata, fetches column examples, and updates the JSON file.
    Returns True if updated, False otherwise.
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        column_names = metadata.get("column_names", [])
        if not column_names:
            print(f"No columns found in metadata for {table_fullname}")
            return False
            
        # 1. Check existing sample_rows in metadata
        sample_rows = metadata.get("sample_rows", [])
        
        # 2. Identify columns that need fetching
        # We assume we need 2 non-null values.
        # We also look at existing 'column_examples' if we ran this script before?
        # The user said: "retrieve values for columns whose example rows show null value at that position".
        
        # Let's populate initial examples from sample_rows
        existing_examples = {}
        for col in column_names:
            existing_examples[col] = []
            
        # Extract from sample_rows
        for row in sample_rows:
            for col in column_names:
                val = row.get(col)
                if val is not None:
                     val = truncate_nested_data(val)
                     if val not in existing_examples[col]:
                         existing_examples[col].append(val)
        
        columns_to_fetch = []
        for col in column_names:
            # Condition: fetch if we have fewer than 2 distinct non-null values?
            # Or strict interpretation of "example rows show null value at that position" (any null implies fetch?)
            # I will use the heuristic: if we have < 2 non-null examples, fetch.
            # This covers the case where we have nulls (likely leading to 0 or 1 value).
            if len(existing_examples[col]) < 2:
                columns_to_fetch.append(col)
        
        if not columns_to_fetch:
            print(f"Sufficient examples already present in sample_rows for {table_fullname}")
            metadata["column_examples"] = existing_examples
        else:
            # Fetch missing
            updated_examples = get_column_examples(session, table_fullname, columns_to_fetch, existing_examples)
            metadata["column_examples"] = updated_examples
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        print(f"Error processing {table_fullname} ({metadata_path}): {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fetch example values for Snowflake tables and update metadata JSONs.")
    parser.add_argument("--db_folder", type=str, default="datasets/Spider2/spider2-snow/resource/databases_no_nulls_2", help="Path to the spider2-snow folder")
    parser.add_argument("--credential_path", type=str, default="datasets/Spider2/spider2-snow/snowflake_credential.json", help="Path to snowflake credentials")
    args = parser.parse_args()

    db_folder = args.db_folder
    credential_path = args.credential_path
    
    if not os.path.exists(credential_path):
        print(f"Credential file not found at {credential_path}")
        sys.exit(1)

    print("Creating Snowpark session...")
    try:
        session = create_snowpark_session(credential_path)
    except Exception as e:
        print(f"Failed to create Snowpark session: {e}")
        sys.exit(1)
        
    databases_folder = os.path.join(db_folder, "resource", "databases")
    if not os.path.exists(databases_folder):
        print(f"Databases folder not found at {databases_folder}")
        sys.exit(1)

    # Traverse directory structure: databases -> schemas -> tables(.json)
    for database_dir in os.listdir(databases_folder):
        db_path = os.path.join(databases_folder, database_dir)
        if not os.path.isdir(db_path):
            continue
            
        for schema_dir in os.listdir(db_path):
            schema_path = os.path.join(db_path, schema_dir)
            if not os.path.isdir(schema_path):
                continue
                
            # Iterate over JSON files in the schema directory
            for file in os.listdir(schema_path):
                if not file.endswith(".json"):
                    continue
                    
                metadata_path = os.path.join(schema_path, file)
                
                # We need table_fullname from the JSON to query Snowflake
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                        table_fullname = meta.get("table_fullname")
                        
                    if table_fullname:
                        process_table_metadata(session, metadata_path, table_fullname)
                    else:
                        print(f"Skipping {metadata_path}: 'table_fullname' not found.")
                        
                except Exception as e:
                    print(f"Failed to read/process {metadata_path}: {e}")

    session.close()
    print("Done.")

if __name__ == "__main__":
    main()
