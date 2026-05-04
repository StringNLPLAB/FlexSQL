import sys
import argparse
import json
import sqlite3
import pandas as pd
import snowflake.connector
from typing import List
import importlib.util
import os

def fetch_table(conn: snowflake.connector.SnowflakeConnection, queries: List[str]) -> List[pd.DataFrame]:
    """
    Execute SQL queries and return the full result set as a pandas DataFrame.
    """
    results = []
    for query in queries:
        cur = conn.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            results.append(pd.DataFrame(rows, columns=col_names))
        finally:
            cur.close()
    return results

def fetch_table_sqlite(db_path: str, queries: List[str]) -> List[pd.DataFrame]:
    """
    Execute SQL queries against a SQLite database and return results as pandas DataFrames.
    """
    results = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for query in queries:
            cur = conn.cursor()
            try:
                cur.execute(query)
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
                results.append(pd.DataFrame(rows, columns=col_names))
            finally:
                cur.close()
    finally:
        conn.close()
    return results

def get_snowflake_connector(sf_cred_path):
    """Load Snowflake connection parameters from a JSON file."""
    with open(f"{sf_cred_path}", "r", encoding="utf-8") as f:
        sf_cred = json.load(f)

    # Set timeout parameters to handle large result sets
    # socket_timeout: Timeout for socket read operations (default is 7 seconds, too short for large results)
    # Set to None to disable timeout, or use a larger value like 300 (5 minutes)
    if 'socket_timeout' not in sf_cred:
        sf_cred['socket_timeout'] = None  # Disable timeout for large result fetches
    
    conn = snowflake.connector.connect(**sf_cred)
    return conn

def save_output(df: pd.DataFrame, save_path: str):
    df.to_csv(save_path, index=False)

def get_llm_processing(program_path, q_id, pd_module, conn=None, db_type=None):
    # 1. Define the full path to the file
    module_path = os.path.join(program_path)
    program_name = module_path.split("/")[-1].split(".")[0]

    # 2. Specify a unique module name
    spec = importlib.util.spec_from_file_location(f"{q_id}_{program_name}", module_path)
    # 3. Create the module object
    program_module = importlib.util.module_from_spec(spec)
    program_module.pd = pd_module
    if conn is not None:
        if db_type == "snowflake":
            # Snowflake connection has no .execute() — inject cursor like python_interpreter_worker does
            program_module.cursor = conn.cursor()
            program_module.get_cursor = conn.cursor
        else:
            # SQLite: conn.execute() and pd.read_sql_query(sql, conn) both work
            program_module.conn = conn
            program_module.get_cursor = lambda: conn.cursor()
    # 4. Execute the module code
    spec.loader.exec_module(program_module)
    # 5. Get the processing function
    processing = program_module.processing
    return processing

def main():
    if len(sys.argv) != 6:
        print("Usage: python program_frame.py <db_path_or_cred_json> <question_id> <program_path> <queries_path> <output_csv>")
        print("  BIRD/sqlite mode : db_path_or_cred_json must end with .sqlite")
        print("  Snowflake mode   : db_path_or_cred_json must end with .json")
        sys.exit(1)

    db_arg, q_id, program_path, queries_path, output_path = [sys.argv[i] for i in range(1, 6)]

    queries = list(json.load(open(queries_path)).values())  # {<table_name>: <query>}

    if db_arg.endswith(".sqlite"):
        # BIRD / SQLite mode
        conn = sqlite3.connect(f"file:{db_arg}?mode=ro", uri=True)
        data_frames = fetch_table_sqlite(db_path=db_arg, queries=queries)
        db_type = "sqlite"
    else:
        # Snowflake mode
        conn = get_snowflake_connector(sf_cred_path=db_arg)
        data_frames = fetch_table(conn=conn, queries=queries)
        db_type = "snowflake"

    processing_function = get_llm_processing(program_path, q_id, pd_module=pd, conn=conn, db_type=db_type)

    result_df = processing_function(data_frames)

    save_output(result_df, output_path)

if __name__ == "__main__":
    main()
