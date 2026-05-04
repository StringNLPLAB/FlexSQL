import json
import sqlite3
import sys

import pandas as pd
import snowflake.connector


def fetch_table(conn: snowflake.connector.SnowflakeConnection, query: str) -> pd.DataFrame:
    """
    Execute SQL queries and return the full result set as a pandas DataFrame.
    """
    result = None
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        result = pd.DataFrame(rows, columns=col_names)
    finally:
        cur.close()
    return result


def fetch_table_sqlite(db_path: str, query: str) -> pd.DataFrame:
    """
    Execute SQL queries against a SQLite database and return results as pandas DataFrames.
    """
    result = None
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        result = pd.DataFrame(rows, columns=col_names)
    finally:
        conn.close()
    return result


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


# def get_llm_processing(program_path, q_id, pd_module):
#     # 1. Define the full path to the file
#     module_path = os.path.join(program_path)
#     program_name = module_path.split("/")[-1].split(".")[0]
#
#     # 2. Specify a unique module name
#     spec = importlib.util.spec_from_file_location(f"{q_id}_{program_name}", module_path)
#     # 3. Create the module object
#     program_module = importlib.util.module_from_spec(spec)
#     program_module.pd = pd_module
#     # 4. Execute the module code
#     spec.loader.exec_module(program_module)
#     # 5. Get the processing function
#     processing = program_module.processing
#     return processing

def main():
    if len(sys.argv) != 6:
        print("Usage: python program_frame.py <cred_json> <question_id> <sql_file> <output_csv>")
        sys.exit(1)

    dialect, sf_cred_path, q_id, sql_file, output_path = sys.argv[1:]

    with open(sql_file, 'r') as reader:
        query = reader.read()

    if dialect == "snowflake":
        snowflake_connector = get_snowflake_connector(sf_cred_path=sf_cred_path)
        exec_result = fetch_table(conn=snowflake_connector, query=query)
    elif dialect == "sqlite":
        exec_result = fetch_table_sqlite(sf_cred_path, query=query)
    else:
        raise ValueError(f"Unknown SQL dialect: {dialect}")

    save_output(exec_result, output_path)


if __name__ == "__main__":
    main()
