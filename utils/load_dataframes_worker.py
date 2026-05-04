"""Subprocess worker: load DataFrames from a database and write them to stdout.

This runs in a child process so that if the OS OOM-killer sends SIGKILL (due to
a very large table), only this child dies — the parent process is unaffected and
can detect the non-zero exit code to trigger SQL-only fallback.

Protocol:
    stdin  – JSON-encoded args dict
    stdout – pickle-encoded list[pd.DataFrame]  (binary)
    stderr – warning/error messages

args_json schema:
    {
        "db_type": "sqlite" | "snowflake",
        // sqlite:
        "db_file_path": "/path/to/db.sqlite",
        // snowflake:
        "snowflake_credential": { ... },
        "database": "SOME_DB",
        // both:
        "queries": ["SELECT ...", ...]
    }

Exit codes:
    0  – success; pickle written to stdout
    1  – handled error (bad args, unknown db_type)
   -9  – SIGKILL from OS OOM killer (caught by parent as returncode == -9)
"""
import sys
import json
import pickle

import pandas as pd


def main():
    args = json.load(sys.stdin)
    db_type = args["db_type"]
    queries = args["queries"]
    data_frames = []

    if db_type == "sqlite":
        import sqlite3
        conn = sqlite3.connect(f"file:{args['db_file_path']}?mode=ro", uri=True)
        cursor = conn.cursor()
        for query in queries:
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                data_frames.append(pd.DataFrame(rows, columns=col_names))
            except Exception as e:
                print(f"Warning: query failed: {e}", file=sys.stderr)
                data_frames.append(pd.DataFrame())
        conn.close()

    elif db_type == "snowflake":
        import snowflake.connector
        conn = snowflake.connector.connect(
            **args["snowflake_credential"],
            database=args["database"]
        )
        for query in queries:
            try:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                data_frames.append(pd.DataFrame(rows, columns=col_names))
                cursor.close()
            except Exception as e:
                print(f"Warning: query failed: {e}", file=sys.stderr)
                data_frames.append(pd.DataFrame())
        conn.close()

    else:
        print(f"Unknown db_type: {db_type}", file=sys.stderr)
        sys.exit(1)

    sys.stdout.buffer.write(pickle.dumps(data_frames))


if __name__ == "__main__":
    main()
