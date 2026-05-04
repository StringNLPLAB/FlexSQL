#!/usr/bin/env python3
"""
MCP server for CCSQL — exposes Snowflake database exploration tools to Claude Code.

Transport:
  - stdio (default): one server per claude subprocess; per-request db_id comes from
    the CCSQL_DB_ID env var.
  - http  (--transport http): one persistent server serves many claude subprocesses;
    per-request db_id comes from the X-CCSQL-DB-ID HTTP header, and python_interpreter
    state is isolated per X-CCSQL-SESSION-ID header.

Reads configuration from environment variables:
  CCSQL_DB_PATH  - path to the data directory (default: spider2-snow)
  CCSQL_DB_TYPE  - database type (default: snowflake)
  CCSQL_DB_ID    - (stdio mode only) current database name

Tools exposed:
  list_schemas()
  list_tables(schema_name)
  list_columns(table_name)
  query_database(query)
  get_distinct_values(table, column)
  search_dimension_values(terms, table, column_name, additional_columns)
  python_interpreter(code)
"""

import argparse
import contextvars
import json
import os
import sys
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from db_tools import (
    create_list_tables_tool,
    create_list_columns_tool,
    create_get_distinct_values_tool,
    create_search_dimension_values_tool,
    create_python_interpreter_tool,
    post_format_generated_query,
    format_results_to_markdown,
)

# ── Configuration ──────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("CCSQL_DB_PATH", "spider2-snow")
DB_TYPE = os.environ.get("CCSQL_DB_TYPE", "snowflake")
TOOL_TIMEOUT_SEC = float(os.environ.get("CCSQL_TOOL_TIMEOUT_SEC", "200"))

_tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ccsql-tool")


def with_timeout(fn):
    """Wrap *fn* so it returns an error string if it runs longer than TOOL_TIMEOUT_SEC.

    Uses contextvars.copy_context() so Starlette's request context (which is how
    get_http_headers() finds X-CCSQL-DB-ID) propagates into the worker thread.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = contextvars.copy_context()
        future = _tool_executor.submit(ctx.run, fn, *args, **kwargs)
        try:
            return future.result(timeout=TOOL_TIMEOUT_SEC)
        except FuturesTimeoutError:
            return (
                f"Error: tool '{fn.__name__}' exceeded timeout of {TOOL_TIMEOUT_SEC:.0f}s. "
            )
    return wrapper

def get_db_id() -> str:
    """Return the current request's db_id.

    HTTP transport: prefer the X-CCSQL-DB-ID header (set per-claude-subprocess
    via the generated .mcp.json). Stdio transport: fall back to the CCSQL_DB_ID
    env var. Raises if neither is set.
    """
    headers = get_http_headers(include={"x-ccsql-db-id"})
    db_id = headers.get("x-ccsql-db-id", "") or os.environ.get("CCSQL_DB_ID", "")
    if not db_id:
        raise ValueError(
            "CCSQL_DB_ID is not set. In HTTP mode the harness must send an "
            "X-CCSQL-DB-ID header; in stdio mode it must set the env var."
        )
    return db_id


def get_session_id() -> str:
    """Per-claude-subprocess session id, used to isolate python_interpreter state.

    HTTP transport: read from X-CCSQL-SESSION-ID header. Stdio transport: a fixed
    constant, since one stdio server == one session by construction.
    """
    headers = get_http_headers(include={"x-ccsql-session-id"})
    return headers.get("x-ccsql-session-id", "") or "stdio-local"


def get_snowflake_cursor():
    """Create a fresh Snowflake cursor for the current CCSQL_DB_ID."""
    import snowflake.connector
    cred_path = os.path.join(DB_PATH, "snowflake_credential.json")
    with open(cred_path) as f:
        creds = json.load(f)
    conn = snowflake.connector.connect(**creds, database=get_db_id())
    return conn, conn.cursor()


def get_sqlite_cursor():
    """Create a fresh SQLite cursor for the current CCSQL_DB_ID."""
    import sqlite3
    db_id = get_db_id()
    db_file = os.path.join(DB_PATH, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_file)
    return conn, conn.cursor()


def get_connection():
    """Return (conn, cursor) for the configured DB_TYPE."""
    if DB_TYPE == "snowflake":
        return get_snowflake_cursor()
    elif DB_TYPE == "sqlite":
        return get_sqlite_cursor()
    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}")


# ── FastMCP server ─────────────────────────────────────────────────────────────

mcp = FastMCP("snowflake-tools")


@mcp.tool()
@with_timeout
def list_schemas() -> str:
    """List all available schemas in the current database (CCSQL_DB_ID).

    Call this first at the start of every task to discover which schemas exist
    before calling list_tables. Each database may have multiple schemas.
    """
    db_id = get_db_id()
    if DB_TYPE == "snowflake":
        base = os.path.join(DB_PATH, "resource", "databases_no_nulls_2", db_id)
        if not os.path.isdir(base):
            return f"No metadata found for database '{db_id}' at {base}"
        schemas = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )
        if not schemas:
            return f"No schemas found for database '{db_id}'."
        lines = [f"Schemas in database '{db_id}':"]
        for s in schemas:
            schema_path = os.path.join(base, s)
            n_tables = sum(
                1 for f in os.listdir(schema_path)
                if f.endswith(".json") and not f.endswith("_M-Schema.json")
            )
            lines.append(f"  - {s}  ({n_tables} tables)")
        return "\n".join(lines)
    else:
        # SQLite: schema is always "main"
        return f"Database '{db_id}' (SQLite) has one schema: main"


@mcp.tool()
@with_timeout
def list_tables(schema_name: str) -> str:
    """List all tables in a specific schema, with their fully-qualified names.

    Args:
        schema_name: The schema name to list tables from (get schema names from list_schemas first).
    """
    db_id = get_db_id()
    fn = create_list_tables_tool(db_type=DB_TYPE, database_name=db_id, db_path=DB_PATH)
    return fn(schema_name)


@mcp.tool()
@with_timeout
def list_columns(table_name: str) -> str:
    """List all columns in a table with data types, descriptions, and example values.

    Args:
        table_name: Fully-qualified table name in format DATABASE.SCHEMA.TABLE
                    (e.g. 'PATENTS.PATENTS.CPC_DEFINITION')
    """
    db_id = get_db_id()
    fn = create_list_columns_tool(db_type=DB_TYPE, database_name=db_id, db_path=DB_PATH)
    return fn(table_name)


@mcp.tool()
@with_timeout
def query_database(query: str, truncate: bool = True) -> str:
    """Execute a SQL query on the Snowflake database.

    Use this to verify your SQL, explore data values, or check joins.
    Always use fully-qualified table names: "DATABASE"."SCHEMA"."TABLE".
    Add a LIMIT clause when exploring to avoid returning too many rows and wasting context.

    By default, long cell values are truncated to save tokens. Set truncate=False
    when you need to see full cell values (e.g. to check exact strings, JSON blobs,
    or long text fields).

    Args:
        query: The SQL query to execute (SELECT only).
        truncate: If True (default), truncate long cell values to 200 chars. Set False to see full values.
    """
    conn, cursor = get_connection()
    try:
        formatted_query = post_format_generated_query(query, db_path=DB_PATH, db_type=DB_TYPE, include_comment=False)
        cursor.execute(formatted_query)
        headers = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result = format_results_to_markdown(headers, rows, truncate_data=truncate, max_truncate_len=200)
        return f"{len(rows)} rows returned:\n{result}"
    except Exception as e:
        return f"Error executing query: {e}"
    finally:
        cursor.close()
        conn.close()


@mcp.tool()
@with_timeout
def get_distinct_values(table: str, column: str, truncate: bool = True) -> str:
    """Get up to 100 distinct values from a specific column in a table.

    Useful for finding exact filter values for WHERE clauses.
    By default, long values are truncated. Set truncate=False to see full values.

    Args:
        table: Fully-qualified table name (e.g. 'PATENTS.PATENTS.CPC_DEFINITION')
        column: Column name to get distinct values from
        truncate: If True (default), truncate long values. Set False to see full values.
    """
    conn, cursor = get_connection()
    cursor_getter = lambda: cursor
    try:
        fn = create_get_distinct_values_tool(db_type=DB_TYPE, cursor_getter=cursor_getter)
        return fn(table, column, truncate=truncate)
    finally:
        cursor.close()
        conn.close()


@mcp.tool()
@with_timeout
def search_dimension_values(
    terms: list,
    table: str,
    column_name: str,
    additional_columns: Optional[list] = None,
    truncate: bool = True,
) -> str:
    """Search a table column for rows matching any of the given terms (case-insensitive partial match).

    Returns up to 5 matching rows. Useful for looking up exact category codes or IDs.
    By default, long cell values are truncated. Set truncate=False to see full values.

    Args:
        terms: List of search terms (e.g. ["patent", "application"])
        table: Fully-qualified table name (e.g. 'PATENTS.PATENTS.CPC_DEFINITION')
        column_name: Column to search in
        additional_columns: Optional extra columns to include in results
        truncate: If True (default), truncate long cell values. Set False to see full values.
    """
    conn, cursor = get_connection()
    cursor_getter = lambda: cursor
    try:
        fn = create_search_dimension_values_tool(db_type=DB_TYPE, cursor_getter=cursor_getter)
        return fn(terms, table, column_name, additional_columns, truncate=truncate)
    finally:
        cursor.close()
        conn.close()


# One interpreter instance per (db_type, db_path, db_id) tuple so state persists
# across multiple python_interpreter() calls within the same Claude session.
_interpreter_cache: dict = {}


@mcp.tool()
@with_timeout
def python_interpreter(code: str) -> str:
    """Execute Python code in a stateful interpreter with database tools pre-loaded.

    The interpreter has access to:
      list_schemas()                                            — list schemas in current DB
      list_tables(schema_name)                                  — list tables in a schema
      list_columns(table_name)                                  — columns + types + examples
      query_database(query)                                     — run SELECT, get markdown table
      get_distinct_values(table, column)                        — up to 100 distinct values
      search_dimension_values(terms, table, col, extra=[])      — ILIKE/LIKE search
      pd, np                                                    — pandas and numpy
      All standard Python builtins

    State (variables you define) persists across multiple calls to this tool.

    Args:
        code: Python code to execute.
    """
    db_id = get_db_id()
    session_id = get_session_id()
    key = (DB_TYPE, DB_PATH, db_id, session_id)
    if key not in _interpreter_cache:
        _interpreter_cache[key] = create_python_interpreter_tool(
            db_type=DB_TYPE,
            db_path=DB_PATH,
            db_id=db_id,
        )
    return _interpreter_cache[key](code)


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCSQL MCP server")
    p.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio",
        help="stdio (default, one server per claude subprocess) or http "
             "(persistent server; db_id/session_id come from headers)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0,
                   help="HTTP port; 0 picks a free port and prints the URL to stdout")
    p.add_argument("--path", default="/mcp")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.transport == "stdio":
        mcp.run()
    else:
        port = args.port
        if port == 0:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((args.host, 0))
                port = s.getsockname()[1]
        # Print the URL on stdout so the parent launcher can read it deterministically.
        url = f"http://{args.host}:{port}{args.path}"
        print(f"CCSQL_MCP_URL={url}", flush=True)
        mcp.run(
            transport="http",
            host=args.host,
            port=port,
            path=args.path,
            show_banner=False,
        )
