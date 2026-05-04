"""Subprocess worker for the python_interpreter MCP tool.

Runs a single Python code snippet in an isolated process with access to:
  - Pre-pickled initial data (_db_path, _db_id, _db_type, _cred_path)
  - Accumulated namespace from prior calls (stateful across tool calls)
  - All six database tool functions that mirror the MCP tools
  - pd, np and all standard Python builtins

CLI:
    python python_interpreter_worker.py <code_file> <init_pkl> <state_pkl> <out_state_pkl>
"""

import sys
import os
import ast
import io
import pickle
import cloudpickle
import contextlib
import traceback

_SKIP_KEYS = {
    "__builtins__", "__name__", "pd", "np", "nx", "gpd",
    "conn", "cursor", "get_cursor",
    # db tool functions — rebuilt each run
    "list_schemas", "list_tables", "list_columns",
    "query_database", "get_distinct_values", "search_dimension_values",
}


def _build_namespace():
    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    for alias, module_name in (
        ("pd", "pandas"),
        ("np", "numpy"),
        ("nx", "networkx"),
        ("gpd", "geopandas"),
    ):
        try:
            namespace[alias] = __import__(module_name)
        except Exception:
            continue
    return namespace


def _load_pickle(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_state(namespace, out_path):
    picklable = {}
    for k, v in namespace.items():
        if k in _SKIP_KEYS:
            continue
        try:
            cloudpickle.dumps(v)
            picklable[k] = v
        except Exception:
            pass
    with open(out_path, "wb") as f:
        cloudpickle.dump(picklable, f)


def _inject_db_tools(namespace, db_type, db_path, db_id, cred_path):
    """Build and inject the six database tool functions into *namespace*."""
    # Make db_tools importable from the same directory as this worker
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    try:
        from db_tools import (
            create_list_tables_tool,
            create_list_columns_tool,
            create_get_distinct_values_tool,
            create_search_dimension_values_tool,
            execute_and_format_query_result,
        )
    except ImportError as e:
        namespace["_db_tools_import_error"] = str(e)
        return

    # ── list_schemas ──────────────────────────────────────────────────────────
    def list_schemas():
        """List all schemas in the current database."""
        if db_type == "snowflake":
            base = os.path.join(db_path, "resource", "databases_no_nulls_2", db_id)
            if not os.path.isdir(base):
                return f"No metadata found for database '{db_id}' at {base}"
            schemas = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
            if not schemas:
                return f"No schemas found for database '{db_id}'."
            lines = [f"Schemas in database '{db_id}':"]
            for s in schemas:
                schema_path = os.path.join(base, s)
                n_tables = sum(
                    1
                    for fn in os.listdir(schema_path)
                    if fn.endswith(".json") and not fn.endswith("_M-Schema.json")
                )
                lines.append(f"  - {s}  ({n_tables} tables)")
            return "\n".join(lines)
        return f"Database '{db_id}' (SQLite) has one schema: main"

    namespace["list_schemas"] = list_schemas

    # ── list_tables / list_columns (no live DB cursor needed) ─────────────────
    namespace["list_tables"] = create_list_tables_tool(
        db_type=db_type, database_name=db_id, db_path=db_path
    )
    namespace["list_columns"] = create_list_columns_tool(
        db_type=db_type, database_name=db_id, db_path=db_path
    )

    # ── tools that need a live cursor ─────────────────────────────────────────
    if db_type == "snowflake" and cred_path:
        try:
            import json as _json
            import snowflake.connector as _sf

            with open(cred_path) as _f:
                _creds = _json.load(_f)
            _conn = _sf.connect(**_creds, database=db_id)
            namespace["conn"] = _conn
            namespace["cursor"] = _conn.cursor()
            namespace["get_cursor"] = _conn.cursor

            _cursor_getter = _conn.cursor

            def query_database(query: str) -> str:
                """Execute a SELECT query and return up to 20 rows as a markdown table."""
                _cur = _cursor_getter()
                try:
                    return execute_and_format_query_result(
                        cursor=_cur,
                        query=query,
                        db_path=db_path,
                        db_type=db_type,
                        n_example_rows=20,
                        truncate_data=True,
                        max_truncate_len=200,
                        include_comment=False,
                        include_query=False,
                    )
                except Exception as exc:
                    return f"Error executing query: {exc}"
                finally:
                    _cur.close()

            namespace["query_database"] = query_database
            namespace["get_distinct_values"] = create_get_distinct_values_tool(
                db_type=db_type, cursor_getter=_cursor_getter
            )
            namespace["search_dimension_values"] = create_search_dimension_values_tool(
                db_type=db_type, cursor_getter=_cursor_getter
            )
        except Exception as exc:
            namespace["_snowflake_connect_error"] = str(exc)
            print(f"[python_interpreter] Snowflake connection failed: {exc}", file=sys.stderr)

    elif db_type == "sqlite":
        import sqlite3 as _sqlite3

        db_file = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        if os.path.isfile(db_file):
            _conn = _sqlite3.connect(db_file)
            namespace["conn"] = _conn
            namespace["get_cursor"] = lambda: _conn.cursor()

            _cursor_getter = lambda: _conn.cursor()

            def query_database(query: str) -> str:
                """Execute a SELECT query and return up to 20 rows as a markdown table."""
                _cur = _cursor_getter()
                try:
                    return execute_and_format_query_result(
                        cursor=_cur,
                        query=query,
                        db_path=db_path,
                        db_type=db_type,
                        n_example_rows=20,
                        truncate_data=True,
                        max_truncate_len=200,
                        include_comment=False,
                        include_query=False,
                    )
                except Exception as exc:
                    return f"Error executing query: {exc}"

            namespace["query_database"] = query_database
            namespace["get_distinct_values"] = create_get_distinct_values_tool(
                db_type=db_type, cursor_getter=_cursor_getter
            )
            namespace["search_dimension_values"] = create_search_dimension_values_tool(
                db_type=db_type, cursor_getter=_cursor_getter
            )


def _run_code(code, namespace):
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = None
    try:
        parsed = ast.parse(code, mode="exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            namespace["__name__"] = "__main__"
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                prefix = ast.Module(body=parsed.body[:-1], type_ignores=[])
                exec(compile(prefix, "<python_interpreter>", "exec"), namespace, namespace)
                result = eval(
                    compile(
                        ast.Expression(parsed.body[-1].value), "<python_interpreter>", "eval"
                    ),
                    namespace,
                    namespace,
                )
            else:
                exec(compile(parsed, "<python_interpreter>", "exec"), namespace, namespace)
    except Exception:
        traceback.print_exc(file=stderr)

    output = stdout.getvalue()
    err_output = stderr.getvalue()
    if result is not None:
        if output and not output.endswith("\n"):
            output += "\n"
        output += repr(result)
    if err_output:
        if output and not output.endswith("\n"):
            output += "\n"
        output += err_output
    return output.strip()


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python python_interpreter_worker.py "
            "<code_file> <init_pkl> <state_pkl> <out_state_pkl>",
            file=sys.stderr,
        )
        sys.exit(1)

    code_file, init_pkl, state_pkl, out_state_pkl = sys.argv[1:]

    namespace = _build_namespace()
    initial_data = _load_pickle(init_pkl)
    namespace.update(initial_data)
    namespace.update(_load_pickle(state_pkl))

    # Extract DB context injected by mcp_server
    db_type = initial_data.get("_db_type", "snowflake")
    db_path = initial_data.get("_db_path", "spider2-snow")
    db_id = initial_data.get("_db_id", "")
    cred_path = initial_data.get("_cred_path", "")

    _inject_db_tools(namespace, db_type, db_path, db_id, cred_path)

    with open(code_file) as f:
        code = f.read()

    output = _run_code(code, namespace)
    print(output if output else "No output.")

    _save_state(namespace, out_state_pkl)


if __name__ == "__main__":
    main()
