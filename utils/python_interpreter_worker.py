"""Subprocess worker for the python_interpreter tool.

Runs a single Python code snippet in an isolated process, with access to
pre-pickled DataFrames and an optional live database connection.

CLI:
    python python_interpreter_worker.py <code_file> <init_pkl> <state_pkl> \
        <out_state_pkl> <db_type> <db_arg>

Args:
    code_file:     Path to a .py file containing the code to execute.
    init_pkl:      Path to pickle file with initial data (DataFrames, etc.).
    state_pkl:     Path to pickle file with accumulated namespace from prior calls
                   (may not exist on the first call).
    out_state_pkl: Path where updated picklable namespace will be written.
    db_type:       'sqlite' | 'snowflake' | '' (empty = no live DB access).
    db_arg:        Path to .sqlite file or snowflake_credential.json, or ''.
"""

import sys
import os
import ast
import io
try:
    import cloudpickle as pickle
except ImportError:
    import pickle
import contextlib
import traceback

_SKIP_KEYS = {
    "__builtins__", "__name__", "pd", "np", "nx", "gpd",
    "conn", "cursor", "get_cursor",
}


def _build_namespace():
    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    for alias, module_name in (("pd", "pandas"), ("np", "numpy"), ("nx", "networkx"), ("gpd", "geopandas")):
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
            pickle.dumps(v)
            picklable[k] = v
        except Exception:
            pass
    with open(out_path, "wb") as f:
        pickle.dump(picklable, f)


def _setup_db(namespace, db_type, db_arg):
    if not db_arg:
        return
    if db_type == "sqlite":
        import sqlite3
        conn = sqlite3.connect(f"file:{db_arg}?mode=ro", uri=True)
        namespace["conn"] = conn
        namespace["get_cursor"] = lambda: conn.cursor()
    elif db_type == "snowflake":
        import json
        try:
            import snowflake.connector
        except ImportError:
            return
        with open(db_arg, "r") as f:
            cred = json.load(f)
        sf_conn = snowflake.connector.connect(**cred)
        sf_cursor = sf_conn.cursor()
        namespace["cursor"] = sf_cursor
        namespace["get_cursor"] = sf_conn.cursor


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
                    compile(ast.Expression(parsed.body[-1].value), "<python_interpreter>", "eval"),
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
    if len(sys.argv) != 7:
        print(
            "Usage: python python_interpreter_worker.py "
            "<code_file> <init_pkl> <state_pkl> <out_state_pkl> <db_type> <db_arg>",
            file=sys.stderr,
        )
        sys.exit(1)

    code_file, init_pkl, state_pkl, out_state_pkl, db_type, db_arg = sys.argv[1:]

    # Build namespace: common libs + initial data + accumulated state
    namespace = _build_namespace()
    namespace.update(_load_pickle(init_pkl))
    namespace.update(_load_pickle(state_pkl))

    # Reconnect to DB
    _setup_db(namespace, db_type, db_arg)

    # Execute code
    with open(code_file, "r") as f:
        code = f.read()

    output = _run_code(code, namespace)
    print(output if output else "No output.")

    # Persist updated namespace (picklable parts only)
    _save_state(namespace, out_state_pkl)


if __name__ == "__main__":
    main()
