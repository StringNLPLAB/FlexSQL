"""
db_tools.py — Self-contained helpers for Snowflake/SQLite database tools.

Extracted from TestSQL/src/{get_ddl,schema_linking,tools}.py so that
mcp_server.py has no cross-project sys.path dependency.
"""

import os
import json
import re
import math
import sys
import pickle
import subprocess
import tempfile
from typing import Callable, Optional, List, Any, Dict, Tuple

from sqlglot import parse_one, exp

# ── get_ddl helpers ────────────────────────────────────────────────────────────

def truncate_nested_data(data, max_str_len=20):
    """Recursively traverse a nested structure and truncate strings to max_str_len."""
    try:
        if isinstance(data, str):
            data = json.loads(data)
    except Exception:
        pass

    if "bytearray" in str(data).lower():
        return "bytearray(b'...')"

    if isinstance(data, dict):
        return {key: truncate_nested_data(value, max_str_len) for key, value in data.items()}

    elif isinstance(data, list):
        if len(data) > 3:
            return [
                truncate_nested_data(data[0], max_str_len),
                truncate_nested_data(data[1], max_str_len),
                "...",
                truncate_nested_data(data[-1], max_str_len),
            ]
        else:
            return [truncate_nested_data(item, max_str_len) for item in data]

    elif isinstance(data, str):
        if len(data) > max_str_len:
            return data[: max_str_len // 2] + "..." + data[-max_str_len // 2 :]
        return data

    else:
        return data


def quote_identifiers(sql_text: str, output_dialect: str = "snowflake") -> str:
    """Quote lowercase identifiers in *sql_text* via sqlglot so Snowflake treats them correctly."""
    expression = parse_one(sql_text, read=output_dialect)

    flatten_aliases: set = set()
    for lateral in expression.find_all(exp.Lateral):
        if isinstance(lateral.this, exp.Explode):
            alias = lateral.args.get("alias")
            if alias:
                flatten_aliases.add(alias.this.this)

    FLATTEN_KEYWORDS = {"value", "key", "path", "index", "seq", "this"}
    never_quoted_node: dict = {}

    for node in expression.walk():
        if isinstance(node, exp.Column):
            table = node.table
            if table and table in flatten_aliases:
                if isinstance(node.this, exp.Identifier):
                    col_name = node.this.this
                    if col_name.lower() in FLATTEN_KEYWORDS:
                        node.this.set("this", col_name.upper())
                        node.this.set("quoted", False)
                        never_quoted_node[node] = True

        if isinstance(node, exp.Identifier):
            if node in never_quoted_node:
                continue
            name = node.this
            if any(char.islower() for char in name):
                node.set("quoted", True)

        if isinstance(node, (exp.Alias, exp.TableAlias)):
            node.set("columns", None)

    return expression.sql(dialect=output_dialect, identify=False, pretty=True)


def add_sql_comment(sql_query: str, column_name: str, comment: str, dialect: str = "snowflake") -> str:
    """Add a trailing inline comment to *column_name* in *sql_query*."""
    try:
        ast = parse_one(sql_query, read=dialect)

        def find_and_comment(node):
            if isinstance(node, (exp.Column, exp.Alias)):
                node_name = ""
                if isinstance(node, exp.Alias):
                    node_name = node.alias_or_name
                elif isinstance(node, exp.Column):
                    node_name = node.this.name
                if node_name == column_name:
                    if node.comments is None:
                        node.comments = []
                    node.comments.append(comment)
            return node

        transformed_ast = ast.transform(find_and_comment)
        return transformed_ast.sql(dialect=dialect, pretty=True)
    except Exception:
        return sql_query


def post_format_generated_query(query: str, db_path: str, db_type: str = "snowflake", include_comment: bool = False) -> str:
    """Format a generated query: optionally annotate columns then quote identifiers."""
    if include_comment:
        parsed_ast = parse_one(query, read=db_type)
        table_names = {table.sql().split()[0] for table in parsed_ast.find_all(exp.Table)}

        for table in table_names:
            pattern = r"'|\"|`"
            table_name = re.sub(pattern, "", table)
            if db_type == "snowflake":
                schema = "/".join(table_name.replace(".", "/").split("/")[:-1])
                table_name = table_name.replace(".", "/").split("/")[-1]
                meta_data_path = None
                schema_dir = os.path.join(db_path, "resource/databases_no_nulls_2", schema)
                if os.path.isdir(schema_dir):
                    for file in os.listdir(schema_dir):
                        if file.endswith(".json") and file.split(".")[0].lower() == table_name.lower():
                            meta_data_path = os.path.join(schema_dir, file)
                            break
                if meta_data_path is None:
                    continue
                with open(meta_data_path, "r") as f:
                    meta_data = json.load(f)
                for column_name, comment, dtype in zip(
                    meta_data["column_names"], meta_data["description"], meta_data["column_types"]
                ):
                    if comment:
                        query = add_sql_comment(query, column_name=column_name, comment=f"{dtype} {comment}")
                    else:
                        query = add_sql_comment(query, column_name=column_name, comment=f"{dtype}")

    query = quote_identifiers(query, output_dialect=db_type)
    return query


# ── schema_linking helpers ─────────────────────────────────────────────────────

def execute_query(cursor, query: str, n_rows: int) -> Tuple[List[str], List[tuple]]:
    """Execute *query*, appending LIMIT if absent, and return (headers, data_rows)."""
    limit_pattern = r"\bLIMIT\s+\d+"
    if not re.search(limit_pattern, query, re.IGNORECASE):
        query = query.rstrip().rstrip(";")
        query = f"{query} LIMIT {n_rows}"

    cursor.execute(query)
    headers = [description[0] for description in cursor.description]
    data_rows = cursor.fetchall()
    return headers, data_rows


def format_results_to_markdown(
    headers: List[str],
    data_rows: List[tuple],
    truncate_data: bool = True,
    max_truncate_len: int = 20,
) -> str:
    """Render *(headers, data_rows)* as a GitHub-flavoured markdown table."""
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in data_rows:
        processed_cells = []
        for value in row:
            if value is None:
                str_value = "NULL"
            elif isinstance(value, float) and math.isnan(value):
                str_value = "NaN"
            elif "bytearray" in str(value).lower():
                str_value = "bytearray(b'...')"
            else:
                if isinstance(value, str):
                    try:
                        if value.strip().startswith("[") or value.strip().startswith("{"):
                            value = json.loads(value)
                        if truncate_data:
                            value = truncate_nested_data(value, max_str_len=max_truncate_len)
                    except json.JSONDecodeError:
                        value = (
                            value[:max_truncate_len] + "..."
                            if len(value) > max_truncate_len and truncate_data
                            else value
                        )
                str_value = str(value).replace("\n", " ").replace("|", "\\|")
            processed_cells.append(str_value)
        markdown += "| " + " | ".join(map(str, processed_cells)) + " |\n"
    return markdown


def execute_and_format_query_result(
    cursor,
    query: str,
    db_path: str,
    db_type: str,
    n_example_rows: int,
    truncate_data: bool = True,
    max_truncate_len: int = 20,
    include_comment: bool = False,
    include_query: bool = True,
) -> str:
    """Execute *query* and return a formatted markdown result block."""
    query = post_format_generated_query(query, db_path=db_path, db_type=db_type, include_comment=False)
    headers, data_rows = execute_query(cursor, query, n_rows=n_example_rows)

    if include_comment:
        query = post_format_generated_query(query, db_path=db_path, db_type=db_type, include_comment=True)

    result_markdown = format_results_to_markdown(
        headers, data_rows, truncate_data=truncate_data, max_truncate_len=max_truncate_len
    )

    n_returned = len(data_rows)
    if n_returned == 0:
        header = "0 rows returned (query executed successfully, empty result set):"
    elif n_returned < n_example_rows:
        header = f"{n_returned} example rows:"
    else:
        header = f"{n_example_rows} example rows:"
    if include_query:
        return f"```sql\n{query}\n```\n\n{header}\n{result_markdown}"
    else:
        return f"{header}\n{result_markdown}"


# ── tools factory functions ────────────────────────────────────────────────────

def create_list_tables_tool(db_type: str, database_name: str, db_path: str) -> Callable:
    """Return a callable that lists tables in a schema from JSON metadata files."""

    def get_base_folder() -> str:
        if db_type == "snowflake":
            return os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
        return os.path.join(db_path, database_name)

    def list_tables(schema_name: str) -> str:
        try:
            base_folder = get_base_folder()
            if not os.path.exists(base_folder):
                return f"Database folder not found: {base_folder}"

            tables = []
            if db_type == "snowflake":
                schema_path = os.path.join(base_folder, schema_name)
                if not os.path.isdir(schema_path):
                    return f"Schema '{schema_name}' not found."
                for file in os.listdir(schema_path):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            with open(os.path.join(schema_path, file)) as f:
                                metadata = json.load(f)
                            tables.append(
                                metadata.get(
                                    "table_fullname",
                                    f"{database_name}.{schema_name}.{file.replace('.json', '')}",
                                )
                            )
                        except Exception:
                            continue
            else:
                for file in os.listdir(base_folder):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            with open(os.path.join(base_folder, file)) as f:
                                metadata = json.load(f)
                            tables.append(metadata.get("table_name", file.replace(".json", "")))
                        except Exception:
                            continue

            if not tables:
                return f"No tables with JSON metadata files found in schema '{schema_name}'."
            result = f"Tables in schema '{schema_name}' (with JSON metadata):\n"
            result += "\n".join(f"- {t}" for t in sorted(tables))
            return result
        except Exception as e:
            return f"Error listing tables: {e}"

    return list_tables


def create_list_columns_tool(db_type: str, database_name: str, db_path: str) -> Callable:
    """Return a callable that lists columns for a table from JSON metadata files."""

    def get_base_folder() -> str:
        if db_type == "snowflake":
            return os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
        return os.path.join(db_path, database_name)

    def list_columns(table_name: str) -> str:
        try:
            base_folder = get_base_folder()
            if not os.path.exists(base_folder):
                return f"Database folder not found: {base_folder}"

            if db_type == "snowflake":
                parts = table_name.split(".")
                if len(parts) != 3:
                    return f"Error: Table name must be 'DATABASE.SCHEMA.TABLE', got '{table_name}'"
                _, schema, table = parts
                schema_path = os.path.join(base_folder, schema)
                if not os.path.isdir(schema_path):
                    return f"Schema '{schema}' not found."

                json_file = None
                for file in os.listdir(schema_path):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            json_path = os.path.join(schema_path, file)
                            with open(json_path) as f:
                                metadata = json.load(f)
                            fqn = metadata.get("table_fullname", "")
                            if fqn.upper() == table_name.upper() or fqn.split(".")[-1].upper() == table.upper():
                                json_file = json_path
                                break
                        except Exception:
                            continue

                if not json_file:
                    return f"Table '{table_name}' not found or has no JSON metadata file."

                with open(json_file) as f:
                    metadata = json.load(f)

                column_names = metadata.get("column_names", [])
                column_types = metadata.get("column_types", [])
                descriptions = metadata.get("description", [])
                column_examples = metadata.get("column_examples", {})

                if not column_names:
                    return f"Table '{table_name}' has no columns in metadata."

                result = f"Columns in table '{table_name}':\n"
                for i, col_name in enumerate(column_names):
                    col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                    desc = descriptions[i] if i < len(descriptions) and descriptions[i] else ""
                    desc_str = f" - {desc}" if desc else ""
                    examples = column_examples.get(col_name, [])
                    if examples:
                        example_values = []
                        for ex in examples[:1]:
                            try:
                                example_values.append(str(truncate_nested_data(ex)))
                            except Exception:
                                example_values.append(str(ex))
                        examples_str = ", ".join(f"'{v}'" for v in example_values)
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: [{examples_str}]\n"
                    else:
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: []\n"
                return result.rstrip()

            else:
                actual = table_name.split(".")[-1] if "." in table_name else table_name
                json_file = None
                for file in os.listdir(base_folder):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            json_path = os.path.join(base_folder, file)
                            with open(json_path) as f:
                                metadata = json.load(f)
                            if metadata.get("table_name", file.replace(".json", "")).upper() == actual.upper():
                                json_file = json_path
                                break
                        except Exception:
                            continue

                if not json_file:
                    return f"Table '{actual}' not found or has no JSON metadata file."

                with open(json_file) as f:
                    metadata = json.load(f)

                column_names = metadata.get("column_names", [])
                column_types = metadata.get("column_types", [])
                descriptions = metadata.get("description", [])
                column_examples = metadata.get("column_examples", {})

                if not column_names:
                    return f"Table '{actual}' has no columns in metadata."

                result = f"Columns in table '{actual}':\n"
                for i, col_name in enumerate(column_names):
                    col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                    desc = descriptions[i] if i < len(descriptions) and descriptions[i] else ""
                    desc_str = f" - {desc}" if desc else ""
                    examples = column_examples.get(col_name, [])
                    if examples:
                        example_values = []
                        for ex in examples[:2]:
                            try:
                                example_values.append(str(truncate_nested_data(ex)))
                            except Exception:
                                example_values.append(str(ex))
                        examples_str = ", ".join(f"'{v}'" for v in example_values)
                        if len(examples) > 2:
                            examples_str += " ... )"
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: [{examples_str}]\n"
                    else:
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: []\n"
                return result.rstrip()

        except Exception as e:
            return f"Error listing columns: {e}"

    return list_columns


def create_get_distinct_values_tool(db_type: str, cursor_getter: Callable) -> Callable:
    """Return a callable that fetches up to 100 distinct values from a column."""

    def get_distinct_values(table: str, column: str, truncate: bool = True) -> str:
        cursor = cursor_getter()
        column = column.replace('"', "").replace("'", "")
        try:
            if db_type == "snowflake":
                quoted_table = ".".join(f'"{p}"' for p in table.split(".")) if table else '""'
                quoted_column = f'"{column}"' if column else '""'
                query = f"SELECT DISTINCT {quoted_column} FROM {quoted_table} ORDER BY {quoted_column} LIMIT 100"
            else:
                query = f'SELECT DISTINCT "{column}" FROM "{table}" ORDER BY "{column}" LIMIT 100'

            _, data_rows = execute_query(cursor, query, n_rows=100)

            if not data_rows:
                return f"No distinct values found in column {column} of table {table}."

            def _fmt(val):
                if val is None:
                    return "NULL"
                s = str(val)
                if truncate and len(s) > 200:
                    return s[:100] + "..." + s[-97:]
                return s

            values = [_fmt(row[0]) for row in data_rows]
            result = f"Distinct values in {table}.{column}:\n" + "\n".join(f"- {v}" for v in values)
            if len(data_rows) >= 100:
                result += "\n(Showing first 100 distinct values — there may be more)"
            return result
        except Exception as e:
            return f"Error getting distinct values: {e}"

    return get_distinct_values


def create_search_dimension_values_tool(db_type: str, cursor_getter: Callable) -> Callable:
    """Return a callable that searches a column for rows matching any of the given terms."""

    def search_dimension_values(
        terms: list,
        table: str,
        column_name: str = "description",
        additional_columns: Optional[list] = None,
        truncate: bool = True,
    ) -> str:
        if not terms:
            return "Error: No search terms provided."
        cursor = cursor_getter()
        try:
            safe_terms = [str(t).replace("'", "''") for t in terms]

            def sanitize_col(c: str) -> str:
                return c.replace('"', "").replace("'", "")

            column_name = sanitize_col(column_name)
            extra_cols = [
                sanitize_col(c)
                for c in (additional_columns or [])
                if c and sanitize_col(c) != column_name
            ]

            if db_type == "snowflake":
                quoted_table = ".".join(f'"{p}"' for p in table.split(".")) if table else '""'
                quoted_column = f'"{column_name}"' if column_name else '""'
                select_cols = ", ".join([quoted_column] + [f'"{c}"' for c in extra_cols])
                conditions = [f"{quoted_column} ILIKE '%{t}%'" for t in safe_terms]
                where_clause = " OR ".join(conditions)
                query = f"SELECT {select_cols} FROM {quoted_table} WHERE {where_clause}"
            else:
                quoted_column = f'"{column_name}"'
                select_cols = ", ".join([quoted_column] + [f'"{c}"' for c in extra_cols])
                conditions = [f'LOWER("{column_name}") LIKE LOWER(\'%{t}%\')' for t in safe_terms]
                where_clause = " OR ".join(conditions)
                query = f'SELECT {select_cols} FROM "{table}" WHERE {where_clause}'

            headers, data_rows = execute_query(cursor, query, n_rows=10)
            if not data_rows:
                return f"No rows found matching '{', '.join(terms)}' in {table}.{column_name}."

            result_markdown = format_results_to_markdown(
                headers, data_rows, truncate_data=truncate, max_truncate_len=200
            )
            return f"Search results for '{', '.join(terms)}' in {table}.{column_name}:\n\n{result_markdown}"
        except Exception as e:
            return f"Error searching dimension values: {e}"

    return search_dimension_values


# ── python interpreter tool ────────────────────────────────────────────────────

def create_python_interpreter_tool(
    db_type: str,
    db_path: str,
    db_id: str,
) -> Callable:
    """Return a stateful python_interpreter callable.

    Each call runs *code* in an isolated subprocess (via python_interpreter_worker.py).
    The worker has access to all six database tool functions mirroring the MCP tools:
      list_schemas(), list_tables(schema), list_columns(table),
      query_database(query), get_distinct_values(table, col),
      search_dimension_values(terms, table, col, extra_cols)
    as well as pd, np, and standard Python builtins.
    """
    _WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_interpreter_worker.py")
    _TIMEOUT = 120

    _tmpdir_obj = tempfile.TemporaryDirectory()
    _tmpdir = _tmpdir_obj.name

    # Resolve to absolute paths before pickling — the worker runs with cwd=tmpdir
    abs_db_path = os.path.abspath(db_path)

    # Pass db context to worker via initial_data pickle
    initial_data: Dict[str, Any] = {
        "_db_path": abs_db_path,
        "_db_id": db_id,
        "_db_type": db_type,
    }
    # Snowflake credential path so worker can connect
    cred_path = os.path.join(abs_db_path, "snowflake_credential.json")
    if os.path.isfile(cred_path):
        initial_data["_cred_path"] = cred_path

    _init_pkl = os.path.join(_tmpdir, "initial_data.pkl")
    with open(_init_pkl, "wb") as f:
        pickle.dump(initial_data, f)

    _state_pkl = os.path.join(_tmpdir, "state.pkl")

    def python_interpreter(code: str) -> str:
        """Execute Python code in a stateful interpreter.

        Database tools available (mirror the MCP tools exactly):
          list_schemas()                                           — list schemas in current DB
          list_tables(schema_name)                                 — list tables in a schema
          list_columns(table_name)                                 — columns + types + examples
          query_database(query)                                    — run SELECT, get markdown table
          get_distinct_values(table, column)                       — up to 100 distinct values
          search_dimension_values(terms, table, col, extra=[])     — ILIKE/LIKE search

        Also available: pd, np, and all standard Python builtins.
        State (variables) persists across multiple calls.
        """
        _ = _tmpdir_obj  # keep alive
        code_file = os.path.join(_tmpdir, "code.py")
        with open(code_file, "w") as f:
            f.write(code)

        cmd = [sys.executable, _WORKER, code_file, _init_pkl, _state_pkl, _state_pkl]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT, cwd=_tmpdir)
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            if not out and err:
                return err
            return out if out else "No output."
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {_TIMEOUT} seconds."

    return python_interpreter
