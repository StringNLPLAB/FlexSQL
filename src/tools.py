"""
Tool definitions for the Chat class.

This module contains all tool functions and their OpenAI/Gemini tool definitions
that can be used with the Chat class for tool calling functionality.
"""
import os
import json
from typing import Callable, Optional, Any, Dict


def create_query_database_tool(db_path: str, db_type: str, cursor_getter: Callable, n_example_rows: int = 1) -> Callable:
    """Create a query_database tool function.
    
    Args:
        db_path: Path to the database folder
        db_type: Database type ("snowflake" or "sqlite")
        cursor_getter: Callable that returns a database cursor when called
        n_example_rows: Number of example rows to return from queries
    
    Returns:
        A callable tool function
    """
    from schema_linking import execute_and_format_query_result
    
    def query_database(query: str) -> str:
        """Execute a SQL query on the database and return formatted results with example rows.
        
        Use this tool when you need to clarify user intentions by examining actual database values.
        This is especially useful during planning when the user's question is ambiguous or you need
        to understand what values exist in specific columns.
        
        Args:
            query: The SQL query to execute
        
        Returns:
            Formatted string containing the SQL query and example rows from the result
        """
        cursor = cursor_getter()
        try:
            # should allow a bit longer token limit to this
            formatted_string = execute_and_format_query_result(
                cursor=cursor,
                query=query,
                db_path=db_path,
                db_type=db_type,
                n_example_rows=n_example_rows,
                truncate_data=True,
                max_truncate_len=200,
                include_comment=False,
                include_query=False
            )
            return formatted_string
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    return query_database


def create_get_distinct_values_tool(db_type: str, cursor_getter: Callable) -> Callable:
    """Create a get_distinct_values tool function.
    
    Args:
        db_type: Database type ("snowflake" or "sqlite")
        cursor_getter: Callable that returns a database cursor when called
    
    Returns:
        A callable tool function
    """
    from schema_linking import execute_query
    
    def get_distinct_values(table: str, column: str) -> str:
        """Get distinct values from a specific column in a table.
        
        Use this tool to see what categorical values exist in a column. This is useful
        when the user mentions a category or type that you need to verify exists in the database.
        
        Args:
            table: The table name (e.g., 'DATABASE.SCHEMA.TABLE' for Snowflake or 'table' for SQLite)
            column: The column name to get distinct values from
        
        Returns:
            Formatted string containing the distinct values from the column
        """
        cursor = cursor_getter()
        column = column.replace('"', "").replace("'", "")
        try:
            # Build query to get distinct values
            if db_type == "snowflake":
                # For Snowflake, split table by "." and quote each component, quote column with ""
                if table:
                    quoted_table = ".".join(f'"{part}"' for part in table.split("."))
                else:
                    quoted_table = '""'
                quoted_column = f'"{column}"' if column else '""'
                query = f"SELECT DISTINCT {quoted_column} FROM {quoted_table} ORDER BY {quoted_column} LIMIT 100"
            else:
                # For SQLite, use simple identifiers
                query = f'SELECT DISTINCT "{column}" FROM "{table}" ORDER BY "{column}" LIMIT 100'
            
            # Execute query directly
            _, data_rows = execute_query(cursor, query, n_rows=100)
            
            if not data_rows:
                return f"No distinct values found in column {column} of table {table}."
            
            # Format as a simple list
            values = [str(row[0]) if row[0] is not None else "NULL" for row in data_rows]
            result = f"Distinct values in {table}.{column}:\n" + "\n".join(f"- {val}" for val in values)
            if len(data_rows) >= 100:
                result += f"\n(Showing first 100 distinct values - there may be more)"
            return result
        except Exception as e:
            return f"Error getting distinct values: {str(e)}"
    
    return get_distinct_values


def create_search_dimension_values_tool(db_type: str, cursor_getter: Callable) -> Callable:
    """Create a search_dimension_values tool function.
    
    Args:
        db_type: Database type ("snowflake" or "sqlite")
        cursor_getter: Callable that returns a database cursor when called
    
    Returns:
        A callable tool function
    """
    from schema_linking import execute_query, format_results_to_markdown
    
    def search_dimension_values(terms: list, table: str, column_name: str = "description", additional_columns: list = None) -> str:
        """Searches a specific table for rows where the column contains the term(s).

        Args:
            terms: A list of search terms to look for (case-insensitive partial match). A single term is a single-element array.
            table: The table name to search in (e.g., 'DATABASE.SCHEMA.TABLE' for Snowflake or 'table' for SQLite)
            column_name: The column name to search in (defaults to "description")
            additional_columns: Optional list of extra column names to include in the results alongside column_name.

        Returns:
            Formatted string containing matching rows
        """
        if not terms:
            return "Error: No search terms provided."
        cursor = cursor_getter()
        try:
            # Sanitize inputs to prevent SQL injection!
            safe_terms = [str(t).replace("'", "''") for t in terms]

            def sanitize_col(c: str) -> str:
                """Strip quotes from a column name to get a safe identifier."""
                return c.replace('"', "").replace("'", "")

            column_name = sanitize_col(column_name)
            extra_cols = [sanitize_col(c) for c in (additional_columns or []) if c and sanitize_col(c) != column_name]

            # Construct a query that looks for any of the terms in the specified column
            if db_type == "snowflake":
                # For Snowflake, split table by "." and quote each component; quote columns with ""
                quoted_table = ".".join(f'"{part}"' for part in table.split(".")) if table else '""'
                quoted_column = f'"{column_name}"' if column_name else '""'
                select_cols = ", ".join([quoted_column] + [f'"{c}"' for c in extra_cols])
                conditions = [f"{quoted_column} ILIKE '%{safe_t}%'" for safe_t in safe_terms]
                where_clause = " OR ".join(conditions)
                query = f"SELECT {select_cols} FROM {quoted_table} WHERE {where_clause} LIMIT 5"
            else:
                # SQLite uses LIKE (case-insensitive with LOWER)
                quoted_column = f'"{column_name}"'
                select_cols = ", ".join([quoted_column] + [f'"{c}"' for c in extra_cols])
                conditions = [f'LOWER("{column_name}") LIKE LOWER(\'%{safe_t}%\')' for safe_t in safe_terms]
                where_clause = " OR ".join(conditions)
                query = f'SELECT {select_cols} FROM "{table}" WHERE {where_clause} LIMIT 5'

            # Execute and format the query
            headers, data_rows = execute_query(cursor, query, n_rows=10)

            if not data_rows:
                terms_str = ", ".join(terms)
                return f"No rows found matching '{terms_str}' in {table}.{column_name}."

            result_markdown = format_results_to_markdown(headers, data_rows)
            terms_str = ", ".join(terms)
            return f"Search results for '{terms_str}' in {table}.{column_name}:\n\n{result_markdown}"
        except Exception as e:
            return f"Error searching dimension values: {str(e)}"
    
    return search_dimension_values



def create_python_interpreter_tool(data_frames_map: Optional[Dict[str, Any]] = None,
                                   additional_context: Optional[Dict[str, Any]] = None,
                                   cursor_getter: Optional[Callable] = None,
                                   db_type: Optional[str] = None,
                                   db_connection_str: Optional[str] = None) -> Callable:
    """Create a python_interpreter tool function.

    Each call to the returned tool runs the provided code in an isolated subprocess,
    protecting the parent process from OOM. State (user-defined variables) is
    persisted across calls via a pickle file in a temporary directory.

    Args:
        data_frames_map: Optional dictionary mapping variable names to DataFrames (or lists of DataFrames)
            to make available in the interpreter. E.g. {"dfs": [df1, df2], "program_output": outputs}
        additional_context: Optional dictionary of additional variables to inject into the interpreter context
        cursor_getter: Optional callable that returns a database cursor. Used to auto-detect the
            SQLite file path (via PRAGMA database_list) when db_connection_str is not provided.
        db_type: Database type ("snowflake" or "sqlite"), used to set up the right DB variable.
        db_connection_str: Path to the .sqlite file (SQLite) or snowflake_credential.json (Snowflake).
            If None and db_type=="sqlite", the path is extracted from cursor_getter automatically.

    Returns:
        A callable tool function
    """
    import sys as _sys
    try:
        import cloudpickle as _pickle
    except ImportError:
        import pickle as _pickle
    import subprocess as _subprocess
    import tempfile as _tempfile

    # Build description of available DB access (for the tool's docstring)
    _db_access_desc = ""
    if cursor_getter is not None:
        if db_type == "sqlite":
            _db_access_desc = (
                "IMPORTANT: A pre-connected sqlite3.Connection is injected as `conn` — do NOT call sqlite3.connect().\n"
                "- conn.execute(sql).fetchall()  — run a query directly\n"
                "- pd.read_sql_query(sql, conn)  — load results into a DataFrame\n"
                "- get_cursor()                  — returns a fresh cursor from the same connection\n"
            )
        elif db_type == "snowflake":
            _db_access_desc = (
                "IMPORTANT: A pre-connected Snowflake cursor is injected as `cursor` — do NOT create new connections.\n"
                "- cursor.execute(sql); cursor.fetchall()  — run a query\n"
                "- get_cursor()                            — returns a fresh cursor from the same connection\n"
            )
        else:
            _db_access_desc = "- get_cursor(): Returns a database cursor to run queries.\n"

    # Auto-detect SQLite file path from cursor_getter if not provided
    _resolved_db_connection_str = db_connection_str
    if cursor_getter is not None and db_type == "sqlite" and not _resolved_db_connection_str:
        try:
            cur = cursor_getter()
            rows = cur.execute("PRAGMA database_list").fetchall()
            for _seq, _name, _file in rows:
                if _name == "main" and _file:
                    _resolved_db_connection_str = _file
                    break
        except Exception:
            pass

    # Persistent temp dir — lives as long as the returned closure keeps a reference
    _tmpdir_obj = _tempfile.TemporaryDirectory()
    _tmpdir = _tmpdir_obj.name

    # Pickle initial data once (DataFrames + picklable additional_context items)
    _initial_data: Dict[str, Any] = {}
    if data_frames_map:
        _initial_data.update(data_frames_map)
    if additional_context:
        for k, v in additional_context.items():
            try:
                _pickle.dumps(v)
                _initial_data[k] = v
            except Exception:
                pass
    _init_pkl = os.path.join(_tmpdir, "initial_data.pkl")
    with open(_init_pkl, "wb") as _f:
        _pickle.dump(_initial_data, _f)

    _state_pkl = os.path.join(_tmpdir, "state.pkl")
    _db_arg = os.path.abspath(_resolved_db_connection_str) if _resolved_db_connection_str else ""
    _db_type_str = db_type or ""
    _WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "python_interpreter_worker.py")
    _TIMEOUT = 120  # seconds per tool call

    def python_interpreter(code: str) -> str:
        """Execute Python code in a stateful interpreter.

        The interpreter has access to:
        - pd, np, nx, gpd: Common data science libraries
        - Variables from data_frames_map (e.g. dfs, program_output)
        - All standard Python builtins
        - __name__ is set to '__main__' so if __name__ == '__main__': blocks will execute
        - All imports and variables persist across multiple executions (stateful)
        - Database access (if configured): conn / cursor / get_cursor()

        You can test your processing function by:
        1. Defining the processing function
        2. Calling it with the appropriate dataframe variable: result = processing(dfs)
        3. Checking the result: print(result.head()) or print(len(result))
        """
        # Keep _tmpdir_obj alive as long as this closure exists
        _ = _tmpdir_obj
        code_file = os.path.join(_tmpdir, "code.py")
        with open(code_file, "w") as f:
            f.write(code)
        cmd = [
            _sys.executable, _WORKER,
            code_file, _init_pkl, _state_pkl, _state_pkl,
            _db_type_str, _db_arg,
        ]
        try:
            proc = _subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT, cwd=_tmpdir)
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            if not out and err:
                return err
            return out if out else "No output."
        except _subprocess.TimeoutExpired:
            return f"Execution timed out after {_TIMEOUT} seconds."

    return python_interpreter


def create_list_tables_tool(db_type: str, database_name: str, db_path: str) -> Callable:
    """Create a list_tables tool function.
    
    Args:
        db_type: Database type ("snowflake" or "sqlite")
        database_name: Name of the database
        db_path: Path to the database folder
    
    Returns:
        A callable tool function
    """
    def get_base_folder() -> str:
        if db_type == "snowflake":
            return os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
        else:
            # SQLite: return db_path directly
            return os.path.join(db_path, database_name)
    
    def list_tables(schema_name: str) -> str:
        """List all tables in a specific schema that have JSON metadata files.
        
        Args:
            schema_name: The schema name to list tables from (for SQLite, use "main")
        
        Returns:
            Formatted string containing a list of fully qualified table names in the schema
        """
        try:
            base_folder = get_base_folder()
            if not os.path.exists(base_folder):
                return f"Database folder not found: {base_folder}"
            
            tables = []
            
            if db_type == "snowflake":
                schema_path = os.path.join(base_folder, schema_name)
                if not os.path.isdir(schema_path):
                    return f"Schema '{schema_name}' not found."
                
                # List all JSON files in the schema folder
                for file in os.listdir(schema_path):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            json_path = os.path.join(schema_path, file)
                            with open(json_path, 'r') as f:
                                metadata = json.load(f)
                            table_fullname = metadata.get("table_fullname", f"{database_name}.{schema_name}.{file.replace('.json', '')}")
                            tables.append(table_fullname)
                        except Exception as e:
                            # Skip files that can't be read
                            continue
            else:
                # SQLite: list JSON files in the database folder
                for file in os.listdir(base_folder):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            json_path = os.path.join(base_folder, file)
                            with open(json_path, 'r') as f:
                                metadata = json.load(f)
                            table_name = metadata.get("table_name", file.replace(".json", ""))
                            tables.append(table_name)
                        except Exception as e:
                            continue
            
            if not tables:
                return f"No tables with JSON metadata files found in schema '{schema_name}'."
            
            result = f"Tables in schema '{schema_name}' (with JSON metadata):\n"
            result += "\n".join(f"- {table}" for table in sorted(tables))
            return result
        except Exception as e:
            return f"Error listing tables: {str(e)}"
    
    return list_tables


def create_list_columns_tool(db_type: str, database_name: str, db_path: str) -> Callable:
    """Create a list_columns tool function.
    
    Args:
        db_type: Database type ("snowflake" or "sqlite")
        database_name: Name of the database
        db_path: Path to the database folder
    
    Returns:
        A callable tool function
    """
    from get_ddl import truncate_nested_data
    
    def get_base_folder() -> str:
        if db_type == "snowflake":
            return os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
        else:
            # SQLite: return db_path directly
            return os.path.join(db_path, database_name)
    
    def list_columns(table_name: str) -> str:
        """List all columns in a specific table from JSON metadata file.
        Includes column names, data types, descriptions, and example values.
        
        Args:
            table_name: Fully qualified table name (e.g., 'DATABASE.SCHEMA.TABLE' for Snowflake or 'table' for SQLite)
        
        Returns:
            Formatted string containing column names, data types, descriptions, and example values
        """
        try:
            base_folder = get_base_folder()
            if not os.path.exists(base_folder):
                return f"Database folder not found: {base_folder}"
            
            if db_type == "snowflake":
                # Parse the fully qualified table name
                parts = table_name.split(".")
                if len(parts) != 3:
                    return f"Error: Table name must be fully qualified as 'DATABASE.SCHEMA.TABLE', got '{table_name}'"
                
                _, schema, table = parts
                
                # Find the JSON file for this table
                schema_path = os.path.join(base_folder, schema)
                if not os.path.isdir(schema_path):
                    return f"Schema '{schema}' not found."
                
                # Look for JSON file matching the table name
                json_file = None
                for file in os.listdir(schema_path):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        # Check if this JSON file matches the table
                        try:
                            json_path = os.path.join(schema_path, file)
                            with open(json_path, 'r') as f:
                                metadata = json.load(f)
                            metadata_table_fullname = metadata.get("table_fullname", "")
                            if metadata_table_fullname.upper() == table_name.upper() or metadata_table_fullname.split(".")[-1].upper() == table.upper():
                                json_file = json_path
                                break
                        except:
                            continue
                
                if not json_file:
                    return f"Table '{table_name}' not found or has no JSON metadata file."
                
                # Read metadata
                with open(json_file, 'r') as f:
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
                    
                    # Get example values for this column
                    examples = column_examples.get(col_name, [])
                    if examples:
                        # Show up to 1 example values
                        example_values = []
                        for ex in examples[:1]:
                            try:
                                truncated = truncate_nested_data(ex)
                                example_values.append(str(truncated))
                            except:
                                example_values.append(str(ex))
                        examples_str = ", ".join([f"'{val}'" for val in example_values])
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: [{examples_str}]\n"
                    else:
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: []\n"
                
                return result.rstrip()
            else:
                # SQLite: find JSON file by table name
                actual_table_name = table_name.split(".")[-1] if "." in table_name else table_name
                
                json_file = None
                for file in os.listdir(base_folder):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        try:
                            json_path = os.path.join(base_folder, file)
                            with open(json_path, 'r') as f:
                                metadata = json.load(f)
                            metadata_table_name = metadata.get("table_name", file.replace(".json", ""))
                            if metadata_table_name.upper() == actual_table_name.upper():
                                json_file = json_path
                                break
                        except:
                            continue
                
                if not json_file:
                    return f"Table '{actual_table_name}' not found or has no JSON metadata file."
                
                # Read metadata
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                column_names = metadata.get("column_names", [])
                column_types = metadata.get("column_types", [])
                descriptions = metadata.get("description", [])
                column_examples = metadata.get("column_examples", {})
                
                if not column_names:
                    return f"Table '{actual_table_name}' has no columns in metadata."
                
                result = f"Columns in table '{actual_table_name}':\n"
                for i, col_name in enumerate(column_names):
                    col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                    desc = descriptions[i] if i < len(descriptions) and descriptions[i] else ""
                    desc_str = f" - {desc}" if desc else ""
                    
                    # Get example values for this column
                    examples = column_examples.get(col_name, [])
                    if examples:
                        # Show up to 2 example values
                        example_values = []
                        for ex in examples[:2]:
                            try:
                                truncated = truncate_nested_data(ex)
                                example_values.append(str(truncated))
                            except:
                                example_values.append(str(ex))
                        examples_str = ", ".join([f"'{val}'" for val in example_values])
                        if len(examples) > 2:
                            examples_str += f" ... )"
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: [{examples_str}]\n"
                    else:
                        result += f"- {col_name}: {col_type}{desc_str}\n  Examples: []\n"
                
                return result.rstrip()
        except Exception as e:
            return f"Error listing columns: {str(e)}"
    
    return list_columns


def _extract_referenced_table_names(program_content: str) -> set:
    """Extract table names referenced after FROM/JOIN keywords in SQL or embedded SQL strings.

    Handles unquoted names and identifiers quoted with double-quotes, backticks,
    or square brackets (common across SQLite, Snowflake, MySQL, SQL Server).
    """
    import re
    # Match: FROM/JOIN  optional-open-quote  word  optional-close-quote
    names = set(re.findall(
        r'\b(?:FROM|JOIN)\s+["`\[]?([a-zA-Z_][a-zA-Z0-9_]*)["`\]]?',
        program_content,
        re.IGNORECASE,
    ))
    _KEYWORDS = {
        'select', 'where', 'order', 'group', 'having', 'limit', 'union',
        'with', 'as', 'on', 'set', 'inner', 'outer', 'left', 'right',
        'cross', 'natural', 'lateral', 'values', 'distinct', 'all',
    }
    return {n for n in names if n.lower() not in _KEYWORDS}


def _format_column_descriptions(
    db_path: str,
    db_id: str,
    table_names: set,
    used_columns: Optional[dict] = None,
) -> str:
    """
    Load database_description CSVs for the given tables and return a formatted
    summary string ready to be injected as a comment block.

    Expected CSV path: {db_path}/{db_id}/database_description/{table_name}.csv
    CSV columns: original_column_name, column_name, column_description,
                 data_format, value_description

    Args:
        used_columns: Optional dict mapping table name → list of original_column_names
                      actually referenced by the program (sourced from plans.json).
                      When provided, only those columns' descriptions are shown.
                      When None, all columns are shown.
    Returns "" if no description files are found.
    """
    import csv as _csv

    desc_dir = os.path.join(db_path, db_id, "database_description")
    if not os.path.isdir(desc_dir):
        return ""

    try:
        available = {f.lower(): f for f in os.listdir(desc_dir) if f.endswith(".csv")}
    except OSError:
        return ""

    # Build a case-insensitive lookup: table_lower → {col_orig_lower, ...}
    used_cols_lower: dict = {}
    if used_columns:
        for tbl, cols in used_columns.items():
            used_cols_lower[tbl.lower()] = {c.lower() for c in cols}

    blocks = []
    for table in sorted(table_names):
        fname = available.get(f"{table}.csv") or available.get(f"{table.lower()}.csv")
        if not fname:
            continue
        csv_path = os.path.join(desc_dir, fname)
        try:
            with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue

        # Column filter for this table (None → accept all)
        allowed_cols = used_cols_lower.get(table.lower()) if used_cols_lower else None

        lines = [f"Table: {table}"]
        for row in rows:
            col_orig  = (row.get("original_column_name") or "").strip()
            col_alias = (row.get("column_name") or "").strip()
            col_desc  = (row.get("column_description") or "").strip()
            data_fmt  = (row.get("data_format") or "").strip()
            val_desc  = (row.get("value_description") or "").strip()

            # Skip columns not used by this program
            if allowed_cols is not None and col_orig.lower() not in allowed_cols:
                continue

            display = col_alias if col_alias else col_orig
            if not display:
                continue

            entry = f"  {display}"
            if data_fmt:
                entry += f" ({data_fmt})"
            if col_desc and col_desc.lower() not in ("unuseful", display.lower()):
                entry += f": {col_desc}"
            if val_desc and val_desc.lower() != "unuseful":
                val_lines = val_desc.strip().splitlines()
                entry += f" → {val_lines[0]}"
                for vl in val_lines[1:]:
                    entry += f"\n    {vl.strip()}"
            lines.append(entry)

        if len(lines) > 1:
            blocks.append("\n".join(lines))

    if not blocks:
        return ""
    return "Column Descriptions:\n\n" + "\n\n".join(blocks)


def create_read_program_tool(
    program_dir: str,
    input_dataframes_formatter: Optional[Callable[[Optional[int]], str]] = None,
    db_path: Optional[str] = None,
    db_id: Optional[str] = None,
) -> Callable:
    """Create a read_program tool function.

    Args:
        program_dir: Path to the directory containing program files.
        input_dataframes_formatter: Optional function(plan_idx) → formatted sub-query text.
        db_path: Root database folder (e.g. dev_20240627/dev_databases). When
                 provided together with db_id, column descriptions from
                 database_description CSVs are prepended to the program.
        db_id: Database name as it appears in the dataset (e.g. 'california_schools').
    """
    def _to_comments(text: str, prefix: str) -> str:
        """Convert plain text to a comment block using the given prefix (-- or #)."""
        lines = []
        for line in text.split("\n"):
            if line.strip():
                lines.append(line if line.startswith(prefix) else f"{prefix} {line}")
            else:
                lines.append(prefix)
        return "\n".join(lines)

    def _build_header(program_content: str, plan_idx, prefix: str) -> str:
        """Assemble column-description + input-dataframe comment header."""
        sections = []

        # 1. Column descriptions from database_description CSVs
        if db_path and db_id:
            try:
                tables = _extract_referenced_table_names(program_content)
                if tables:
                    # Load used columns from plans.json[plan_idx][1] so we only
                    # show descriptions for columns actually referenced by this plan.
                    used_columns = None
                    if plan_idx is not None:
                        plans_path = os.path.join(program_dir, "plans.json")
                        try:
                            with open(plans_path, "r", encoding="utf-8") as _pf:
                                _plans = json.load(_pf)
                            if plan_idx < len(_plans):
                                used_columns = _plans[plan_idx][1]  # {table: [col, ...]}
                        except Exception:
                            pass
                    col_desc = _format_column_descriptions(db_path, db_id, tables, used_columns)
                    if col_desc:
                        sections.append(col_desc)
            except Exception:
                pass

        # 2. Input dataframes from sub-queries
        if input_dataframes_formatter:
            try:
                df_text = input_dataframes_formatter(plan_idx)
                if df_text:
                    sections.append("Input DataFrames (from sub-queries):\n" + df_text)
            except Exception:
                pass

        if not sections:
            return ""

        sep = f"\n{prefix}\n"
        body = sep.join(_to_comments(s, prefix) for s in sections)
        return body + f"\n{prefix}\n"

    def read_program(program_id: str) -> str:
        """Read a program file (SQL or Python) by its group index."""
        import glob

        plan_idx = None
        if '_' in program_id:
            try:
                plan_idx = int(program_id.split('_')[0])
            except ValueError:
                pass

        for path, prefix in [
            (os.path.join(program_dir, f"program_{program_id}.sql"), "--"),
            (os.path.join(program_dir, f"program_{program_id}.py"),  "#"),
        ]:
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                header = _build_header(content, plan_idx, prefix)
                return header + content if header else content
            except Exception as e:
                return f"Error reading program {program_id}: {e}"

        available = sorted([
            os.path.basename(f).replace('program_', '').replace('.sql', '').replace('.py', '')
            for f in glob.glob(os.path.join(program_dir, 'program_*.*'))
        ])
        return f"Program {program_id} not found. Available programs: {available}"

    return read_program


def get_tools_definition(tool_functions: Dict[str, Callable], db_type: Optional[str] = None) -> list:
    """Get the tools definition in OpenAI format.

    Args:
        tool_functions: Dictionary mapping tool names to their callable functions
        db_type: Database type ("snowflake" or "sqlite"), used to tailor parameter descriptions

    Returns:
        List of tool definitions in OpenAI format
    """
    # Table name description is tailored to the actual database being used
    if db_type == "snowflake":
        table_name_desc = "The table name in the format 'DATABASE.SCHEMA.TABLE' (e.g., 'MYDB.PUBLIC.USERS')"
    elif db_type == "sqlite":
        table_name_desc = "The table name (e.g., 'users')"
    else:
        table_name_desc = "The table name (e.g., 'DATABASE.SCHEMA.TABLE' for Snowflake or 'table' for SQLite)"

    tools = []

    if "query_database" in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute a SQL query on the database and return formatted results with 20 example rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute."
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    if "get_distinct_values" in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": "get_distinct_values",
                "description": "Get 100 distinct values from a specific column in a table. Use this tool to see what categorical values exist in a column. This is useful when the user mentions a category or type that you need to verify exists in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": table_name_desc
                        },
                        "column": {
                            "type": "string",
                            "description": "The column name to get distinct values from"
                        }
                    },
                    "required": ["table", "column"]
                }
            }
        })

    if "search_dimension_values" in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": "search_dimension_values",
                "description": "Searches a specific table for rows where a column contains the term(s) (case-insensitive partial match). Returns up to 5 matching rows. By default only the searched column is returned, which is useful for checking whether a value exists. If you need other columns from the matching rows (e.g. to look up a key or retrieve related fields), specify them in `additional_columns`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "terms": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "A list of search terms to look for (case-insensitive partial match). A single term is a single-element array like [\"term\"]."
                        },
                        "table": {
                            "type": "string",
                            "description": table_name_desc
                        },
                        "column_name": {
                            "type": "string",
                            "description": "The column name to search in",
                            "default": "description"
                        },
                        "additional_columns": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Optional list of extra column names to include in the result alongside the search column."
                        }
                    },
                    "required": ["terms", "table", "column_name"]
                }
            }
        })
    
    # Add hierarchical schema linking tools if they're registered
    if "list_tables" in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": "list_tables",
                "description": "List all tables in a specific schema. Use this to explore what tables exist in a schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schema_name": {
                            "type": "string",
                            "description": "The schema name to list tables from"
                        }
                    },
                    "required": ["schema_name"]
                }
            }
        })
    
    if "list_columns" in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": "list_columns",
                "description": "List all columns in a specific table along with their data types and example values (up to 1 example value). Use this to explore the structure of a table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": table_name_desc
                        }
                    },
                    "required": ["table_name"]
                }
            }
        })

    if "python_interpreter" in tool_functions:
        if db_type == "sqlite":
            _db_access_note = (
                "IMPORTANT: A pre-connected sqlite3.Connection is injected as `conn` — do NOT call sqlite3.connect().\n"
                "- conn.execute(sql).fetchall()  — run a query directly\n"
                "- pd.read_sql_query(sql, conn)  — load results into a DataFrame\n"
                "- get_cursor()                  — returns a fresh cursor from the same connection"
            )
        elif db_type == "snowflake":
            _db_access_note = (
                "IMPORTANT: A pre-connected Snowflake cursor is injected as `cursor` — do NOT create new connections.\n"
                "- cursor.execute(sql); cursor.fetchall()  — run a query\n"
                "- get_cursor()                            — returns a fresh cursor from the same connection"
            )
        else:
            _db_access_note = "- get_cursor(): Returns a database cursor (if a DB connection was configured)."

        tools.append({
            "type": "function",
            "function": {
                "name": "python_interpreter",
                "description": f"""Execute Python code in a stateful interpreter.

The interpreter has access to:
- pd, np, nx, gpd: Common data science libraries
- DataFrames: List of pandas DataFrames (the variable name depends on context - check the prompt for the specific variable name to use, e.g., 'dfs' or 'program_output')
- All standard Python builtins
- __name__ is set to '__main__' so if __name__ == '__main__': blocks will execute
- All imports and variables persist across multiple executions (stateful)
- Database access:
{_db_access_note}

You can test your processing function by:
1. Defining the processing function
2. Calling it with the DataFrame list
3. Checking the result: print(result.head()) or print(len(result))""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute."
                        }
                    },
                    "required": ["code"]
                }
            }
        })
    
    if "read_program" in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": "read_program",
                "description": "Read a program file (SQL or Python) by its group index. Pass the group index as a string: '0' for the first group, '1' for the second, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "program_id": {
                            "type": "string",
                            "description": "The group index as a string, e.g. '0', '1', '2'. Do NOT pass 'plan_idx_program_idx' format like '0_0' or '1_0'."
                        }
                    },
                    "required": ["program_id"]
                }
            }
        })
        
    return tools
