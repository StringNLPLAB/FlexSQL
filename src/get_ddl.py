import sqlite3
import re
import os
import pandas as pd
import chardet
import json
from sqlglot import parse_one, exp
import sys

# Ensure we can import from utils if running from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def calculate_similarity(s1: str, s2: str) -> dict:
    """
    Calculates the Levenshtein distance and similarity percentage between two strings.

    The calculation is case-insensitive.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        A dictionary containing the edit 'distance' and 'similarity' percentage.
    """
    # Normalize strings to lower case for case-insensitive comparison
    s1_lower = s1.lower()
    s2_lower = s2.lower()

    len1, len2 = len(s1_lower), len(s2_lower)

    # Handle the trivial case of empty strings
    max_len = max(len1, len2)
    if max_len == 0:
        return {'distance': 0, 'similarity': 100.0}

    # Initialize the distance matrix (len1+1 x len2+1)
    # The extra row and column are for comparisons with an empty string
    dist_matrix = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

    # Initialize the first row and column of the matrix
    # This represents the cost of converting an empty string to a prefix of the other string
    for i in range(len1 + 1):
        dist_matrix[i][0] = i
    for j in range(len2 + 1):
        dist_matrix[0][j] = j

    # Fill the rest of the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Cost is 0 if characters are the same, 1 if they're different
            cost = 0 if s1_lower[i-1] == s2_lower[j-1] else 1
            
            # The value of each cell is the minimum of three operations:
            # 1. Deletion (from s1)
            # 2. Insertion (into s1)
            # 3. Substitution
            dist_matrix[i][j] = min(
                dist_matrix[i-1][j] + 1,        # Deletion
                dist_matrix[i][j-1] + 1,        # Insertion
                dist_matrix[i-1][j-1] + cost    # Substitution
            )

    # The final edit distance is in the bottom-right cell
    distance = dist_matrix[len1][len2]

    # Calculate similarity as a percentage
    similarity = (1 - distance / max_len)

    return round(similarity, 2)

def quote_identifiers(sql_text: str, output_dialect: str = "snowflake") -> str:
    expression = parse_one(sql_text, read=output_dialect)

    # 1. SCANNER: Identify aliases specifically created by FLATTEN
    # We find all aliases (like "ep") that are attached to a LATERAL FLATTEN.
    flatten_aliases = set()
    for lateral in expression.find_all(exp.Lateral):
        if isinstance(lateral.this, exp.Explode):
            alias = lateral.args.get("alias")
            if alias:
                flatten_aliases.add(alias.this.this)

    FLATTEN_KEYWORDS = {"value", "key", "path", "index", "seq", "this"}
    never_quoted_node = {}
    # 2. WALKER: Mutate the tree in-place
    # We use walk() instead of transform() to avoid the "assert root" error.
    for node in expression.walk():
        
        # --- CASE A: Handle "Flatten" Columns (ep.value) ---
        if isinstance(node, exp.Column):
            table = node.table
            # Check if this column belongs to a known Flatten alias
            if table and table in flatten_aliases:
                # Check if the column name is one of the reserved keywords
                # Note: node.this is usually the Identifier object
                if isinstance(node.this, exp.Identifier):
                    col_name = node.this.this
                    if col_name.lower() in FLATTEN_KEYWORDS:
                        # FORCE UPPERCASE and UNQUOTED
                        node.this.set("this", col_name.upper())
                        node.this.set("quoted", False)
                        never_quoted_node[node] = True
                        
                        # We continue; the next check (Case B) will see 
                        # this is now Uppercase and skip quoting it.
        # --- CASE B: Handle All Identifiers (General Rule) ---
        if isinstance(node, exp.Identifier):
            # If explicitly marked as unquoted (from Case A), skip it
            if node in never_quoted_node:
                continue

            name = node.this
            # RULE: If it contains lowercase letters, quote it.
            if any(char.islower() for char in name):
                node.set("quoted", True)

        # --- CASE C: Handle Aliases (Clean up) ---
        if isinstance(node, (exp.Alias, exp.TableAlias)):
            # Remove verbose column definitions: (seq, key, path...)
            node.set("columns", None)

    # Generate SQL
    return expression.sql(dialect=output_dialect, identify=False, pretty=True)


def quote_identifiers_(sql_text: str, output_dialect: str) -> str:
    # Parse the query
    expression = parse_one(sql_text, read=output_dialect)

    def transform_node(node):
        # 1. Handle Identifiers (Columns/Tables)
        if isinstance(node, exp.Identifier):
            name = node.this
            # Only quote if there are lowercase letters
            node.set("quoted", any(char.islower() for char in name))
            return node

        # 2. THE KEY FIX: Prevent column list injection in aliases
        # This stops (SEQ, KEY, PATH, etc.) from being added if they weren't there.
        if isinstance(node, (exp.Alias, exp.TableAlias)):
            # Force columns to None so sqlglot doesn't render the default list
            node.set("columns", None)
            
            # Ensure the alias itself follows the casing rule
            if node.args.get("alias"):
                alias_name = node.args["alias"].this
                node.args["alias"].set("quoted", any(char.islower() for char in alias_name))
        
        return node

    # Apply transformations
    transformed_ast = expression.transform(transform_node)

    # Generate SQL. 
    # 'identify=False' ensures it doesn't try to force-quote everything.
    return transformed_ast.sql(dialect=output_dialect, identify=False, pretty=True)

def post_format_generated_query(query, db_path, db_type="snowflake", include_comment=False):
    """
        only use for subqueries composed from plans
    """
    
    if include_comment:
        parsed_ast = parse_one(query, read=f"{db_type}")

        table_names = {
            table.sql().split()[0] for table in parsed_ast.find_all(exp.Table)
        }

        for table in table_names:
            pattern = r"'|\"|`"
            table_name = re.sub(pattern, "", table)
            if db_type == "snowflake":
                schema = "/".join(table_name.replace(".", "/").split("/")[:-1])
                table_name = table_name.replace(".", "/").split("/")[-1]
                
                for file in os.listdir(os.path.join(db_path, "resource/databases_no_nulls_2", schema)):
                    if file.endswith(".json") and file.split(".")[0].lower() == table_name.lower():
                        meta_data_path = os.path.join(db_path, "resource/databases_no_nulls_2", schema, file)
                        break
                with open(meta_data_path, 'r') as f:
                    meta_data = json.load(f)
                for column_name, comment, dtype in zip(meta_data["column_names"], meta_data["description"], meta_data["column_types"]):
                    if comment:
                        query = add_sql_comment(query, column_name=column_name, comment=f"{dtype} " + comment)
                    else:
                        query = add_sql_comment(query, column_name=column_name, comment=f"{dtype}")
                    
    query = quote_identifiers(query, output_dialect=db_type)

    return query


def add_ddl_comment(ddl: str, column_name: str, comment: str, examples: list = None):
    """
    Adds a comment to a specific column definition in a SQL DDL statement.
    Only if that column's comment is less than 50% similar to this column name

    This version correctly handles column definitions that contain commas,
    such as NUMBER(38, 0).

    Args:
        ddl: The full SQL DDL string.
        column_name: The unquoted name of the column (e.g., "comment_count").
        comment: The comment string to add.
        examples: Optional list of example values.

    """
    # Clean unecessary numerical value for datatype definition.
    pattern = re.compile(
        r'\b(VARCHAR|CHAR|NVARCHAR|CHARACTER|DECIMAL|NUMERIC|NUMBER|FLOAT|REAL|BINARY|VARBINARY)\s*\(\s*\d+\s*(,\s*\d+\s*)?\)',
        re.IGNORECASE
    )

    # Replace the matched pattern (e.g., "VARCHAR(16777216)") with just the data type ("VARCHAR")
    # The r'\1' refers to the first captured group in the pattern, which is the data type name.
    ddl = pattern.sub(r'\1', ddl)

    # Escape the column name in case it contains special regex characters.
    safe_name = re.escape(column_name)

    # This pattern matches the column name, allowing it to be unquoted,
    # double-quoted (`"`), or backticked (`` ` ``). The `\b` ensures we match
    # the whole word for the unquoted case.
    column_identifier = rf'(?:"{safe_name}"|`{safe_name}`|\b{safe_name}\b)'

    # The corrected regex pattern.
    # Group 1: `(^\s*{column_identifier}\s+.*?)` captures the column name and full data type.
    #          The `.*?` is non-greedy and captures everything until the next part of the pattern.
    # Group 2: `(,?)` optionally captures the trailing comma.
    # Group 3: `(\s*)$` captures trailing whitespace at the end of the line.
    pattern = re.compile(
        rf"(^\s*{column_identifier}\s+.*?)(,?)(\s*)$",
        re.MULTILINE | re.IGNORECASE
    )

    # Define the replacement string.
    # Handles `None`, empty strings, and "nan".
    
    has_comment = comment and str(comment).strip().lower() != "nan"
    has_examples = examples and len(examples) > 0
    
    if has_comment or has_examples:
        # Reconstructs the line: \1 is the definition, \2 is the comma.
        # Then, it adds the comment.
        final_comment_parts = []
        if has_comment:
            name_comment_simlarity = calculate_similarity(column_name, comment)
            if name_comment_simlarity < 0.5:
                # Escape single quotes in the comment itself to prevent SQL injection/errors
                # safe_comment = str(comment).replace("'", "\\'")
                final_comment_parts.append(comment)
        
        if has_examples:
            # Format examples, escaping single quotes in values
            ex_str = ", ".join([f"'{str(truncate_nested_data(e))}'" for e in examples[:1]])
            final_comment_parts.append(f"example values: {ex_str}")
            
        if final_comment_parts:
            final_comment = "; ".join(final_comment_parts)
            # Escape single quotes for SQL: ' -> ''
            escaped_comment = final_comment.replace("'", "''")
            
            # Use a lambda function for replacement to avoid regex escape issues
            # This way we can safely insert the comment without worrying about
            # backslashes or other special characters being interpreted as regex escapes
            def replacement_func(match):
                col_def = match.group(1)  # Column definition
                comma = match.group(2)     # Optional comma
                return f"{col_def} COMMENT '{escaped_comment}'{comma}"
            
            # Perform the substitution and return the result.
            modified_ddl = pattern.sub(replacement_func, ddl)
            
            return modified_ddl
    return ddl

def add_sql_comment(
    sql_query: str,
    column_name: str,
    comment: str,
    dialect: str = "snowflake"
) -> str:
    """
    Adds a custom inline TRAILING comment to a specific column in a SQL query.

    Args:
        sql_query: The original SQL query string.
        column_name: The name (or alias) of the column to comment.
        comment: The text for the comment.
        dialect: The SQL dialect for parsing and generation.

    Returns:
        The new SQL query with the comment added.
    """
    
    try:
        # Parse the SQL into an AST
        ast = parse_one(sql_query, read=dialect)

        # Define a transformer function to find and modify the node
        def find_and_comment(node):
            # Check if the node is a column expression
            if isinstance(node, (exp.Column, exp.Alias)):
                
                # Check if the node's name or alias matches our target
                node_name = ""
                if isinstance(node, exp.Alias):
                    node_name = node.alias_or_name
                elif isinstance(node, exp.Column):
                    node_name = node.this.name

                if node_name == column_name:
                    # Initialize the comments list if it doesn't exist
                    if node.comments is None:
                        node.comments = []
                    
                    # Use append() for a trailing comment
                    node.comments.append(comment)
                    # --------------------------
            
            return node

        # Apply the transformation to the entire AST
        transformed_ast = ast.transform(find_and_comment)

        # Generate the new SQL string in the specified dialect
        return transformed_ast.sql(dialect=dialect, pretty=True)

    except Exception as e:
        print(f"An error occurred: {e}")
        return sql_query


def extract_ddl(db_folder="dev_20240627/dev_databases", db_type="sqlite", question_id="", database_id=None, use_gold_tables=True, include_example_values=True):
    """
    Connects to a SQLite database and extracts the DDL statements
    for all tables.

    Args:
        db_name: Database name
        db_folder: Path to database folder
        db_type: Type of database ("sqlite" or "snowflake")
        question_id: Question/instance ID for snowflake databases (used for gold_tables mode)
        database_id: Database ID for snowflake databases (used for non-gold_tables mode)
        use_gold_tables: If True, only process tables from gold_tables in JSONL file.
                        If False, process all tables found in ddl_folder.
        include_example_values: If True, include example values in the DDL comments.
    """
    table_ddl_statements = {}
    if db_type == "sqlite":
        _db_file = os.path.join(db_folder, database_id, f"{database_id}.sqlite")
        conn = sqlite3.connect(f"file:{_db_file}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Query the sqlite_master table for the DDL statements
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;")
        
        # print("Extracting DDL from the database...")
        for row in cursor.fetchall():
            table_name, table_ddl = row

            try:
                description_path = os.path.join(db_folder, database_id, f"{table_name}.csv")
                with open(description_path, 'rb') as f:
                    encoding = chardet.detect(f.read())

                table_meta_data = pd.read_csv(os.path.join(db_folder, database_id, f"{table_name}.csv"), encoding=encoding["encoding"])
                table_meta_data = table_meta_data.set_index("original_column_name").to_dict(orient="index")
            except FileNotFoundError as filenotfound:
                if table_name == "sqlite_sequence":
                    continue
                print(filenotfound)
                continue
            for original_column_name in table_meta_data.keys():
                if str(table_meta_data[original_column_name]["column_description"]).lower() == str(original_column_name).lower():
                    continue

                table_ddl = add_ddl_comment(table_ddl, column_name=original_column_name, comment=table_meta_data[original_column_name]["column_description"])

            table_ddl_statements[table_name] = table_ddl
    elif db_type == "snowflake":
        if use_gold_tables:
            # Original mode: use gold_tables from JSONL file
            with open(os.path.join(db_folder, "spider2-snow-gold-tables.jsonl"), "r") as f:
                for line in f:
                    tmp = json.loads(line)
                    key = tmp["instance_id"]
                    if key != question_id:
                        continue
                    gold_tables = tmp["gold_tables"]
                    for table in gold_tables:
                        schema = "/".join(table.replace(".", "/").split("/")[:-1])
                        table_name = table.split(".")[-1]
                        ddl_folder = f"{db_folder}/resource/databases_no_nulls_2/{schema}/"

                        for table_metadata_file in os.listdir(ddl_folder):
                            if table_metadata_file.endswith(".json") and not table_metadata_file.endswith("_M-Schema.json") and table_metadata_file.split(".")[0].lower() == table_name.lower():
                                
                                table_meta_data = json.load(open(os.path.join(ddl_folder, table_metadata_file)))
                                break
                        else:
                            raise FileNotFoundError(f"Table metadata file not found for table {table_name} in schema {schema}")

                        table_fullname = table_meta_data["table_fullname"]

                        all_table_ddls = pd.read_csv(os.path.join(ddl_folder, "DDL.csv"), usecols=[0, 2], index_col=0).squeeze("columns").to_dict()
                        
                        all_table_ddls = {key.split(".")[-1]: value for key, value in all_table_ddls.items()}

                        column_descriptions_mapping = {key: value for key, value in zip(table_meta_data["column_names"], table_meta_data["description"])}

                        # Replace table full name in ddl to make queries precise
                        table_ddl = replace_table_name(ddl_string=all_table_ddls[table_fullname.split(".")[-1]], new_table_name=table_fullname)

                        for orig_column_name in column_descriptions_mapping.keys():
                            col_examples = table_meta_data.get("column_examples", {}).get(orig_column_name, [])
                            if include_example_values:
                                table_ddl = add_ddl_comment(table_ddl, orig_column_name, comment=column_descriptions_mapping[orig_column_name], examples=col_examples)
                            else:
                                table_ddl = add_ddl_comment(table_ddl, orig_column_name, comment=column_descriptions_mapping[orig_column_name])

                        table_ddl_statements[table_fullname] = table_ddl
        else:
            # New mode: process all tables in ddl_folder
            if database_id is None:
                raise ValueError("database_id must be provided when use_gold_tables=False")
            
            base_ddl_folder = f"{db_folder}/resource/databases_no_nulls_2/{database_id}/"
            
            if not os.path.exists(base_ddl_folder):
                raise FileNotFoundError(f"DDL folder not found: {base_ddl_folder}")
            
            # Iterate through all schema directories
            for schema_dir in os.listdir(base_ddl_folder):
                ddl_folder = os.path.join(base_ddl_folder, schema_dir)
                if not os.path.isdir(ddl_folder):
                    continue

                # Check if DDL.csv exists in this schema folder
                ddl_csv_path = os.path.join(ddl_folder, "DDL.csv")

                if not os.path.exists(ddl_csv_path):
                    continue
                
                # Load all DDLs for this schema
                all_table_ddls = pd.read_csv(ddl_csv_path, usecols=[0, 2], index_col=0).squeeze("columns").to_dict()
                all_table_ddls = {key.split(".")[-1]: value for key, value in all_table_ddls.items()}
                
                # Process all JSON metadata files in this schema folder
                for table_metadata_file in os.listdir(ddl_folder):
                    if not table_metadata_file.endswith(".json") or table_metadata_file.endswith("_M-Schema.json"): # skipping table without *.JSON metadata file
                        continue
                    
                    table_meta_data_path = os.path.join(ddl_folder, table_metadata_file)
                    table_meta_data = json.load(open(table_meta_data_path))
                    
                    table_fullname = table_meta_data["table_fullname"]
                    table_name_short = table_fullname.split(".")[-1]
                    
                    # Skip if DDL not found for this table
                    if table_name_short not in all_table_ddls:
                        continue
                    
                    column_descriptions_mapping = {key: value for key, value in zip(table_meta_data["column_names"], table_meta_data["description"])}

                    # Replace table full name in ddl to make queries precise
                    table_ddl = replace_table_name(ddl_string=all_table_ddls[table_name_short], new_table_name=table_fullname)

                    for orig_column_name in column_descriptions_mapping.keys():
                        col_examples = table_meta_data.get("column_examples", {}).get(orig_column_name, [])
                        if include_example_values:
                            table_ddl = add_ddl_comment(table_ddl, orig_column_name, comment=column_descriptions_mapping[orig_column_name], examples=col_examples)
                        else:
                            table_ddl = add_ddl_comment(table_ddl, orig_column_name, comment=column_descriptions_mapping[orig_column_name])

                    table_ddl_statements[table_fullname] = table_ddl
    return table_ddl_statements


def list_tables(db_folder="dev_20240627/dev_databases", db_type="sqlite", question_id="", database_id=None, use_gold_tables=True):
    """
    Return a list of table names for the given database without building DDL strings.

    Mirrors the enumeration logic of extract_ddl but skips all per-table metadata
    reads and DDL formatting. Used by callers (e.g. planning) that only need the
    set of table identifiers.

    Args:
        db_folder: Path to database folder
        db_type: "sqlite" or "snowflake"
        question_id: Instance ID (used when use_gold_tables=True for snowflake)
        database_id: Database ID (required for both sqlite and non-gold snowflake)
        use_gold_tables: If True (snowflake only), only return tables from gold_tables JSONL

    Returns:
        List[str] of table names (short names for sqlite, fully-qualified for snowflake).
    """
    if db_type == "sqlite":
        _db_file = os.path.join(db_folder, database_id, f"{database_id}.sqlite")
        conn = sqlite3.connect(f"file:{_db_file}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        names = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
        conn.close()
        return names

    if db_type == "snowflake":
        table_names = []
        if use_gold_tables:
            with open(os.path.join(db_folder, "spider2-snow-gold-tables.jsonl"), "r") as f:
                for line in f:
                    tmp = json.loads(line)
                    if tmp["instance_id"] != question_id:
                        continue
                    for table in tmp["gold_tables"]:
                        schema = "/".join(table.replace(".", "/").split("/")[:-1])
                        table_name = table.split(".")[-1]
                        ddl_folder = f"{db_folder}/resource/databases_no_nulls_2/{schema}/"

                        for table_metadata_file in os.listdir(ddl_folder):
                            if (table_metadata_file.endswith(".json")
                                and not table_metadata_file.endswith("_M-Schema.json")
                                and table_metadata_file.split(".")[0].lower() == table_name.lower()):
                                table_meta_data = json.load(open(os.path.join(ddl_folder, table_metadata_file)))
                                table_names.append(table_meta_data["table_fullname"])
                                break
                        else:
                            raise FileNotFoundError(f"Table metadata file not found for table {table_name} in schema {schema}")
            return table_names

        if database_id is None:
            raise ValueError("database_id must be provided when use_gold_tables=False")
        base_ddl_folder = f"{db_folder}/resource/databases_no_nulls_2/{database_id}/"
        if not os.path.exists(base_ddl_folder):
            raise FileNotFoundError(f"DDL folder not found: {base_ddl_folder}")

        for schema_dir in os.listdir(base_ddl_folder):
            ddl_folder = os.path.join(base_ddl_folder, schema_dir)
            if not os.path.isdir(ddl_folder):
                continue
            for table_metadata_file in os.listdir(ddl_folder):
                if not table_metadata_file.endswith(".json") or table_metadata_file.endswith("_M-Schema.json"):
                    continue
                table_meta_data = json.load(open(os.path.join(ddl_folder, table_metadata_file)))
                table_names.append(table_meta_data["table_fullname"])
        return table_names

    raise ValueError(f"Unsupported db_type: {db_type}")

def load_table_similarities(similarities_path: str) -> dict:
    """
    Load table similarities report and build a mapping of similar tables.
    
    Args:
        similarities_path: Path to the table_similarities_report JSON file
        
    Returns:
        Dictionary mapping table_fullname -> list of similar table_fullnames
    """
    if not os.path.exists(similarities_path):
        return {}
    
    try:
        with open(similarities_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load table similarities report: {e}")
        return {}
    
    # Build mapping: table_name -> list of similar table names
    similar_tables_map = {}
    
    for db_name, db_data in data.items():
        if not isinstance(db_data, dict):
            continue
        
        for group_list in db_data.values():
            if not isinstance(group_list, list):
                continue
            
            for group in group_list:
                # Expected shape: [<prefix_or_label>, [ [table_name, [column_names, column_types]], ... ]]
                if not (isinstance(group, list) and len(group) == 2 and isinstance(group[1], list)):
                    continue
                
                table_entries = group[1]
                table_names = []
                
                # Extract all table names from this group
                for entry in table_entries:
                    if isinstance(entry, list) and entry and isinstance(entry[0], str):
                        table_names.append(entry[0])
                
                # If there are multiple tables in this group, they are similar
                if len(table_names) > 1:
                    # For each table, store the list of other similar tables
                    for table_name in table_names:
                        similar_tables_map[table_name] = [t for t in table_names if t != table_name]
    
    return similar_tables_map


def replace_table_name(ddl_string: str, new_table_name: str) -> str:
    """
    Replaces the table name in a CREATE TABLE DDL statement using regex.

    Args:
        ddl_string: The original DDL string.
        new_table_name: The new table name to substitute.

    Returns:
        The DDL string with the table name replaced.
    """
    # Regex to find and capture parts of the CREATE TABLE statement
    # Group 1: 'create or replace TABLE '
    # Group 2: The actual table name
    # Group 3: The opening parenthesis '(' with any preceding whitespace
    pattern = r'(?i)(create(?:\s+or\s+replace)?\s+TABLE\s+)([\w."`\']+)(\s*\()'
    
    # The replacement pattern uses backreferences to the captured groups.
    # \1 refers to the first group, and \3 refers to the third group.
    # The new table name is inserted between them.
    replacement = rf'\1{new_table_name}\3'
    
    # Perform the substitution
    new_ddl_string = re.sub(pattern, replacement, ddl_string, count=1)
    
    return new_ddl_string

def remove_byte_arrays(text: str) -> str:
    """
    Removes bytearray string representations from a given text using regex.
    """
    # This pattern specifically looks for "bytearray(b'...')"
    # The '.*?' part is a non-greedy match for any characters inside the single quotes.
    pattern = r"bytearray\(b'.*?'\)"
    
    # Replace any found patterns with an empty string
    return re.sub(pattern, "bytearray(b'...')", text)

def truncate_nested_data(data, max_str_len=20):
    """
    Recursively traverses a nested data structure (dicts, lists) and 
    truncates all string values to max_str_len.
    Keeps numbers, booleans, and None types as-is.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
    except Exception:
        pass
    
    if "bytearray" in str(data).lower():
        return "bytearray(b'...')"

    if isinstance(data, dict):
        # Recursively process dictionary values
        return {key: truncate_nested_data(value, max_str_len) 
                for key, value in data.items()}
    
    elif isinstance(data, list):
        # Recursively process list items
        # If list has more than 3 elements, show [element1, element2, ... last element]
        if len(data) > 3:
            result = [
                truncate_nested_data(data[0], max_str_len),
                truncate_nested_data(data[1], max_str_len),
                '...',
                truncate_nested_data(data[-1], max_str_len)
            ]
            return result
        else:
            # Process all elements for lists with 5 or fewer elements
            result = []
            for item in data:
                result.append(truncate_nested_data(item, max_str_len))
            return result
    elif isinstance(data, str):
        # Truncate the string if it's longer than max_str_len
        if len(data) > max_str_len:
            return data[:max_str_len//2] + '...' + data[-max_str_len//2:]
        return data
        
    else:
        # Keep numbers (int, float), bools, None, etc. as-is
        return data


if __name__ == "__main__":
    ddls = extract_ddl(database_id="codebase_community")
    print(ddls.keys())
