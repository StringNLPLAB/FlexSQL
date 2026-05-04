import os
import sys
import json
import re
import random
import sqlite3
import logging
import math
import traceback
import snowflake.connector
from typing import Any, Dict, Tuple, List, Callable, Set
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger import initialize_logger
from utils import get_db_id
from difflib import SequenceMatcher


import sqlglot
from sqlglot import exp
from chat import Chat

from get_ddl import (
    post_format_generated_query,
    truncate_nested_data,
    load_table_similarities
)
from prompt_templates import (
    HIERARCHICAL_SCHEMA_LINKING_PROMPT_TEMPLATE,
)

import time


def execute_query(cursor, query: str, n_rows: int) -> Tuple[List[str], List[Tuple]]:
    """
    Executes a query and fetches a limited number of rows.
    Returns headers and data rows separately.
    Appends LIMIT clause to the query if not already present.
    """
    # Check if query already has a LIMIT clause (case-insensitive)
    # Look for LIMIT followed by a number, possibly with OFFSET
    limit_pattern = r'\bLIMIT\s+\d+'
    
    if not re.search(limit_pattern, query, re.IGNORECASE):
        # Remove trailing semicolon and whitespace if present
        query = query.rstrip().rstrip(';')
        # Append LIMIT clause
        query = f"{query} LIMIT {n_rows}"
    
    cursor.execute(query)
    headers = [description[0] for description in cursor.description]
    # Use fetchall() since we're limiting at SQL level
    data_rows = cursor.fetchall()
    return headers, data_rows


def format_results_to_markdown(headers: List[str], data_rows: List[Tuple], truncate_data: bool =True, max_truncate_len:int =20) -> str:
    """Formats query results into a markdown table."""
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in data_rows:
        processed_cells = []
        for value in row:
            if value is None:
                str_value = "NULL"
            # Check if the value is a float NaN (Not a Number)
            elif isinstance(value, float) and math.isnan(value):
                str_value = "NaN"
            elif "bytearray" in str(value).lower():
                str_value = "bytearray(b'...')"
            else:
                # Convert to string, replace newlines and pipes to avoid breaking the table
                if isinstance(value, str):
                    try:
                        if value.strip().startswith("[") or value.strip().startswith("{"):
                            value = json.loads(value)
                        if truncate_data:
                            value = truncate_nested_data(value, max_str_len=max_truncate_len)
                    except json.decoder.JSONDecodeError:
                        # print("Normal string value:", value, type(value))
                        value = value[:max_truncate_len] + "..." if len(value) > max_truncate_len and truncate_data else value
                        # exit()
                
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
    truncate_data=True,
    max_truncate_len=20,
    include_comment=False,
    include_query=True
):
    """
    Execute a query, validate the target table, and return formatted results.
    """
    query = post_format_generated_query(query, db_path=db_path, db_type=db_type, include_comment=False)

    headers, data_rows = execute_query(cursor, query, n_rows=n_example_rows)

    if not data_rows:
        raise ValueError("Query returned no results.")
    
    if include_comment:
        query = post_format_generated_query(query, db_path=db_path, db_type=db_type, include_comment=True)
    
    result_markdown = format_results_to_markdown(headers, data_rows, truncate_data=truncate_data, max_truncate_len=max_truncate_len)
    
    if include_query:
        formatted_string = (
            f"```sql\n{query}\n```\n\n"
            f"{n_example_rows} example rows:\n{result_markdown}"
        )
    else:
        formatted_string = f"{n_example_rows} example rows:\n{result_markdown}"

    return formatted_string

def match_table_name_flexible(pattern: str, available_tables: Set[str], logger: logging.Logger = None) -> List[str]:
    """
    Flexibly match a table name pattern against available tables.
    
    Handles:
    - Wildcard patterns (e.g., "GA_SESSIONS_*" matches "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170401")
    - Case-insensitive matching
    - Partial matches (e.g., "GA_SESSIONS" matches tables ending with "GA_SESSIONS_*")
    - Fuzzy matching for spelling mistakes (using sequence similarity)
    
    Args:
        pattern: Table name pattern from model (may contain wildcards or spelling mistakes)
        available_tables: Set of available fully-qualified table names (e.g., "DATABASE.SCHEMA.TABLE")
        logger: Optional logger for debugging
    
    Returns:
        List of matched table names
    """
    if not pattern or not available_tables:
        return []
    
    pattern = pattern.strip()
    matches = []
    
    # Normalize pattern: handle case-insensitive matching
    pattern_lower = pattern.lower()
    
    # Strategy 1: Exact match (case-insensitive)
    for table in available_tables:
        if table.lower() == pattern_lower:
            matches.append(table)
            if logger:
                logger.debug(f"Exact match: '{pattern}' -> '{table}'")
    
    if matches:
        return matches
    
    # Strategy 2: Wildcard pattern matching
    # Convert wildcard pattern to regex (escape special chars except *)
    pattern_escaped = re.escape(pattern)
    pattern_escaped = pattern_escaped.replace(r'\*', '.*')  # Convert \* to .*
    pattern_regex = re.compile(f'^{pattern_escaped}$', re.IGNORECASE)
    
    for table in available_tables:
        if pattern_regex.match(table):
            matches.append(table)
            # if logger:
            #     logger.debug(f"Wildcard match: '{pattern}' -> '{table}'")
    
    if matches:
        return matches
    
    # Strategy 3: Partial match - check if pattern matches the table name part (last component)
    # e.g., "GA_SESSIONS" should match "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170401"
    # e.g., "GA_SESSIONS_*" should match "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170401"
    pattern_table_part = pattern.split('.')[-1].lower()  # Get last component
    
    for table in available_tables:
        table_parts = table.split('.')
        table_name_part = table_parts[-1].lower() if table_parts else ""
        
        # Check if pattern matches the table name part (with wildcard support)
        # Remove trailing wildcard for prefix matching
        pattern_for_match = pattern_table_part.rstrip('*')
        table_pattern_escaped = re.escape(pattern_for_match)
        table_pattern_regex = re.compile(f'^{table_pattern_escaped}', re.IGNORECASE)
        
        if table_pattern_regex.match(table_name_part):
            matches.append(table)
            if logger:
                logger.debug(f"Partial match: '{pattern}' (table part: '{pattern_table_part}') -> '{table}'")
    
    if matches:
        return matches
    
    # Strategy 4: Fuzzy matching for spelling mistakes (only if no matches found)
    # Use sequence similarity to find close matches
    best_matches = []
    best_ratio = 0.7  # Minimum similarity threshold
    
    for table in available_tables:
        table_lower = table.lower()
        table_name_part = table.split('.')[-1].lower() if '.' in table else table_lower
        
        # Compare with full table name
        ratio_full = SequenceMatcher(None, pattern_lower, table_lower).ratio()
        # Compare with table name part only
        ratio_part = SequenceMatcher(None, pattern_table_part, table_name_part).ratio()
        
        ratio = max(ratio_full, ratio_part)
        
        if ratio >= best_ratio:
            if ratio > best_ratio:
                best_matches = [table]
                best_ratio = ratio
            elif ratio == best_ratio:
                best_matches.append(table)
    
    if best_matches:
        if logger:
            logger.debug(f"Fuzzy match (ratio={best_ratio:.2f}): '{pattern}' -> {best_matches}")
        return best_matches
    
    # No matches found
    if logger:
        logger.warning(f"No match found for table pattern: '{pattern}'")
    return []


def _normalize_column_list(column_names: Any) -> List[str]:
    if column_names is None:
        return []
    if isinstance(column_names, str):
        if "," in column_names:
            items = [item.strip() for item in column_names.split(",")]
        else:
            items = [column_names.strip()]
        # Remove all backslashes from column names
        return [item.replace("\\", "") for item in items if item]
    if isinstance(column_names, list):
        normalized = []
        for item in column_names:
            if item is None:
                continue
            if isinstance(item, str):
                normalized_item = item.strip()
                # Remove all backslashes from column names
                normalized_item = normalized_item.replace("\\", "")
                if normalized_item:
                    normalized.append(normalized_item)
        return normalized
    return []


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            cleaned = code_blocks[0].strip()
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    start_idx = cleaned.find("{")
    if start_idx == -1:
        return ""
    depth = 0
    for idx in range(start_idx, len(cleaned)):
        if cleaned[idx] == "{":
            depth += 1
        elif cleaned[idx] == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start_idx:idx + 1]
    return ""


def fix_and_escape_long_chains(raw_input):
    # Regex Explanation:
    # "[^"]+"          -> Matches the first quoted string (e.g., "Part1")
    # (?: ... )+       -> A non-capturing group that repeats 1 or more times
    # \s*:\s*"[^"]+"   -> Matches the colon and the next string (e.g., :"Part2")
    #
    # This grabs the entire chain like "A":"B":"C" as a single match.
    raw_input = raw_input.replace('\""', '"')
    
    pattern = r'"[^"]+"(?:\s*:\s*"[^"]+")+'
    
    def replace_match(match):
        # 1. Get the full matched text (e.g., "Version":"Info":"Release")
        full_chain = match.group(0)
        
        # 2. Escape existing quotes inside the text ( " -> \" )
        escaped_content = full_chain.replace('"', '\\"')
        
        # 3. Wrap the whole thing in outer quotes to make it a JSON string
        return f'"{escaped_content}"'

    # 1. Apply the regex fix
    fixed_json = re.sub(pattern, replace_match, raw_input)
    
    # 2. Parse to verify it is valid JSON
    try:
        parsed_obj = json.loads(fixed_json)
    except json.JSONDecodeError as e:
        return f"Error: Could not fix JSON. {e}"

    # 3. Dump to string (this adds the final layer of escaping for your output)
    return json.dumps(parsed_obj)

def parse_columns_response(
    model_response_txt: str,
    expected_tables: List[str],
    logger: logging.Logger,
    allow_wildcards: bool = True
) -> Dict[str, List[str]]:
    try:
        parsed = json.loads(fix_and_escape_long_chains(model_response_txt))
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse JSON column selection: {exc}")
        return {}
    if isinstance(parsed, list):
        if len(expected_tables) == 1:
            return {expected_tables[0]: _normalize_column_list(parsed)}
        logger.warning("JSON list returned for multiple tables; skipping.")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Parsed JSON is not an object; skipping.")
        return {}
    table_key_map = {table.lower(): table for table in expected_tables}
    mapped = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        normalized_key = key.strip()
        table_name = table_key_map.get(normalized_key.lower())
        if table_name:
            mapped[table_name] = _normalize_column_list(value)
            continue
        if allow_wildcards and ("*" in normalized_key or "?" in normalized_key):
            matched_tables = match_table_name_flexible(normalized_key, set(expected_tables), logger)
            if not matched_tables:
                logger.warning(f"No tables matched wildcard key '{normalized_key}'.")
                continue
            columns = _normalize_column_list(value)
            for matched_table in matched_tables:
                mapped[matched_table] = columns
    if len(mapped) == 0 and len(expected_tables) == 1:
        for alt_key in ("columns", "column_names"):
            if alt_key in parsed:
                return {expected_tables[0]: _normalize_column_list(parsed.get(alt_key))}
    return mapped


def compose_select_query(table_name: str, column_names: List[str]) -> str:
    if not column_names:
        return ""
    column_names_ = column_names.copy()
    for i, column_name in enumerate(column_names):
        
        if "\"" in column_name:
            column_name = column_name.replace("\"", "")
        if "'" in column_name:
            column_name = column_name.replace("'", "")

        if ":" in column_name:
            parts = column_name.split(":")
            column_name_quoted = []
            for part in parts:
                if "[" in part:
                    # Split at the first '[' to separate the part before from the bracket part
                    bracket_idx = part.index("[")
                    part_before = part[:bracket_idx]
                    bracket_part = part[bracket_idx:]
                    column_name_quoted.append(f"\"{part_before}\"{bracket_part}")
                else:
                    column_name_quoted.append(f"\"{part}\"")
            
            alias = f"{parts[0].upper().replace('[', '_').replace(']', '')}"
            for idx, part in enumerate(parts):
                if idx > 0:
                    alias += f"_{part.upper().replace('[', '_').replace(']', '')}"
            column_names_[i] = ":".join(column_name_quoted) + " AS " + alias

        else:
            column_names_[i] = f"\"{column_name}\""
            
    column_list = ", ".join(column_names_)
    table_name = table_name.replace("\"", "")
    table_name_parts = table_name.split(".")
    for i, part in enumerate(table_name_parts):
        table_name_parts[i] = f"\"{part}\""
    table_name = ".".join(table_name_parts)

    return f"SELECT {column_list} FROM {table_name}"    

def table_has_variant_columns(table_name: str, db_path: str, db_type: str, database_id: str) -> bool:
    """
    Check if a table has VARIANT columns by loading its JSON metadata file.
    
    Args:
        table_name: Fully qualified table name (e.g., "DATABASE.SCHEMA.TABLE")
        db_path: Path to the database folder (e.g., "datasets/Spider2/spider2-snow")
        db_type: Database type ("snowflake" or "sqlite")
        database_id: Database ID/name

    Returns:
        True if the table has at least one VARIANT column, False otherwise
    """
    if db_type != "snowflake" or db_path != "datasets/Spider2/spider2-snow":
        return False
    
    try:
        # Parse table name: DATABASE.SCHEMA.TABLE -> [DATABASE, SCHEMA, TABLE]
        table_parts = table_name.split(".")
        if len(table_parts) < 2:
            return False
        
        # For spider2-snow, the structure is: {db_path}/resource/databases_no_nulls_2/{database_id}/{schema}/{table}.json
        # The table_name might be fully qualified, so we need to extract schema and table name
        schema_name = table_parts[-2] if len(table_parts) >= 2 else table_parts[0]
        table_short_name = table_parts[-1]
        
        # Try to find the JSON file
        base_folder = os.path.join(db_path, "resource", "databases_no_nulls_2", database_id)
        schema_path = os.path.join(base_folder, schema_name)
        
        if not os.path.exists(schema_path):
            return False
        
        # Look for the JSON file matching the table name
        json_file = None
        for file in os.listdir(schema_path):
            if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                # Check if the file name matches (case-insensitive)
                file_base = file.replace(".json", "")
                if file_base.lower() == table_short_name.lower():
                    json_file = os.path.join(schema_path, file)
                    break
        
        if not json_file or not os.path.exists(json_file):
            return False
        
        # Load JSON metadata and check column types
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        column_types = metadata.get("column_types", [])
        # Check if any column type contains "VARIANT" (case-insensitive)
        return any("VARIANT" in str(col_type).upper() for col_type in column_types)
        
    except Exception as e:
        logging.warning(f"Error checking VARIANT columns for table {table_name}: {e}")
        return False


def group_has_variant_columns(table_group: List[str], db_path: str, db_type: str, database_id: str) -> bool:
    """
    Check if any table in a group has VARIANT columns.
    
    Args:
        table_group: List of table names in the group
        db_path: Path to the database folder
        db_type: Database type
        database_id: Database ID/name
        
    Returns:
        True if any table in the group has VARIANT columns, False otherwise
    """
    return any(table_has_variant_columns(table, db_path, db_type, database_id) for table in table_group)


def build_similarity_groups(
    table_names: List[str],
    similarities_path: str,
    logger: logging.Logger
) -> List[List[str]]:
    if not similarities_path:
        return [[table_name] for table_name in table_names]
    similar_tables_map = load_table_similarities(similarities_path)
    if not similar_tables_map:
        return [[table_name] for table_name in table_names]
    table_set = set(table_names)
    adjacency = {table_name: set() for table_name in table_set}
    for table_name, similar_tables in similar_tables_map.items():
        if table_name not in table_set:
            continue
        for similar_table in similar_tables:
            if similar_table in table_set:
                adjacency[table_name].add(similar_table)
                adjacency[similar_table].add(table_name)
    visited = set()
    groups = []
    for table_name in table_names:
        if table_name in visited:
            continue
        queue = [table_name]
        group = []
        visited.add(table_name)
        while queue:
            current = queue.pop(0)
            group.append(current)
            for neighbor in sorted(adjacency.get(current, [])):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        groups.append(group)
    if any(len(group) > 1 for group in groups):
        logger.info(f"Grouped tables by structure: {groups}")
    return groups


def get_schema_list(database_name: str, db_type: str, cursor_getter: Callable, db_path: str) -> List[str]:
    """
    Get list of schemas in the database by reading folder structure.
    
    Args:
        database_name: Name of the database
        db_type: Database type ("snowflake" or "sqlite")
        cursor_getter: Callable that returns a database cursor (not used, kept for compatibility)
        db_path: Path to the database folder
        
    Returns:
        List of schema names. For SQLite, returns ["main"] as a placeholder.
    """
    import os
    try:
        if db_type == "snowflake":
            # Schema names are subdirectories in: {db_path}/resource/databases_no_nulls_2/{database_name}/
            base_folder = os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
            
            if not os.path.exists(base_folder):
                logging.warning(f"Database folder not found: {base_folder}")
                return []
            
            # List all subdirectories (these are schema names)
            schemas = []
            for item in os.listdir(base_folder):
                item_path = os.path.join(base_folder, item)
                if os.path.isdir(item_path):
                    schemas.append(item)
            
            return sorted(schemas) if schemas else []
        else:
            # SQLite doesn't have schemas, return a placeholder
            return ["main"]
    except Exception as e:
        logging.warning(f"Error getting schema list: {str(e)}")
        return []


def get_schema_statistics(database_name: str, schema_list: List[str], db_type: str, db_path: str) -> List[Dict[str, Any]]:
    """
    Get statistics for each schema including number of tables and total columns.
    Only tables with JSON metadata files are counted.
    
    Args:
        database_name: Name of the database
        schema_list: List of schema names
        db_type: Database type ("snowflake" or "sqlite")
        db_path: Path to the database folder
        
    Returns:
        List of dictionaries with schema statistics
    """
    import os
    import json
    
    schema_stats = []
    
    try:
        if db_type == "snowflake":
            base_folder = os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
            
            if not os.path.exists(base_folder):
                return schema_stats
            
            for schema_name in schema_list:
                schema_path = os.path.join(base_folder, schema_name)
                if not os.path.isdir(schema_path):
                    continue
                
                # Count JSON files (tables) and total columns
                table_count = 0
                total_columns = 0

                for file in os.listdir(schema_path):
                    if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                        table_count += 1
                        try:
                            json_path = os.path.join(schema_path, file)
                            with open(json_path, 'r') as f:
                                metadata = json.load(f)
                            column_count = len(metadata.get("column_names", []))
                            total_columns += column_count
                        except Exception:
                            continue
                
                if table_count > 0:
                    schema_stats.append({
                        "schema": schema_name,
                        "table_count": table_count,
                        "total_columns": total_columns
                    })
        else:
            # SQLite: check for JSON files in the database folder
            base_folder = os.path.join(db_path, database_name)
            if not os.path.exists(base_folder):
                return schema_stats
            
            table_count = 0
            total_columns = 0
            
            for file in os.listdir(base_folder):
                if file.endswith(".json") and not file.endswith("_M-Schema.json"):
                    table_count += 1
                    try:
                        json_path = os.path.join(base_folder, file)
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                        column_count = len(metadata.get("column_names", []))
                        total_columns += column_count
                    except Exception:
                        continue
            
            if table_count > 0:
                schema_stats.append({
                    "schema": "main",
                    "table_count": table_count,
                    "total_columns": total_columns
                })
    except Exception as e:
        logging.warning(f"Error computing schema statistics: {str(e)}")
    
    return schema_stats


def hierarchical_schema_linking(
    agent: 'Chat',
    question: Dict,
    database_name: str,
    schema_list: List[str],
    db_type: str,
    cursor_getter: Callable,
    db_path: str,
    logger: logging.Logger,
) -> List[str]:
    """
    Perform hierarchical schema linking by allowing the model to explore schemas,
    tables, and columns using tools, then return relevant tables.
    
    Args:
        agent: Chat agent instance
        question: Question dictionary
        database_name: Name of the database
        schema_list: List of schema names available in the database
        db_type: Database type ("snowflake" or "sqlite")
        cursor_getter: Callable that returns a database cursor
        db_path: Path to the database folder
        logger: Logger instance

    Returns:
        List of fully qualified table names (DATABASE.SCHEMA.TABLE format) that are relevant
    """
    from utils import get_question_str
    agent.enable_tools(
        ["list_tables", "list_columns"],
        db_path=db_path,
        db_type=db_type,
        cursor_getter=cursor_getter,
        database_name=database_name,
    )
    logger.info(f"Tool calling enabled for hierarchical schema linking: {agent.tool_calling_enabled}, tool functions: {agent.tool_functions}")
    
    logger.info(f"Found {len(schema_list)} schemas: {schema_list}")

    
    # Get question string
    question_str = get_question_str(question, db_type)
    
    # Get schema statistics
    schema_stats = get_schema_statistics(database_name, schema_list, db_type, db_path)
    
    # Format schema statistics
    random.shuffle(schema_stats)
    schema_stats_str = ""
    for stat in schema_stats:
        schema_stats_str += f"  - {stat['schema']}: {stat['table_count']} tables, {stat['total_columns']} columns\n"
    
    # Create prompt
    prompt = HIERARCHICAL_SCHEMA_LINKING_PROMPT_TEMPLATE.format(
        database_name=database_name,
        schema_statistics=schema_stats_str.rstrip(),
        question=question_str
    )
    
    logger.info(f"<hierarchical-schema-linking>\nPrompt:\n{prompt}\n</hierarchical-schema-linking>")
    
    # Get model response with tool calling
    try:
        response = agent.get_code_blocks(
            prompt,
            code_format="json",
            logger=logger,
            example_json_structure=["string"]  # Expecting array of strings
        )
        
        # Extract JSON from response
        json_str = response["code_blocks"][0]
        relevant_tables = json.loads(json_str)
        
        if not isinstance(relevant_tables, list):
            logger.warning(f"Expected list of tables, got {type(relevant_tables)}. Converting to list.")
            relevant_tables = [relevant_tables] if relevant_tables else []
        
        logger.info(f"<hierarchical-schema-linking-result>Thoughts: {response['thoughts']}\nRelevant schemas: {relevant_tables}\n</hierarchical-schema-linking-result>")
        
        return relevant_tables
        
    except Exception as e:
        logger.error(f"Error in hierarchical schema linking: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return empty list on error - will fall back to using all tables
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/Spider2/spider2-snow/sampled_data_50.jsonl")
    parser.add_argument("--db_path", type=str, default="datasets/Spider2/spider2-snow")
    parser.add_argument("--db_type", type=str, default="snowflake")
    parser.add_argument("--ip", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-oss-120b")

    args = parser.parse_args()
    
    questions = [json.loads(line) for line in open(args.dataset)]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    experiment_name = f"schema_linking_{args.model}_{args.db_type}_{time_str}"
    
    os.makedirs(os.path.join("inference_res", experiment_name), exist_ok=True)
    
    