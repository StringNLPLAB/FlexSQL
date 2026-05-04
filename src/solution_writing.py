import json
import os
import sqlite3
import subprocess
import sys
import traceback
from typing import Dict, List, Tuple

import pandas as pd
import snowflake.connector

from chat import Chat, extract_all_blocks
from get_ddl import load_table_similarities, post_format_generated_query
from prompt_templates import (
    HYBRID_STITCHING_PROMPT_TEMPLATE,
    HYBRID_STITCHING_PLANNING_PROMPT_TEMPLATE,
    SQL_ONLY_STITCHING_PROMPT_TEMPLATE,
    SQL_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE,
    PYTHON_ONLY_STITCHING_PROMPT_TEMPLATE,
    PYTHON_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE,
    SNOWFLAKE_VARIANT_ACCESS_RULES,
)
from schema_linking import execute_and_format_query_result, group_has_variant_columns
from utils import get_db_id, get_question_str

MAX_RETRIES = 3


def _trim_to_processing_function(code: str) -> str:
    """Remove executable statements that appear after the `processing` function.

    Blank lines and comment lines outside the function are preserved.
    Falls back to the original code on SyntaxError.
    """
    import ast
    try:
        tree = ast.parse(code)
        func_end_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "processing":
                func_end_line = node.end_lineno  # 1-indexed
                break
        if func_end_line is None:
            return code
        lines = code.splitlines()
        result = lines[:func_end_line]
        for line in lines[func_end_line:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                result.append(line)
        return "\n".join(result)
    except SyntaxError:
        pass
    return code


def _group_sub_sqls_by_similarity(
    sub_sqls: Dict[str, str],
    similarities_path: str,
    logger
) -> List[Tuple[str, List[str]]]:
    table_names = list(sub_sqls.keys())
    if not similarities_path or not table_names:
        return [(table_name, []) for table_name in table_names]
    similar_tables_map = load_table_similarities(similarities_path)
    if not similar_tables_map:
        return [(table_name, []) for table_name in table_names]
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
        ordered_group = [name for name in table_names if name in group]
        if not ordered_group:
            continue
        representative = ordered_group[0]
        similar_tables = ordered_group[1:]
        groups.append((representative, similar_tables))
    if any(similar for _, similar in groups):
        logger.info(f"Grouped sub-SQLs by structure: {groups}")
    return groups


def _format_sub_sqls_for_prompt(
    sub_sqls: Dict[str, str],
    similarities_path: str,
    db_type: str,
    db_path: str,
    question: dict,
    logger,
    include_df_indices: bool = True,
) -> str:
    grouped = _group_sub_sqls_by_similarity(sub_sqls, similarities_path, logger)
    conn = None
    cursor = None
    if db_path:
        if db_type == "sqlite":
            db_name = get_db_id(question)
            db_file_path = os.path.join(db_path, db_name, f"{db_name}.sqlite")
            conn = sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
            cursor = conn.cursor()
        elif db_type == "snowflake":
            snowflake_credential = json.load(open(os.path.join(db_path, "snowflake_credential.json")))
            conn = snowflake.connector.connect(**snowflake_credential)
            cursor = conn.cursor()
    table_names = list(sub_sqls.keys())
    blocks = []
    for representative, similar_tables in grouped:
        query = sub_sqls.get(representative, "")
        if not query:
            continue
        rep_idx = table_names.index(representative)
        header = f"SQL Query {rep_idx + 1} — dfs[{rep_idx}] (Table: {representative})" if include_df_indices else f"SQL Query {rep_idx + 1} (Table: {representative})"
        if cursor:
            try:
                formatted_string = execute_and_format_query_result(
                    cursor=cursor,
                    query=query,
                    db_path=db_path,
                    db_type=db_type,
                    n_example_rows=1,
                    truncate_data=True,
                )
                block = f"{header}\n{formatted_string}"
            except Exception as exc:
                logger.warning(f"Failed to fetch example rows for {representative}: {exc}")
                block = f"{header}\n```sql\n{query}\n```"
        else:
            block = f"{header}\n```sql\n{query}\n```"
        if similar_tables:
            sim_info = ", ".join(f"{t} → dfs[{table_names.index(t)}]" for t in similar_tables)
            block += f"\nSimilar structure tables (same schema, also available in dfs): {sim_info}"
        blocks.append(block)
    if conn:
        cursor.close()
        conn.close()
    return "\n\n---\n\n".join(blocks)


def _load_data_frames_for_interpreter(
    db_type: str, db_path: str, question: dict, sub_sqls: Dict[str, str],
    sub_queries_save_path: str, logger
) -> Tuple[List[pd.DataFrame], bool]:
    """Load data frames by spawning a child subprocess (utils/load_dataframes_worker.py).

    Running the load in a child process means that if the OS OOM-killer sends
    SIGKILL because a table is too large to fit in RAM, only the child dies.
    The parent detects returncode == -9 and falls back to SQL-only mode without
    crashing the overall question-processing worker.

    Returns:
        (data_frames, oom_killed): list of DataFrames (empty list when oom_killed
        is True).
    """
    import pickle

    # Build the args dict for the worker
    if db_type == "sqlite":
        db_name = get_db_id(question)
        db_file_path = os.path.join(db_path, db_name, f"{db_name}.sqlite")
        worker_args = {
            "db_type": "sqlite",
            "db_file_path": db_file_path,
            "queries": list(sub_sqls.values()),
        }
    elif db_type == "snowflake":
        snowflake_credential = json.load(open(os.path.join(db_path, "snowflake_credential.json")))
        if sub_queries_save_path and os.path.exists(sub_queries_save_path):
            with open(sub_queries_save_path) as f:
                queries = list(json.load(f).values())
        else:
            queries = list(sub_sqls.values())
        worker_args = {
            "db_type": "snowflake",
            "snowflake_credential": snowflake_credential,
            "database": get_db_id(question),
            "queries": queries,
        }
    else:
        return [], False

    try:
        result = subprocess.run(
            [sys.executable, "utils/load_dataframes_worker.py"],
            input=json.dumps(worker_args).encode(),
            capture_output=True,
            timeout=300,
        )

        if result.returncode == 0:
            try:
                return pickle.loads(result.stdout), False
            except Exception as e:
                logger.warning(f"Failed to deserialise data frames from worker: {e}; switching to SQL-only mode")
                return [], True
        else:
            if result.returncode == -9:
                logger.warning("Data-loading subprocess was OOM-killed (SIGKILL); switching to SQL-only mode")
            else:
                logger.warning(
                    f"Data-loading subprocess failed (rc={result.returncode}); switching to SQL-only mode"
                    + (f"\n  stderr: {result.stderr.decode(errors='replace')[:300]}" if result.stderr else "")
                )
            return [], True

    except subprocess.TimeoutExpired:
        logger.warning("Data-loading subprocess timed out; switching to SQL-only mode")
        return [], True


def write_solution(agent: Chat, sub_sqls: Dict[str, str], question: dict, db_type: str, logger,
                   program_output_dir: str = None, program_candidate_id: int = None, q_id: str = None,
                   sub_queries_save_path: str = None, db_path: str = None, plan: str = None, external_knowledge_summary: str = None,
                   similarities_path: str = None, evaluation_feedback: str = None, flexible_format_retry: bool = False,
                   sql_only: bool = False, python_only: bool = False):
    # Check if any tables have VARIANT columns
    table_names = list(sub_sqls.keys())
    has_variant = False
    if db_type == "snowflake" and db_path == "datasets/Spider2/spider2-snow" and table_names:
        has_variant = group_has_variant_columns(table_names, db_path, db_type, get_db_id(question))

    sub_sqls_formatted = _format_sub_sqls_for_prompt(sub_sqls, similarities_path, db_type, db_path, question, logger, include_df_indices=not sql_only)

    # Setup cursor getter for query_database tool
    def get_cursor():
        if db_type == "sqlite":
            db_name = get_db_id(question)
            db_file_path = os.path.join(db_path, db_name, f"{db_name}.sqlite")
            conn = sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
            return conn.cursor()
        elif db_type == "snowflake":
            snowflake_credential = json.load(open(os.path.join(db_path, "snowflake_credential.json")))
            conn = snowflake.connector.connect(**snowflake_credential)
            return conn.cursor()

    if python_only:
        # Python-only mode: force Python program output; data frames must be loaded
        data_frames, oom_occurred = _load_data_frames_for_interpreter(
            db_type, db_path, question, sub_sqls, sub_queries_save_path, logger
        )
        if oom_occurred:
            logger.error(f"[{q_id}] OOM loading data frames in python_only mode — cannot fall back to SQL; skipping question")
            return None, oom_occurred
        python_only_template = PYTHON_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE if plan is not None else PYTHON_ONLY_STITCHING_PROMPT_TEMPLATE
        if db_type == "sqlite":
            db_access_note = (
                "**Database Access in the interpreter:** A pre-connected `conn` (sqlite3.Connection) is already injected — "
                "do NOT call `sqlite3.connect()` or create new connections. Use it directly:\n"
                "- `conn.execute(\"SELECT ...\").fetchall()`\n"
                "- `pd.read_sql_query(\"SELECT ...\", conn)`\n"
                "- `get_cursor()` returns a fresh cursor from the same connection."
            )
        else:
            db_access_note = (
                "**Database Access in the interpreter:** A pre-connected `cursor` (Snowflake cursor) is already injected — "
                "do NOT create new Snowflake connections. Use it directly:\n"
                "- `cursor.execute(\"SELECT ...\"); cursor.fetchall()`\n"
                "- `get_cursor()` returns a fresh cursor from the same connection."
            )
        if plan is not None:
            stitching_prompt = python_only_template.format(
                db_type=db_type, sub_sqls=sub_sqls_formatted,
                original_question=get_question_str(question, db_type),
                plan=plan, external_knowledge=external_knowledge_summary or "",
                db_access_note=db_access_note
            )
        else:
            stitching_prompt = python_only_template.format(
                db_type=db_type, sub_sqls=sub_sqls_formatted,
                original_question=get_question_str(question, db_type),
                db_access_note=db_access_note
            )
        if evaluation_feedback:
            stitching_prompt += f"\n\n### Evaluation feedback (fix the implementation accordingly)\nA previous attempt was evaluated as incorrect. Use this feedback to produce a corrected program:\n{evaluation_feedback}"
        agent.set_system_prompt("You are an expert-level data scientist. You write Python to answer questions using pre-loaded DataFrames. You can use the python_interpreter tool to test and debug your code interactively.")
        agent.enable_tools(
            ["python_interpreter", "query_database", "get_distinct_values", "search_dimension_values"],
            db_path=db_path,
            db_type=db_type,
            cursor_getter=get_cursor,
            n_example_rows=20,
            data_frames_map={"dfs": data_frames}
        )
    elif sql_only:
        # SQL-only mode: skip Python interpreter entirely
        oom_occurred = False
        sql_only_template = SQL_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE if plan is not None else SQL_ONLY_STITCHING_PROMPT_TEMPLATE
        if has_variant:
            sql_only_template = sql_only_template.replace(
                "### Your Task",
                SNOWFLAKE_VARIANT_ACCESS_RULES + "### Your Task"
            )
        if plan is not None:
            stitching_prompt = sql_only_template.format(
                db_type=db_type, sub_sqls=sub_sqls_formatted,
                original_question=get_question_str(question, db_type),
                plan=plan, external_knowledge=external_knowledge_summary or ""
            )
        else:
            stitching_prompt = sql_only_template.format(
                db_type=db_type, sub_sqls=sub_sqls_formatted,
                original_question=get_question_str(question, db_type)
            )
        if evaluation_feedback:
            stitching_prompt += f"\n\n### Evaluation feedback (fix the implementation accordingly)\nA previous attempt was evaluated as incorrect. Use this feedback to produce a corrected program:\n{evaluation_feedback}"
        agent.set_system_prompt("You are an expert-level data analyst. You write SQL to answer questions. You can use the query_database tool to test and debug your queries interactively.")
        agent.enable_tools(
            ["query_database", "get_distinct_values", "search_dimension_values"],
            db_path=db_path,
            db_type=db_type,
            cursor_getter=get_cursor,
            n_example_rows=20,
        )
    else:
        # Hybrid mode: model can choose Python or SQL
        hybrid_stitching_prompt_template = HYBRID_STITCHING_PLANNING_PROMPT_TEMPLATE if plan is not None else HYBRID_STITCHING_PROMPT_TEMPLATE

        # Add VARIANT section to prompt if needed (insert before "If choosing SQL" section)
        if has_variant:
            if "**If choosing SQL:**" in hybrid_stitching_prompt_template:
                hybrid_stitching_prompt_template = hybrid_stitching_prompt_template.replace(
                    "**If choosing SQL:**",
                    SNOWFLAKE_VARIANT_ACCESS_RULES + "**If choosing SQL:**"
                )

        agent.set_system_prompt("You are an expert-level data scientist who can work with both Python and SQL. You can use the Python interpreter tool or the query_database tool to test and debug your code interactively.")

        if db_type == "sqlite":
            db_access_note = (
                "**Database Access in the interpreter:** A pre-connected `conn` (sqlite3.Connection) is already injected — "
                "do NOT call `sqlite3.connect()` or create new connections. Use it directly:\n"
                "- `conn.execute(\"SELECT ...\").fetchall()`\n"
                "- `pd.read_sql_query(\"SELECT ...\", conn)`\n"
                "- `get_cursor()` returns a fresh cursor from the same connection."
            )
        else:
            db_access_note = (
                "**Database Access in the interpreter:** A pre-connected `cursor` (Snowflake cursor) is already injected — "
                "do NOT create new Snowflake connections. Use it directly:\n"
                "- `cursor.execute(\"SELECT ...\"); cursor.fetchall()`\n"
                "- `get_cursor()` returns a fresh cursor from the same connection."
            )

        if plan is not None:
            stitching_prompt = hybrid_stitching_prompt_template.format(db_type=db_type, sub_sqls=sub_sqls_formatted, original_question=get_question_str(question, db_type), plan=plan, external_knowledge=external_knowledge_summary or "", db_access_note=db_access_note)
        else:
            stitching_prompt = hybrid_stitching_prompt_template.format(db_type=db_type, sub_sqls=sub_sqls_formatted, original_question=get_question_str(question, db_type), db_access_note=db_access_note)
        if evaluation_feedback:
            stitching_prompt += f"\n\n### Evaluation feedback (fix the implementation accordingly)\nA previous attempt was evaluated as incorrect. Use this feedback to produce a corrected program:\n{evaluation_feedback}"

        # Enable tools; fall back to SQL-only mode if data frames cannot fit in memory
        data_frames, oom_occurred = _load_data_frames_for_interpreter(
            db_type, db_path, question, sub_sqls, sub_queries_save_path, logger
        )
        if oom_occurred:
            logger.warning(f"[{q_id}] OOM loading data frames; switching to SQL-only mode")
            sql_only_template = SQL_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE if plan is not None else SQL_ONLY_STITCHING_PROMPT_TEMPLATE
            if has_variant:
                sql_only_template = sql_only_template.replace(
                    "### Your Task",
                    SNOWFLAKE_VARIANT_ACCESS_RULES + "### Your Task"
                )
            if plan is not None:
                stitching_prompt = sql_only_template.format(
                    db_type=db_type, sub_sqls=sub_sqls_formatted,
                    original_question=get_question_str(question, db_type),
                    plan=plan, external_knowledge=external_knowledge_summary or ""
                )
            else:
                stitching_prompt = sql_only_template.format(
                    db_type=db_type, sub_sqls=sub_sqls_formatted,
                    original_question=get_question_str(question, db_type)
                )
            if evaluation_feedback:
                stitching_prompt += f"\n\n### Evaluation feedback (fix the implementation accordingly)\nA previous attempt was evaluated as incorrect. Use this feedback to produce a corrected program:\n{evaluation_feedback}"
            agent.enable_tools(
                ["query_database", "get_distinct_values", "search_dimension_values"],
                db_path=db_path,
                db_type=db_type,
                cursor_getter=get_cursor,
                n_example_rows=20,
            )
        else:
            agent.enable_tools(
                ["python_interpreter", "query_database", "get_distinct_values", "search_dimension_values"],
                db_path=db_path,
                db_type=db_type,
                cursor_getter=get_cursor,
                n_example_rows=20,
                data_frames_map={"dfs": data_frames}
            )

    logger.info(f"Tool calling enabled: {agent.tool_calling_enabled}, tool functions: {agent.tool_functions}")

    # Get response - model can choose Python or SQL
    try:
        model_response = agent.get_response(stitching_prompt, logger=logger)
        response_text = model_response["text"]
    except (Exception, SystemExit) as e:
        logger.error(f"Failed to get response from model for question {q_id}: {e}")
        traceback.print_exc()
        return None, oom_occurred

    logger.info(f"<hybrid-gen>\n\nPrompt:\n\n{stitching_prompt}\n\nThoughts:\n\n{model_response.get('thoughts', '')}\n\nResponse:\n\n{response_text}</hybrid-gen>")

    # Extract code blocks - check for both Python and SQL
    python_blocks = extract_all_blocks(response_text, "python")
    sql_blocks = extract_all_blocks(response_text, "sql")

    # Determine which approach was chosen; sql_only never allows Python, python_only never allows SQL
    use_python = len(python_blocks) > 0 and not sql_only
    use_sql = len(sql_blocks) > 0 and not python_only

    for _nudge in range(MAX_RETRIES):
        if use_python or use_sql:
            break
        if oom_occurred or sql_only:
            nudge = (
                "Your response did not contain a code block. "
                "Please provide your final answer now in a ```sql``` code block."
            )
        elif python_only:
            nudge = (
                "Your response did not contain a code block. "
                "Please provide your final answer now in a ```python``` code block."
            )
        else:
            nudge = (
                "Your response did not contain a code block. "
                "Please provide your final answer now in either a ```python``` or ```sql``` code block."
            )
        logger.warning(f"No code block in response for {q_id} (nudge {_nudge + 1}/{MAX_RETRIES})")
        try:
            retry_response = agent.get_response(nudge, logger=logger)
            response_text = retry_response["text"]
            python_blocks = extract_all_blocks(response_text, "python")
            sql_blocks = extract_all_blocks(response_text, "sql")
            use_python = len(python_blocks) > 0 and not sql_only
            use_sql = len(sql_blocks) > 0 and not python_only
        except (Exception, SystemExit) as e:
            logger.error(f"Nudge {_nudge + 1} failed for question {q_id}: {e}")
            break

    if not use_python and not use_sql:
        logger.error(f"Neither Python nor SQL code block found after nudges for question {q_id}")
        logger.error(f"Response text: {response_text[:500]}...")
        return None, oom_occurred

    failed_to_solve = True
    result_csv_path = os.path.join(program_output_dir, f"program_output_{program_candidate_id}.csv")

    for turn in range(MAX_RETRIES):
        if use_python:
            # Python mode - existing logic
            python_program = _trim_to_processing_function(python_blocks[-1])
            program_path = os.path.join(program_output_dir, f"program_{program_candidate_id}.py")
            with open(program_path, "w") as f:
                f.write(python_program)

            # Ensure sub-queries are on disk so program_frame.py can read them
            _sub_queries_path = sub_queries_save_path or os.path.join(program_output_dir, f"sub_queries_{program_candidate_id}.json")
            if not os.path.exists(_sub_queries_path) and sub_sqls:
                with open(_sub_queries_path, "w") as _f:
                    json.dump(sub_sqls, _f)

            if db_type == "sqlite":
                db_name = get_db_id(question)
                db_file_path = os.path.join(db_path, db_name, f"{db_name}.sqlite")
                command = [sys.executable, "utils/program_frame.py", db_file_path, str(q_id), program_path, _sub_queries_path, result_csv_path]
            elif db_type == "snowflake":
                command = [sys.executable, "utils/program_frame.py", os.path.join(db_path, "snowflake_credential.json"), q_id, program_path, _sub_queries_path, result_csv_path]

            try:
                timeout = 300 # 5 mins #TODO: make this configurable
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
            except subprocess.CalledProcessError as e:
                error_lines = e.stderr.splitlines()
                relevant_lines = []

                i = 0
                while i < len(error_lines) - 1: # Stop before the last line
                    line = error_lines[i]

                    # Check if it's the File line we care about
                    if line.strip().startswith("File ") and f"program_{program_candidate_id}.py" in line:
                        # Add the 'File ...' line
                        relevant_lines.append(line)

                        # Add the next line, which is the code snippet
                        # We also check it's not another 'File' line, just in case
                        next_line = error_lines[i + 1]
                        if not next_line.strip().startswith("File "):
                            relevant_lines.append(next_line)
                            i += 1 # Skip the next line since we just added it

                    i += 1

                # Always add the final exception message (the last line)
                try:
                    relevant_lines.append(error_lines[-1])
                except IndexError:
                    relevant_lines.append(f"The program timed out after {timeout} seconds!")
                llm_trace_str = "\n".join(relevant_lines)

                if len(llm_trace_str) == 0:
                    logger.info("[NOTICE]Somehow the error trace doesn't contain the LLMs script error")
                    llm_trace_str = str(e.stderr)

                if flexible_format_retry:
                    err_message = f"The model got the following error:\n\n{llm_trace_str}\n\nPlease revise. You may fix the Python program (```python``` block) or switch to SQL (```sql``` block) — whichever you think is more reliable."
                else:
                    err_message = f"The model got the following error:\n\n{llm_trace_str}\n\nPlease revise the program. Remember to put the revised program within a ```python``` block"

                try:
                    if flexible_format_retry:
                        retry_response = agent.get_response(err_message, logger=logger)
                        retry_text = retry_response["text"]
                        new_python = extract_all_blocks(retry_text, "python")
                        new_sql = extract_all_blocks(retry_text, "sql")
                        if new_python:
                            logger.info(f"<error>\n{err_message}\n\nRevised Program (python):\n{new_python[-1]}\n</error>")
                            python_blocks = new_python
                            use_python, use_sql = True, False
                        elif new_sql:
                            logger.info(f"<error>\n{err_message}\n\nRevised Program (sql):\n{new_sql[-1]}\n</error>")
                            sql_blocks = new_sql
                            use_python, use_sql = False, True
                        else:
                            logger.error("Neither Python nor SQL block found in retry response. Giving up.")
                            failed_to_solve = True
                            break
                    else:
                        python_program = agent.get_code_blocks(err_message, code_format="python", logger=logger)["code_blocks"][-1]
                        logger.info(f"<error>\n{err_message}\n\nRevised Program:\n{python_program}\n</error>")
                        python_blocks = [python_program]
                except (Exception, SystemExit) as retry_error:
                    logger.error(f"Failed to get revised program: {retry_error}")
                    logger.error(f"Original error trace: {llm_trace_str}")
                    failed_to_solve = True
                    break
                continue
            except subprocess.TimeoutExpired:
                logger.error(f"Process for question {q_id} timed out. Path: {program_path}")
                break

        elif use_sql:
            # SQL mode - execute query and save to CSV
            sql_query = sql_blocks[-1]

            # format the sql query to quote the identifiers
            try:
                sql_query = post_format_generated_query(sql_query, db_path, db_type, include_comment=False)
            except Exception as e:
                logger.error(f"Failed to format SQL query: {e}. But continue to give it a try for execution anyway.")

            sql_path = os.path.join(program_output_dir, f"program_{program_candidate_id}.sql")
            with open(sql_path, "w") as f:
                f.write(sql_query)

            conn = None
            cursor = None
            try:
                # Get connection and cursor
                if db_type == "sqlite":
                    db_name = get_db_id(question)
                    db_file_path = os.path.join(db_path, db_name, f"{db_name}.sqlite")
                    conn = sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
                    cursor = conn.cursor()
                elif db_type == "snowflake":
                    snowflake_credential = json.load(open(os.path.join(db_path, "snowflake_credential.json")))
                    conn = snowflake.connector.connect(**snowflake_credential)
                    cursor = conn.cursor()

                cursor.execute(sql_query)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                result_df = pd.DataFrame(rows, columns=col_names)

                # Save to CSV following the same convention as Python programs
                result_df.to_csv(result_csv_path, index=False)

            except Exception as e:
                if flexible_format_retry:
                    error_msg = f"The SQL query got the following error:\n\n{str(e)}\n\nPlease revise. You may fix the SQL query (```sql``` block) or switch to Python (```python``` block) — whichever you think is more reliable."
                else:
                    error_msg = f"The SQL query got the following error:\n\n{str(e)}\n\nPlease revise the query. Remember to put the revised query within a ```sql``` block"
                try:
                    if flexible_format_retry:
                        retry_response = agent.get_response(error_msg, logger=logger)
                        retry_text = retry_response["text"]
                        new_sql = extract_all_blocks(retry_text, "sql")
                        new_python = extract_all_blocks(retry_text, "python")
                        if new_sql:
                            logger.info(f"<error>\n{error_msg}\n\nRevised Program (sql):\n{new_sql[-1]}\n</error>")
                            sql_blocks = new_sql
                            use_python, use_sql = False, True
                        elif new_python:
                            logger.info(f"<error>\n{error_msg}\n\nRevised Program (python):\n{new_python[-1]}\n</error>")
                            python_blocks = new_python
                            use_python, use_sql = True, False
                        else:
                            logger.error("Neither SQL nor Python block found in retry response. Giving up.")
                            failed_to_solve = True
                            break
                    else:
                        sql_query = agent.get_code_blocks(error_msg, code_format="sql", logger=logger)["code_blocks"][-1]
                        logger.info(f"<error>\n{error_msg}\n\nRevised Query:\n{sql_query}\n</error>")
                        sql_blocks = [sql_query]
                except (Exception) as retry_error:
                    logger.error(f"Failed to get revised query: {retry_error}")
                    logger.error(f"Original error: {e}")
                    failed_to_solve = True
                    break
                continue
            finally:
                # Clean up connections
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

        failed_to_solve = False
        break

    if failed_to_solve:
        logger.error(f"[Program_{program_candidate_id}]Failed to process question {q_id} after retries.</program>")
    else:
        logger.info(f"[Program_{program_candidate_id}]Successfully processed question {q_id}!</program>")

    # Return the final code (Python or SQL) and whether OOM occurred
    if use_python:
        return (python_blocks[-1] if python_blocks else None), oom_occurred
    else:
        return (sql_blocks[-1] if sql_blocks else None), oom_occurred
