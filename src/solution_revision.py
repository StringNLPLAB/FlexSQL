import json
import os
import re
import sqlite3
from typing import Tuple

import pandas as pd
import snowflake.connector

from chat import Chat
from solution_writing import _load_data_frames_for_interpreter
from utils import get_db_id, get_question_str


def evaluate_program_result(
    evaluation_agent: Chat,
    question: dict,
    program_path: str,
    result_csv_path: str,
    program_output_dir: str,
    db_type: str,
    db_path: str,
    sub_queries_save_path: str,
    logger,
    q_id: str,
    external_knowledge_summary: str = None,
    oom_occurred: bool = False,
    no_planning_repair: bool = False,
    plan: str = None,
) -> Tuple[bool, str, str]:
    """
    Use an evaluation agent with python_interpreter and query_database tools to verify
    if the generated program's output makes sense for the question, and to classify
    the cause if it does not (plan vs implementation).
    Returns (makes_sense: bool, feedback: str, issue_type: str).
    issue_type is one of "plan" | "implementation" when makes_sense is False;
    otherwise "implementation" (arbitrary, not used).
    When no_planning_repair=True, only execution repair is available; the evaluator
    prompt is simplified to ask for correctness + feedback only (no issue classification).
    """
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
        return None

    sub_sqls = {}
    if sub_queries_save_path and os.path.exists(sub_queries_save_path):
        with open(sub_queries_save_path, "r") as f:
            sub_sqls = json.load(f)

    if oom_occurred:
        data_frames = []
    else:
        data_frames, _ = _load_data_frames_for_interpreter(
            db_type, db_path, question, sub_sqls, sub_queries_save_path, logger
        )

    output_df = pd.DataFrame()
    if os.path.exists(result_csv_path):
        try:
            output_df = pd.read_csv(result_csv_path)
        except Exception as e:
            logger.warning(f"Failed to load program output CSV for evaluation: {e}")

    dfs_note = (
        " Note: input dataframes (dfs) are not available because the source tables were too large to load into memory; use query_database to inspect the data instead."
        if oom_occurred else "The input dataframes are available in a variable named 'dfs'"
    )
    # the input and output dataframes are pre-loaded. and
    evaluation_agent.clear_chat_history()

    if no_planning_repair:
        evaluation_agent.set_system_prompt(
            "You are an evaluator. Your job is to verify whether a program's output correctly answers a question. "
            "You have access to the python_interpreter and query_database tools to run checks (e.g. spot-check values, run queries). "
            "In the interpreter, the output dataframe is available in a variable named 'program_output'."
            + dfs_note
            + "You must respond with a single JSON object in a ```json``` block with: {\"makes_sense\": true or false, \"feedback\": \"brief explanation\"}. "
            "Set makes_sense to true only if the result is correct and answers the question. When makes_sense is false, provide actionable feedback describing what is wrong with the implementation."
        )
    else:
        evaluation_agent.set_system_prompt(
            "You are an evaluator. Your job is to verify whether a program's output correctly answers a question, and if not, to decide whether the problem is due to the high-level plan (wrong steps, wrong tables, wrong logic) or due to the program implementation (bugs in code, wrong joins/filters, aggregation errors). "
            "You have access to the python_interpreter and query_database tools to run checks (e.g. spot-check values, run queries). "
            "In the interpreter, the output dataframe is available in a variable named 'program_output'."
            + dfs_note
            + "You must respond with a single JSON object in a ```json``` block with: {\"makes_sense\": true or false, \"feedback\": \"brief explanation\", \"issue_type\": \"plan\" or \"implementation\"}. "
            "Set makes_sense to true only if the result is correct and answers the question. When makes_sense is false, set issue_type to \"plan\" if the error comes from the plan (wrong approach, wrong tables/steps); set issue_type to \"implementation\" if the plan is fine but the code/SQL is wrong."
        )
    database_name = get_db_id(question)
    data_frames_map = {"program_output": output_df}
    if not oom_occurred:
        data_frames_map["dfs"] = data_frames
    evaluation_agent.enable_tools(
        ["python_interpreter", "query_database", "list_columns", "list_tables", "get_distinct_values", "search_dimension_values"],
        db_path=db_path,
        db_type=db_type,
        cursor_getter=get_cursor,
        n_example_rows=20,
        database_name=database_name,
        data_frames_map=data_frames_map,
    )

    program_content = ""
    code_lang = "python"
    if os.path.exists(program_path):
        with open(program_path, "r") as f:
            program_content = f.read()
        if program_path.endswith(".sql"):
            code_lang = "sql"

    question_str = get_question_str(question, db_type)
    knowledge_section = f"\n### External Knowledge\n{external_knowledge_summary}\n" if external_knowledge_summary and external_knowledge_summary.strip() else ""
    plan_section = f"\n### Natural Language Plan\n{plan}\n" if plan else ""
    prompt = f"""### Question
{question_str}
{knowledge_section}{plan_section}
### Generated program
```{code_lang}
{program_content}
```

### Program output as dataframe
The program's output is available in the python_interpreter in the variable 'program_output'. You can also use dfs (input dataframes). Verify whether the output correctly answers the question above.

Use the python_interpreter or query_database tools if needed to verify. Then respond with exactly one JSON object in a ```json``` code block."""
    if no_planning_repair:
        prompt += " When makes_sense is false, provide actionable feedback on what is wrong with the implementation.\n### Output Format\n```json\n{\"makes_sense\": true or false, \"feedback\": \"brief explanation\"}\n```\n"
    else:
        prompt += " When makes_sense is false, you must set issue_type to \"plan\" (wrong high-level plan/steps/tables) or \"implementation\" (plan is OK but code/SQL has bugs).\n### Output Format\n```json\n{\"makes_sense\": true or false, \"feedback\": \"brief explanation\", \"issue_type\": \"plan\" or \"implementation\"}\n```\n"
    logger.info(f"[self-eval] Evaluation prompt:\n{prompt}")
    try:
        response = evaluation_agent.get_code_blocks(
            prompt, code_format="json", logger=logger
        )
        thoughts = response.get("thoughts")
        if thoughts:
            logger.info(f"[self-eval] Evaluation thoughts:\n{thoughts}")
        blocks = response.get("code_blocks", [])
        if blocks:
            logger.info(f"[self-eval] Evaluation output text (JSON block):\n{blocks[0]}")
        if not blocks:
            logger.warning("Evaluation agent returned no JSON block; assuming makes_sense=False")
            return False, "Evaluation agent did not return a valid JSON verdict.", "implementation" if no_planning_repair else "plan"
        raw = blocks[0].strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        ev = json.loads(raw)
        makes_sense = bool(ev.get("makes_sense", False))
        feedback = str(ev.get("feedback", "")) or "No feedback provided."
        if no_planning_repair:
            issue_type = "implementation"
        else:
            issue_type = (ev.get("issue_type") or "plan").strip().lower()
            if issue_type not in ("plan", "implementation"):
                issue_type = "plan"  # default to plan when unclear
        return makes_sense, feedback, issue_type
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse evaluation response: {e}")
        return False, f"Failed to parse evaluation: {e}", "implementation" if no_planning_repair else "plan"
