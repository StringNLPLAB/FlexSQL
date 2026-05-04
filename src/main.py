from chat import Chat
from get_ddl import list_tables
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import random
import sqlite3
import sys

import datetime
import traceback
import snowflake.connector
from typing import Dict, Optional
from logger import initialize_logger, _zero_token_usage, _merge_usage
from planning import planning_batch_generate, plan_clarification, planning_no_diverse_generate
from schema_linking import hierarchical_schema_linking, get_schema_list
from solution_revision import evaluate_program_result
from solution_writing import write_solution
from utils import get_q_id, get_db_id, get_question_str

def _stash_candidate_files(program_output_dir: str, program_candidate_id_str: str) -> Dict[str, str]:
    """Rename program + CSV for this candidate to *.bak siblings; return {orig: bak}.

    os.replace overwrites any stale *.bak left from a prior crashed run.
    """
    backups: Dict[str, str] = {}
    for ext in (".py", ".sql"):
        p = os.path.join(program_output_dir, f"program_{program_candidate_id_str}{ext}")
        if os.path.exists(p):
            bak = p + ".bak"
            os.replace(p, bak)
            backups[p] = bak
    csv_p = os.path.join(program_output_dir, f"program_output_{program_candidate_id_str}.csv")
    if os.path.exists(csv_p):
        bak = csv_p + ".bak"
        os.replace(csv_p, bak)
        backups[csv_p] = bak
    return backups

def _commit_or_rollback_candidate_files(
    program_output_dir: str,
    program_candidate_id_str: str,
    backups: Dict[str, str],
    logger,
    log_prefix: str = "",
) -> bool:
    """Commit new (program, csv) iff both exist; otherwise restore backups.

    Returns True on commit, False on rollback.
    """
    new_prog = any(
        os.path.exists(os.path.join(program_output_dir, f"program_{program_candidate_id_str}{ext}"))
        for ext in (".py", ".sql")
    )
    new_csv = os.path.exists(
        os.path.join(program_output_dir, f"program_output_{program_candidate_id_str}.csv")
    )
    if new_prog and new_csv:
        for bak in backups.values():
            if os.path.exists(bak):
                os.remove(bak)
        logger.info(f"{log_prefix}retry committed (new program+csv written).")
        return True
    for orig, bak in backups.items():
        if os.path.exists(orig):
            os.remove(orig)
        if os.path.exists(bak):
            os.replace(bak, orig)
    logger.warning(
        f"{log_prefix}retry rolled back (new_prog={new_prog}, new_csv={new_csv}); "
        f"preserved previous (program, output)."
    )
    return False

def process_question(question, args, output_dir):
    """
    This function contains the logic for processing a single question.
    It's the body of the original for loop.
    """
    question_usage = _zero_token_usage()
    sub_sqls_agent: Optional[Chat] = None
    sql_stitching_agent: Optional[Chat] = None
    evaluation_agent: Optional[Chat] = None
    q_id = None
    status = "success"
    error_message = ""
    try:
        q_id = get_q_id(question, db_type=args.db_type)
            
        program_output_dir = os.path.join(output_dir, str(q_id))
        os.makedirs(program_output_dir, exist_ok=True)
        sub_queries_save_path = os.path.join(program_output_dir, "sub_queries.json")
        plans_save_path = os.path.join(program_output_dir, "plans.json")

        logger = initialize_logger(log_path=os.path.join(program_output_dir, "log.txt"))
        logger.info(f"Processing question {q_id}")

    
        # Each thread creates its own agent to avoid sharing state
        sub_sqls_agent = Chat(
            args.model,
            base_url=args.base_url,
            ip=args.ip,
            port=args.port,
        )
        
        sub_sqls_per_plan = [] # list of sub_sqls dicts for each plan or single run
        external_knowledge_summary = ""
        
        if args.db_type == "sqlite": 
            db_name = get_db_id(question)
            db_file_path = os.path.join(args.db_path, db_name, f"{db_name}.sqlite")
            conn = sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
            cursor = conn.cursor()
        elif args.db_type == "snowflake":
            snowflake_credential = json.load(open(os.path.join(args.db_path, "snowflake_credential.json")))
            conn = snowflake.connector.connect(**snowflake_credential, database = get_db_id(question))
            cursor = conn.cursor()
        cursor_getter = lambda: cursor

        if args.use_2step_batch_planning:
                planning_agent = Chat(
                    args.model,
                    base_url=args.base_url,
                    ip=args.ip,
                    port=args.port,
                    gen_config=args.planning_gen_config or None,
                )
                table_names = list_tables(
                    db_folder=args.db_path,
                    db_type=args.db_type,
                    question_id=q_id,
                    database_id=get_db_id(question),
                    use_gold_tables=args.use_gold_tables,
                )
                # Get schema list
                schema_list = get_schema_list(
                    database_name=get_db_id(question),
                    db_type=args.db_type,
                    cursor_getter=cursor_getter,
                    db_path=args.db_path
                )

                if args.db_type == "snowflake" and len(schema_list) <= 1 and not args.use_gold_tables:
                    logger.warning(f"Only {len(schema_list)} schemas found for question {q_id}. Skipping hierarchical schema linking.")
                    args.hierarchical_sl = False

                # Hierarchical schema linking step
                if args.hierarchical_sl and not args.use_gold_tables and args.db_type == "snowflake":
                    logger.info(f"Performing hierarchical schema linking for question {q_id}")
                    hierarchical_sl_agent = Chat(
                        args.model,
                        base_url=args.base_url,
                        ip=args.ip,
                        port=args.port,
                    )

                    # Perform hierarchical schema linking
                    relevant_schemas = hierarchical_schema_linking(
                        agent=hierarchical_sl_agent,
                        question=question,
                        database_name=get_db_id(question),
                        schema_list=schema_list,
                        db_type=args.db_type,
                        cursor_getter=cursor_getter,
                        db_path=args.db_path,
                        logger=logger,
                    )

                    logger.info(f"Hierarchical schema linking identified {len(relevant_schemas)} relevant schemas: {relevant_schemas}")
                    logger.debug(f"All tables: {table_names}")
                    logger.debug(f"Relevant schemas: {relevant_schemas}")
                    if relevant_schemas:
                        filtered = [
                            t for t in table_names
                            if any(schema.lower() in t.split(".")[-2].lower() for schema in relevant_schemas)
                        ]
                        table_names = filtered
                        logger.info(f"Filtered to {len(table_names)} tables: {table_names}")
                    else:
                        logger.warning(f"No relevant schemas found from hierarchical schema linking. Using all schemas.")

                # Step 1: Generate plans without value grounding
                logger.info(f"Step 1: Generating plans for question {q_id}")

                plans, external_knowledge_summary = planning_batch_generate(
                    planning_agent,
                    question,
                    table_names=table_names,
                    logger=logger,
                    top_k=args.planning_top_k,
                    table_similarities_path=args.similarities_path,
                    db_type=args.db_type,
                    db_path=args.db_path,
                    cursor_getter=cursor_getter,
                    batch_size=args.planning_batch_size,
                )

                # Step 2: Clarify plans and generate sub-SQLs
                if plans:
                    logger.info(f"Step 2: Clarifying {len(plans)} plans for question {q_id}")
                    # Clear chat history before clarification step
                    planning_agent.clear_chat_history()

                    plans = plan_clarification(
                        planning_agent,
                        question,
                        plans,
                        logger=logger,
                        db_type=args.db_type,
                        db_path=args.db_path,
                        cursor_getter=cursor_getter,
                        external_knowledge_summary=external_knowledge_summary,
                        similarities_path=args.similarities_path,
                        use_gold_tables=args.use_gold_tables,
                        table_names=table_names,
                    )
                else:
                    logger.warning(f"No plans generated in step 1 for question {q_id}")

                with open(plans_save_path, "w") as f:
                    json.dump(plans, f, indent="\t", ensure_ascii=False)
                    
                for plan_idx, plan_item in enumerate(plans):
                    if plan_idx == args.planning_top_k:
                        break
                    # Plans from plan_clarification always have format: (plan_text, plan_tables, sub_sqls)
                    if len(plan_item) == 3:
                        plan_text, plan_tables, sub_sqls_plan = plan_item
                    else:
                        logger.error(f"Unexpected plan format for plan {plan_idx}: expected 3-tuple from plan_clarification, got {len(plan_item)}-tuple. Skipping.")
                        continue
                    
                    sub_sqls_per_plan.append(sub_sqls_plan)
                    with open(os.path.join(program_output_dir, f"sub_queries_plan_{plan_idx}.json"), "w") as f:
                        json.dump(sub_sqls_plan, f, indent="\t", ensure_ascii=False)

        elif args.use_no_diverse_planning:
                planning_agent = Chat(
                    args.model,
                    base_url=args.base_url,
                    ip=args.ip,
                    port=args.port,
                    gen_config=args.planning_gen_config or None,
                )
                table_names = list_tables(
                    db_folder=args.db_path,
                    db_type=args.db_type,
                    question_id=q_id,
                    database_id=get_db_id(question),
                    use_gold_tables=args.use_gold_tables,
                )
                # Get schema list
                schema_list = get_schema_list(
                    database_name=get_db_id(question),
                    db_type=args.db_type,
                    cursor_getter=cursor_getter,
                    db_path=args.db_path
                )

                if args.db_type == "snowflake" and len(schema_list) <= 1 and not args.use_gold_tables:
                    logger.warning(f"Only {len(schema_list)} schemas found for question {q_id}. Skipping hierarchical schema linking.")
                    args.hierarchical_sl = False

                # Hierarchical schema linking step
                if args.hierarchical_sl and not args.use_gold_tables and args.db_type == "snowflake":
                    logger.info(f"Performing hierarchical schema linking for question {q_id}")
                    hierarchical_sl_agent = Chat(
                        args.model,
                        base_url=args.base_url,
                        ip=args.ip,
                        port=args.port,
                    )

                    # Perform hierarchical schema linking
                    relevant_schemas = hierarchical_schema_linking(
                        agent=hierarchical_sl_agent,
                        question=question,
                        database_name=get_db_id(question),
                        schema_list=schema_list,
                        db_type=args.db_type,
                        cursor_getter=cursor_getter,
                        db_path=args.db_path,
                        logger=logger,
                    )

                    logger.info(f"Hierarchical schema linking identified {len(relevant_schemas)} relevant schemas: {relevant_schemas}")
                    logger.debug(f"All tables: {table_names}")
                    logger.debug(f"Relevant schemas: {relevant_schemas}")
                    if relevant_schemas:
                        filtered = [
                            t for t in table_names
                            if any(schema.lower() in t.split(".")[-2].lower() for schema in relevant_schemas)
                        ]
                        table_names = filtered
                        logger.info(f"Filtered to {len(table_names)} tables: {table_names}")
                    else:
                        logger.warning(f"No relevant schemas found from hierarchical schema linking. Using all schemas.")

                logger.info(f"No-diverse planning: generating {args.planning_top_k} plans for question {q_id}")
                plans, external_knowledge_summary = planning_no_diverse_generate(
                    planning_agent,
                    question,
                    table_names=table_names,
                    logger=logger,
                    top_k=args.planning_top_k,
                    table_similarities_path=args.similarities_path,
                    db_type=args.db_type,
                    db_path=args.db_path,
                    cursor_getter=cursor_getter,
                )

                if not plans:
                    logger.warning(f"No plans generated for question {q_id}")

                with open(plans_save_path, "w") as f:
                    json.dump(plans, f, indent="\t", ensure_ascii=False)

                for plan_idx, plan_item in enumerate(plans):
                    if plan_idx == args.planning_top_k:
                        break
                    if len(plan_item) == 3:
                        plan_text, plan_tables, sub_sqls_plan = plan_item
                    else:
                        logger.error(f"Unexpected plan format for plan {plan_idx}: expected 3-tuple from planning_no_diverse_generate, got {len(plan_item)}-tuple. Skipping.")
                        continue

                    sub_sqls_per_plan.append(sub_sqls_plan)
                    with open(os.path.join(program_output_dir, f"sub_queries_plan_{plan_idx}.json"), "w") as f:
                        json.dump(sub_sqls_plan, f, indent="\t", ensure_ascii=False)

        else:
            # Without planning.
            # TODO: Look at this later if we want to do ablation study on this.
            pass
            
        # if schema_filtering fails -> we'll try to do some schema compression later too.
        sub_sqls_agent.clear_chat_history()
    
        sql_stitching_agent = Chat(
            args.model,
            base_url=args.base_url,
            ip=args.ip,
            port=args.port,
            gen_config=args.stitching_gen_config or None,
        )
        if args.use_2step_batch_planning or args.use_no_diverse_planning:
            evaluation_agent = Chat(
                args.model,
                base_url=args.base_url,
                ip=args.ip,
                port=args.port,
                gen_config=args.eval_gen_config or None,
            )

        for plan_idx, sub_sqls_for_plan in enumerate(sub_sqls_per_plan):
            _keys = list(sub_sqls_for_plan.keys())
            random.shuffle(_keys)
            sub_sqls_for_plan = {k: sub_sqls_for_plan[k] for k in _keys}
            if args.use_2step_batch_planning or args.use_no_diverse_planning:
                # Overwrite the pre-written JSON so Snowflake df loading matches the shuffled order
                with open(os.path.join(program_output_dir, f"sub_queries_plan_{plan_idx}.json"), "w") as _f:
                    json.dump(sub_sqls_for_plan, _f, indent="\t", ensure_ascii=False)
            for program_candidate_id in range(args.num_programs):
                logger.info(f"<program plan_idx={plan_idx} program_id={program_candidate_id}> Start generation!")
                if args.use_2step_batch_planning or args.use_no_diverse_planning:
                    sub_queries_save_path = os.path.join(program_output_dir, f"sub_queries_plan_{plan_idx}.json")
                else:
                    sub_queries_save_path = sub_queries_save_path

                sql_stitching_agent.clear_chat_history()
                if args.use_2step_batch_planning or args.use_no_diverse_planning:
                    plan_item = plans[plan_idx]
                    # Handle both old format (plan_text, plan_tables) and new format (plan_text, plan_tables, columns)
                    if isinstance(plan_item, tuple) and len(plan_item) >= 2:
                        current_plan = plan_item[0]  # Extract plan_text
                    elif isinstance(plan_item, (str, list)):
                        current_plan = plan_item  # Old format where plan is directly the text
                    else:
                        current_plan = None
                        logger.warning(f"Unexpected plan format for plan {plan_idx}, setting current_plan to None")

                program_candidate_id_str = f"{plan_idx}_{program_candidate_id}" if (args.use_2step_batch_planning or args.use_no_diverse_planning) else program_candidate_id
                _, oom_occurred = write_solution(
                    sql_stitching_agent,
                    sub_sqls_for_plan,
                    question,
                    args.db_type,
                    logger=logger,
                    program_output_dir=program_output_dir,
                    program_candidate_id=program_candidate_id_str,
                    q_id=q_id,
                    sub_queries_save_path=sub_queries_save_path,
                    db_path=args.db_path,
                    similarities_path=args.similarities_path,
                    plan=current_plan,
                    external_knowledge_summary=external_knowledge_summary,
                    flexible_format_retry=args.flexible_format_retry,
                    sql_only=args.sql_only,
                    python_only=args.python_only,
                )

                if (args.use_2step_batch_planning or args.use_no_diverse_planning) and evaluation_agent is not None:
                    for _eval_round in range(args.max_self_eval_rounds):
                        logger.info(f"[self-eval] plan_idx={plan_idx} program_candidate_id={program_candidate_id} round={_eval_round}: starting evaluation.")
                        if oom_occurred:
                            program_path = os.path.join(program_output_dir, f"program_{program_candidate_id_str}.sql")
                            if not os.path.exists(program_path):
                                program_path = os.path.join(program_output_dir, f"program_{program_candidate_id_str}.py")
                        else:
                            program_path = os.path.join(program_output_dir, f"program_{program_candidate_id_str}.py")
                            if not os.path.exists(program_path):
                                program_path = os.path.join(program_output_dir, f"program_{program_candidate_id_str}.sql")
                        result_csv_path = os.path.join(program_output_dir, f"program_output_{program_candidate_id_str}.csv")
                        makes_sense, feedback, issue_type = evaluate_program_result(
                            evaluation_agent,
                            question=question,
                            program_path=program_path,
                            result_csv_path=result_csv_path,
                            program_output_dir=program_output_dir,
                            db_type=args.db_type,
                            db_path=args.db_path,
                            sub_queries_save_path=sub_queries_save_path,
                            logger=logger,
                            q_id=q_id,
                            external_knowledge_summary=external_knowledge_summary,
                            oom_occurred=oom_occurred,
                            no_planning_repair=args.no_planning_repair,
                            plan=current_plan,
                        )
                        logger.info(f"[self-eval] plan_idx={plan_idx} program_candidate_id={program_candidate_id} round={_eval_round}: makes_sense={makes_sense}, issue_type={issue_type}" + (f", feedback={feedback[:200]}..." if feedback and len(feedback) > 200 else (f", feedback={feedback}" if feedback else "")))
                        if makes_sense:
                            logger.info(f"[self-eval] plan_idx={plan_idx}: passed.")
                            break
                        if not makes_sense and not feedback:
                            logger.info(f"[self-eval] plan_idx={plan_idx} round={_eval_round}: rejected but no feedback, cannot improve. Stopping.")
                            break
                        if not makes_sense and feedback:
                            if issue_type == "plan":
                                logger.info(f"[self-eval] plan_idx={plan_idx}: rejected (plan issue). Re-running plan_clarification with feedback.")
                                plan_input_single = [(plans[plan_idx][0], plans[plan_idx][1])]
                                planning_agent.clear_chat_history()
                                clarified_single = plan_clarification(
                                    planning_agent,
                                    question,
                                    plan_input_single,
                                    logger=logger,
                                    db_type=args.db_type,
                                    db_path=args.db_path,
                                    cursor_getter=cursor_getter,
                                    external_knowledge_summary=external_knowledge_summary,
                                    similarities_path=args.similarities_path,
                                    use_gold_tables=args.use_gold_tables,
                                    table_names=table_names,
                                    feedback_per_plan={0: feedback},
                                )
                                plans[plan_idx] = clarified_single[0]
                                with open(plans_save_path, "w") as f:
                                    json.dump(plans, f, indent="\t", ensure_ascii=False)
                                logger.info(f"[self-eval] plan_idx={plan_idx}: plan_clarification with feedback done. Retrying write_solution.")
                                plan_item_new = plans[plan_idx]
                                if len(plan_item_new) == 3:
                                    _, _, sub_sqls_plan_new = plan_item_new
                                    _keys = list(sub_sqls_plan_new.keys())
                                    random.shuffle(_keys)
                                    sub_sqls_plan_new = {k: sub_sqls_plan_new[k] for k in _keys}
                                    sub_sqls_per_plan[plan_idx] = sub_sqls_plan_new
                                    with open(os.path.join(program_output_dir, f"sub_queries_plan_{plan_idx}.json"), "w") as f:
                                        json.dump(sub_sqls_plan_new, f, indent="\t", ensure_ascii=False)
                                sub_sqls_for_plan = sub_sqls_per_plan[plan_idx]
                                current_plan = plans[plan_idx][0] if len(plans[plan_idx]) >= 1 else current_plan
                                _backups = _stash_candidate_files(program_output_dir, program_candidate_id_str)
                                sql_stitching_agent.clear_chat_history()
                                try:
                                    _, oom_occurred = write_solution(
                                        sql_stitching_agent,
                                        sub_sqls_for_plan,
                                        question,
                                        args.db_type,
                                        logger=logger,
                                        program_output_dir=program_output_dir,
                                        program_candidate_id=program_candidate_id_str,
                                        q_id=q_id,
                                        sub_queries_save_path=sub_queries_save_path,
                                        db_path=args.db_path,
                                        similarities_path=args.similarities_path,
                                        plan=current_plan,
                                        external_knowledge_summary=external_knowledge_summary,
                                        flexible_format_retry=args.flexible_format_retry,
                                        sql_only=args.sql_only,
                                        python_only=args.python_only,
                                    )
                                finally:
                                    _commit_or_rollback_candidate_files(
                                        program_output_dir,
                                        program_candidate_id_str,
                                        _backups,
                                        logger,
                                        log_prefix=f"[self-eval] q_id={q_id} plan_idx={plan_idx} program_candidate_id={program_candidate_id} (plan-issue): ",
                                    )
                                logger.info(f"[self-eval] plan_idx={plan_idx}: write_solution retry complete (after plan_clarification).")
                            else:
                                # issue_type == "implementation": re-run write_solution with old program + feedback
                                logger.info(f"[self-eval] plan_idx={plan_idx}: rejected (implementation issue). Re-running write_solution with evaluation feedback.")
                                old_program_content = ""
                                if os.path.exists(program_path):
                                    with open(program_path) as _f:
                                        old_program_content = _f.read()
                                ext = "sql" if program_path.endswith(".sql") else "python"
                                full_feedback = (
                                    f"### Previous program:\n```{ext}\n{old_program_content}\n```\n\n### Evaluation feedback:\n{feedback}"
                                    if old_program_content else feedback
                                )
                                _backups = _stash_candidate_files(program_output_dir, program_candidate_id_str)
                                sql_stitching_agent.clear_chat_history()
                                try:
                                    _, oom_occurred = write_solution(
                                        sql_stitching_agent,
                                        sub_sqls_for_plan,
                                        question,
                                        args.db_type,
                                        logger=logger,
                                        program_output_dir=program_output_dir,
                                        program_candidate_id=program_candidate_id_str,
                                        q_id=q_id,
                                        sub_queries_save_path=sub_queries_save_path,
                                        db_path=args.db_path,
                                        similarities_path=args.similarities_path,
                                        plan=current_plan,
                                        external_knowledge_summary=external_knowledge_summary,
                                        evaluation_feedback=full_feedback,
                                        flexible_format_retry=args.flexible_format_retry,
                                        sql_only=args.sql_only,
                                        python_only=args.python_only,
                                    )
                                finally:
                                    _commit_or_rollback_candidate_files(
                                        program_output_dir,
                                        program_candidate_id_str,
                                        _backups,
                                        logger,
                                        log_prefix=f"[self-eval] q_id={q_id} plan_idx={plan_idx} program_candidate_id={program_candidate_id} (impl-issue): ",
                                    )
                                logger.info(f"[self-eval] plan_idx={plan_idx}: write_solution retry complete (with feedback).")
    
    except Exception as e:
        q_id = get_q_id(question, db_type=args.db_type)
        print(f"An unexpected error occurred while processing question {q_id}: {e}")
        traceback.print_exc()
        status = "error"
        error_message = str(e)
    finally:
        for agent in (sub_sqls_agent, sql_stitching_agent, evaluation_agent):
            if agent is not None:
                _merge_usage(question_usage, agent.get_usage())

    result_payload = {
        "status": status,
        "question_id": q_id,
        "usage": question_usage,
    }
    if status == "error":
        result_payload["message"] = error_message
    return result_payload

def main(args, output_dir: str):
    # Load dataset - supports both JSON array (e.g., BIRD dev.json) and JSONL (e.g., spider2-snow)
    with open(args.dataset) as f:
        raw = f.read().strip()
    if raw.startswith('['):
        dataset = json.loads(raw)
    else:
        dataset = [json.loads(line) for line in raw.splitlines() if line.strip()]

    total_usage = _zero_token_usage()

    # Use ProcessPoolExecutor to run tasks concurrently
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_question, question, args, output_dir): question for question in dataset}

        # Process results as they are completed
        for future in as_completed(futures):
            q_id = get_q_id(futures[future], db_type=args.db_type)
            try:
                result = future.result()
                print(f"Completed processing for question {q_id}: {result}")
                if isinstance(result, dict) and "usage" in result:
                    _merge_usage(total_usage, result["usage"])
            except Exception as e:
                print(f"An exception was raised for question {q_id}: {e}")

    print(
        "Aggregate token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}".format(
            **total_usage
        )
    )
    return total_usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="gpt-oss-120b", help="model identifier passed to the OpenAI-compatible client (model name for hosted APIs, or local model directory name for a self-hosted server)")
    parser.add_argument('--dataset', type=str, default="datasets/Spider2/spider2-snow/spider2-snow.jsonl", help="path to the JSON/JSONL file containing the input questions to run inference on (one question per record)")
    parser.add_argument("--db_path", default="datasets/Spider2/spider2-snow/resource/databases_no_nulls_2", help="path to the database folder; for sqlite this is the directory containing per-database subfolders with .sqlite files, for snowflake this is the spider2-snow folder containing snowflake_credential.json and resource/")
    parser.add_argument('--base_url', type=str, default=None, help="full base URL for the OpenAI-compatible API endpoint; takes precedence over --ip/--port when set")
    parser.add_argument('--ip', type=str, default=None, help="hostname or IP of a self-hosted OpenAI-compatible server; combined with --port to construct the base URL when --base_url is not provided")
    parser.add_argument("--workers", type=int, default=2, help="number of worker processes for the ProcessPoolExecutor that runs questions in parallel")
    parser.add_argument('--port', type=str, default=None, help="port of a self-hosted OpenAI-compatible server; combined with --ip to construct the base URL when --base_url is not provided")
    parser.add_argument('--db_type', default="sqlite", choices=["sqlite", "snowflake"], help="database backend; controls credential handling, DDL extraction, and SQL dialect ('sqlite' for local .sqlite files, 'snowflake' for Spider2-Snow)")
    parser.add_argument('--num_programs', type=int, default=1, help="number of independent program candidates to generate per (question, plan) pair; downstream voting/selection picks the final answer across these candidates")
    parser.add_argument('--use_2step_batch_planning', action="store_true", help="run the two-stage planning pipeline: planning_batch_generate produces diverse candidate plans, then plan_clarification grounds each plan against the schema and emits per-plan sub-SQLs")
    parser.add_argument('--use_no_diverse_planning', action="store_true", help="ablation of --use_2step_batch_planning: generate top_k plans sequentially with full tool access from the start (no batch diversity, no separate clarification step). Mutually exclusive with --use_2step_batch_planning")
    parser.add_argument('--planning_top_k', type=int, default=8, help="number of plans to keep per question; for batch planning this is the total across all batches and must be divisible by --planning_batch_size")
    parser.add_argument('--planning_batch_size', type=int, default=4, help="number of plans the model is asked to produce in a single batched call; --planning_top_k must be divisible by this value (number of batched calls = top_k / batch_size)")
    parser.add_argument('--custom_exp_name', type=str, default="", help="custom suffix appended to the auto-generated output directory name; when empty the run is timestamped instead")
    parser.add_argument('--similarities_path', type=str, default=None, help="path to the table_similarities_report JSON file used to group near-duplicate tables in DDL/table overviews and to expand wildcard table patterns in plans")
    parser.add_argument('--use_gold_tables', action="store_true", help="restrict DDL extraction and downstream planning to the gold-labeled tables for each question, skipping schema linking; used for upper-bound evaluation")
    parser.add_argument('--hierarchical-sl', action="store_true", dest="hierarchical_sl", help="enable an extra schema-linking pass that uses list_tables/list_columns tools to filter the schema list down to relevant schemas before DDL extraction; only takes effect when --db_type=snowflake and --use_gold_tables is not set")
    parser.add_argument('--planning_gen_config', type=str, default=None, help='JSON string of OpenAI-compatible generation parameters (temperature, top_p, etc.) applied to the planning agent only, e.g. \'{"temperature": 0.7}\'')
    parser.add_argument('--stitching_gen_config', type=str, default=None, help='JSON string of OpenAI-compatible generation parameters applied to the SQL-stitching agent only, e.g. \'{"temperature": 0.2}\'')
    parser.add_argument('--eval_gen_config', type=str, default=None, help='JSON string of OpenAI-compatible generation parameters applied to the self-evaluation agent only, e.g. \'{"temperature": 0.0}\'')
    parser.add_argument('--max_self_eval_rounds', type=int, default=3, help="maximum number of self-evaluation feedback rounds per program candidate; each round may trigger plan-level or implementation-level repair before giving up")
    parser.add_argument('--flexible_format_retry', action='store_true', help="when retrying after an execution error during stitching, allow the model to switch output format between Python and SQL instead of being locked to whichever format the original attempt used")
    parser.add_argument('--sql_only', action='store_true', help="restrict the stitching step to SQL output only: skip Python interpreter setup and dataframe loading entirely. Mutually exclusive with --python_only")
    parser.add_argument('--python_only', action='store_true', help="restrict the stitching step to Python output only: force the final program to be Python and reject SQL blocks as final answers. Mutually exclusive with --sql_only")
    parser.add_argument('--no-planning-repair', action='store_true', dest='no_planning_repair', help="ablation of self-evaluation repair: the evaluator only emits a correctness verdict + feedback (no plan/implementation issue classification), and only execution-level repair is attempted — plan_clarification is never re-run")

    args = parser.parse_args()

    def _parse_gen_config(raw: Optional[str], name: str) -> dict:
        if raw is None:
            return {}
        try:
            cfg = json.loads(raw)
            if not isinstance(cfg, dict):
                raise ValueError("must be a JSON object")
            return cfg
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Invalid --{name}: {exc}") from exc

    args.planning_gen_config = _parse_gen_config(args.planning_gen_config, "planning_gen_config")
    args.stitching_gen_config = _parse_gen_config(args.stitching_gen_config, "stitching_gen_config")
    args.eval_gen_config = _parse_gen_config(args.eval_gen_config, "eval_gen_config")

    model_name = os.path.normpath(args.model).split("/")[-1]
    
    now = datetime.datetime.now()
    if args.use_2step_batch_planning:
        beam_size_str = "2step-batch-planning"
    elif args.use_no_diverse_planning:
        beam_size_str = "no-diverse-planning"
    else:
        beam_size_str = "no-planning"
    if args.no_planning_repair:
        beam_size_str += "-no-planning-repair"

    # Include microseconds to avoid collisions in parallel jobs
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    sql_only_str = "-sql-only" if args.sql_only else ""
    python_only_str = "-python-only" if args.python_only else ""
    if args.custom_exp_name:
        output_dir = f"inference_res/{model_name}-{args.custom_exp_name}--{args.planning_top_k}-plans-{args.num_programs}-programs-{beam_size_str}{sql_only_str}{python_only_str}/"
    else:
        output_dir = f"inference_res/{model_name}-{time_string}--{args.planning_top_k}-plans-{args.num_programs}-programs-{beam_size_str}{sql_only_str}{python_only_str}/"

    os.makedirs("inference_res/", exist_ok=True)
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs(f"{output_dir}/total_usage", exist_ok=True)

    total_usage = main(args, output_dir)
    all_ids = []
    for q in open(args.dataset):
        q = json.loads(q)
        q_id = get_q_id(q, db_type=args.db_type)
        all_ids.append(q_id)
    total_usage["all_ids"] = all_ids

    usage_summary_path = os.path.join(output_dir, "total_usage", f"aggregate_usage_{time_string}.json")
    with open(usage_summary_path, "w") as usage_file:
        json.dump(total_usage, usage_file, indent=2)
    print(f"Saved aggregate usage to {usage_summary_path}")

