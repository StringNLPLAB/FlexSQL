# -*- coding:utf-8 -*-

import argparse
import csv
import itertools
import json
import logging
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, Manager
from typing import List

import numpy as np
import pandas as pd

from src.chat import Chat
from src.logger import initialize_logger
from src.py2sql.pandas_formatter import pandas_format
from src.utils import divide, compare_pandas_table, now, diff_outputs
from pathlib import Path

DIR = os.path.dirname(__file__)
MAX_RETRIES = 3

DB_PATH = os.path.join(DIR, "../../spider2-lite")
DB_TYPE = "sqlite"
SUBQUERY_TYPE_PATH = os.path.join(DIR, "../../spider2-lite/resource/databases/sqlite")


def py2sql(agent: Chat, qid: str, question: dict, plan: str,
           py_file: str, program: str, py_out_file: str, sql_file: str, sql_out_file: str,
           subqueries: dict, subqueries_schema: dict = None,
           logger: logging.Logger = None, db_type: str = "sqlite", db_path: str = None, **kwargs) -> bool:
    # Python program -> SQL query within X iterations and repair x time for each iteration
    prompt_template = """
### Your Task:
{task}

Please format your SQL query in a JSON object like
```json
{{"sql": "<SQL query>"}}
```

### user question:
{question}

### plan:
{plan}

### Python program:
{code}

### atomic SQL sub-queries:
```json
{subqueries}
```

### schemas of atomic SQL sub-queries:
```json
{subqueries_schema}
```

### Your Query:
    """.strip()
    task = """
You are given:
* A user question
* A plan that partitions the user question into smaller sub-questions
* A Python program that exactly implements the plan
* A set of atomic SQL sub-queries, each representing a base table
* The schemas of atomic SQL sub-queries, each representing column names and data types of a base table
The Python program operates on the atomic SQL sub-queries as pandas DataFrames.

Your task is to translate the Python program into a SQL query. The SQL query must satisfy:
1. It uses the provided atomic sub-queries as common table expressions
2. It strictly follows the semantics of the Python program; if there is any discrepancy between the Python program and the plan, the Python program takes precedence.
""".strip()
    code = """
```python
{program}
```
""".strip().format(program=program)

    def translate_program(prompt, sql_file, sql_out_file):
        try:
            # Get response from model - generates all plans at once
            response = agent.get_code_blocks(
                prompt,
                code_format="json",
                logger=logger,
                example_json_structure={"sql": "WITH ..."},
            )
            logger.info(f"<thoughts>\n{response['thoughts']}\n</thoughts>")

            query = json.loads(response['code_blocks'][0])['sql']

            for turn in range(MAX_RETRIES):
                with open(sql_file, 'w') as writer:
                    writer.write(query)

                command = [sys.executable, "utils/sql_frame.py", "sqlite", kwargs['credential'], qid, sql_file, sql_out_file]

                try:
                    # Using the program file with subprocess is more robust
                    timeout = 3000
                    logger.info(command)
                    subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)  # too short timeout
                except subprocess.CalledProcessError as e:
                    error_lines = e.stderr.splitlines()
                    relevant_lines = []

                    # Always add the final exception message (the last line)
                    try:
                        relevant_lines.append(error_lines[-1])
                    except IndexError:
                        relevant_lines.append(f"The SQL query timed out after {timeout} seconds!")
                    llm_trace_str = "\n".join(relevant_lines)

                    if len(llm_trace_str) == 0:
                        logger.info("[NOTICE]Somehow the error trace doesn't contain the LLMs script error")
                        llm_trace_str = str(e.stderr)

                    err_message = f"The model got the following error:\n\n{llm_trace_str}\n\nPlease revise the SQL query. Remember to put the revised query within a ```json``` block"

                    try:
                        response = agent.get_code_blocks(err_message, code_format="json", logger=logger)
                        query = json.loads(response['code_blocks'][0])['sql']
                        logger.info(f"<error>\n{err_message}\n\nRevised SQL query:\n{query}\n</error>")
                        continue
                    except Exception as retry_error:
                        logger.error(f"Failed to get revised SQL query: {retry_error}")
                except subprocess.TimeoutExpired:
                    logger.error(f"Process for question {qid} timed out. Path: {sql_file}")
                    break

        except Exception as e:
            logger.error(f"Error generating batch plans: {e}", exc_info=True)

    def gen_sql_query(sql_file, sql_out_file):
        os.makedirs(os.path.dirname(sql_file), exist_ok=True)
        agent.clear_chat_history()
        agent.set_system_prompt(
            f"You are a helpful assistant. Given a user question and a plan, translate a Python program in {db_type} based on a set of atomic SQL sub-queries."
            f"Format your {db_type} query in a JSON object with a single field \"sql\"."
            f"CRITICAL: Use the provided atomic sub-queries as common table expressions."
        )

        # Build prompt
        prompt = prompt_template.format(
            task=task,
            question=question['question'],
            plan=plan,
            code=code,
            subqueries=str(subqueries),
            subqueries_schema=str(subqueries_schema),
        )
        logger.info(f"<prompt>\n{prompt}\n</prompt>")
        # Define example structure for validation
        translate_program(prompt, sql_file, sql_out_file)

    def repair_sql(code, py_diff, sql_file, sql_out_file, incorrect_sql, sql_diff):
        agent.clear_chat_history()
        agent.set_system_prompt(
            f"You are a helpful assistant. Given a user question and a plan, correct a {db_type} query translated from a Python program based on their inconsistent outputs."
            f"Format your {db_type} query in a JSON object with a single field \"sql\"."
            f"CRITICAL: Correct the query based on the Python program and their outputs."
        )

        atomic_subqueries = subqueries
        repair_prompt = f"""
### Your Task:
You are given:
* A user question
* A plan that partitions the user question into smaller sub-questions
* A Python program that exactly implements the plan
* An incorrect SQL query that was translated from the Python program but produces outputs inconsistent with the program
* A Python program's unique output that can be produced by the Python program but cannot be produced by the incorrect SQL query
* A SQL query's unique output that can be produced by the incorrect SQL query but cannot be produced by the Python program
The SQL query must have different outputs from the Python program.

Your task is to correct the SQL query and make it consistent with the Python program. The corrected SQL query must satisfy:
1. It produces the Python program’s unique output.
2. It never produces the SQL query’s unique (incorrect) output.
3. It strictly follows the semantics of the Python program; if there is any discrepancy between the Python program and the plan, the Python program takes precedence.

Do NOT revise, rewrite, or remove the sub-queries in the common table expressions (CTEs) listed below.
These CTEs are treated as fixed base tables for translation and must remain unchanged.
atomic SQL sub-queries:
```json
{atomic_subqueries}
```
schemas of atomic SQL sub-queries:
```json
{subqueries_schema}
```

Please format your SQL query in a JSON object like
```json
{{"sql": "<SQL query>"}}
```

### user question:
{question["question"]}

### plan:
{plan}

### Python program:
{code}

### Python program's unique output (Expected rows produced by the Python program but not produced by the incorrect SQL query):
```csv
{py_diff}
```

### incorrect SQL query:
```sql
{incorrect_sql}
```

### SQL query's unique output (Incorrect rows produced by the SQL query but not produced by the Python program):
```csv
{sql_diff}
```

### Your corrected Query:
""".strip()
        logger.info(f"<prompt>\n{repair_prompt}\n</prompt>")
        # Define example structure for validation
        translate_program(repair_prompt, sql_file, sql_out_file)

    py_out = pd.read_csv(py_out_file)
    done_qids = kwargs.get('done_qids')

    def _signal_done():
        if done_qids is not None:
            done_qids[qid] = True

    for trans_idx in range(kwargs['start_iter'], kwargs['end_iter']):
        if done_qids is not None and done_qids.get(qid, False):
            logger.info(f"[Aborting {qid} program {kwargs.get('plan_id')}_{kwargs.get('impl_id')}: another program for this qid already found a consistent translation]")
            return False
        logger.info(f"{now()} | Translation: {trans_idx + 1}/{args.num_translations} | Transpiled: {os.path.realpath(sql_file)}")
        sql_file_idx, sql_out_file_idx = sql_file[:-3] + f"{trans_idx}.sql", sql_out_file[:-3] + f"{trans_idx}.csv"

        logger.info(f"[Translation {trans_idx + 1}/{args.num_translations} begin]")
        gen_sql_query(sql_file_idx, sql_out_file_idx)

        if not os.path.exists(sql_out_file_idx):
            logger.info(f"[Translation {trans_idx + 1}/{args.num_translations} failed: cannot find a syntactically correct SQL query and turn to the next translation]")
            continue

        sql_out = pd.read_csv(sql_out_file_idx)
        consistent = compare_pandas_table(sql_out, py_out, ignore_order=True)
        if consistent:
            logger.info(f"[Translation {trans_idx + 1}/{args.num_translations} succeeded: find a consistent SQL query w/o repair]")
            _signal_done()
            return True
        else:
            logger.info(f"[Translation {trans_idx + 1}/{args.num_translations} failed: find an inconsistent SQL query]")

            with open(sql_file_idx, 'r') as reader:
                incorrect_sql = reader.read()

            for repair_idx in range(1, 1 + args.num_repairs):
                repaired_sql_file_idx, repaired_sql_out_file_idx = sql_file[:-3] + f"{trans_idx}.repair.{repair_idx}.sql", sql_out_file[:-3] + f"{trans_idx}.repair.{repair_idx}.csv"
                sql_diff, py_diff = diff_outputs(sql_out, py_out)
                logger.info(f"[Translation {trans_idx + 1}/{args.num_translations} + Repair {repair_idx}/{args.num_repairs}]")
                repair_sql(code, py_diff, repaired_sql_file_idx, repaired_sql_out_file_idx, incorrect_sql, sql_diff)

                if not os.path.exists(repaired_sql_out_file_idx):
                    logger.info(
                        f"[Translation {trans_idx + 1}/{args.num_translations} + Repair {repair_idx}/{args.num_repairs} failed: cannot find a syntactically correct SQL query and turn to the next repair]")
                    continue

                sql_out = pd.read_csv(repaired_sql_out_file_idx)
                consistent = compare_pandas_table(sql_out, py_out, ignore_order=True)
                if consistent:
                    logger.info(f"[Translation {trans_idx + 1}/{args.num_translations} + Repair {repair_idx}/{args.num_repairs} succeeded: find a consistent SQL query w repair]")
                    _signal_done()
                    return True
                else:
                    logger.info(
                        f"[Translation {trans_idx + 1}/{args.num_translations} + Repair {repair_idx}/{args.num_repairs} failed: find an inconsistent SQL query after repair and turn to the next translation]")
    return False


def _process_translate_task(params, args, done_qids):
    """Worker: translate one (qid, program) task with up to args.num_translations attempts.

    Aborts early if `done_qids[qid]` is set (another program for this qid has already
    found a consistent translation). On success, sets `done_qids[qid] = True` so siblings
    of the same qid can exit at their next loop iteration.
    """
    log_path = os.path.join(
        args.output_dir, "logs",
        f"{params['qid']}.plan{params['plan_id']}_{params['impl_id']}.log",
    )
    logger = initialize_logger(log_path=log_path)

    if done_qids.get(params['qid'], False):
        logger.info(f"Skipping {params['qid']} program {params['plan_id']}_{params['impl_id']}: another program already found a consistent translation")
        return {"qid": params['qid'], "plan_id": params['plan_id'], "impl_id": params['impl_id'], "success": False, "skipped": True}

    agent = Chat(model=args.model, base_url=args.base_url, ip=args.ip, port=args.port)
    logger.info(f"{now()} | Qid: {params['qid']} | Python: {os.path.basename(params['py_file'])}")
    try:
        success = py2sql(agent, **params, db_path=DB_PATH, logger=logger, done_qids=done_qids)
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        success = False

    return {"qid": params['qid'], "plan_id": params['plan_id'], "impl_id": params['impl_id'], "success": success, "skipped": False}


def run_translate(args, parameters: List):
    with Manager() as manager:
        done_qids = manager.dict()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_translate_task, params, args, done_qids): params for params in parameters}
            for future in as_completed(futures):
                params = futures[future]
                try:
                    result = future.result()
                    status = "skipped" if result.get('skipped') else ("succeeded" if result['success'] else "failed")
                    print(f"{now()} | qid={result['qid']} program={result['plan_id']}_{result['impl_id']} {status}")
                except Exception as e:
                    print(f"{now()} | Exception qid={params['qid']} program={params['plan_id']}_{params['impl_id']}: {e}")
                    traceback.print_exc()


def cmp_outs(parameters: List, args):
    results = []
    for params in parameters:
        res = {'qid': params['qid'], 'plan_id': params.get('plan_id'), 'impl_id': params.get('impl_id')}
        use_repair = "repair" in os.path.basename(params['sql_file'])

        basename = os.path.basename(params['sql_file'])
        basename = basename[basename.find('.') + 1:]
        num_iteration = int(basename[:basename.find('.')])
        if use_repair:
            num_repair = int(basename[basename.find("repair") + 7:-4])

        # num_iteration is the trans_idx parsed from the filename (0-indexed: 0..num_translations-1).
        # Convert to a 1-indexed count when reporting "number of translations attempted".
        if os.path.exists(params['sql_out_file']):
            py_out = pd.read_csv(params['py_out_file'])
            sql_out = pd.read_csv(params['sql_out_file'])
            eval_res = compare_pandas_table(sql_out, py_out, ignore_order=True)
            if use_repair:  # consistency reached on this attempt's num_repair-th repair turn
                num_repair = num_iteration * args.num_repairs + num_repair
                res['wo_repair'] = 0
                res['w_repair'] = eval_res
            else:  # consistency reached on this attempt's fresh translation (no repair this round)
                num_repair = num_iteration * args.num_repairs
                res['wo_repair'] = eval_res
                res['w_repair'] = 0
        else:
            # no SQL output: cannot find a correct translation
            res['wo_repair'] = res['w_repair'] = 0
            if use_repair:
                num_repair = num_iteration * args.num_repairs
            else:
                num_repair = (num_iteration + 1) * args.num_repairs

        # consistent results
        res['consistent'] = res['wo_repair'] or res['w_repair']
        res["translations"] = num_iteration + 1
        res["repairs"] = num_repair

        results.append(res)
    return results


def run_eval(args, parameters: List):
    with Pool(args.workers) as mpool:
        parameters = list(divide(parameters, args.workers))
        results = [
            mpool.apply_async(cmp_outs, args=(parameters[worker_idx], args,))
            for worker_idx in range(len(parameters))
        ]
        results = [res.get() for res in results]
        results = list(itertools.chain(*results))

    # Aggregate per-qid: a qid is consistent if any of its programs is consistent.
    # wo_repair takes precedence over w_repair to keep them mutually exclusive at the qid level.
    per_qid_progs = {}
    for res in results:
        per_qid_progs.setdefault(res['qid'], []).append(res)

    qid_results = []
    for qid, progs in per_qid_progs.items():
        wo = 1 if any(p['wo_repair'] for p in progs) else 0
        wr = 0 if wo else (1 if any(p['w_repair'] for p in progs) else 0)
        qid_results.append({
            'qid': qid,
            'wo_repair': wo,
            'w_repair': wr,
            'consistent': wo or wr,
            'translations': max(p['translations'] for p in progs),
            'repairs': max(p['repairs'] for p in progs),
        })

    if not qid_results:
        print("No questions to evaluate (no translation outputs found).")
        return

    print(f"#Questions: {len(qid_results)}")
    print(f"#Consistent results: {sum(r['consistent'] for r in qid_results)}/{len(qid_results)}")
    print(f"#Consistent results w/o repair: {sum(r['wo_repair'] for r in qid_results)}/{len(qid_results)}")
    print(f"#Consistent results w repair: {sum(r['w_repair'] for r in qid_results)}/{len(qid_results)}")
    print(f"Failure questions: {[r['qid'] for r in qid_results if r['consistent'] == 0]}")

    num_translations = [r['translations'] for r in qid_results]
    print(f"#Translations: Min={min(num_translations)}, Max={max(num_translations)}, Avg={np.mean(num_translations):.2f}, Median={np.median(num_translations):.2f}")
    num_repairs = [r['repairs'] for r in qid_results]
    print(f"#Repairs: Min={min(num_repairs)}, Max={max(num_repairs)}, Avg={np.mean(num_repairs):.2f}, Median={np.median(num_repairs):.2f}")

    # for res in results:
    #     print(res['qid'], res["translations"], res['repairs'])


def main(args):
    print(f"-------------- Python programs => SQL queries --------------")
    print(args)

    def read_translate_tasks(args):
        # read questions
        question_file = os.path.join(DB_PATH, "spider2-sqlite.jsonl")
        nl_questions = {}
        qid2db = {}
        with open(question_file, 'r') as reader:
            for line in reader:
                line = json.loads(line)
                nl_questions[line["instance_id"]] = line
                qid2db[line["instance_id"]] = line["db"]

        # read correct python program that can produce the same results as the gold queries
        # gold_exec_path = os.path.join(args.db_path, "datasets/Spider2/spider2-snow/evaluation_suite/gold/exec_result")
        tasks = []

        # test
        mv_qids_file = os.path.join(args.input_experiment_dir, "major_vote.csv")
        with open(mv_qids_file, 'r') as reader:
            reader = csv.reader(reader)
            next(reader)
            mv_qids = set([qid[0] for qid in reader])

        # qid -> a group of Python/SQL's outputs s.t. the group has the most major voting
        # (1) all outputs in this group are the same, and (2)
        correct_impls = {}
        major_voting_file = os.path.join(args.input_experiment_dir, "major_vote", "majority_vote_summary.json")
        with open(major_voting_file, 'r') as reader:
            groups = json.load(reader)
            for qid, group_info in groups.items():

                if qid not in mv_qids or 'groups' not in group_info: continue  # only consider the questions that have major voting
                idx2num = {int(idx): len(candidates) for idx, candidates in group_info['groups'].items()}
                max_idx, max_num = -1, 0
                for idx, num in sorted(idx2num.items(), key=lambda x: x[0]):
                    if num > max_num:
                        max_idx, max_num = idx, num
                output_files = group_info['groups'][str(max_idx)]

                # if this question has any SQL query, then this question has been resolved
                # otherwise, we translate its python programs in SQL

                # read plan
                plan_file = os.path.join(args.input_experiment_dir, qid, "plans.json")
                with open(plan_file, 'r') as reader:
                    plans = []
                    subqueries = []
                    for plan in json.load(reader):
                        plans.append('\n'.join(plan[0]))
                        if isinstance(plan[1], dict):
                            subqueries.append(plan[1])
                        else:
                            subqueries.append(None)  # ignore python programs that have wrong subquery info

                gen_sql_files = []  # SQL queries generated from frontend LLMs
                sql_files, sql_out_files, py_files, py_out_files, qid_plans, qid_subqueries, qid_subquery_schemas = [], [], [], [], [], [], []
                credential_file = None
                for ofile in output_files:
                    plan_id, impl_id = map(int, ofile[len("program_output_"):-len(".csv")].split("_"))
                    sql_file = os.path.join(args.input_experiment_dir, qid, f"program_{plan_id}_{impl_id}.sql")
                    if os.path.exists(sql_file):
                        gen_sql_files.append(sql_file)
                    py_file = os.path.join(args.input_experiment_dir, qid, f"program_{plan_id}_{impl_id}.py")
                    if os.path.exists(py_file) and subqueries[plan_id] is not None:
                        # translated SQL query file
                        sql_files.append(os.path.join(args.output_dir, qid, f"query_{plan_id}_{impl_id}.sql"))
                        # translated SQL query's output file
                        sql_out_files.append(os.path.join(args.output_dir, qid, f"query_output_{plan_id}_{impl_id}.csv"))
                        py_files.append(py_file)
                        py_out_files.append(os.path.join(args.input_experiment_dir, qid, ofile))
                        qid_plans.append(plans[plan_id])

                        qid_subqueries.append({db: f"SELECT {', '.join(columns)} FROM {db}" for db, columns in subqueries[plan_id].items()})
                        subqueries_schema = {}
                        for db, columns in subqueries[plan_id].items():
                            subquery_type_file = os.path.join(SUBQUERY_TYPE_PATH, qid2db[qid], f'{db}.json')
                            with open(subquery_type_file, 'r') as datatype_reader:
                                datatype = json.load(datatype_reader)
                                datatype = {str.lower(col): type for col, type in zip(datatype['column_names'], datatype['column_types'])}
                            subqueries_schema[db] = {str.lower(col): datatype[str.lower(col)] for col in columns}
                        qid_subquery_schemas.append(subqueries_schema)
                        credential_file = os.path.join(SUBQUERY_TYPE_PATH, qid2db[qid], f'{qid2db[qid]}.sqlite')

                if len(gen_sql_files) == 0:
                    # we have no SQL queries, do translation for their python programs
                    correct_impls[qid] = py_files

                    if len(py_files) == 0:
                        print(f"{qid} has no SQL queries or Python programs")
                        continue

                    qid_tasks = []
                    for sql_file, sql_out_file, py_file, py_out_file, plan, subqueries, subquery_schemas in \
                            zip(sql_files, sql_out_files, py_files, py_out_files, qid_plans, qid_subqueries, qid_subquery_schemas):
                        # read program
                        with open(py_file, 'r') as reader:
                            program = reader.read().strip()
                            program = pandas_format(program)

                        assert len(subqueries) == len(subquery_schemas), (qid, len(subqueries), len(subquery_schemas))

                        # py_file is .../program_{plan_id}_{impl_id}.py
                        _, plan_id_str, impl_id_str = os.path.basename(py_file)[:-3].split('_')
                        plan_id, impl_id = int(plan_id_str), int(impl_id_str)

                        qid_tasks.append({
                            "qid": qid, "question": nl_questions[qid], "plan": plan,
                            "sql_file": sql_file, "sql_out_file": sql_out_file,
                            "py_file": py_file, "program": program, "py_out_file": py_out_file,
                            "subqueries": subqueries, "subqueries_schema": subquery_schemas, "credential": credential_file,
                            "plan_id": plan_id, "impl_id": impl_id,
                            "start_iter": 0, "end_iter": args.num_translations,
                        })
                    tasks.append(qid_tasks)

        tasks = list(itertools.chain(*tasks))
        print(f"Total {len(tasks)}/{len(correct_impls)} programs")
        return tasks

    def read_eval_tasks(args):
        tasks = []

        mv_qids_file = os.path.join(args.input_experiment_dir, "major_vote.csv")
        with open(mv_qids_file, 'r') as reader:
            reader = csv.reader(reader)
            next(reader)
            mv_qids = set([qid[0] for qid in reader])

        # qid -> a group of Python/SQL's outputs s.t. the group has the most major voting
        # (1) all outputs in this group are the same, and (2)
        correct_impls = {}
        major_voting_file = os.path.join(args.input_experiment_dir, "major_vote", "majority_vote_summary.json")
        with open(major_voting_file, 'r') as reader:
            groups = json.load(reader)
            for qid, group_info in groups.items():
                if qid not in mv_qids or 'groups' not in group_info: continue
                idx2num = {int(idx): len(candidates) for idx, candidates in group_info['groups'].items()}
                max_idx, max_num = -1, 0
                for idx, num in sorted(idx2num.items(), key=lambda x: x[0]):
                    if num > max_num:
                        max_idx, max_num = idx, num
                output_files = group_info['groups'][str(max_idx)]

                gen_sql_files = []
                py_files, py_out_files, plan_ids, impl_ids = [], [], [], []
                for ofile in output_files:
                    plan_id, impl_id = map(int, ofile[len("program_output_"):-len(".csv")].split("_"))
                    sql_file = os.path.join(args.input_experiment_dir, qid, f"program_{plan_id}_{impl_id}.sql")
                    if os.path.exists(sql_file):
                        gen_sql_files.append(sql_file)
                    py_file = os.path.join(args.input_experiment_dir, qid, f"program_{plan_id}_{impl_id}.py")
                    if os.path.exists(py_file):
                        py_files.append(py_file)
                        py_out_files.append(os.path.join(args.input_experiment_dir, qid, ofile))
                        plan_ids.append(plan_id)
                        impl_ids.append(impl_id)

                if len(gen_sql_files) == 0:
                    correct_impls[qid] = py_files

                    if len(py_files) == 0:
                        print(f"{qid} has no SQL queries or Python programs")
                        continue

                    out_path = os.path.join(args.output_dir, qid)
                    qid_has_files = False
                    # Build one task per program (plan_id, impl_id); pick the latest existing
                    # file for that program (highest trans_idx, then highest repair_idx).
                    for plan_id, impl_id, py_out_file in zip(plan_ids, impl_ids, py_out_files):
                        program_gen_sql_file = None
                        for i in range(args.num_translations):
                            file_i = os.path.join(out_path, f"query_{plan_id}_{impl_id}.{i}.sql")
                            if os.path.exists(file_i):
                                program_gen_sql_file = file_i
                                for j in range(1, 1 + args.num_repairs):
                                    file_i_repair_j = os.path.join(out_path, f"query_{plan_id}_{impl_id}.{i}.repair.{j}.sql")
                                    if os.path.exists(file_i_repair_j):
                                        program_gen_sql_file = file_i_repair_j
                        if program_gen_sql_file is not None:
                            qid_has_files = True
                            d, b = os.path.dirname(program_gen_sql_file), os.path.basename(program_gen_sql_file)
                            program_gen_sql_out_file = os.path.join(d, "query_output" + b[5:-3] + "csv")
                            tasks.append({
                                "qid": qid, "plan_id": plan_id, "impl_id": impl_id,
                                "sql_file": program_gen_sql_file, "sql_out_file": program_gen_sql_out_file,
                                "py_out_file": py_out_file,
                            })

                    if not qid_has_files:
                        print(f"No translation found for {qid}")
        return tasks

    if args.mode == "translate":
        tasks = read_translate_tasks(args)
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
        run_translate(args, tasks)
    elif args.mode == "eval":
        tasks = read_eval_tasks(args)
        run_eval(args, tasks)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # LLM setup
    parser.add_argument('--model', type=str, default="gpt-oss-120b", help="path to the model dir")
    parser.add_argument('--ip', type=str, default="localhost", help="Optional IP for a self-hosted OpenAI-compatible server")
    parser.add_argument('--port', type=str, default=None, help="Optional port for a self-hosted OpenAI-compatible server")
    parser.add_argument('--base_url', type=str, default=None, help="Override OpenAI-compatible server base URL")

    parser.add_argument('--time', type=str, default=now(), help="Time to run experiments")

    # (1) translate: translate Python programs into SQL queries
    # (2) eval: compute consistency between programs' outputs and SQL queries' outputs
    parser.add_argument('--mode', type=str, choices=['translate', 'eval'], default='translate')
    parser.add_argument('--input_experiment_dir', type=str, help="the experiment directory output by main.py to run translation on")

    parser.add_argument('--num_translations', type=int, default=64, help="number of generation iteration")
    parser.add_argument('--num_repairs', type=int, default=3, help="number of repair iteration")

    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (used in both --mode translate and --mode eval)")
    args = parser.parse_args()

    args.output_dir = os.path.join(DIR, f"../../py2sql/{DB_TYPE}/{Path(args.input_experiment_dir).name}", args.time)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
