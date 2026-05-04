import json
import re
import pandas as pd
import math
from typing import List, Union
import os
import os.path as osp
import argparse
# from google.cloud import bigquery
import shutil
import sqlite3
from tqdm import tqdm
import snowflake.connector
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from threading import Lock
import multiprocessing
import signal
import time
from functools import lru_cache
import subprocess

import sys
class TeeOutput:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
        self.lock = Lock()
    
    def write(self, message):
        with self.lock:
            self.console.write(message)
            self.file.write(message)
    
    def flush(self):
        with self.lock:
            self.console.flush()
            self.file.flush()
    
    def close(self):
        self.file.close()

sys.stdout = TeeOutput('log.txt')
sys.stderr = sys.stdout

TOTAL_GB_PROCESSED = 0.0
GB_LOCK = Lock()  

byte_output_dict = {}

@lru_cache(maxsize=None)
def load_gold_csv(file_path: str) -> pd.DataFrame:
    """Cache gold CSV loads to avoid repeated disk reads during evaluation."""
    return pd.read_csv(file_path)

def load_jsonl_to_dict(jsonl_file):
    data_dict = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            instance_id = item['instance_id']
            data_dict[instance_id] = item
    return data_dict

def per_gold_scores_multi(pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False):
    """Return a list of 0/1 scores, one per gold variant, without early exit."""
    if multi_condition_cols == [] or multi_condition_cols == [[]] or multi_condition_cols == [None] or multi_condition_cols == None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]
    if len(multi_condition_cols) < len(multi_gold):
        multi_condition_cols = list(multi_condition_cols) + [[] for _ in range(len(multi_gold) - len(multi_condition_cols))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]
    return [
        int(compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]))
        for i, gold in enumerate(multi_gold)
    ]


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    # print('condition_cols', condition_cols)
    # Quick rejection: different row counts cannot be equivalent.
    if len(pred) != len(gold):
        return 0

    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    
    if condition_cols != []:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for gold in t_gold_list:
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0

    return score



def get_snowflake_sql_result(sql_query, database_id, is_save, save_dir=None, file_name="result.csv", timeout=30, instance_id=None, credential_path=None):
    if credential_path is None:
        # Try CWD first, then fall back to <project_root>/datasets/Spider2/spider2-snow/snowflake_credential.json.
        credential_path = 'snowflake_credential.json'
        if not os.path.exists(credential_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            credential_path = os.path.join(project_root, 'datasets', 'Spider2', 'spider2-snow', 'snowflake_credential.json')
            credential_path = os.path.abspath(credential_path)
    credential_path = os.path.abspath(credential_path)
    snowflake_credential = json.load(open(credential_path))
    connection_kwargs = {k: v for k, v in snowflake_credential.items() if k != "session_parameters"}
    session_parameters = snowflake_credential.get("session_parameters", {}).copy()
    session_parameters["STATEMENT_TIMEOUT_IN_SECONDS"] = timeout
    connection_kwargs["session_parameters"] = session_parameters

    conn = snowflake.connector.connect(
        database=database_id,
        **connection_kwargs
    )
    cursor = conn.cursor()
    
    prefix = f"[{instance_id}] " if instance_id else ""

    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        if df.empty:
            message = "No data found for the specified query."
            print(f"{prefix}{message}")
            return False, message
        else:
            if is_save:
                df.to_csv(os.path.join(save_dir, file_name), index=False)
                return True, None
    except snowflake.connector.errors.ProgrammingError as e:
        error_message = str(e)
        if "STATEMENT_TIMEOUT" in error_message or "SQL execution canceled" in error_message:
            timeout_msg = f"Query execution timed out after {timeout} seconds"
            print(f"{prefix}{timeout_msg}")
            return False, timeout_msg
        print(f"{prefix}Error occurred while fetching data: {error_message}")
        return False, error_message
    except Exception as e:
        error_message = str(e)
        print(f"{prefix}Error occurred while fetching data: {error_message}")
        return False, error_message
    finally:
        cursor.close()
        conn.close()


def extract_sql_query(pred_sql_query):
    pattern = r'```sql\n(.*?)\n```'
    match = re.search(pattern, pred_sql_query, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return pred_sql_query



def evaluate_single_sql_instance(pred_id, base_id, eval_standard_dict, spider2sql_metadata, pred_result_dir, gold_sql_dir, gold_result_dir, temp_dir, result_csv_dir=None, timeout=30, credential_path=None):
    error_info = None
    
    try:
        pred_sql_query = open(os.path.join(pred_result_dir, f"{pred_id}.sql")).read()
        pred_sql_query = extract_sql_query(pred_sql_query)
        
        process_temp_dir = os.path.join(temp_dir, f"process_{os.getpid()}")
        os.makedirs(process_temp_dir, exist_ok=True)
        
        if base_id.startswith("sf"):
            database_id = spider2sql_metadata[base_id]['db_id']
            exe_flag, dbms_error_info = get_snowflake_sql_result(
                pred_sql_query,
                database_id,
                True,
                process_temp_dir,
                f"{pred_id}.csv",
                timeout=timeout,
                instance_id=pred_id,
                credential_path=credential_path,
            )  
            if exe_flag == False:
                score = 0
                per_gold_scores = None
                error_info = dbms_error_info
            else:
                pred_pd = pd.read_csv(os.path.join(process_temp_dir, f"{pred_id}.csv"))

                if result_csv_dir:
                    shutil.copy2(os.path.join(process_temp_dir, f"{pred_id}.csv"),
                               os.path.join(result_csv_dir, f"{pred_id}.csv"))

                if '_' in base_id:
                    pattern = re.compile(rf'^{re.escape(base_id)}(_[a-z])?\.csv$')
                else:
                    pattern = re.compile(rf'^{re.escape(base_id)}(_[a-z])?\.csv$')

                all_files = os.listdir(gold_result_dir)
                csv_files = [file for file in all_files if pattern.match(file)]
                csv_files = sorted(csv_files)
                base_file_path = os.path.join(gold_result_dir, f"{base_id}.csv")
                per_gold_scores = None
                if os.path.exists(base_file_path):
                    try:
                        gold_pd = load_gold_csv(base_file_path)
                        score = compare_pandas_table(
                            pred_pd,
                            gold_pd,
                            eval_standard_dict.get(base_id)['condition_cols'],
                            eval_standard_dict.get(base_id)['ignore_order'],
                        )
                        per_gold_scores = [score]
                    except Exception as e:
                        print(f"{base_id}: compare against {base_file_path} failed: {e}")
                        score = 0
                        per_gold_scores = [0]
                        error_info = 'Python Script Error:' + str(e)
                    if score == 0 and error_info is None:
                        error_info = 'Result Error'
                elif csv_files:
                    try:
                        csv_file_paths = [os.path.join(gold_result_dir, file) for file in csv_files]
                        gold_pds = [load_gold_csv(file_path) for file_path in csv_file_paths]
                        per_gold_scores = per_gold_scores_multi(
                            pred_pd,
                            gold_pds,
                            eval_standard_dict.get(base_id)['condition_cols'],
                            eval_standard_dict.get(base_id)['ignore_order'],
                        )
                        score = int(any(per_gold_scores))
                    except Exception as e:
                        print(f"{base_id}: multi-compare against {csv_file_paths} failed: {e}")
                        score = 0
                        per_gold_scores = [0] * len(csv_files)
                        error_info = 'Python Script Error:' + str(e)
                    if score == 0 and error_info is None:
                        error_info = 'Result Error'
                else:
                    score = 0
                    per_gold_scores = []
                    if error_info is None:
                        error_info = 'No matching gold file found'

    except Exception as e:
        print(f"Error evaluating {pred_id}: {e}")
        score = 0
        per_gold_scores = None
        error_info = f"Evaluation Error: {str(e)}"
        pred_sql_query = ""

    return {
        "instance_id": pred_id,
        "score": score,
        "pred_sql": pred_sql_query,
        "error_info": error_info,
        "per_gold_scores": per_gold_scores,
    }


def evaluate_single_exec_result_instance(pred_id, base_id, eval_standard_dict, pred_result_dir, gold_result_dir):
    error_info = None
    
    try:
        pred_pd = pd.read_csv(os.path.join(pred_result_dir, f"{pred_id}.csv"))
        
        if '_' in base_id:
            pattern = re.compile(rf'^{re.escape(base_id)}(_[a-z])?\.csv$')
        else:
            pattern = re.compile(rf'^{re.escape(base_id)}(_[a-z])?\.csv$')
            
        all_files = os.listdir(gold_result_dir)
        csv_files = [file for file in all_files if pattern.match(file)]
        csv_files = sorted(csv_files)
        base_file_path = os.path.join(gold_result_dir, f"{base_id}.csv")
        per_gold_scores = None
        if os.path.exists(base_file_path):
            try:
                gold_pd = load_gold_csv(base_file_path)
                score = compare_pandas_table(
                    pred_pd,
                    gold_pd,
                    eval_standard_dict.get(base_id)['condition_cols'],
                    eval_standard_dict.get(base_id)['ignore_order'],
                )
                per_gold_scores = [score]
            except Exception as e:
                print(f"{base_id}: compare against {base_file_path} failed: {e}")
                score = 0
                per_gold_scores = [0]
                error_info = 'Python Script Error:' + str(e)
            if score == 0 and error_info is None:
                error_info = 'Result Error'

        elif csv_files:
            try:
                csv_file_paths = [os.path.join(gold_result_dir, file) for file in csv_files]
                gold_pds = [load_gold_csv(file_path) for file_path in csv_file_paths]
                per_gold_scores = per_gold_scores_multi(
                    pred_pd,
                    gold_pds,
                    eval_standard_dict.get(base_id)['condition_cols'],
                    eval_standard_dict.get(base_id)['ignore_order'],
                )
                score = int(any(per_gold_scores))
            except Exception as e:
                print(f"{base_id}: multi-compare against {csv_file_paths} failed: {e}")
                score = 0
                per_gold_scores = [0] * len(csv_files)
                error_info = 'Python Script Error:' + str(e)
            if score == 0 and error_info is None:
                error_info = 'Result Error'
        else:
            score = 0
            per_gold_scores = []
            if error_info is None:
                error_info = 'No matching gold file found'

    except Exception as e:
        print(f"{pred_id} ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {e}")
        score = 0
        per_gold_scores = None
        error_info = f"Evaluation Error: {str(e)}"

    return {
        "instance_id": pred_id,
        "score": score,
        "pred_sql": None,
        "error_info": error_info,
        "per_gold_scores": per_gold_scores,
    }


def save_correct_ids_to_csv(output_results, result_dir):
    correct_ids = [item['instance_id'] for item in output_results if item['score'] == 1]
    
    df = pd.DataFrame({'output': correct_ids})
    
    parent_dir = os.path.dirname(result_dir)
    result_dir_name = os.path.basename(result_dir)
    csv_file_path = os.path.join(parent_dir, f"{result_dir_name}.csv")
    
    df.to_csv(csv_file_path, index=False)
    print(f"Correct IDs saved to: {csv_file_path}")
    
    return csv_file_path


def evaluate_group(base_id, prediction_ids, mode, eval_standard_dict, spider2sql_metadata, 
                   pred_result_dir, gold_sql_dir, gold_result_dir, temp_dir, result_csv_dir, timeout, credential_path=None):
    """
    Evaluate a group of predictions for a single base_id (pass@k logic).
    This function is designed to be picklable for multiprocessing.
    """
    errors = []
    correct_program_names = []
    last_pred_sql = None
    per_pred_scores = {}  # pred_id -> 0 or 1
    # For micro accuracy: element-wise OR of per_gold_scores across all predictions
    micro_gold_covered = None  # list[int], one entry per gold variant

    # Iterate through each prediction for the current base_id
    for pred_id in prediction_ids:
        if mode == "sql":
            result = evaluate_single_sql_instance(
                pred_id, base_id, eval_standard_dict, spider2sql_metadata,
                pred_result_dir, gold_sql_dir, gold_result_dir, temp_dir, result_csv_dir,
                timeout=timeout, credential_path=credential_path
            )
        elif mode == "exec_result":
            result = evaluate_single_exec_result_instance(
                pred_id, base_id, eval_standard_dict, pred_result_dir, gold_result_dir
            )

        per_pred_scores[pred_id] = result['score']

        # Track all correct predictions
        if result['score'] == 1:
            correct_program_names.append(pred_id)
            if result.get('pred_sql'):
                last_pred_sql = result.get('pred_sql')
        else:
            errors.append(f"{pred_id}: {result['error_info']}")

        # Accumulate per-gold coverage (element-wise OR across predictions)
        pgs = result.get('per_gold_scores')
        if pgs is not None:
            if micro_gold_covered is None:
                micro_gold_covered = list(pgs)
            else:
                for i in range(min(len(micro_gold_covered), len(pgs))):
                    micro_gold_covered[i] = max(micro_gold_covered[i], pgs[i])

    # Return score of 1 if any prediction was correct (pass@k)
    score = 1 if correct_program_names else 0
    return {
        "instance_id": base_id,
        "score": score,
        "pred_sql": last_pred_sql if correct_program_names else "All predictions failed or were incorrect.",
        "error_info": None if correct_program_names else "; ".join(errors),
        "correct_program_names": correct_program_names,
        "micro_gold_covered": micro_gold_covered,
        "per_pred_scores": per_pred_scores,
    }


def save_annotation_dict(annotation_dict, result_dir):
    """
    Save annotation dictionary mapping question IDs to lists of correct program names.
    
    Args:
        annotation_dict: Dictionary with keys as question IDs and values as lists of correct program names
        result_dir: Result directory (parent directory will be used to save the file)
    """
    parent_dir = os.path.dirname(result_dir)
    result_dir_name = os.path.basename(result_dir)
    json_file_path = os.path.join(parent_dir, f"{result_dir_name}_annotation.json")
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Annotation dictionary saved to: {json_file_path}")
    
    return json_file_path


def execute_single_program_wrapper(program_path, base_id, sub_queries_path, output_path, credential_path, timeout):
    """
    Wrapper function for execute_program to make it picklable for multiprocessing.
    """
    program_id = os.path.basename(program_path).replace('program_', '').replace('.py', '')
    prefix = f"[{base_id}/program_{program_id}] "
    
    print(f"{prefix}Executing...")
    success, error_msg = execute_program(
        program_path,
        base_id,
        sub_queries_path,
        output_path,
        credential_path,
        timeout
    )
    
    if success:
        print(f"{prefix}✓ Success")
        return {'base_id': base_id, 'program_id': program_id, 'success': True, 'error': None}
    else:
        print(f"{prefix}✗ Failed: {error_msg}")
        return {'base_id': base_id, 'program_id': program_id, 'success': False, 'error': error_msg}


def execute_program(program_path, base_id, sub_queries_path, output_path, credential_path="spider2-snow/snowflake_credential.json", timeout=300):
    """
    Execute a program file using program_frame.py
    
    Args:
        program_path: Path to the program_<id>.py file
        base_id: Base ID (e.g., sf_bq263)
        sub_queries_path: Path to sub_queries.json
        output_path: Path where program_output_<id>.csv should be saved
        credential_path: Path to snowflake_credential.json
        timeout: Execution timeout in seconds
    
    Returns:
        (success: bool, error_message: str or None)
    """
    cmd = [
        "python", "utils/program_frame.py",
        credential_path,
        base_id,
        program_path,
        sub_queries_path,
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        if result.returncode == 0:
            # Verify output file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True, None
            else:
                return False, "Output file was not created or is empty"
        else:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            return False, f"Execution failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def execute_missing_programs(args):
    """
    Scan result_dir for subdirectories containing program_<id>.py files,
    and execute those that don't have corresponding program_output_<id>.csv files.
    """
    result_dir = args.result_dir
    credential_path = getattr(args, 'credential_path', 'spider2-snow/snowflake_credential.json')
    timeout = getattr(args, 'program_timeout', 3600)
    max_workers = min(args.max_workers if hasattr(args, 'max_workers') else 8, 20)
    
    # Collect all programs that need execution
    programs_to_execute = []
    
    # Scan result_dir for subdirectories
    if not os.path.isdir(result_dir):
        print(f"Error: {result_dir} is not a directory")
        return
    
    for item_name in os.listdir(result_dir):
        subfolder_path = os.path.join(result_dir, item_name)
        
        # Process only if the item is a directory
        if os.path.isdir(subfolder_path):
            sub_queries_path = os.path.join(subfolder_path, "sub_queries.json")
            
            # Check if sub_queries.json exists
            if not os.path.exists(sub_queries_path):
                print(f"Warning: sub_queries.json not found in {subfolder_path}, skipping")
                continue
            
            # Find all program_<id>.py files
            for filename in os.listdir(subfolder_path):
                if filename.startswith("program_") and filename.endswith(".py"):
                    # Extract program_id from filename (e.g., "program_2.py" -> "2")
                    match = re.match(r'program_(\d+)\.py$', filename)
                    if match:
                        program_id = match.group(1)
                        program_path = os.path.join(subfolder_path, filename)
                        output_filename = f"program_output_{program_id}.csv"
                        output_path = os.path.join(subfolder_path, output_filename)
                        
                        # Check if output file exists
                        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                            programs_to_execute.append({
                                'base_id': item_name,
                                'program_id': program_id,
                                'program_path': program_path,
                                'sub_queries_path': sub_queries_path,
                                'output_path': output_path
                            })
                        else:
                            print(f"Skipping {item_name}/program_{program_id}.py (output already exists)")
    
    if not programs_to_execute:
        print("No programs need execution. All output files exist.")
        return
    
    print(f"Found {len(programs_to_execute)} programs to execute")
    
    # Execute in parallel with ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_program = {
            executor.submit(
                execute_single_program_wrapper,
                prog_info['program_path'],
                prog_info['base_id'],
                prog_info['sub_queries_path'],
                prog_info['output_path'],
                credential_path,
                timeout
            ): prog_info 
            for prog_info in programs_to_execute
        }
        
        for future in tqdm(as_completed(future_to_program), total=len(programs_to_execute), desc="Executing programs"):
            result = future.result()
            results.append(result)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    print(f"\nExecution Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed programs:")
        for r in results:
            if not r['success']:
                print(f"  {r['base_id']}/program_{r['program_id']}: {r['error']}")


def build_multi_gold_set(gold_result_dir, jsonl_path):
    """
    Return a set of base_ids that have >1 gold answer variant AND exist in the jsonl dataset.
    These are the questions used for micro accuracy@k.
    """
    with open(jsonl_path) as f:
        jsonl_ids = {json.loads(line)['instance_id'] for line in f}

    from collections import defaultdict
    gold_groups = defaultdict(list)
    for fname in os.listdir(gold_result_dir):
        if fname.endswith('.csv'):
            base = re.sub(r'_[a-z]\.csv$', '', fname)
            gold_groups[base].append(fname)

    return {
        base_id: files
        for base_id, files in gold_groups.items()
        if len(files) > 1 and base_id in jsonl_ids
    }


def evaluate_spider2sql(args):
    mode = args.mode
    gold_sql_dir = os.path.join(args.gold_dir, "sql")
    gold_result_dir = os.path.join(args.gold_dir, "exec_result")
    pred_result_dir = args.result_dir

    eval_standard_dict = load_jsonl_to_dict(os.path.join(args.gold_dir, f"spider2{args.task}_eval.jsonl"))
    spider2sql_metadata = load_jsonl_to_dict("spider2-snow/spider2-snow.jsonl")

    # Pre-compute the set of multi-gold questions (in jsonl) for micro accuracy@k
    dataset_jsonl = getattr(args, 'dataset_jsonl', 'spider2-snow/spider2-snow.jsonl')
    multi_gold_set = build_multi_gold_set(gold_result_dir, dataset_jsonl)
    print(f"Multi-gold questions (in jsonl, >1 gold answer): {len(multi_gold_set)}")
    
    result_csv_dir = None
    if mode == "sql":
        result_csv_dir = f"{pred_result_dir}_csv"
        if os.path.exists(result_csv_dir):
            shutil.rmtree(result_csv_dir)
        os.makedirs(result_csv_dir)
    
    # --- MODIFICATION START: Group predictions for pass@k ---
    pred_groups = {}
    file_extension = ".sql" if mode == "sql" else ".csv"
    for file in os.listdir(args.result_dir):
        if file.endswith(file_extension):
            file_base_name = file.rsplit('.', 1)[0]
            # Support filenames that end with one or two numeric suffixes, e.g.
            #   base_id_3
            #   base_id_12_0
            parts = file_base_name.split('_')
            base_id = file_base_name
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                base_id = '_'.join(parts[:-2])
            elif len(parts) >= 2 and parts[-1].isdigit():
                base_id = '_'.join(parts[:-1])
            if not base_id:
                base_id = file_base_name
            
            if base_id not in pred_groups:
                pred_groups[base_id] = []
            pred_groups[base_id].append(file_base_name)

    gold_ids = list(eval_standard_dict.keys())
    # Evaluate base_ids that exist in both the gold set and have predictions.
    eval_base_ids = sorted(list(set(gold_ids).intersection(pred_groups.keys())))
    if getattr(args, 'restrict_to_gold', False):
        gold_sql_ids = {f[:-4] for f in os.listdir(gold_sql_dir) if f.endswith('.sql')}
        eval_base_ids = [bid for bid in eval_base_ids if bid in gold_sql_ids]
        print(f"--restrict_to_gold: restricted to {len(eval_base_ids)} questions with gold SQL")
    
    output_results = []
    max_workers = min(args.max_workers if hasattr(args, 'max_workers') else 8, len(eval_base_ids))

    # Prepare arguments for multiprocessing
    temp_dir_path = os.path.abspath("temp")
    
    # --- MODIFIED: Use ProcessPoolExecutor with module-level function ---
    print(f"Evaluating {mode} in pass@k mode with {max_workers} workers (multiprocessing)")
    print(f"Total groups to evaluate: {len(eval_base_ids)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(
                evaluate_group,
                base_id,
                pred_groups[base_id],
                mode,
                eval_standard_dict,
                spider2sql_metadata,
                pred_result_dir,
                gold_sql_dir,
                gold_result_dir,
                temp_dir_path,
                result_csv_dir,
                args.timeout,
                getattr(args, 'credential_path', None)
            ): base_id for base_id in eval_base_ids
        }
        
        completed = 0
        correct_count = 0
        for future in as_completed(future_to_id):
            result = future.result()
            output_results.append(result)
            completed += 1
            
            # Update pass@k statistics
            if result['score'] == 1:
                correct_count += 1
            
            # Print running update every 10 completions or for every completion if < 50 total
            if completed % 10 == 0 or len(eval_base_ids) < 50 or completed == len(eval_base_ids):
                elapsed = time.time() - start_time
                pass_at_k = correct_count / completed if completed > 0 else 0.0
                progress_pct = (completed / len(eval_base_ids)) * 100
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(eval_base_ids) - completed) / rate if rate > 0 else 0
                
                print(f"[Pass@K Update] Completed: {completed}/{len(eval_base_ids)} ({progress_pct:.1f}%) | "
                      f"Correct: {correct_count} | Pass@K: {pass_at_k:.4f} | "
                      f"Rate: {rate:.2f} groups/s | ETA: {eta:.0f}s")
    # --- MODIFICATION END ---
    
    print({item['instance_id']: item['score'] for item in output_results})
    correct_examples = sum([item['score'] for item in output_results])

    total_examples = len(eval_base_ids)
    print(f"Final score: {correct_examples / total_examples if total_examples > 0 else 0}, Correct examples: {correct_examples}, Total examples: {total_examples}")
    print(f"Real score: {correct_examples / 547}, Correct examples: {correct_examples}, Total examples: 547")

    # --- Micro accuracy@k ---
    # For each multi-gold question (in jsonl, >1 gold), score = covered_variants / total_variants.
    # Micro accuracy@k = mean of per-question scores.
    micro_scores = []
    results_by_id = {item['instance_id']: item for item in output_results}
    for base_id, gold_files in multi_gold_set.items():
        if base_id not in results_by_id:
            # No prediction for this question; score 0
            micro_scores.append(0.0)
            continue
        covered = results_by_id[base_id].get('micro_gold_covered')
        n_gold = len(gold_files)
        if covered is None or len(covered) == 0:
            micro_scores.append(0.0)
        else:
            micro_scores.append(sum(covered) / n_gold)
    micro_acc = sum(micro_scores) / len(micro_scores) if micro_scores else 0.0
    print(f"Micro accuracy@K: {micro_acc:.4f} "
          f"(avg gold-variant coverage over {len(multi_gold_set)} multi-gold questions in dataset)")
    
    if mode == "sql" and result_csv_dir:
        print(f"Execution results saved to: {result_csv_dir}")
    
    csv_file_path = save_correct_ids_to_csv(output_results, pred_result_dir)
    
    # Build and save annotation dictionary
    annotation_dict = {}
    for item in output_results:
        base_id = item['instance_id']
        correct_programs = item.get('correct_program_names', [])
        if correct_programs:  # Only include entries with at least one correct program
            annotation_dict[base_id] = correct_programs
    
    save_annotation_dict(annotation_dict, pred_result_dir)

    # --- Min-plan (plan-0) accuracy ---
    # For each question pick the single output with the lowest (plan_id, prog_id).
    def _plan_prog_key(pred_id):
        parts = pred_id.split('_')
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
            return (int(parts[-2]), int(parts[-1]))
        if len(parts) >= 1 and parts[-1].isdigit():
            return (int(parts[-1]), 0)
        return (0, 0)

    min_plan_correct = 0
    min_plan_total = 0
    results_by_id = {r['instance_id']: r for r in output_results}
    for base_id in eval_base_ids:
        result = results_by_id.get(base_id)
        if result is None:
            continue
        min_pred = min(pred_groups[base_id], key=_plan_prog_key)
        score = result['per_pred_scores'].get(min_pred, 0)
        min_plan_correct += score
        min_plan_total += 1

    print(f"\nPlan-0 (min-plan) accuracy: "
          f"{min_plan_correct / min_plan_total if min_plan_total else 0:.4f} "
          f"| Correct: {min_plan_correct} / {min_plan_total}")
    print(f"Plan-0 real score (over 547): {min_plan_correct / 547:.4f}")

    # --- Gold-SQL subset report ---
    gold_sql_ids = {f[:-4] for f in os.listdir(gold_sql_dir) if f.endswith('.sql')}
    subset_base_ids = [bid for bid in eval_base_ids if bid in gold_sql_ids]
    subset_results = [r for r in output_results if r['instance_id'] in gold_sql_ids]
    subset_total = len(subset_base_ids)
    subset_correct = sum(r['score'] for r in subset_results)
    N_GOLD_SQL = len(gold_sql_ids)

    print(f"\n{'='*60}")
    print(f"Gold-SQL subset report ({subset_total} evaluated / {N_GOLD_SQL} total gold-SQL questions)")
    print(f"{'='*60}")
    print(f"Final score: {subset_correct / subset_total if subset_total else 0:.4f}, "
          f"Correct: {subset_correct}, Total evaluated: {subset_total}")
    print(f"Real score (over {N_GOLD_SQL}): {subset_correct / N_GOLD_SQL:.4f}")

    subset_micro_scores = []
    subset_results_by_id = {r['instance_id']: r for r in subset_results}
    for base_id, gold_files in multi_gold_set.items():
        if base_id not in gold_sql_ids:
            continue
        covered = subset_results_by_id.get(base_id, {}).get('micro_gold_covered')
        n_gold = len(gold_files)
        if covered is None or len(covered) == 0:
            subset_micro_scores.append(0.0)
        else:
            subset_micro_scores.append(sum(covered) / n_gold)
    subset_micro_acc = sum(subset_micro_scores) / len(subset_micro_scores) if subset_micro_scores else 0.0
    print(f"Micro accuracy@K: {subset_micro_acc:.4f} "
          f"(over {len(subset_micro_scores)} multi-gold questions in subset)")

    subset_min_plan_correct = 0
    subset_min_plan_total = 0
    for base_id in subset_base_ids:
        result = subset_results_by_id.get(base_id)
        if result is None:
            continue
        min_pred = min(pred_groups[base_id], key=_plan_prog_key)
        score = result['per_pred_scores'].get(min_pred, 0)
        subset_min_plan_correct += score
        subset_min_plan_total += 1

    print(f"Plan-0 (min-plan) accuracy: "
          f"{subset_min_plan_correct / subset_min_plan_total if subset_min_plan_total else 0:.4f} "
          f"| Correct: {subset_min_plan_correct} / {subset_min_plan_total}")
    print(f"Plan-0 real score (over {N_GOLD_SQL}): {subset_min_plan_correct / N_GOLD_SQL:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations for NLP models.")
    parser.add_argument("--mode", type=str, choices=["sql", "exec_result", "execute_programs"], default='sql', help="Mode of submission results")
    parser.add_argument("--result_dir", type=str, default="spider2sql_example_submit_result", help="Result directory")
    parser.add_argument("--gold_dir", type=str, default="gold", help="Result directory")
    parser.add_argument("--is_sql_debug", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of worker processes")
    parser.add_argument("--timeout", type=int, default=3600, help="SQL execution timeout in seconds")
    parser.add_argument("--credential_path", type=str, default="spider2-snow/snowflake_credential.json", help="Path to snowflake credential JSON file")
    parser.add_argument("--program_timeout", type=int, default=3600, help="Program execution timeout in seconds")
    parser.add_argument("--task", type=str, default="snow", choices=["snow", "lite"], help="Task to evaluate")
    parser.add_argument("--dataset_jsonl", type=str, default="spider2-snow/spider2-snow.jsonl", help="Path to dataset JSONL file (used for multi-gold question detection)")
    parser.add_argument("--restrict_to_gold", action="store_true", default=False, help="Restrict evaluation to questions that have a gold SQL file in gold/sql/")
    args = parser.parse_args()
    
    if args.mode == "execute_programs":
        execute_missing_programs(args)
    else:
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        os.makedirs("temp")
        
        evaluate_spider2sql(args)