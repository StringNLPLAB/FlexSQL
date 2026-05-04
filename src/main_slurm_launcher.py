#!/usr/bin/env python3
"""
main_slurm_launcher.py

Splits a dataset into N folds and submits a SLURM array job where each
task processes one fold via main.py (agentic text-to-SQL inference).

Two operating modes:
  - vLLM mode  (default): spins up a local vLLM server on each node, then
                           calls main.py --ip / --port.
  - API mode (--api_mode): no GPU / no vLLM server; calls main.py
                           directly (works with OpenAI, Gemini, etc.).

The number of GPUs requested per SLURM task is automatically derived from
the -tp (tensor-parallel) value in --vllm_server_extra_args (default: 4).

Usage examples:

    # Agentic inference with an OpenAI API model
    python src/main_slurm_launcher.py \\
        --input_file dev_20240627/dev.json \\
        --api_mode \\
        --num_folds 4 \\
        --agent_script_extra_args "--model gpt-4o \\
            --db_path dev_20240627/dev_databases \\
            --db_type sqlite \\
            --num_programs 3 \\
            --planning_top_k 4"

    # Agentic inference with a local vLLM model
    python src/main_slurm_launcher.py \\
        --input_file dev_20240627/dev.json \\
        --vllm_model gpt-oss-20b \\
        --num_folds 8 \\
        --agent_script_extra_args "--model gpt-oss-20b \\
            --db_path dev_20240627/dev_databases \\
            --db_type sqlite \\
            --num_programs 3 \\
            --planning_top_k 4"
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (identical to baseline_launcher.py)
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(input_file: Path, num_folds: int, output_dir: Path, priority_ids=None) -> list:
    """
    Split a JSON array or JSONL dataset file into N JSONL fold files.
    If priority_ids is given (a set of question/instance id strings), those
    questions are placed first so they land in the earliest folds.
    Returns a list of absolute Paths to the fold files.
    """
    try:
        raw = input_file.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"Error reading {input_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not raw:
        print(f"Input file {input_file} is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        if raw.startswith("["):
            data = json.loads(raw)
        else:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception as exc:
        print(f"Error parsing {input_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print(f"Input file {input_file} is empty after parsing.", file=sys.stderr)
        sys.exit(1)

    if priority_ids:
        id_key = "instance_id" if "instance_id" in data[0] else "question_id"
        priority = [e for e in data if str(e.get(id_key, "")) in priority_ids]
        rest     = [e for e in data if str(e.get(id_key, "")) not in priority_ids]
        print(f"Priority questions: {len(priority)} (with gold SQL) placed first, {len(rest)} remaining after.")
        data = priority + rest

    chunk_size = (len(data) + num_folds - 1) // num_folds
    fold_files = []

    for i in range(num_folds):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(data))
        if start >= len(data):
            break

        fold_path = output_dir / f"fold_{i+1:03d}_of_{num_folds}.jsonl"
        try:
            with open(fold_path, "w", encoding="utf-8") as fh:
                for entry in data[start:end]:
                    fh.write(json.dumps(entry) + "\n")
            fold_files.append(fold_path.resolve())
        except Exception as exc:
            print(f"Error writing fold file {fold_path}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"Split {input_file.name} ({len(data)} samples) into {len(fold_files)} folds in {output_dir}")
    return fold_files


def submit_sbatch(script_content: str) -> str:
    """Submit a bash script to sbatch and return the job ID."""
    try:
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script_content,
            text=True,
            capture_output=True,
            check=True,
        )
        job_id = result.stdout.strip()
        if ";" in job_id:
            job_id = job_id.split(";")[-1]
        if not job_id.isdigit():
            raise ValueError(f"sbatch returned unexpected output: {job_id!r}")
        return job_id
    except subprocess.CalledProcessError as exc:
        print("--- sbatch submission FAILED ---", file=sys.stderr)
        print("STDOUT:", exc.stdout, file=sys.stderr)
        print("STDERR:", exc.stderr, file=sys.stderr)
        print("SCRIPT:", file=sys.stderr)
        print(script_content, file=sys.stderr)
        raise


def _parse_tp(vllm_server_extra_args: str, default: int = 4) -> int:
    """Extract the -tp / --tensor-parallel-size value from vllm serve args."""
    m = re.search(r"(?:^|\s)-tp\s+(\d+)", vllm_server_extra_args)
    if not m:
        m = re.search(r"(?:^|\s)--tensor-parallel-size\s+(\d+)", vllm_server_extra_args)
    return int(m.group(1)) if m else default


# ─────────────────────────────────────────────────────────────────────────────
# SLURM script templates
# ─────────────────────────────────────────────────────────────────────────────

# GPU node: spin up vLLM, wait for health, then run main.py.
VLLM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=main2_vllm
#SBATCH --output={log_dir}/main2_vllm_%A_%a.log
#SBATCH --error={log_dir}/main2_vllm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus={gpu_name}{num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --array=0-{array_max}{array_throttle}
{sbatch_args}

fold_file_path=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{folds_list_file}")
echo "Array task $SLURM_ARRAY_TASK_ID  fold: $fold_file_path"

cd {project_home}
source .venv/bin/activate

port=$(( 10000 + SLURM_JOB_ID % 50000 ))
echo "Using port $port (SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID)"
export TIKTOKEN_RS_CACHE_DIR=/scratch/qhp/models/tiktoken_cache/

echo "Starting vLLM server..."
vllm serve {vllm_model} --served-model-name {served_model_name} --host "0.0.0.0" --port $port {vllm_server_extra_args} &
server_pid=$!

sleep 2
if ! kill -0 $server_pid 2>/dev/null; then
    echo "vLLM server died immediately."
    exit 1
fi

healthcheck_url="http://127.0.0.1:$port/health"
echo "Waiting for $healthcheck_url ..."
timeout=1800
start_time=$(date +%s)
until curl -s --fail $healthcheck_url > /dev/null; do
    sleep 10
    if [ $(( $(date +%s) - start_time )) -ge $timeout ]; then
        echo "Health check timed out."
        kill $server_pid 2>/dev/null
        exit 1
    fi
done

node_ip=$(ifconfig eth0 2>/dev/null | grep 'inet ' | awk '{{print $2}}')
if [ -z "$node_ip" ]; then
    node_ip=$(hostname -i 2>/dev/null | awk '{{print $1}}')
fi
if [ -z "$node_ip" ]; then
    node_ip=127.0.0.1
fi
echo "vLLM ready at $node_ip:$port"

python {worker_script} \\
    --dataset "$fold_file_path" \\
    --ip "$node_ip" \\
    --port "$port" \\
    {agent_script_extra_args}

worker_exit=$?
echo "Worker finished with exit code $worker_exit"

echo "Shutting down vLLM (PID $server_pid)..."
kill $server_pid 2>/dev/null
sleep 5
kill -9 $server_pid 2>/dev/null

exit $worker_exit
"""

# CPU-only / API model: no vLLM server, run main.py directly.
API_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=main2_api
#SBATCH --output={log_dir}/main2_api_%A_%a.log
#SBATCH --error={log_dir}/main2_api_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --array=0-{array_max}{array_throttle}
{sbatch_args}

fold_file_path=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{folds_list_file}")
echo "Array task $SLURM_ARRAY_TASK_ID  fold: $fold_file_path"

cd {project_home}
source .venv/bin/activate

python {worker_script} \\
    --dataset "$fold_file_path" \\
    {agent_script_extra_args}

worker_exit=$?
echo "Worker finished with exit code $worker_exit"
exit $worker_exit
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a SLURM array job for agentic text-to-SQL inference. "
            "Each array task processes one fold of the dataset via main.py."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required
    parser.add_argument("--input_file", type=Path, required=True,
                        help="Dataset file (JSON array or JSONL, e.g. datasets/Spider2/spider2-snow/spider2-snow.jsonl)")

    # Mode
    parser.add_argument("--api_mode", action="store_true",
                        help="API-based model (OpenAI/Gemini). No GPU, no vLLM server.")
    parser.add_argument("--vllm_model", default=None,
                        help="Model path/name for vLLM (required unless --api_mode).")
    parser.add_argument("--vllm_server_extra_args", default="-tp 1 --enable-auto-tool-choice --tool-call-parser openai",
                        help=(
                            "Extra args forwarded to `vllm serve`.\n"
                            "The -tp / --tensor-parallel-size value is also used to set\n"
                            "#SBATCH --gpus automatically (default: 4)."
                        ))
    parser.add_argument("--gpu_name", default=None,
                        help="Optional GPU type for #SBATCH --gpus (e.g. v100l, a100). "
                             "When set, the directive becomes --gpus=<gpu_name>:<num_gpus>.")

    # Folding & throttling
    parser.add_argument("--num_folds", type=int, default=4,
                        help="Number of parallel SLURM tasks.")
    parser.add_argument("--max_concurrent_jobs", type=int, default=None,
                        help="Cap on simultaneously running array tasks.")

    # SLURM & paths
    parser.add_argument("--sbatch_args", default="#SBATCH --time=3:00:00",
                        help="Extra #SBATCH directives inserted into the script.")
    parser.add_argument("--worker_script", type=Path,
                        default=Path(__file__).parent / "main.py",
                        help="Path to the per-fold worker script.")
    parser.add_argument("--agent_script_extra_args", default=None,
                        help=(
                            "All remaining args passed verbatim to main.py.\n"
                            "Must include at minimum:\n"
                            "  --model <name>\n"
                            "  --db_path <path>\n"
                            "  --db_type sqlite|snowflake\n"
                            "Common optional args:\n"
                            "  --num_programs N\n"
                            "  --planning_top_k N\n"
                            "  --planning_beam_size N\n"
                            "  --workers N\n"
                            "  --custom_exp_name <name>"
                        ))

    # Gold SQL prioritization
    parser.add_argument("--gold_sql_dir", type=Path, default=None,
                        help="Directory of gold .sql files (e.g. datasets/Spider2/spider2-snow/evaluation_suite/gold/sql). "
                             "Questions whose instance_id/question_id matches a filename stem are placed "
                             "in the earliest folds.")

    # Run naming
    parser.add_argument("--slurm_log_dir", default=f"run_{int(time.time())}")
    parser.add_argument("--base_dir", type=Path, default=Path.cwd())
    parser.add_argument("--project_home", type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help="Project root the SLURM task should `cd` into before "
                             "activating .venv (default: parent dir of this script).")

    args = parser.parse_args()

    if not args.api_mode and not args.vllm_model:
        parser.error("--vllm_model is required unless --api_mode is used.")

    # ── Set up run directory ─────────────────────────────────────────────────
    run_dir = (args.base_dir / args.slurm_log_dir).resolve()
    log_dir = run_dir / "slurm_logs"
    folds_dir = run_dir / "folds"

    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        folds_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Error: run directory '{run_dir}' already exists.", file=sys.stderr)
        sys.exit(1)

    print(f"--- main.py SLURM Launcher ---")
    print(f"Run directory : {run_dir}")

    # ── Split dataset ────────────────────────────────────────────────────────
    priority_ids = None
    if args.gold_sql_dir:
        gold_dir = args.gold_sql_dir.resolve()
        priority_ids = {p.stem for p in gold_dir.glob("*.sql")}
        print(f"Gold SQL dir  : {gold_dir} ({len(priority_ids)} files → priority questions)")

    fold_files = split_dataset(args.input_file.resolve(), args.num_folds, folds_dir, priority_ids=priority_ids)
    if not fold_files:
        print("No fold files created. Exiting.", file=sys.stderr)
        sys.exit(1)

    folds_list_file = run_dir / "folds_list.txt"
    folds_list_file.write_text("\n".join(str(p) for p in fold_files) + "\n")
    print(f"Folds list    : {folds_list_file}")

    # ── Build SLURM script ───────────────────────────────────────────────────
    array_max = len(fold_files) - 1
    array_throttle = f"%{args.max_concurrent_jobs}" if args.max_concurrent_jobs else ""

    if args.api_mode:
        template = API_TEMPLATE
    else:
        template = VLLM_TEMPLATE

    # Derive GPU count from -tp in vllm_server_extra_args.
    num_gpus = _parse_tp(args.vllm_server_extra_args) if not args.api_mode else 0

    fmt_kwargs = dict(
        log_dir=log_dir,
        array_max=array_max,
        array_throttle=array_throttle,
        sbatch_args=args.sbatch_args,
        folds_list_file=folds_list_file,
        worker_script=args.worker_script.resolve(),
        agent_script_extra_args=args.agent_script_extra_args,
        project_home=args.project_home.resolve(),
    )
    if not args.api_mode:
        fmt_kwargs["vllm_model"] = args.vllm_model
        fmt_kwargs["served_model_name"] = Path(args.vllm_model).name
        fmt_kwargs["vllm_server_extra_args"] = args.vllm_server_extra_args
        fmt_kwargs["num_gpus"] = num_gpus
        fmt_kwargs["gpu_name"] = f"{args.gpu_name}:" if args.gpu_name else ""

    script = template.format(**fmt_kwargs)

    # ── Submit ───────────────────────────────────────────────────────────────
    mode_label = "API mode" if args.api_mode else f"vLLM ({args.vllm_model})"
    throttle_msg = f" (max {args.max_concurrent_jobs} concurrent)" if args.max_concurrent_jobs else ""
    gpu_msg = f", {num_gpus} GPUs/task" if not args.api_mode else ""
    print(f"\nSubmitting {len(fold_files)}-task array job [{mode_label}{gpu_msg}]{throttle_msg} ...")

    try:
        job_id = submit_sbatch(script)
    except Exception as exc:
        print(f"Submission failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Submission complete ---")
    print(f"Job ID        : {job_id}")
    print(f"Array range   : 0-{array_max}  ({len(fold_files)} tasks)")
    print(f"Monitor       : squeue -u $USER")
    print(f"Cancel all    : scancel {job_id}")


if __name__ == "__main__":
    main()
