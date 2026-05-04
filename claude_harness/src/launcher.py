#!/usr/bin/env python3
"""
launcher.py — SLURM array job launcher for CCSQL.

Splits a dataset into N folds and submits a SLURM array job where each task
processes one fold via run_fold.py (Claude Code text-to-SQL inference).

Each array task runs Claude Code (the CLI) sequentially over the questions in
its fold, with MCP tools for database exploration.

Usage example:
  python src/launcher.py \\
      --input_file spider2-snow/spider2-snow.jsonl \\
      --num_folds 20 \\
      --run_name my-run-001 \\
      --output_dir inference_res \\
      --sbatch_args $'#SBATCH --time=4:00:00\\n#SBATCH --account=def-xiye17_cpu' \\
      --max_concurrent_jobs 10 \\
      --timeout 600

The generated SLURM script activates CCSQL's .venv and calls run_fold.py.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CCSQL_ROOT = Path(__file__).parent.parent.resolve()


# ── Dataset splitting ──────────────────────────────────────────────────────────

def split_dataset(input_file: Path, num_folds: int, output_dir: Path) -> list:
    """Split a JSON array or JSONL dataset into N JSONL fold files."""
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

    chunk_size = (len(data) + num_folds - 1) // num_folds
    fold_files = []

    for i in range(num_folds):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(data))
        if start >= len(data):
            break

        fold_path = output_dir / f"fold_{i+1:03d}_of_{num_folds}.jsonl"
        with open(fold_path, "w", encoding="utf-8") as fh:
            for entry in data[start:end]:
                fh.write(json.dumps(entry) + "\n")
        fold_files.append(fold_path.resolve())

    print(f"Split {input_file.name} ({len(data)} samples) into {len(fold_files)} folds in {output_dir}")
    return fold_files


# ── SLURM submission ───────────────────────────────────────────────────────────

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


# ── SLURM template ─────────────────────────────────────────────────────────────

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=ccsql
#SBATCH --output={log_dir}/ccsql_%A_%a.log
#SBATCH --error={log_dir}/ccsql_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --array=0-{array_max}{array_throttle}
{dependency_line}{sbatch_args}

fold_file_path=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{folds_list_file}")
echo "Array task $SLURM_ARRAY_TASK_ID  fold: $fold_file_path"

cd {ccsql_root}
source {ccsql_root}/.venv/bin/activate

python src/run_fold.py \\
    --fold "$fold_file_path" \\
    --output_dir "{output_dir}" \\
    --db_path spider2-snow \\
    --timeout {timeout} \\
    --claude_path "{claude_path}" \\
    --model "{model}" \\
    {extra_args}

exit_code=$?
echo "run_fold.py exited with code $exit_code"
exit $exit_code
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch a SLURM array job for CCSQL (Claude Code text-to-SQL inference).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input_file", type=Path, required=True,
                        help="Dataset file (JSON array or JSONL)")
    parser.add_argument("--num_folds", type=int, default=20,
                        help="Number of parallel SLURM tasks (default: 20)")
    parser.add_argument("--max_concurrent_jobs", type=int, default=None,
                        help="Cap on simultaneously running array tasks")
    parser.add_argument("--sbatch_args",
                        default="#SBATCH --time=2:00:00\n#SBATCH --account=def-xiye17_cpu",
                        help="Extra #SBATCH directives inserted into the script")
    parser.add_argument("--timeout", type=int, default=660,
                        help="Timeout in seconds per question (default: 600)")
    parser.add_argument("--claude_path", default="claude",
                        help="Path to the claude CLI binary (default: claude)")
    parser.add_argument("--model", default="sonnet",
                        help="Claude model to use (e.g. sonnet, opus, haiku)")
    parser.add_argument("--effort", default=None,
                        help="Thinking effort level (low, medium, high, max)")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Max revision retries per question (default: 1). 0 = no retries.")
    parser.add_argument("--dependency", default=None,
                        help="SLURM dependency (e.g. afterok:12345678). Omit for none.")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-run questions even if answer.sql already exists")
    parser.add_argument("--run_name", default=f"run_{int(time.time())}",
                        help="Name for this run (used as subdirectory name)")
    parser.add_argument("--output_dir", type=Path, default=CCSQL_ROOT / "inference_res",
                        help="Root output directory (default: inference_res/)")
    args = parser.parse_args()

    # ── Directories ────────────────────────────────────────────────────────────
    run_dir = (args.output_dir / args.run_name).resolve()
    log_dir = run_dir / "slurm_logs"
    folds_dir = run_dir / "folds"

    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        folds_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Error: run directory '{run_dir}' already exists.", file=sys.stderr)
        sys.exit(1)

    print(f"--- CCSQL SLURM Launcher ---")
    print(f"Run directory : {run_dir}")
    print(f"Output dir    : {run_dir}")

    # ── Split dataset ──────────────────────────────────────────────────────────
    fold_files = split_dataset(args.input_file.resolve(), args.num_folds, folds_dir)
    if not fold_files:
        print("No fold files created. Exiting.", file=sys.stderr)
        sys.exit(1)

    folds_list_file = run_dir / "folds_list.txt"
    folds_list_file.write_text("\n".join(str(p) for p in fold_files) + "\n")
    print(f"Folds list    : {folds_list_file}")

    # ── Build SLURM script ─────────────────────────────────────────────────────
    array_max = len(fold_files) - 1
    array_throttle = f"%{args.max_concurrent_jobs}" if args.max_concurrent_jobs else ""
    extra_parts = []
    if args.no_skip:
        extra_parts.append("--no_skip")
    if args.effort:
        extra_parts.append(f"--effort {args.effort}")
    if args.max_retries is not None:
        extra_parts.append(f"--max_retries {args.max_retries}")
    extra_args = " \\\n    ".join(extra_parts)

    script = SLURM_TEMPLATE.format(
        log_dir=log_dir,
        array_max=array_max,
        array_throttle=array_throttle,
        dependency_line=f"#SBATCH --dependency={args.dependency}\n" if args.dependency else "",
        sbatch_args=args.sbatch_args,
        folds_list_file=folds_list_file,
        ccsql_root=CCSQL_ROOT,
        output_dir=run_dir,
        timeout=args.timeout,
        claude_path=args.claude_path,
        model=args.model,
        extra_args=extra_args,
    )

    # Save the script for reference
    script_path = run_dir / "slurm_job.sh"
    script_path.write_text(script)
    print(f"SLURM script  : {script_path}")

    # ── Submit ─────────────────────────────────────────────────────────────────
    throttle_msg = f" (max {args.max_concurrent_jobs} concurrent)" if args.max_concurrent_jobs else ""
    print(f"\nSubmitting {len(fold_files)}-task array job{throttle_msg} ...")

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
    print(f"Output dir    : {run_dir}")


if __name__ == "__main__":
    main()
