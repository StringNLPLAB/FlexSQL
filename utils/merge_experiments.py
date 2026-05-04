#!/usr/bin/env python3
"""
Merge two experiment folders into a combined folder for higher pass@k evaluation.

Each experiment folder contains question subfolders (e.g., sf001, sf_bq002) with:
  - plans.json: list of plans
  - program_{plan_idx}_{prog_idx}.sql: SQL program files
  - program_output_{plan_idx}_{prog_idx}.csv: execution output files
  - sub_queries_plan_{plan_idx}.json: sub-query details per plan
  - log.txt: execution log

The merge combines plans from both experiments per question:
  - Exp1 keeps plan indices 0..N-1
  - Exp2 plans are renumbered to N..N+M-1
  - All corresponding program/output/sub_queries files are renumbered accordingly

Derived files (evaluation/, major_vote/, etc.) are skipped since they
need to be regenerated on the merged data.

Usage:
    python merge_experiments.py <exp1_dir> <exp2_dir> <output_dir>
    python merge_experiments.py <exp1_dir> <exp2_dir>  # auto-generates output name
"""

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SKIP_ENTRIES = {
    "evaluation", "evaluation.csv", "evaluation_annotation.json",
    "major_vote", "major_vote.csv",
}

PROGRAM_RE = re.compile(r"^program_(\d+)_(\d+)\.(sql|py)$")
PROGRAM_OUTPUT_RE = re.compile(r"^program_output_(\d+)_(\d+)\.csv$")
SUB_QUERIES_RE = re.compile(r"^sub_queries_plan_(\d+)\.json$")


def get_question_dirs(exp_dir: Path) -> set[str]:
    """Return set of question folder names (skip derived outputs)."""
    return {
        d.name for d in exp_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_ENTRIES
    }


def count_plans(question_dir: Path) -> int:
    """Count the number of plans in a question folder from plans.json."""
    plans_file = question_dir / "plans.json"
    if plans_file.exists():
        with open(plans_file) as f:
            plans = json.load(f)
        return len(plans) if isinstance(plans, list) else 0
    return 0


def get_max_plan_index(question_dir: Path) -> int:
    """Get the highest plan index from files in the question directory.

    This is more robust than counting plans.json entries, since some plans
    may not have generated programs but still occupy index slots.
    """
    max_idx = -1
    for fname in os.listdir(question_dir):
        for pattern in (PROGRAM_RE, PROGRAM_OUTPUT_RE, SUB_QUERIES_RE):
            m = pattern.match(fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
                break
    # Also check plans.json
    plans_count = count_plans(question_dir)
    return max(max_idx + 1, plans_count)


def copy_question_files(src_dir: Path, dst_dir: Path, plan_offset: int = 0):
    """Copy all question files from src to dst, renumbering plan indices by offset."""
    for fname in sorted(os.listdir(src_dir)):
        src_path = src_dir / fname

        # Handle program files: program_{plan}_{prog}.{sql,py}
        m = PROGRAM_RE.match(fname)
        if m:
            plan_idx = int(m.group(1)) + plan_offset
            prog_idx = int(m.group(2))
            ext = m.group(3)
            new_name = f"program_{plan_idx}_{prog_idx}.{ext}"
            shutil.copy2(src_path, dst_dir / new_name)
            continue

        # Handle program output files: program_output_{plan}_{prog}.csv
        m = PROGRAM_OUTPUT_RE.match(fname)
        if m:
            plan_idx = int(m.group(1)) + plan_offset
            prog_idx = int(m.group(2))
            new_name = f"program_output_{plan_idx}_{prog_idx}.csv"
            shutil.copy2(src_path, dst_dir / new_name)
            continue

        # Handle sub_queries files: sub_queries_plan_{plan}.json
        m = SUB_QUERIES_RE.match(fname)
        if m:
            plan_idx = int(m.group(1)) + plan_offset
            new_name = f"sub_queries_plan_{plan_idx}.json"
            shutil.copy2(src_path, dst_dir / new_name)
            continue

        # Skip plans.json and log.txt here (handled separately)
        if fname in ("plans.json", "log.txt"):
            continue

        # Copy any other files as-is (only from first exp, or if not already there)
        dst_path = dst_dir / fname
        if not dst_path.exists():
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            elif src_path.is_dir():
                shutil.copytree(src_path, dst_path)


def merge_plans_json(dir1: Path, dir2: Path, dst_dir: Path):
    """Merge plans.json from both experiments by concatenating the lists."""
    plans1 = []
    plans2 = []

    plans1_file = dir1 / "plans.json"
    plans2_file = dir2 / "plans.json"

    if plans1_file.exists():
        with open(plans1_file) as f:
            plans1 = json.load(f)

    if plans2_file.exists():
        with open(plans2_file) as f:
            plans2 = json.load(f)

    merged = plans1 + plans2
    with open(dst_dir / "plans.json", "w") as f:
        json.dump(merged, f, indent=2)


def merge_logs(dir1: Path, dir2: Path, dst_dir: Path):
    """Concatenate log files from both experiments."""
    log_parts = []
    for i, d in enumerate([dir1, dir2], 1):
        log_file = d / "log.txt"
        if log_file.exists():
            with open(log_file) as f:
                content = f.read()
            log_parts.append(f"{'='*60}\n=== Experiment {i}: {d.parent.name}\n{'='*60}\n{content}")

    if log_parts:
        with open(dst_dir / "log.txt", "w") as f:
            f.write("\n".join(log_parts))


def merge_question(q_name: str, dir1: Path | None, dir2: Path | None, dst_dir: Path):
    """Merge a single question from both experiments."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    if dir1 and dir2:
        # Both experiments have this question
        plan_offset = get_max_plan_index(dir1)
        copy_question_files(dir1, dst_dir, plan_offset=0)
        copy_question_files(dir2, dst_dir, plan_offset=plan_offset)
        merge_plans_json(dir1, dir2, dst_dir)
        merge_logs(dir1, dir2, dst_dir)
    elif dir1:
        # Only in exp1
        shutil.copytree(dir1, dst_dir, dirs_exist_ok=True)
    elif dir2:
        # Only in exp2
        shutil.copytree(dir2, dst_dir, dirs_exist_ok=True)


def _process_question(args: tuple) -> tuple[str, int, int, str]:
    """Worker function: merge one question and return stats. Must be top-level for pickling."""
    q_name, dir1_str, dir2_str, out_q_str = args
    dir1 = Path(dir1_str) if dir1_str else None
    dir2 = Path(dir2_str) if dir2_str else None
    out_q = Path(out_q_str)

    merge_question(q_name, dir1, dir2, out_q)

    merged_plans = count_plans(out_q) if (out_q / "plans.json").exists() else 0
    merged_programs = len([f for f in os.listdir(out_q) if PROGRAM_RE.match(f)])
    src_label = "both" if dir1 and dir2 else ("exp1" if dir1 else "exp2")
    return q_name, merged_plans, merged_programs, src_label


def merge_experiments(exp1_dir: str, exp2_dir: str, output_dir: str, workers: int | None = None):
    """Merge two experiment folders into a combined output folder."""
    exp1 = Path(exp1_dir)
    exp2 = Path(exp2_dir)
    out = Path(output_dir)

    if not exp1.is_dir():
        raise FileNotFoundError(f"Experiment 1 not found: {exp1}")
    if not exp2.is_dir():
        raise FileNotFoundError(f"Experiment 2 not found: {exp2}")

    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out}\n"
            "Remove it first or choose a different name."
        )

    out.mkdir(parents=True)

    questions1 = get_question_dirs(exp1)
    questions2 = get_question_dirs(exp2)
    all_questions = sorted(questions1 | questions2)

    print(f"Experiment 1: {exp1.name}")
    print(f"  Questions: {len(questions1)}")
    print(f"Experiment 2: {exp2.name}")
    print(f"  Questions: {len(questions2)}")
    print(f"Union questions: {len(all_questions)}")
    print(f"Shared questions: {len(questions1 & questions2)}")
    print(f"Only in exp1: {len(questions1 - questions2)}")
    print(f"Only in exp2: {len(questions2 - questions1)}")
    print(f"Output: {out}")
    print()

    task_args = [
        (
            q_name,
            str(exp1 / q_name) if q_name in questions1 else None,
            str(exp2 / q_name) if q_name in questions2 else None,
            str(out / q_name),
        )
        for q_name in all_questions
    ]

    results: dict[str, tuple[int, int, str]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_q = {executor.submit(_process_question, args): args[0] for args in task_args}
        for future in as_completed(future_to_q):
            q_name, merged_plans, merged_programs, src_label = future.result()
            results[q_name] = (merged_plans, merged_programs, src_label)

    for q_name in all_questions:
        merged_plans, merged_programs, src_label = results[q_name]
        print(f"  {q_name}: {merged_plans} plans, {merged_programs} programs (from {src_label})")

    print(f"\nDone! Merged experiment saved to: {out}")
    print("Re-run evaluation on the merged folder to compute pass@k metrics.")


def auto_output_name(exp1_dir: str, exp2_dir: str) -> str:
    """Generate a sensible output directory name."""
    exp1_name = Path(exp1_dir).name
    parent = Path(exp1_dir).parent
    return str(parent / f"{exp1_name}--MERGED")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two experiment folders for higher pass@k evaluation."
    )
    parser.add_argument("exp1", help="Path to first experiment folder")
    parser.add_argument("exp2", help="Path to second experiment folder")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Path for merged output folder (auto-generated if omitted)"
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=None,
        help="Number of worker processes (default: number of CPUs)"
    )
    args = parser.parse_args()

    output = args.output or auto_output_name(args.exp1, args.exp2)
    merge_experiments(args.exp1, args.exp2, output, workers=args.workers)


if __name__ == "__main__":
    main()
