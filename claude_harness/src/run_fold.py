#!/usr/bin/env python3
"""
run_fold.py — process a JSONL fold of questions with Claude Code.

Iterates over every question in the fold file and calls run_question for each.
Continues on individual question failures; writes a per-fold summary log.

Usage:
  python src/run_fold.py --fold path/to/fold_001.jsonl --output_dir inference_res/run_name
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure we can import run_question from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from run_question import run_question
from mcp_launcher import running_mcp_server


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude Code on all questions in a JSONL fold file."
    )
    parser.add_argument("--fold", type=Path, required=True, help="Path to fold JSONL file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory root")
    parser.add_argument("--db_path", default="spider2-snow")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per question (seconds)")
    parser.add_argument(
        "--claude_path", default="claude", help="Path to the claude CLI binary"
    )
    parser.add_argument("--model", default="sonnet", help="Claude model (e.g. sonnet, opus, haiku)")
    parser.add_argument("--effort", default=None, help="Thinking effort level (low, medium, high, max)")
    parser.add_argument(
        "--no_skip", action="store_true", help="Re-run questions even if answer.sql exists"
    )
    parser.add_argument(
        "--max_retries", type=int, default=1,
        help="Max revision retries per question (default: 1). 0 = no retries.",
    )
    args = parser.parse_args()

    if not args.fold.exists():
        print(f"Fold file not found: {args.fold}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CCSQL_DB_PATH"] = args.db_path

    # Load questions
    questions = []
    with open(args.fold) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Fold: {args.fold.name}  |  {len(questions)} questions  |  output: {args.output_dir}")

    fold_log_path = args.output_dir / f"fold_log_{args.fold.stem}.jsonl"
    mcp_log_path = args.output_dir / f"mcp_server_{args.fold.stem}.log"
    results = []
    t_fold_start = time.time()

    # One persistent HTTP MCP server per fold. Dynamic port (bind 0) so
    # concurrent folds on the same node don't collide. Per-question db_id and
    # session_id are passed as HTTP headers via a generated --mcp-config file.
    with running_mcp_server(
        db_path=args.db_path, db_type="snowflake", log_path=mcp_log_path,
    ) as mcp:
        print(f"MCP server: {mcp.url}  (pid={mcp.proc.pid}, log={mcp_log_path})", flush=True)

        for i, question in enumerate(questions, 1):
            instance_id = question.get("instance_id") or question.get("question_id", f"q{i}")
            print(f"  [{i}/{len(questions)}] {instance_id} (db={question.get('db_id', '?')}) ...", flush=True)

            if not mcp.is_alive():
                print("  [FATAL] MCP server died mid-fold; aborting remaining questions", file=sys.stderr)
                break

            result = run_question(
                question=question,
                output_dir=args.output_dir,
                db_path=args.db_path,
                timeout=args.timeout,
                claude_path=args.claude_path,
                model=args.model,
                effort=args.effort,
                skip_existing=not args.no_skip,
                max_retries=args.max_retries,
                mcp_url=mcp.url,
            )
            result["fold"] = args.fold.name
            results.append(result)

            status_symbol = {"success": "OK", "skipped": "--", "timeout": "TO", "error": "ERR", "no_output": "??"}.get(
                result["status"], "?"
            )
            print(f"    [{status_symbol}] {result['message']}", flush=True)

            # Append to fold log incrementally
            with open(fold_log_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    elapsed = time.time() - t_fold_start
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(f"\nFold complete in {elapsed:.0f}s:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"Log written to: {fold_log_path}")


if __name__ == "__main__":
    main()
