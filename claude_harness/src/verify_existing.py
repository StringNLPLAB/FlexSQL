#!/usr/bin/env python3
"""Run the independent verifier on existing fold outputs (no re-run of the
main agent loop). Writes verifier.json next to each answer.sql and prints a
summary table comparing verdict vs Spider2 eval score.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_question import run_verifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=Path, required=True, help="JSONL fold with questions")
    parser.add_argument("--output_dir", type=Path, required=True, help="Dir containing per-question subdirs + CSVs")
    parser.add_argument("--scores_json", type=Path, help="Optional JSON dict {instance_id: 0|1} from Spider2 eval")
    parser.add_argument("--claude_path", default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    questions = {}
    for line in open(args.fold):
        d = json.loads(line)
        questions[d["instance_id"]] = d

    scores = {}
    if args.scores_json and args.scores_json.exists():
        scores = json.loads(args.scores_json.read_text())

    results = []
    for i, (qid, q) in enumerate(questions.items(), 1):
        qdir = args.output_dir / qid
        csv_path = qdir / f"{qid}.csv"
        verifier_file = qdir / "verifier.json"
        if not qdir.exists():
            print(f"  [{i}/{len(questions)}] {qid}: no output dir — skip")
            continue
        t0 = time.time()
        verdict = run_verifier(
            question=q,
            csv_path=csv_path,
            verifier_file=verifier_file,
            claude_path=args.claude_path,
            model=args.model,
            timeout=args.timeout,
        )
        elapsed = time.time() - t0
        v = verdict.get("verdict", "?")
        score = scores.get(qid, "?")
        print(f"  [{i}/{len(questions)}] {qid}  verdict={v:<9}  score={score}  ({elapsed:.0f}s)  {verdict.get('reason','')[:80]}")
        results.append({"qid": qid, "verdict": v, "score": score, "elapsed": elapsed})

    # Summary confusion matrix
    print()
    print(f"{'verdict':<12} {'score=1':>8} {'score=0':>8} {'no-csv':>8}")
    print('-'*40)
    for v in ("correct", "incorrect", "uncertain"):
        row = [r for r in results if r["verdict"] == v]
        n1 = sum(1 for r in row if r["score"] == 1)
        n0 = sum(1 for r in row if r["score"] == 0)
        nx = sum(1 for r in row if r["score"] == "?")
        print(f"{v:<12} {n1:>8} {n0:>8} {nx:>8}")


if __name__ == "__main__":
    main()
