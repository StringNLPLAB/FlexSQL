#!/usr/bin/env python3
"""Recover chosen_file in majority_vote_summary.json so it points to a CSV
whose corresponding program (.sql preferred over .py) actually exists on disk.

For each entry whose current chosen_file has no sibling .sql/.py program,
look for another member of the same vote group that does and switch
chosen_file to it. SQL beats Python; ties broken alphabetically.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Optional

OUTPUT_RE = re.compile(r"program_output_(\d+)_(\d+)\.csv")


def _program_paths(parent: Path, csv_name: str) -> tuple[Path, Path]:
    m = OUTPUT_RE.match(csv_name)
    if not m:
        raise ValueError(f"unexpected CSV name: {csv_name}")
    x, y = m.group(1), m.group(2)
    return parent / f"program_{x}_{y}.sql", parent / f"program_{x}_{y}.py"


def _program_priority(parent: Path, csv_name: str) -> int:
    """0 = SQL exists, 1 = Python exists, 2 = neither. Lower is better."""
    sql, py = _program_paths(parent, csv_name)
    if sql.exists():
        return 0
    if py.exists():
        return 1
    return 2


def _find_substitute(parent: Path, group_members: list[str], current: str) -> Optional[str]:
    candidates = [m for m in group_members
                  if m != current and _program_priority(parent, m) < 2]
    if not candidates:
        return None
    candidates.sort(key=lambda m: (_program_priority(parent, m), m))
    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference_dir", type=Path, required=True,
                   help="Per-question root, e.g. inference_res/<run_id>/")
    p.add_argument("--summary", type=Path, default=None,
                   help="Path to majority_vote_summary.json "
                        "(default: <inference_dir>/major_vote/majority_vote_summary.json)")
    p.add_argument("--majority_dir", type=Path, default=None,
                   help="Directory holding per-question majority CSVs "
                        "(default: <inference_dir>/major_vote/)")
    p.add_argument("--copy_csv", action="store_true",
                   help="Re-copy the new chosen CSV over <majority_dir>/<question>.csv after switching.")
    p.add_argument("--dry_run", action="store_true", help="Report only; do not write changes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inference_dir = args.inference_dir.expanduser().resolve()
    summary_path = (args.summary or inference_dir / "major_vote" / "majority_vote_summary.json").resolve()
    majority_dir = (args.majority_dir or inference_dir / "major_vote").resolve()

    if not summary_path.exists():
        raise SystemExit(f"Summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text())

    fixed: list[tuple[str, str, str]] = []  # (qid, old, new)
    lone: list[str] = []
    none_in_group: list[str] = []
    already_ok = 0

    for qid, info in summary.items():
        chosen = info.get("chosen_file")
        if not chosen:
            continue
        qdir = inference_dir / qid
        if _program_priority(qdir, chosen) < 2:
            already_ok += 1
            continue
        group_members = next(
            (m for m in info.get("groups", {}).values() if chosen in m),
            None,
        )
        if group_members is None or len(group_members) <= 1:
            lone.append(qid)
            continue
        sub = _find_substitute(qdir, group_members, chosen)
        if sub is None:
            none_in_group.append(qid)
            continue
        fixed.append((qid, chosen, sub))

    print(f"Already OK: {already_ok}")
    print(f"Fixable (will switch chosen_file): {len(fixed)}")
    print(f"Unfixable - lone group: {len(lone)}")
    print(f"Unfixable - no member of the group has a program: {len(none_in_group)}")

    if fixed:
        print("\n-- Switches --")
        for qid, old, new in fixed:
            old_kind = "py" if (inference_dir / qid / old.replace("_output", "", 1).replace(".csv", ".py")).exists() else "?"
            new_kind = "sql" if _program_priority(inference_dir / qid, new) == 0 else "py"
            print(f"  {qid}: {old} ({old_kind}) -> {new} ({new_kind})")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    if not fixed:
        return

    backup = summary_path.with_suffix(summary_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(summary_path, backup)
        print(f"\nBackup written: {backup}")

    for qid, _, new in fixed:
        summary[qid]["chosen_file"] = new
        if args.copy_csv:
            src = inference_dir / qid / new
            dst = majority_dir / f"{qid}.csv"
            if src.exists():
                shutil.copy2(src, dst)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Updated summary: {summary_path}")
    if args.copy_csv:
        print(f"Refreshed per-question CSVs in: {majority_dir}")


if __name__ == "__main__":
    main()
