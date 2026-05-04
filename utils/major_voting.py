#!/usr/bin/env python3
"""
Majority voting for pass@k outputs.

This script scans a directory of per-question subfolders (each containing
`program_output_*.csv` files), groups equivalent outputs, and copies the
majority result for each question into a flat output directory as
`<question_id>.csv`.
"""

import argparse
import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser

def is_valid_timestamp_format(date_string):
    """
    Uses dateutil to automatically detect and parse almost any date string.
    Returns the datetime object if successful, else False.
    """
    try:
        # automatically detects format and precision.
        value = parser.parse(date_string)        
        return value.replace(tzinfo=None)
        
    except (ValueError, parser.ParserError, TypeError, OverflowError):
        # parser.parse raises ParserError for invalid strings
        # ValueError might be raised for out-of-bounds math
        return False

def jaccard_similarity(str1, str2):
    # Convert strings to sets of words
    a = set(str1.split()) 
    b = set(str2.split())
    
    # Intersection / Union
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    
    return intersection / union

def _ws_norm(s: str) -> str:
    """Lowercase and collapse whitespace — basic text normalization."""
    return ' '.join(s.lower().split())


_LIST_SINGLE_SINGLE = re.compile(r"^\['(.*)'\]$", re.DOTALL)
_LIST_SINGLE_DOUBLE = re.compile(r'^\["(.*)"\]$', re.DOTALL)


def _normalize_cell(val: Any) -> Any:
    """Normalize a single cell value to remove Snowflake serialization artifacts.

    Applied in order:
      1. Unwrap single-element Python list repr: ['x'] / ["x"] → x
      2. Strip surrounding literal double-quotes: "x" → x

    Whitespace is intentionally NOT stripped — leading/trailing spaces can be
    meaningful in TO_CHAR-formatted values that must match specific gold variants.
    """
    if not isinstance(val, str):
        return val

    # 1. Unwrap Python list repr with a single element
    m = _LIST_SINGLE_SINGLE.match(val) or _LIST_SINGLE_DOUBLE.match(val)
    if m:
        val = m.group(1)
        return val

    # 2. Strip surrounding literal double-quotes (only when no inner quotes)
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        inner = val[1:-1]
        if '"' not in inner:
            val = inner

    return val


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with all cell values normalized."""
    return df.map(_normalize_cell)


def _sort_key(x):
    """Sort key: None first, then numbers by value, then strings."""
    if x is None:
        return (0,)
    if isinstance(x, (int, float)):
        return (1, float(x))
    return (2, str(x))


def compare_tables(pred: pd.DataFrame, gold: pd.DataFrame, ignore_order: bool, fuzzy_threshold: float = 0.0) -> bool:
    # Quick rejection: different row counts cannot be equivalent.
    if len(pred) != len(gold):
        return False

    # Drop all-NaN columns to avoid wildcard matching
    pred = pred.dropna(axis=1, how="all")
    gold = gold.dropna(axis=1, how="all")

    # Guard: 0-column DataFrames after NaN drop must not vacuously match
    if pred.shape[1] == 0 or gold.shape[1] == 0:
        return pred.shape[1] == 0 and gold.shape[1] == 0

    tolerance = 1e-2

    def prep(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return _ws_norm(value)
        return value

    def vectors_match(v1, v2, tol: float, ignore_order_: bool) -> bool:
        v1 = [prep(x) for x in v1]
        v2 = [prep(x) for x in v2]
        if ignore_order_:
            v1 = sorted(v1, key=_sort_key)
            v2 = sorted(v2, key=_sort_key)

        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False

            if isinstance(a, str) and isinstance(b, str):
                a_timestamp = is_valid_timestamp_format(a)
                b_timestamp = is_valid_timestamp_format(b)
                if a_timestamp != b_timestamp:
                    return False

            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                if (fuzzy_threshold > 0
                        and isinstance(a, str) and isinstance(b, str)
                        and len(a) >= 4 and len(b) >= 4
                        and SequenceMatcher(None, a, b).ratio() >= fuzzy_threshold):
                    continue
                return False
        return True

    t_gold_list = gold.transpose().values.tolist()
    t_pred_list = pred.transpose().values.tolist()
    for gold_col in t_gold_list:
        if not any(vectors_match(gold_col, pred_col, tolerance, ignore_order) for pred_col in t_pred_list):
            return False
    return True


def tables_equivalent(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    ignore_row_order: bool,
    allow_superset: bool,
    fuzzy_threshold: float = 0.0,
) -> bool:
    if allow_superset:
        return (compare_tables(df_a, df_b, ignore_row_order, fuzzy_threshold)
                or compare_tables(df_b, df_a, ignore_row_order, fuzzy_threshold))
    return (compare_tables(df_a, df_b, ignore_row_order, fuzzy_threshold)
            and compare_tables(df_b, df_a, ignore_row_order, fuzzy_threshold))


def read_csv_safe(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty"
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


def select_majority_group(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select winning group. Tiebreaker: prefer more columns, then lexicographic."""
    best_group: Optional[Dict[str, Any]] = None
    best_size = 0
    best_ncols = 0
    best_first: Optional[str] = None
    for group in groups:
        size = len(group["files"])
        ncols = len(group["df"].columns)
        first = sorted(group["files"])[0].name if group["files"] else ""
        if (size > best_size
            or (size == best_size and ncols > best_ncols)
            or (size == best_size and ncols == best_ncols
                and (best_first is None or first < best_first))):
            best_group = group
            best_size = size
            best_ncols = ncols
            best_first = first
    return best_group or {"files": []}


def _program_stem(csv_file: Path) -> str:
    """Strip the first ``_output`` infix from the CSV stem.

    Mapping: ``program_output_1_0.csv`` → ``program_1_0``.
    """
    return csv_file.stem.replace("_output", "", 1)


def _program_priority(csv_file: Path) -> int:
    """0 = SQL source exists, 1 = Python source exists, 2 = neither. Lower is better."""
    stem = _program_stem(csv_file)
    parent = csv_file.parent
    if (parent / (stem + ".sql")).exists():
        return 0
    if (parent / (stem + ".py")).exists():
        return 1
    return 2


def _has_program_file(csv_file: Path, filename_base: str) -> bool:
    """Return True if a corresponding .sql or .py program file exists next to the CSV."""
    return _program_priority(csv_file) < 2


def process_subdir(
    subdir: Path,
    output_dir: Path,
    filename_base: str,
    tol: float,
    ignore_row_order: bool,
    include_empty: bool,
    allow_superset: bool,
    fuzzy_threshold: float = 0.0,
    require_program: bool = False,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not subdir.is_dir():
        return None
    base_id = subdir.name
    csv_files = sorted(
        [p for p in subdir.iterdir() if p.is_file() and p.name.startswith(filename_base) and p.suffix == ".csv"]
    )

    orphaned: List[str] = []
    if require_program:
        orphaned = [p.name for p in csv_files if not _has_program_file(p, filename_base)]
        csv_files = [p for p in csv_files if _has_program_file(p, filename_base)]

    if not csv_files:
        return None

    groups: List[Dict[str, Any]] = []
    errors: List[str] = []
    skipped: List[str] = []

    for csv_file in csv_files:
        df, err = read_csv_safe(csv_file)
        if df is None:
            errors.append(f"{csv_file.name}: {err}")
            continue
        if df.empty and not include_empty:
            skipped.append(csv_file.name)
            continue
        df = normalize_df(df)
        matched = False
        for group in groups:
            if tables_equivalent(
                df,
                group["df"],
                ignore_row_order=ignore_row_order,
                allow_superset=allow_superset,
                fuzzy_threshold=fuzzy_threshold,
            ):
                # Keep the widest df as the group's representative for
                # tables_equivalent comparison and inter-group tiebreaking.
                if len(df.columns) > len(group["df"].columns):
                    group["df"] = df

                group["files"].append(csv_file)
                group["dfs"][csv_file] = df
                matched = True
                break
        if not matched:
            groups.append({"df": df, "files": [csv_file], "dfs": {csv_file: df}})

    # Pick chosen within each group by program-source priority
    # (SQL > Py > none), then more columns, then alphabetical name.
    # The chosen file's df is exported, ensuring chosen_file matches the output CSV.
    for group in groups:
        chosen = min(
            group["files"],
            key=lambda p: (_program_priority(p), -len(group["dfs"][p].columns), p.name),
        )
        group["chosen"] = chosen
        group["chosen_df"] = group["dfs"][chosen]

    if not groups:
        return base_id, {
            "status": "no_valid_outputs",
            "total_files": len(csv_files),
            "errors": errors,
            "skipped": skipped,
            "orphaned": orphaned,
        }

    best_group = select_majority_group(groups)
    chosen_file = best_group["chosen"]
    output_path = output_dir / f"{base_id}.csv"
    best_group["chosen_df"].to_csv(output_path, index=False)

    return base_id, {
        "status": "ok",
        "chosen_file": chosen_file.name,
        "vote_count": len(best_group["files"]),
        "total_considered": sum(len(group["files"]) for group in groups),
        "total_files": len(csv_files),
        "groups": {str(i): [p.name for p in group["files"]] for i, group in enumerate(groups)},
        "errors": errors,
        "skipped": skipped,
        "orphaned": orphaned,
    }


def majority_vote(
    input_dir: Path,
    output_dir: Path,
    filename_base: str,
    tol: float,
    ignore_row_order: bool,
    include_empty: bool,
    allow_superset: bool,
    max_workers: int,
    fuzzy_threshold: float = 0.0,
    require_program: bool = False,
) -> Dict[str, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Any]] = {}

    subdirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not subdirs:
        return summary

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_subdir,
                subdir,
                output_dir,
                filename_base,
                tol,
                ignore_row_order,
                include_empty,
                allow_superset,
                fuzzy_threshold,
                require_program,
            )
            for subdir in subdirs
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            base_id, record = result
            summary[base_id] = record

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose majority outputs from pass@k samples and export one CSV per question."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with per-question subfolders containing program_output_*.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for majority-voted CSVs. Defaults to <input_dir>_majority.",
    )
    parser.add_argument(
        "--filename_base",
        type=str,
        default="program_output",
        help="Base filename to match inside each subfolder.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Absolute tolerance for numeric equivalence.",
    )
    parser.add_argument(
        "--respect_row_order",
        action="store_true",
        help="Respect row order (do not sort values within columns).",
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Include empty CSVs as valid candidates.",
    )
    parser.add_argument(
        "--allow_superset",
        action="store_true",
        help="Treat outputs as equivalent if one table is a column superset of the other.",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.0,
        help="SequenceMatcher ratio threshold for fuzzy string matching (0 = disabled, recommended: 0.90).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="Maximum number of worker processes.",
    )
    parser.add_argument(
        "--require_program",
        action="store_true",
        help="Ignore CSV files that have no corresponding .sql or .py program file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = Path(f"{input_dir}_majority").resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    summary = majority_vote(
        input_dir=input_dir,
        output_dir=output_dir,
        filename_base=args.filename_base,
        tol=args.tolerance,
        ignore_row_order=not args.respect_row_order,
        include_empty=args.include_empty,
        allow_superset=args.allow_superset,
        max_workers=args.max_workers,
        fuzzy_threshold=args.fuzzy_threshold,
        require_program=args.require_program,
    )

    summary_path = output_dir / "majority_vote_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total_ok = sum(1 for v in summary.values() if v.get("status") == "ok")
    total_total = len(summary)
    print(f"Majority voting complete: {total_ok}/{total_total} questions exported.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

