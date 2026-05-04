#!/usr/bin/env python3

import argparse
import json
import pathlib
from typing import List, Dict, Set, Any, Tuple

import sqlglot
import sqlglot.expressions as exp

from attr_analyzer import (
    get_schema,
    get_involved_attrs,
)


# --- --- --- --- --- --- --- --- --- --- ---
#      NEW HELPER & CORRECTED get_assets
# --- --- --- --- --- --- --- --- --- --- ---

def normalize_node_name(node) -> str:
    """
    Recursively normalizes a sqlglot node (Identifier, Dot, etc.)
    based on Snowflake's quoting rules.
    """
    if isinstance(node, exp.Identifier):
        # This is the base case (e.g., "RESULTS" or "year_points")
        if not node.quoted:
            return node.name.upper()
        return node.name  # Quoted, return as-is

    if isinstance(node, exp.Dot):
        # Recursively build the dotted name (e.g., "F1.F1.RESULTS")
        left = normalize_node_name(node.this)
        right = normalize_node_name(node.expression)
        return f"{left}.{right}"

    # Fallback for the 'str' object error or other simple nodes
    if isinstance(node, str):
        return node.upper()

    # Last resort, e.g. for nodes that aren't part of a name
    return node.sql().upper()


def get_assets(sql: str) -> Tuple[Set[str], Set[str]]:
    """
    Parses a single SQL query and returns two sets: tables and columns.
    
    This version correctly normalizes all identifiers, handles aliases, 
    and filters out Common Table Expressions (CTEs) from the table list.
    """
    tables: Set[str] = set()
    columns: Set[str] = set()
    try:
        parsed = sqlglot.parse_one(sql, dialect="snowflake")
        if not parsed:
            return tables, columns
    except Exception:
        return tables, columns

    # 1. Find all defined CTE names
    cte_names: Set[str] = set()
    for cte in parsed.find_all(exp.CTE):
        # cte.this is the Identifier for the CTE name
        cte_names.add(normalize_node_name(cte.this))

    # 2. Extract all table references
    all_table_references: Set[str] = set()
    for table in parsed.find_all(exp.Table):
        # 'table.this' is the source node (e.g., F1.F1.RESULTS or year_points)
        # This correctly ignores the alias (table.name)
        all_table_references.add(normalize_node_name(table.this))

    # 3. The "true" tables are all table references *except* the CTEs
    tables = all_table_references - cte_names

    # 4. Extract column aliases and the *original* columns they are aliasing
    aliases: Set[str] = set()
    aliased_columns: Set[str] = set()

    for alias in parsed.find_all(exp.Alias):
        if isinstance(alias.this, exp.Table):
            continue  # This is a table alias, skip it

        # 'alias.alias' is the Identifier for the alias name
        aliases.add(normalize_node_name(alias.alias))

        # Find all *original* columns inside the aliased expression
        for col in alias.this.find_all(exp.Column):
            # 'col.this' is the Identifier for the original column
            aliased_columns.add(normalize_node_name(col.this))

    # 5. Extract all column identifiers referenced *anywhere* in the query
    all_column_references: Set[str] = set()
    for col in parsed.find_all(exp.Column):
        # 'col.this' is the Identifier
        all_column_references.add(normalize_node_name(col.this))

    # 6. The "true" columns are:
    #    (All column references) - (The column aliases) | (The original aliased columns)
    columns = (all_column_references - aliases) | aliased_columns

    return tables, columns


# --- --- --- --- --- --- --- --- --- --- ---
#      NEW UTILITIES FOR ATTR ANALYZER API
# --- --- --- --- --- --- --- --- --- --- ---

def extract_assets_from_involved_attrs(
    involved_attrs: Dict[str, Set[str]]
) -> Tuple[Set[str], Set[str]]:
    """
    Converts the mapping returned by get_involved_attrs into the
    table and column sets expected by the metrics calculation.
    """
    tables: Set[str] = set()
    columns: Set[str] = set()

    if not involved_attrs:
        return tables, columns

    for table_name, column_names in involved_attrs.items():
        normalized_table = table_name.upper()
        tables.add(normalized_table)

        for column_name in column_names or []:
            normalized_column = column_name.upper()
            columns.add(f"{normalized_table}.{normalized_column}")

    return tables, columns


# --- --- --- --- --- --- --- --- --- --- ---
#      UNCHANGED FUNCTIONS
# --- --- --- --- --- --- --- --- --- --- ---

def calculate_metrics_for_set(gold_set: Set[str], component_set: Set[str]) -> Dict[str, Any]:
    """
    Calculates precision/recall metrics for a single asset class (tables or columns).
    (Unchanged)
    """

    true_positives_set = gold_set.intersection(component_set)
    false_positives_set = component_set.difference(gold_set)
    false_negatives_set = gold_set.difference(component_set)

    tp_count = len(true_positives_set)
    fp_count = len(false_positives_set)
    fn_count = len(false_negatives_set)

    precision_denominator = tp_count + fp_count
    precision = (tp_count / precision_denominator) if precision_denominator > 0 else 1.0

    recall_denominator = tp_count + fn_count
    recall = (tp_count / recall_denominator) if recall_denominator > 0 else 1.0

    return {
        "metrics": {
            "precision": precision,
            "recall": recall
        },
        "counts": {
            "gold_assets_total": recall_denominator,
            "component_assets_total": precision_denominator,
            "true_positives": tp_count,
            "false_positives (found_but_not_needed)": fp_count,
            "false_negatives (missed_but_needed)": fn_count
        },
        "details": {
            "gold_assets (total_needed)": sorted(list(gold_set)),
            "component_assets (total_found)": sorted(list(component_set)),
            "true_positives (found_and_needed)": sorted(list(true_positives_set)),
            "false_positives (found_but_not_needed)": sorted(list(false_positives_set)),
            "false_negatives (missed_but_needed)": sorted(list(false_negatives_set))
        }
    }
import re

def normalize_sql(sql_query):
    """
    Removes SQL comments and normalizes whitespace.
    
    Handles:
    1. Block comments: /* ... */ (including multi-line)
    2. Line comments: -- ...
    3. Normalizes all whitespace (spaces, tabs, newlines) to a single space.
    """
    
    # 1. Remove block comments (/* ... */)
    # The re.DOTALL flag makes '.' match newline characters,
    # and '?' makes the '.*' non-greedy.
    query = re.sub(r"/\*.*?\*/", "", sql_query, flags=re.DOTALL)
    
    # 2. Remove line comments (-- ...)
    # This matches '--' and all characters until the end of the line.
    query = re.sub(r"--.*", "", query)
    
    # 3. Normalize whitespace
    # Replace multiple whitespace characters (spaces, tabs, newlines) 
    # with a single space and remove leading/trailing whitespace.
    query = re.sub(r'\"', '', query)
    query = re.sub(r"\s+", " ", query).strip()

    return query

def analyze_query_coverage(id: str, gold_query: str, component_queries: List[str]) -> Dict[str, Any]:
    """
    Analyzes table and column coverage separately and returns a nested report.
    (Unchanged)
    """

    schema = get_schema(id)

    # 1. Get assets from the gold query
    # gold_tables, gold_columns = get_assets(gold_query)
    #
    # if not gold_tables and not gold_columns:
    #     try:
    #         sqlglot.parse_one(gold_query, dialect="snowflake")
    #     except Exception:
    #         return {"error": "Gold query could not be parsed."}
    #     if not gold_query.strip():
    #         return {"error": "Gold query file is empty."}

    try:
        gold_involved_attrs = get_involved_attrs(schema, gold_query)
        gold_tables, gold_columns = extract_assets_from_involved_attrs(gold_involved_attrs)
        # if id == 'sf_bq012':
        #     print(gold_involved_attrs)
        #     print("--------------------------------")
        #     print(gold_tables, gold_columns)
        #     print("--------------------------------")
        #     print(gold_query)
        #     exit()
    except Exception:
        return {"error": "Gold query could not be parsed."}

    # 2. Get all assets from the component queries
    component_tables: Set[str] = set()
    component_columns: Set[str] = set()

    component_query_parsed = False
    for q in component_queries:
        if not q.strip():
            continue

        try:
            involved_attrs = get_involved_attrs(schema, q)
            ct, cc = extract_assets_from_involved_attrs(involved_attrs)
            component_query_parsed = True
            if len(ct) == 0: 
                raise ValueError("Can't parse tables from generated subqueries")
            elif len(cc) == 0:
                raise ValueError("Can't parser columns from generated subqueries")
        except Exception as e:
            print("ERROR:", str(e))
            print(f"Failed to parse query '{q}'")
            print(involved_attrs)
            print(normalize_sql(q))
            print(gold_involved_attrs)
            print(gold_query)
            print("--------------------------------")
            continue

        component_tables.update(ct)
        component_columns.update(cc)

        # if not component_query_parsed:
        #     try:
        #         sqlglot.parse_one(q, dialect="snowflake")
        #         component_query_parsed = True
        #     except Exception:
        #         pass

    if not component_query_parsed and any(q.strip() for q in component_queries):
        return {"error": "All component queries failed to parse."}

    # 3. Calculate metrics for each asset type
    table_report = calculate_metrics_for_set(gold_tables, component_tables)
    column_report = calculate_metrics_for_set(gold_columns, component_columns)

    return {
        "table_level_report": table_report,
        "column_level_report": column_report
    }


def process_all_queries(gold_dir: str, component_dir: str) -> Dict[str, Any]:
    """
    Finds matching gold queries and sub-queries, runs coverage analysis,
    and returns a consolidated report including average metrics.
    (Unchanged)
    """
    gold_path = pathlib.Path(gold_dir)
    component_path = pathlib.Path(component_dir)

    instance_reports = {}

    table_precision_scores = []
    table_recall_scores = []
    column_precision_scores = []
    column_recall_scores = []

    if not gold_path.is_dir():
        return {"error": f"Gold query directory not found: {gold_dir}"}
    if not component_path.is_dir():
        return {"error": f"Component query directory not found: {component_dir}"}
    
    num_fully_covered_queries = 0

    for component_instance_dir in component_path.iterdir():
        if not component_instance_dir.is_dir():
            continue

        instance_id = component_instance_dir.name
        sub_query_file = component_instance_dir / "sub_queries.json"
        gold_file = gold_path / f"{instance_id}.sql"

        if not sub_query_file.exists():
            instance_reports[instance_id] = {
                "error": "Instance directory exists but sub_queries.json is missing."
            }
            continue

        if not gold_file.exists():
            instance_reports[instance_id] = {
                "error": "sub_queries.json exists but corresponding gold .sql file is missing."
            }
            continue

        try:
            gold_query_sql = gold_file.read_text()
        except Exception as e:
            instance_reports[instance_id] = {"error": f"Failed to read gold query: {e}"}
            continue

        try:
            component_data = json.loads(sub_query_file.read_text())
            component_query_list = list(component_data.values())
        except Exception as e:
            instance_reports[instance_id] = {"error": f"Failed to read/parse sub_queries.json: {e}"}
            continue

        report = analyze_query_coverage(instance_id, gold_query_sql, component_query_list)
        instance_reports[instance_id] = report

        if "table_level_report" in report:
            table_precision_scores.append(report["table_level_report"]["metrics"]["precision"])
            table_recall_scores.append(report["table_level_report"]["metrics"]["recall"])
            column_precision_scores.append(report["column_level_report"]["metrics"]["precision"])
            column_recall_scores.append(report["column_level_report"]["metrics"]["recall"])

            if report["column_level_report"]["metrics"]["recall"] == 1:
                num_fully_covered_queries += 1

    total_successful = len(table_precision_scores)
    total_errors = len(instance_reports) - total_successful

    avg_table_precision = (sum(table_precision_scores) / total_successful) if total_successful > 0 else 0.0
    avg_table_recall = (sum(table_recall_scores) / total_successful) if total_successful > 0 else 0.0
    avg_column_precision = (sum(column_precision_scores) / total_successful) if total_successful > 0 else 0.0
    avg_column_recall = (sum(column_recall_scores) / total_successful) if total_successful > 0 else 0.0

    final_report = {
        "overall_metrics": {
            "average_table_precision": avg_table_precision,
            "average_table_recall": avg_table_recall,
            "average_column_precision": avg_column_precision,
            "average_column_recall": avg_column_recall,
            "total_instances_processed_successfully": total_successful,
            "total_instances_with_errors": total_errors,
            "num_fully_covered_queries": num_fully_covered_queries
        },
        "instance_reports": instance_reports
    }

    return final_report


def main():
    """
    Main function to parse command-line arguments and run the analysis.
    (Unchanged)
    """
    parser = argparse.ArgumentParser(
        description="Analyze SQL query coverage between component queries and a gold standard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 analyze_coverage.py ./all_gold_sql ./all_component_folders
"""
    )

    parser.add_argument(
        "--gold_dir",
        type=str,
        help="Path to the directory containing gold standard <instance_id>.sql files."
    )

    parser.add_argument(
        "--component_dir",
        type=str,
        help="Path to the directory containing <instance_id>/sub_queries.json folders."
    )

    args = parser.parse_args()

    results = process_all_queries(args.gold_dir, args.component_dir)

    json.dump(results, open("tmp.json", "w"), indent=2)


if __name__ == "__main__":
    main()
