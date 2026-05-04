#!/usr/bin/env python3
"""
Analyze table names in spider2-snow databases to find similar prefixes and suffixes.
Also checks that similar tables have the same column names and types.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

def normalize_schema(column_names: List[str], column_types: List[str]) -> frozenset:
    """
    Normalize a schema to ignore column order.
    Returns a frozenset of (column_name, column_type) tuples.
    Two schemas are considered the same if they have the same set of (name, type) pairs.
    """
    if len(column_names) != len(column_types):
        # Shouldn't happen, but handle gracefully
        return frozenset()
    
    # Create a set of (column_name, column_type) pairs
    # This ignores the order of columns
    return frozenset(zip(column_names, column_types))

def extract_prefixes_suffixes(table_names: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Extract all meaningful prefixes and suffixes from table names.
    Returns two dicts: prefix -> set of full names, suffix -> set of full names
    """
    prefix_groups = defaultdict(set)
    suffix_groups = defaultdict(set)
    
    for name in table_names:
        parts = name.split('_')
        
        # Case 1: Handle names with underscores (original logic)
        if len(parts) >= 2:
            # Generate all prefixes (from start, including single part if there are multiple parts)
            for i in range(1, len(parts) + 1):
                prefix = '_'.join(parts[:i])
                prefix_groups[prefix].add(name)
            
            # Generate all suffixes (from end, including single part if there are multiple parts)
            for i in range(1, len(parts) + 1):
                suffix = '_'.join(parts[-i:])
                suffix_groups[suffix].add(name)
        
        # Case 2: Handle names without underscores - split by character and numerical parts
        # Pattern: characters followed by numbers (e.g., "GSOD2007" -> prefix: "GSOD", suffix: "2007")
        # or numbers followed by characters (e.g., "2007GSOD" -> prefix: "2007", suffix: "GSOD")
        else:
            # Match pattern: one or more letters followed by one or more digits
            match_chars_then_nums = re.match(r'^([A-Za-z]+)(\d+)$', name)
            if match_chars_then_nums:
                char_part = match_chars_then_nums.group(1)
                num_part = match_chars_then_nums.group(2)
                prefix_groups[char_part].add(name)
                suffix_groups[num_part].add(name)
            else:
                # Match pattern: one or more digits followed by one or more letters
                match_nums_then_chars = re.match(r'^(\d+)([A-Za-z]+)$', name)
                if match_nums_then_chars:
                    num_part = match_nums_then_chars.group(1)
                    char_part = match_nums_then_chars.group(2)
                    prefix_groups[num_part].add(name)
                    suffix_groups[char_part].add(name)
    
    return prefix_groups, suffix_groups

def find_maximal_prefixes(prefix_groups: Dict[str, Set[str]], 
                          table_schemas: Dict[str, Tuple[List[str], List[str]]]) -> Dict[str, List[str]]:
    """
    Find maximal prefixes with exact schema matching.
    Returns: prefix -> list of table names (sorted) that share the prefix AND have identical schemas
    """
    # Sort by length (longest first)
    sorted_prefixes = sorted(prefix_groups.items(), key=lambda x: (-len(x[0].split('_')), x[0]))
    
    result = {}
    covered_tables = set()
    
    for prefix, tables in sorted_prefixes:
        if len(tables) < 2:
            continue
        
        # Group tables by schema - tables with identical schemas get grouped together
        schema_groups = defaultdict(list)
        for table_name in tables:
            if table_name in table_schemas:
                schema = table_schemas[table_name]
                column_names = schema[0]
                column_types = schema[1]
                
                # Normalize schema to ignore column order
                schema_key = normalize_schema(column_names, column_types)
                schema_groups[schema_key].append(table_name)
        
        # Only keep schema groups with at least 2 tables (tables with matching schemas)
        for schema_key, matching_tables in schema_groups.items():
            if len(matching_tables) >= 2:
                # Sort table names alphabetically
                matching_tables_sorted = sorted(matching_tables)
                # Check if this adds new tables
                new_tables = set(matching_tables_sorted) - covered_tables
                
                # Only add if it has at least 2 tables, and either:
                # - Adds new tables, OR
                # - Has significantly more tables than already covered
                if len(new_tables) >= 2 or (len(matching_tables_sorted) >= 3 and 
                                            len(matching_tables_sorted) > len(covered_tables & set(matching_tables_sorted))):
                    key = f"{prefix}__{len(matching_tables_sorted)}"
                    result[key] = matching_tables_sorted
                    covered_tables.update(matching_tables_sorted)
    
    return result

def find_maximal_suffixes(suffix_groups: Dict[str, Set[str]], 
                          table_schemas: Dict[str, Tuple[List[str], List[str]]]) -> Dict[str, List[str]]:
    """Find maximal suffixes with exact schema matching."""
    sorted_suffixes = sorted(suffix_groups.items(), key=lambda x: (-len(x[0].split('_')), x[0]))
    
    result = {}
    covered_tables = set()
    
    for suffix, tables in sorted_suffixes:
        if len(tables) < 2:
            continue
        
        # Group tables by schema - tables with identical schemas get grouped together
        schema_groups = defaultdict(list)
        for table_name in tables:
            if table_name in table_schemas:
                schema = table_schemas[table_name]
                column_names = schema[0]
                column_types = schema[1]
                
                # Normalize schema to ignore column order
                schema_key = normalize_schema(column_names, column_types)
                schema_groups[schema_key].append(table_name)
        
        # Only keep schema groups with at least 2 tables (tables with matching schemas)
        for schema_key, matching_tables in schema_groups.items():
            if len(matching_tables) >= 2:
                # Sort table names alphabetically
                matching_tables_sorted = sorted(matching_tables)
                new_tables = set(matching_tables_sorted) - covered_tables
                
                if len(new_tables) >= 2 or (len(matching_tables_sorted) >= 3 and 
                                            len(matching_tables_sorted) > len(covered_tables & set(matching_tables_sorted))):
                    key = f"{suffix}__{len(matching_tables_sorted)}"
                    result[key] = matching_tables_sorted
                    covered_tables.update(matching_tables_sorted)
    
    return result

def analyze_table_similarities(base_path: str) -> Dict[str, Dict[str, List]]:
    """
    Analyze all databases and find tables with similar prefixes/suffixes.
    Only returns tables that have matching prefixes/suffixes AND identical schemas.
    
    Args:
        base_path: Path to databases folder
    
    Returns:
        Dict mapping database_name -> similarity_type -> list of matches
    """
    results = defaultdict(lambda: defaultdict(list))
    base = Path(base_path)
    
    if not base.exists():
        print(f"Error: Path {base_path} does not exist", file=sys.stderr)
        return results
    
    # Iterate through each database folder
    for db_folder in sorted(base.iterdir()):
        if not db_folder.is_dir():
            continue
        
        db_name = db_folder.name
        print(f"Analyzing database: {db_name}", file=sys.stderr)
        
        # Collect all tables in this database with their schemas
        table_map = {}  # table_name -> table_fullname
        table_schemas = {}  # table_name -> (column_names, column_types)
        
        # Walk through schema folders
        for schema_folder in sorted(db_folder.iterdir()):
            if not schema_folder.is_dir():
                continue
            
            # Find all JSON files in this schema
            for json_file in sorted(schema_folder.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        table_name = data.get("table_name", "")
                        table_fullname = data.get("table_fullname", "")
                        column_names = data.get("column_names", [])
                        column_types = data.get("column_types", [])
                        
                        if table_name and table_fullname and column_names and column_types:
                            # Extract just the table name part (without schema prefix)
                            simple_name = table_name.split('.')[-1]
                            table_map[simple_name] = table_fullname
                            table_schemas[simple_name] = (column_names, column_types)
                except Exception as e:
                    print(f"  Warning: Could not read {json_file}: {e}", file=sys.stderr)
                    continue
        
        if len(table_map) < 2:
            continue
        
        table_names = list(table_map.keys())
        
        # Extract prefixes and suffixes
        prefix_groups, suffix_groups = extract_prefixes_suffixes(table_names)
        
        # Find maximal prefixes with exact schema matching
        maximal_prefixes = find_maximal_prefixes(prefix_groups, table_schemas)
        for key, names in sorted(maximal_prefixes.items(), key=lambda x: (-len(x[1]), x[0])):
            # Extract the prefix from the key (format: "prefix__count")
            prefix = key.rsplit('__', 1)[0]
            
            # Get fullnames and schemas
            table_info = []
            for name in names:
                fullname = table_map[name]
                schema = table_schemas[name]
                table_info.append((fullname, schema))
            
            results[db_name]["prefix"].append((prefix, table_info))
        
        # Find maximal suffixes with exact schema matching
        maximal_suffixes = find_maximal_suffixes(suffix_groups, table_schemas)
        for key, names in sorted(maximal_suffixes.items(), key=lambda x: (-len(x[1]), x[0])):
            # Extract the suffix from the key (format: "suffix__count")
            suffix = key.rsplit('__', 1)[0]
            
            # Get fullnames and schemas
            table_info = []
            for name in names:
                fullname = table_map[name]
                schema = table_schemas[name]
                table_info.append((fullname, schema))
            
            results[db_name]["suffix"].append((suffix, table_info))
    
    return results

def print_results(results: Dict[str, Dict[str, List]], output_file=None):
    """Print the analysis results in a readable format."""
    output = output_file if output_file else sys.stdout
    
    databases_with_similarities = [db for db, res in results.items() if any(res.values())]
    
    if not databases_with_similarities:
        print("No table similarities found.", file=output)
        return
    
    print(f"Found similarities in {len(databases_with_similarities)} databases\n", file=output)
    print("=" * 80, file=output)
    print("TABLES WITH MATCHING PREFIXES OR SUFFIXES AND IDENTICAL SCHEMAS", file=output)
    print("=" * 80, file=output)
    print()
    
    for db_name in sorted(databases_with_similarities):
        db_results = results[db_name]
        
        print(f"\n{'='*80}", file=output)
        print(f"Database: {db_name}", file=output)
        print(f"{'='*80}", file=output)
        
        # Print prefix similarities with exact schema matching
        if "prefix" in db_results and db_results["prefix"]:
            print(f"\n[PREFIX SIMILARITIES WITH EXACT SCHEMA MATCH]", file=output)
            for prefix, table_info in db_results["prefix"]:
                print(f"  Prefix: '{prefix}' ({len(table_info)} tables with identical schema)", file=output)
                
                # Show schema info (from first table, since they all match)
                if table_info:
                    first_fullname, first_schema = table_info[0]
                    column_names, column_types = first_schema
                    print(f"    Schema: {len(column_names)} columns", file=output)
                    print(f"    Columns: {', '.join(f'{name}({type_})' for name, type_ in zip(column_names[:5], column_types[:5]))}", end="", file=output)
                    if len(column_names) > 5:
                        print(f" ... ({len(column_names) - 5} more)", file=output)
                    else:
                        print("", file=output)
                
                for fullname, schema in table_info:
                    print(f"    - {fullname}", file=output)
        
        # Print suffix similarities with exact schema matching
        if "suffix" in db_results and db_results["suffix"]:
            print(f"\n[SUFFIX SIMILARITIES WITH EXACT SCHEMA MATCH]", file=output)
            for suffix, table_info in db_results["suffix"]:
                print(f"  Suffix: '{suffix}' ({len(table_info)} tables with identical schema)", file=output)
                
                # Show schema info (from first table, since they all match)
                if table_info:
                    first_fullname, first_schema = table_info[0]
                    column_names, column_types = first_schema
                    print(f"    Schema: {len(column_names)} columns", file=output)
                    print(f"    Columns: {', '.join(f'{name}({type_})' for name, type_ in zip(column_names[:5], column_types[:5]))}", end="", file=output)
                    if len(column_names) > 5:
                        print(f" ... ({len(column_names) - 5} more)", file=output)
                    else:
                        print("", file=output)
                
                for fullname, schema in table_info:
                    print(f"    - {fullname}", file=output)

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(
        description="Find tables across each database that share a prefix/suffix and have identical schemas.",
    )
    parser.add_argument(
        "--metadata-root",
        default=os.path.join(project_root, "datasets", "Spider2", "spider2-snow", "resource", "databases_no_nulls_2"),
        help="Root folder containing per-database metadata (output of remove_null_columns.py).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "table_similarities_report.json"),
        help="Path to write the similarity report JSON.",
    )
    args = parser.parse_args()

    print(
        "Analyzing table similarities (prefix/suffix matching with exact schema matching) across all databases...",
        file=sys.stderr,
    )
    results = analyze_table_similarities(args.metadata_root)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nFull report saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
