"""
Planning module for generating natural language plans to answer database queries.
"""

import gc
import json
import random
import logging
from typing import List, Dict, Tuple, Optional, Callable
from utils import get_db_id, get_question_str
import os
from get_ddl import load_table_similarities
from chat import Chat
from schema_linking import (
    match_table_name_flexible, parse_columns_response,
    compose_select_query, execute_and_format_query_result
)
from get_ddl import post_format_generated_query
from prompt_templates import (
    NO_DIVERSE_PLANNING_SINGLE_PROMPT_TEMPLATE,
)
import sqlite3
import snowflake.connector

MAX_RETRIES = 3

def _parse_plan_candidates(response_text: str, beam_size: int, logger: logging.Logger):
    """
    Parse candidate steps with scores from model response.
    
    Expected formats (flexible parsing):
    1. "1. Step text (score: 0.9)"
    2. "Candidate 1: Step text\nScore: 0.8"
    3. "Step text | Score: 0.7"
    4. JSON format: {"step": "...", "score": 0.9}
    
    Args:
        response_text: Raw response text from model
        beam_size: Expected number of candidates
    
    Returns:
        List of dicts with 'text' and 'score' keys
    """
    candidates = []
    
    # Try to parse JSON format first
    # First, try parsing the entire response_text directly (in case it's already clean JSON)
    try:
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, list):
            logger.debug(f"Successfully parsed JSON directly from response_text")
            return parsed[:beam_size]
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"Direct parse failed: {e}, trying to extract JSON array...")
    
    # If direct parse fails, look for JSON array in the text
    # Find the first '[' and then find its matching ']' by counting brackets
    try:
        start_idx = response_text.find('[')
        if start_idx == -1:
            logger.warning("No opening bracket '[' found in response text")
            return []
        
        # Find matching closing bracket by counting
        bracket_count = 0
        end_idx = -1
        for i in range(start_idx, len(response_text)):
            if response_text[i] == '[':
                bracket_count += 1
            elif response_text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx == -1:
            logger.warning("No matching closing bracket ']' found for JSON array")
            return []
        
        json_str = response_text[start_idx:end_idx]
        logger.debug(f"Extracted JSON array, length: {len(json_str)} characters")
        
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            logger.debug(f"Successfully parsed JSON array with {len(parsed)} items")
            return parsed[:beam_size]
        else:
            logger.warning(f"Parsed JSON is not a list, got type: {type(parsed)}")
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e.msg} at line {e.lineno}, column {e.colno}")
        # Log a snippet around the error location for debugging
        error_pos = e.pos if hasattr(e, 'pos') else None
        if error_pos and error_pos < len(response_text):
            start_snippet = max(0, error_pos - 100)
            end_snippet = min(len(response_text), error_pos + 100)
            logger.debug(f"JSON error context: ...{response_text[start_snippet:end_snippet]}...")
        else:
            logger.debug(f"Failed response text: {response_text[:]}...")
    except Exception as e:
        logger.warning(f"Unexpected error parsing response text: {type(e).__name__}: {e}")
        logger.debug(f"Failed response text: {response_text[:]}...")
    
    return []


def _revise_plans_with_invalid_tables(agent, question: dict, plans_to_revise: List[Tuple[int, str, List[str], List[str]]],
                                      valid_table_names: set, logger: logging.Logger,
                                      db_type: str, db_path: str) -> List[Tuple[float, str, List[str]]]:
    """
    Revise plans that contain invalid table names by sending correction messages to the model.
    
    Args:
        agent: Chat agent instance
        question: Question dictionary
        plans_to_revise: List of tuples (plan_idx, plan_text, tables, invalid_tables)
        valid_table_names: Set of valid table names from DDL dictionary
        logger: Logger instance
        db_type: Database type
        db_path: Path to database folder
        cursor_getter: Optional cursor getter for enabling metadata tools
        similarities_path: Optional path for grouping similar tables
        cursor_getter: Optional cursor getter for enabling metadata tools
    
    Returns:
        List of revised plans as tuples (score, plan_text, tables)
    """
    revised_plans = []
    
    for plan_idx, plan_text, tables, invalid_tables in plans_to_revise:
        try:
            logger.info(f"Revising plan {plan_idx + 1} with invalid tables: {invalid_tables}")
            
            # Create list of valid table names for the prompt
            valid_tables_list = sorted(list(valid_table_names))
            
            revision_prompt = f"""The following plan contains invalid table names that do not exist in the database schema.

### Original Plan:
{plan_text}

### Tables listed in the plan:
{', '.join(tables)}

### Invalid table names found:
{', '.join(invalid_tables)}

### Valid table names in the database schema:
{', '.join(valid_tables_list)}


### User Question:
{get_question_str(question, db_type)}

### Your Task:
Please revise the plan by replacing the invalid table names with the correct table names from the valid table names list above. 
Make sure to:
1. Use the EXACT table names from the valid table names list (case-sensitive)
2. Keep the same plan structure and logic
3. Only change the table names, not the plan content
4. Format your response as a JSON object with "text" (the revised plan), "tables" (list of corrected table names under the format "DATABASE.SCHEMA.TABLE"), and "score" (a probability score) keys
5. Wrap the JSON object in a ```json``` code block

### Revised Plan:
"""
            
            # Define example structure for validation
            example_structure = {
                "text": "step 1... step 2...",
                "tables": ["DATABASE.SCHEMA.TABLE_1", "DATABASE.SCHEMA.TABLE_2"],
                "score": 0.9
            }
            
            # Get revised plan from model
            response = agent.get_code_blocks(
                revision_prompt,
                code_format="json",
                logger=logger,
                example_json_structure=example_structure
            )
            response_blocks = response["code_blocks"]
            
            if not response_blocks:
                logger.warning(f"No revision blocks found for plan {plan_idx + 1}, keeping original plan")
                # Use original plan with a lower score
                revised_plans.append((0.5, plan_text, tables))
                continue
            
            # Parse revised plan
            try:
                revision_result = json.loads(response_blocks[0])
                if not isinstance(revision_result, dict):
                    logger.warning(f"Revision response for plan {plan_idx + 1} is not a dictionary, keeping original plan")
                    revised_plans.append((0.5, plan_text, tables))
                    continue
                
                revised_plan_text = revision_result.get('text', '').strip()
                revised_tables = revision_result.get('tables', [])
                revised_score = revision_result.get('score', 0.5)
                
                if not revised_plan_text:
                    logger.warning(f"Revised plan {plan_idx + 1} has no text, keeping original plan")
                    revised_plans.append((0.5, plan_text, tables))
                    continue
                
                # Validate revised tables
                still_invalid = []
                for table in revised_tables:
                    if table.lower() not in {name.lower() for name in valid_table_names}:
                        still_invalid.append(table)
                
                if still_invalid:
                    logger.warning(f"Revised plan {plan_idx + 1} still has invalid tables: {still_invalid}, keeping original plan")
                    revised_plans.append((0.5, plan_text, tables))
                else:
                    logger.info(f"Successfully revised plan {plan_idx + 1}")
                    revised_plans.append((revised_score, revised_plan_text, revised_tables))
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse revision JSON for plan {plan_idx + 1}: {e}, keeping original plan")
                revised_plans.append((0.5, plan_text, tables))
                
        except Exception as e:
            logger.error(f"Error revising plan {plan_idx + 1}: {e}", exc_info=True)
            # Keep original plan on error
            revised_plans.append((0.5, plan_text, tables))
    
    return revised_plans


def _get_table_overview_str(
    database_name: Optional[str],
    db_type: str,
    db_path: Optional[str],
    logger: logging.Logger,
    table_names: Optional[List[str]] = None,
    similarities_path: Optional[str] = None
) -> str:
    if not database_name or not db_path:
        return "No table metadata available."
    
    table_stats: Dict[str, int] = {}
    try:
        if db_type == "snowflake":
            base_folder = os.path.join(db_path, "resource", "databases_no_nulls_2", database_name)
            if not os.path.exists(base_folder):
                logger.warning(f"Database folder not found for table overview: {base_folder}")
                return "No table metadata available."
            
            for schema_name in sorted(os.listdir(base_folder)):
                schema_path = os.path.join(base_folder, schema_name)
                if not os.path.isdir(schema_path):
                    continue
                
                for file in os.listdir(schema_path):
                    if not file.endswith(".json") or file.endswith("_M-Schema.json"):
                        continue
                    json_path = os.path.join(schema_path, file)
                    try:
                        with open(json_path, "r") as f:
                            metadata = json.load(f)
                        table_fullname = metadata.get(
                            "table_fullname",
                            f"{database_name}.{schema_name}.{file.replace('.json', '')}"
                        )
                        column_count = len(metadata.get("column_names", []))
                        table_stats[table_fullname] = column_count
                    except Exception:
                        continue
        else:
            base_folder = os.path.join(db_path, database_name)
            if not os.path.exists(base_folder):
                logger.warning(f"Database folder not found for table overview: {base_folder}")
                return "No table metadata available."
            
            for file in os.listdir(base_folder):
                if not file.endswith(".json") or file.endswith("_M-Schema.json"):
                    continue
                json_path = os.path.join(base_folder, file)
                try:
                    with open(json_path, "r") as f:
                        metadata = json.load(f)
                    table_name = metadata.get("table_name", file.replace(".json", ""))
                    column_count = len(metadata.get("column_names", []))
                    table_stats[table_name] = column_count
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Error building table overview: {str(e)}")
        return "No table metadata available."
    
    if not table_stats:
        return "No table metadata available."
    
    entries = []
    lower_stats = {name.lower(): count for name, count in table_stats.items()}
    if table_names:
        ordered_tables = list(table_names)
    else:
        ordered_tables = sorted(table_stats.keys())
    random.shuffle(ordered_tables)
    
    similar_tables_map = load_table_similarities(similarities_path) if similarities_path else {}
    processed_tables = set()
    
    for name in ordered_tables:
        name_key = name.lower()
        if name_key in processed_tables:
            continue
        
        count = lower_stats.get(name_key)
        if count is None:
            entries.append(f"- {name}: unknown columns")
        else:
            entries.append(f"- {name}: {count} columns")
        
        similar_tables = similar_tables_map.get(name, [])
        similar_tables_in_list = [
            t for t in similar_tables
            if t.lower() in lower_stats and t.lower() not in processed_tables
        ]
        if similar_tables_in_list:
            similar_tables_str = ", ".join(similar_tables_in_list)
            entries.append(f"- tables with similar structure to {name}: {similar_tables_str}")
            for similar_table in similar_tables_in_list:
                processed_tables.add(similar_table.lower())
        
        processed_tables.add(name_key)
    
    return "\n".join(entries)


def planning_batch_generate(agent: Chat, question: dict, table_names: List[str], logger: logging.Logger,
                   top_k: int = 5, table_similarities_path: str = None, db_type: str = "snowflake", db_path: str = None,
                   cursor_getter: Optional[Callable] = None, batch_size: int = 4) -> Tuple[List[Tuple[str, List[str]]], str]:
    """
    Generate multiple complete plans all at once (batch mode) - Plan Generation Step.
    This function focuses on generating diverse plans without value grounding.
    Validates table names against the provided DDL dictionary and revises plans if needed.
    
    Args:
        agent: Chat agent instance
        question: Question dictionary with 'instruction' key
        table_names: List of table names available in the database
        logger: Logger instance for logging activities
        top_k: Number of top plans to return
        table_similarities_path: Path to the table_similarities_report JSON file
        db_type: Database type (snowflake, sqlite, etc.)
        db_path: Path to database folder
    
    Returns:
        List of top k plan texts (strings)
    """
    logger.info(f"Starting batch plan generation with top_k={top_k}")
    
    agent.enable_tools(
        tool_names = ["list_columns"],
        db_path=db_path,
        db_type=db_type,
        cursor_getter=cursor_getter,
        database_name=get_db_id(question),
    )
    logger.info(f"Tool calling enabled for planning generation: {agent.tool_calling_enabled}, tool functions: {agent.tool_functions}")


    database_name = get_db_id(question)
    table_overview_str = _get_table_overview_str(
        database_name=database_name,
        db_type=db_type,
        db_path=db_path,
        logger=logger,
        table_names=list(table_names),
        similarities_path=table_similarities_path
    )


    # Ensure planning generation cannot access data-query tools.
    for tool_name in ("query_database", "get_distinct_values", "search_dimension_values"):
        if tool_name in agent.tool_functions:
            agent.tool_functions.pop(tool_name, None)

    valid_table_names = set(table_names)

    num_batches = top_k // batch_size
    assert num_batches * batch_size == top_k, "top_k must be divisible by batch_size = {batch_size}"

    agent.set_system_prompt(
        f"You are a helpful assistant. For each user question, generate a set of {batch_size} possible plans to answer the user question."
        f"Format your plans in a JSON array of {batch_size} objects, each object containing a \"text\" key "
        f"containing the plan, a \"tables\" key containing the tables used in the plan, and a \"score\" key containing the probability of the plan. "
        f"Please sample at random from the full distribution of possible plans in your imagination. "
        f"CRITICAL: If the User Question is ambiguous or could map to multiple parts of the DB Schema, "
        f"use the {batch_size} plans to cover these different valid interpretations. "
        f"Do not just rephrase the same plans; propose semantically different valid plans based on the schema."
    )

    if db_type == "snowflake":
        _batch_table_name_note = ' Please use the FULL TABLE NAMES under the format "DATABASE.SCHEMA.TABLE".'
        _batch_wildcard_note = ' If some tables within one plan are sharing the same prefix or suffix, please use a wildcard representation to represent those table names.'
        _batch_example_tables = '"DATABASE.SCHEMA.TABLE_1", "DATABASE.SCHEMA.TABLE_2"'
    else:
        _batch_table_name_note = ''
        _batch_wildcard_note = ''
        _batch_example_tables = '"TABLE_1", "TABLE_2"'

    prompt_template = """

    ### Your Task:
    You are given a user question and a table overview. Your task is to analyze the user's intentions to resolve any ambiguity and generate {batch_size} high-level natural language plans to extract the data.

    **Tools**: Use the following tools to explore the database structure, if necessary:
   - `list_columns(table_name)`: Lists all columns in a specific table

    **Step 1: 5W1H Intent Analysis**
    Analyze the user question using this framework. Focus strictly on mapping user terms to the provided schema:
    
    1.  **What (Entities):** Identify the nouns in the request and map them to specific tables or columns. Note if a noun could map to multiple different columns.
    2.  **Who (Subjects):** Identify the primary data subject or entity ID being queried (e.g., a specific user, account, or item ID).
    3.  **Where (Scope):** Identify any filtering conditions, categorization requirements, or specific data sources within the schema.
    4.  **When (Time):** Identify any time-based constraints. Determine if the timeframe is explicit (specific dates) or relative (needs definition).
    5.  **Why (Goal):** Determine the output goal. Is the user requesting a calculation/aggregation (analysis) or a direct retrieval of rows (lookup)?
    6.  **How (Logic):** Identify specific operations required to format the result, such as sorting order, record limits, or mathematical formulas.

    **Step 2: Plan Generation**
    Based on the analysis, create {batch_size} step-by-step plans. 
    * Each step of a plan must represent a **single**, distinct logical action.
    * **DO NOT use any SQL syntax**. Use natural language only.
    * If a critical element (like the specific column for a vague noun) is missing, flag the ambiguity.    
    * Don't include the analysis, just the plans. 
    * Each plan should be independent of each other, do not do cross-references between plans when writing them.
    * Format your plans in a JSON array of {batch_size} objects, each object containing a \"text\" key containing the plan, a \"tables\" key containing the tables used in the plan, and a \"score\" key containing the probability of the plan.{batch_table_name_note} Wrap the JSON array in a ```json``` code block.
    * You must specifically list all the tables used in each plan.{batch_wildcard_note}

    Example plans format:
    ```json
        [{{"text": "<content of plan 1>", "tables": [{batch_example_tables}, ...], "score": <a probability score>}}, ..., {{"text": "<content of plan {batch_size}>", "tables": [{batch_example_tables}, ...], "score": <a probability score>}}]
    ```

    ### Table Overview (table name -> number of columns):
    {table_overview}

    ### User Question:
    {question}
    ### Some External Knowledge that might be useful:
    {external_knowledge}

    ### Your Plans:
    """

    external_knowledge_summary = ""
    if question.get("external_knowledge", ""): # snowflake and sqlite of spider2-format
        external_knowledge_path = os.path.join(db_path, "resource/documents", question["external_knowledge"]) if db_type == "snowflake" else os.path.join(os.path.dirname(os.path.dirname(db_path)), "documents", question["external_knowledge"])
        if os.path.exists(external_knowledge_path):
            external_doc = open(external_knowledge_path, "r").read()
            
            summary_agent = Chat(model=agent.model, ip=agent.ip, port=agent.port, base_url=agent.base_url)
            
            # Create prompt to summarize external document
            summary_prompt = f"""You are given a user question, a table overview, and an external document. Your task is to summarize the external document, keeping only information that is relevant to answering the user question given the table overview.

Focus on:
- Information that helps understand the user question in the context of the table overview
- Definitions, relationships, or business rules that connect to the listed tables/columns
- Any domain-specific knowledge that clarifies what data means or how it should be interpreted

Exclude:
- Information unrelated to the user question or table overview
- Redundant or overly detailed information
- Information that doesn't help with answering the user question

Provide a concise summary

### Table Overview (table name -> number of columns):
{table_overview_str}

### User Question:
The current database system is {db_type}. {get_question_str(question, db_type)}

### External Document:
{external_doc}

### Summary of the external document (keep only relevant information):
"""
                
            try:
                logger.info("Summarizing external knowledge document...")
                summary_response = summary_agent.get_model_response_txt(summary_prompt)
                external_knowledge_summary = summary_response["text"].strip()
                logger.debug(f"External knowledge summary thoughts: {summary_response['thoughts']}")
                logger.debug(f"External knowledge summary: {external_knowledge_summary}")
                
                # Escape curly braces to prevent .format() from treating them as placeholders
                external_knowledge_summary = external_knowledge_summary.replace("{", "{{").replace("}", "}}")
                external_knowledge_summary = f"\n {external_knowledge_summary}\n"
            except Exception as e:
                logger.warning(f"Failed to summarize external knowledge document: {e}")
                external_knowledge_summary = ""
            gc.collect()

    if question.get('evidence', ""): # bird-format
        external_knowledge_summary = f"\n {question.get('evidence', '')}\n"

    # Define example structure for validation
    example_structure = [
        {"text": "step 1...", "tables": ["DATABASE.SCHEMA.TABLE_1", "DATABASE.SCHEMA.TABLE_2"], "score": 0.9}
    ]
    
    top_plans = []
    for mini_batch_idx in range(num_batches):
        logger.info(f"Generating {batch_size} plans for batch {mini_batch_idx + 1}/{num_batches}")
        prompt = prompt_template.format(
            question=get_question_str(question, db_type),
            table_overview=table_overview_str,
            batch_size=batch_size,
            external_knowledge=external_knowledge_summary,
            batch_table_name_note=_batch_table_name_note,
            batch_wildcard_note=_batch_wildcard_note,
            batch_example_tables=_batch_example_tables,
        )
        if mini_batch_idx ==0:
            logger.debug(f"Prompt: {prompt}")
        
        try:
            # Get response from model - generates all plans at once
            agent.messages = agent.messages[:1] # keep only the system prompt
            response = agent.get_code_blocks(
                prompt,
                code_format="json",
                logger=logger,
                example_json_structure=example_structure
            )
            response_blocks = response["code_blocks"]
            
            logger.debug(f"Planning thoughts: {response['thoughts']}")

            if not response_blocks:
                logger.warning("No plan blocks found in response")
                return [], external_knowledge_summary

            logger.debug(f"Planning response: {response_blocks[0]}")
            
            # Parse all plans from response
            candidates = _parse_plan_candidates(response_blocks[0], top_k, logger)
            
            if not candidates:
                logger.warning("No valid plans found in response")
                return [], external_knowledge_summary
            
            # Process all plans - each candidate should be a complete plan
            all_plans = []
            plans_to_revise = []
            for candidate_idx, candidate in enumerate(candidates):
                plan_text_raw = candidate.get('text', '')
                if isinstance(plan_text_raw, list):
                    plan_text_raw = "\n".join(str(s) for s in plan_text_raw)
                plan_text = str(plan_text_raw).strip()
                plan_score = candidate.get('score', 0.0)
                tables = candidate.get('tables', [])
                
                if not plan_text:
                    continue
                
                # Validate table names using flexible matching
                invalid_tables = []
                matched_tables_set = set()
                
                for table_pattern in tables:
                    # Try flexible matching first
                    matches = match_table_name_flexible(table_pattern, valid_table_names, logger)
                    
                    if matches:
                        # Pattern matched one or more tables
                        matched_tables_set.update(matches)
                        if len(matches) > 1:
                            logger.info(f"Plan {candidate_idx + 1}: Pattern '{table_pattern}' matched {len(matches)} tables: {matches[:5]}{'...' if len(matches) > 5 else ''}")
                    else:
                        # No match found even with flexible matching
                        invalid_tables.append(table_pattern)
                        logger.debug(f"Plan {candidate_idx + 1}: No match found for table pattern '{table_pattern}'")
                
                if invalid_tables:
                    # Store plan for revision
                    plans_to_revise.append((candidate_idx, plan_text, tables, invalid_tables))
                    logger.warning(f"Plan {candidate_idx + 1} has invalid table names (after flexible matching): {invalid_tables}. Valid table names: {valid_table_names}")
                else:
                    # All table patterns matched successfully
                    # Use the matched tables (expanded from patterns)
                    final_tables = list(matched_tables_set) if matched_tables_set else tables
                    all_plans.append((plan_score, plan_text, final_tables))
                    if matched_tables_set and len(matched_tables_set) > len(tables):
                        logger.debug(f"Plan {candidate_idx + 1}: Expanded {len(tables)} table patterns to {len(matched_tables_set)} matched tables")
            
            # Revise plans with invalid table names
            if plans_to_revise:
                logger.info(f"Revising {len(plans_to_revise)} plans with invalid table names")
                revised_plans = _revise_plans_with_invalid_tables(
                    agent, question, plans_to_revise, valid_table_names,
                    logger, db_type, db_path
                )
                
                # Add revised plans back
                for plan_score, plan_text, tables in revised_plans:
                    all_plans.append((plan_score, plan_text, tables))
            
            # Sort by score and return top k
            all_plans.sort(key=lambda x: x[0], reverse=True)
            top_plans.extend([(plan_text, plan_tables) for _, plan_text, plan_tables in all_plans[:batch_size]])
        
        except Exception as e:
            logger.error(f"Error generating batch plans: {e}", exc_info=True)
            continue
    
    logger.info(f"Batch plan generation completed. Generated {len(top_plans)} plans.")
    return top_plans, external_knowledge_summary


def _compose_queries_from_columns(
    plan_tables: Dict[str, List[str]],
    db_path: str,
    db_type: str,
    question: dict,
    cursor_getter: Optional[Callable],
    logger: logging.Logger,
    cache_dir: str = "./cache",
    agent: Optional[Chat] = None,
    table_names: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Compose SELECT queries directly from columns already identified by plan clarification.
    Handles wildcard table patterns by matching them against available tables.
    If there are errors, ask the model to fix column names.
    """
    from utils import get_q_id
    from get_ddl import list_tables

    if not plan_tables:
        logger.warning("No tables or columns provided for query composition")
        return {}

    q_id = get_q_id(question, db_type)

    if table_names is None:
        table_names = list_tables(
            db_folder=db_path,
            db_type=db_type,
            question_id=q_id,
            database_id=get_db_id(question),
            use_gold_tables=False,
        )

    available_tables = set(table_names)
    
    # Create database connection
    if db_type == "sqlite":
        _db_file = os.path.join(db_path, get_db_id(question), f"{get_db_id(question)}.sqlite")
        conn = sqlite3.connect(f"file:{_db_file}?mode=ro", uri=True)
    elif db_type == "snowflake":
        snowflake_credential = json.load(open(os.path.join(db_path, "snowflake_credential.json")))
        conn = snowflake.connector.connect(**snowflake_credential, database=get_db_id(question))
    else:
        logger.error(f"Unsupported db_type: {db_type}")
        return {}
    
    cursor = conn.cursor()
    all_queries = {}
    n_example_rows = 1
    
    # Compose and execute queries for each table pattern
    # plan_tables is a dict where keys are table patterns (may contain wildcards) and values are column lists
    for table_pattern, columns in plan_tables.items():
        if not columns:
            logger.info(f"No columns provided for table pattern '{table_pattern}'. Skipping.")
            continue
        
        # Match table pattern against available tables (handles wildcards)
        matched_tables = match_table_name_flexible(table_pattern, available_tables, logger)
        
        if not matched_tables:
            logger.warning(f"No tables matched for pattern '{table_pattern}'. Available tables: {sorted(available_tables)[:10]}...")
            continue
        
        if len(matched_tables) > 1:
            logger.info(f"Pattern '{table_pattern}' matched {len(matched_tables)} tables: {matched_tables}")
        
        # For each matched table, compose and execute query with the same columns
        for table_name in matched_tables:
            query = compose_select_query(table_name, columns)
            logger.info(f"COMPOSED QUERY for {table_name}: {query}")
            if not query:
                logger.info(f"Failed to build query for table '{table_name}'. Skipping.")
                continue
            
            query = post_format_generated_query(query, db_path=db_path, db_type=db_type, include_comment=False)
            logger.info(f"FORMATTED QUERY: {query}")
            
            # Execute query with retry logic
            out_of_retry = True
            for attempt in range(MAX_RETRIES):
                try:
                    execute_and_format_query_result(
                        cursor=cursor,
                        query=query,
                        db_path=db_path,
                        db_type=db_type,
                        n_example_rows=n_example_rows,
                        include_comment=False
                    )
                    all_queries[table_name] = query
                    out_of_retry = False
                    logger.info(f"Successfully executed query for table '{table_name}'. ✅")
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/{MAX_RETRIES} for '{table_name}' failed. Error: {str(e)}")
                    if "Query returned no results" in str(e):
                        logger.info(f"No rows returned for table '{table_name}'. Skipping retries.")
                        break
                    
                    # Ask model to fix column names if agent is available; only then do we retry (no break = loop continues)
                    if agent:
                        fix_prompt = (
                            f"The column list you provided resulted in an error for table {table_name}.\n"
                            f"**Error Message:** {str(e)}\n\n"
                            f"Please provide a corrected JSON object with the same format: {{\"{table_name}\": [\"col1\", \"col2\", ...]}}"
                        )
                        
                        fix_response = agent.get_code_blocks(fix_prompt, code_format="json")
                        fixed_columns = parse_columns_response(fix_response["code_blocks"][-1], [table_name], logger, allow_wildcards=False)
                        
                        if fixed_columns and fixed_columns.get(table_name):
                            columns = fixed_columns[table_name]
                            query = compose_select_query(table_name, columns)
                            if not query:
                                logger.error(f"Received empty column list for '{table_name}'. Breaking retry loop.")
                                break
                            query = post_format_generated_query(query, db_path=db_path, db_type=db_type, include_comment=False)
                            logger.info(f"Received corrected column list for '{table_name}'. Retrying...")
                            logger.info(f"Revised columns: {columns}")
                            # no break: loop continues with new query (retry)
                        else:
                            logger.error(f"Model failed to provide corrected columns for '{table_name}'. Will retry (attempt {attempt + 1}/{MAX_RETRIES}).")
                            # no break: loop continues so we ask the model again next attempt
                    else:
                        logger.warning(f"No agent provided, cannot fix column names. Breaking retry loop.")
                        break
            
            if out_of_retry:
                logger.error(f"Skipping query for table '{table_name}' after {MAX_RETRIES} failed attempts. ❌")
    
    # Cleanup
    if conn:
        cursor.close()
        conn.close()
    
    # Save queries to cache
    os.makedirs(os.path.join(cache_dir, q_id), exist_ok=True)
    json.dump(all_queries, open(os.path.join(cache_dir, q_id, "sub_sqls.json"), "w"), ensure_ascii=False, indent="\t")
    
    return all_queries


def plan_clarification(agent: Chat, question: dict, plans: List, logger: logging.Logger,
                      db_type: str = "snowflake", db_path: str = None,
                      cursor_getter: Optional[Callable] = None, external_knowledge_summary: str = None,
                      similarities_path: Optional[str] = None, cache_dir: str = "./cache",
                      use_gold_tables: bool = False, table_names: List[str] = None,
                      feedback_per_plan: Optional[Dict[int, str]] = None,
                      ) -> List:
    """
    Clarify ambiguous terms in generated plans by grounding them to actual database values,
    identify relevant columns, and generate sub-SQL queries for each table.
    
    Args:
        agent: Chat agent instance (must have tool calling enabled)
        question: Question dictionary with 'instruction' key
        plans: List of plans to clarify, where each plan is a tuple of (plan_text, plan_tables)
        logger: Logger instance for logging activities
        db_type: Database type (snowflake, sqlite, etc.)
        db_path: Path to database folder
        cursor_getter: Optional function to get database cursor
        external_knowledge_summary: Optional pre-generated summary from planning_batch_generate
        similarities_path: Optional path to table similarities file
        cache_dir: Directory for caching sub-SQL queries
        use_gold_tables: Whether to use gold tables
        table_names: List of table names available in the database
        feedback_per_plan: Optional dict mapping plan index to evaluation feedback string;
            when provided, that feedback is injected into the clarification prompt for re-runs.
    
    Returns:
        List of clarified plans with sub-SQL queries, where each plan is a tuple of:
        (plan_text, plan_tables, columns, sub_sqls):
        - plan_text: List of plan steps (sentences) or a string
        - plan_tables: List of table names used in the plan
        - columns: Dictionary mapping table names to lists of relevant column names
        - sub_sqls: Dictionary mapping table names to generated SQL queries
    """
    logger.info(f"Starting plan clarification (value grounding) for {len(plans)} plans")
    agent.enable_tools(
        tool_names = ["list_columns", "query_database", "get_distinct_values", "search_dimension_values"],
        db_path=db_path,
        db_type=db_type,
        cursor_getter=cursor_getter,
        database_name=get_db_id(question)
    )
    logger.info(f"Tool calling enabled for plan clarification: {agent.tool_calling_enabled}, tool functions: {agent.tool_functions}")

    if not plans:
        logger.warning("No plans provided for clarification")
        return []
    
    database_name = get_db_id(question)
    table_overview_str = _get_table_overview_str(
        database_name=database_name,
        db_type=db_type,
        db_path=db_path,
        logger=logger,
        similarities_path=similarities_path
    )
        
    agent.set_system_prompt(
        f"You are a Rigorous Data Architect. Your goal is to refine execution plans by grounding "
        f"ambiguous terms, but you must strictly avoid redundant verification for values that allow flexibility."
        f"\n\n"
        f"### CORE DIRECTIVE: SMART GROUNDING STRATEGY\n"
        f"1. **IDENTIFY VALUE TYPES:**\n"
        f"   - **Flexible Text (NO TOOLS):** If a plan step involves filtering by names, titles, descriptions, or high-cardinality string fields, assume the downstream implementation will use partial/fuzzy matching. **Do not verify these values.**\n"
        f"   - **Strict System Keys (TOOL REQUIRED):** If a plan step involves a specific domain classification, status code, type identifier, or foreign key that requires an exact string literal to match, you **MUST** use tools to find the precise value.\n"
        f"\n"
        f"2. **SELF-CRITICISM AND FEASIBILITY CHECK:**\n"
        f"   - Before refining the plan, critically examine each step to ensure it is **actually doable**.\n"
        f"   - Verify that all referenced tables exist in the database schema.\n"
        f"   - Verify that all referenced columns exist in their respective tables.\n"
        f"   - Check that the logical flow and dependencies between steps are sound.\n"
        f"   - Identify any missing prerequisites, invalid operations, or impossible conditions.\n"
        f"   - If a step is not feasible, you must either correct it using available tools or flag it for revision.\n"
    )

    prompt_template = """### Table Overview (table name -> number of columns)
        {table_overview}
        
        You may call the `list_columns(table_name)` tool to inspect column names and examples if needed.

        ### User Question
        {question}
        ### Some External Knowledge that might be useful:
        {external_knowledge}

        ### Proposed Execution Plan
        {plan}

        ### Task
        Refine the execution plan by determining which filter values require **exact database grounding** and which can safely rely on **flexible matching**.

        ### Step 1: Self-Criticism and Feasibility Check
        Before refining filter values, critically examine the entire plan to ensure every step is **actually doable**:
        
        1. **Table Verification:** For each table mentioned in the plan, verify it exists in the Table Overview. If a table is not listed, use `list_columns(table_name)` to check if it exists, or flag it as potentially incorrect.
        
        2. **Column Verification:** For each column referenced in filtering, joining, or selection operations, verify it exists in the corresponding table. Use `list_columns(table_name)` if needed to inspect the actual column names.
        
        3. **Logical Flow Check:** Verify that:
           - Steps can be executed in the order specified
           - Dependencies between steps are satisfied (e.g., if Step 2 uses results from Step 1, Step 1 must produce those results)
           - Join conditions reference valid columns from the joined tables
           - Filter conditions are logically sound and can be evaluated
        
        4. **Operation Feasibility:** Ensure that:
           - Aggregations are applied to appropriate data types
           - Comparisons are between compatible types
           - Required data exists or can be derived from available tables
        
        5. **Correction Protocol:** If you identify any infeasible steps:
           - Use tools to discover the correct table/column names
           - Revise the plan to use valid references
           - If correction is not possible with available information, note the issue but proceed with the best available refinement

        ### Step 2: Decision Procedure: Smart Grounding Filter
        After verifying feasibility, inspect every filtering condition in the plan and classify it into exactly one of the following categories:

        #### Category A — Flexible / Descriptive Values (NO TOOLS)
        - Targets free-form or high-cardinality text fields (e.g., names, titles, descriptions).
        - Minor string variation is acceptable and expected.
        - Downstream logic can apply partial or fuzzy matching.
        - **Action:** Leave unchanged. Do NOT verify or normalize.

        #### Category B — Strict / Structural Values (TOOLS REQUIRED)
        - Targets normalized fields such as:
        - status codes
        - type identifiers
        - enumerations
        - domain classifications
        - foreign keys or lookup-table values
        - Requires an exact literal match to function correctly.
        - **Action:** You have to use the appropriate tool to resolve the precise value.

        ### Step 3: Identify Relevant Columns
        After refining the plan, for each table mentioned in the plan, identify the **relevant column names** needed to answer the user's question. This includes:
        - Columns needed for filtering conditions
        - Columns needed for joining operations
        - Columns needed for selection/aggregation
        - Columns needed for sorting or grouping
        
        Use the `list_columns(table_name)` tool if needed to inspect the actual column names in each table.

        ### Execution Rules
        1. **First, perform the feasibility check (Step 1)** - verify tables, columns, and logical flow before proceeding.
        2. Resolve **only** Category B values.
        3. Describe resolved values in natural language (no SQL or code syntax). Each plan should be independent of each other, do not do cross-references between plans when writing them.
        4. If you corrected any infeasible steps during the feasibility check, incorporate those corrections into the refined plan.
        5. If no Category B values exist and no corrections are needed, return the plan unchanged.
        6. **List relevant columns for each table** as part of Step 3. Exactly following the format given under ### Output Format.
        7. Output **only** a JSON object with:
           - A `\"text\"` key containing the final refined plan
           - A `\"tables\"` key containing a dictionary mapping each table name{clarify_table_name_rule} to a list of relevant column names.

        8. Wrap the JSON object in a ```json``` code block. Don't say anything about category A or B in your response.{clarify_wildcard_rule}

        ### Output Format
        ```json
        {output_format}
        ```
    """ 

    if db_type == "snowflake":
        _clarify_table_name_rule = ' (use FULL TABLE NAMES under the format "DATABASE.SCHEMA.TABLE")'
        _clarify_wildcard_rule = (
            '\n        9. If some tables within one plan are sharing the same prefix or suffix, please use a wildcard'
            ' representation to represent those table names. For example, if the plan uses the tables "TABLE_1",'
            ' "DATABASE.SCHEMA.TABLE_2", and "DATABASE.SCHEMA.TABLE_3", and "DATABASE.SCHEMA.TABLE_1" and'
            ' "DATABASE.SCHEMA.TABLE_2" share the same prefix "DATABASE.SCHEMA.TABLE", you can use'
            ' "DATABASE.SCHEMA.TABLE_*" to represent both "DATABASE.SCHEMA.TABLE_1" and "DATABASE.SCHEMA.TABLE_2".'
        )
        output_format = """{{"text": "<refined execution plan>", "tables": {{"DATABASE.SCHEMA.TABLE_1": ["'col_1'", "'col_2'", ...], "DATABASE.SCHEMA.TABLE_2": ["'col_x'", "'col_y'", ...]}}}}"""
        output_format_variant = """{{"text": "<refined execution plan>", "tables": {{"DATABASE.SCHEMA.TABLE_1": ["'col_1':'key_1'", "'some_array_col'[0]:'key_2'", ...], "DATABASE.SCHEMA.TABLE_2": ["'col_x':'key_x'", "'some_other_array_key'[0]:'key_y'", ...]}}}}"""
    else:
        _clarify_table_name_rule = ''
        _clarify_wildcard_rule = ''
        output_format = """{{"text": "<refined execution plan>", "tables": {{"TABLE_1": ["col_1", "col_2", ...], "TABLE_2": ["col_x", "col_y", ...]}}}}"""
        output_format_variant = output_format
    
    # Process each plan individually
    clarified_plans = []
    for plan_idx, (plan, plan_tables) in enumerate(plans):
        try:
            logger.info(f"Clarifying plan {plan_idx + 1}/{len(plans)}")
            
            # Format plan for display
            if isinstance(plan, str):
                plan_str = plan
            elif isinstance(plan, list):
                plan_str = "\n".join([f"  {j+1}. {step}" for j, step in enumerate(plan)])
            else:
                plan_str = str(plan)
            
            # Check if any tables in this plan have VARIANT columns
            current_prompt_template = prompt_template
            has_variant = False
            
            # Build prompt for this single plan
            external_knowledge_str = external_knowledge_summary if external_knowledge_summary else ""
            prompt = current_prompt_template.format(
                question=get_question_str(question, db_type),
                table_overview=table_overview_str,
                plan=plan_str,
                external_knowledge=external_knowledge_str,
                output_format=output_format if not has_variant else output_format_variant,
                clarify_table_name_rule=_clarify_table_name_rule,
                clarify_wildcard_rule=_clarify_wildcard_rule,
            )
            if feedback_per_plan and plan_idx in feedback_per_plan:
                feedback_section = (
                    "\n\n### Previous attempt feedback\n"
                    "A program was generated from this plan and its output was evaluated. The evaluation found issues:\n\n"
                    f"{feedback_per_plan[plan_idx]}\n\n"
                    "Please refine the plan and sub-queries to address the above feedback.\n\n"
                )
                prompt = prompt.replace("        ### Task\n", feedback_section + "        ### Task\n")
            logger.debug(f"Clarification prompt for plan {plan_idx + 1}: {prompt}")
            
            # Clear chat history before each plan clarification to keep them independent
            agent.messages = agent.messages[:1] # keep only the system prompt
            
            # Get response from model - may involve tool calls
            response = agent.get_code_blocks(
                prompt,
                code_format="json",
                logger=logger
            )
            response_blocks = response["code_blocks"]
            
            logger.debug(f"Clarification thoughts for plan {plan_idx + 1}: {response['thoughts']}")
            logger.debug(f"Clarification response for plan {plan_idx + 1}: {response_blocks[0] if response_blocks else 'No blocks'}")
            
            if not response_blocks:
                logger.warning(f"No clarification blocks found for plan {plan_idx + 1}, using original plan")
                clarified_plans.append((plan, plan_tables, {}))
                continue

            # Parse clarification result
            try:
                clarification_result = json.loads(response_blocks[0])
                if not isinstance(clarification_result, dict):
                    logger.warning(f"Clarification response for plan {plan_idx + 1} is not a dictionary, using original plan")
                    clarified_plans.append((plan, plan_tables, {}))
                    continue
                
                plan_text = clarification_result.get('text', '')
                plan_tables_raw = clarification_result.get('tables', {})
                
                # Handle both formats: dict (new format) or list (old format for backward compatibility)
                if isinstance(plan_tables_raw, dict):
                    # New format: "tables" is a dict mapping table names to column lists
                    plan_tables = plan_tables_raw
                elif isinstance(plan_tables_raw, list):
                    # Old format: "tables" is a list, columns should be in separate "columns" field
                    plan_tables = clarification_result.get('columns', {})
                    if not plan_tables:
                        # Fallback: create empty dict with table names as keys
                        plan_tables = {table: [] for table in plan_tables_raw}
                else:
                    plan_tables = {}
                    logger.warning(f"Plan {plan_idx + 1} has unexpected 'tables' format: {type(plan_tables_raw)}")
                                   
                # Convert plan text to string and parse into steps
                plan_text_str = str(plan_text).strip()
                if not plan_text_str:
                    # Compose queries directly from columns already identified
                    sub_sqls = {}
                    if plan_tables:
                        try:
                            sub_sqls = _compose_queries_from_columns(
                                plan_tables=plan_tables,
                                db_path=db_path,
                                db_type=db_type,
                                question=question,
                                cursor_getter=cursor_getter,
                                logger=logger,
                                cache_dir=cache_dir,
                                agent=agent,
                                table_names=table_names,
                            )
                        except Exception as e:
                            logger.error(f"Error composing queries for plan {plan_idx + 1}: {e}", exc_info=True)
                    clarified_plans.append((plan, plan_tables, sub_sqls))
                    continue
                
                # Try to parse as JSON array first (in case model returns array format)
                try:
                    parsed = json.loads(plan_text_str)
                    if isinstance(parsed, list):
                        # It's a JSON array, use it as steps
                        steps = [str(step).strip() for step in parsed if str(step).strip()]
                    else:
                        # Not a list, treat as single string
                        steps = [plan_text_str]
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, treat as plain text - split by newlines or keep as single step
                    if '\n' in plan_text_str:
                        steps = [step.strip() for step in plan_text_str.split('\n') if step.strip()]
                    else:
                        steps = [plan_text_str]
                
                # Filter out empty steps
                cleaned_steps = [step.strip() for step in steps if step.strip()]
                
                # Compose queries directly from columns already identified by plan clarification
                sub_sqls = {}
                if cleaned_steps and plan_tables:
                    try:
                        sub_sqls = _compose_queries_from_columns(
                            plan_tables=plan_tables,
                            db_path=db_path,
                            db_type=db_type,
                            question=question,
                            cursor_getter=cursor_getter,
                            logger=logger,
                            agent=agent,
                        )
                    except Exception as e:
                        logger.error(f"Error composing queries for plan {plan_idx + 1}: {e}", exc_info=True)
                        sub_sqls = {}
                
                if cleaned_steps:
                    clarified_plans.append((cleaned_steps, plan_tables, sub_sqls))
                else:
                    # Fallback to original plan if clarified plan is empty
                    logger.warning(f"Clarified plan {plan_idx + 1} is empty, using original plan")
                    clarified_plans.append((plan, plan_tables, sub_sqls))
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse clarification JSON for plan {plan_idx + 1}: {e}, using original plan")
                clarified_plans.append((plan, plan_tables, {}))
                
        except Exception as e:
            logger.error(f"Error clarifying plan {plan_idx + 1}: {e}", exc_info=True)
            clarified_plans.append((plan, plan_tables, {}))  # Use original plan on error
    
    logger.info(f"Plan clarification completed. Clarified {len(clarified_plans)} plans.")
    return clarified_plans


def planning_no_diverse_generate(
    agent: Chat,
    question: dict,
    table_names: List[str],
    logger: logging.Logger,
    top_k: int = 5,
    table_similarities_path: Optional[str] = None,
    db_type: str = "snowflake",
    db_path: Optional[str] = None,
    cursor_getter: Optional[Callable] = None,
    cache_dir: str = "./cache",
) -> Tuple[List[Tuple], str]:
    """
    Generate top_k plans sequentially, one per agent invocation, each with full tool access.
    Ablation of planning_batch_generate + plan_clarification: skips batch generation and
    instead generates plans as independent passes with all 4 tools enabled from the start.

    Returns:
        Tuple of (plans, external_knowledge_summary) where plans is a list of 3-tuples
        (plan_text, plan_tables_dict, sub_sqls_dict) — same format as plan_clarification output.
    """
    logger.info(f"Starting no-diverse plan generation with top_k={top_k}")

    # --- Build table overview ---
    database_name = get_db_id(question)
    table_overview_str = _get_table_overview_str(
        database_name=database_name,
        db_type=db_type,
        db_path=db_path,
        logger=logger,
        table_names=list(table_names),
        similarities_path=table_similarities_path,
    )

    # --- Extract external knowledge summary (same logic as planning_batch_generate) ---
    external_knowledge_summary = ""
    if question.get("external_knowledge", ""):
        external_knowledge_path = (
            os.path.join(db_path, "resource/documents", question["external_knowledge"])
            if db_type == "snowflake"
            else os.path.join(os.path.dirname(os.path.dirname(db_path)), "documents", question["external_knowledge"])
        )
        if os.path.exists(external_knowledge_path):
            external_doc = open(external_knowledge_path, "r").read()
            summary_agent = Chat(model=agent.model, ip=agent.ip, port=agent.port, base_url=agent.base_url)
            summary_prompt = f"""You are given a user question, a table overview, and an external document. Your task is to summarize the external document, keeping only information that is relevant to answering the user question given the table overview.

Focus on:
- Information that helps understand the user question in the context of the table overview
- Definitions, relationships, or business rules that connect to the listed tables/columns
- Any domain-specific knowledge that clarifies what data means or how it should be interpreted

Exclude:
- Information unrelated to the user question or table overview
- Redundant or overly detailed information
- Information that doesn't help with answering the user question

Provide a concise summary

### Table Overview (table name -> number of columns):
{table_overview_str}

### User Question:
The current database system is {db_type}. {get_question_str(question, db_type)}

### External Document:
{external_doc}

### Summary of the external document (keep only relevant information):
"""
            try:
                logger.info("Summarizing external knowledge document...")
                summary_response = summary_agent.get_model_response_txt(summary_prompt)
                external_knowledge_summary = summary_response["text"].strip()
                external_knowledge_summary = external_knowledge_summary.replace("{", "{{").replace("}", "}}")
                external_knowledge_summary = f"\n {external_knowledge_summary}\n"
            except Exception as e:
                logger.warning(f"Failed to summarize external knowledge document: {e}")
                external_knowledge_summary = ""
            gc.collect()

    if question.get("evidence", ""):
        external_knowledge_summary = f"\n {question.get('evidence', '')}\n"

    # --- DB-type-specific format strings (same as plan_clarification) ---
    if db_type == "snowflake":
        _clarify_table_name_rule = ' (use FULL TABLE NAMES under the format "DATABASE.SCHEMA.TABLE")'
        _clarify_wildcard_rule = (
            '\n- If some tables share the same prefix or suffix, use a wildcard representation'
            ' (e.g., "DATABASE.SCHEMA.TABLE_*") to represent them.'
        )
        output_format = """{{"text": "<execution plan>", "tables": {{"DATABASE.SCHEMA.TABLE_1": ["'col_1'", "'col_2'", ...], "DATABASE.SCHEMA.TABLE_2": ["'col_x'", "'col_y'", ...]}}}}"""
    else:
        _clarify_table_name_rule = ""
        _clarify_wildcard_rule = ""
        output_format = """{{"text": "<execution plan>", "tables": {{"TABLE_1": ["col_1", "col_2", ...], "TABLE_2": ["col_x", "col_y", ...]}}}}"""

    # --- Set system prompt (same as plan_clarification) ---
    agent.set_system_prompt(
        f"You are a Rigorous Data Architect. Your goal is to refine execution plans by grounding "
        f"ambiguous terms, but you must strictly avoid redundant verification for values that allow flexibility."
        f"\n\n"
        f"### CORE DIRECTIVE: SMART GROUNDING STRATEGY\n"
        f"1. **IDENTIFY VALUE TYPES:**\n"
        f"   - **Flexible Text (NO TOOLS):** If a plan step involves filtering by names, titles, descriptions, or high-cardinality string fields, assume the downstream implementation will use partial/fuzzy matching. **Do not verify these values.**\n"
        f"   - **Strict System Keys (TOOL REQUIRED):** If a plan step involves a specific domain classification, status code, type identifier, or foreign key that requires an exact string literal to match, you **MUST** use tools to find the precise value.\n"
        f"\n"
        f"2. **SELF-CRITICISM AND FEASIBILITY CHECK:**\n"
        f"   - Before refining the plan, critically examine each step to ensure it is **actually doable**.\n"
        f"   - Verify that all referenced tables exist in the database schema.\n"
        f"   - Verify that all referenced columns exist in their respective tables.\n"
        f"   - Check that the logical flow and dependencies between steps are sound.\n"
        f"   - Identify any missing prerequisites, invalid operations, or impossible conditions.\n"
        f"   - If a step is not feasible, you must either correct it using available tools or flag it for revision.\n"
    )

    # --- Enable all tools once (same as plan_clarification) ---
    agent.enable_tools(
        tool_names=["list_columns", "query_database", "get_distinct_values", "search_dimension_values"],
        db_path=db_path,
        db_type=db_type,
        cursor_getter=cursor_getter,
        database_name=database_name,
    )
    logger.info(f"Tools enabled for no-diverse planning: {list(agent.tool_functions.keys())}")

    # --- Generate plans one by one ---
    results = []
    for plan_num in range(1, top_k + 1):
        logger.info(f"No-diverse planning: generating plan {plan_num}/{top_k}")
        try:
            # Reset to system prompt only — each pass is fully independent
            agent.messages = agent.messages[:1]

            prompt = NO_DIVERSE_PLANNING_SINGLE_PROMPT_TEMPLATE.format(
                table_overview=table_overview_str,
                question=get_question_str(question, db_type),
                external_knowledge=external_knowledge_summary if external_knowledge_summary else "",
                output_format=output_format,
                clarify_table_name_rule=_clarify_table_name_rule,
                clarify_wildcard_rule=_clarify_wildcard_rule,
            )
            if plan_num == 1:
                logger.debug(f"No-diverse planning prompt: {prompt}")

            response = agent.get_code_blocks(
                prompt,
                code_format="json",
                logger=logger,
            )
            response_blocks = response["code_blocks"]
            logger.debug(f"Plan {plan_num} thoughts: {response['thoughts']}")
            logger.debug(f"Plan {plan_num} response: {response_blocks[0] if response_blocks else 'No blocks'}")

            if not response_blocks:
                logger.warning(f"No code blocks for plan {plan_num}, appending empty fallback")
                results.append(([], {}, {}))
                continue

            # Parse response
            try:
                parsed = json.loads(response_blocks[0])
                if not isinstance(parsed, dict):
                    logger.warning(f"Plan {plan_num} response is not a dict, appending empty fallback")
                    results.append(([], {}, {}))
                    continue

                plan_text = parsed.get("text", "")
                plan_tables_raw = parsed.get("tables", {})

                # Handle both dict (new) and list (old) formats
                if isinstance(plan_tables_raw, dict):
                    plan_tables = plan_tables_raw
                elif isinstance(plan_tables_raw, list):
                    plan_tables = {table: [] for table in plan_tables_raw}
                else:
                    plan_tables = {}
                    logger.warning(f"Plan {plan_num} has unexpected 'tables' format: {type(plan_tables_raw)}")

                # Parse plan text into steps
                plan_text_str = str(plan_text).strip()
                try:
                    parsed_text = json.loads(plan_text_str)
                    if isinstance(parsed_text, list):
                        steps = [str(s).strip() for s in parsed_text if str(s).strip()]
                    else:
                        steps = [plan_text_str]
                except (json.JSONDecodeError, TypeError):
                    if "\n" in plan_text_str:
                        steps = [s.strip() for s in plan_text_str.split("\n") if s.strip()]
                    else:
                        steps = [plan_text_str]
                cleaned_steps = [s for s in steps if s]

                # Compose sub-SQL queries from identified columns
                sub_sqls = {}
                if plan_tables:
                    try:
                        sub_sqls = _compose_queries_from_columns(
                            plan_tables=plan_tables,
                            db_path=db_path,
                            db_type=db_type,
                            question=question,
                            cursor_getter=cursor_getter,
                            logger=logger,
                            cache_dir=cache_dir,
                            agent=agent,
                            table_names=table_names,
                        )
                    except Exception as e:
                        logger.error(f"Error composing queries for plan {plan_num}: {e}", exc_info=True)

                results.append((cleaned_steps if cleaned_steps else plan_text, plan_tables, sub_sqls))

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for plan {plan_num}: {e}, appending empty fallback")
                results.append(([], {}, {}))

        except Exception as e:
            logger.error(f"Error generating plan {plan_num}: {e}", exc_info=True)
            results.append(([], {}, {}))

    logger.info(f"No-diverse planning completed. Generated {len(results)} plans.")
    return results, external_knowledge_summary


if __name__ == "__main__":
    """
    Standalone script to run batch planning offline and collect multiple plans per question.
    Saves plans to a JSON file with question_id as keys and lists of plans as values.
    """
    import argparse
    from chat import Chat
    from logger import initialize_logger
    
    parser = argparse.ArgumentParser(description="Offline batch planning to collect multiple plans")
    parser.add_argument('--model', type=str, required=True, help="Model name or path")
    parser.add_argument('--dataset', type=str, required=True, help="Path to question JSONL file")
    parser.add_argument('--db_path', type=str, required=True, help="Path to database folder")
    parser.add_argument('--db_type', type=str, default="snowflake", choices=["snowflake", "sqlite"], help="Database type")
    parser.add_argument('--ip', type=str, default=None, help="IP for OpenAI-compatible server")
    parser.add_argument('--port', type=str, default=None, help="Port for OpenAI-compatible server")
    parser.add_argument('--base_url', type=str, default=None, help="Base URL for OpenAI API")
    parser.add_argument('--top_k', type=int, default=5, help="Number of plans to generate per batch")
    parser.add_argument('--num_batches', type=int, default=10, help="Number of batch planning calls per question")
    parser.add_argument('--output', type=str, default="collected_plans.json", help="Output JSON file path")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for log files")
    parser.add_argument('--similarities_path', type=str, default=None, help="path to the table_similarities_report JSON file")
    parser.add_argument('--use_gold_tables', action="store_true", help="use gold tables to generate program")
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize logger
    logger = initialize_logger(log_path=os.path.join(args.log_dir, "planning_collection.log"))
    logger.info(f"Starting offline batch planning collection")
    logger.info(f"Model: {args.model}, Dataset: {args.dataset}, DB Type: {args.db_type}")
    logger.info(f"Generating {args.top_k} plans per batch, {args.num_batches} batches per question")
    
    # Initialize agent
    # agent = Chat(
    #     args.model,
    #     base_url=args.base_url,
    #     ip=args.ip,
    #     port=args.port,
    # )

    # cursor_getter = None
    # import snowflake.connector

    # if args.db_type == "snowflake":
    #     snowflake_credential = json.load(open(os.path.join(args.db_path, "snowflake_credential.json")))
    #     conn = snowflake.connector.connect(**snowflake_credential)
    #     cursor_getter = lambda: conn.cursor()

    # # Enable tool calling
    # agent.enable_tools(
    #     ["query_database", "get_distinct_values", "search_dimension_values"],
    #     db_path=args.db_path,
    #     db_type=args.db_type,
    #     cursor_getter=cursor_getter,
    #     n_example_rows=100
    # )

    # # Load questions
    # logger.info(f"Loading questions from {args.dataset}")
    # questions = []
    # with open(args.dataset, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         try:
    #             json_str = line[line.find('{'):] if '{' in line else line
    #             question = json.loads(json_str)
    #             questions.append(question)
    #         except (json.JSONDecodeError, ValueError) as e:
    #             logger.warning(f"Failed to parse line: {line.strip()}, error: {e}")
    #             continue
    
    # logger.info(f"Loaded {len(questions)} questions")
    
    # # Collect plans for each question
    # all_collected_plans = {}
    
    # for idx, question in enumerate(questions):
    #     q_id = get_q_id(question, args.db_type)
    #     logger.info(f"Processing question {idx+1}/{len(questions)}: {q_id}")
        
    #     try:
    #         # Extract DDL
    #         table_ddls = extract_ddl(
    #             get_db_id(question), 
    #             db_folder=args.db_path, 
    #             db_type=args.db_type, 
    #             question_id=q_id,
    #             database_id=get_db_id(question),
    #             use_gold_tables=args.use_gold_tables,
    #             similarities_path=args.similarities_path
    #         )
    #         all_ddls_str = "\n".join([value for value in table_ddls.values()])
            
    #         # Collect plans from multiple batches
    #         collected_plans_for_question = []
            
    #         for batch_idx in range(args.num_batches):
    #             logger.info(f"  Batch {batch_idx+1}/{args.num_batches} for question {q_id}")
                
    #             # Clear agent history for each batch to get diverse plans
    #             agent.init_messages()
                
    #             # Generate plans
    #             plans = planning_batch(
    #                 agent=agent,
    #                 question=question,
    #                 all_ddls_str=all_ddls_str,
    #                 logger=logger,
    #                 top_k=args.top_k,
    #                 db_type=args.db_type,
    #                 db_path=args.db_path
    #             )
                
    #             if plans:
    #                 collected_plans_for_question.extend(plans)
    #                 logger.info(f"    Generated {len(plans)} plans in this batch")
    #             else:
    #                 logger.warning(f"    No plans generated in batch {batch_idx+1}")
            
    #         if collected_plans_for_question:
    #             all_collected_plans[q_id] = collected_plans_for_question
    #             logger.info(f"  Total plans collected for {q_id}: {len(collected_plans_for_question)}")
    #         else:
    #             logger.warning(f"  No plans collected for {q_id}")
                
    #     except Exception as e:
    #         logger.error(f"Error processing question {q_id}: {e}", exc_info=True)
    #         continue
    
    # # Save collected plans
    # logger.info(f"Saving {len(all_collected_plans)} questions' plans to {args.output}")
    # with open(args.output, 'w', encoding='utf-8') as f:
    #     json.dump(all_collected_plans, f, indent=2, ensure_ascii=False)
    
    # # Print summary
    # total_plans = sum(len(plans) for plans in all_collected_plans.values())
    # logger.info(f"Collection complete!")
    # logger.info(f"  Questions processed: {len(all_collected_plans)}/{len(questions)}")
    # logger.info(f"  Total plans collected: {total_plans}")
    # logger.info(f"  Average plans per question: {total_plans/len(all_collected_plans) if all_collected_plans else 0:.2f}")
    # logger.info(f"  Output saved to: {args.output}")

