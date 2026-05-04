"""
Prompt templates used throughout the SQL generation pipeline.
"""

HYBRID_STITCHING_PROMPT_TEMPLATE = """
### INSTRUCTIONS
You have the freedom to choose between two approaches to answer the user's question:

**Option 1: Python Approach**
Write a Python script that processes DataFrames loaded from the sub-queries.

**Option 2: SQL Approach**
Write a single SQL query that directly answers the question using the database.

Choose the approach that best fits the question. Use Python when you need complex data manipulation, string processing, or calculations that are easier in Python. Use SQL when a straightforward query can answer the question directly.

### SQL QUERIES AND EXAMPLE ROWS:
    {sub_sqls}

### User's Question
    {original_question}

### Your Task
Choose your approach and implement it:

**If choosing Python:**

You will be given the user's question and the list of SQL queries written in {db_type} dialect that were run to generate the DataFrames. The DataFrames are passed into your function as a list named `dfs`.

-   The DataFrame from `SQL Query 1` is at `dfs[0]`.
-   The DataFrame from `SQL Query 2` is at `dfs[1]`.
-   ...and so on.

**Schema & Data Types:**
Use the SQL queries and corresponding example rows to understand the schema (column names, data types) of each DataFrame in the `dfs` list.
Pay attention to the comments on datatypes for each column. If a column type is a nested dictionary or list, it may be loaded as a string. You should parse these using `json.loads()` to retrieve the actual nested dictionary or list object.

**Business Logic:**
Use the user question to understand the business logic required (e.g., joins, filtering, aggregation, calculations).

### SCRIPT REQUIREMENTS
**Libraries & Imports:**
* **Import Freely:** You are free to import and use any standard Python libraries necessary to complete the task. **You must include all necessary `import` statements at the beginning of your script.**
* **Recommended Toolkit:** We strongly suggest using the following libraries where appropriate, as they are well-suited for this data:
    * **Pandas** (standard for tabular data)
    * **Geopandas** (if spatial data/geometry is involved)
    * **NetworkX** (if graph/network algorithms are required)
    * **Numpy** (for numerical operations)

**Implementation Guidelines:**
* Enclose your final, complete Python script within a single ```python``` code block.
* Your function must be named `processing` and accept exactly one argument: the list of dataframes (`dfs`).
* Your function must return a **Pandas DataFrame** containing the answer to the question.
* **String Matching:** Only use direct string comparisons (e.g., `==`) when the question specifically demands an exact match (e.g., values enclosed in quotes). Otherwise, use pandas string search methods with regex patterns to ensure broad coverage.

### INTERACTIVE DEBUGGING
You have access to a Python interpreter tool that can execute code and test your `processing` function. Use it to:
1. Test your code incrementally
2. Debug any errors you encounter
3. Verify that your `processing` function works correctly with the data. **The list `dfs` is pre-loaded and available in the interpreter for you to access directly.** If there is anything you are unsure about regarding the data schema or content, use the interpreter to inspect `dfs` (e.g., `print(dfs[0].head())`).
4. Check that the output DataFrame is not empty

{db_access_note}

Write your complete program, then use the interpreter to test it. If you encounter errors, fix them and test again until the program works correctly.

**If choosing SQL:**
- Write a single SQL query within a ```sql``` code block
- The query should directly answer the user's question
- You can use the `query_database` tool to test your query interactively
- The query must be written in {db_type} dialect
- Make sure to use the correct table names and column names from the sub-queries provided

**Important:** Clearly indicate your choice at the beginning of your response. For example:
- "I'll use Python to solve this..." or "I'll use SQL to solve this..."
- Then provide your code in the appropriate code block.
"""

HYBRID_STITCHING_PLANNING_PROMPT_TEMPLATE = """
### INSTRUCTIONS
You have the freedom to choose between two approaches to answer the user's question:

**Option 1: Python Approach**
Write a Python script that processes DataFrames loaded from the sub-queries.

**Option 2: SQL Approach**
Write a single SQL query that directly answers the question using the database.

Choose the approach that best fits the question. Use Python when you need complex data manipulation, string processing, or calculations that are easier in Python. Use SQL when a straightforward query can answer the question directly.

### Some External Knowledge that might be useful:
    {external_knowledge}

### SQL QUERIES AND EXAMPLE ROWS:
    {sub_sqls}

### User's Question
    {original_question}

### Plan
    {plan}

### Your Task
Choose your approach and implement it following the plan:

**If choosing Python:**

You will be given the user's question and the list of SQL queries written in {db_type} dialect that were run to generate the DataFrames. The DataFrames are passed into your function as a list named `dfs`.

-   The DataFrame from `SQL Query 1` is at `dfs[0]`.
-   The DataFrame from `SQL Query 2` is at `dfs[1]`.
-   ...and so on.

**Schema & Data Types:**
Use the SQL queries and corresponding example rows to understand the schema (column names, data types) of each DataFrame in the `dfs` list.
Pay attention to the comments on datatypes for each column. If a column type is a nested dictionary or list, it may be loaded as a string. You should parse these using `json.loads()` to retrieve the actual nested dictionary or list object.

**Business Logic:**
Use the user question to understand the business logic required (e.g., joins, filtering, aggregation, calculations). A plan has been provided below to guide your implementation. Please refer to the plan when writing your Python script.

### SCRIPT REQUIREMENTS
**Libraries & Imports:**
* **Import Freely:** You are free to import and use any standard Python libraries necessary to complete the task. **You must include all necessary `import` statements at the beginning of your script.**
* **Recommended Toolkit:** We strongly suggest using the following libraries where appropriate, as they are well-suited for this data:
    * **Pandas** (standard for tabular data)
    * **Geopandas** (if spatial data/geometry is involved)
    * **NetworkX** (if graph/network algorithms are required)
    * **Numpy** (for numerical operations)

**Implementation Guidelines:**
* Enclose your final, complete Python function within a single ```python``` code block. Don't include any other code to launch the function. The function will be tested automatically.
* Your function must be named `processing` and accept exactly one argument: the list of dataframes (`dfs`).
* Your function must return a **Pandas DataFrame** containing the answer to the question.
* **String Matching:** Only use direct string comparisons (e.g., `==`) when the question specifically demands an exact match (e.g., values enclosed in quotes). Otherwise, use pandas string search methods with regex patterns to ensure broad coverage.

### INTERACTIVE DEBUGGING
You have access to a Python interpreter tool that can execute code and test your `processing` function. Use it to:
1. Test your code incrementally. Ensure that special characters are properly escaped in the code.
2. Debug any errors you encounter
3. Verify that your `processing` function works correctly with the data. **The list `dfs` is pre-loaded and available in the interpreter for you to access directly.** If there is anything you are unsure about regarding the data schema or content, use the interpreter to inspect `dfs` (e.g., `print(dfs[0].head())`).
4. You don't need to see the whole output DataFrame, just use the interpreter to debug your code incrementally, we will execute the whole function later and tell you if you need to do any revisions.

{db_access_note}

Write your complete program, then use the interpreter to test it. If you encounter errors, fix them and test again until the program works correctly.

**If choosing SQL:**
- Write a single SQL query within a ```sql``` code block
- The query should directly answer the user's question following the plan
- You can use the `query_database` tool to test your query interactively. Note that the tool will only return the first 10 rows of the result and is just used for debugging. We will execute the query later and tell you if you need to do any revisions.
- The query must be written in {db_type} dialect
- Make sure to use the correct table names and column names from the sub-queries provided. Don't use any wildcard patterns, simply list all the necessary tables and columns.
- Refer to the plan when constructing your query

**Important:** Clearly indicate your choice at the beginning of your response. For example:
- "I'll use Python to solve this..." or "I'll use SQL to solve this..."
- Then provide your code in the appropriate code block.
- Please make sure that the code is fully functional and can be executed directly without any errors.
"""

SQL_ONLY_STITCHING_PROMPT_TEMPLATE = """
### INSTRUCTIONS
Write a single SQL query that directly answers the user's question using the database.

### SQL QUERIES AND EXAMPLE ROWS:
    {sub_sqls}

### User's Question
    {original_question}

### Your Task
Write a SQL query that answers the question:

- Write a single SQL query within a ```sql``` code block
- The query should directly answer the user's question
- You can use the `query_database` tool to test your query interactively
- The query must be written in {db_type} dialect
- Make sure to use the correct table names and column names from the sub-queries provided

**Important:** Provide your SQL query in a ```sql``` code block. Make sure the code is fully functional and can be executed directly without any errors.
"""

SQL_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE = """
### INSTRUCTIONS
Write a single SQL query that directly answers the user's question using the database.

### Some External Knowledge that might be useful:
    {external_knowledge}

### SQL QUERIES AND EXAMPLE ROWS:
    {sub_sqls}

### User's Question
    {original_question}

### Plan
    {plan}

### Your Task
Write a SQL query that answers the question following the plan:

- Write a single SQL query within a ```sql``` code block
- The query should directly answer the user's question following the plan
- You can use the `query_database` tool to test your query interactively. Note that the tool will only return the first 10 rows of the result and is just used for debugging. We will execute the query later and tell you if you need to do any revisions.
- The query must be written in {db_type} dialect
- Make sure to use the correct table names and column names from the sub-queries provided. Don't use any wildcard patterns, simply list all the necessary tables and columns.
- Refer to the plan when constructing your query

**Important:** Provide your SQL query in a ```sql``` code block. Please make sure that the code is fully functional and can be executed directly without any errors.
"""

PYTHON_ONLY_STITCHING_PROMPT_TEMPLATE = """
### INSTRUCTIONS
Write a Python script that processes DataFrames loaded from the sub-queries to answer the user's question.

### SQL QUERIES AND EXAMPLE ROWS:
    {sub_sqls}

### User's Question
    {original_question}

### Your Task
Write a Python script that answers the question:

The DataFrames are passed into your function as a list named `dfs`.

-   The DataFrame from `SQL Query 1` is at `dfs[0]`.
-   The DataFrame from `SQL Query 2` is at `dfs[1]`.
-   ...and so on.

**Schema & Data Types:**
Use the SQL queries and corresponding example rows to understand the schema (column names, data types) of each DataFrame in the `dfs` list.
Pay attention to the comments on datatypes for each column. If a column type is a nested dictionary or list, it may be loaded as a string. You should parse these using `json.loads()` to retrieve the actual nested dictionary or list object.

### SCRIPT REQUIREMENTS
**Libraries & Imports:**
* **Import Freely:** You are free to import and use any standard Python libraries necessary to complete the task. **You must include all necessary `import` statements at the beginning of your script.**
* **Recommended Toolkit:** We strongly suggest using the following libraries where appropriate, as they are well-suited for this data:
    * **Pandas** (standard for tabular data)
    * **Geopandas** (if spatial data/geometry is involved)
    * **NetworkX** (if graph/network algorithms are required)
    * **Numpy** (for numerical operations)

**Implementation Guidelines:**
* Enclose your final, complete Python script within a single ```python``` code block.
* Your function must be named `processing` and accept exactly one argument: the list of dataframes (`dfs`).
* Your function must return a **Pandas DataFrame** containing the answer to the question.
* **String Matching:** Only use direct string comparisons (e.g., `==`) when the question specifically demands an exact match (e.g., values enclosed in quotes in the question). Otherwise, use pandas string search methods with regex patterns to ensure broad coverage.

### INTERACTIVE DEBUGGING
You have access to a Python interpreter tool that can execute code and test your `processing` function. Use it to:
1. Test your code incrementally
2. Debug any errors you encounter
3. Verify that your `processing` function works correctly with the data. **The list `dfs` is pre-loaded and available in the interpreter for you to access directly.** If there is anything you are unsure about regarding the data schema or content, use the interpreter to inspect `dfs` (e.g., `print(dfs[0].head())`).
4. Check that the output DataFrame is not empty

{db_access_note}

Write your complete program, then use the interpreter to test it. If you encounter errors, fix them and test again until the program works correctly.

**Important:** Provide your Python script in a ```python``` code block. Make sure the code is fully functional and can be executed directly without any errors.
"""

PYTHON_ONLY_STITCHING_PLANNING_PROMPT_TEMPLATE = """
### INSTRUCTIONS
Write a Python script that processes DataFrames loaded from the sub-queries to answer the user's question, following the provided plan.

### Some External Knowledge that might be useful:
    {external_knowledge}

### SQL QUERIES AND EXAMPLE ROWS:
    {sub_sqls}

### User's Question
    {original_question}

### Plan
    {plan}

### Your Task
Write a Python script that answers the question following the plan:

The DataFrames are passed into your function as a list named `dfs`.

-   The DataFrame from `SQL Query 1` is at `dfs[0]`.
-   The DataFrame from `SQL Query 2` is at `dfs[1]`.
-   ...and so on.

**Schema & Data Types:**
Use the SQL queries and corresponding example rows to understand the schema (column names, data types) of each DataFrame in the `dfs` list.
Pay attention to the comments on datatypes for each column. If a column type is a nested dictionary or list, it may be loaded as a string. You should parse these using `json.loads()` to retrieve the actual nested dictionary or list object.

### SCRIPT REQUIREMENTS
**Libraries & Imports:**
* **Import Freely:** You are free to import and use any standard Python libraries necessary to complete the task. **You must include all necessary `import` statements at the beginning of your script.**
* **Recommended Toolkit:** We strongly suggest using the following libraries where appropriate, as they are well-suited for this data:
    * **Pandas** (standard for tabular data)
    * **Geopandas** (if spatial data/geometry is involved)
    * **NetworkX** (if graph/network algorithms are required)
    * **Numpy** (for numerical operations)

**Implementation Guidelines:**
* Enclose your final, complete Python script within a single ```python``` code block.
* Your function must be named `processing` and accept exactly one argument: the list of dataframes (`dfs`).
* Your function must return a **Pandas DataFrame** containing the answer to the question.
* **String Matching:** Only use direct string comparisons (e.g., `==`) when the question specifically demands an exact match (e.g., values enclosed in quotes in the question). Otherwise, use pandas string search methods with regex patterns to ensure broad coverage.
* Refer to the plan when writing your Python script.

### INTERACTIVE DEBUGGING
You have access to a Python interpreter tool that can execute code and test your `processing` function. Use it to:
1. Test your code incrementally
2. Debug any errors you encounter
3. Verify that your `processing` function works correctly with the data. **The list `dfs` is pre-loaded and available in the interpreter for you to access directly.** If there is anything you are unsure about regarding the data schema or content, use the interpreter to inspect `dfs` (e.g., `print(dfs[0].head())`).
4. Check that the output DataFrame is not empty

{db_access_note}

Write your complete program, then use the interpreter to test it. If you encounter errors, fix them and test again until the program works correctly.

**Important:** Provide your Python script in a ```python``` code block. Please make sure that the code is fully functional and can be executed directly without any errors.
"""

HIERARCHICAL_SCHEMA_LINKING_PROMPT_TEMPLATE = """
**Persona**: You are an expert SQL schema analyst. Your task is to identify which SCHEMAS in a database are relevant for answering a natural language question. Within each schema, you can use the tools to explore the tables and columns.

**Database**: {database_name}

**Schema Statistics**:
{schema_statistics}

**Question**: {question}

**Instructions**:
1. You have access to tools that allow you to explore the database structure hierarchically:
   - `list_tables(schema_name)`: Lists all tables in a specific schema
   - `list_columns(table_name)`: Lists all columns in a specific table

2. Start by exploring the schemas systematically. Use the tools to:
   - Explore schemas that might be relevant to the question

3. **Bias towards inclusion**: If you are unsure whether a schema is relevant, include it. Only exclude schemas if you are confident they are not needed.

4. After exploring, output a JSON list of fully qualified schema names that are relevant to answering the question. Use the format: SCHEMA

**Output Format**:
Provide your final answer as a JSON array of relevant schema names, enclosed in a ```json``` block:

```json
[
  "SCHEMA1",
  "SCHEMA2",
  ...
]
```

**Important**:
- Use schema names (SCHEMA format)
- Only include schemas that are relevant to answering the question
- If no schemas are relevant, return an empty array []
"""

SNOWFLAKE_VARIANT_ACCESS_RULES = """
### **Snowflake VARIANT Column Access Rules**
When writing SQL queries that access VARIANT data types in Snowflake (columns where each value is a JSON object), you must specify the full access path including nested keys and array indices. **Always treat key names and column names as case-sensitive** by using double quotes.

* **Case-Sensitive Keys:** Use double quotes for exact matches: "column_name":"KeyName" (always use this for key names to ensure case sensitivity).
* **Keys with Special Characters:** Use bracket notation: "column_name"["Key with Space"] or "column_name"["key.with.dots"].
* **Nested Object Access:** "column_name":"OuterKey":"InnerKey".
* **Array Element Access:** Use zero-based indexing: "column_name"[0].
* **Dynamic Key Access:** Use GET_PATH() function for dynamic keys: GET_PATH("column_name", "dynamic_key_string").

**Important:** For VARIANT columns, always include the full access path to keys you need (e.g., "column_name":"nested_key":"sub_key"[0]) rather than just the base column name.

---

"""

NO_DIVERSE_PLANNING_SINGLE_PROMPT_TEMPLATE = """
### Table Overview (table name -> number of columns)
{table_overview}

You may call `list_columns(table_name)` to inspect column names, `query_database(sql)` to check actual values, `get_distinct_values(table, col)` to enumerate a column, and `search_dimension_values(table, col, keyword)` to search for specific values.

### User Question
{question}

### External Knowledge that might be useful:
{external_knowledge}

### Your Task

Generate an execution plan to answer the user's question.

**Step 1: Intent Analysis**
Map the question to schema elements:
1. **What (Entities):** Map nouns to tables/columns. Note ambiguous mappings.
2. **Where (Scope):** Identify filtering conditions or data sources.
3. **When (Time):** Identify time-based constraints, if any.
4. **How (Logic):** Sorting, limits, aggregations, formulas.

**Step 2: Feasibility Check and Value Grounding**
Before writing the plan:
1. Confirm each table exists in the Table Overview. Use `list_columns` to inspect columns.
2. For strict system keys (status codes, type identifiers, enums), use `get_distinct_values` or `search_dimension_values` to find exact values. Use `query_database` when you need to verify logic, check join conditions, or sample data. Do NOT verify flexible text fields (names, descriptions).
3. Verify join conditions and logical flow are sound.

**Step 3: Write the Plan and Identify Columns**
Write a step-by-step execution plan (natural language only, no SQL). Then for each table, list the exact columns needed (for filtering, joining, selecting, aggregating, sorting).

### Output Format
Return a single JSON object in a ```json``` code block with this structure:
```json
{output_format}
```

Rules:
- `"text"`: Full execution plan as a string (each step numbered on a new line).
- `"tables"`: Dict mapping each table name{clarify_table_name_rule} to a list of relevant column names.{clarify_wildcard_rule}
"""
