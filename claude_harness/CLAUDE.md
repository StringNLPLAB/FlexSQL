# CCSQL — Text-to-SQL on Snowflake (Spider2)

## Task

You are solving a text-to-SQL benchmark task on Snowflake databases. You receive a natural language question about a Snowflake database and must produce a single correct SQL query that answers it.

See `workflow.md` for the high-level approach and the recommended use of the `sql-planning`, `sql-writing`, and `sql-revision` skills. This file is your reference for tools and SQL rules — consult it at every decision point.

## Environment

- `CCSQL_DB_ID` — the Snowflake database name (e.g., `PATENTS`, `ADVENTUREWORKS`)
- `CCSQL_DB_PATH` — path to the data directory (default: `spider2-snow`)

---

## Available MCP Tools

All tools operate on the database specified by `CCSQL_DB_ID`.

| Tool | Purpose |
|---|---|
| `list_schemas()` | List all schemas in the current database. |
| `list_tables(schema_name)` | List all tables in a schema (returns fully-qualified `DB.SCHEMA.TABLE` names). |
| `list_columns(table_name)` | Column names, types, descriptions, and example values for a table. Input must be fully-qualified. |
| `query_database(query)` | Execute a Snowflake SQL query; returns up to 20 result rows. |
| `get_distinct_values(table, column)` | Up to 100 distinct values from a column. Useful for verifying WHERE-clause filter values. |
| `search_dimension_values(terms, table, column_name, additional_columns?)` | Case-insensitive partial-match search in a column (up to 5 rows). Good for looking up codes, categories, or IDs. |
| `python_interpreter(code)` | Stateful Python interpreter with all DB tools above as functions + `pd`, `np`. Use when you need to: (1) batch-explore many tables/columns in a loop instead of many separate tool calls, (2) post-process query results with Python logic to inform your SQL (e.g. compare values across tables, compute stats), (3) do complex string/data wrangling to find correct filter values or join keys, or (4) do any kind of computations accurately. Don't do any math on your own, better use this tool. State persists across calls. |

---

## Snowflake SQL Reference

- Use double quotes for identifiers: `"DATABASE"."SCHEMA"."TABLE"."COLUMN"`
- `ILIKE` for case-insensitive matching: `WHERE name ILIKE '%value%'`
- `QUALIFY` clause filters window functions directly: `QUALIFY ROW_NUMBER() OVER (...) = 1`
- VARIANT/ARRAY access: `col:field::STRING`
- Date functions: `YEAR()`, `MONTH()`, `DATE_TRUNC('month', col)`, `DATEDIFF('day', start, end)`
- `TRY_CAST(x AS INT)` for NULL-safe casting
- Always add `LIMIT` when exploring to avoid huge result sets

---

## Rules

- Only run SELECT queries — never INSERT, UPDATE, DELETE, or DDL.
- Do not modify any source files in this repository.
- Do not import snowflake.connector or create database connections yourself — use the MCP tools.
- If a tool returns an error, try a different approach rather than giving up.
- Follow the instructions in the prompt. Do not attempt to do more than what is asked.
