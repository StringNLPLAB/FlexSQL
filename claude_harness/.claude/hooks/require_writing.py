#!/usr/bin/env python3
"""PreToolUse hook: block the first data-facing database call (query_database
or python_interpreter) until the sql-writing skill has been invoked at least
once. Schema-inspection tools (list_schemas, list_tables, list_columns,
get_distinct_values, search_dimension_values) remain free — the agent can
orient itself in the schema without triggering the gate.

The gate fires the moment the agent tries to *read data*, which is the point
where an interpretation commitment would otherwise harden. Once sql-writing
has been invoked the flag file is present and all subsequent data calls pass
through."""
import json, os, sys

DATA_TOOLS = {
    "mcp__snowflake-tools__query_database",
    "mcp__snowflake-tools__python_interpreter",
}

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_name = data.get("tool_name", "")
if tool_name not in DATA_TOOLS:
    sys.exit(0)

qdir = os.environ.get("CCSQL_QDIR")
if not qdir:
    sys.exit(0)

if os.path.exists(os.path.join(qdir, ".writing_invoked")):
    sys.exit(0)

print(
    "BLOCKED: You must invoke the sql-writing skill via the Skill tool before "
    "reading data from the database. Schema-inspection tools (list_schemas, "
    "list_tables, list_columns, get_distinct_values, search_dimension_values) "
    "are still available for orientation, but running query_database or "
    "python_interpreter requires the sql-writing mindset to be loaded. Invoke "
    "the skill now and follow its principles — the question is a specification, "
    "ambiguity is more common than it looks, and interpretations should be "
    "tested empirically, not assumed.",
    file=sys.stderr,
)
sys.exit(2)
