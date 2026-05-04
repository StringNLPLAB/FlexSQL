#!/usr/bin/env python3
"""PostToolUse hook: when the Skill tool loads sql-writing, touch a flag file
so the pre-data-query hook knows it has been invoked at least once."""
import json, os, sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_name = data.get("tool_name", "")
if tool_name != "Skill":
    sys.exit(0)

skill = (data.get("tool_input") or {}).get("skill", "") or ""
if "sql-writing" not in skill:
    sys.exit(0)

qdir = os.environ.get("CCSQL_QDIR")
if not qdir:
    sys.exit(0)

try:
    os.makedirs(qdir, exist_ok=True)
    open(os.path.join(qdir, ".writing_invoked"), "w").close()
except Exception:
    pass

sys.exit(0)
