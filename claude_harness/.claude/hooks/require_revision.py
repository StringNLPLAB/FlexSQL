#!/usr/bin/env python3
"""PreToolUse hook: block writes to answer.sql unless sql-revision was invoked
first. Exit 2 with a message on stderr blocks the tool call and feeds the
message back to the agent."""
import json, os, sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_name = data.get("tool_name", "")
if tool_name != "Write":
    sys.exit(0)

file_path = (data.get("tool_input") or {}).get("file_path", "") or ""
if not file_path.endswith("answer.sql"):
    sys.exit(0)

qdir = os.environ.get("CCSQL_QDIR")
if not qdir:
    sys.exit(0)

flag = os.path.join(qdir, ".revision_invoked")
if os.path.exists(flag):
    sys.exit(0)

print(
    "BLOCKED: You must invoke the sql-revision skill via the Skill tool before "
    "writing answer.sql. Revision carries the *breaking* mindset — you treat "
    "your own SQL skeptically, diff your output against the literal text of "
    "the question, and run probes that would contradict your answer if it were "
    "wrong. Load the skill now and follow its principles before finalizing.",
    file=sys.stderr,
)
sys.exit(2)
