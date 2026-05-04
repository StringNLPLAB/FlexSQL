---
name: sql-revision
description: Verify that a SQL query's actual output faithfully answers the natural-language question.
---

# SQL Revision

A query that runs is not the same as a query that is right. The goal of revision is to catch the gap — the cases where the SQL executes cleanly but produces an answer that doesn't match what the question literally asks for. Treat your own SQL skeptically: the agent that wrote it has already convinced itself it's correct, and that agent is you.

## Principles

- **The literal text of the question is the arbiter.** Not your memory of it, not the plan you wrote, not your reasoning about it. Quote the parts of the question that describe the output and the constraints so you have something concrete to diff against.
- **Check the shape before the values.** If the question asks for specific columns in a specific order and your output has different ones, that is a real mistake — do not rationalize it as a naming convention or "extra context." Compare column lists directly.
- **Check the constraints individually.** Walk through every filter, grouping, scope, and condition in the question and find where your SQL implements it. If you cannot point to the implementation, the constraint is silently missing.
- **Probe to contradict, not to confirm.** Run queries whose result should match your answer *if* the answer is correct — a different aggregation, a row-count sanity check, a spot check against the raw tables. If a probe disagrees, the answer is wrong. If every probe agrees, your answer is consistent — do not manufacture fixes for problems the probes did not reveal.
- **Only act on evidence.** Revision is a test, not a license to rewrite. Change the query only when a specific disagreement has been observed.

Consult `CLAUDE.md` for tool signatures and Snowflake syntax rules.
