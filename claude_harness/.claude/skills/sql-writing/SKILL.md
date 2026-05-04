---
name: sql-writing
description: Construct a SQL query that correctly answers a natural-language question grounded in the actual database.
---

# SQL Writing

Text-to-SQL is harder than it looks, and most of the difficulty is interpretation. Before a single clause is written, the question has to be read precisely, the database has to be understood as it actually is, and the possible readings of the question have to be checked against the data. Most mistakes are baked in at interpretation time — a word misread, a scope assumed, a grain inferred — and no amount of SQL polish will fix them afterwards.

## Principles

- **The question is a specification, not a description.** Every filter, every constraint, every column in the output shape is part of what you have to satisfy. The `SELECT` list is part of the spec, not an afterthought.
- **Ambiguity is more common than it looks.** Most real questions have at least one phrase that can be read more than one way — in scope, in grain, in aggregation, in filter semantics, or in output shape. Find them.
- **Test interpretations with data, not with reasoning.** When a phrase could mean two things, the cheapest way to tell which it means is to run a small query under each reading and look at the output. The surprise in the data is often the ambiguity you missed.
- **Candidates, not commitments.** Until the data has ruled out alternatives, your first interpretation is a hypothesis. Be willing to replace it if an alternative produces a more faithful answer.
- **External documents are specifications, not guidelines.** When the question references a formula, regex, or function definition, implement it exactly as written.
- **Simplest faithful reading wins.** Add no constraints the question did not ask for; drop none that it did.

## Tools

You have schema-inspection tools (`list_schemas`, `list_tables`, `list_columns`, `get_distinct_values`, `search_dimension_values`) and data-query tools (`query_database`, `python_interpreter`). Use them freely — each query is cheap compared to a wrong answer.

Consult `CLAUDE.md` for tool signatures and Snowflake syntax rules.
