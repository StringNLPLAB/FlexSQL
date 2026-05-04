# -*- coding:utf-8 -*-

from typing import Any

import datetime
import json
import math
import os
import re

from difflib import SequenceMatcher
import pandas as pd
import parso
import sqlglot
from dateutil import parser


def divide(lst, partitions):
    chunck_size = math.ceil(len(lst) / partitions)
    for i in range(0, len(lst), chunck_size):
        yield lst[i:i + chunck_size]


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    # print('condition_cols', condition_cols)

    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                      sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols != []:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score


def is_valid_timestamp_format(date_string):
    """
    Uses dateutil to automatically detect and parse almost any date string.
    Returns the datetime object if successful, else False.
    """
    try:
        # automatically detects format and precision.
        value = parser.parse(date_string)
        return value.replace(tzinfo=None)

    except (ValueError, parser.ParserError, TypeError, OverflowError):
        # parser.parse raises ParserError for invalid strings
        # ValueError might be raised for out-of-bounds math
        return False

def _ws_norm(s: str) -> str:
    """Lowercase and collapse whitespace — basic text normalization."""
    return ' '.join(s.lower().split())

def _sort_key(x):
    """Sort key: None first, then numbers by value, then strings."""
    if x is None:
        return (0,)
    if isinstance(x, (int, float)):
        return (1, float(x))
    return (2, str(x))

def compare_tables(pred: pd.DataFrame, gold: pd.DataFrame, ignore_order: bool, fuzzy_threshold: float = 0.0) -> bool:
    # Quick rejection: different row counts cannot be equivalent.
    if len(pred) != len(gold):
        return False

    # Drop all-NaN columns to avoid wildcard matching
    pred = pred.dropna(axis=1, how="all")
    gold = gold.dropna(axis=1, how="all")

    # Guard: 0-column DataFrames after NaN drop must not vacuously match
    if pred.shape[1] == 0 or gold.shape[1] == 0:
        return pred.shape[1] == 0 and gold.shape[1] == 0

    tolerance = 1e-2

    def prep(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return _ws_norm(value)
        return value

    def vectors_match(v1, v2, tol: float, ignore_order_: bool) -> bool:
        v1 = [prep(x) for x in v1]
        v2 = [prep(x) for x in v2]
        if ignore_order_:
            v1 = sorted(v1, key=_sort_key)
            v2 = sorted(v2, key=_sort_key)

        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False

            if isinstance(a, str) and isinstance(b, str):
                a_timestamp = is_valid_timestamp_format(a)
                b_timestamp = is_valid_timestamp_format(b)
                if a_timestamp != b_timestamp:
                    return False

            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                if (fuzzy_threshold > 0
                        and isinstance(a, str) and isinstance(b, str)
                        and len(a) >= 4 and len(b) >= 4
                        and SequenceMatcher(None, a, b).ratio() >= fuzzy_threshold):
                    continue
                return False
        return True

    t_gold_list = gold.transpose().values.tolist()
    t_pred_list = pred.transpose().values.tolist()
    for gold_col in t_gold_list:
        if not any(vectors_match(gold_col, pred_col, tolerance, ignore_order) for pred_col in t_pred_list):
            return False
    return True

PythonNode = "PythonNode"
Newline = "Newline"
ImportName = "ImportName"
ImportFrom = "ImportFrom"
String = "String"
SimpleStmt = "simple_stmt"
ExprStmt = "expr_stmt"

getType = lambda x: type(x).__name__
is_newline = lambda x: getType(x) == Newline
is_import = lambda x: x.type == SimpleStmt and getType(x.children[0]) in {ImportName, ImportFrom}
is_docstring = lambda x: x.type == SimpleStmt and getType(x.children[0]) == String
is_separate_comment = lambda x: re.match(r'^#\s*-+$', x.get_code().strip().split('\n', 1)[0])


def chunk_program(program: str):
    ast = parso.parse(program, version="3.9")

    func_body = None
    for node in ast.children:
        if getType(node) == "Function" and node.name.value == "processing":
            func_body = node.children[-1]
            break

    assert func_body is not None, "No processing function."

    def filter_children(children, filter_import=True):
        out = []
        for child in children:
            if is_newline(child):
                continue
            elif filter_import and is_import(child):
                continue
            out.append(child)
        return out

    children = filter_children(func_body.children)

    if is_docstring(children[0]):
        docstring = children[0].get_code()
    else:
        docstring = None

    indices = []
    code = children[1:]
    for idx, child in enumerate(code):
        if is_separate_comment(child):
            indices.append(idx)
    indices.append(len(code) + 1)

    snippets = []
    for start, end in zip(indices[:-1], indices[1:]):
        snippet = ''.join([child.get_code() for child in code[start:end]])
        snippets.append(snippet.strip('\n'))

    return "dfs", docstring, snippets


def get_heuristic_prompts():
    sys_prompt = """
You are a helpful assistant specialized in translating Python (Pandas-style) code into SQL.

Given:
1) a Python code snippet, and
2) a set of heuristic translation rules from Python to SQL,
your task is to identify the subset of rules that are relevant to the given code snippet and could be applied during translation.

Do NOT generate SQL queries.  
Only select, filter, and present the applicable heuristic rules.
    """.strip()
    usr_prompt = """
### Task Description
You are given:
* A Python code snippet
* A set of heuristic rules for translating Python (Pandas-style) operations into SQL
Your task is to identify the heuristic rules that are relevant to the given Python code snippet and could be used during translation.

Do not perform the translation itself.
Do not invent new rules.
Only select rules that are applicable to the code snippet.

Please format your identified heuristic rules in a JSON List like
```json
{{"rules": ["<heuristic rule>", ...]}}
```
If no rules apply, return:
```json
{{"rules": []}}
```

### Python Code Snippet
```python
{snippet}
```

### Translation Assumptions
When reasoning about translation, assume the following Python operations should be ignored because they have no effect on SQL semantics:
* `.copy()` because SQL populates new table for each (sub-)query
* `.reset_index()` and `as_index=False` because SQL auto resets indices for rows
* `errors="coerce"` because SQL auto converts invalid data to NULL in type conversion
* `.get("Any string", None)` because there is no empty cell in a SQL table
* `.astype(str)` and `.str` because all non-numeric types are stored in strings

### Heuristic Rules
Assume `dataframe -> T` where `T` denotes the table name.

#### SELECT:
* `dataframe.iloc[:, [INDEX1, ..., INDEX_n]] -> SELECT COLUMN_1, ..., COLUMN_n FROM T` where `COLUMN_i` denote the (INDEX_i + 1)-th column name of T as INDEX_i starts from 0

#### WHERE:
* `dataframe.loc[MASK, [COLUMN1, ..., COLUMN_n]].copy() -> SELECT COLUMN1, ..., COLUMN_n # WHERE PREDICATE` where `PREDICATE` is transpiled from `MASK`

#### GROUP BY + HAVING:
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].count() -> SELECT COUNT(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].min() -> SELECT MIN(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].max() -> SELECT MAX(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].sum() -> SELECT SUM(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].mean() -> SELECT AVG(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`
* `dataframe.groupby([KEY1, ..., KEY_n]).agg(cols) -> SELECT Expression_i(COLUMN_i) AS alias_i FROM T GROUP BY KEY1, ..., KEY_n` where `cols` is a list of `alias_i=(COLUMN_i, Expression_i)`

#### ORDER BY + LIMIT + FETCH:
* `result.sort_values(by=COLUMN, ascending=BOOL).reset_index(drop=True) -> ORDER BY COLUMN_i (ASC if BOOL is True else DESC)`
* `result.sort_values(by=[COLUMN1, ..., COLUMN_n], ascending=[BOOL1, ..., BOOL_n]).reset_index(drop=True) -> ORDER BY COLUMN1 (ASC if BOOL1 is True else DESC), ..., COLUMN_n (ASC if BOOL_n is True else DESC)`
* `result.sort_values(by=COLUMN, ascending=BOOL).head(NUM) -> ORDER BY COLUMN_i (ASC if BOOL is True else DESC) LIMIT NUM`
* `dataframe.iloc[0] -> FETCH FIRST 1 ROWS ONLY`

#### DISTINCT:
* `dataframe[COLUMN1, ..., COLUMN_n].unique() -> SELECT DISTINCT COLUMN1, ..., COLUMN_n FROM T`
* `dataframe.unique() -> SELECT DISTINCT * FROM T`

#### JOIN:
* `pd.merge(dataframe1, dataframe2, how='inner', on=COLUMN) -> T1 INNER JOIN T2 ON T1.COLUMN = T2.COLUMN`
* `pd.merge(dataframe1, dataframe2, how='cross', on=COLUMN) -> T1 CROSS JOIN T2 ON T1.COLUMN = T2.COLUMN`
* `pd.merge(dataframe1, dataframe2, how='left', on=COLUMN) -> T1 LEFT JOIN T2 ON T1.COLUMN = T2.COLUMN`
* `pd.merge(dataframe1, dataframe2, how='right', on=COLUMN) -> T1 RIGHT JOIN T2 ON T1.COLUMN = T2.COLUMN`
* `pd.merge(dataframe1, dataframe2, how='outer', on=COLUMN) -> T1 FULL JOIN T2 ON T1.COLUMN = T2.COLUMN`

#### Flatten:
* `flatten_function(dataframe[COLUMN]) -> LATERAL FLATTEN (input COLUMN) as COLUMN` where `flatten_function` flatten a nested list/dict to a list, such as the following implementation:
```
def flatten_function(obj):
    if isinstance(obj, dict): for v in obj.values(): yield from flatten_function(v)
    elif isinstance(obj, (list, tuple)): for i in obj: yield from flatten_function(i)
    else: yield str(obj)
```

#### Alias:
* `dataframe.rename(columns={{COLUMN1: ALIAS1, ..., COLUMN_n: ALIAS_n}}) -> SELECT COLUMN1 AS ALIAS1, ..., COLUMN_n AS ALIAS_n`

#### Predicates:
* `~PREDICATE -> NOT PREDICATE`
* `PREDICATE1 & PREDICATE2 -> PREDICATE1 AND PREDICATE2`
* `PREDICATE1 | PREDICATE2 -> PREDICATE1 OR PREDICATE2`
* `dataframe[COLUMN].isna() -> COLUMN IS NULL`
* `dataframe[COLUMN].notna() | pd.notnull(dataframe[COLUMN]) | dataframe[COLUMN].notnull() -> COLUMN IS NOT NULL`
* `dataframe[COLUMN].str.fullmatch(r"REGEX") | dataframe[COLUMN].str.contains(r"REGEX") -> COLUMN LIKE "REGEX" | REGEXP_LIKE(COLUMN, "REGEX", parameters)` where parameters: 'c' for case-sensitive matching and 'i' for case-insensitive matching
* `dataframe[COLUMN].str.contains(STRING) -> STRING IN COLUMN`

#### Expressions:
* `dataframe[COLUMN].round(NUM) -> ROUND(COLUMN, NUM)`
* `log10(COLUMN) -> LOG(10, COLUMN)`
* `dataframe[COLUMN].str.strip() -> TRIM(COLUMN)`
* `dataframe[COLUMN].str.split(STRING) -> COLUMN.split(STRING)`,
* `pd.to_datetime(dataframe[COLUMN]) -> COLUMN::DATE`
* `pd.to_numeric(dataframe[COLUMN]) -> COLUMN::FLOAT`
* `int(dataframe[COLUMN]) -> COLUMN::INTEGER,
* `dataframe[COLUMN].str.replace(r"REGEX", STRING2, regex=True) -> REGEXP_REPLACE(COLUMN, "REGEX", STRING2)`
* `dataframe[COLUMN].str.replace(STRING1, STRING2, regex=False) -> REPLACE(COLUMN, STRING1, STRING2)`
* `json.loads(dataframe[COLUMN]) | dataframe[COLUMN].apply(lambda x: json.loads(x) if isinstance(x, str) else x) | _maybe_json(dataframe[COLUMN]) | _parse_maybe_json(dataframe[COLUMN]) | dataframe[COLUMN].apply(_maybe_json) | dataframe[COLUMN].apply(_parse_maybe_json) -> PARSE_JSON(COLUMN)`
* `dataframe[COLUMN].isin([STRING1, ..., STRING_n]) -> COLUMN IN [STRING1, ..., STRING_n]`

#### Window functions:
* `df.groupby([KEY1, ..., KEY_n]).first() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY KEY1 ASC, ..., KEY_n ASC) = 1`
* `df.groupby([KEY1, ..., KEY_n]).last() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY KEY1 DESC, ..., KEY_n DESC) = 1`
* `df.groupby([KEY1, ..., KEY_n]).tail(NUM) -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY KEY1 DESC, ..., KEY_n DESC) <= 10`
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].cumsum() -> SELECT KEY1, ..., KEY_n, SUM(COLUMN) OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY COLUMN) FROM T`
* `dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].idxmax() -> SELECT KEY1, ..., KEY_n, ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY COLUMN DESC) AS RN FROM T QUALIFY RN = 1`

#### Window functions + ORDER BY:
* `dataframe.sort_values([ORDER_KEY1, ..., ORDER_KEY_n], [ASC_FLAG1, ..., ASC_FLAG_n]).groupby([GROUPBY_KEY1, ..., GROUPBY_KEY_n], as_index=False).first() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER ( PARTITION BY GROUPBY_KEY1, ..., GROUPBY_KEY_n ORDER BY ORDER_KEY1 ASC_KEY1, ..., ORDER_KEY_n ASC_KEY_n) = 1` where `ASC_KEY_i = ASC` if `ASC_FLAG_i = True`; otherwise, `ASC_KEY_i = DESC`
* `dataframe.sort_values([ORDER_KEY1, ..., ORDER_KEY_n], [ASC_FLAG1, ..., ASC_FLAG_n]).groupby([GROUPBY_KEY1, ..., GROUPBY_KEY_n], as_index=False).last() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER ( PARTITION BY GROUPBY_KEY1, ..., GROUPBY_KEY_n ORDER BY ORDER_KEY1 ASC_KEY1, ..., ORDER_KEY_n ASC_KEY_n) = 1` where `ASC_KEY_i = DESC` if `ASC_FLAG_i = True`; otherwise, `ASC_KEY_i = ASC`

#### Subqueries:
* `dataframe.dropna(subset=[COLUMN1, ..., COLUMN_n]) -> SELECT * FROM T WHERE COLUMN1 IS NOT NULL OR ... OR COLUMN_n IS NOT NULL`

### Your Identified rules:
```
    """.strip()
    qwen_prompt = """
### Task Description
You are given:
* A Python code snippet
* A set of heuristic rules for translating Python (Pandas-style) operations into SQL
Your task is to identify the heuristic rules that are relevant to the given Python code snippet and could be used during translation.

Do not perform the translation itself.
Do not invent new rules.
Only select rules that are applicable to the code snippet.

Please format your identified heuristic rules in a JSON List like
```json
{{"rules": ["<heuristic rule>", ...]}}
```
If no rules apply, return:
```json
{{"rules": []}}
```

### Python Code Snippet
```python
{snippet}
```

### Translation Assumptions
When reasoning about translation, assume the following Python operations should be ignored because they have no effect on SQL semantics:
* `.copy()` because SQL populates new table for each (sub-)query
* `.reset_index()` and `as_index=False` because SQL auto resets indices for rows
* `errors="coerce"` because SQL auto converts invalid data to NULL in type conversion
* `.get("Any string", None)` because there is no empty cell in a SQL table
* `.astype(str)` and `.str` because all non-numeric types are stored in strings

### Post-Translation Validation Rules
After completing the translation, double-check the following:
* Ensure all identifiers are used consistently. In Snowflake, quoted column names are case-sensitive.
* When using `FLATTEN`, strictly follow the syntax `FLATTEN(input => COLUMN)`. Do not replace `COLUMN` with an expression.
* Avoid using `TRY_CAST` or `TRY_TO_<TYPE>`. Prefer explicit casting with `COLUMN::TYPE`.

### Heuristic Rules
Assume `dataframe -> T` where `T` denotes the table name.

{rules}

### Your Identified rules:
```
    """.strip()
    return sys_prompt, usr_prompt, qwen_prompt


task = "Given a Python code snippet, find relevant rules that can be used for translating code into SQL"
rules = [
    "`dataframe.iloc[:, [INDEX1, ..., INDEX_n]] -> SELECT COLUMN_1, ..., COLUMN_n FROM T` where `COLUMN_i` denote the (INDEX_i + 1)-th column name of T as INDEX_i starts from 0",
    "`dataframe.loc[MASK, [COLUMN1, ..., COLUMN_n]].copy() -> SELECT COLUMN1, ..., COLUMN_n # WHERE PREDICATE` where `PREDICATE` is transpiled from `MASK`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].count() -> SELECT COUNT(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].min() -> SELECT MIN(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].max() -> SELECT MAX(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].sum() -> SELECT SUM(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].mean() -> SELECT AVG(COLUMN) FROM T GROUP BY KEY1, ..., KEY_n`",
    "`dataframe.groupby([KEY1, ..., KEY_n]).agg(cols) -> SELECT Expression_i(COLUMN_i) AS alias_i FROM T GROUP BY KEY1, ..., KEY_n` where `cols` is a list of `alias_i=(COLUMN_i, Expression_i)`",
    "`result.sort_values(by=COLUMN, ascending=BOOL).reset_index(drop=True) -> ORDER BY COLUMN_i (ASC if BOOL is True else DESC)`",
    "`result.sort_values(by=[COLUMN1, ..., COLUMN_n], ascending=[BOOL1, ..., BOOL_n]).reset_index(drop=True) -> ORDER BY COLUMN1 (ASC if BOOL1 is True else DESC), ..., COLUMN_n (ASC if BOOL_n is True else DESC)`",
    "`result.sort_values(by=COLUMN, ascending=BOOL).head(NUM) -> ORDER BY COLUMN_i (ASC if BOOL is True else DESC) LIMIT NUM`",
    "`dataframe.iloc[0] -> FETCH FIRST 1 ROWS ONLY`",
    "`dataframe[COLUMN1, ..., COLUMN_n].unique() -> SELECT DISTINCT COLUMN1, ..., COLUMN_n FROM T`",
    "`dataframe.unique() -> SELECT DISTINCT * FROM T`",
    "`pd.merge(dataframe1, dataframe2, how='inner', on=COLUMN) -> T1 INNER JOIN T2 ON T1.COLUMN = T2.COLUMN`",
    "`pd.merge(dataframe1, dataframe2, how='cross', on=COLUMN) -> T1 CROSS JOIN T2 ON T1.COLUMN = T2.COLUMN`",
    "`pd.merge(dataframe1, dataframe2, how='left', on=COLUMN) -> T1 LEFT JOIN T2 ON T1.COLUMN = T2.COLUMN`",
    "`pd.merge(dataframe1, dataframe2, how='right', on=COLUMN) -> T1 RIGHT JOIN T2 ON T1.COLUMN = T2.COLUMN`",
    "`pd.merge(dataframe1, dataframe2, how='outer', on=COLUMN) -> T1 FULL JOIN T2 ON T1.COLUMN = T2.COLUMN`",
    "`flatten_function(dataframe[COLUMN]) -> LATERAL FLATTEN (input COLUMN) as COLUMN` where `flatten_function` flatten a nested list/dict to a list, such as the following implementation: ```def flatten_function(obj):\tif isinstance(obj, dict): for v in obj.values(): yield from flatten_function(v)\telif isinstance(obj, (list, tuple)): for i in obj: yield from flatten_function(i)\telse: yield str(obj)```",
    "`dataframe.rename(columns={{COLUMN1: ALIAS1, ..., COLUMN_n: ALIAS_n}}) -> SELECT COLUMN1 AS ALIAS1, ..., COLUMN_n AS ALIAS_n`",
    "`~PREDICATE -> NOT PREDICATE`",
    "`PREDICATE1 & PREDICATE2 -> PREDICATE1 AND PREDICATE2`",
    "`PREDICATE1 | PREDICATE2 -> PREDICATE1 OR PREDICATE2`",
    "`dataframe[COLUMN].isna() -> COLUMN IS NULL`",
    "`dataframe[COLUMN].notna() | pd.notnull(dataframe[COLUMN]) | dataframe[COLUMN].notnull() -> COLUMN IS NOT NULL`",
    "`dataframe[COLUMN].str.fullmatch(r\"REGEX\") | dataframe[COLUMN].str.contains(r\"REGEX\") -> COLUMN LIKE \"REGEX\"`",
    "`dataframe[COLUMN].str.contains(STRING) -> STRING IN COLUMN`",
    "`dataframe[COLUMN].round(NUM) -> ROUND(COLUMN, NUM)`",
    "`log10(COLUMN) -> LOG(10, COLUMN)`",
    "`dataframe[COLUMN].str.strip() -> TRIM(COLUMN)`",
    "`dataframe[COLUMN].str.split(STRING) -> COLUMN.split(STRING)`",
    "`pd.to_datetime(dataframe[COLUMN]) -> COLUMN::DATE`",
    "`pd.to_numeric(dataframe[COLUMN]) -> COLUMN::FLOAT`",
    "`int(dataframe[COLUMN]) -> COLUMN::INTEGER",
    "`dataframe[COLUMN].str.replace(r\"REGEX\", STRING2, regex=True) -> REGEXP_REPLACE(COLUMN, \"REGEX\", STRING2)`",
    "`dataframe[COLUMN].str.replace(STRING1, STRING2, regex=False) -> REPLACE(COLUMN, STRING1, STRING2)`",
    "`json.loads(dataframe[COLUMN]) | dataframe[COLUMN].apply(lambda x: json.loads(x) if isinstance(x, str) else x) | _maybe_json(dataframe[COLUMN]) | _parse_maybe_json(dataframe[COLUMN]) | dataframe[COLUMN].apply(_maybe_json) | dataframe[COLUMN].apply(_parse_maybe_json) -> PARSE_JSON(COLUMN)`",
    "`dataframe[COLUMN].isin([STRING1, ..., STRING_n]) -> COLUMN IN [STRING1, ..., STRING_n]`",
    "`df.groupby([KEY1, ..., KEY_n]).first() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY KEY1 ASC, ..., KEY_n ASC) = 1`",
    "`df.groupby([KEY1, ..., KEY_n]).last() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY KEY1 DESC, ..., KEY_n DESC) = 1`",
    "`df.groupby([KEY1, ..., KEY_n]).tail(NUM) -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY KEY1 DESC, ..., KEY_n DESC) <= 10`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].cumsum() -> SELECT KEY1, ..., KEY_n, SUM(COLUMN) OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY COLUMN) FROM T`",
    "`dataframe.groupby([KEY1, ..., KEY_n])[COLUMN].idxmax() -> SELECT KEY1, ..., KEY_n, ROW_NUMBER() OVER (PARTITION BY KEY1, ..., KEY_n ORDER BY COLUMN DESC) AS RN FROM T QUALIFY RN = 1`",
    "`dataframe.sort_values([ORDER_KEY1, ..., ORDER_KEY_n], [ASC_FLAG1, ..., ASC_FLAG_n]).groupby([GROUPBY_KEY1, ..., GROUPBY_KEY_n], as_index=False).first() -> SELECT \"FROM T QUALIFY ROW_NUMBER() OVER ( PARTITION BY GROUPBY_KEY1, ..., GROUPBY_KEY_n ORDER BY ORDER_KEY1 ASC_KEY1, ..., ORDER_KEY_n ASC_KEY_n) = 1` where `ASC_KEY_i = ASC` if `ASC_FLAG_i = True`; otherwise, `ASC_KEY_i = DESC`",
    "`dataframe.sort_values([ORDER_KEY1, ..., ORDER_KEY_n], [ASC_FLAG1, ..., ASC_FLAG_n]).groupby([GROUPBY_KEY1, ..., GROUPBY_KEY_n], as_index=False).last() -> SELECT * FROM T QUALIFY ROW_NUMBER() OVER ( PARTITION BY GROUPBY_KEY1, ..., GROUPBY_KEY_n ORDER BY ORDER_KEY1 ASC_KEY1, ..., ORDER_KEY_n ASC_KEY_n) = 1` where `ASC_KEY_i = DESC` if `ASC_FLAG_i = True`; otherwise, `ASC_KEY_i = ASC`",
    "`dataframe.dropna(subset=[COLUMN1, ..., COLUMN_n]) -> SELECT * FROM T WHERE COLUMN1 IS NOT NULL OR ... OR COLUMN_n IS NOT NULL`",
]


def get_topk_rules(topk, snippets, qwen_model):
    def get_instruct(task_desc, snippet):
        return f"Instruct: {task_desc}\nCode snippet: {snippet}"

    queries = [get_instruct(task, snippet) for snippet in snippets]
    input_texts = queries + rules
    outputs = qwen_model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    query_embeddings, rule_embeddings = embeddings[:len(queries)], embeddings[len(queries):]

    topk_rules = []
    for embedding in query_embeddings:
        scores = embedding[None, :] @ rule_embeddings.T
        _, ids = scores.topk(k=topk)
        topk_rules.append([rules[idx] for idx in ids[0].tolist()])
    return topk_rules


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def is_syntactically_valid(query):
    try:
        sqlglot.parse_one(query, read="snowflake")
        return True
    except:
        return False


def diff_outputs(pred, gold, condition_cols=[], ignore_order=False, to_csv=True, topk=10):
    # find different prediction and gold outputs

    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def tuple_match(v1, v2, tol=tolerance):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols != []:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    gold_headers = gold_cols.columns.values.tolist()
    pred_headers = pred_cols.columns.values.tolist()

    t_gold_list = gold_cols.values.tolist()
    t_pred_list = pred_cols.values.tolist()

    # t_gold_list = {tuple(t): 0 for t in t_gold_list}
    # t_pred_list = {tuple(t): 0 for t in t_pred_list}
    #
    # for gold_t in t_gold_list.keys():
    #     for pred_t in t_pred_list.keys():
    #         if tuple_match(gold_t, pred_t):
    #             t_pred_list[gold_t] = t_gold_list[gold_t] = 1
    #             break

    # gold_diff = [t for t, used_flag in t_gold_list.items() if not used_flag]
    # pred_diff = [t for t, used_flag in t_pred_list.items() if not used_flag]

    t_gold_set = {tuple(t) for t in t_gold_list}
    t_pred_set = {tuple(t) for t in t_pred_list}

    gold_diff = list(t_gold_set - t_pred_set)
    pred_diff = list(t_pred_set - t_gold_set)

    if to_csv:
        gold_diff = [gold_headers] + gold_diff[:topk]
        pred_diff = [pred_headers] + pred_diff[:topk]

        gold_diff = '\n'.join([','.join(map(str, t)) for t in gold_diff])
        pred_diff = '\n'.join([','.join(map(str, t)) for t in pred_diff])

    return pred_diff, gold_diff


def get_q_id(question, db_type):
    q_id = question.get("question_id", question.get("instance_id"))
    if q_id is None:
        raise ValueError(f"Question ID not found in question: {question}")

    return str(q_id)


def get_db_id(question):
    db_id = question.get("db_id", question.get("db"))
    if db_id is None:
        raise ValueError(f"Database ID not found in question: {question}")

    return str(db_id)


def get_question_str(question, db_type):
    if db_type == "sqlite":
        return question["question"]
    elif db_type == "snowflake":
        return question["instruction"]
    return None


def escape_format_braces(text):
    """
    Escape curly braces in text to prevent .format() from treating them as placeholders.
    Replaces { with {{ and } with }}.
    """
    if text is None:
        return ""
    return str(text).replace("{", "{{").replace("}", "}}")
def get_subquery_schema(qid: str, query: str, schema_file: str, sql: str = "snowflake"):
    from sqlglot import exp
    ast = sqlglot.parse_one(query, read=sql)
    columns = [col.name for col in ast.find_all(exp.Column)]
    with open(schema_file, 'r') as reader:
        schema = json.load(reader)
        schema_types = {name: type for name, type in zip(schema['column_names'], schema['column_types'])}
    # types = {col: schema_types[col] for col in columns}
    types = {}
    for col in columns:
        if col in schema_types:
            types[col] = schema_types[col]
        else:
            if qid == "sf_bq037":
                types[col] = {"IMPRECISE": "BOOLEAN", "MEND": "NUMBER", "MLEN": "NUMBER", "MSTART": "NUMBER", "OLD_VARIANT": "TEXT"}[col]
            elif qid == "sf_bq255" or qid == "sf_bq359":
                types[col] = {"difference_truncated": "BOOLEAN"}[col]
            else:
                # raise NotImplementedError(qid)
                print(qid, col)
    # ['sf_bq248', 'sf_bq255']
    return types
