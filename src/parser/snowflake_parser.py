# -*- coding:utf-8 -*-
from collections import defaultdict

import sqlglot
from sqlglot import exp
from parser import visitor

DEFAULT_ATTRS = ['SEQ', 'KEY', 'PATH', 'INDEX', 'VALUE', 'THIS']
CURR_CTX_NAME = "CURR"
TMP_CTX_NAME = "TMP"


class SnowflakeParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.schema = None

    def parse(self, schema, query):
        self.reset()

        # get involved attributes in the query
        self.involved_attributes = defaultdict(dict)
        # build mapping between attributes and their aliases
        self.attr_aliases = {}
        for table, attrs in schema.items():
            for attr in attrs.keys():
                self.involved_attributes[table][attr] = 0
                attr = f"{table}.{attr}"
                self.attr_aliases[attr] = attr

        ast = sqlglot.parse_one(query, read="snowflake")
        self.with_ctx = self.schema = {table: set(attrs.keys())
                                       for table, attrs in schema.items()}
        self.init_tables = [self.schema.keys()]
        ast, _ = self.visit(ast, ctx=self.schema.copy())

        involved_attrs = defaultdict(set)
        for table, attrs in self.involved_attributes.items():
            iattrs = {attr for attr, occurrence in attrs.items() if occurrence > 0}
            if len(iattrs) != 0:
                involved_attrs[table] = iattrs

        return ast, involved_attrs

    def get_init_ctx(self):
        return {table: self.schema[table] for table in self.init_tables}

    @staticmethod
    def get_ast_size(ast):

        def get_size(ast):
            if isinstance(ast, dict):
                size = 0
                for key, value in ast.items():
                    if not (key in {'value', 'literal', 'number', 'null', 'table', 'current_data', 'columns'} or isinstance(value,
                                                                                                                            bool)):
                        size += 1
                    size += get_size(value)
                return size
            elif isinstance(ast, int | float | str):
                return 1
            elif ast is None:
                return 1
            elif isinstance(ast, list | tuple):
                return sum([get_size(node) for node in ast])
            else:
                raise NotImplementedError(ast)

        return get_size(ast)

    @staticmethod
    def get_max_ast_height(ast):

        def get_max_height(ast):
            if isinstance(ast, dict):
                sizes = []
                for key, value in ast.items():
                    size = 0
                    if not (key in {'value', 'literal', 'number', 'null', 'table', 'current_data', 'columns'} or isinstance(value,
                                                                                                                            bool)):
                        size += 1
                    size += get_max_height(value)
                    sizes.append(size)
                return 0 if len(sizes) == 0 else max(sizes)
            elif isinstance(ast, int | float | str):
                return 1
            elif ast is None:
                return 1
            elif isinstance(ast, list | tuple):
                sizes = [get_max_height(node) for node in ast]
                return 0 if len(sizes) == 0 else max(sizes)
            else:
                raise NotImplementedError(ast)

        return get_max_height(ast)

    @staticmethod
    def get_statistics(ast):
        out = {"join": 0, "outer_join": 0, "window": 0, "qualify": 0, "agg": 0, "alias": 0, "with": 0, "group_by": 0,
               "having": 0, "order_by": 0, "limit": 0, "regexp": 0, "subquery": 0, "set_op": 0, "structure": 0}

        def traverse(ast):
            if isinstance(ast, dict):
                for key, value in ast.items():
                    if 'join' in key:
                        out["join"] += 1
                    elif key in {'left_join', 'right_join', 'full_join'}:
                        out["outer_join"] += 1
                    elif key == "window":
                        out["window"] += 1
                    elif key == "qualify":
                        out["qualify"] += 1
                    elif key in {'min', 'max', 'sum', 'avg', 'count', 'count_if', 'corr', 'list_agg', 'median', 'max_by',
                                 'min_by', 'array_agg'}:
                        out["agg"] += 1
                    elif key == "alias":
                        out["alias"] += 1
                    elif key == "with":
                        out["with"] += 1
                    elif key == "group_by":
                        out["group_by"] += 1
                    elif key == "having":
                        out["having"] += 1
                    elif key == "order_by":
                        out["order_by"] += 1
                    elif key in {'limit', 'fetch'}:
                        out["limit"] += 1
                    elif 'like' in key or 'regexp' in key:
                        out["regexp"] += 1
                    elif key == "subquery":
                        out["subquery"] += 1
                    elif key == "union":
                        out["set_op"] += 1
                    elif any(v in key for v in
                             ['array', 'array_concat', 'json_extract', 'array_contains', 'array_generate_range', 'parse_json',
                              'explode', 'lateral', 'st_point', 'st_distance', 'within_group']):
                        out["structure"] += 1

                    traverse(value)
                return
            elif isinstance(ast, int | float | str):
                return
            elif ast is None:
                return
            elif isinstance(ast, list | tuple):
                for node in ast:
                    traverse(node)
                return
            else:
                raise NotImplementedError(ast)

        traverse(ast)
        return out

    ####################### static analysis #########################

    def get_atom_subquery(self, query):
        atom_subquery = query
        while 'union' in atom_subquery: # intersect and except are not considered
            atom_subquery = atom_subquery['union'][0]
        return atom_subquery

    def get_attr_from_ctx(self, attr, ctx):
        if not isinstance(attr, str):
            return None

        if '.' in attr:  # T.a
            return attr
        else:  # a
            for table, attrs in ctx.items():
                if attr in attrs:
                    return f"{table}.{attr}"
        # raise ValueError(attr, ctx)
        # SELECT f.value:"code"::STRING AS cpc_code, SUBSTRING(cpc_code, 1, 4) AS cpc_level4
        return None

    def get_upper_attr(self, attr):
        idx = attr.rfind('.')
        return str.upper(attr[:idx]) + attr[idx:]

    def register_involved_attr(self, attr):
        upper_attr = self.get_upper_attr(attr)
        if attr in self.attr_aliases:
            pass
        elif upper_attr in self.attr_aliases:
            attr = upper_attr
        else:
            return
        if attr in self.attr_aliases:
            ori_attr = self.attr_aliases[attr]
            table, attr = ori_attr.rsplit('.', 1)
            if table in self.involved_attributes and attr in self.involved_attributes[table]:
                self.involved_attributes[table][attr] += 1
                return

    def register_attr_alias(self, src_attr, tgt_attr):
        # only allow aliases for atomic attributes
        if src_attr in self.attr_aliases:
            self.attr_aliases[tgt_attr] = self.attr_aliases[src_attr]
        elif self.get_upper_attr(src_attr) in self.attr_aliases:
            self.attr_aliases[tgt_attr] = self.attr_aliases[self.get_upper_attr(src_attr)]

    def get_table_from_ctx(self, table, ctx):
        if 'table_from_rows' in table:  # as it does not have alias, we temporarily create one
            return {TMP_CTX_NAME: set(DEFAULT_ATTRS)}

        if 'lateral' in table:
            table = table['lateral']

        if 'alias' in table:
            table_name = table['alias']['table']
        elif 'table' in table:
            table_name = table['table']
        elif 'subquery' in table:
            table_name = None  # this subquery has no alias
        else:
            raise NotImplementedError(table, list(ctx.keys()))

        if table_name is None:
            atom_query = self.get_atom_subquery(table['subquery'])
            return {TMP_CTX_NAME: self.get_attrs_from_select(atom_query['select'])}
        elif table_name in ctx:
            return {table_name: ctx[table_name]}
        elif str.upper(table_name) in ctx:
            return {table_name: ctx[str.upper(table_name)]}
        else:
            raise ValueError(f"Unknown table {table_name} in {list(ctx.keys())}.")

    ####################### statements #########################

    def get_attrs_from_select(self, select_clause):
        attrs = set()
        for column in select_clause:
            if isinstance(column, str):
                attrs.add(column)
            elif isinstance(column, dict) and 'alias' in column:
                attrs.add(column['alias'])
            else:
                raise NotImplementedError(column)
        return attrs

    def register_start(self, table, ctx):
        # register involved attributes
        for attr in ctx[table]:
            if f'{table}.{attr}' in self.attr_aliases:
                ori_attr = self.attr_aliases[f'{table}.{attr}']
            elif f'{str.upper(table)}.{attr}' in self.attr_aliases:
                ori_attr = self.attr_aliases[f'{str.upper(table)}.{attr}']
            else:
                continue
            table, attr = ori_attr.rsplit('.', 1)
            self.involved_attributes[table][attr] += 1

    @visitor(exp.Select)
    def visit(self, ast, ctx=None, **kwargs):
        init_tables = set(ctx.keys())
        outer_from_tables = set(kwargs.get('from_ctx', {}).keys())
        out = {}
        # WITH
        if ast.args.get('with', None) is not None:
            out['with'], ctx = self.visit(ast.args['with'], ctx=ctx, **kwargs)
        # FROM
        from_table = None
        if ast.args.get('from', None) is not None:
            out['from'], from_ctx = self.visit(ast.args['from'], ctx=ctx, **kwargs)
            from_table = self.get_table_from_ctx(out['from'], from_ctx)
        # JOIN
        join_tables = None
        if ast.args.get('joins', None) is not None:
            # SELECT FLIGHTS."flight_id" FROM AIRLINES.AIRLINES.FLIGHTS JOIN XX
            table = list(from_table.keys())[0]
            alias = table.rsplit('.', 1)[-1]
            for attr in from_table[table]:
                self.register_attr_alias(f"{table}.{attr}", f"{alias}.{attr}")

            # assert len(from_ctx) >= len(ctx)
            join_clause, join_tables = [], {}
            for join_stmt in ast.args['joins']:
                if 'from_ctx' in kwargs:
                    kwargs['from_ctx'].update(from_table)
                else:
                    kwargs['from_ctx'] = from_table
                join_stmt, join_ctx = self.visit(join_stmt, ctx=ctx, **kwargs)
                join_clause.append(join_stmt)

                # assert len(join_stmt) == 1, join_stmt
                join_type = list(join_stmt.keys())[0]
                join_table = self.get_table_from_ctx(join_stmt[join_type], join_ctx)

                join_tables.update(join_table)
            out['join'] = join_clause

        # current ctx
        curr_ctx = {}
        if from_table is not None:
            curr_ctx.update(from_table)
        if join_tables is not None:
            curr_ctx.update(join_tables)

        # SELECT
        curr_attrs = set()
        out['select'] = []
        for expr in ast.args['expressions']:
            expr, _ = self.visit(expr, ctx=curr_ctx, **kwargs)
            out['select'].append(expr)
            if 'alias' in expr:
                curr_attrs.add(expr['alias'])
            elif 'column' in expr and isinstance(expr['column'], str):
                curr_attrs.add(expr['column'])
            elif isinstance(expr, str) and str.endswith(expr, '.*'):  # expand *
                table, _ = expr.rsplit('.', 1)
                self.register_start(table, curr_ctx)
                curr_attrs |= curr_ctx[table]
            elif isinstance(expr, str) and expr == '*':
                assert len(curr_ctx) == 1
                table = list(curr_ctx.keys())[0]
                self.register_start(table, ctx)
                curr_attrs |= curr_ctx[table]

        # SELECT DISTINCT
        if ast.args.get('distinct', None) is not None:
            out['select_distinct'] = ast.args['distinct'] is not None
        # WHERE
        if ast.args.get('where', None) is not None:
            out['where'], _ = self.visit(ast.args['where'], ctx=curr_ctx, **kwargs)
        # QUALIFY
        if ast.args.get('qualify', None) is not None:
            out['qualify'], _ = self.visit(ast.args['qualify'], ctx=curr_ctx, **kwargs)

        # In GROUP BY + HAVING and ORDER BY, they can use aliases in SELECT.
        tmp_ctx = {**curr_ctx, CURR_CTX_NAME: curr_attrs}
        # GROUP BY + HAVING
        if ast.args.get('group', None) is not None:
            out['group_by'], _ = self.visit(ast.args['group'], ctx=tmp_ctx, **kwargs)
        if ast.args.get('having', None) is not None:
            out['having'], _ = self.visit(ast.args['having'], ctx=tmp_ctx, **kwargs)
        # ORDER BY
        if ast.args.get('order', None) is not None:
            out['order_by'], _ = self.visit(ast.args['order'], ctx=tmp_ctx, **kwargs)

        # LIMIT / FETCH
        if ast.args.get('limit', None) is not None:
            limit_fetch, _ = self.visit(ast.args['limit'], ctx=curr_ctx, **kwargs)
            out.update(limit_fetch)

        # pop out the intermediate table in this query
        pop_keys = {key for key in ctx.keys() if key not in init_tables}
        for key in pop_keys:
            ctx.pop(key)
        pop_keys = {key for key in kwargs.get('from_ctx', {}).keys() if key not in outer_from_tables}
        for key in pop_keys:
            kwargs['from_ctx'].pop(key)
        return out, curr_ctx

    @visitor(exp.With)
    def visit(self, ast, ctx=None, **kwargs):
        out = {}
        if ast.args.get('recursive', None) is not None:
            out['recursive'] = True
        out['cte'] = []
        for expr in ast.args['expressions']:
            expr, ctx = self.visit(expr, ctx=ctx, **kwargs)
            self.with_ctx = ctx
            out['cte'].append(expr)
        return out, ctx

    @visitor(exp.CTE)
    def visit(self, ast, ctx=None, **kwargs):
        # note that by default, snowflake allow recursively define CTE tables without recusive declaration
        alias, _ = self.visit(ast.args['alias'], ctx=ctx, **kwargs)
        if 'columns' in alias:
            ctx[alias['table']] = set(alias['columns'])
        else:
            # get attributes from subquery
            atom_subquery = ast.args['this']
            while isinstance(atom_subquery, exp.Union | exp.Bracket):
                atom_subquery = atom_subquery.args['this']
            ctx[alias['table']] = set()
            drop_table = False
            for expr in atom_subquery.expressions:
                if 'alias' in expr.args:
                    ctx[alias['table']].add(expr.args['alias'].name)
                elif 'this' in expr.args:
                    ctx[alias['table']].add(expr.args['this'].name)
                elif isinstance(expr, exp.Star):
                    drop_table = True
                    break
                else:
                    raise NotImplementedError
            if drop_table:
                ctx.pop(alias['table'])
        subquery, subquery_ctx = self.visit(ast.args['this'], ctx=ctx, **kwargs)

        # update ctx for WTIH
        if 'columns' in alias:
            ctx[alias['table']] = set(alias['columns'])
        else:
            ctx[alias['table']] = set()

            atomic_subquery = self.get_atom_subquery(subquery)
            for column in atomic_subquery['select']:
                if 'alias' in column:
                    ctx[alias['table']].add(column['alias'])
                elif isinstance(column, str):
                    if column.endswith('.*'):
                        table, _ = column.rsplit('.', 1)
                        ctx[alias['table']] |= subquery_ctx[table]
                    elif column == '*':
                        tables = [atomic_subquery['from']]
                        if 'join' in atomic_subquery:
                            tables += [join_stmt['join'] for join_stmt in atomic_subquery['join']]
                        for table in tables:
                            table = table['alias' if 'alias' in table else 'table']
                            ctx[alias['table']] |= ctx[table]
                    elif '.' in column:
                        column = column.rsplit('.', 1)[-1]
                        ctx[alias['table']].add(column)
                    else:
                        ctx[alias['table']].add(column)
                else:
                    raise NotImplementedError(column)

        return {'alias': alias, 'subquery': subquery}, ctx

    @visitor(exp.Union)
    def visit(self, ast, ctx=None, **kwargs):
        subqueries, _ = self.visit_exprs(ast.args, ctx=ctx, **kwargs)
        out = {'union': subqueries, 'distinct': ast.args['distinct']}
        return out, ctx

    @visitor(exp.All)
    def visit(self, ast, ctx=None, **kwargs):  # xx NOT IN (SELECT XX FROM ...)
        curr_ctx = {**self.with_ctx.copy(), **ctx}
        return {'subquery': self.visit(ast.args['this'], ctx=curr_ctx, **kwargs)[0]}, ctx

    @visitor(exp.Subquery)
    def visit(self, ast, ctx=None, **kwargs):
        # subquery widely appear, e.g., WHERE i."block_timestamp" BETWEEN (SELECT start_timestamp FROM date_range) AND (SELECT end_timestamp FROM date_range)
        curr_ctx = {**self.with_ctx.copy(), **ctx}
        subquery, _ = self.visit(ast.args['this'], ctx=curr_ctx, **kwargs)
        out = {'subquery': subquery}
        if ast.args.get('alias', None) is not None:
            out['alias'], _ = self.visit(ast.args['alias'], ctx=curr_ctx, **kwargs)
            ctx[out['alias']['table']] = set()
            for column in self.get_atom_subquery(subquery)['select']:
                if 'alias' in column:
                    ctx[out['alias']['table']].add(column['alias'])
                elif isinstance(column, str):
                    ctx[out['alias']['table']].add(column)
                else:
                    raise NotImplementedError(column)

        return out, ctx

    @visitor(exp.Tuple)
    def visit(self, ast, ctx=None, **kwargs):
        tuple = [self.visit(t, ctx=ctx, **kwargs)[0] for t in ast.args['expressions']]
        return tuple, ctx

    @visitor(exp.TableFromRows)
    def visit(self, ast, ctx=None, **kwargs):
        out = {}
        out['table'], _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        if ast.args.get('alias', None) is not None:
            out['alias'], _ = self.visit(ast.args['alias'], ctx=ctx, **kwargs)
            if ast.args.get('columns', None) is not None:
                out['alias']['columns'] = []
                for col in ast.args['columns']:
                    col, _ = self.visit(col, ctx=ctx, **kwargs)
                    out['alias']['columns'].append(col)
                ctx[out['alias']['table']] = set(out['alias']['columns'])
        return {"table_from_rows": out}, ctx

    @visitor(exp.Values)
    def visit(self, ast, ctx=None, **kwargs):
        tuples = [self.visit(t, ctx=ctx, **kwargs)[0] for t in ast.args['expressions']]
        alias = self.visit(ast.args['alias'], ctx=ctx, **kwargs)[0]
        ctx[alias['table']] = set(alias['columns'])
        return {**alias, 'values': tuples}, ctx

    @visitor(exp.Table)
    def visit(self, ast, ctx=None, **kwargs):
        names = []
        for key in ['catalog', 'db', 'table', 'this']:
            if ast.args.get(key, None) is not None:
                names.append(ast.args[key])
        assert len(names) > 0, names
        table = [self.visit(name, ctx=ctx, **kwargs)[0] for name in names]
        table = '.'.join(table)
        out = {'table': table}

        table_ctx = {}
        if 'alias' in ast.args:
            out['alias'], _ = self.visit(ast.args['alias'], ctx=ctx, **kwargs)
            if table in ctx:
                table_ctx[out['alias']['table']] = ctx[table]
            elif str.upper(table) in ctx:
                table_ctx[out['alias']['table']] = ctx[str.upper(table)]
            else:
                print(f"Unknown table {table} (It might be a recursive CTE!).")
                # raise ValueError(f"Unknown table {table}.")
            for attr in table_ctx[out['alias']['table']]:
                self.register_attr_alias(f"{table}.{attr}", f"{out['alias']['table']}.{attr}")
        else:
            if table in ctx:
                table_ctx[table] = ctx[table]
            elif str.upper(table) in ctx:
                table_ctx[table] = ctx[str.upper(table)]
        return out, table_ctx

    @visitor(exp.TableAlias)
    def visit(self, ast, ctx=None, **kwargs):
        alias, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'table': alias}
        if ast.args.get('columns', None) is not None:
            out['columns'] = [self.visit(column, ctx=ctx, **kwargs)[0] for column in ast.args['columns']]
        return out, ctx

    @visitor(exp.Distinct)
    def visit(self, ast, ctx=None, **kwargs):
        columns = [self.visit(opd, ctx=ctx, **kwargs)[0]
                   for opd in ast.args['expressions']]
        return {'distinct': True, 'columns': columns}, ctx

    @visitor(exp.Alias)
    def visit(self, ast, ctx=None, **kwargs):
        column, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        alias, _ = self.visit(ast.args['alias'], ctx=ctx, **kwargs)
        return {'column': column, 'alias': alias}, ctx

    @visitor(exp.Where)
    def visit(self, ast, ctx=None, **kwargs):
        return self.visit(ast.args['this'], ctx=ctx, **kwargs)[0], ctx

    @visitor(exp.Join)
    def visit(self, ast, ctx=None, **kwargs):
        stmt, ctx = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        join_type = 'join'
        if ast.args.get('side', None) is not None:  # OUTER
            join_type = f"{str.lower(ast.args['side'])}_join"
        elif ast.args.get('method', None) is not None:  # NATURAL
            join_type = "natural_join"
        elif ast.args.get('kind', None) is not None:  # CROSS/INNER
            join_type = f"{str.lower(ast.args['kind'])}_join"
        out = {join_type: stmt}
        if ast.args.get('on', None) is not None:
            out['on'] = self.visit(ast.args['on'], **kwargs)
        elif ast.args.get('using', None) is not None:
            out['using'] = [self.visit(key, **kwargs) for key in ast.args['using']]
        return out, ctx

    @visitor(exp.Group)
    def visit(self, ast, ctx=None, **kwargs):
        keys = [self.visit(key, ctx=ctx, **kwargs)[0] for key in ast.args['expressions']]
        return keys, ctx

    @visitor(exp.Having)
    def visit(self, ast, ctx=None, **kwargs):
        return self.visit(ast.args['this'], ctx=ctx, **kwargs)[0], ctx

    @visitor(exp.Order)
    def visit(self, ast, ctx=None, **kwargs):
        out = {}
        out['keys'] = [self.visit(key, ctx=ctx, **kwargs)[0]
                       for key in ast.args['expressions']]
        if ast.args.get('this', None) is not None:  # group_concat
            out['value'], _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        return out, ctx

    @visitor(exp.Ordered)
    def visit(self, ast, ctx=None, **kwargs):
        key, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'key': key}
        if ast.args.get('desc', None) is not None:
            out['asc'] = not ast.args['desc']
        if ast.args.get('nulls_first', False):
            out['nulls_first'] = True
        return out, ctx

    @visitor(exp.Limit)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        return {'limit': expr}, ctx

    @visitor(exp.Fetch)
    def visit(self, ast, ctx=None, **kwargs):
        first = ast.args['direction'] == "FIRST"
        count, _ = self.visit(ast.args['count'], ctx=ctx, **kwargs)
        return {'fetch': {'first': first, 'count': count}}, ctx

    ####################### conditional #########################

    @visitor(exp.Nullif)
    def visit(self, ast, ctx=None, **kwargs):
        exprs, _ = self.visit_exprs(ast.args, ctx=ctx, **kwargs)
        return {'nullif': exprs}, ctx

    @visitor(exp.Case)
    def visit(self, ast, ctx=None, **kwargs):
        if_stmts = [self.visit(stmt, ctx=ctx, **kwargs)[0] for stmt in ast.args['ifs']]
        if ast.args['this'] is None:
            out = {'ifs': if_stmts}
            if ast.args.get('default', None) is not None:
                out['else_stmt'], _ = self.visit(ast.args['default'], ctx=ctx, **kwargs)
            return {'case': out}, ctx
        else:
            expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
            return {'switch': {'value': expr, 'ifs': if_stmts}}, ctx

    @visitor(exp.If)
    def visit(self, ast, ctx=None, **kwargs):
        condition, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        then_stmt, _ = self.visit(ast.args['true'], ctx=ctx, **kwargs)
        out = {'if': condition, 'then': then_stmt}
        if 'false' in ast.args:
            else_stmt, _ = self.visit(ast.args['false'], ctx=ctx, **kwargs)
            out['else'] = else_stmt
        return out, ctx

    @visitor(exp.Coalesce)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]]
        exprs = exprs + [self.visit(expr, ctx=ctx, **kwargs)[0]
                         for expr in ast.args['expressions']]
        return {'coalesce': exprs}, ctx

    ####################### semi-structure #########################

    @visitor(exp.Array)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(expr, ctx=ctx, **kwargs)[0]
                 for expr in ast.args['expressions']]
        return {'array': exprs}, ctx

    @visitor(exp.ArrayConcat)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]]
        exprs += [self.visit(expr, ctx=ctx, **kwargs)[0]
                  for expr in ast.args['expressions']]
        return {'array_concat': exprs}, ctx

    @visitor(exp.ArraySize)
    def visit(self, ast, ctx=None, **kwargs):
        return {'array_size': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.JSONPathRoot)
    def visit(self, ast, ctx=None, **kwargs):
        assert len(ast.args) == 0, ast.args
        return None, ctx

    @visitor(exp.JSONPathKey)
    def visit(self, ast, ctx=None, **kwargs):
        return {'key': ast.args['this']}, ctx

    @visitor(exp.JSONPath)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(expr, ctx=ctx, **kwargs)[0]
                 for expr in ast.args['expressions']]
        return exprs[-1], ctx

    @visitor(exp.JSONExtract)
    def visit(self, ast, ctx=None, **kwargs):
        return {'json_extract': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.ArrayContains)
    def visit(self, ast, ctx=None, **kwargs):
        return {'array_contains': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.GenerateSeries)
    def visit(self, ast, ctx=None, **kwargs):
        start, _ = self.visit(ast.args['start'], ctx=ctx, **kwargs)
        end, _ = self.visit(ast.args['end'], ctx=ctx, **kwargs)
        return {'array_generate_range': [start, end]}, ctx

    @visitor(exp.Kwarg)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        return {'input': expr}, ctx

    @visitor(exp.ParseJSON)
    def visit(self, ast, ctx=None, **kwargs):
        return {'parse_json': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Explode)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]]
        exprs += [self.visit(expr, ctx=ctx, **kwargs)[0]
                  for expr in ast.args['expressions']]
        return {'explode': exprs}, ctx

    @visitor(exp.Lateral)
    def visit(self, ast, ctx=None, **kwargs):
        table, _ = self.visit(ast.args['this'], ctx=kwargs['from_ctx'], **kwargs)
        alias, _ = self.visit(ast.args['alias'], ctx=kwargs['from_ctx'], **kwargs)
        if alias is not None:
            if 'columns' in alias:
                ctx[alias['table']] = set(alias['columns'])
            else:
                ctx[alias['table']] = set(DEFAULT_ATTRS)
        return {'lateral': {'table': table, 'alias': alias}}, ctx

    @visitor(exp.StPoint)
    def visit(self, ast, ctx=None, **kwargs):
        return {'st_point': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.StDistance)
    def visit(self, ast, ctx=None, **kwargs):
        return {'st_distance': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    ####################### string #########################

    @visitor(exp.Right)
    def visit(self, ast, ctx=None, **kwargs):
        return {'right': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.StrPosition)
    def visit(self, ast, ctx=None, **kwargs):
        substring, _ = self.visit(ast.args['substr'], ctx=ctx, **kwargs)
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        return {'str_position': {'substring': substring, 'string': expr}}, ctx

    @visitor(exp.Substring)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args[key], ctx=ctx, **kwargs)[0]
                 for key in ['this', 'start', 'length']]
        return {'substr': exprs}, ctx

    @visitor(exp.Split)
    def visit(self, ast, ctx=None, **kwargs):
        exprs, _ = self.visit_exprs(ast.args, ctx=ctx, **kwargs)
        return {'split': exprs}, ctx

    @visitor(exp.Replace)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args[key], ctx=ctx, **kwargs)[0]
                 for key in ['this', 'expression', 'replacement']]
        return {'replace': exprs}, ctx

    @visitor(exp.Trim)
    def visit(self, ast, ctx=None, **kwargs):
        return {'trim': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.DPipe)
    def visit(self, ast, ctx=None, **kwargs):
        return {'concat': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Concat)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(expr, ctx=ctx, **kwargs)[0]
                 for expr in ast.args['expressions']]
        return {'concat': exprs}, ctx

    @visitor(exp.SplitPart)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args[key], ctx=ctx, **kwargs)[0]
                 for key in ['this', 'delimiter', 'part_index']]
        return {'split_part': exprs}, ctx

    @visitor(exp.Pad)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        pattern, _ = self.visit(ast.args['fill_pattern'], ctx=ctx, **kwargs)
        pattern_num, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        out = {'value': expr, 'pattern': pattern, 'pattern_num': pattern_num}
        if ast.args.get('is_left', None) is not None:
            out['is_left'] = ast.args['is_left']
        return {'pad': out}, ctx

    @visitor(exp.Lower)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        return {'lower': expr}, ctx

    @visitor(exp.Length)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        return {'length': expr}, ctx

    @visitor(exp.Upper)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        return {'upper': expr}, ctx

    ####################### time #########################

    @visitor(exp.Year)
    def visit(self, ast, ctx=None, **kwargs):
        return {'year': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Month)
    def visit(self, ast, ctx=None, **kwargs):
        return {'month': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Day)
    def visit(self, ast, ctx=None, **kwargs):
        return {'day': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.TsOrDsToDate)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'value': expr}
        if ast.args.get('format', None) is not None:
            out['format'], _ = self.visit(ast.args['format'], ctx=ctx, **kwargs)
        return {'ts_to_date': out}, ctx

    @visitor(exp.TimeToStr)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        format, _ = self.visit(ast.args['format'], ctx=ctx, **kwargs)
        return {'time_to_str': {'value': expr, 'format': format}}, ctx

    @visitor(exp.Extract)
    def visit(self, ast, ctx=None, **kwargs):
        part, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        expr, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        return {'extract': {'value': expr, 'part': part}}, ctx

    @visitor(exp.DateAdd)
    def visit(self, ast, ctx=None, **kwargs):
        value, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        offset, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        unit, _ = self.visit(ast.args['unit'], ctx=ctx, **kwargs)
        return {'date_add': {'value': value, 'offset': offset, 'unit': unit}}, ctx

    @visitor(exp.DateDiff)
    def visit(self, ast, ctx=None, **kwargs):
        value, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        offset, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        unit, _ = self.visit(ast.args['unit'], ctx=ctx, **kwargs)
        return {'date_diff': {'value': value, 'offset': offset, 'unit': unit}}, ctx

    @visitor(exp.DateFromParts)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args[key], ctx=ctx, **kwargs)[0]['number']
                 for key in ['year', 'month', 'day']]
        return {'date': exprs}, ctx

    @visitor(exp.Anonymous)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(expr, ctx=ctx, **kwargs)[0]
                 for expr in ast.args['expressions']]
        return {ast.args['this']: exprs}, ctx

    @visitor(exp.TimestampTrunc)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'value': expr}
        if ast.args.get('scale', None) is not None:
            out['scale'], _ = self.visit(ast.args['scale'], ctx=ctx, **kwargs)
        if ast.args.get('unit', None) is not None:
            out['unit'], _ = self.visit(ast.args['unit'], ctx=ctx, **kwargs)
        return {'timestamp_trunc': out}, ctx

    @visitor(exp.UnixToTime)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        scale, _ = self.visit(ast.args['scale'], ctx=ctx, **kwargs)
        return {'unix_to_time': {'value': expr, 'scale': scale}}, ctx

    @visitor(exp.TimeToUnix)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        return {'time_to_unix': {'value': expr}}, ctx

    ####################### type conversion #########################

    @visitor(exp.ToNumber)
    def visit(self, ast, ctx=None, **kwargs):
        return {'to_number': {'value': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}}, ctx

    @visitor(exp.ToChar)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'value': expr}
        if ast.args.get('format', None) is not None:
            out['format'], _ = self.visit(ast.args['format'], ctx=ctx, **kwargs)
        return {'to_char': out}, ctx

    @visitor(exp.StrToTime)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        format, _ = self.visit(ast.args['format'], ctx=ctx, **kwargs)
        return {'str_to_time': {'value': expr, 'format': format}}, ctx

    @visitor(exp.Boolean)
    def visit(self, ast, ctx=None, **kwargs):
        return ast.args['this'], ctx

    @visitor(exp.DataType)
    def visit(self, ast, ctx=None, **kwargs):
        return ast.args['this'].name, ctx

    @visitor(exp.Cast)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        type, _ = self.visit(ast.args['to'], ctx=ctx, **kwargs)
        return {'cast': {'value': expr, 'type': type}}, ctx

    @visitor(exp.TryCast)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        type, _ = self.visit(ast.args['to'], ctx=ctx, **kwargs)
        return {'try_cast': {'value': expr, 'type': type}}, ctx

    ####################### predicate/expression #########################

    def visit_exprs(self, args, ctx=None, **kwargs):
        exprs = [args['this'], args['expression']]
        exprs = [self.visit(expr, ctx=ctx, **kwargs)[0] for expr in exprs]
        return exprs, ctx

    @visitor(exp.Least)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]]
        exprs = exprs + [self.visit(expr, ctx=ctx, **kwargs)[0]
                         for expr in ast.args['expressions']]
        return {'least': exprs}, ctx

    @visitor(exp.Greatest)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]]
        exprs = exprs + [self.visit(expr, ctx=ctx, **kwargs)[0]
                         for expr in ast.args['expressions']]
        return {'greatest': exprs}, ctx

    @visitor(exp.Sqrt)
    def visit(self, ast, ctx=None, **kwargs):
        return {'sqrt': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Asin)
    def visit(self, ast, ctx=None, **kwargs):
        return {'asin': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Sin)
    def visit(self, ast, ctx=None, **kwargs):
        return {'sin': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Cos)
    def visit(self, ast, ctx=None, **kwargs):
        return {'cos': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Radians)
    def visit(self, ast, ctx=None, **kwargs):
        return {'radians': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Pow)
    def visit(self, ast, ctx=None, **kwargs):
        return {'power': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Log)
    def visit(self, ast, ctx=None, **kwargs):
        return {'log': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Round)
    def visit(self, ast, ctx=None, **kwargs):
        expr, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'value': expr}
        if ast.args.get('decimals', None) is not None:
            out['decimals'], _ = self.visit(ast.args['decimals'], ctx=ctx, **kwargs)
        return {'extract': out}, ctx

    @visitor(exp.Floor)
    def visit(self, ast, ctx=None, **kwargs):
        return {'floor': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Is)
    def visit(self, ast, ctx=None, **kwargs):
        return {'is': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Not)
    def visit(self, ast, ctx=None, **kwargs):
        return {'not': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.And)
    def visit(self, ast, ctx=None, **kwargs):
        return {'and': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Or)
    def visit(self, ast, ctx=None, **kwargs):
        return {'or': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.EQ)
    def visit(self, ast, ctx=None, **kwargs):
        return {'=': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.NEQ)
    def visit(self, ast, ctx=None, **kwargs):
        return {'!=': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Add)
    def visit(self, ast, ctx=None, **kwargs):
        return {'+': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Neg)
    def visit(self, ast, ctx=None, **kwargs):
        return {'-': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Sub)
    def visit(self, ast, ctx=None, **kwargs):
        return {'-': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Mul)
    def visit(self, ast, ctx=None, **kwargs):
        return {'*': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Div)
    def visit(self, ast, ctx=None, **kwargs):
        return {'/': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.IntDiv)
    def visit(self, ast, ctx=None, **kwargs):
        return {'//': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.GTE)
    def visit(self, ast, ctx=None, **kwargs):
        return {'>=': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.GT)
    def visit(self, ast, ctx=None, **kwargs):
        return {'>': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Corr)
    def visit(self, ast, ctx=None, **kwargs):
        return {'corr': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    def visit(self, ast, ctx=None, **kwargs):
        return {'>': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.LTE)
    def visit(self, ast, ctx=None, **kwargs):
        return {'<=': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.LT)
    def visit(self, ast, ctx=None, **kwargs):
        return {'<': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Between)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [ast.args['this'], ast.args['low'], ast.args['high']]
        exprs = [self.visit(expr, ctx=ctx, **kwargs)[0] for expr in exprs]
        return {'between': exprs}, ctx

    @visitor(exp.Abs)
    def visit(self, ast, ctx=None, **kwargs):
        return {'abs': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Ln)
    def visit(self, ast, ctx=None, **kwargs):
        return {'ln': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Exp)
    def visit(self, ast, ctx=None, **kwargs):
        return {'exp': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Exists)
    def visit(self, ast, ctx=None, **kwargs):
        subquery, _ = self.visit(ast.args['this'], ctx={**self.with_ctx.copy(), **ctx}, **kwargs)
        return {'exists': {'subquery': subquery}}, ctx

    @visitor(exp.In)
    def visit(self, ast, ctx=None, **kwargs):
        lhs, _ = self.visit_exprs(ast.args['this'], ctx=ctx, **kwargs)
        out = [lhs]
        if ast.args.get('expressions', None) is not None:
            rhs = [self.visit_exprs(expr, ctx=ctx, **kwargs)[0]
                   for expr in ast.args['expressions']]
            out.append(rhs)
        if ast.args.get('query', None) is not None:
            subquery, _ = self.visit(ast.args['query'], ctx={**self.with_ctx.copy(), **ctx}, **kwargs)
            out.append(subquery)
        return {'in': out}, ctx

    ####################### aggregation #########################

    @visitor(exp.ArgMax)
    def visit(self, ast, ctx=None, **kwargs):
        return {'max_by': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.ArgMin)
    def visit(self, ast, ctx=None, **kwargs):
        return {'min_by': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Corr)
    def visit(self, ast, ctx=None, **kwargs):
        return {'corr': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.CountIf)
    def visit(self, ast, ctx=None, **kwargs):
        return {'count_if': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Count)
    def visit(self, ast, ctx=None, **kwargs):
        return {'count': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Min)
    def visit(self, ast, ctx=None, **kwargs):
        return {'min': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Max)
    def visit(self, ast, ctx=None, **kwargs):
        return {'max': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Sum)
    def visit(self, ast, ctx=None, **kwargs):
        return {'sum': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Avg)
    def visit(self, ast, ctx=None, **kwargs):
        return {'avg': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Median)
    def visit(self, ast, ctx=None, **kwargs):
        return {'median': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.ArrayAgg)
    def visit(self, ast, ctx=None, **kwargs):
        return {'array_agg': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.GroupConcat)
    def visit(self, ast, ctx=None, **kwargs):
        key, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        separator, _ = self.visit(ast.args['separator'], ctx=ctx, **kwargs)
        return {'list_agg': {'order_by': key, 'separator': separator}}, ctx

    ####################### window #########################

    @visitor(exp.DenseRank)
    def visit(self, ast, ctx=None, **kwargs):
        return {'dense_rank': None}, ctx

    @visitor(exp.RowNumber)
    def visit(self, ast, ctx=None, **kwargs):
        return {'row_number': None}, ctx

    @visitor(exp.Rank)
    def visit(self, ast, ctx=None, **kwargs):
        return {'rank': None}, ctx

    @visitor(exp.Lag)
    def visit(self, ast, ctx=None, **kwargs):
        return {'lag': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Lead)
    def visit(self, ast, ctx=None, **kwargs):
        return {'lead': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Ntile)
    def visit(self, ast, ctx=None, **kwargs):
        return {'ntile': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.PercentileCont)
    def visit(self, ast, ctx=None, **kwargs):
        return {'percentile_cont': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.IgnoreNulls)
    def visit(self, ast, ctx=None, **kwargs):
        return {'ignore_nulls': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Window)
    def visit(self, ast, ctx=None, **kwargs):
        value, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        out = {'value': value}
        if ast.args.get('over', None) == 'OVER':
            out['over'] = {}
            if ast.args.get('partition_by', None) is not None or len(ast.args['partition_by']) > 0:
                out['over']['partition_by'] = \
                    [self.visit(column, ctx=ctx, **kwargs)[0] for column in ast.args['partition_by']]
            if ast.args.get('order', None) is not None:
                out['over']['order_by'] = self.visit(ast.args['order'], ctx=ctx, **kwargs)[0]
        return {'window': out}, ctx

    @visitor(exp.Qualify)
    def visit(self, ast, ctx=None, **kwargs):
        return {'qualify': self.visit(ast.args['this'], ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.WithinGroup)
    def visit(self, ast, ctx=None, **kwargs):
        value, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        key, _ = self.visit(ast.args['expression'], ctx=ctx, **kwargs)
        return {'within_group': {'value': value, 'order_by': key}}, ctx

    ####################### constants #########################

    @visitor(exp.Literal)
    def visit(self, ast, ctx=None, **kwargs):
        if ast.args['is_string']:
            return {'literal': ast.args['this']}, ctx
        else:
            try:
                value = float(ast.args['this'])
            except:
                raise ValueError(ast.args['this'])
            if int(value) == value:
                value = int(value)
            return {'number': value}, ctx

    @visitor(exp.Identifier)
    def visit(self, ast, ctx=None, **kwargs):
        return ast.args['this'], ctx

    @visitor(exp.Column)
    def visit(self, ast, ctx=None, **kwargs):
        names = []
        for key in ['catalog', 'db', 'table']:
            if ast.args.get(key, None) is not None:
                names.append(ast.args[key])
        if len(names) > 0:
            table = []
            for name in names:
                name, _ = self.visit(name, ctx=ctx, **kwargs)
                table.append(name)
            table = '.'.join(table)
        column, _ = self.visit(ast.args['this'], ctx=ctx, **kwargs)
        if len(names) > 0:
            column = f'{table}.{column}'
        attr = self.get_attr_from_ctx(column, ctx)
        if attr is not None:
            self.register_involved_attr(attr)
        return column, ctx

    @visitor(exp.CurrentDate)
    def visit(self, ast, ctx=None, **kwargs):
        return {'current_data': None}, ctx

    @visitor(exp.Null)
    def visit(self, ast, ctx=None, **kwargs):
        return {'null': None}, ctx

    @visitor(exp.Star)
    def visit(self, ast, ctx=None, **kwargs):
        return '*', ctx

    @visitor(exp.Var)
    def visit(self, ast, ctx=None, **kwargs):
        return ast.args['this'], ctx

    ####################### regexp #########################

    @visitor(exp.RegexpLike)
    def visit(self, ast, ctx=None, **kwargs):
        return {'regexp_like': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.RegexpReplace)
    def visit(self, ast, ctx=None, **kwargs):
        exprs = [self.visit(ast.args[key], ctx=ctx, **kwargs)[0]
                 for key in ['this', 'expression', 'replacement']]
        return {'regexp_replace': exprs}, ctx

    @visitor(exp.ILike)
    def visit(self, ast, ctx=None, **kwargs):
        return {'ilike': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    @visitor(exp.Like)
    def visit(self, ast, ctx=None, **kwargs):
        return {'like': self.visit_exprs(ast.args, ctx=ctx, **kwargs)[0]}, ctx

    ####################### intermediate #########################

    @visitor(exp.Bracket)
    def visit(self, ast, ctx=None, **kwargs):
        return self.visit(ast.args['this'], ctx=ctx, **kwargs)

    @visitor(exp.From)
    def visit(self, ast, ctx=None, **kwargs):
        return self.visit(ast.args['this'], ctx=ctx, **kwargs)

    @visitor(exp.Paren)
    def visit(self, ast, ctx=None, **kwargs):
        return self.visit(ast.args['this'], ctx=ctx, **kwargs)
