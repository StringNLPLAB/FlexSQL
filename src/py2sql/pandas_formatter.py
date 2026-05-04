# -*- coding:utf-8 -*-

from copy import deepcopy

import libcst as cst
from libcst import *

EQ = AssignEqual(whitespace_before=SimpleWhitespace(""), whitespace_after=SimpleWhitespace(""))
BY = Name(value='by')
TRUE = Name(value='True')
FALSE = Name(value='False')
ASCENDING = Name(value='ascending')
AS_INDEX = Name(value='as_index')


class GroupByTransformer(cst.CSTTransformer):
    def leave_Call(self, original_node: Call, updated_node: Call) -> Call:
        # Check if the call is a .groupby call
        if isinstance(original_node.func, cst.Attribute):
            if original_node.func.attr.value == "groupby":
                if isinstance(updated_node.args[0].value, Name | SimpleString):
                    first_arg = Arg(keyword=BY, equal=EQ, value=List(elements=[cst.Element(value=original_node.args[0].value)]))
                    updated_node = updated_node.with_changes(args=[first_arg, *updated_node.args[1:]])
                if len(updated_node.args) == 1:
                    new_arg = Arg(keyword=AS_INDEX, equal=EQ, value=TRUE)
                    updated_node = updated_node.with_changes(args=[updated_node.args[0], new_arg])
                first_arg = updated_node.args[0].with_changes(keyword=BY, equal=EQ)
                updated_node = updated_node.with_changes(args=[first_arg, *updated_node.args[1:]])
                return updated_node
        return updated_node


class OrderByTransformer(cst.CSTTransformer):
    def leave_Call(self, original_node: Call, updated_node: Call) -> Call:
        # Check if the call is a .groupby call
        if isinstance(original_node.func, cst.Attribute):
            if original_node.func.attr.value == "sort_values":
                if isinstance(updated_node.args[0].value, Name | SimpleString):
                    first_arg = Arg(keyword=BY, equal=EQ, value=List(elements=[cst.Element(value=original_node.args[0].value)]))
                    if len(updated_node.args) == 2 and isinstance(updated_node.args[1].value, Name):
                        second_arg = Arg(keyword=ASCENDING, equal=EQ, value=List(elements=[cst.Element(value=original_node.args[1].value)]))
                        updated_node = updated_node.with_changes(args=[first_arg, second_arg])
                    else:
                        updated_node = updated_node.with_changes(args=[first_arg, *updated_node.args[1:]])
                if len(updated_node.args) == 1:
                    new_arg = Arg(keyword=ASCENDING, equal=EQ, value=List(elements=[cst.Element(value=TRUE) for _ in range(len(updated_node.args[0].value.elements))]))
                    updated_node = updated_node.with_changes(args=[updated_node.args[0], new_arg])
                first_arg = updated_node.args[0].with_changes(keyword=BY, equal=EQ)
                if len(updated_node.args) == 2 and isinstance(updated_node.args[1].value, Name):
                    second_arg = Arg(keyword=ASCENDING, equal=EQ, value=List(elements=[cst.Element(value=original_node.args[1].value) for _ in range(len(updated_node.args[0].value.elements))]))
                    updated_node = updated_node.with_changes(args=[first_arg, second_arg])
                else:
                    updated_node = updated_node.with_changes(args=[first_arg, *updated_node.args[1:]])
                return updated_node
        return updated_node


def pandas_format(code):
    ast = cst.parse_module(code)

    def __format(ast, formatter):
        ast_copy = deepcopy(ast)

        try:
            ast = ast.visit(formatter())
            return ast
        except Exception as err:
            # cannot format this code, use the old ast
            return ast_copy

    ast = __format(ast, GroupByTransformer)
    ast = __format(ast, OrderByTransformer)
    return ast.code


if __name__ == '__main__':
    code = """
ts.sort_values(date_col, ascending=False)
df.sort_values(["UNINSURED_PCT", "UNINSURED_PCT"])
df.sort_values("UNINSURED_PCT")
df.sort_values(by="UNINSURED_PCT", ascending=False)
df.sort_values(["UNINSURED_PCT", "UNINSURED_PCT"], ascending=False)
df.sort_values(["UNINSURED_PCT", "UNINSURED_PCT"], ascending=True)

# Original code
df.groupby(by=["ID_RSSD", "VARIABLE"])
df.groupby("ID_RSSD")
df.groupby("ID_RSSD", as_index=False)
df.groupby(["ID_RSSD", "VARIABLE"], as_index=False)
df.groupby(["ID_RSSD", "VARIABLE"], as_index=True)
"""
    ast = cst.parse_module(code)
    ast = ast.visit(GroupByTransformer())
    ast = ast.visit(OrderByTransformer())
    print(ast.code)
