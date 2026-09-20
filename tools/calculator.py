import ast
import operator
from langchain_core.tools import tool


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    """
    Recursively evaluate a restricted Python arithmetic expression.

    Only numeric constants and operators explicitly listed above
    are allowed.
    """

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operator_function = _OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported operator.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        return operator_function(left, right)

    if isinstance(node, ast.UnaryOp):
        operator_function = _OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported unary operator.")

        return operator_function(_evaluate(node.operand))

    raise ValueError("Unsupported expression.")

@tool
def calculator(expression: str):
    """
    Safely evaluate a mathematical expression.

    Example:
        calculator("(5600 - 4250) / 4250 * 100")
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)

        return {
            "success": True,
            "expression": expression,
            "result": result,
        }

    except Exception as exc:
        return {
            "success": False,
            "expression": expression,
            "error": str(exc),
        }