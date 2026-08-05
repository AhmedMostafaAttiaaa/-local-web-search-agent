"""Calculator tool: safely evaluate a math expression locally (no network).

LLMs are unreliable at arithmetic; this gives the agent an exact, offline way to
compute values (unit conversions, totals, percentages) instead of guessing.

Safety: we do NOT use eval(). The expression is parsed to an AST and only a small
allow-list of numeric operators and functions is permitted, so arbitrary code
cannot run.
"""
from __future__ import annotations

import ast
import math
import operator

# Binary operators the calculator understands.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Unary operators (e.g. -3, +3).
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Named constants and single-argument functions the model may reference.
_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an allow-listed AST node, or raise ValueError."""
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError("unsupported or unsafe expression element")


def calculator(expression: str) -> str:
    """Evaluate a math ``expression`` and return the result as a string.

    Never raises: on any problem it returns a readable ``[calculator error] ...``
    string so the agent can read the failure and react.

    Args:
        expression: A math expression, e.g. ``"1280 * 30.9"`` or ``"sqrt(2) / 2"``.
            Supports + - * / // % **, parentheses, the constants pi/e/tau, and the
            functions sqrt, abs, round, floor, ceil, log, log10, sin, cos, tan.

    Returns:
        The computed value as a string, or a ``[calculator error]`` message.
    """
    expr = (expression or "").strip()
    if not expr:
        return "[calculator error] Empty expression."
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return "[calculator error] Division by zero."
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"[calculator error] Could not evaluate {expr!r}: {exc}"
    # Present whole-number floats cleanly (4.0 -> "4").
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
