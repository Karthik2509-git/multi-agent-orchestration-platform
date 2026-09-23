"""Local in-process MCP server implementation exposing safe demonstration tools."""

import ast
import operator
from typing import Any, Callable, Dict

from mcp.server.mcpserver import MCPServer

from src.app.core.logging import get_logger

logger = get_logger(__name__)

# Supported safe arithmetic operators
SAFE_OPERATORS: Dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

MAX_EXPONENT_CEILING = 1000


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node strictly restricted to safe arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_func = SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = _safe_eval_node(node.operand)
        return op_func(operand)

    if isinstance(node, ast.BinOp):
        op_func = SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT_CEILING:
                raise ValueError(
                    f"Exponent {right} exceeds safe calculation ceiling ({MAX_EXPONENT_CEILING})"
                )

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Division by zero is not permitted")

        return op_func(left, right)

    raise ValueError(f"Unsupported AST syntax node: {type(node).__name__}")


def evaluate_safe_expression(expression: str) -> float | int:
    """Safely parse and evaluate an arithmetic expression string."""
    clean_expr = expression.strip()
    if not clean_expr:
        raise ValueError("Expression cannot be empty")

    parsed_tree = ast.parse(clean_expr, mode="eval")
    result = _safe_eval_node(parsed_tree)

    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def create_local_mcp_server(name: str = "local") -> MCPServer:
    """Construct an in-process MCPServer instance configured with safe tools."""
    server = MCPServer(name)

    @server.tool(
        name="calculator",
        description=(
            "Safely evaluate arithmetic mathematical expressions (e.g. '100 * 4 + 25'). "
            "Supports addition (+), subtraction (-), multiplication (*), division (/), "
            "modulo (%), and exponentiation (**)."
        ),
    )
    def calculator(expression: str) -> str:
        """Safely evaluate arithmetic expression without eval/exec."""
        try:
            val = evaluate_safe_expression(expression)
            return str(val)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    @server.tool(
        name="text_stats",
        description=(
            "Calculate quantitative text statistics for a given string, including "
            "character count, word count, and line count."
        ),
    )
    def text_stats(text: str) -> str:
        """Calculate text metrics."""
        chars = len(text)
        words = len(text.split())
        lines = len(text.splitlines()) if text else 0
        return (
            f"Characters: {chars}, Words: {words}, Lines: {lines}, Non-whitespace: "
            f"{len(''.join(text.split()))}"
        )

    logger.debug("Created local MCPServer '%s' with registered tools", name)
    return server
