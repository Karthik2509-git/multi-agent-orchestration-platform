"""Safe mathematical expression calculator tool using AST parsing."""

import ast
import operator
from typing import Any, Dict, Union

from src.app.core.logging import get_logger
from src.app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class CalculatorTool(BaseTool):
    """Safely evaluates mathematical expressions without eval() or arbitrary code execution."""

    name: str = "calculator"
    description: str = (
        "Calculates the result of a mathematical expression. "
        "Supports standard arithmetic: +, -, *, /, //, %, ** and parentheses. "
        "Example: '25 * 17' or '(1000 / 8) + 42'."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '(1000 / 8) + 42'",
            }
        },
        "required": ["expression"],
    }

    _OPERATORS: Dict[Any, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            # Prevent resource exhaustion via giant exponents
            if op_type == ast.Pow:
                if abs(right) > 1000:
                    raise ValueError("Exponent exceeds maximum allowed limit of 1000")
                if left > 10000 and right > 100:
                    raise ValueError("Base/exponent calculation too large")

            # Check division / modulo by zero
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division or modulo by zero is undefined")

            return self._OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self._OPERATORS[op_type](operand)

        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute calculation safely."""
        expression = kwargs.get("expression", "")
        if not expression or not isinstance(expression, str):
            return ToolResult(
                success=False,
                error="Invalid input: 'expression' must be a non-empty string",
            )

        trimmed = expression.strip()
        if len(trimmed) > 300:
            return ToolResult(
                success=False,
                error="Expression exceeds maximum character length of 300",
            )

        logger.info("Executing calculator tool with expression: %s", trimmed)

        try:
            parsed = ast.parse(trimmed, mode="eval")
            result = self._eval_node(parsed)
            return ToolResult(
                success=True,
                data={"expression": trimmed, "result": result},
            )
        except ZeroDivisionError as zde:
            logger.warning("Calculator ZeroDivisionError: %s", str(zde))
            return ToolResult(success=False, error="Calculation error: Division by zero")
        except (SyntaxError, ValueError, OverflowError) as err:
            logger.warning("Calculator evaluation error: %s", str(err))
            return ToolResult(success=False, error=f"Calculation error: {str(err)}")
        except Exception as ex:
            logger.error("Unexpected calculator error: %s", str(ex))
            return ToolResult(success=False, error=f"Unexpected calculation error: {str(ex)}")
