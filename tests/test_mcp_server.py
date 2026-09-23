"""Tests for local in-process MCPServer and tool implementations."""

import pytest

from src.app.mcp.servers.local_tools import (
    create_local_mcp_server,
    evaluate_safe_expression,
)


def test_evaluate_safe_expression_valid_arithmetic():
    """Verify safe arithmetic evaluation with AST parser."""
    assert evaluate_safe_expression("10 + 20") == 30
    assert evaluate_safe_expression("100 / 4") == 25
    assert evaluate_safe_expression("2 ** 8") == 256
    assert evaluate_safe_expression("(10 + 5) * 2 - 6 / 2") == 27
    assert evaluate_safe_expression("-5 + 10") == 5


def test_evaluate_safe_expression_division_by_zero():
    """Verify division by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError, match="Division by zero"):
        evaluate_safe_expression("100 / 0")


def test_evaluate_safe_expression_power_ceiling():
    """Verify exponent ceiling prevents denial of service."""
    with pytest.raises(ValueError, match="exceeds safe calculation ceiling"):
        evaluate_safe_expression("2 ** 1001")


def test_evaluate_safe_expression_disallows_code_execution():
    """Verify code injection and arbitrary syntax is blocked."""
    with pytest.raises(ValueError, match="Unsupported AST syntax node"):
        evaluate_safe_expression("__import__('os').system('ls')")

    with pytest.raises(ValueError, match="Unsupported AST syntax node"):
        evaluate_safe_expression("open('/etc/passwd')")


def test_create_local_mcp_server_initialization():
    """Verify local MCPServer instantiation with proper name."""
    server = create_local_mcp_server(name="test_local")
    assert server.name == "test_local"
