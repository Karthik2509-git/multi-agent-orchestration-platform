"""Tests for the Calculator tool."""

import pytest

from src.app.tools.calculator import CalculatorTool


@pytest.fixture
def calculator() -> CalculatorTool:
    return CalculatorTool()


@pytest.mark.asyncio
async def test_calculator_basic_operations(calculator: CalculatorTool) -> None:
    """Test standard arithmetic operations."""
    res1 = await calculator.execute(expression="25 * 17")
    assert res1.success is True
    assert res1.data["result"] == 425

    res2 = await calculator.execute(expression="(1000 / 8) + 42")
    assert res2.success is True
    assert res2.data["result"] == 167.0

    res3 = await calculator.execute(expression="2 ** 10")
    assert res3.success is True
    assert res3.data["result"] == 1024

    res4 = await calculator.execute(expression="100 // 9")
    assert res4.success is True
    assert res4.data["result"] == 11

    res5 = await calculator.execute(expression="100 % 9")
    assert res5.success is True
    assert res5.data["result"] == 1


@pytest.mark.asyncio
async def test_calculator_negative_numbers(calculator: CalculatorTool) -> None:
    """Test unary operators and negative results."""
    res = await calculator.execute(expression="-50 + 20")
    assert res.success is True
    assert res.data["result"] == -30


@pytest.mark.asyncio
async def test_calculator_division_by_zero(calculator: CalculatorTool) -> None:
    """Test graceful handling of division and modulo by zero."""
    res = await calculator.execute(expression="100 / 0")
    assert res.success is False
    assert "Division by zero" in res.error

    res_mod = await calculator.execute(expression="100 % 0")
    assert res_mod.success is False
    assert "Division by zero" in res_mod.error


@pytest.mark.asyncio
async def test_calculator_invalid_syntax_and_types(calculator: CalculatorTool) -> None:
    """Test syntax errors, non-arithmetic syntax, and invalid inputs."""
    res1 = await calculator.execute(expression="2 + * 3")
    assert res1.success is False
    assert "Calculation error" in res1.error

    res2 = await calculator.execute(expression="")
    assert res2.success is False
    assert "non-empty string" in res2.error

    res3 = await calculator.execute(expression="import os; os.system('echo hi')")
    assert res3.success is False
    assert "Calculation error" in res3.error

    res4 = await calculator.execute(expression="print('hello')")
    assert res4.success is False


@pytest.mark.asyncio
async def test_calculator_power_abuse_protection(calculator: CalculatorTool) -> None:
    """Test that enormous exponents are rejected to protect CPU and memory."""
    res = await calculator.execute(expression="9 ** 999999")
    assert res.success is False
    assert "maximum allowed limit" in res.error
