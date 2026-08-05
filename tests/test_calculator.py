"""Tests for the offline calculator tool (tools/calculator.py)."""
from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.calculator import calculator  # noqa: E402


def test_basic_arithmetic():
    assert calculator("2 + 3 * 4") == "14"


def test_whole_float_is_shown_as_int():
    assert calculator("10 / 2") == "5"


def test_parentheses_and_power():
    assert calculator("(1 + 2) ** 3") == "27"


def test_functions_and_constants():
    assert calculator("sqrt(16)") == "4"
    assert calculator("round(pi, 2)") == "3.14"


def test_division_by_zero_is_reported():
    assert calculator("1 / 0").startswith("[calculator error]")


def test_empty_expression_is_reported():
    assert calculator("   ").startswith("[calculator error]")


def test_unsafe_expression_is_rejected():
    # Attribute access / arbitrary names must not evaluate.
    assert calculator("__import__('os').system('echo hi')").startswith("[calculator error]")
    assert calculator("open('x')").startswith("[calculator error]")


if __name__ == "__main__":
    for expr in ("2 + 3 * 4", "sqrt(2)", "1 / 0", "open('x')"):
        print(f"{expr!r:40} -> {calculator(expr)}")
