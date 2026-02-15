"""
Tests for the calculator module.

These tests will be used to generate execution snapshots for I/O equivalence verification.
"""

import pytest

from calculator import (
    MAX_VALUE,
    MIN_VALUE,
    Calculator,
    add,
    calculate_average,
    divide,
    multiply,
    subtract,
)


class TestBasicOperations:
    """Test basic arithmetic operations."""

    def test_add_positive(self):
        assert add(2, 3) == 5

    def test_add_negative(self):
        assert add(-5, 3) == -2

    def test_add_zero(self):
        assert add(0, 0) == 0

    def test_subtract_positive(self):
        assert subtract(10, 4) == 6

    def test_subtract_negative(self):
        assert subtract(5, 10) == -5

    def test_multiply_positive(self):
        assert multiply(4, 5) == 20

    def test_multiply_by_zero(self):
        assert multiply(100, 0) == 0

    def test_multiply_negative(self):
        assert multiply(-3, 4) == -12

    def test_divide_positive(self):
        assert divide(10, 2) == 5.0

    def test_divide_with_remainder(self):
        assert divide(7, 2) == 3.5

    def test_divide_by_zero_raises_error(self):
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)


class TestBoundaryConditions:
    """Test boundary conditions and error cases."""

    def test_multiply_exceeds_max(self):
        with pytest.raises(ValueError):
            multiply(MAX_VALUE, 2)

    def test_multiply_exceeds_min(self):
        with pytest.raises(ValueError):
            multiply(MIN_VALUE, 2)


class TestAverage:
    """Test average calculation."""

    def test_average_simple(self):
        assert calculate_average([1, 2, 3, 4, 5]) == 3.0

    def test_average_single_value(self):
        assert calculate_average([42]) == 42.0

    def test_average_negative_values(self):
        assert calculate_average([-10, -20, -30]) == -20.0

    def test_average_empty_list_raises_error(self):
        with pytest.raises(ValueError):
            calculate_average([])


class TestCalculatorClass:
    """Test Calculator class with memory."""

    def test_initial_memory_is_zero(self):
        calc = Calculator()
        assert calc.get_memory() == 0

    def test_add_to_memory(self):
        calc = Calculator()
        calc.add_to_memory(10)
        assert calc.get_memory() == 10

    def test_add_to_memory_multiple_times(self):
        calc = Calculator()
        calc.add_to_memory(5)
        calc.add_to_memory(3)
        assert calc.get_memory() == 8

    def test_clear_memory(self):
        calc = Calculator()
        calc.add_to_memory(100)
        calc.clear_memory()
        assert calc.get_memory() == 0

    def test_compute_add(self):
        calc = Calculator()
        result = calc.compute(10, 5, "add")
        assert result == 15
        assert calc.get_memory() == 15

    def test_compute_subtract(self):
        calc = Calculator()
        result = calc.compute(10, 5, "subtract")
        assert result == 5
        assert calc.get_memory() == 5

    def test_compute_multiply(self):
        calc = Calculator()
        result = calc.compute(6, 7, "multiply")
        assert result == 42
        assert calc.get_memory() == 42

    def test_compute_divide(self):
        calc = Calculator()
        result = calc.compute(20, 4, "divide")
        assert result == 5.0
        assert calc.get_memory() == 5.0

    def test_compute_unknown_operation(self):
        calc = Calculator()
        with pytest.raises(ValueError):
            calc.compute(10, 5, "modulo")
