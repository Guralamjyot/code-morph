"""
Simple calculator module for testing CodeMorph.

This module demonstrates basic arithmetic operations that can be translated to Java.
"""


# Constants
MAX_VALUE = 1000000
MIN_VALUE = -1000000


def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of a and b
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """
    Subtract two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Difference of a and b
    """
    return a - b


def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Product of a and b
    """
    result = a * b
    if result > MAX_VALUE or result < MIN_VALUE:
        raise ValueError(f"Result {result} exceeds bounds [{MIN_VALUE}, {MAX_VALUE}]")
    return result


def divide(a: int, b: int) -> float:
    """
    Divide two integers.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient of a and b

    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def calculate_average(numbers: list[int]) -> float:
    """
    Calculate the average of a list of numbers.

    Args:
        numbers: List of integers

    Returns:
        Average value

    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")

    total = sum(numbers)
    return total / len(numbers)


class Calculator:
    """Calculator class with memory functionality."""

    def __init__(self):
        """Initialize calculator with zero memory."""
        self.memory = 0

    def add_to_memory(self, value: int) -> None:
        """
        Add a value to memory.

        Args:
            value: Value to add
        """
        self.memory += value

    def get_memory(self) -> int:
        """
        Get the current memory value.

        Returns:
            Current memory value
        """
        return self.memory

    def clear_memory(self) -> None:
        """Reset memory to zero."""
        self.memory = 0

    def compute(self, a: int, b: int, operation: str) -> int | float:
        """
        Perform a calculation and store result in memory.

        Args:
            a: First operand
            b: Second operand
            operation: Operation to perform (add, subtract, multiply, divide)

        Returns:
            Result of the operation

        Raises:
            ValueError: If operation is not recognized
        """
        if operation == "add":
            result = add(a, b)
        elif operation == "subtract":
            result = subtract(a, b)
        elif operation == "multiply":
            result = multiply(a, b)
        elif operation == "divide":
            result = divide(a, b)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        self.memory = int(result) if isinstance(result, int) else result
        return result


if __name__ == "__main__":
    # Simple test
    print(f"2 + 3 = {add(2, 3)}")
    print(f"10 - 4 = {subtract(10, 4)}")
    print(f"5 * 6 = {multiply(5, 6)}")
    print(f"20 / 4 = {divide(20, 4)}")
    print(f"Average of [1, 2, 3, 4, 5] = {calculate_average([1, 2, 3, 4, 5])}")

    calc = Calculator()
    calc.compute(10, 5, "add")
    print(f"Calculator memory: {calc.get_memory()}")
