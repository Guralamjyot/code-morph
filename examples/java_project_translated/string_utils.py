"""
Simple string utility module for testing CodeMorph Java to Python translation.

Translated from: com/example/StringUtils.java
"""

import re
from typing import Optional

# Constants
MAX_LENGTH: int = 1000
DEFAULT_SEPARATOR: str = ","


def reverse(input_str: Optional[str]) -> Optional[str]:
    """
    Reverse a string.

    Args:
        input_str: The string to reverse

    Returns:
        The reversed string
    """
    if input_str is None:
        return None
    return input_str[::-1]


def is_palindrome(input_str: Optional[str]) -> bool:
    """
    Check if a string is a palindrome.

    Args:
        input_str: The string to check

    Returns:
        True if the string is a palindrome
    """
    if input_str is None or input_str == "":
        return False
    cleaned = re.sub(r'[^a-z0-9]', '', input_str.lower())
    reversed_str = reverse(cleaned)
    return cleaned == reversed_str


def count_char(input_str: Optional[str], target: str) -> int:
    """
    Count occurrences of a character in a string.

    Args:
        input_str: The string to search
        target: The character to count

    Returns:
        The number of occurrences
    """
    if input_str is None:
        return 0
    count = 0
    for c in input_str:
        if c == target:
            count += 1
    return count


def split_to_list(input_str: Optional[str], separator: str) -> list[str]:
    """
    Split a string by a separator and return as list.

    Args:
        input_str: The string to split
        separator: The separator

    Returns:
        List of substrings
    """
    result: list[str] = []
    if input_str is None or input_str == "":
        return result
    parts = input_str.split(separator)
    for part in parts:
        trimmed = part.strip()
        if trimmed:
            result.append(trimmed)
    return result


def join(items: Optional[list[str]], separator: str) -> str:
    """
    Join a list of strings with a separator.

    Args:
        items: The list of strings to join
        separator: The separator to use

    Returns:
        The joined string
    """
    if items is None or len(items) == 0:
        return ""
    return separator.join(items)


def truncate(input_str: Optional[str], max_length: int) -> Optional[str]:
    """
    Truncate a string to a maximum length.

    Args:
        input_str: The string to truncate
        max_length: Maximum length

    Returns:
        Truncated string with ellipsis if needed

    Raises:
        ValueError: if max_length is negative
    """
    if max_length < 0:
        raise ValueError("max_length cannot be negative")
    if input_str is None or len(input_str) <= max_length:
        return input_str
    if max_length <= 3:
        return input_str[:max_length]
    return input_str[:max_length - 3] + "..."


if __name__ == "__main__":
    # Simple test
    print(f"reverse('hello') = {reverse('hello')}")
    print(f"is_palindrome('racecar') = {is_palindrome('racecar')}")
    print(f"count_char('hello world', 'l') = {count_char('hello world', 'l')}")
    print(f"split_to_list('a, b, c', ',') = {split_to_list('a, b, c', ',')}")
    print(f"join(['a', 'b', 'c'], '-') = {join(['a', 'b', 'c'], '-')}")
    print(f"truncate('hello world', 8) = {truncate('hello world', 8)}")
