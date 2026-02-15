"""
Tests for string_utils module.

Translated from: com/example/StringUtilsTest.java
"""

import pytest
from string_utils import (
    reverse,
    is_palindrome,
    count_char,
    split_to_list,
    join,
    truncate,
)


class TestReverse:
    def test_reverse_hello(self):
        assert reverse("hello") == "olleh"

    def test_reverse_empty(self):
        assert reverse("") == ""

    def test_reverse_none(self):
        assert reverse(None) is None


class TestIsPalindrome:
    def test_palindrome_simple(self):
        assert is_palindrome("racecar") is True

    def test_palindrome_with_spaces(self):
        assert is_palindrome("A man a plan a canal Panama") is True

    def test_not_palindrome(self):
        assert is_palindrome("hello") is False

    def test_empty_string(self):
        assert is_palindrome("") is False

    def test_none(self):
        assert is_palindrome(None) is False


class TestCountChar:
    def test_count_l_in_hello_world(self):
        assert count_char("hello world", "l") == 3

    def test_count_not_found(self):
        assert count_char("hello", "x") == 0

    def test_count_none(self):
        assert count_char(None, "a") == 0


class TestSplitToList:
    def test_split_comma_separated(self):
        result = split_to_list("a, b, c", ",")
        assert result == ["a", "b", "c"]

    def test_split_empty(self):
        assert split_to_list("", ",") == []

    def test_split_none(self):
        assert split_to_list(None, ",") == []


class TestJoin:
    def test_join_with_dash(self):
        items = ["a", "b", "c"]
        assert join(items, "-") == "a-b-c"

    def test_join_none(self):
        assert join(None, ",") == ""

    def test_join_empty_list(self):
        assert join([], ",") == ""


class TestTruncate:
    def test_truncate_long_string(self):
        assert truncate("hello world", 8) == "hello..."

    def test_truncate_short_string(self):
        assert truncate("hi", 10) == "hi"

    def test_truncate_none(self):
        assert truncate(None, 5) is None

    def test_truncate_negative_raises(self):
        with pytest.raises(ValueError):
            truncate("test", -1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
