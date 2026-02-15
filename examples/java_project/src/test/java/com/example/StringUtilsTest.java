package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;

/**
 * Tests for StringUtils class.
 */
public class StringUtilsTest {

    @Test
    public void testReverse() {
        assertEquals("olleh", StringUtils.reverse("hello"));
        assertEquals("", StringUtils.reverse(""));
        assertNull(StringUtils.reverse(null));
    }

    @Test
    public void testIsPalindrome() {
        assertTrue(StringUtils.isPalindrome("racecar"));
        assertTrue(StringUtils.isPalindrome("A man a plan a canal Panama"));
        assertFalse(StringUtils.isPalindrome("hello"));
        assertFalse(StringUtils.isPalindrome(""));
        assertFalse(StringUtils.isPalindrome(null));
    }

    @Test
    public void testCountChar() {
        assertEquals(3, StringUtils.countChar("hello world", 'l'));
        assertEquals(0, StringUtils.countChar("hello", 'x'));
        assertEquals(0, StringUtils.countChar(null, 'a'));
    }

    @Test
    public void testSplitToList() {
        List<String> result = StringUtils.splitToList("a, b, c", ",");
        assertEquals(Arrays.asList("a", "b", "c"), result);

        assertTrue(StringUtils.splitToList("", ",").isEmpty());
        assertTrue(StringUtils.splitToList(null, ",").isEmpty());
    }

    @Test
    public void testJoin() {
        List<String> items = Arrays.asList("a", "b", "c");
        assertEquals("a-b-c", StringUtils.join(items, "-"));
        assertEquals("", StringUtils.join(null, ","));
    }

    @Test
    public void testTruncate() {
        assertEquals("hello...", StringUtils.truncate("hello world", 8));
        assertEquals("hi", StringUtils.truncate("hi", 10));
        assertNull(StringUtils.truncate(null, 5));
        assertThrows(IllegalArgumentException.class, () -> StringUtils.truncate("test", -1));
    }
}
