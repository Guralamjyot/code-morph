package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple string utility class for testing CodeMorph Java to Python translation.
 */
public class StringUtils {

    // Constants
    public static final int MAX_LENGTH = 1000;
    public static final String DEFAULT_SEPARATOR = ",";

    /**
     * Reverse a string.
     *
     * @param input The string to reverse
     * @return The reversed string
     */
    public static String reverse(String input) {
        if (input == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder(input);
        return sb.reverse().toString();
    }

    /**
     * Check if a string is a palindrome.
     *
     * @param input The string to check
     * @return true if the string is a palindrome
     */
    public static boolean isPalindrome(String input) {
        if (input == null || input.isEmpty()) {
            return false;
        }
        String cleaned = input.toLowerCase().replaceAll("[^a-z0-9]", "");
        String reversed = reverse(cleaned);
        return cleaned.equals(reversed);
    }

    /**
     * Count occurrences of a character in a string.
     *
     * @param input The string to search
     * @param target The character to count
     * @return The number of occurrences
     */
    public static int countChar(String input, char target) {
        if (input == null) {
            return 0;
        }
        int count = 0;
        for (char c : input.toCharArray()) {
            if (c == target) {
                count++;
            }
        }
        return count;
    }

    /**
     * Split a string by a separator and return as list.
     *
     * @param input The string to split
     * @param separator The separator
     * @return List of substrings
     */
    public static List<String> splitToList(String input, String separator) {
        List<String> result = new ArrayList<>();
        if (input == null || input.isEmpty()) {
            return result;
        }
        String[] parts = input.split(separator);
        for (String part : parts) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty()) {
                result.add(trimmed);
            }
        }
        return result;
    }

    /**
     * Join a list of strings with a separator.
     *
     * @param items The list of strings to join
     * @param separator The separator to use
     * @return The joined string
     */
    public static String join(List<String> items, String separator) {
        if (items == null || items.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            sb.append(items.get(i));
            if (i < items.size() - 1) {
                sb.append(separator);
            }
        }
        return sb.toString();
    }

    /**
     * Truncate a string to a maximum length.
     *
     * @param input The string to truncate
     * @param maxLength Maximum length
     * @return Truncated string with ellipsis if needed
     * @throws IllegalArgumentException if maxLength is negative
     */
    public static String truncate(String input, int maxLength) {
        if (maxLength < 0) {
            throw new IllegalArgumentException("maxLength cannot be negative");
        }
        if (input == null || input.length() <= maxLength) {
            return input;
        }
        if (maxLength <= 3) {
            return input.substring(0, maxLength);
        }
        return input.substring(0, maxLength - 3) + "...";
    }
}
