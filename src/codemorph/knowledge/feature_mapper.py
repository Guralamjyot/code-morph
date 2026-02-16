"""
Feature mapping system for CodeMorph.

Defines language-specific transformation rules that guide LLM translation.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from codemorph.config.models import CodeFragment, LanguageType


@dataclass
class FeatureMappingRule:
    """
    A rule that guides translation of a specific language feature.

    Example:
        Python list comprehension -> Java Stream API
    """

    id: str
    name: str
    source_language: LanguageType
    target_language: LanguageType
    description: str
    premise: str  # What to look for in source code
    instruction: str  # How to translate it
    validation: str | None = None  # How to verify the translation
    examples: list[dict[str, str]] | None = None
    priority: int = 1  # Higher priority rules are checked first

    def check_premise(self, fragment: CodeFragment, source_ast: Any | None = None) -> bool:
        """
        Check if this rule applies to the given fragment.

        Args:
            fragment: The code fragment to check
            source_ast: Optional parsed AST for more sophisticated checks

        Returns:
            True if the rule applies
        """
        # Simple keyword-based check
        return self.premise.lower() in fragment.source_code.lower()

    def validate_translation(self, translated_code: str) -> bool:
        """
        Validate that the translation follows the rule.

        Args:
            translated_code: The translated code to validate

        Returns:
            True if validation passes or no validation defined
        """
        if not self.validation:
            return True

        # Simple keyword-based validation
        return self.validation.lower() in translated_code.lower()


class FeatureMapper:
    """Manages feature mapping rules for code translation."""

    def __init__(self):
        self.rules: dict[tuple[LanguageType, LanguageType], list[FeatureMappingRule]] = {}

    def load_rules_from_yaml(self, rules_file: Path):
        """
        Load rules from a YAML file.

        Args:
            rules_file: Path to the rules YAML file
        """
        with open(rules_file) as f:
            rules_data = yaml.safe_load(f)

        for rule_dict in rules_data.get("rules", []):
            rule = FeatureMappingRule(
                id=rule_dict["id"],
                name=rule_dict["name"],
                source_language=LanguageType(rule_dict["source_language"]),
                target_language=LanguageType(rule_dict["target_language"]),
                description=rule_dict.get("description", ""),
                premise=rule_dict["premise"],
                instruction=rule_dict["instruction"],
                validation=rule_dict.get("validation"),
                examples=rule_dict.get("examples"),
                priority=rule_dict.get("priority", 1),
            )
            self.add_rule(rule)

    def add_rule(self, rule: FeatureMappingRule):
        """Add a rule to the mapper."""
        key = (rule.source_language, rule.target_language)
        if key not in self.rules:
            self.rules[key] = []
        self.rules[key].append(rule)

        # Sort by priority (higher first)
        self.rules[key].sort(key=lambda r: r.priority, reverse=True)

    def get_applicable_rules(
        self,
        fragment: CodeFragment,
        source_language: LanguageType,
        target_language: LanguageType,
        source_ast: Any | None = None,
    ) -> list[FeatureMappingRule]:
        """
        Get all rules that apply to a fragment.

        Args:
            fragment: The fragment to check
            source_language: Source language
            target_language: Target language
            source_ast: Optional AST for sophisticated checking

        Returns:
            List of applicable rules, sorted by priority
        """
        key = (source_language, target_language)
        if key not in self.rules:
            return []

        applicable = []
        for rule in self.rules[key]:
            if rule.check_premise(fragment, source_ast):
                applicable.append(rule)

        return applicable

    def get_instructions_for_fragment(
        self,
        fragment: CodeFragment,
        source_language: LanguageType,
        target_language: LanguageType,
    ) -> list[str]:
        """
        Get translation instructions for a fragment.

        Args:
            fragment: The fragment to translate
            source_language: Source language
            target_language: Target language

        Returns:
            List of instruction strings for the LLM
        """
        rules = self.get_applicable_rules(fragment, source_language, target_language)
        return [rule.instruction for rule in rules]

    def validate_translation(
        self,
        fragment: CodeFragment,
        translated_code: str,
        source_language: LanguageType,
        target_language: LanguageType,
    ) -> tuple[bool, list[str]]:
        """
        Validate a translation against applicable rules.

        Args:
            fragment: Original fragment
            translated_code: The translation to validate
            source_language: Source language
            target_language: Target language

        Returns:
            Tuple of (all_valid, list of failed rules)
        """
        rules = self.get_applicable_rules(fragment, source_language, target_language)
        failed = []

        for rule in rules:
            if not rule.validate_translation(translated_code):
                failed.append(rule.id)

        return (len(failed) == 0, failed)


# =============================================================================
# Built-in Rules
# =============================================================================


def get_default_python_to_java_rules() -> list[FeatureMappingRule]:
    """Get default Python -> Java transformation rules."""
    return [
        FeatureMappingRule(
            id="PY_LIST_COMP",
            name="List Comprehension",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert Python list comprehensions to Java Streams or loops",
            premise=" for ",  # Comprehensions always contain ' for '
            instruction=(
                "Convert list comprehensions to Java Stream API or traditional loops. "
                "Example: [x for x in items] becomes items.stream().collect(Collectors.toList())"
            ),
            validation=None,  # Can translate to stream, loop, or other patterns
            priority=5,
        ),
        FeatureMappingRule(
            id="PY_DICT_COMP",
            name="Dictionary Comprehension",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert Python dict comprehensions to Java Streams or loops",
            premise=" for ",  # Comprehensions always contain ' for '
            instruction=(
                "Convert dictionary comprehensions to Java Stream API with Collectors.toMap() or loops. "
                "Example: {k: v for k, v in items} becomes items.stream().collect(Collectors.toMap(...))"
            ),
            validation=None,  # Can translate to stream, loop, HashMap, etc.
            priority=5,
        ),
        FeatureMappingRule(
            id="PY_CONTEXT_MGR",
            name="Context Manager (with statement)",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert with statements to try-with-resources",
            premise="with ",
            instruction=(
                "Convert 'with' statements to Java try-with-resources. "
                "Example: with open(file) as f: becomes try (FileReader f = new FileReader(file)) { ... }"
            ),
            validation=None,  # Not all 'with' translates to try-with-resources
            priority=8,
        ),
        FeatureMappingRule(
            id="PY_STRING_FORMAT",
            name="F-String Formatting",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert f-strings to String.format or concatenation",
            premise='f"',
            instruction=(
                "Convert f-strings to String.format() or StringBuilder. "
                'Example: f"Hello {name}" becomes String.format("Hello %s", name)'
            ),
            validation=None,  # Hard to validate
            priority=3,
        ),
        FeatureMappingRule(
            id="PY_NONE",
            name="None to null",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert Python None to Java null",
            premise="None",
            instruction="Replace 'None' with 'null'",
            validation=None,  # 'None' in docstrings/comments doesn't require 'null' in output
            priority=1,
        ),
        FeatureMappingRule(
            id="PY_BOOL",
            name="Boolean True/False",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert Python True/False to Java true/false",
            premise="True",
            instruction="Replace 'True' with 'true' and 'False' with 'false' (lowercase)",
            validation="true",
            priority=1,
        ),
        FeatureMappingRule(
            id="PY_LEN",
            name="len() function",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert len() to .length or .size()",
            premise="len(",
            instruction=(
                "Replace len(obj) with obj.length (for arrays) or obj.size() (for Collections)"
            ),
            validation=None,
            priority=2,
        ),
        FeatureMappingRule(
            id="PY_PRINT",
            name="print() function",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert print() to System.out.println()",
            premise="print(",
            instruction="Replace print(...) with System.out.println(...)",
            validation="System.out.println",
            priority=2,
        ),
        FeatureMappingRule(
            id="PY_RANGE",
            name="range() function",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert range() to for loop",
            premise="range(",
            instruction=(
                "Convert range(n) to a traditional for loop: for (int i = 0; i < n; i++)"
            ),
            validation="for (",
            priority=4,
        ),
        FeatureMappingRule(
            id="PY_LAMBDA",
            name="Lambda expressions",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert Python lambdas to Java lambdas",
            premise="lambda ",
            instruction=(
                "Convert Python lambda to Java lambda expression. "
                "Example: lambda x: x + 1 becomes (x) -> x + 1"
            ),
            validation="->",
            priority=6,
        ),
        FeatureMappingRule(
            id="PY_SELF",
            name="self parameter",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Remove self parameter, use this",
            premise="def ",
            instruction=(
                "Remove 'self' parameter from method definitions (it's implicit in Java). "
                "Use 'this' keyword when needed for clarity."
            ),
            validation=None,
            priority=7,
        ),
        FeatureMappingRule(
            id="PY_ISINSTANCE",
            name="isinstance() checks",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert isinstance to instanceof",
            premise="isinstance(",
            instruction="Replace isinstance(obj, Type) with (obj instanceof Type)",
            validation="instanceof",
            priority=3,
        ),
        FeatureMappingRule(
            id="PY_TRY_EXCEPT",
            name="Exception handling",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert try/except to try/catch",
            premise="except ",
            instruction=(
                "Convert 'except' to 'catch'. Map Python exception types to Java equivalents: "
                "KeyError -> NoSuchElementException, ValueError -> IllegalArgumentException, etc."
            ),
            validation="catch",
            priority=6,
        ),
        FeatureMappingRule(
            id="PY_RAISE",
            name="raise statement",
            source_language=LanguageType.PYTHON,
            target_language=LanguageType.JAVA,
            description="Convert raise to throw",
            premise="raise ",
            instruction="Replace 'raise Exception(...)' with 'throw new Exception(...)'",
            validation="throw new",
            priority=4,
        ),
    ]


def get_default_java_to_python_rules() -> list[FeatureMappingRule]:
    """Get default Java -> Python transformation rules."""
    return [
        FeatureMappingRule(
            id="JAVA_STREAM",
            name="Stream API",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java Streams to Python list comprehensions",
            premise=".stream(",
            instruction=(
                "Convert Java Stream operations to Python list comprehensions or generator expressions. "
                "Example: items.stream().map(x -> x + 1).collect() becomes [x + 1 for x in items]"
            ),
            validation="[",
            priority=7,
        ),
        FeatureMappingRule(
            id="JAVA_TRY_RESOURCES",
            name="Try-with-resources",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert try-with-resources to with statement",
            premise="try (",
            instruction=(
                "Convert Java try-with-resources to Python 'with' statement. "
                "Example: try (FileReader f = ...) becomes with open(...) as f:"
            ),
            validation="with ",
            priority=8,
        ),
        FeatureMappingRule(
            id="JAVA_NULL",
            name="null to None",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java null to Python None",
            premise="null",
            instruction="Replace 'null' with 'None'",
            validation=None,  # 'null' in javadoc/comments triggers premise; can't reliably validate
            priority=1,
        ),
        FeatureMappingRule(
            id="JAVA_BOOL",
            name="Boolean true/false",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java true/false to Python True/False",
            premise="true",
            instruction="Replace 'true' with 'True' and 'false' with 'False' (capitalized)",
            validation=None,  # LLMs handle this correctly; strict check causes false positives on CLASS fragments
            priority=1,
        ),
        FeatureMappingRule(
            id="JAVA_PRINTLN",
            name="System.out.println",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert println to print",
            premise="System.out.println",
            instruction="Replace System.out.println(...) with print(...)",
            validation="print(",
            priority=2,
        ),
        FeatureMappingRule(
            id="JAVA_FOR_LOOP",
            name="Traditional for loop",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java for loops to Python",
            premise="for (",
            instruction=(
                "Convert Java for loops to Python. "
                "for (int i = 0; i < n; i++) becomes for i in range(n)"
            ),
            validation="for ",
            priority=5,
        ),
        FeatureMappingRule(
            id="JAVA_INSTANCEOF",
            name="instanceof operator",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert instanceof to isinstance",
            premise="instanceof",
            instruction="Replace (obj instanceof Type) with isinstance(obj, Type)",
            validation="isinstance(",
            priority=3,
        ),
        FeatureMappingRule(
            id="JAVA_CATCH",
            name="Exception handling",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert try/catch to try/except",
            premise="catch (",
            instruction=(
                "Convert 'catch' to 'except'. Map Java exceptions to Python equivalents: "
                "NullPointerException -> AttributeError, IllegalArgumentException -> ValueError, "
                "IndexOutOfBoundsException -> IndexError, NoSuchElementException -> KeyError"
            ),
            validation="except ",
            priority=6,
        ),
        FeatureMappingRule(
            id="JAVA_THROW",
            name="throw statement",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert throw to raise",
            premise="throw new",
            instruction="Replace 'throw new Exception(...)' with 'raise Exception(...)'",
            validation="raise ",
            priority=4,
        ),
        FeatureMappingRule(
            id="JAVA_ILLEGAL_ARG",
            name="IllegalArgumentException",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert IllegalArgumentException to ValueError",
            premise="IllegalArgumentException",
            instruction=(
                "Replace 'throw new IllegalArgumentException(...)' with 'raise ValueError(...)'. "
                "Java IllegalArgumentException maps to Python ValueError, NOT Exception."
            ),
            validation="ValueError",
            priority=5,
        ),
        FeatureMappingRule(
            id="JAVA_INT_DIVISION",
            name="Integer division",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java integer division to Python floor division",
            premise="/",
            instruction=(
                "When translating Java integer division (int/int or long/long), "
                "use Python's floor division operator '//' instead of '/'. "
                "Java '/' on integers truncates toward zero; Python '//' floors toward negative infinity. "
                "For exact Java behavior, use 'int(a / b)' to truncate toward zero."
            ),
            validation=None,  # Can't validate without knowing the types
            priority=2,
        ),
        FeatureMappingRule(
            id="JAVA_LAMBDA",
            name="Lambda expressions",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java lambdas to Python equivalents",
            premise="->",
            instruction=(
                "Convert Java lambda expressions to Python equivalents. "
                "Prefer list comprehensions or generator expressions for Stream API lambdas "
                "(e.g., .filter(x -> cond) becomes [x for x in ... if cond]). "
                "Use Python lambda only for simple inline callbacks."
            ),
            validation=None,  # Comprehensions are more Pythonic than lambdas; don't reject them
            priority=6,
        ),
        FeatureMappingRule(
            id="JAVA_THIS",
            name="this keyword",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert this to self",
            premise="this.",
            instruction=(
                "Replace 'this.' with 'self.'. Add 'self' as the first parameter in method definitions."
            ),
            validation="self",
            priority=7,
        ),
        FeatureMappingRule(
            id="JAVA_STRING_FORMAT",
            name="String.format",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert String.format to f-strings",
            premise="String.format(",
            instruction=(
                "Convert String.format() to Python f-strings or .format(). "
                'Example: String.format("Hello %s", name) becomes f"Hello {name}"'
            ),
            validation=None,
            priority=3,
        ),
        FeatureMappingRule(
            id="JAVA_ARRAYLIST",
            name="ArrayList usage",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert ArrayList to Python list",
            premise="ArrayList",
            instruction=(
                "Replace ArrayList<T> with Python list. "
                "new ArrayList<>() becomes []. Use list methods: add() -> append(), get() -> [], size() -> len()"
            ),
            validation="[",
            priority=4,
        ),
        FeatureMappingRule(
            id="JAVA_HASHMAP",
            name="HashMap usage",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert HashMap to Python dict",
            premise="HashMap",
            instruction=(
                "Replace HashMap<K,V> with Python dict. "
                "new HashMap<>() becomes {}. Use dict methods: put() -> [], get() -> .get(), containsKey() -> 'in'"
            ),
            validation="{",
            priority=4,
        ),
        FeatureMappingRule(
            id="JAVA_OPTIONAL",
            name="Optional handling",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Optional to direct None checks",
            premise="Optional",
            instruction=(
                "Replace Optional<T> with direct None checks in Python. "
                "Optional.of(x) becomes x. Optional.empty() becomes None. "
                "opt.isPresent() becomes 'x is not None'. opt.orElse(default) becomes 'x if x is not None else default'"
            ),
            validation="None",
            priority=5,
        ),
        FeatureMappingRule(
            id="JAVA_GETTER_SETTER",
            name="Getter/Setter methods",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert getters/setters to properties or direct access",
            premise="get",
            instruction=(
                "Convert Java getters/setters to Python @property decorators or direct attribute access. "
                "For simple data classes, use direct attribute access. "
                "For validation logic, use @property and @attr.setter decorators."
            ),
            validation=None,
            priority=3,
        ),
        FeatureMappingRule(
            id="JAVA_STATIC",
            name="Static methods",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert static methods",
            premise="static ",
            instruction=(
                "Convert Java static methods to Python @staticmethod or @classmethod. "
                "Use @staticmethod for methods that don't access class state. "
                "Use @classmethod for factory methods or methods that need the class. "
                "For private static helpers, module-level functions are also acceptable."
            ),
            validation=None,  # Module-level functions are valid alternatives; don't reject them
            priority=4,
        ),
        FeatureMappingRule(
            id="JAVA_FINAL",
            name="Final variables",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert final to constants",
            premise="final ",
            instruction=(
                "Python doesn't have final, but use UPPER_SNAKE_CASE naming convention for constants. "
                "For class constants, use class attributes. "
                "Consider using typing.Final for type hints: MY_CONST: Final = value"
            ),
            validation=None,
            priority=2,
        ),
        FeatureMappingRule(
            id="JAVA_INTERFACE",
            name="Interface definitions",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert interfaces to ABC",
            premise="interface ",
            instruction=(
                "Convert Java interfaces to Python Abstract Base Classes (ABC). "
                "Use 'from abc import ABC, abstractmethod'. "
                "Interface methods become @abstractmethod decorated methods."
            ),
            validation="ABC",
            priority=6,
        ),
        FeatureMappingRule(
            id="JAVA_IMPLEMENTS",
            name="Implements clause",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert implements to inheritance",
            premise="implements ",
            instruction=(
                "Convert 'implements InterfaceName' to class inheritance. "
                "class MyClass implements Interface becomes class MyClass(Interface)"
            ),
            validation="(",
            priority=5,
        ),
        FeatureMappingRule(
            id="JAVA_EXTENDS",
            name="Extends clause",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert extends to inheritance",
            premise="extends ",
            instruction=(
                "Convert 'extends ClassName' to Python inheritance. "
                "class Child extends Parent becomes class Child(Parent). "
                "Use super().__init__() to call parent constructor. "
                "Note: 'T extends X' in generics is a type bound, not inheritance."
            ),
            validation=None,  # 'extends' in generics (type bounds) doesn't need super()
            priority=5,
        ),
        FeatureMappingRule(
            id="JAVA_SYNCHRONIZED",
            name="Synchronized blocks",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert synchronized to threading.Lock",
            premise="synchronized",
            instruction=(
                "Convert synchronized blocks to Python threading.Lock. "
                "Use 'with lock:' context manager pattern. "
                "For synchronized methods, use a Lock instance as a class attribute."
            ),
            validation="Lock",
            priority=4,
        ),
        FeatureMappingRule(
            id="JAVA_ASSERT",
            name="Assert statement",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java assert to Python assert",
            premise="assert ",
            instruction=(
                "Convert Java assert to Python assert. "
                "Java: assert condition : message becomes Python: assert condition, message"
            ),
            validation="assert ",
            priority=2,
        ),
        FeatureMappingRule(
            id="JAVA_FOREACH",
            name="Enhanced for loop",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert enhanced for loop",
            premise="for (",
            instruction=(
                "Convert Java enhanced for loop to Python for-in loop. "
                "for (Type item : collection) becomes for item in collection:"
            ),
            validation="for ",
            priority=5,
        ),
        FeatureMappingRule(
            id="JAVA_TERNARY",
            name="Ternary operator",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert ternary to Python conditional",
            premise="? ",
            instruction=(
                "Convert Java ternary to Python conditional expression. "
                "condition ? valueIfTrue : valueIfFalse becomes valueIfTrue if condition else valueIfFalse"
            ),
            validation=None,  # LLMs may use if/else statement instead of expression; both valid
            priority=3,
        ),
        FeatureMappingRule(
            id="JAVA_STRINGBUILDER",
            name="StringBuilder usage",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert StringBuilder to list join",
            premise="StringBuilder",
            instruction=(
                "Convert StringBuilder to Python list + join pattern or f-strings. "
                "Use ''.join(parts) for building strings from lists. "
                "For simple cases, use f-strings or string concatenation."
            ),
            validation=None,  # f-strings are valid alternatives to join; don't reject them
            priority=4,
        ),
        FeatureMappingRule(
            id="JAVA_ANNOTATIONS",
            name="Annotations",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert annotations to decorators",
            premise="@",
            instruction=(
                "Convert Java annotations to Python decorators where applicable. "
                "@Override can be removed (Python uses duck typing). "
                "@Deprecated becomes @deprecated from warnings module."
            ),
            validation=None,  # @Override→nothing is valid; can't demand @ in output
            priority=3,
        ),
        FeatureMappingRule(
            id="JAVA_RECORD",
            name="Record classes",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert records to dataclasses",
            premise="record ",
            instruction=(
                "Convert Java records to Python dataclasses. "
                "record Point(int x, int y) becomes @dataclass class Point: x: int; y: int"
            ),
            validation="@dataclass",
            priority=6,
        ),
        FeatureMappingRule(
            id="JAVA_ENUM",
            name="Enum definitions",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java enum to Python Enum",
            premise="enum ",
            instruction=(
                "Convert Java enum to Python Enum class. "
                "Use 'from enum import Enum'. "
                "enum Color { RED, GREEN } becomes class Color(Enum): RED = auto(); GREEN = auto()"
            ),
            validation="Enum",
            priority=5,
        ),
        # =====================================================================
        # Generalized Java→Python overload / builtin / iterator rules
        # =====================================================================
        FeatureMappingRule(
            id="JAVA_OVERLOADED_METHOD",
            name="Overloaded method dispatch",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Handle Java method overloading in Python",
            premise="(",  # Matches any method with parameters
            instruction=(
                "Python lacks method overloading. If this Java method has multiple "
                "overloads with different parameter types/counts, combine them into "
                "a single Python method using `*args` + `isinstance` dispatch or "
                "optional parameters with defaults. NEVER create self-recursive "
                "delegation between methods of the same name."
            ),
            validation=None,
            priority=1,  # Low priority — only fires when overloads detected
        ),
        FeatureMappingRule(
            id="JAVA_NO_BUILTIN_SHADOW",
            name="No Python builtin shadowing",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Prevent shadowing Python builtin names as variables",
            premise=".iterator()",
            instruction=(
                "Do NOT use Python builtin names as variable names. "
                "Common Java→Python trap: `iter = iter(x)` shadows the builtin. "
                "Use `_iter`, `it`, or `iterator` instead. Also avoid: list, dict, "
                "type, id, range, map, filter, set, hash, next, tuple, str, int, "
                "float, bool, open, format, input, object, super."
            ),
            validation=None,
            priority=8,
        ),
        FeatureMappingRule(
            id="JAVA_ITERATOR_PATTERN",
            name="Java Iterator pattern",
            source_language=LanguageType.JAVA,
            target_language=LanguageType.PYTHON,
            description="Convert Java hasNext/next to Pythonic iteration",
            premise="hasNext",
            instruction=(
                "Java `hasNext()`/`next()` → Python `for x in iterable:` or "
                "`try: x = next(it) except StopIteration`. "
                "Do NOT use `next(it, None) is None` as an exhaustion check — "
                "`None` can be a valid element. Use a sentinel: "
                "`_SENTINEL = object(); val = next(it, _SENTINEL); if val is _SENTINEL:`"
            ),
            validation=None,
            priority=7,
        ),
    ]


def create_default_mapper() -> FeatureMapper:
    """Create a FeatureMapper with default rules loaded."""
    mapper = FeatureMapper()

    for rule in get_default_python_to_java_rules():
        mapper.add_rule(rule)

    for rule in get_default_java_to_python_rules():
        mapper.add_rule(rule)

    return mapper
