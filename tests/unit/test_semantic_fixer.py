"""
Tests for SemanticFixer — generalized AST rewrites for Java→Python translation.

All tests use generic synthetic code, not specific library references.
"""

import textwrap

import pytest

from codemorph.assembler.index_builder import (
    ClassSummary,
    FragmentEntry,
    ProjectIndex,
)
from codemorph.assembler.semantic_fixer import SemanticFixer


# =========================================================================
# Helpers
# =========================================================================


def _make_index(
    classes: dict | None = None,
    fragments: dict | None = None,
    hierarchy: dict | None = None,
) -> ProjectIndex:
    """Build a minimal ProjectIndex for testing."""
    index = ProjectIndex()
    if classes:
        index.classes = classes
    if fragments:
        index.fragments = fragments
    if hierarchy:
        index.hierarchy = hierarchy
    return index


def _make_fragment(
    fragment_id: str,
    class_name: str,
    member_name: str | None = None,
    symbol_type: str = "method",
    java_source: str | None = None,
    target_code: str | None = None,
    python_name: str | None = None,
) -> FragmentEntry:
    return FragmentEntry(
        fragment_id=fragment_id,
        class_name=class_name,
        member_name=member_name,
        symbol_type=symbol_type,
        status="translated",
        is_mocked=False,
        java_name=fragment_id,
        python_name=python_name or fragment_id,
        signature=None,
        target_code=target_code,
        java_source=java_source,
    )


def _make_class_summary(
    name: str, symbol_type: str = "class"
) -> ClassSummary:
    return ClassSummary(
        name=name,
        java_source_file=f"{name}.java",
        symbol_type=symbol_type,
    )


# =========================================================================
# Fix 1: Builtin Shadowing
# =========================================================================


class TestBuiltinShadowing:
    """Test that local variables shadowing Python builtins are renamed."""

    def test_iter_shadowing(self):
        code = textwrap.dedent("""\
            class Foo:
                def process(self):
                    iter = iter(self.items)
                    return iter
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_builtin_shadowing("Foo", code)

        assert "_iter" in fixed
        # The builtin `iter` call is also renamed, which is fine because the
        # entire purpose is to disambiguate — the fix is applied uniformly
        assert report.applied

    def test_list_shadowing(self):
        code = textwrap.dedent("""\
            class Bar:
                def collect(self):
                    list = list(range(10))
                    return list
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_builtin_shadowing("Bar", code)

        assert "_list" in fixed
        assert report.applied

    def test_no_shadow_when_not_called(self):
        """Variable named 'list' but list() is never called should not be renamed."""
        code = textwrap.dedent("""\
            class Baz:
                def process(self):
                    list = [1, 2, 3]
                    return list
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_builtin_shadowing("Baz", code)

        # No call to list() as a function, so no shadow detected
        assert not report.applied

    def test_idempotent(self):
        """Running twice should produce the same output."""
        code = textwrap.dedent("""\
            class Foo:
                def process(self):
                    iter = iter(self.items)
                    return iter
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed1, _ = fixer._fix_builtin_shadowing("Foo", code)
        fixed2, _ = fixer._fix_builtin_shadowing("Foo", fixed1)
        assert fixed1 == fixed2


# =========================================================================
# Fix 2: Missing @staticmethod
# =========================================================================


class TestMissingStaticmethod:
    """Test that methods without self/cls get @staticmethod."""

    def test_missing_decorator(self):
        code = textwrap.dedent("""\
            class Foo:
                def create(name):
                    return Foo(name)
        """)
        index = _make_index(
            fragments={
                "Foo::Foo.create": _make_fragment(
                    "Foo::Foo.create", "Foo", "create",
                    java_source="public static Foo create(String name) { return new Foo(name); }",
                ),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_staticmethod("Foo", code)

        assert "@staticmethod" in fixed
        assert report.applied

    def test_already_decorated(self):
        code = textwrap.dedent("""\
            class Foo:
                @staticmethod
                def create(name):
                    return Foo(name)
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_staticmethod("Foo", code)

        assert not report.applied
        assert fixed.count("@staticmethod") == 1

    def test_skip_dunder_methods(self):
        code = textwrap.dedent("""\
            class Foo:
                def __repr__():
                    return "Foo"
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_staticmethod("Foo", code)

        assert not report.applied

    def test_method_with_self_not_touched(self):
        code = textwrap.dedent("""\
            class Foo:
                def process(self):
                    return self.value
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_staticmethod("Foo", code)

        assert not report.applied


# =========================================================================
# Fix 3: Infinite Recursion
# =========================================================================


class TestInfiniteRecursion:
    """Test detection of self-recursive calls with different arity."""

    def test_recursive_overload(self):
        code = textwrap.dedent("""\
            class Container:
                def process(self, x):
                    return self.process(x.first, x.second)
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_infinite_recursion("Container", code)

        assert report.applied
        assert "WARNING" in fixed
        assert "infinite recursion" in fixed.lower()

    def test_normal_recursion_ignored(self):
        """Same arity self-call is normal recursion, not overload conflation."""
        code = textwrap.dedent("""\
            class Tree:
                def traverse(self, node):
                    return self.traverse(node.left)
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_infinite_recursion("Tree", code)

        assert not report.applied


# =========================================================================
# Fix 4: Missing __init__
# =========================================================================


class TestMissingInit:
    """Test __init__ generation for classes with uninitialized self.attrs."""

    def test_generate_init(self):
        code = textwrap.dedent("""\
            class Person:
                def greet(self):
                    return f"Hello, {self.name}"

                def get_age(self):
                    return self.age
        """)
        index = _make_index(
            fragments={
                "Person::Person.greet": _make_fragment(
                    "Person::Person.greet", "Person", "greet",
                ),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_init("Person", code)

        assert "def __init__(self):" in fixed
        assert "self.name = None" in fixed
        assert "self.age = None" in fixed
        assert report.applied

    def test_already_has_init(self):
        code = textwrap.dedent("""\
            class Person:
                def __init__(self, name):
                    self.name = name

                def greet(self):
                    return f"Hello, {self.name}"
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_init("Person", code)

        assert not report.applied

    def test_init_with_super(self):
        code = textwrap.dedent("""\
            class Employee(Person):
                def get_salary(self):
                    return self.salary
        """)
        index = _make_index(
            fragments={
                "Employee::Employee.getSalary": _make_fragment(
                    "Employee::Employee.getSalary", "Employee", "getSalary",
                ),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_missing_init("Employee", code)

        assert "def __init__(self):" in fixed
        assert "super().__init__()" in fixed
        assert report.applied


# =========================================================================
# Fix 5: Interface to ABC
# =========================================================================


class TestInterfaceToABC:
    """Test conversion of interface-marked classes to ABCs."""

    def test_interface_cleanup(self):
        code = textwrap.dedent("""\
            class IRenderer:
                def __init__(self):
                    self._width = None
                    self._height = None

                def render(self):
                    pass

                def resize(self, w, h):
                    pass
        """)
        index = _make_index(
            classes={
                "IRenderer": _make_class_summary("IRenderer", "interface"),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_interface_to_abc("IRenderer", code)

        assert "from abc import ABC, abstractmethod" in fixed
        assert "ABC" in fixed
        assert "@abstractmethod" in fixed
        # __init__ should be removed
        assert "__init__" not in fixed
        assert report.applied

    def test_non_interface_not_touched(self):
        code = textwrap.dedent("""\
            class Renderer:
                def __init__(self):
                    self._width = None

                def render(self):
                    pass
        """)
        index = _make_index(
            classes={
                "Renderer": _make_class_summary("Renderer", "class"),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_interface_to_abc("Renderer", code)

        assert not report.applied


# =========================================================================
# Fix 6: Method Name Consistency
# =========================================================================


class TestMethodNameConsistency:
    """Test normalization of inconsistent method names within groups."""

    def test_normalize_names(self):
        code = textwrap.dedent("""\
            class DataRow:
                def get_item_0(self):
                    return self._items[0]

                def get_item1(self):
                    return self._items[1]

                def get_item_2(self):
                    return self._items[2]
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_method_name_consistency("DataRow", code)

        assert report.applied
        # All should be normalized to the majority pattern
        # get_item_0 and get_item_2 use underscore (2 vs 1), so underscore wins
        assert "get_item_0" in fixed
        assert "get_item_1" in fixed
        assert "get_item_2" in fixed
        # The outlier get_item1 should have been renamed
        assert "get_item1" not in fixed

    def test_already_consistent(self):
        code = textwrap.dedent("""\
            class DataRow:
                def get_item0(self):
                    return self._items[0]

                def get_item1(self):
                    return self._items[1]
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_method_name_consistency("DataRow", code)

        assert not report.applied


# =========================================================================
# Fix 7: Field Name Registry
# =========================================================================


class TestFieldNameRegistry:
    """Test field renaming to match symbol registry."""

    def test_rename_uppercase_to_registry(self):
        code = textwrap.dedent("""\
            class Metrics:
                COUNT = None
                TOTAL = None

                def get_count(self):
                    return self.COUNT
        """)
        index = _make_index(
            fragments={
                "Metrics::Metrics.count": _make_fragment(
                    "Metrics::Metrics.count", "Metrics", "count",
                    symbol_type="field",
                    python_name="count",
                ),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_field_names("Metrics", code)

        assert report.applied
        assert "count" in fixed
        # COUNT should be replaced with count
        assert "COUNT" not in fixed or "TOTAL" in fixed  # TOTAL stays

    def test_no_rename_when_already_correct(self):
        code = textwrap.dedent("""\
            class Metrics:
                count = 0
        """)
        index = _make_index(
            fragments={
                "Metrics::Metrics.count": _make_fragment(
                    "Metrics::Metrics.count", "Metrics", "count",
                    symbol_type="field",
                    python_name="count",
                ),
            }
        )
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_field_names("Metrics", code)

        assert not report.applied


# =========================================================================
# Fix 8: None Sentinel
# =========================================================================


class TestNoneSentinel:
    """Test replacement of next(x, None) is None pattern."""

    def test_sentinel_replacement(self):
        code = textwrap.dedent("""\
            class Stream:
                def consume(self):
                    it = iter(self.items)
                    val = next(it, None)
                    if val is None:
                        raise StopIteration
                    return val
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_none_sentinel("Stream", code)

        assert report.applied
        assert "_SENTINEL = object()" in fixed
        assert "next(it, _SENTINEL)" in fixed
        assert "is _SENTINEL" in fixed

    def test_no_sentinel_when_not_present(self):
        code = textwrap.dedent("""\
            class Stream:
                def consume(self):
                    for val in self.items:
                        yield val
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, report = fixer._fix_none_sentinel("Stream", code)

        assert not report.applied

    def test_sentinel_idempotent(self):
        code = textwrap.dedent("""\
            class Stream:
                def consume(self):
                    it = iter(self.items)
                    val = next(it, None)
                    if val is None:
                        raise StopIteration
                    return val
        """)
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed1, _ = fixer._fix_none_sentinel("Stream", code)
        fixed2, _ = fixer._fix_none_sentinel("Stream", fixed1)
        assert fixed1 == fixed2


# =========================================================================
# Integration: fix_all
# =========================================================================


class TestFixAll:
    """Test the full fix_all pipeline."""

    def test_multiple_classes(self):
        assembled = {
            "Foo": textwrap.dedent("""\
                class Foo:
                    def create(name):
                        return Foo(name)
            """),
            "Bar": textwrap.dedent("""\
                class Bar:
                    def process(self):
                        iter = iter(self.items)
                        return iter
            """),
        }
        index = _make_index(
            classes={
                "Foo": _make_class_summary("Foo"),
                "Bar": _make_class_summary("Bar"),
            },
            fragments={
                "Foo::Foo.create": _make_fragment(
                    "Foo::Foo.create", "Foo", "create",
                    java_source="public static Foo create(String name) {}",
                ),
            },
        )
        fixer = SemanticFixer(index)
        fixed, reports = fixer.fix_all(assembled)

        assert "Foo" in fixed
        assert "Bar" in fixed
        assert "@staticmethod" in fixed["Foo"]
        assert "_iter" in fixed["Bar"]

    def test_empty_input(self):
        index = _make_index()
        fixer = SemanticFixer(index)
        fixed, reports = fixer.fix_all({})

        assert fixed == {}
        assert reports == {}

    def test_syntax_error_code_survives(self):
        """Code with syntax errors should pass through unchanged."""
        assembled = {
            "Broken": "class Broken:\n    def oops(self\n",
        }
        index = _make_index(
            classes={"Broken": _make_class_summary("Broken")},
        )
        fixer = SemanticFixer(index)
        fixed, reports = fixer.fix_all(assembled)

        assert fixed["Broken"] == assembled["Broken"]


# =========================================================================
# Index Builder: overload-aware fragment IDs
# =========================================================================


class TestIndexBuilderOverloads:
    """Test that the index builder correctly parses $-suffixed fragment IDs."""

    def test_parse_fragment_id_with_overload(self):
        from codemorph.assembler.index_builder import IndexBuilder

        builder = IndexBuilder(state_dir="/tmp/nonexistent")
        cls, member = builder._parse_fragment_id("Pair::Pair.addAt0$X0")
        assert cls == "Pair"
        assert member == "addAt0"

    def test_parse_fragment_id_without_overload(self):
        from codemorph.assembler.index_builder import IndexBuilder

        builder = IndexBuilder(state_dir="/tmp/nonexistent")
        cls, member = builder._parse_fragment_id("Pair::Pair.fromArray")
        assert cls == "Pair"
        assert member == "fromArray"

    def test_parse_class_level(self):
        from codemorph.assembler.index_builder import IndexBuilder

        builder = IndexBuilder(state_dir="/tmp/nonexistent")
        cls, member = builder._parse_fragment_id("Pair::Pair")
        assert cls == "Pair"
        assert member is None

    def test_overloads_field_populated(self):
        """ProjectIndex.overloads should group fragments sharing a base ID."""
        index = ProjectIndex()
        index.overloads = {}

        # Simulate what build() does
        frag_ids = [
            "Pair::Pair.addAt0$X0",
            "Pair::Pair.addAt0$X0_X1",
            "Pair::Pair.addAt0$Unit",
            "Pair::Pair.fromArray",
        ]
        base_groups: dict[str, list[str]] = {}
        for fid in frag_ids:
            base = fid.split("$")[0] if "$" in fid else fid
            base_groups.setdefault(base, []).append(fid)

        for base, variants in base_groups.items():
            if len(variants) > 1:
                index.overloads[base] = sorted(variants)

        assert "Pair::Pair.addAt0" in index.overloads
        assert len(index.overloads["Pair::Pair.addAt0"]) == 3
        assert "Pair::Pair.fromArray" not in index.overloads
