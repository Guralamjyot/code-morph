"""
Unit tests for feature mapping system.
"""

from pathlib import Path

import pytest

from codemorph.config.models import CodeFragment, FragmentType, LanguageType
from codemorph.knowledge.feature_mapper import (
    FeatureMappingRule,
    FeatureMapper,
    create_default_mapper,
    get_default_python_to_java_rules,
)


@pytest.fixture
def sample_python_fragment():
    """Create a sample Python fragment with list comprehension."""
    return CodeFragment(
        id="test::process_items",
        name="process_items",
        fragment_type=FragmentType.FUNCTION,
        source_file=Path("test.py"),
        start_line=1,
        end_line=3,
        source_code="""def process_items(items):
    return [x * 2 for x in items]
""",
        dependencies=[],
    )


@pytest.fixture
def sample_python_context_manager():
    """Create a sample Python fragment with context manager."""
    return CodeFragment(
        id="test::read_file",
        name="read_file",
        fragment_type=FragmentType.FUNCTION,
        source_file=Path("test.py"),
        start_line=1,
        end_line=4,
        source_code="""def read_file(path):
    with open(path) as f:
        return f.read()
""",
        dependencies=[],
    )


def test_create_feature_mapping_rule():
    """Test creating a feature mapping rule."""
    rule = FeatureMappingRule(
        id="TEST_RULE",
        name="Test Rule",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="A test rule",
        premise="test",
        instruction="Do something",
        validation="result",
        priority=5,
    )

    assert rule.id == "TEST_RULE"
    assert rule.priority == 5


def test_rule_check_premise(sample_python_fragment):
    """Test checking if a rule applies to a fragment."""
    rule = FeatureMappingRule(
        id="LIST_COMP",
        name="List Comprehension",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Convert list comprehensions",
        premise="[",
        instruction="Use Stream API",
    )

    # Should match because fragment contains "["
    assert rule.check_premise(sample_python_fragment)

    # Create fragment without list comprehension
    simple_fragment = CodeFragment(
        id="test::simple",
        name="simple",
        fragment_type=FragmentType.FUNCTION,
        source_file=Path("test.py"),
        start_line=1,
        end_line=2,
        source_code="def simple():\n    pass",
        dependencies=[],
    )

    # Should not match
    assert not rule.check_premise(simple_fragment)


def test_rule_validate_translation():
    """Test validating a translation against a rule."""
    rule = FeatureMappingRule(
        id="LIST_COMP",
        name="List Comprehension",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Convert list comprehensions",
        premise="[",
        instruction="Use Stream API",
        validation="stream(",
    )

    # Valid translation
    good_translation = "return items.stream().map(x -> x * 2).collect(Collectors.toList());"
    assert rule.validate_translation(good_translation)

    # Invalid translation
    bad_translation = "return items.map(x -> x * 2).toList();"
    assert not rule.validate_translation(bad_translation)


def test_feature_mapper_add_rule():
    """Test adding rules to the mapper."""
    mapper = FeatureMapper()

    rule = FeatureMappingRule(
        id="TEST",
        name="Test",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Test",
        premise="test",
        instruction="Do something",
    )

    mapper.add_rule(rule)

    key = (LanguageType.PYTHON, LanguageType.JAVA)
    assert key in mapper.rules
    assert len(mapper.rules[key]) == 1


def test_feature_mapper_get_applicable_rules(sample_python_fragment):
    """Test getting applicable rules for a fragment."""
    mapper = FeatureMapper()

    # Add two rules, one that matches and one that doesn't
    rule1 = FeatureMappingRule(
        id="MATCH",
        name="Match",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Matches",
        premise="[",  # Will match list comprehension
        instruction="Use Stream API",
        priority=5,
    )

    rule2 = FeatureMappingRule(
        id="NO_MATCH",
        name="No Match",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Doesn't match",
        premise="async",  # Won't match
        instruction="Use CompletableFuture",
        priority=3,
    )

    mapper.add_rule(rule1)
    mapper.add_rule(rule2)

    # Get applicable rules
    applicable = mapper.get_applicable_rules(
        sample_python_fragment,
        LanguageType.PYTHON,
        LanguageType.JAVA,
    )

    assert len(applicable) == 1
    assert applicable[0].id == "MATCH"


def test_feature_mapper_get_instructions(sample_python_context_manager):
    """Test getting translation instructions."""
    mapper = FeatureMapper()

    rule = FeatureMappingRule(
        id="WITH_STMT",
        name="With Statement",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Context manager",
        premise="with ",
        instruction="Use try-with-resources",
    )

    mapper.add_rule(rule)

    instructions = mapper.get_instructions_for_fragment(
        sample_python_context_manager,
        LanguageType.PYTHON,
        LanguageType.JAVA,
    )

    assert len(instructions) == 1
    assert "try-with-resources" in instructions[0]


def test_feature_mapper_validate_translation(sample_python_fragment):
    """Test validating a translation."""
    mapper = FeatureMapper()

    rule = FeatureMappingRule(
        id="LIST_COMP",
        name="List Comprehension",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Convert list comp",
        premise="[",
        instruction="Use Stream API",
        validation="stream(",
    )

    mapper.add_rule(rule)

    # Valid translation
    valid_code = "return items.stream().map(x -> x * 2).collect(Collectors.toList());"
    is_valid, failed = mapper.validate_translation(
        sample_python_fragment,
        valid_code,
        LanguageType.PYTHON,
        LanguageType.JAVA,
    )

    assert is_valid
    assert len(failed) == 0

    # Invalid translation
    invalid_code = "return items.map(x -> x * 2).toList();"
    is_valid, failed = mapper.validate_translation(
        sample_python_fragment,
        invalid_code,
        LanguageType.PYTHON,
        LanguageType.JAVA,
    )

    assert not is_valid
    assert "LIST_COMP" in failed


def test_default_mapper_creation():
    """Test creating default mapper with built-in rules."""
    mapper = create_default_mapper()

    # Should have both Python->Java and Java->Python rules
    py_to_java_key = (LanguageType.PYTHON, LanguageType.JAVA)
    java_to_py_key = (LanguageType.JAVA, LanguageType.PYTHON)

    assert py_to_java_key in mapper.rules
    assert java_to_py_key in mapper.rules

    # Check some rules exist
    py_to_java_rules = mapper.rules[py_to_java_key]
    rule_ids = [r.id for r in py_to_java_rules]

    assert "PY_LIST_COMP" in rule_ids
    assert "PY_CONTEXT_MGR" in rule_ids
    assert "PY_NONE" in rule_ids


def test_get_default_python_to_java_rules():
    """Test getting default Python to Java rules."""
    rules = get_default_python_to_java_rules()

    assert len(rules) > 0

    # Check for essential rules
    rule_ids = [r.id for r in rules]

    assert "PY_LIST_COMP" in rule_ids
    assert "PY_CONTEXT_MGR" in rule_ids
    assert "PY_NONE" in rule_ids
    assert "PY_BOOL" in rule_ids

    # Check that rules are properly configured
    none_rule = next(r for r in rules if r.id == "PY_NONE")
    assert none_rule.source_language == LanguageType.PYTHON
    assert none_rule.target_language == LanguageType.JAVA
    assert "None" in none_rule.premise
    assert "null" in none_rule.instruction


def test_rule_priority_ordering():
    """Test that rules are ordered by priority."""
    mapper = FeatureMapper()

    # Add rules with different priorities
    rule1 = FeatureMappingRule(
        id="LOW",
        name="Low Priority",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="Low",
        premise="test",
        instruction="Low priority",
        priority=1,
    )

    rule2 = FeatureMappingRule(
        id="HIGH",
        name="High Priority",
        source_language=LanguageType.PYTHON,
        target_language=LanguageType.JAVA,
        description="High",
        premise="test",
        instruction="High priority",
        priority=10,
    )

    mapper.add_rule(rule1)
    mapper.add_rule(rule2)

    key = (LanguageType.PYTHON, LanguageType.JAVA)
    rules = mapper.rules[key]

    # Higher priority should come first
    assert rules[0].id == "HIGH"
    assert rules[1].id == "LOW"
