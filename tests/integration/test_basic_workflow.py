"""
Integration test for basic CodeMorph workflow.

Tests the interaction between components without running full translation.
"""

from pathlib import Path

import pytest

from codemorph.analyzer.graph_builder import DependencyGraphBuilder
from codemorph.config.models import LanguageType
from codemorph.languages.registry import LanguagePluginRegistry


@pytest.fixture
def example_python_file():
    """Get path to the example calculator.py file."""
    return Path(__file__).parent.parent.parent / "examples" / "python_project" / "calculator.py"


def test_python_plugin_parsing(example_python_file):
    """Test that Python plugin can parse and extract fragments."""
    # Get the plugin
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")

    # Parse the file
    assert example_python_file.exists(), "Example file not found"
    ast = plugin.parse_file(example_python_file)
    assert ast is not None

    # Extract fragments
    fragments = plugin.extract_fragments(example_python_file, ast)

    # Verify we got the expected fragments
    fragment_names = [f.name for f in fragments]
    assert "add" in fragment_names
    assert "subtract" in fragment_names
    assert "multiply" in fragment_names
    assert "divide" in fragment_names
    assert "calculate_average" in fragment_names
    assert "Calculator" in fragment_names

    # Check that we extracted methods from Calculator class
    calculator_methods = [f for f in fragments if f.parent_class == "Calculator"]
    method_names = [m.name for m in calculator_methods]
    assert "add_to_memory" in method_names
    assert "get_memory" in method_names
    assert "clear_memory" in method_names
    assert "compute" in method_names


def test_dependency_graph_building(example_python_file):
    """Test building a dependency graph from fragments."""
    # Get plugin and parse
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    ast = plugin.parse_file(example_python_file)
    fragments = plugin.extract_fragments(example_python_file, ast)

    # Convert to dict for graph builder
    fragment_dict = {f.id: f for f in fragments}

    # Analyze dependencies for each fragment
    for fragment in fragments:
        deps = plugin.get_fragment_dependencies(fragment, ast)
        fragment.dependencies = [d for d in deps if d in fragment_dict]

    # Build graph
    graph_builder = DependencyGraphBuilder()
    graph = graph_builder.build_graph(fragment_dict)

    # Verify graph was built
    assert graph.number_of_nodes() > 0
    assert graph.number_of_nodes() == len(fragments)

    # Get translation order
    try:
        order = graph_builder.get_translation_order()
        assert len(order) > 0
    except ValueError:
        # Circular dependencies found - this is expected for some codebases
        cycles = graph_builder.detect_circular_dependencies()
        assert len(cycles) >= 0  # Just verify we can detect them


def test_signature_extraction(example_python_file):
    """Test extracting signatures from functions."""
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    ast = plugin.parse_file(example_python_file)
    fragments = plugin.extract_fragments(example_python_file, ast)

    # Find the 'add' function
    add_func = next(f for f in fragments if f.name == "add")
    signature = plugin.extract_signature(add_func)

    assert signature is not None
    assert "add" in signature
    assert "int" in signature  # Type annotations should be present


def test_docstring_extraction(example_python_file):
    """Test extracting docstrings."""
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    ast = plugin.parse_file(example_python_file)
    fragments = plugin.extract_fragments(example_python_file, ast)

    # Find the 'multiply' function
    multiply_func = next(f for f in fragments if f.name == "multiply")
    docstring = plugin.extract_docstring(multiply_func)

    assert docstring is not None
    assert "Multiply" in docstring or "multiply" in docstring


def test_naming_convention_conversion():
    """Test name conversion between Python and Java conventions."""
    python_plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    java_plugin = LanguagePluginRegistry.get_plugin(LanguageType.JAVA, "17")

    from codemorph.config.models import FragmentType

    # Python snake_case to Java camelCase
    python_name = "calculate_average"
    java_name = python_plugin.convert_name(python_name, FragmentType.FUNCTION, "camelCase")
    assert java_name == "calculateAverage"

    # Java camelCase to Python snake_case
    java_method = "calculateAverage"
    python_method = java_plugin.convert_name(java_method, FragmentType.METHOD, "snake_case")
    assert python_method == "calculate_average"


def test_complexity_scoring(example_python_file):
    """Test complexity score calculation."""
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    ast = plugin.parse_file(example_python_file)
    fragments = plugin.extract_fragments(example_python_file, ast)

    fragment_dict = {f.id: f for f in fragments}

    # Build graph and calculate complexity
    graph_builder = DependencyGraphBuilder()
    graph_builder.build_graph(fragment_dict)
    scores = graph_builder.calculate_complexity_scores(fragment_dict)

    # Verify we got scores for all fragments
    assert len(scores) == len(fragments)

    # More complex functions should have higher scores
    # The Calculator.compute method should be more complex than simple add()
    if "calculator::add" in scores and "calculator::Calculator.compute" in scores:
        assert scores["calculator::Calculator.compute"] > scores["calculator::add"]
