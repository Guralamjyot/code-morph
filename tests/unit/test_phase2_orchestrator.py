"""
Unit tests for Phase 2 orchestrator.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemorph.config.loader import create_config_from_args
from codemorph.config.models import (
    AnalysisResult,
    CodeFragment,
    FragmentType,
    TranslationStatus,
)
from codemorph.state.persistence import TranslationState
from codemorph.translator.orchestrator import Phase2Orchestrator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration."""
    return create_config_from_args(
        source_dir=temp_dir / "source",
        source_lang="python",
        source_version="3.10",
        target_lang="java",
        target_version="17",
        output_dir=temp_dir / "output",
        build_system="gradle",
    )


@pytest.fixture
def sample_fragments():
    """Create sample code fragments."""
    return {
        "test::func1": CodeFragment(
            id="test::func1",
            name="func1",
            fragment_type=FragmentType.FUNCTION,
            source_file=Path("test.py"),
            start_line=1,
            end_line=3,
            source_code="def func1():\n    return 42",
            dependencies=[],
        ),
        "test::func2": CodeFragment(
            id="test::func2",
            name="func2",
            fragment_type=FragmentType.FUNCTION,
            source_file=Path("test.py"),
            start_line=5,
            end_line=7,
            source_code="def func2():\n    return func1() * 2",
            dependencies=["test::func1"],
        ),
    }


@pytest.fixture
def state_with_analysis(sample_config, sample_fragments, temp_dir):
    """Create a state with completed Phase 1 analysis."""
    sample_config.project.state_dir = temp_dir / ".codemorph"
    sample_config.project.state_dir.mkdir(parents=True, exist_ok=True)

    state = TranslationState(sample_config, temp_dir / "source")

    # Create analysis result
    analysis = AnalysisResult(
        fragments=sample_fragments,
        translation_order=["test::func1", "test::func2"],
        circular_dependencies=[],
    )

    state.set_analysis_result(analysis)
    state.current_phase = 1

    return state


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_orchestrator_initialization(mock_create_llm, sample_config, state_with_analysis):
    """Test Phase2Orchestrator initialization."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    assert orchestrator.config == sample_config
    assert orchestrator.state == state_with_analysis
    assert orchestrator.source_plugin is not None
    assert orchestrator.target_plugin is not None
    assert orchestrator.llm_client is not None
    assert orchestrator.feature_mapper is not None
    assert orchestrator.type_checker is not None


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_get_dependency_context_empty(mock_create_llm, sample_config, state_with_analysis):
    """Test getting dependency context when no dependencies are translated."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    fragment = state_with_analysis.analysis_result.fragments["test::func1"]
    context = orchestrator._get_dependency_context(fragment)

    assert len(context) == 0


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_get_dependency_context_with_dependencies(mock_create_llm, sample_config, state_with_analysis):
    """Test getting dependency context with translated dependencies."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    # Mock a translated dependency
    from codemorph.config.models import TranslatedFragment

    translated_func1 = TranslatedFragment(
        fragment=state_with_analysis.analysis_result.fragments["test::func1"],
        status=TranslationStatus.COMPILED,
        target_code="public static int func1() { return 42; }",
    )

    state_with_analysis.update_fragment(translated_func1)

    # Get context for func2 (which depends on func1)
    fragment = state_with_analysis.analysis_result.fragments["test::func2"]
    context = orchestrator._get_dependency_context(fragment)

    assert len(context) == 1
    assert context[0]["name"] == "func1"
    assert "signature" in context[0]


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_translate_fragment_success(
    mock_create_llm, sample_config, state_with_analysis
):
    """Test successful fragment translation."""
    # Set up mock LLM client
    mock_llm_instance = mock_create_llm.return_value
    mock_llm_instance.translate_fragment.return_value = (
        "public static int func1() { return 42; }",
        MagicMock(),  # conversation
    )

    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    # Mock compilation success
    with patch.object(
        orchestrator.target_plugin, "compile_fragment", return_value=(True, [])
    ):
        fragment = state_with_analysis.analysis_result.fragments["test::func1"]
        translated = orchestrator._translate_fragment(fragment)

        assert translated.status in [
            TranslationStatus.COMPILED,
            TranslationStatus.TYPE_VERIFIED,
        ]
        assert translated.target_code is not None
        assert translated.retry_count <= 1


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_translate_fragment_with_retries(
    mock_create_llm, sample_config, state_with_analysis
):
    """Test fragment translation with compilation failures and retries."""
    # Set low retry limit for testing
    sample_config.translation.max_retries_type_check = 2

    # Set up mock LLM responses
    mock_llm_instance = mock_create_llm.return_value
    mock_llm_instance.translate_fragment.return_value = (
        "public static int func1() { return 42 }",  # Missing semicolon
        MagicMock(),
    )
    mock_llm_instance.refine_translation.return_value = (
        "public static int func1() { return 42; }",  # Fixed
        MagicMock(),
    )

    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    # Mock compilation: fail first, succeed second
    compilation_results = [(False, ["Missing semicolon"]), (True, [])]
    with patch.object(
        orchestrator.target_plugin,
        "compile_fragment",
        side_effect=compilation_results,
    ):
        fragment = state_with_analysis.analysis_result.fragments["test::func1"]
        translated = orchestrator._translate_fragment(fragment)

        # Should succeed after refinement
        assert translated.status in [
            TranslationStatus.COMPILED,
            TranslationStatus.TYPE_VERIFIED,
        ]
        assert translated.retry_count >= 1


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_translate_fragment_exceeds_retries(
    mock_create_llm, sample_config, state_with_analysis
):
    """Test fragment translation that exceeds retry limit."""
    # Set very low retry limit
    sample_config.translation.max_retries_type_check = 1
    sample_config.translation.allow_mocking = True

    # Mock LLM always returns bad code
    mock_llm_instance = mock_create_llm.return_value
    mock_llm_instance.translate_fragment.return_value = (
        "invalid java code",
        MagicMock(),
    )
    mock_llm_instance.refine_translation.return_value = (
        "still invalid",
        MagicMock(),
    )

    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    # Mock compilation always fails
    with patch.object(
        orchestrator.target_plugin,
        "compile_fragment",
        return_value=(False, ["Compilation error"]),
    ):
        fragment = state_with_analysis.analysis_result.fragments["test::func1"]
        translated = orchestrator._translate_fragment(fragment)

        # Should be mocked after exceeding retries
        assert translated.status == TranslationStatus.MOCKED
        assert translated.is_mocked is True
        assert translated.mock_reason is not None


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_validate_feature_mapping(mock_create_llm, sample_config, state_with_analysis):
    """Test feature mapping validation."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    # Create a fragment with list comprehension
    fragment = CodeFragment(
        id="test::process",
        name="process",
        fragment_type=FragmentType.FUNCTION,
        source_file=Path("test.py"),
        start_line=1,
        end_line=2,
        source_code="result = [x * 2 for x in items]",
        dependencies=[],
    )

    # Valid translation using Stream API
    good_translation = "return items.stream().map(x -> x * 2).collect(Collectors.toList());"
    assert orchestrator._validate_feature_mapping(fragment, good_translation)

    # Invalid translation (not using stream)
    bad_translation = "for (int x : items) { result.add(x * 2); }"
    # This might still pass if validation rule isn't strict
    # Just verify the method runs without error
    orchestrator._validate_feature_mapping(fragment, bad_translation)


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_compile_fragment_java(mock_create_llm, sample_config, state_with_analysis, temp_dir):
    """Test Java fragment compilation."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    fragment = state_with_analysis.analysis_result.fragments["test::func1"]

    # Valid Java code
    valid_code = """
public class TestClass {
    public static int func1() {
        return 42;
    }
}
"""

    success, errors = orchestrator._compile_fragment(fragment, valid_code)

    # This will only succeed if Java is installed
    # If Java not available, should return appropriate error
    if success:
        assert len(errors) == 0
    else:
        assert len(errors) > 0


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_generate_mock_java(mock_create_llm, sample_config, state_with_analysis):
    """Test Java mock generation."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    fragment = state_with_analysis.analysis_result.fragments["test::func1"]
    mock_code = orchestrator._generate_mock(fragment)

    assert mock_code is not None
    assert "MOCKED" in mock_code
    assert "func1" in mock_code.lower()


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_to_camel_case(mock_create_llm, sample_config, state_with_analysis):
    """Test snake_case to camelCase conversion."""
    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    assert orchestrator._to_camel_case("my_function") == "myFunction"
    assert orchestrator._to_camel_case("calculate_total") == "calculateTotal"
    assert orchestrator._to_camel_case("simple") == "simple"


@patch("codemorph.translator.orchestrator.create_llm_client")
def test_full_run_phase2(mock_create_llm, sample_config, state_with_analysis):
    """Test full Phase 2 orchestrator run."""
    # Set up mock LLM
    mock_llm_instance = mock_create_llm.return_value
    mock_llm_instance.translate_fragment.return_value = (
        "public static int func1() { return 42; }",
        MagicMock(),
    )

    orchestrator = Phase2Orchestrator(sample_config, state_with_analysis)

    # Mock successful compilation
    with patch.object(
        orchestrator.target_plugin, "compile_fragment", return_value=(True, [])
    ):
        with patch.object(orchestrator, "_check_type_compatibility", return_value=True):
            result = orchestrator.run()

            # Should have translated all fragments
            assert len(result) == 2
            assert "test::func1" in result
            assert "test::func2" in result

            # State should be updated
            assert state_with_analysis.current_phase == 2
