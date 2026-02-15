"""
Unit tests for state persistence system.
"""

import tempfile
from pathlib import Path

import pytest

from codemorph.config.loader import create_config_from_args
from codemorph.config.models import (
    AnalysisResult,
    CodeFragment,
    FragmentType,
    TranslatedFragment,
    TranslationStatus,
)
from codemorph.state.persistence import StateManager, TranslationState


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return create_config_from_args(
        source_dir=Path("."),
        source_lang="python",
        source_version="3.10",
        target_lang="java",
        target_version="17",
        output_dir=Path("./output"),
        build_system="gradle",
    )


@pytest.fixture
def sample_fragment():
    """Create a sample code fragment."""
    return CodeFragment(
        id="test::sample_function",
        name="sample_function",
        fragment_type=FragmentType.FUNCTION,
        source_file=Path("test.py"),
        start_line=1,
        end_line=5,
        source_code="def sample_function():\n    pass",
        dependencies=[],
        signature="def sample_function() -> None",
        docstring="A sample function for testing",
    )


def test_create_translation_state(sample_config):
    """Test creating a new translation state."""
    state = TranslationState(sample_config, Path("."))

    assert state.session_id is not None
    assert state.current_phase == 0
    assert state.analysis_result is None
    assert len(state.translated_fragments) == 0


def test_save_and_load_state(temp_state_dir, sample_config):
    """Test saving and loading state."""
    # Override state dir
    sample_config.project.state_dir = temp_state_dir

    # Create state
    state = TranslationState(sample_config, Path("."))
    state.current_phase = 1
    state.metadata["test_key"] = "test_value"

    # Save
    state_file = state.save()
    assert state_file.exists()

    # Load
    loaded_state = TranslationState.load(state_file)

    assert loaded_state.session_id == state.session_id
    assert loaded_state.current_phase == 1
    assert loaded_state.metadata["test_key"] == "test_value"


def test_state_with_analysis_result(temp_state_dir, sample_config, sample_fragment):
    """Test state with analysis result."""
    sample_config.project.state_dir = temp_state_dir

    state = TranslationState(sample_config, Path("."))

    # Create analysis result
    analysis = AnalysisResult(
        fragments={"test::sample_function": sample_fragment},
        translation_order=["test::sample_function"],
        circular_dependencies=[],
    )

    state.set_analysis_result(analysis)

    # Save and load
    state_file = state.save()
    loaded_state = TranslationState.load(state_file)

    assert loaded_state.analysis_result is not None
    assert len(loaded_state.analysis_result.fragments) == 1
    assert "test::sample_function" in loaded_state.analysis_result.fragments


def test_update_fragment_status(sample_config, sample_fragment):
    """Test updating fragment translation status."""
    state = TranslationState(sample_config, Path("."))

    # Create translated fragment
    translated = TranslatedFragment(
        fragment=sample_fragment,
        status=TranslationStatus.PENDING,
    )

    state.update_fragment(translated)
    assert sample_fragment.id in state.translated_fragments

    # Update status
    state.mark_fragment_status(sample_fragment.id, TranslationStatus.COMPILED)
    assert state.translated_fragments[sample_fragment.id].status == TranslationStatus.COMPILED


def test_get_progress(sample_config, sample_fragment):
    """Test progress tracking."""
    state = TranslationState(sample_config, Path("."))

    # Initially no progress
    progress = state.get_progress()
    assert progress["total_fragments"] == 0
    assert progress["completed"] == 0

    # Add analysis result
    analysis = AnalysisResult(
        fragments={
            "frag1": sample_fragment,
            "frag2": sample_fragment,
        },
        translation_order=["frag1", "frag2"],
    )
    state.set_analysis_result(analysis)

    # Add some translated fragments
    translated1 = TranslatedFragment(
        fragment=sample_fragment,
        status=TranslationStatus.IO_VERIFIED,
    )
    state.translated_fragments["frag1"] = translated1

    translated2 = TranslatedFragment(
        fragment=sample_fragment,
        status=TranslationStatus.IN_PROGRESS,
    )
    state.translated_fragments["frag2"] = translated2

    progress = state.get_progress()
    assert progress["total_fragments"] == 2
    assert progress["completed"] == 1
    assert progress["in_progress"] == 1
    assert progress["percentage"] == 50.0


def test_checkpoint_creation(temp_state_dir, sample_config):
    """Test creating checkpoints."""
    sample_config.project.state_dir = temp_state_dir

    state = TranslationState(sample_config, Path("."))
    state.current_phase = 1

    # Create checkpoint
    checkpoint_path = state.create_checkpoint("phase1_complete")
    assert checkpoint_path.exists()
    assert "phase1_complete" in checkpoint_path.name

    # List checkpoints
    checkpoints = TranslationState.list_checkpoints(temp_state_dir)
    assert len(checkpoints) >= 1
    assert checkpoints[0]["name"] == "phase1_complete"


def test_state_manager(temp_state_dir, sample_config):
    """Test StateManager functionality."""
    # Ensure state saves go to the same dir that the manager searches
    sample_config.project.state_dir = temp_state_dir
    manager = StateManager(temp_state_dir)

    # Create session
    state = manager.create_session(sample_config, Path("."))
    state.current_phase = 1
    state.save()

    # List sessions
    sessions = manager.list_sessions()
    assert len(sessions) >= 1

    # Load latest
    latest_state = manager.load_latest_session()
    assert latest_state.current_phase == 1


def test_export_report(temp_state_dir, sample_config, sample_fragment):
    """Test exporting translation report."""
    sample_config.project.state_dir = temp_state_dir

    state = TranslationState(sample_config, Path("."))

    # Add some data
    analysis = AnalysisResult(
        fragments={"frag1": sample_fragment},
        translation_order=["frag1"],
    )
    state.set_analysis_result(analysis)

    translated = TranslatedFragment(
        fragment=sample_fragment,
        status=TranslationStatus.MOCKED,
        is_mocked=True,
        mock_reason="Could not translate decorator",
    )
    state.update_fragment(translated)

    # Export report
    report_path = temp_state_dir / "report.md"
    state.export_report(report_path)

    assert report_path.exists()

    # Check content
    content = report_path.read_text()
    assert "Translation Report" in content
    assert "Mocked Fragments" in content
    assert "sample_function" in content
