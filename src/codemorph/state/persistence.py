"""
State persistence layer for CodeMorph.

Handles saving and restoring translation state, enabling resumable sessions.
"""

import json
import time
from pathlib import Path
from typing import Any

import orjson

from codemorph.config.models import (
    AnalysisResult,
    CodeFragment,
    CodeMorphConfig,
    TranslatedFragment,
    TranslationStatus,
)


class TranslationState:
    """
    Represents the complete state of a translation session.

    This allows the system to resume from any point if interrupted.
    """

    def __init__(
        self,
        config: CodeMorphConfig,
        project_root: Path,
        session_id: str | None = None,
    ):
        self.config = config
        self.project_root = project_root
        self.session_id = session_id or self._generate_session_id()
        self.state_dir = config.project.state_dir
        self.created_at = time.time()
        self.updated_at = time.time()

        # State components
        self.analysis_result: AnalysisResult | None = None
        self.translated_fragments: dict[str, TranslatedFragment] = {}
        self.current_phase: int = 0  # 0=not started, 1=partitioning, 2=type-driven, 3=semantics
        self.current_fragment_index: int = 0
        self.errors: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        from uuid import uuid4
        return f"session_{int(time.time())}_{str(uuid4())[:8]}"

    # =========================================================================
    # State Updates
    # =========================================================================

    def set_analysis_result(self, result: AnalysisResult):
        """Store the Phase 1 analysis result."""
        self.analysis_result = result
        self.current_phase = 1
        self.updated_at = time.time()

    def update_fragment(self, fragment: TranslatedFragment):
        """Update the state of a translated fragment."""
        self.translated_fragments[fragment.fragment.id] = fragment
        self.updated_at = time.time()

    def mark_fragment_status(self, fragment_id: str, status: TranslationStatus):
        """Update just the status of a fragment."""
        if fragment_id in self.translated_fragments:
            self.translated_fragments[fragment_id].status = status
            self.updated_at = time.time()

    def add_error(self, fragment_id: str, error_type: str, message: str):
        """Log an error during translation."""
        self.errors.append({
            "fragment_id": fragment_id,
            "error_type": error_type,
            "message": message,
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    def set_current_phase(self, phase: int):
        """Update the current phase."""
        self.current_phase = phase
        self.updated_at = time.time()

    def advance_fragment(self):
        """Move to the next fragment in the translation order."""
        self.current_fragment_index += 1
        self.updated_at = time.time()

    # =========================================================================
    # Query State
    # =========================================================================

    def get_progress(self) -> dict[str, Any]:
        """Get translation progress statistics."""
        if not self.analysis_result:
            return {
                "phase": "Not started",
                "total_fragments": 0,
                "completed": 0,
                "in_progress": 0,
                "failed": 0,
                "percentage": 0.0,
            }

        total = len(self.analysis_result.fragments)
        completed = sum(
            1 for f in self.translated_fragments.values()
            if f.status in (TranslationStatus.IO_VERIFIED, TranslationStatus.MOCKED)
        )
        in_progress = sum(
            1 for f in self.translated_fragments.values()
            if f.status == TranslationStatus.IN_PROGRESS
        )
        failed = sum(
            1 for f in self.translated_fragments.values()
            if f.status == TranslationStatus.FAILED
        )

        return {
            "phase": f"Phase {self.current_phase}",
            "total_fragments": total,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "percentage": (completed / total * 100) if total > 0 else 0.0,
        }

    def get_next_fragment_id(self) -> str | None:
        """Get the next fragment to translate."""
        if not self.analysis_result or not self.analysis_result.translation_order:
            return None

        if self.current_fragment_index >= len(self.analysis_result.translation_order):
            return None  # All done

        return self.analysis_result.translation_order[self.current_fragment_index]

    def get_pending_fragments(self) -> list[str]:
        """Get all fragments that haven't been translated yet."""
        if not self.analysis_result:
            return []

        return [
            fid for fid in self.analysis_result.translation_order
            if fid not in self.translated_fragments
            or self.translated_fragments[fid].status == TranslationStatus.PENDING
        ]

    def get_mocked_fragments(self) -> list[TranslatedFragment]:
        """Get all fragments that are mocked."""
        return [
            f for f in self.translated_fragments.values()
            if f.status == TranslationStatus.MOCKED
        ]

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> Path:
        """
        Save the complete state to disk.

        Returns:
            Path to the saved state file
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.state_dir / f"{self.session_id}.json"

        state_dict = self._to_dict()

        # Use orjson for fast serialization
        with open(state_file, "wb") as f:
            f.write(orjson.dumps(
                state_dict,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            ))

        # Also save a "latest" symlink for easy resume
        latest_link = self.state_dir / "latest.json"
        if latest_link.exists():
            latest_link.unlink()

        # Write the same content to latest
        with open(latest_link, "wb") as f:
            f.write(orjson.dumps(
                state_dict,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            ))

        return state_file

    def _to_dict(self) -> dict[str, Any]:
        """Convert state to a serializable dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_phase": self.current_phase,
            "current_fragment_index": self.current_fragment_index,
            "config": self.config.model_dump(mode='json'),
            "analysis_result": self.analysis_result.model_dump(mode='json') if self.analysis_result else None,
            "translated_fragments": {
                fid: frag.model_dump(mode='json')
                for fid, frag in self.translated_fragments.items()
            },
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, state_file: Path) -> "TranslationState":
        """
        Load state from a file.

        Args:
            state_file: Path to the state file

        Returns:
            Restored TranslationState instance
        """
        with open(state_file, "rb") as f:
            state_dict = orjson.loads(f.read())

        # Reconstruct config
        config = CodeMorphConfig(**state_dict["config"])

        # Create instance
        instance = cls(
            config=config,
            project_root=Path(config.project.source.root),
            session_id=state_dict["session_id"],
        )

        # Restore state
        instance.created_at = state_dict["created_at"]
        instance.updated_at = state_dict["updated_at"]
        instance.current_phase = state_dict["current_phase"]
        instance.current_fragment_index = state_dict["current_fragment_index"]
        instance.errors = state_dict["errors"]
        instance.metadata = state_dict["metadata"]

        # Restore analysis result
        if state_dict["analysis_result"]:
            instance.analysis_result = AnalysisResult(**state_dict["analysis_result"])

        # Restore translated fragments
        for fid, frag_dict in state_dict["translated_fragments"].items():
            instance.translated_fragments[fid] = TranslatedFragment(**frag_dict)

        return instance

    @classmethod
    def load_latest(cls, state_dir: Path) -> "TranslationState":
        """
        Load the latest state from a directory.

        Args:
            state_dir: Directory containing state files

        Returns:
            Most recent TranslationState

        Raises:
            FileNotFoundError: If no state files exist
        """
        latest_file = state_dir / "latest.json"
        if latest_file.exists():
            return cls.load(latest_file)

        # Fallback: find most recent session file
        state_files = sorted(state_dir.glob("session_*.json"), reverse=True)
        if not state_files:
            raise FileNotFoundError(f"No state files found in {state_dir}")

        return cls.load(state_files[0])

    # =========================================================================
    # Checkpoints
    # =========================================================================

    def create_checkpoint(self, name: str) -> Path:
        """
        Create a named checkpoint.

        Args:
            name: Checkpoint name (e.g., "phase1_complete")

        Returns:
            Path to the checkpoint file
        """
        checkpoint_dir = self.state_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{name}_{int(time.time())}.json"

        state_dict = self._to_dict()
        state_dict["checkpoint_name"] = name

        with open(checkpoint_file, "wb") as f:
            f.write(orjson.dumps(
                state_dict,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            ))

        return checkpoint_file

    @classmethod
    def list_checkpoints(cls, state_dir: Path) -> list[dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata
        """
        checkpoint_dir = state_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for ckpt_file in sorted(checkpoint_dir.glob("*.json")):
            try:
                with open(ckpt_file, "rb") as f:
                    data = orjson.loads(f.read())
                checkpoints.append({
                    "file": ckpt_file,
                    "name": data.get("checkpoint_name", "unknown"),
                    "session_id": data.get("session_id"),
                    "phase": data.get("current_phase"),
                    "timestamp": data.get("updated_at"),
                })
            except:
                continue

        return checkpoints

    # =========================================================================
    # Export
    # =========================================================================

    def export_report(self, output_path: Path):
        """
        Export a human-readable translation report.

        Args:
            output_path: Path for the report (Markdown format)
        """
        progress = self.get_progress()
        mocked = self.get_mocked_fragments()

        report_lines = [
            f"# CodeMorph Translation Report",
            f"",
            f"**Session ID**: {self.session_id}",
            f"**Created**: {time.ctime(self.created_at)}",
            f"**Last Updated**: {time.ctime(self.updated_at)}",
            f"**Translation Type**: {self.config.get_translation_type()}",
            f"",
            f"## Progress",
            f"",
            f"- **Phase**: {progress['phase']}",
            f"- **Total Fragments**: {progress['total_fragments']}",
            f"- **Completed**: {progress['completed']} ({progress['percentage']:.1f}%)",
            f"- **In Progress**: {progress['in_progress']}",
            f"- **Failed**: {progress['failed']}",
            f"",
        ]

        # Mocked fragments
        if mocked:
            report_lines.extend([
                f"## Mocked Fragments (Require Manual Attention)",
                f"",
                f"The following fragments could not be automatically translated:",
                f"",
            ])
            for frag in mocked:
                report_lines.append(f"- **{frag.fragment.name}** ({frag.fragment.fragment_type.value})")
                if frag.mock_reason:
                    report_lines.append(f"  - Reason: {frag.mock_reason}")
                report_lines.append("")

        # Errors
        if self.errors:
            report_lines.extend([
                f"## Errors Encountered",
                f"",
            ])
            for error in self.errors[-10:]:  # Last 10 errors
                report_lines.append(f"- **{error['fragment_id']}**: {error['error_type']}")
                report_lines.append(f"  - {error['message']}")
                report_lines.append("")

        # Fragment details
        if self.translated_fragments:
            report_lines.extend([
                f"## Fragment Status",
                f"",
                f"| Fragment | Type | Status | Retries |",
                f"|----------|------|--------|---------|",
            ])
            for frag in self.translated_fragments.values():
                report_lines.append(
                    f"| {frag.fragment.name} | {frag.fragment.fragment_type.value} | "
                    f"{frag.status.value} | {frag.retry_count} |"
                )
            report_lines.append("")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))


class StateManager:
    """Manager for handling multiple translation states."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, config: CodeMorphConfig, project_root: Path) -> TranslationState:
        """Create a new translation session."""
        return TranslationState(config, project_root)

    def load_session(self, session_id: str) -> TranslationState:
        """Load a specific session by ID."""
        state_file = self.state_dir / f"{session_id}.json"
        return TranslationState.load(state_file)

    def load_latest_session(self) -> TranslationState:
        """Load the most recent session."""
        return TranslationState.load_latest(self.state_dir)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all available sessions."""
        sessions = []
        for state_file in sorted(self.state_dir.glob("session_*.json"), reverse=True):
            try:
                with open(state_file, "rb") as f:
                    data = orjson.loads(f.read())
                sessions.append({
                    "session_id": data.get("session_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "phase": data.get("current_phase"),
                    "file": state_file,
                })
            except:
                continue
        return sessions

    def cleanup_old_sessions(self, keep_recent: int = 5):
        """
        Remove old session files, keeping only the most recent.

        Args:
            keep_recent: Number of recent sessions to keep
        """
        sessions = self.list_sessions()
        if len(sessions) <= keep_recent:
            return

        for session in sessions[keep_recent:]:
            try:
                session["file"].unlink()
            except:
                pass
