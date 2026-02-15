"""
Interactive Checkpoint UI for CodeMorph.

Rich-based terminal interface for human-in-the-loop code review.
Provides per-function approval with multi-choice options.
"""

import os
import subprocess
import tempfile
from enum import Enum

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from codemorph.config.models import (
    CodeFragment,
    TranslatedFragment,
    TranslationStatus,
)


class CheckpointAction(str, Enum):
    """Actions available at a checkpoint."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    MOCK = "mock"
    SKIP = "skip"
    APPROVE_ALL = "approve_all"


class CheckpointUI:
    """
    Rich-based interactive checkpoint interface.

    Provides per-function review with code comparison and multiple actions:
    - Approve: Accept the translation
    - Reject: Request re-translation with optional hint
    - Edit: Manually edit the translated code
    - Mock: Replace with mock implementation
    - Skip: Defer decision
    - Approve All: Skip remaining checkpoints

    Usage:
        ui = CheckpointUI()

        for fragment, translated in translations:
            action = ui.show_translation_checkpoint(fragment, translated, phase=2)

            if action == CheckpointAction.APPROVE:
                continue
            elif action == CheckpointAction.EDIT:
                edited = ui.show_edit_dialog(translated)
                translated.translated_code = edited
            # ... handle other actions
    """

    def __init__(self, console: Console | None = None):
        """
        Initialize the checkpoint UI.

        Args:
            console: Rich Console instance (creates one if not provided)
        """
        self.console = console or Console()
        self._approve_all_remaining = False
        self._current_phase = 0

    def show_translation_checkpoint(
        self,
        fragment: CodeFragment,
        translated: TranslatedFragment,
        phase: int = 2,
        index: int | None = None,
        total: int | None = None,
    ) -> CheckpointAction:
        """
        Display checkpoint for a single translation and get user action.

        Shows side-by-side comparison of source and translated code,
        along with status information and available actions.

        Args:
            fragment: The source code fragment
            translated: The translated fragment
            phase: Current phase number (2 or 3)
            index: Current fragment index (for progress display)
            total: Total fragments (for progress display)

        Returns:
            CheckpointAction indicating user's choice
        """
        # If user chose "approve all", skip interactive checkpoint
        if self._approve_all_remaining:
            return CheckpointAction.APPROVE

        self._current_phase = phase

        # Clear and show header
        self.console.print()
        progress_str = ""
        if index is not None and total is not None:
            progress_str = f" ({index + 1}/{total})"
        self.console.rule(
            f"[bold cyan]Checkpoint: {fragment.name}{progress_str}[/bold cyan]"
        )

        # Fragment metadata
        self._show_metadata(fragment, translated)

        # Side-by-side code comparison
        self._show_code_comparison(fragment, translated)

        # Show errors if any
        self._show_errors(translated)

        # Action prompt
        return self._get_user_action(translated)

    def _show_metadata(self, fragment: CodeFragment, translated: TranslatedFragment):
        """Display fragment metadata."""
        status_color = self._get_status_color(translated.status)

        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column("Label", style="dim")
        info_table.add_column("Value")

        info_table.add_row("Status", f"[{status_color}]{translated.status.value}[/{status_color}]")
        info_table.add_row("Type", fragment.fragment_type.value)
        info_table.add_row("File", str(fragment.source_file))
        info_table.add_row("Lines", f"{fragment.start_line}-{fragment.end_line}")
        if translated.retry_count > 0:
            info_table.add_row("Retries", str(translated.retry_count))

        self.console.print(info_table)
        self.console.print()

    def _show_code_comparison(
        self, fragment: CodeFragment, translated: TranslatedFragment
    ):
        """Display side-by-side source and translated code."""
        # Determine languages for syntax highlighting
        source_lang = "python"  # TODO: Get from config
        target_lang = "java"  # TODO: Get from config

        # Source code panel
        source_syntax = Syntax(
            fragment.source_code.strip(),
            source_lang,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        source_panel = Panel(
            source_syntax,
            title=f"[yellow]Source ({source_lang.title()})[/yellow]",
            border_style="yellow",
            expand=True,
        )

        # Target code panel
        target_code = translated.target_code or "// Translation failed or pending"
        target_syntax = Syntax(
            target_code.strip(),
            target_lang,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        target_panel = Panel(
            target_syntax,
            title=f"[green]Target ({target_lang.title()})[/green]",
            border_style="green",
            expand=True,
        )

        # Show side-by-side
        self.console.print(Columns([source_panel, target_panel], equal=True))

    def _show_errors(self, translated: TranslatedFragment):
        """Display any errors from translation."""
        if translated.compilation_errors:
            error_text = "\n".join(translated.compilation_errors)
            self.console.print(Panel(
                f"[red]{error_text}[/red]",
                title="[red]Compilation Errors[/red]",
                border_style="red",
            ))

        if translated.type_errors:
            error_text = "\n".join(translated.type_errors)
            self.console.print(Panel(
                f"[yellow]{error_text}[/yellow]",
                title="[yellow]Type Errors[/yellow]",
                border_style="yellow",
            ))

        if translated.io_mismatches:
            mismatch_lines = []
            for mm in translated.io_mismatches[:3]:  # Show first 3
                mismatch_lines.append(
                    f"Input: {mm.get('input', 'N/A')}\n"
                    f"Expected: {mm.get('expected', 'N/A')}\n"
                    f"Actual: {mm.get('actual', 'N/A')}"
                )
            self.console.print(Panel(
                "\n---\n".join(mismatch_lines),
                title="[magenta]I/O Mismatches[/magenta]",
                border_style="magenta",
            ))

    def _get_user_action(self, translated: TranslatedFragment) -> CheckpointAction:
        """Prompt user for action and return choice."""
        self.console.print()

        # Determine default based on status
        if translated.status in (
            TranslationStatus.COMPILED,
            TranslationStatus.TYPE_VERIFIED,
            TranslationStatus.IO_VERIFIED,
        ):
            default = "approve"
        elif translated.status == TranslationStatus.FAILED:
            default = "mock"
        else:
            default = "reject"

        # Show available actions
        actions_help = (
            "[dim]Actions: "
            "[green]approve[/green] (accept) | "
            "[yellow]reject[/yellow] (retry with hint) | "
            "[cyan]edit[/cyan] (manual edit) | "
            "[red]mock[/red] (use fallback) | "
            "[dim]skip[/dim] (defer) | "
            "[magenta]approve_all[/magenta] (accept remaining)"
            "[/dim]"
        )
        self.console.print(actions_help)

        action = Prompt.ask(
            "[bold]Action[/bold]",
            choices=["approve", "reject", "edit", "mock", "skip", "approve_all"],
            default=default,
        )

        if action == "approve_all":
            if Prompt.ask(
                "[yellow]Are you sure you want to approve all remaining?[/yellow]",
                choices=["yes", "no"],
                default="no",
            ) == "yes":
                self._approve_all_remaining = True
                return CheckpointAction.APPROVE

        return CheckpointAction(action)

    def show_edit_dialog(self, translated: TranslatedFragment) -> str:
        """
        Allow user to manually edit the translation.

        Opens the user's preferred editor ($EDITOR) with the current
        translated code. Returns the edited code.

        Args:
            translated: The translated fragment to edit

        Returns:
            The edited code string
        """
        self.console.print("[yellow]Opening editor...[/yellow]")
        self.console.print(
            "[dim]Edit the code, save, and close the editor to continue.[/dim]"
        )

        # Get editor from environment
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))

        # Create temp file with current code
        with tempfile.NamedTemporaryFile(
            suffix=".java",  # TODO: Get from target language
            mode="w",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(translated.target_code or "")
            temp_path = f.name

        try:
            # Open editor
            subprocess.call([editor, temp_path])

            # Read edited content
            with open(temp_path, "r", encoding="utf-8") as f:
                edited_code = f.read()

            self.console.print("[green]Edit saved.[/green]")
            return edited_code

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def show_reject_hint_prompt(self) -> str:
        """
        Prompt user for a hint when rejecting a translation.

        Returns:
            The hint string to provide to the LLM
        """
        self.console.print()
        self.console.print(
            "[dim]Provide a hint to help improve the translation "
            "(or press Enter to skip):[/dim]"
        )
        hint = Prompt.ask("Hint", default="")
        return hint

    def show_phase_summary(
        self,
        phase: int,
        stats: dict,
        show_continue_prompt: bool = True,
    ) -> bool:
        """
        Show phase completion summary.

        Args:
            phase: Phase number (1, 2, or 3)
            stats: Dictionary with statistics
            show_continue_prompt: Whether to ask to continue

        Returns:
            True if user wants to continue, False otherwise
        """
        self.console.print()
        self.console.rule(f"[bold cyan]Phase {phase} Summary[/bold cyan]")

        # Create stats table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in stats.items():
            # Format the key nicely
            display_key = key.replace("_", " ").title()
            table.add_row(display_key, str(value))

        self.console.print(table)
        self.console.print()

        if show_continue_prompt:
            return Prompt.ask(
                f"[bold]Proceed to Phase {phase + 1}?[/bold]",
                choices=["yes", "no"],
                default="yes",
            ) == "yes"

        return True

    def show_library_mapping_checkpoint(
        self,
        library: str,
        suggested_mapping: dict,
    ) -> tuple[str, dict | None]:
        """
        Show checkpoint for library mapping verification.

        Args:
            library: Source library name
            suggested_mapping: LLM-suggested mapping details

        Returns:
            Tuple of (action, optional user_mapping)
            action is one of: "approve", "edit", "skip"
        """
        self.console.print()
        self.console.rule(f"[bold yellow]Library Mapping: {library}[/bold yellow]")

        # Display suggested mapping
        table = Table(title="Suggested Mapping", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Source Library", library)
        table.add_row("Target Library", suggested_mapping.get("target_library", "N/A"))
        table.add_row(
            "Maven Dependency",
            suggested_mapping.get("maven_dependency") or "None (built-in)",
        )
        table.add_row(
            "Imports",
            "\n".join(suggested_mapping.get("target_imports", [])) or "N/A",
        )
        if suggested_mapping.get("notes"):
            table.add_row("Notes", suggested_mapping["notes"])

        self.console.print(table)
        self.console.print()

        action = Prompt.ask(
            "[bold]Action[/bold]",
            choices=["approve", "edit", "skip"],
            default="approve",
        )

        if action == "edit":
            # Allow user to provide custom mapping
            target = Prompt.ask(
                "Target library",
                default=suggested_mapping.get("target_library", ""),
            )
            maven = Prompt.ask(
                "Maven dependency (or leave empty)",
                default=suggested_mapping.get("maven_dependency", ""),
            )
            imports_str = Prompt.ask(
                "Imports (comma-separated)",
                default=",".join(suggested_mapping.get("target_imports", [])),
            )
            imports = [i.strip() for i in imports_str.split(",") if i.strip()]

            user_mapping = {
                "target_library": target,
                "maven_dependency": maven or None,
                "target_imports": imports,
            }
            return ("edit", user_mapping)

        return (action, None)

    def reset_approve_all(self):
        """Reset the approve-all flag (e.g., for new phase)."""
        self._approve_all_remaining = False

    def _get_status_color(self, status: TranslationStatus) -> str:
        """Get color for a translation status."""
        colors = {
            TranslationStatus.PENDING: "dim",
            TranslationStatus.IN_PROGRESS: "yellow",
            TranslationStatus.TRANSLATED: "blue",
            TranslationStatus.COMPILED: "green",
            TranslationStatus.TYPE_VERIFIED: "green",
            TranslationStatus.IO_VERIFIED: "bold green",
            TranslationStatus.MOCKED: "magenta",
            TranslationStatus.FAILED: "red",
            TranslationStatus.HUMAN_REVIEW: "yellow",
        }
        return colors.get(status, "white")

    def show_warning(self, message: str):
        """Display a warning message."""
        self.console.print(f"[yellow]Warning: {message}[/yellow]")

    def show_error(self, message: str):
        """Display an error message."""
        self.console.print(f"[red]Error: {message}[/red]")

    def show_success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green]{message}[/green]")

    def show_info(self, message: str):
        """Display an info message."""
        self.console.print(f"[cyan]{message}[/cyan]")
