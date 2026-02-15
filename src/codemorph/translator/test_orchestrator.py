"""
Test Translation Orchestrator for CodeMorph.

Orchestrates the translation of test files after Phase 2.
Converts source language tests (e.g., pytest) to target language tests (e.g., JUnit).
"""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codemorph.config.models import (
    CodeMorphConfig,
    LanguageType,
    TranslatedFragment,
)
from codemorph.languages.registry import get_plugin
from codemorph.state.symbol_registry import SymbolRegistry
from codemorph.translator.test_translator import TestTranslator


@dataclass
class TranslatedTestFile:
    """Result of translating a test file."""

    source_file: Path
    target_file: Path
    translated_code: str
    compilation_success: bool
    compilation_errors: list[str] = field(default_factory=list)
    test_count: int = 0  # Number of test methods detected


class TestTranslationOrchestrator:
    """
    Orchestrates translation of test files after Phase 2.

    Workflow:
    1. Discover test files in source test directory
    2. For each test file:
       a. Use LLM to translate to target language
       b. Verify compilation
       c. Save to output directory
    3. Report results

    Usage:
        orchestrator = TestTranslationOrchestrator(config, symbol_registry, llm_client)

        results = orchestrator.translate_test_suite(
            test_dir=Path("tests"),
            translated_fragments=state.translated_fragments,
            output_dir=Path("output")
        )
    """

    def __init__(
        self,
        config: CodeMorphConfig,
        symbol_registry: SymbolRegistry,
        llm_client,  # LLMClient
    ):
        """
        Initialize the test translation orchestrator.

        Args:
            config: CodeMorph configuration
            symbol_registry: Symbol registry with name mappings
            llm_client: LLM client for generating translations
        """
        self.config = config
        self.symbol_registry = symbol_registry
        self.llm_client = llm_client
        self.translator = TestTranslator(config, llm_client, symbol_registry)
        self.console = Console()

        # Get target language plugin for compilation
        self.target_plugin = get_plugin(
            config.project.target.language,
            config.project.target.version
        )

    def translate_test_suite(
        self,
        test_dir: Path,
        translated_fragments: dict[str, TranslatedFragment],
        output_dir: Path,
    ) -> list[TranslatedTestFile]:
        """
        Translate all test files in the test directory.

        Args:
            test_dir: Directory containing source test files
            translated_fragments: Dict of translated code fragments
            output_dir: Directory for translated output

        Returns:
            List of TranslatedTestFile results
        """
        results: list[TranslatedTestFile] = []

        # Discover test files
        test_files = self._discover_test_files(test_dir)

        if not test_files:
            self.console.print("[yellow]No test files found to translate.[/yellow]")
            return results

        self.console.print(
            f"\n[bold cyan]Translating {len(test_files)} test file(s)...[/bold cyan]\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Translating tests...", total=len(test_files))

            for test_file in test_files:
                progress.update(task, description=f"Translating: {test_file.name}")

                result = self._translate_test_file(
                    test_file, translated_fragments, output_dir
                )
                results.append(result)

                progress.advance(task)

        # Display summary
        self._display_summary(results)

        return results

    def _discover_test_files(self, test_dir: Path) -> list[Path]:
        """
        Discover test files in the test directory.

        Args:
            test_dir: Directory to search

        Returns:
            List of test file paths
        """
        if not test_dir.exists():
            return []

        source_lang = self.config.project.source.language
        patterns = []

        if source_lang == LanguageType.PYTHON:
            patterns = ["**/test_*.py", "**/*_test.py"]
        elif source_lang == LanguageType.JAVA:
            patterns = ["**/*Test.java", "**/*Tests.java"]

        test_files: list[Path] = []
        for pattern in patterns:
            test_files.extend(test_dir.glob(pattern))

        # Filter out __pycache__ and other non-test files
        test_files = [
            f for f in test_files
            if "__pycache__" not in str(f) and f.is_file()
        ]

        return sorted(set(test_files))

    def _translate_test_file(
        self,
        test_file: Path,
        translated_fragments: dict[str, TranslatedFragment],
        output_dir: Path,
    ) -> TranslatedTestFile:
        """
        Translate a single test file.

        Args:
            test_file: Path to source test file
            translated_fragments: Translated code fragments
            output_dir: Output directory

        Returns:
            TranslatedTestFile result
        """
        # Get output path
        target_file = self.translator.get_test_file_mapping(test_file, output_dir)

        try:
            # Translate the test file
            translated_code = self.translator.translate_test_file(
                test_file, translated_fragments
            )

            # Count test methods (approximate)
            test_count = self._count_test_methods(translated_code)

            # Verify compilation
            compile_dir = output_dir / "build" / "test"
            compile_dir.mkdir(parents=True, exist_ok=True)

            success, errors = self.target_plugin.compile_fragment(
                translated_code,
                compile_dir,
            )

            result = TranslatedTestFile(
                source_file=test_file,
                target_file=target_file,
                translated_code=translated_code,
                compilation_success=success,
                compilation_errors=errors,
                test_count=test_count,
            )

            # Save to output if compilation succeeded
            if success:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(translated_code, encoding="utf-8")

            return result

        except Exception as e:
            return TranslatedTestFile(
                source_file=test_file,
                target_file=target_file,
                translated_code="",
                compilation_success=False,
                compilation_errors=[str(e)],
                test_count=0,
            )

    def _count_test_methods(self, code: str) -> int:
        """
        Count the number of test methods in translated code.

        This is an approximation based on annotations/patterns.
        """
        target_lang = self.config.project.target.language

        if target_lang == LanguageType.JAVA:
            # Count @Test annotations
            return code.count("@Test")
        elif target_lang == LanguageType.PYTHON:
            # Count test_ functions
            import re
            return len(re.findall(r"def test_", code))

        return 0

    def _display_summary(self, results: list[TranslatedTestFile]):
        """Display translation summary."""
        self.console.print()

        table = Table(title="Test Translation Summary", show_header=True)
        table.add_column("Test File", style="cyan")
        table.add_column("Tests", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Output", style="dim")

        success_count = 0
        total_tests = 0

        for result in results:
            status = "[green]✓ Compiled[/green]" if result.compilation_success else "[red]✗ Failed[/red]"
            output = str(result.target_file.name) if result.compilation_success else "N/A"

            table.add_row(
                result.source_file.name,
                str(result.test_count),
                status,
                output,
            )

            if result.compilation_success:
                success_count += 1
                total_tests += result.test_count

        self.console.print(table)
        self.console.print()

        # Summary line
        self.console.print(
            f"[bold]Translated: {success_count}/{len(results)} test files "
            f"({total_tests} test methods)[/bold]"
        )

        # Show errors for failed files
        failed = [r for r in results if not r.compilation_success]
        if failed:
            self.console.print()
            self.console.print("[bold red]Failed Translations:[/bold red]")
            for result in failed:
                self.console.print(f"  [red]• {result.source_file.name}[/red]")
                for error in result.compilation_errors[:3]:  # Show first 3 errors
                    self.console.print(f"    [dim]{error}[/dim]")


def run_test_translation(
    config: CodeMorphConfig,
    symbol_registry: SymbolRegistry,
    llm_client,
    test_dir: Path,
    translated_fragments: dict[str, TranslatedFragment],
    output_dir: Path,
) -> list[TranslatedTestFile]:
    """
    Convenience function to run test translation.

    Args:
        config: CodeMorph configuration
        symbol_registry: Symbol registry with mappings
        llm_client: LLM client
        test_dir: Source test directory
        translated_fragments: Translated code fragments
        output_dir: Output directory

    Returns:
        List of TranslatedTestFile results
    """
    orchestrator = TestTranslationOrchestrator(config, symbol_registry, llm_client)
    return orchestrator.translate_test_suite(test_dir, translated_fragments, output_dir)
