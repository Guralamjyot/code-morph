"""
Phase 1 Orchestrator: Project Partitioning and Analysis.

This orchestrator coordinates the analysis phase, which includes:
1. Discovering and parsing source files
2. Extracting code fragments
3. Analyzing dependencies
4. Building dependency graph
5. Determining translation order
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codemorph.analyzer.graph_builder import DependencyGraphBuilder
from codemorph.config.models import AnalysisResult, CodeFragment, CodeMorphConfig
from codemorph.languages.registry import LanguagePluginRegistry
from codemorph.state.persistence import TranslationState

console = Console()


class Phase1Orchestrator:
    """Orchestrates Phase 1: Project Partitioning."""

    def __init__(self, config: CodeMorphConfig, state: TranslationState):
        self.config = config
        self.state = state

        # Get source language plugin
        self.source_plugin = LanguagePluginRegistry.get_plugin(
            config.project.source.language,
            config.project.source.version,
        )

    def run(self) -> AnalysisResult:
        """
        Execute Phase 1: Project Partitioning.

        Returns:
            AnalysisResult with all fragments and dependency graph
        """
        console.print("\n[bold cyan]Phase 1: Project Partitioning[/bold cyan]\n")

        # Step 1: Discover source files
        source_files = self._discover_source_files()
        console.print(f"[green]✓[/green] Found {len(source_files)} source files")

        # Step 2: Parse and extract fragments
        all_fragments = self._extract_fragments(source_files)
        console.print(f"[green]✓[/green] Extracted {len(all_fragments)} code fragments")

        # Step 3: Analyze dependencies
        self._analyze_dependencies(all_fragments)
        console.print(f"[green]✓[/green] Analyzed dependencies")

        # Step 4: Build dependency graph
        graph_builder = DependencyGraphBuilder()
        fragment_dict = {f.id: f for f in all_fragments}
        graph_builder.build_graph(fragment_dict)

        # Step 5: Create analysis result
        result = graph_builder.create_analysis_result(fragment_dict)

        # Save graph visualization if possible
        graph_path = self.config.project.state_dir / "dependency_graph.png"
        try:
            graph_builder.visualize_graph(graph_path)
            result.dependency_graph_path = graph_path
            console.print(f"[green]✓[/green] Saved dependency graph to {graph_path}")
        except:
            # Matplotlib not available, save as GraphML
            graph_path = self.config.project.state_dir / "dependency_graph.graphml"
            graph_builder.save_graph(graph_path)
            result.dependency_graph_path = graph_path
            console.print(f"[green]✓[/green] Saved dependency graph to {graph_path}")

        # Step 6: Display results
        self._display_analysis_results(result)

        # Step 7: Save state
        self.state.set_analysis_result(result)
        self.state.save()
        console.print(f"\n[green]✓[/green] Saved analysis state")

        return result

    def _discover_source_files(self) -> list[Path]:
        """
        Discover all source files in the source root.

        Returns:
            List of source file paths
        """
        source_root = self.config.project.source.root
        extensions = self.source_plugin.file_extensions
        exclude_patterns = self.config.project.source.exclude_patterns

        source_files = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering source files...", total=None)

            for ext in extensions:
                for file_path in source_root.rglob(f"*{ext}"):
                    # Skip excluded patterns
                    if any(pattern in str(file_path) for pattern in exclude_patterns):
                        continue

                    # Skip test files if separate test root is specified
                    if self.config.project.source.test_root:
                        if self.config.project.source.test_root in file_path.parents:
                            continue

                    source_files.append(file_path)

            progress.update(task, completed=True)

        return source_files

    def _extract_fragments(self, source_files: list[Path]) -> list[CodeFragment]:
        """
        Parse files and extract code fragments.

        Args:
            source_files: List of source files to parse

        Returns:
            List of all extracted fragments
        """
        all_fragments = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing source files...", total=len(source_files))

            for file_path in source_files:
                try:
                    # Parse the file
                    ast = self.source_plugin.parse_file(file_path)

                    # Extract fragments
                    fragments = self.source_plugin.extract_fragments(file_path, ast)

                    # Extract signatures and docstrings
                    for fragment in fragments:
                        fragment.signature = self.source_plugin.extract_signature(fragment)
                        fragment.docstring = self.source_plugin.extract_docstring(fragment)

                    all_fragments.extend(fragments)

                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Failed to parse {file_path}: {e}")

                progress.advance(task)

        return all_fragments

    def _analyze_dependencies(self, fragments: list[CodeFragment]):
        """
        Analyze dependencies for each fragment.

        Args:
            fragments: List of fragments to analyze
        """
        # Create a map of fragment names to IDs for dependency resolution
        name_to_id = {}
        for fragment in fragments:
            name_to_id[fragment.name] = fragment.id

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing dependencies...", total=len(fragments))

            for fragment in fragments:
                # Get raw dependencies (these are names, not IDs)
                try:
                    # This is a simplified dependency analysis
                    # In a real implementation, we'd need to pass the full AST
                    raw_deps = self.source_plugin.get_fragment_dependencies(fragment, None)

                    # Resolve names to fragment IDs
                    resolved_deps = []
                    for dep_name in raw_deps:
                        if dep_name in name_to_id:
                            resolved_deps.append(name_to_id[dep_name])

                    fragment.dependencies = resolved_deps

                except Exception as e:
                    console.print(
                        f"[yellow]⚠[/yellow] Failed to analyze dependencies for {fragment.name}: {e}"
                    )
                    fragment.dependencies = []

                progress.advance(task)

    def _display_analysis_results(self, result: AnalysisResult):
        """
        Display analysis results in a nice format.

        Args:
            result: The analysis result to display
        """
        console.print("\n[bold]Analysis Summary[/bold]\n")

        # Basic stats
        stats_table = Table(show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Fragments", str(len(result.fragments)))
        stats_table.add_row("Translation Order Determined", str(len(result.translation_order) > 0))
        stats_table.add_row("Circular Dependencies", str(len(result.circular_dependencies)))

        console.print(stats_table)

        # Fragment type breakdown
        from collections import Counter

        type_counts = Counter(f.fragment_type.value for f in result.fragments.values())

        console.print("\n[bold]Fragment Types[/bold]\n")
        type_table = Table()
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", justify="right", style="green")

        for ftype, count in type_counts.most_common():
            type_table.add_row(ftype, str(count))

        console.print(type_table)

        # Top 10 most complex fragments
        if result.complexity_scores:
            console.print("\n[bold]Most Complex Fragments[/bold]\n")

            sorted_by_complexity = sorted(
                result.complexity_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            complexity_table = Table()
            complexity_table.add_column("Fragment", style="cyan")
            complexity_table.add_column("Type", style="yellow")
            complexity_table.add_column("Complexity", justify="right", style="green")

            for frag_id, score in sorted_by_complexity:
                fragment = result.fragments.get(frag_id)
                if fragment:
                    complexity_table.add_row(
                        fragment.name,
                        fragment.fragment_type.value,
                        str(score),
                    )

            console.print(complexity_table)

        # Circular dependencies warning
        if result.circular_dependencies:
            console.print(
                f"\n[yellow]⚠ Warning: Found {len(result.circular_dependencies)} "
                f"circular dependency groups[/yellow]"
            )
            for i, cycle in enumerate(result.circular_dependencies[:3], 1):
                console.print(f"  {i}. {' → '.join(cycle)}")

            if len(result.circular_dependencies) > 3:
                console.print(f"  ... and {len(result.circular_dependencies) - 3} more")


def run_phase1_analysis(config: CodeMorphConfig) -> tuple[AnalysisResult, TranslationState]:
    """
    Convenience function to run Phase 1 analysis.

    Args:
        config: CodeMorph configuration

    Returns:
        Tuple of (AnalysisResult, TranslationState)
    """
    # Create state
    state = TranslationState(config, config.project.source.root)

    # Create and run orchestrator
    orchestrator = Phase1Orchestrator(config, state)
    result = orchestrator.run()

    return result, state
