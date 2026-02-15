#!/usr/bin/env python3
"""
Phase 1 Demo: Project Partitioning and Analysis

Demonstrates the complete Phase 1 workflow:
  - Source file discovery
  - AST parsing and fragment extraction
  - Dependency graph building
  - Translation order determination

Usage:
    # Default: analyze tictactoe Java project (Java 17 → Python 3.11)
    python demo_phase1.py

    # Custom source directory and languages
    python demo_phase1.py --source-dir examples/python_project --source-lang python --target-lang java

    # Use a YAML config file instead (overrides all other flags)
    python demo_phase1.py --config tictactoe_config.yaml
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from codemorph.analyzer.orchestrator import run_phase1_analysis
from codemorph.config.loader import create_config_from_args, load_config_from_yaml

console = Console()

# Sensible version defaults per language
DEFAULT_VERSIONS = {
    "java": "17",
    "python": "3.11",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CodeMorph Phase 1: Project Partitioning and Analysis",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (overrides all other options)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("examples/tictactoe/src"),
        help="Source directory to analyze (default: examples/tictactoe/src)",
    )
    parser.add_argument(
        "--source-lang",
        choices=["java", "python"],
        default="java",
        help="Source language (default: java)",
    )
    parser.add_argument(
        "--target-lang",
        choices=["java", "python"],
        default="python",
        help="Target language (default: python)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Test directory (auto-detected from <source-dir>/../tests if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./demo_output/<project>)",
    )
    return parser.parse_args()


def auto_detect_test_dir(source_dir: Path) -> Path | None:
    """Try common test directory locations relative to the source."""
    for name in ("tests", "test"):
        candidate = source_dir.parent / name
        if candidate.is_dir():
            return candidate
    return None


def build_config(args):
    """Build a CodeMorphConfig from CLI args or a YAML file."""
    if args.config:
        console.print(f"\n[cyan]Loading config from {args.config}...[/cyan]")
        return load_config_from_yaml(args.config)

    source_dir = args.source_dir.resolve()
    source_lang = args.source_lang
    target_lang = args.target_lang

    if source_lang == target_lang:
        console.print("[red]Error: source and target languages must differ.[/red]")
        raise SystemExit(1)

    source_version = DEFAULT_VERSIONS[source_lang]
    target_version = DEFAULT_VERSIONS[target_lang]

    test_dir = args.test_dir or auto_detect_test_dir(source_dir)
    project_name = source_dir.parent.name if source_dir.name == "src" else source_dir.name
    output_dir = args.output_dir or Path(f"./demo_output/{project_name}")

    console.print(f"\n[cyan]Configuring analysis...[/cyan]")

    # Java targets require build system + package name
    build_system = "gradle" if target_lang == "java" else None
    package_name = f"com.example.{project_name}" if target_lang == "java" else None

    return create_config_from_args(
        source_dir=source_dir,
        source_lang=source_lang,
        source_version=source_version,
        target_lang=target_lang,
        target_version=target_version,
        output_dir=output_dir,
        build_system=build_system,
        package_name=package_name,
        test_dir=test_dir,
        project_name=project_name,
    )


def main():
    args = parse_args()

    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]CodeMorph Phase 1 Demo[/bold cyan]\n\n"
            "This demonstrates complete project analysis:\n"
            "  Source file discovery\n"
            "  AST parsing and fragment extraction\n"
            "  Dependency graph building\n"
            "  Translation order determination",
            border_style="cyan",
        )
    )

    config = build_config(args)

    console.print(f"[green]\u2713[/green] Configuration created")
    console.print(f"  Project:  {config.project.name}")
    console.print(
        f"  Source:   {config.project.source.language.value} "
        f"{config.project.source.version} ({config.project.source.root})"
    )
    console.print(
        f"  Target:   {config.project.target.language.value} "
        f"{config.project.target.version}"
    )
    console.print(f"  Output:   {config.project.target.output_dir}")
    if config.project.source.test_root:
        console.print(f"  Tests:    {config.project.source.test_root}")

    # Run Phase 1 analysis
    try:
        result, state = run_phase1_analysis(config)

        # Display translation order
        if result.translation_order:
            console.print("\n[bold]Translation Order (First 15 Fragments)[/bold]\n")
            for i, fragment_id in enumerate(result.translation_order[:15], 1):
                fragment = result.fragments[fragment_id]
                console.print(
                    f"  {i:2d}. [cyan]{fragment.name}[/cyan] "
                    f"({fragment.fragment_type.value}) "
                    f"- {len(fragment.dependencies)} dependencies"
                )

            if len(result.translation_order) > 15:
                console.print(
                    f"\n  ... and {len(result.translation_order) - 15} more fragments"
                )

        # Show where state was saved
        console.print(f"\n[bold]State Saved[/bold]")
        console.print(f"  Session ID: {state.session_id}")
        console.print(f"  State file: {config.project.state_dir}/latest.json")

        if result.dependency_graph_path:
            console.print(f"  Graph:      {result.dependency_graph_path}")

        # What's next
        console.print("\n[bold cyan]What's Next?[/bold cyan]")
        console.print(
            "  Run demo_phase2.py with the same arguments to translate these fragments."
        )
        console.print(
            "  Phase 2 requires an LLM — by default it connects to Ollama at localhost:11434."
        )
        console.print(
            "  Or pass --config <file>.yaml to use OpenAI/OpenRouter instead.\n"
        )

    except Exception as e:
        console.print(f"\n[red]Error during analysis: {e}[/red]")
        console.print_exception()


if __name__ == "__main__":
    main()
