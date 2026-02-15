#!/usr/bin/env python3
"""
Phase 2 Demo: Type-Driven Translation (Phase 1 + Phase 2)

Runs Phase 1 analysis then Phase 2 translation end-to-end:
  - Phase 1: file discovery, AST parsing, fragment extraction, dependency graph
  - Phase 2: feature mapping, LLM translation, compilation checks, type verification,
             retry logic with error refinement, overload merging

Prerequisites:
  - An LLM backend must be available. By default CodeMorph connects to Ollama
    at http://localhost:11434 (model: deepseek-coder:6.7b).
  - To use OpenAI or OpenRouter instead, pass --config with a YAML config file
    that specifies the provider and API key (see tictactoe_config.yaml for an example).

Usage:
    # Default: translate tictactoe (Java 17 → Python 3.11) via Ollama
    python demo_phase2.py

    # Custom source directory and languages
    python demo_phase2.py --source-dir examples/python_project --source-lang python --target-lang java

    # Use a YAML config file (includes LLM provider/key settings)
    python demo_phase2.py --config tictactoe_config.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from codemorph.analyzer.orchestrator import run_phase1_analysis
from codemorph.config.loader import create_config_from_args, load_config_from_yaml
from codemorph.config.models import TranslationStatus
from codemorph.translator.orchestrator import run_phase2_translation

console = Console()

# Sensible version defaults per language
DEFAULT_VERSIONS = {
    "java": "17",
    "python": "3.11",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CodeMorph Phase 1 + 2 Demo: Analysis and Type-Driven Translation",
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
        help="Source directory to analyze and translate (default: examples/tictactoe/src)",
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

    console.print(f"\n[cyan]Configuring translation...[/cyan]")

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
            "[bold cyan]CodeMorph Phase 1 + 2 Demo[/bold cyan]\n\n"
            "This demonstrates:\n"
            "  Phase 1: Project analysis and partitioning\n"
            "  Phase 2: Type-driven LLM translation\n"
            "  Compilation verification\n"
            "  Type compatibility checking\n"
            "  Feature mapping validation\n"
            "  Retry logic with error refinement\n"
            "  Overload detection and merging",
            border_style="cyan",
        )
    )

    config = build_config(args)

    console.print(f"[green]\u2713[/green] Configuration created")
    console.print(f"  Project:    {config.project.name}")
    console.print(
        f"  Source:     {config.project.source.language.value} "
        f"{config.project.source.version} ({config.project.source.root})"
    )
    console.print(
        f"  Target:     {config.project.target.language.value} "
        f"{config.project.target.version}"
    )
    console.print(f"  Output:     {config.project.target.output_dir}")
    console.print(f"  LLM:        {config.llm.provider.value} / {config.llm.model}")
    console.print(f"  Max retries: {config.translation.max_retries_type_check}")
    if config.project.source.test_root:
        console.print(f"  Tests:      {config.project.source.test_root}")

    try:
        # ── Phase 1: Analysis ──────────────────────────────────────────
        console.print("\n[bold cyan]Starting Phase 1: Analysis[/bold cyan]")
        result, state = run_phase1_analysis(config)

        console.print(f"\n[green]\u2713[/green] Phase 1 complete")
        console.print(f"  Fragments:         {len(result.fragments)}")
        console.print(f"  Translation order: {len(result.translation_order)}")

        # ── Phase 2: Translation ───────────────────────────────────────
        console.print("\n[bold cyan]Starting Phase 2: Type-Driven Translation[/bold cyan]")
        translated_fragments, symbol_registry = run_phase2_translation(config, state)

        # Translation summary
        console.print(f"\n[bold]Translation Complete![/bold]")
        console.print(f"  Total fragments: {len(translated_fragments)}")

        status_counts = Counter(f.status for f in translated_fragments.values())

        console.print(f"\n[bold]Status Breakdown:[/bold]")
        for status, count in status_counts.most_common():
            color = {
                TranslationStatus.TYPE_VERIFIED: "green",
                TranslationStatus.COMPILED: "green",
                TranslationStatus.FAILED: "red",
                TranslationStatus.MOCKED: "yellow",
            }.get(status, "white")
            console.print(f"  [{color}]{status.value}:[/{color}] {count}")

        # Sample translations
        console.print(f"\n[bold]Sample Translations (first 3):[/bold]")
        shown = 0
        for frag_id, translated in translated_fragments.items():
            if shown >= 3:
                break
            if translated.target_code:
                console.print(f"\n  [cyan]{translated.fragment.name}[/cyan]")
                console.print(f"    Status:  {translated.status.value}")
                console.print(f"    Retries: {translated.retry_count}")
                if translated.is_mocked:
                    console.print(f"    [yellow]Mocked: {translated.mock_reason}[/yellow]")
                shown += 1

        # State saved
        console.print(f"\n[bold]State Saved[/bold]")
        console.print(f"  Session ID:      {state.session_id}")
        console.print(f"  State file:      {config.project.state_dir}/latest.json")
        console.print(f"  Symbol registry: {config.project.state_dir}/symbol_registry.json")
        console.print(f"  Current phase:   {state.current_phase}")

        # What's next
        console.print("\n[bold cyan]What's Next?[/bold cyan]")
        console.print(
            "  Phase 3: Semantics-driven refinement — runs I/O equivalence tests"
        )
        console.print(
            "  Phase 4: Assembly — combines fragments into a runnable Python/Java project\n"
        )

    except Exception as e:
        console.print(f"\n[red]Error during translation: {e}[/red]")
        console.print_exception()


if __name__ == "__main__":
    main()
