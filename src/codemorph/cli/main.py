"""
CodeMorph CLI - Main entry point.

Provides commands for translating code between languages and upgrading language versions.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codemorph.config.loader import (
    ConfigurationError,
    create_config_from_args,
    generate_default_config,
    load_config_from_yaml,
)
from codemorph.config.models import CheckpointMode, LanguageType

app = typer.Typer(
    name="codemorph",
    help="Bidirectional code translation and version upgrades for Python and Java",
    no_args_is_help=True,
)

console = Console()


# =============================================================================
# Helper Functions
# =============================================================================


def print_banner():
    """Print CodeMorph banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗ ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ██████╗██╗  ██╗ ║
    ║  ██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║██╔═══██╗██╔══██╗██║  ██║ ║
    ║  ██║     ██║   ██║██║  ██║█████╗  ██╔████╔██║██║   ██║██████╔╝███████║ ║
    ║  ██║     ██║   ██║██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║ ║
    ║  ╚██████╗╚██████╔╝██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║ ║
    ║   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ║
    ║                                                               ║
    ║              Code Translation & Version Upgrades              ║
    ║                        Version 1.0.0                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def validate_path(path: str, must_exist: bool = True) -> Path:
    """Validate and return a Path object."""
    p = Path(path)
    if must_exist and not p.exists():
        raise typer.BadParameter(f"Path does not exist: {path}")
    return p


# =============================================================================
# Commands
# =============================================================================


@app.command()
def translate(
    source: str = typer.Argument(..., help="Source directory containing code to translate"),
    target_lang: Optional[str] = typer.Option(None, "--target-lang", "-t", help="Target language (python/java)"),
    target_version: Optional[str] = typer.Option(None, "--target-version", "-tv", help="Target language version"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    source_lang: str = typer.Option("python", "--source-lang", "-s", help="Source language (python/java)"),
    source_version: str = typer.Option("3.10", "--source-version", "-sv", help="Source language version"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration YAML file"),
    test_dir: Optional[str] = typer.Option(None, "--test-dir", help="Test suite directory"),
    build_system: Optional[str] = typer.Option(None, "--build-system", help="Build system for Java (maven/gradle)"),
    package_name: Optional[str] = typer.Option(None, "--package-name", help="Base package name for Java"),
    checkpoint_mode: str = typer.Option("batch", "--checkpoint-mode", help="Checkpoint mode (interactive/batch/auto)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Translate code from one language to another.

    Examples:
        codemorph translate ./my_python_app -t java -tv 17 --build-system gradle
        codemorph translate ./my_java_app -s java -sv 11 -t python -tv 3.11
        codemorph translate ./legacy_py -s python -sv 2.7 -t python -tv 3.10
    """
    print_banner()

    # Configure logging
    import logging
    import os
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(name)s: %(message)s")

    # Ensure JAVA_HOME is set if not already
    if not os.environ.get("JAVA_HOME"):
        java_path = Path("/workspace/persistent/java/jdk-17.0.2")
        if java_path.exists():
            os.environ["JAVA_HOME"] = str(java_path)
            os.environ["PATH"] = f"{java_path / 'bin'}:{os.environ.get('PATH', '')}"

    try:
        # Load or create configuration
        if config:
            console.print(f"[cyan]Loading configuration from {config}...[/cyan]")
            cfg = load_config_from_yaml(Path(config))
        else:
            if not target_lang or not target_version:
                console.print("[bold red]Error:[/bold red] --target-lang and --target-version required when not using --config")
                raise typer.Exit(1)
            console.print("[cyan]Creating configuration from arguments...[/cyan]")
            cfg = create_config_from_args(
                source_dir=validate_path(source),
                source_lang=source_lang,
                source_version=source_version,
                target_lang=target_lang,
                target_version=target_version,
                output_dir=Path(output),
                build_system=build_system,
                package_name=package_name,
                test_dir=Path(test_dir) if test_dir else None,
                checkpoint_mode=CheckpointMode(checkpoint_mode),
            )

        # Display configuration
        display_config(cfg, verbose)

        # Confirm before proceeding (skip in auto mode)
        if cfg.checkpoint_mode != CheckpointMode.AUTO:
            if not typer.confirm("\nProceed with translation?"):
                console.print("[yellow]Translation cancelled.[/yellow]")
                raise typer.Abort()

        # Run Phase 1: Analysis
        console.print("\n[bold cyan]Phase 1: Project Analysis[/bold cyan]")
        from codemorph.analyzer.orchestrator import run_phase1_analysis

        analysis_result, state = run_phase1_analysis(cfg)
        console.print(f"[green]✓[/green] Phase 1 complete")

        # Checkpoint: Review analysis
        if cfg.checkpoint_mode == CheckpointMode.INTERACTIVE:
            console.print(f"\n[bold]Analysis Summary:[/bold]")
            console.print(f"  Fragments: {len(analysis_result.fragments)}")
            console.print(f"  Translation order: {len(analysis_result.translation_order)}")
            if analysis_result.circular_dependencies:
                console.print(
                    f"  [yellow]Circular dependencies: {len(analysis_result.circular_dependencies)}[/yellow]"
                )

            if not typer.confirm("\nProceed with translation?"):
                console.print("[yellow]Translation stopped after Phase 1.[/yellow]")
                console.print(f"[cyan]State saved to: {state.state_file}[/cyan]")
                raise typer.Abort()

        # Library Mapping Checkpoint (after Phase 1)
        console.print("\n[bold cyan]Library Mapping Analysis[/bold cyan]")
        from codemorph.knowledge.library_mapper import LibraryMappingService
        from codemorph.translator.llm_client import create_llm_client
        from codemorph.languages.registry import get_plugin

        llm_client = create_llm_client(cfg.llm)
        library_service = LibraryMappingService(cfg, llm_client)

        # Extract imports from all source files
        source_plugin = get_plugin(cfg.project.source.language)
        all_imports = []
        for fragment in state.analysis_result.fragments.values():
            if fragment.source_file:
                try:
                    tree = source_plugin.parse_file(fragment.source_file)
                    imports = source_plugin.extract_imports(tree, fragment.source_file)
                    all_imports.extend(imports)
                except Exception:
                    pass  # Skip files that can't be parsed for imports

        # Analyze imports and get LLM suggestions for unknown libraries
        if all_imports:
            library_result = library_service.analyze_imports(all_imports)

            if library_result.unknown_imports:
                console.print(f"  Found {len(library_result.unknown_imports)} unknown libraries")

                # Query LLM for suggestions
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "Getting LLM suggestions for library mappings...",
                        total=len(library_result.unknown_imports),
                    )
                    for imp in library_result.unknown_imports:
                        progress.update(task, description=f"Analyzing: {imp.module}")
                        library_service.suggest_mapping(imp.module)
                        progress.advance(task)

            # Interactive checkpoint for library verification
            if cfg.checkpoint_mode == CheckpointMode.INTERACTIVE:
                from codemorph.cli.checkpoint_ui import CheckpointUI

                checkpoint_ui = CheckpointUI(console)
                checkpoint_ui.show_library_mapping_checkpoint(library_service)

            # Display library mapping summary
            pending = library_service.get_pending_for_user_review()
            if pending:
                console.print(
                    f"\n[yellow]⚠ {len(pending)} library mappings are unverified. "
                    f"Translation will proceed but may have issues.[/yellow]"
                )
        else:
            console.print("  [dim]No external library imports detected[/dim]")

        console.print(f"[green]✓[/green] Library mapping analysis complete")

        # Run Phase 2: Type-Driven Translation
        console.print("\n[bold cyan]Phase 2: Type-Driven Translation[/bold cyan]")
        from codemorph.translator.orchestrator import run_phase2_translation

        translated_fragments, symbol_registry = run_phase2_translation(cfg, state)
        console.print(f"[green]✓[/green] Phase 2 complete")

        # Checkpoint: Review type-driven translation
        if cfg.checkpoint_mode == CheckpointMode.INTERACTIVE:
            from collections import Counter
            from codemorph.config.models import TranslationStatus

            status_counts = Counter(f.status for f in translated_fragments.values())

            console.print(f"\n[bold]Phase 2 Summary:[/bold]")
            for status, count in status_counts.most_common():
                console.print(f"  {status.value}: {count}")

            if not typer.confirm("\nProceed with Test Translation?"):
                console.print("[yellow]Translation stopped after Phase 2.[/yellow]")
                console.print(f"[cyan]State saved to: {state.state_file}[/cyan]")
                raise typer.Abort()

        # Test Translation (after Phase 2, before Phase 3)
        test_results = []
        test_dir_path = Path(test_dir) if test_dir else cfg.project.source.test_root
        if test_dir_path and test_dir_path.exists() and cfg.verification.generate_tests:
            console.print("\n[bold cyan]Test Translation[/bold cyan]")
            from codemorph.translator.test_orchestrator import run_test_translation

            test_results = run_test_translation(
                config=cfg,
                symbol_registry=symbol_registry,
                llm_client=llm_client,
                test_dir=test_dir_path,
                translated_fragments=translated_fragments,
                output_dir=cfg.project.target.output_dir,
            )

            # Display test translation summary
            success_count = sum(1 for r in test_results if r.compilation_success)
            total_tests = sum(r.test_count for r in test_results if r.compilation_success)
            console.print(
                f"[green]✓[/green] Test translation complete: "
                f"{success_count}/{len(test_results)} files ({total_tests} test methods)"
            )

            if cfg.checkpoint_mode == CheckpointMode.INTERACTIVE:
                if not typer.confirm("\nProceed with Phase 3 (I/O Equivalence)?"):
                    console.print("[yellow]Translation stopped after Test Translation.[/yellow]")
                    console.print(f"[cyan]State saved to: {state.state_file}[/cyan]")
                    raise typer.Abort()
        elif test_dir_path and test_dir_path.exists():
            console.print("\n[dim]Test translation skipped (generate_tests=false)[/dim]")
        else:
            console.print("\n[dim]No test directory found, skipping test translation[/dim]")

        # Run Phase 3: Semantics-Driven Translation
        console.print("\n[bold cyan]Phase 3: Semantics-Driven Translation[/bold cyan]")
        from codemorph.verifier.orchestrator import run_phase3_semantics

        equivalence_reports = run_phase3_semantics(cfg, state, translated_fragments)
        console.print(f"[green]✓[/green] Phase 3 complete")

        # Display final summary
        from collections import Counter
        from codemorph.config.models import TranslationStatus

        status_counts = Counter(f.status for f in translated_fragments.values())

        console.print("\n[bold]Translation Summary:[/bold]")
        for status, count in status_counts.most_common():
            color = "green"
            if status == TranslationStatus.FAILED:
                color = "red"
            elif status == TranslationStatus.MOCKED:
                color = "yellow"
            console.print(f"  [{color}]{status.value}:[/{color}] {count}")

        console.print(f"\n[green]✓[/green] Translation complete!")
        console.print(f"  State saved to: {state.state_dir}")
        console.print(f"  Output directory: {cfg.project.target.output_dir}")
        console.print(f"  Symbol registry: {len(symbol_registry)} symbols mapped")
        if test_results:
            test_success = sum(1 for r in test_results if r.compilation_success)
            console.print(f"  Test files translated: {test_success}/{len(test_results)}")

        # Show warnings if needed
        if status_counts.get(TranslationStatus.MOCKED, 0) > 0:
            console.print(
                f"\n[yellow]⚠ {status_counts[TranslationStatus.MOCKED]} fragment(s) were mocked "
                f"and require manual attention[/yellow]"
            )

        if status_counts.get(TranslationStatus.FAILED, 0) > 0:
            console.print(
                f"\n[red]✗ {status_counts[TranslationStatus.FAILED]} fragment(s) failed translation[/red]"
            )

    except ConfigurationError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def analyze(
    source: str = typer.Argument(..., help="Source directory to analyze"),
    output: str = typer.Option("./analysis.json", "--output", "-o", help="Output file for analysis results"),
    source_lang: str = typer.Option("python", "--source-lang", "-s", help="Source language"),
    source_version: str = typer.Option("3.10", "--source-version", "-sv", help="Source version"),
    target_lang: str = typer.Option("java", "--target-lang", "-t", help="Target language (for planning)"),
    target_version: str = typer.Option("17", "--target-version", "-tv", help="Target version"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Analyze a codebase and generate a translation plan (Phase 1 only).

    This command performs project partitioning without translating.
    """
    print_banner()

    try:
        from codemorph.analyzer.orchestrator import run_phase1_analysis
        import orjson

        # Create minimal config
        config = create_config_from_args(
            source_dir=validate_path(source),
            source_lang=source_lang,
            source_version=source_version,
            target_lang=target_lang,
            target_version=target_version,
            output_dir=Path(output).parent / "output",
            build_system="gradle" if target_lang.lower() == "java" else None,
        )

        # Run Phase 1
        result, state = run_phase1_analysis(config)

        # Save analysis results to JSON
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))

        console.print(f"\n[green]✓[/green] Analysis saved to {output_path}")

    except ConfigurationError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def verify(
    target_file: str = typer.Argument(..., help="Translated file to verify"),
    source_file: str = typer.Argument(..., help="Original source file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Verify a specific translated file against its source.

    Runs I/O equivalence checks using test snapshots.
    """
    print_banner()
    console.print("[cyan]Verifying translation...[/cyan]")

    # TODO: Implement verifier
    console.print("[bold red]Verification engine not yet implemented![/bold red]")


@app.command()
def init(
    output: str = typer.Option("./codemorph.yaml", "--output", "-o", help="Output path for config file"),
):
    """
    Generate a default configuration file.

    Creates a codemorph.yaml with sensible defaults that you can customize.
    """
    output_path = Path(output)

    if output_path.exists():
        if not typer.confirm(f"{output} already exists. Overwrite?"):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Abort()

    generate_default_config(output_path)
    console.print(f"[green]✓[/green] Generated configuration file: {output}")
    console.print("\nEdit this file to customize your translation settings.")


@app.command()
def doctor():
    """
    Check system dependencies and configuration.

    Verifies that all required tools are available (Java, Python, Ollama, etc.).
    """
    print_banner()
    console.print("[cyan]Checking system dependencies...[/cyan]\n")

    # TODO: Implement comprehensive checks
    checks = [
        ("Python 3.10+", "✓", "green"),
        ("Java JDK 11+", "?", "yellow"),
        ("Ollama", "?", "yellow"),
        ("Tree-sitter", "?", "yellow"),
    ]

    table = Table(title="Dependency Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")

    for component, status, color in checks:
        table.add_row(component, f"[{color}]{status}[/{color}]")

    console.print(table)
    console.print("\n[yellow]Full dependency check not yet implemented.[/yellow]")


@app.command()
def assemble(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration YAML file"),
    state_dir: str = typer.Option(".codemorph", "--state-dir", help="Directory containing latest.json and symbol_registry.json"),
    java_source: Optional[str] = typer.Option(None, "--java-source", "-j", help="Java source directory"),
    output: str = typer.Option("./assembled", "--output", "-o", help="Output directory for assembled project"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
    fill_mocks: bool = typer.Option(True, "--fill-mocks/--no-fill-mocks", help="Fill mocked fragments with LLM"),
    max_fix_iterations: int = typer.Option(5, "--max-fix-iterations", help="Max compile-fix iterations"),
    use_opencode: bool = typer.Option(False, "--use-opencode/--no-opencode", help="Use OpenCode external agent for compile-fix instead of internal LLM"),
    interactive: bool = typer.Option(False, "--interactive/--no-interactive", help="Interactive OpenCode mode: display errors and prompt Retry/Edit/Skip before each iteration"),
):
    """
    Phase 4: Assemble translated fragments into a working Python project.

    Reads the translation state (latest.json) and symbol registry from the
    state directory, then uses an LLM-driven agent to structure, draft,
    inject, and fix the assembled project.

    Examples:
        codemorph assemble -c config.yaml --state-dir .codemorph -o ./assembled
        codemorph assemble --state-dir .codemorph -j ./java_src --no-fill-mocks
    """
    print_banner()

    import logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(name)s: %(message)s")

    try:
        # Load configuration
        if config:
            console.print(f"[cyan]Loading configuration from {config}...[/cyan]")
            cfg = load_config_from_yaml(Path(config))
        else:
            # Try to load from state dir
            state_path = Path(state_dir)
            latest_file = state_path / "latest.json"
            if latest_file.exists():
                console.print(f"[cyan]Loading config from {latest_file}...[/cyan]")
                from codemorph.state.persistence import TranslationState
                state = TranslationState.load(latest_file)
                cfg = state.config
            else:
                console.print("[bold red]Error:[/bold red] No config file or state directory found. "
                              "Use --config or ensure --state-dir has latest.json")
                raise typer.Exit(1)

        # Resolve source directory
        source_dir = Path(java_source) if java_source else Path(cfg.project.source.root)

        console.print(f"[cyan]Source: {source_dir}[/cyan]")
        console.print(f"[cyan]State: {state_dir}[/cyan]")
        console.print(f"[cyan]Output: {output}[/cyan]")

        # Run Phase 4
        from codemorph.assembler.orchestrator import Phase4Orchestrator

        orchestrator = Phase4Orchestrator(
            config=cfg,
            state_dir=Path(state_dir),
            source_dir=source_dir,
            output_dir=Path(output),
            verbose=verbose,
            fill_mocks=fill_mocks,
            max_fix_iterations=max_fix_iterations,
            use_opencode=use_opencode,
            interactive=interactive,
        )

        results = orchestrator.run()

        # Final status
        if results.get("compile_fix", {}).get("final_status") == "clean":
            console.print("\n[bold green]Assembly complete! All files compile cleanly.[/bold green]")
        else:
            console.print("\n[bold yellow]Assembly complete with some issues.[/bold yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def resume(
    project_dir: str = typer.Argument(..., help="Project directory with .codemorph state"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Resume a translation from the last checkpoint.

    Loads the saved state and continues from where it left off.
    """
    print_banner()
    console.print(f"[cyan]Resuming translation from {project_dir}...[/cyan]")

    # TODO: Implement resume logic
    console.print("[bold red]Resume functionality not yet implemented![/bold red]")


# =============================================================================
# Helper Display Functions
# =============================================================================


def display_config(cfg, verbose: bool = False):
    """Display the loaded configuration in a nice format."""
    console.print("\n")

    # Create translation info panel
    translation_type = cfg.get_translation_type()
    is_upgrade = cfg.is_version_upgrade()

    info_text = f"""
[bold cyan]Translation Type:[/bold cyan] {translation_type}
[bold cyan]Source:[/bold cyan] {cfg.project.source.root}
[bold cyan]Target:[/bold cyan] {cfg.project.target.output_dir}
[bold cyan]Checkpoint Mode:[/bold cyan] {cfg.checkpoint_mode.value}
    """

    if is_upgrade:
        panel_title = "Version Upgrade"
        panel_style = "bold yellow"
    else:
        panel_title = "Cross-Language Translation"
        panel_style = "bold green"

    console.print(Panel(info_text.strip(), title=panel_title, border_style=panel_style))

    if verbose:
        # Show detailed configuration
        console.print("\n[bold]Configuration Details:[/bold]")
        console.print(f"  LLM Model: {cfg.llm.model}")
        console.print(f"  Max Type Check Retries: {cfg.translation.max_retries_type_check}")
        console.print(f"  Max Semantics Retries: {cfg.translation.max_retries_semantics}")
        console.print(f"  Allow Mocking: {cfg.translation.allow_mocking}")
        console.print(f"  RAG Enabled: {cfg.rag.enabled}")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
