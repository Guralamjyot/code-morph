#!/usr/bin/env python3
"""
CodeMorph Demo Script

Demonstrates the current working components of CodeMorph.
This is NOT a full translation, but shows individual subsystems working.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codemorph.analyzer.graph_builder import DependencyGraphBuilder
from codemorph.config.models import FragmentType, LanguageType
from codemorph.languages.registry import LanguagePluginRegistry

console = Console()


def print_header(title: str):
    """Print a section header."""
    console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(80)}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")


def demo_python_parsing():
    """Demonstrate Python AST parsing and fragment extraction."""
    print_header("1. Python AST Parsing & Fragment Extraction")

    # Get the Python plugin
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    console.print("[cyan]Loaded Python 3.10 plugin[/cyan]")

    # Parse the example file
    example_file = Path("examples/python_project/calculator.py")
    console.print(f"[cyan]Parsing {example_file}...[/cyan]\n")

    ast = plugin.parse_file(example_file)
    fragments = plugin.extract_fragments(example_file, ast)

    # Display results in a table
    table = Table(title=f"Extracted {len(fragments)} Code Fragments")
    table.add_column("Name", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Lines", justify="center")
    table.add_column("Parent Class", style="blue")

    for fragment in fragments:
        table.add_row(
            fragment.name,
            fragment.fragment_type.value,
            f"{fragment.start_line}-{fragment.end_line}",
            fragment.parent_class or "-",
        )

    console.print(table)

    # Show a specific fragment
    add_func = next(f for f in fragments if f.name == "add")
    console.print("\n[bold]Example Fragment: 'add' function[/bold]")
    syntax = Syntax(add_func.source_code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Source Code"))

    sig = plugin.extract_signature(add_func)
    console.print(f"\n[cyan]Signature:[/cyan] {sig}")
    console.print(f"[cyan]Docstring:[/cyan] {add_func.docstring[:50]}...")

    return fragments


def demo_dependency_graph(fragments):
    """Demonstrate dependency graph building."""
    print_header("2. Dependency Graph Analysis")

    # Get plugin for dependency analysis
    plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")

    # Analyze dependencies (simplified)
    fragment_dict = {}
    for fragment in fragments:
        fragment_dict[fragment.id] = fragment

    console.print(f"[cyan]Analyzing dependencies for {len(fragments)} fragments...[/cyan]\n")

    # Build the graph
    graph_builder = DependencyGraphBuilder()
    graph = graph_builder.build_graph(fragment_dict)

    console.print(f"[green]✓[/green] Built dependency graph with {graph.number_of_nodes()} nodes")

    # Calculate complexity scores
    complexity_scores = graph_builder.calculate_complexity_scores(fragment_dict)

    # Show top 5 most complex fragments
    sorted_fragments = sorted(complexity_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    table = Table(title="Top 5 Most Complex Fragments")
    table.add_column("Fragment", style="cyan")
    table.add_column("Complexity Score", justify="center", style="yellow")

    for frag_id, score in sorted_fragments:
        # Extract just the name from the ID
        name = frag_id.split("::")[-1]
        table.add_row(name, str(score))

    console.print(table)

    # Check for circular dependencies
    cycles = graph_builder.detect_circular_dependencies()
    if cycles:
        console.print(f"\n[yellow]⚠[/yellow] Found {len(cycles)} circular dependencies")
    else:
        console.print("\n[green]✓[/green] No circular dependencies detected")

    return graph_builder


def demo_naming_conventions():
    """Demonstrate name conversion between Python and Java."""
    print_header("3. Naming Convention Conversion")

    python_plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")
    java_plugin = LanguagePluginRegistry.get_plugin(LanguageType.JAVA, "17")

    conversions = [
        ("calculate_average", FragmentType.FUNCTION, "Python → Java (camelCase)"),
        ("calculateAverage", FragmentType.METHOD, "Java → Python (snake_case)"),
        ("MyClass", FragmentType.CLASS, "Python → Java (PascalCase)"),
        ("MAX_VALUE", FragmentType.CONSTANT, "Python → Java (SCREAMING_SNAKE)"),
    ]

    table = Table(title="Name Convention Conversions")
    table.add_column("Original", style="cyan")
    table.add_column("Direction", style="yellow")
    table.add_column("Converted", style="green")

    for original, frag_type, direction in conversions:
        if "Python → Java" in direction:
            target_convention = python_plugin.get_naming_convention(frag_type)
            if "camel" in direction.lower():
                target_convention = "camelCase"
            converted = python_plugin.convert_name(original, frag_type, target_convention)
        else:
            converted = java_plugin.convert_name(original, frag_type, "snake_case")

        table.add_row(original, direction, converted)

    console.print(table)


def demo_llm_client():
    """Demonstrate LLM client (if Ollama is available)."""
    print_header("4. LLM Integration (Ollama)")

    try:
        from codemorph.config.models import LLMConfig
        from codemorph.translator.llm_client import OllamaClient

        config = LLMConfig()
        client = OllamaClient(config)

        # Check if model is available
        available_models = client.get_available_models()
        console.print(f"[cyan]Connected to Ollama at {config.host}[/cyan]")
        console.print(f"[green]✓[/green] Found {len(available_models)} models\n")

        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan")

        for model in available_models[:5]:  # Show first 5
            marker = " [default]" if model == config.model else ""
            table.add_row(f"{model}{marker}")

        console.print(table)

        if config.model in available_models:
            console.print(f"\n[green]✓[/green] Default model '{config.model}' is available")
        else:
            console.print(
                f"\n[yellow]⚠[/yellow] Default model '{config.model}' not found. "
                f"Run: ollama pull {config.model}"
            )

    except ConnectionError as e:
        console.print(f"[yellow]⚠[/yellow] Could not connect to Ollama: {e}")
        console.print("[cyan]Make sure Ollama is running: ollama serve[/cyan]")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")


def main():
    """Run the demo."""
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]CodeMorph Demo[/bold cyan]\n\n"
            "This demonstrates the current working components.\n"
            "Full translation pipeline is still under development.",
            border_style="cyan",
        )
    )

    try:
        # Run all demos
        fragments = demo_python_parsing()
        demo_dependency_graph(fragments)
        demo_naming_conventions()
        demo_llm_client()

        # Summary
        print_header("Summary")
        console.print("[green]✓[/green] Python AST parsing and fragment extraction")
        console.print("[green]✓[/green] Dependency graph analysis")
        console.print("[green]✓[/green] Naming convention conversion")
        console.print("[green]✓[/green] LLM client integration")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Implement state persistence")
        console.print("  2. Create feature mapping rules")
        console.print("  3. Build type compatibility checker")
        console.print("  4. Implement execution snapshot system")
        console.print("  5. Create main orchestrator\n")

    except Exception as e:
        console.print(f"\n[red]Error running demo: {e}[/red]")
        console.print_exception()


if __name__ == "__main__":
    main()
