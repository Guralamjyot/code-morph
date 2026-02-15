"""
Phase 3 Demo: Semantics-Driven Translation

Demonstrates the complete Phase 3 workflow:
1. Capture execution snapshots from Python tests
2. Execute translated Java code with same inputs
3. Verify I/O equivalence
4. Refine translations that fail
5. Generate mocks for untranslatable functions

Run with:
    python demo_phase3.py
"""

import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codemorph.verifier.snapshot_capture import (
    SnapshotCollector,
    log_snapshot,
    set_global_collector,
)
from codemorph.verifier.equivalence_checker import (
    EquivalenceChecker,
    EquivalenceStatus,
)
from codemorph.verifier.mocker import (
    FunctionMocker,
    MockStrategy,
    MockRegistry,
)

console = Console()


def print_header(title: str):
    """Print a section header."""
    console.print(Panel(title, style="bold cyan"))


def demo_snapshot_capture():
    """Demonstrate snapshot capture."""
    print_header("Step 1: Capture Execution Snapshots")

    # Create temporary directory for snapshots
    tmpdir = Path(tempfile.mkdtemp())
    collector = SnapshotCollector(tmpdir)
    set_global_collector(collector)

    console.print("\n[cyan]Instrumenting Python functions...[/cyan]")

    # Define test functions with decorator
    @log_snapshot
    def add(a, b):
        """Add two numbers."""
        return a + b

    @log_snapshot
    def calculate_total(items, tax_rate=0.05):
        """Calculate total with tax."""
        total = sum(item['price'] for item in items)
        return total * (1 + tax_rate)

    @log_snapshot
    def divide(a, b):
        """Divide with error handling."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    console.print("[green]✓[/green] Functions instrumented")

    # Execute functions to generate snapshots
    console.print("\n[cyan]Running test cases...[/cyan]")

    test_cases = [
        ("add", lambda: add(2, 3)),
        ("add", lambda: add(10, 20)),
        ("calculate_total", lambda: calculate_total([{'price': 100}], 0.05)),
        ("calculate_total", lambda: calculate_total([{'price': 50}, {'price': 30}], 0.10)),
        ("divide", lambda: divide(10, 2)),
        ("divide", lambda: divide(100, 5)),
    ]

    # Test exception case
    try:
        divide(10, 0)
    except ValueError:
        pass  # Expected

    for func_name, test_func in test_cases:
        result = test_func()
        console.print(f"  [green]✓[/green] {func_name}: {result}")

    # Display summary
    summary = collector.get_summary()

    table = Table(title="Captured Snapshots", show_header=True)
    table.add_column("Function", style="cyan")
    table.add_column("Snapshots", justify="right", style="green")

    for func_id, count in summary.items():
        simple_name = func_id.split("::")[-1]
        table.add_row(simple_name, str(count))

    console.print("\n")
    console.print(table)

    return tmpdir, collector


def demo_equivalence_checking(snapshot_dir: Path):
    """Demonstrate I/O equivalence checking."""
    print_header("Step 2: I/O Equivalence Verification")

    console.print("\n[cyan]Creating Java executors (simulated)...[/cyan]")

    # Simulate Java implementations
    def java_add(*args, **kwargs):
        """Java implementation of add."""
        return args[0] + args[1]

    def java_calculate_total(*args, **kwargs):
        """Java implementation of calculate_total."""
        items = args[0]
        tax_rate = kwargs.get('tax_rate', 0.05) if kwargs else (args[1] if len(args) > 1 else 0.05)
        total = sum(item['price'] for item in items)
        return total * (1 + tax_rate)

    def java_divide(*args, **kwargs):
        """Java implementation of divide."""
        if args[1] == 0:
            raise ValueError("Cannot divide by zero")
        return args[0] / args[1]

    # Buggy implementation for demonstration
    def buggy_add(*args, **kwargs):
        """Buggy Java implementation (multiplies instead of adds)."""
        return args[0] * args[1]

    console.print("[green]✓[/green] Executors created")

    # Create equivalence checker
    checker = EquivalenceChecker(snapshot_dir)

    # Check each function
    console.print("\n[cyan]Running equivalence checks...[/cyan]")

    results = {}

    # Check correct implementations
    for func_id, executor in [
        ("__main__::add", java_add),
        ("__main__::calculate_total", java_calculate_total),
        ("__main__::divide", java_divide),
    ]:
        report = checker.check_function(func_id, executor)
        results[func_id] = report

        func_name = func_id.split("::")[-1]
        if report.is_equivalent:
            console.print(
                f"  [green]✓[/green] {func_name}: "
                f"{report.passed}/{report.total_snapshots} passed"
            )
        else:
            console.print(
                f"  [red]✗[/red] {func_name}: "
                f"{report.passed}/{report.total_snapshots} passed, "
                f"{report.failed} failed"
            )

    # Check buggy implementation
    console.print("\n[cyan]Testing buggy implementation...[/cyan]")
    buggy_report = checker.check_function("__main__::add", buggy_add)

    if not buggy_report.is_equivalent:
        console.print(
            f"  [red]✗[/red] buggy_add: Detected mismatch "
            f"({buggy_report.failed}/{buggy_report.total_snapshots} failed)"
        )

        # Show first failure
        for result in buggy_report.results:
            if result.status == EquivalenceStatus.FAILED:
                console.print(f"\n  [yellow]Example failure:[/yellow]")
                console.print(f"    Input: {result.inputs}")
                console.print(f"    Expected: {result.expected_output}")
                console.print(f"    Actual: {result.actual_output}")
                console.print(f"    Diff: {result.diff}")
                break

    # Summary table
    table = Table(title="Equivalence Results", show_header=True)
    table.add_column("Function", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Status", justify="center")

    for func_id, report in results.items():
        func_name = func_id.split("::")[-1]
        status = "✓ PASS" if report.is_equivalent else "✗ FAIL"
        status_style = "green" if report.is_equivalent else "red"

        table.add_row(
            func_name,
            str(report.total_snapshots),
            str(report.passed),
            str(report.failed),
            f"[{status_style}]{status}[/{status_style}]"
        )

    console.print("\n")
    console.print(table)

    return results


def demo_function_mocking():
    """Demonstrate function mocking."""
    print_header("Step 3: Function Mocking")

    console.print("\n[cyan]Creating mock registry...[/cyan]")

    registry = MockRegistry()
    mocker = FunctionMocker("python", "java")

    console.print("[green]✓[/green] Registry created")

    console.print("\n[cyan]Generating mocks for failed translations...[/cyan]")

    # Generate different types of mocks
    mocks_to_create = [
        ("complex_algorithm", "public int complexAlgorithm(Data d)", MockStrategy.STUB, "C-extension dependency"),
        ("legacy_function", "public void legacyFunction()", MockStrategy.MANUAL, "Uses deprecated APIs"),
        ("platform_specific", "public String platformSpecific()", MockStrategy.BRIDGE, "Platform-specific code"),
    ]

    for func_name, signature, strategy, reason in mocks_to_create:
        mock = mocker.generate_mock(
            function_id=f"module::{func_name}",
            function_name=func_name,
            signature=signature,
            strategy=strategy,
            reason=reason
        )

        registry.register_mock(mock)

        console.print(f"  [yellow]⚠[/yellow] {func_name}: {strategy.value} mock ({reason})")

    # Display mock report
    console.print("\n")
    console.print(Panel(registry.generate_report(), title="Mock Report", border_style="yellow"))

    # Show example generated code
    console.print("\n[cyan]Example generated mock code:[/cyan]")

    stub_mock = registry.get_mocks_by_strategy(MockStrategy.STUB)[0]

    console.print(Panel(
        stub_mock.generated_code,
        title=f"{stub_mock.function_name} (STUB)",
        border_style="dim"
    ))

    return registry


def demo_phase3_statistics():
    """Display Phase 3 statistics."""
    print_header("Step 4: Phase 3 Statistics")

    # Simulated statistics
    stats = {
        "Total Functions": 8,
        "Functions Tested": 8,
        "Functions Passed": 5,
        "Functions Failed": 1,
        "Functions Mocked": 2,
        "Total Snapshots": 15,
        "Snapshots Passed": 13,
        "Snapshots Failed": 2,
        "Refinement Attempts": 3,
        "Successful Refinements": 1,
    }

    table = Table(title="Phase 3 Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for metric, value in stats.items():
        if "Failed" in metric or "Mocked" in metric:
            style = "yellow" if value > 0 else "green"
            table.add_row(metric, f"[{style}]{value}[/{style}]")
        else:
            table.add_row(metric, str(value))

    console.print("\n")
    console.print(table)

    # Success rate
    success_rate = (stats["Functions Passed"] / stats["Total Functions"]) * 100
    snapshot_rate = (stats["Snapshots Passed"] / stats["Total Snapshots"]) * 100

    console.print(f"\n[bold]Overall Success Rate:[/bold]")
    console.print(f"  Functions: [green]{success_rate:.1f}%[/green]")
    console.print(f"  Snapshots: [green]{snapshot_rate:.1f}%[/green]")


def main():
    """Run the complete Phase 3 demo."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]CodeMorph Phase 3 Demo[/bold cyan]\n"
        "Semantics-Driven Translation with I/O Equivalence",
        border_style="cyan"
    ))

    try:
        # Step 1: Capture snapshots
        snapshot_dir, collector = demo_snapshot_capture()

        # Step 2: Check equivalence
        input("\n[dim]Press Enter to continue to equivalence checking...[/dim]")
        equivalence_results = demo_equivalence_checking(snapshot_dir)

        # Step 3: Generate mocks
        input("\n[dim]Press Enter to continue to function mocking...[/dim]")
        mock_registry = demo_function_mocking()

        # Step 4: Show statistics
        input("\n[dim]Press Enter to see final statistics...[/dim]")
        demo_phase3_statistics()

        # Conclusion
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]✓ Phase 3 Demo Complete![/bold green]\n\n"
            "Phase 3 ensures translated code behaves exactly like the source:\n"
            "  • Captures execution snapshots from tests\n"
            "  • Verifies I/O equivalence\n"
            "  • Refines failed translations\n"
            "  • Generates mocks for untranslatable code\n\n"
            "[cyan]Next:[/cyan] Run full translation with: codemorph translate ./examples/python_project",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
