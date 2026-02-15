"""
I/O Equivalence Verification (Phase 3)

Verifies that translated code produces the same outputs as the source code
for the same inputs. This is the "Judge" module.

Based on Section 12 of the CodeMorph v2.0 plan.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .snapshot_capture import ExecutionSnapshot, SnapshotCollector, SnapshotSerializer

logger = logging.getLogger(__name__)


class EquivalenceStatus(str, Enum):
    """Status of equivalence check."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EquivalenceResult:
    """Result of comparing one execution."""

    snapshot_id: int
    status: EquivalenceStatus

    # Input/Output data
    inputs: Dict[str, Any]
    expected_output: Optional[Any]
    actual_output: Optional[Any] = None

    # Exception handling
    expected_exception: Optional[str] = None
    actual_exception: Optional[str] = None

    # Comparison details
    diff: Optional[str] = None
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if self.status == EquivalenceStatus.PASSED:
            return f"✓ Snapshot {self.snapshot_id}: PASSED"
        elif self.status == EquivalenceStatus.FAILED:
            return f"✗ Snapshot {self.snapshot_id}: FAILED - {self.error_message}"
        elif self.status == EquivalenceStatus.ERROR:
            return f"⚠ Snapshot {self.snapshot_id}: ERROR - {self.error_message}"
        else:
            return f"○ Snapshot {self.snapshot_id}: SKIPPED"


@dataclass
class FunctionEquivalenceReport:
    """Equivalence report for a single function."""

    function_id: str
    total_snapshots: int
    passed: int
    failed: int
    errors: int
    skipped: int

    results: List[EquivalenceResult]

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 - 1.0)."""
        if self.total_snapshots == 0:
            return 0.0
        return self.passed / self.total_snapshots

    @property
    def is_equivalent(self) -> bool:
        """Check if function is fully equivalent."""
        return self.passed == self.total_snapshots and self.failed == 0 and self.errors == 0

    def __str__(self) -> str:
        return (
            f"{self.function_id}:\n"
            f"  Total: {self.total_snapshots}\n"
            f"  Passed: {self.passed} ({self.success_rate:.1%})\n"
            f"  Failed: {self.failed}\n"
            f"  Errors: {self.errors}\n"
            f"  Skipped: {self.skipped}"
        )


class OutputComparator:
    """Handles deep comparison of Python and Java outputs."""

    @staticmethod
    def are_equal(expected: Any, actual: Any, tolerance: float = 1e-9) -> Tuple[bool, Optional[str]]:
        """
        Deep equality check with tolerance for floating point.

        Args:
            expected: Expected value (from source)
            actual: Actual value (from target)
            tolerance: Tolerance for float comparison

        Returns:
            Tuple of (is_equal, diff_message)
        """
        return OutputComparator._compare_recursive(expected, actual, tolerance, path="root")

    @staticmethod
    def _compare_recursive(
        expected: Any,
        actual: Any,
        tolerance: float,
        path: str
    ) -> Tuple[bool, Optional[str]]:
        """Recursive comparison with path tracking."""

        # Type checking
        if type(expected) != type(actual):
            # Special case: int vs float
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if abs(expected - actual) <= tolerance:
                    return True, None
                return False, f"At {path}: {expected} != {actual} (numeric difference)"

            return False, f"At {path}: Type mismatch - {type(expected).__name__} vs {type(actual).__name__}"

        # None
        if expected is None:
            return True, None

        # Booleans
        if isinstance(expected, bool):
            if expected == actual:
                return True, None
            return False, f"At {path}: {expected} != {actual}"

        # Numbers
        if isinstance(expected, (int, float)):
            if abs(expected - actual) <= tolerance:
                return True, None
            return False, f"At {path}: {expected} != {actual} (difference: {abs(expected - actual)})"

        # Strings
        if isinstance(expected, str):
            if expected == actual:
                return True, None
            return False, f"At {path}: '{expected}' != '{actual}'"

        # Lists
        if isinstance(expected, list):
            if len(expected) != len(actual):
                return False, f"At {path}: List length mismatch - {len(expected)} vs {len(actual)}"

            for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                equal, diff = OutputComparator._compare_recursive(
                    exp_item, act_item, tolerance, f"{path}[{i}]"
                )
                if not equal:
                    return False, diff

            return True, None

        # Dictionaries
        if isinstance(expected, dict):
            if set(expected.keys()) != set(actual.keys()):
                missing_in_actual = set(expected.keys()) - set(actual.keys())
                extra_in_actual = set(actual.keys()) - set(expected.keys())
                msg = f"At {path}: Key mismatch"
                if missing_in_actual:
                    msg += f" - Missing: {missing_in_actual}"
                if extra_in_actual:
                    msg += f" - Extra: {extra_in_actual}"
                return False, msg

            for key in expected.keys():
                equal, diff = OutputComparator._compare_recursive(
                    expected[key], actual[key], tolerance, f"{path}[{key}]"
                )
                if not equal:
                    return False, diff

            return True, None

        # Sets
        if isinstance(expected, set):
            if expected == actual:
                return True, None
            missing = expected - actual
            extra = actual - expected
            msg = f"At {path}: Set mismatch"
            if missing:
                msg += f" - Missing: {missing}"
            if extra:
                msg += f" - Extra: {extra}"
            return False, msg

        # Fallback: use equality operator
        if expected == actual:
            return True, None

        return False, f"At {path}: {expected} != {actual}"


class ExceptionMapper:
    """Maps exceptions between Python and Java."""

    # Standard exception mappings
    EXCEPTION_MAP = {
        # Python -> Java
        "KeyError": "java.util.NoSuchElementException",
        "ValueError": "java.lang.IllegalArgumentException",
        "TypeError": "java.lang.IllegalArgumentException",
        "IndexError": "java.lang.IndexOutOfBoundsException",
        "AttributeError": "java.lang.NoSuchFieldException",
        "FileNotFoundError": "java.io.FileNotFoundException",
        "PermissionError": "java.lang.SecurityException",
        "ZeroDivisionError": "java.lang.ArithmeticException",
        "RuntimeError": "java.lang.RuntimeException",
        "NotImplementedError": "java.lang.UnsupportedOperationException",
    }

    @classmethod
    def are_equivalent_exceptions(cls, py_exception: str, java_exception: str) -> bool:
        """Check if Python and Java exceptions are semantically equivalent."""

        # Extract exception type name (remove package path)
        java_type = java_exception.split(".")[-1] if "." in java_exception else java_exception

        # Direct match
        if py_exception == java_type:
            return True

        # Check mapping
        expected_java = cls.EXCEPTION_MAP.get(py_exception)
        if expected_java:
            return java_exception == expected_java or java_type in expected_java

        # Fallback: both are some kind of exception
        return "Exception" in py_exception and "Exception" in java_exception


class EquivalenceChecker:
    """
    Main equivalence checker.
    Loads snapshots, executes target code, compares results.
    """

    def __init__(
        self,
        snapshot_dir: Path,
        float_tolerance: float = 1e-9
    ):
        """
        Initialize equivalence checker.

        Args:
            snapshot_dir: Directory containing snapshot files
            float_tolerance: Tolerance for floating point comparisons
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.float_tolerance = float_tolerance
        self.collector = SnapshotCollector(snapshot_dir)
        self.serializer = SnapshotSerializer()
        self.comparator = OutputComparator()

    def check_function(
        self,
        function_id: str,
        executor: callable,
        max_snapshots: Optional[int] = None
    ) -> FunctionEquivalenceReport:
        """
        Check equivalence for a specific function.

        Args:
            function_id: Function identifier (e.g., "module::function")
            executor: Callable that executes the target function
                      Should accept (*args, **kwargs) and return output or raise exception
            max_snapshots: Maximum number of snapshots to test (None = all)

        Returns:
            FunctionEquivalenceReport with results
        """
        # Load snapshots
        snapshots = self.collector.load_snapshots(function_id)

        if not snapshots:
            logger.warning(f"No snapshots found for {function_id}")
            return FunctionEquivalenceReport(
                function_id=function_id,
                total_snapshots=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                results=[]
            )

        # Limit snapshots if requested
        if max_snapshots is not None:
            snapshots = snapshots[:max_snapshots]

        logger.info(f"Testing {len(snapshots)} snapshots for {function_id}")

        # Check each snapshot
        results = []
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        for idx, snapshot in enumerate(snapshots):
            result = self._check_snapshot(idx, snapshot, executor)
            results.append(result)

            if result.status == EquivalenceStatus.PASSED:
                passed += 1
            elif result.status == EquivalenceStatus.FAILED:
                failed += 1
            elif result.status == EquivalenceStatus.ERROR:
                errors += 1
            else:
                skipped += 1

        return FunctionEquivalenceReport(
            function_id=function_id,
            total_snapshots=len(snapshots),
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            results=results
        )

    def _check_snapshot(
        self,
        idx: int,
        snapshot: ExecutionSnapshot,
        executor: callable
    ) -> EquivalenceResult:
        """Check a single snapshot."""

        # Deserialize inputs
        args = self.serializer.deserialize(snapshot.args)
        kwargs = self.serializer.deserialize(snapshot.kwargs)

        # Create result object
        result = EquivalenceResult(
            snapshot_id=idx,
            status=EquivalenceStatus.PASSED,
            inputs={"args": args, "kwargs": kwargs},
            expected_output=snapshot.output,
            expected_exception=snapshot.exception
        )

        try:
            # Execute target function
            actual_output = executor(*args, **kwargs)

            # Deserialize expected output
            expected_output = self.serializer.deserialize(snapshot.output)

            # If source raised exception but target didn't
            if snapshot.exception is not None:
                result.status = EquivalenceStatus.FAILED
                result.error_message = (
                    f"Expected exception {snapshot.exception_type}, "
                    f"but got output: {actual_output}"
                )
                result.actual_output = actual_output
                return result

            # Compare outputs
            are_equal, diff = self.comparator.are_equal(
                expected_output,
                actual_output,
                self.float_tolerance
            )

            result.actual_output = actual_output

            if are_equal:
                result.status = EquivalenceStatus.PASSED
            else:
                result.status = EquivalenceStatus.FAILED
                result.error_message = "Output mismatch"
                result.diff = diff

        except Exception as e:
            actual_exception_type = type(e).__name__
            result.actual_exception = str(e)

            # If source also raised exception, check if they're equivalent
            if snapshot.exception is not None:
                if ExceptionMapper.are_equivalent_exceptions(
                    actual_exception_type,   # Python exception type
                    snapshot.exception_type   # Java exception type
                ):
                    result.status = EquivalenceStatus.PASSED
                else:
                    result.status = EquivalenceStatus.FAILED
                    result.error_message = (
                        f"Exception mismatch: expected {snapshot.exception_type}, "
                        f"got {actual_exception_type}"
                    )
            else:
                # Source didn't raise exception, but target did
                result.status = EquivalenceStatus.ERROR
                result.error_message = f"Unexpected exception: {actual_exception_type}: {e}"

        return result

    def generate_report(self, reports: List[FunctionEquivalenceReport]) -> str:
        """Generate a human-readable equivalence report."""
        lines = []
        lines.append("=" * 80)
        lines.append("I/O EQUIVALENCE VERIFICATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        total_functions = len(reports)
        total_snapshots = sum(r.total_snapshots for r in reports)
        total_passed = sum(r.passed for r in reports)
        total_failed = sum(r.failed for r in reports)
        total_errors = sum(r.errors for r in reports)

        lines.append(f"Functions tested: {total_functions}")
        lines.append(f"Total snapshots: {total_snapshots}")
        lines.append(f"Passed: {total_passed} ({total_passed/total_snapshots:.1%})")
        lines.append(f"Failed: {total_failed}")
        lines.append(f"Errors: {total_errors}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("DETAILED RESULTS")
        lines.append("=" * 80)
        lines.append("")

        for report in reports:
            lines.append(str(report))

            # Show first few failures
            failures = [r for r in report.results if r.status in (EquivalenceStatus.FAILED, EquivalenceStatus.ERROR)]
            if failures:
                lines.append("  First failures:")
                for failure in failures[:3]:  # Show first 3
                    lines.append(f"    {failure}")

            lines.append("")

        return "\n".join(lines)
