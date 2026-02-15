"""
Unit tests for Phase 3 components.

Tests for:
- Snapshot capture
- Equivalence checking
- Function mocking
- Java executor
"""

import json
import tempfile
from pathlib import Path
import pytest

from codemorph.verifier.snapshot_capture import (
    ExecutionSnapshot,
    SnapshotCollector,
    SnapshotSerializer,
    log_snapshot,
    set_global_collector,
)
from codemorph.verifier.equivalence_checker import (
    EquivalenceChecker,
    EquivalenceStatus,
    OutputComparator,
    ExceptionMapper,
)
from codemorph.verifier.mocker import (
    FunctionMocker,
    MockStrategy,
    MockRegistry,
)


# =============================================================================
# Snapshot Capture Tests
# =============================================================================


class TestSnapshotSerializer:
    """Test snapshot serialization."""

    def test_serialize_primitives(self):
        """Test serialization of primitive types."""
        serializer = SnapshotSerializer()

        assert serializer.serialize(None) is None
        assert serializer.serialize(True) is True
        assert serializer.serialize(42) == 42
        assert serializer.serialize(3.14) == 3.14
        assert serializer.serialize("hello") == "hello"

    def test_serialize_collections(self):
        """Test serialization of collections."""
        serializer = SnapshotSerializer()

        # List
        assert serializer.serialize([1, 2, 3]) == [1, 2, 3]

        # Dict
        assert serializer.serialize({"a": 1, "b": 2}) == {"a": 1, "b": 2}

        # Set
        result = serializer.serialize({1, 2, 3})
        assert result["_type"] == "set"
        assert set(result["values"]) == {1, 2, 3}

    def test_serialize_custom_object(self):
        """Test serialization of custom objects."""
        serializer = SnapshotSerializer()

        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        person = Person("Alice", 30)
        result = serializer.serialize(person)

        assert result["_type"] == "Person"
        assert result["attributes"]["name"] == "Alice"
        assert result["attributes"]["age"] == 30


class TestSnapshotCollector:
    """Test snapshot collection."""

    def test_record_and_load_snapshot(self):
        """Test recording and loading snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SnapshotCollector(Path(tmpdir))

            snapshot = ExecutionSnapshot(
                function_name="test_func",
                function_id="test_module::test_func",
                timestamp="2024-01-01T00:00:00",
                args=[1, 2],
                kwargs={"x": 3},
                output=6,
            )

            collector.record_snapshot(snapshot)

            # Load snapshots
            loaded = collector.load_snapshots("test_module::test_func")

            assert len(loaded) == 1
            assert loaded[0].function_name == "test_func"
            assert loaded[0].args == [1, 2]
            assert loaded[0].output == 6

    def test_get_summary(self):
        """Test summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SnapshotCollector(Path(tmpdir))

            for i in range(3):
                snapshot = ExecutionSnapshot(
                    function_name="func1",
                    function_id="module::func1",
                    timestamp="2024-01-01T00:00:00",
                    args=[i],
                    kwargs={},
                    output=i * 2,
                )
                collector.record_snapshot(snapshot)

            summary = collector.get_summary()
            assert summary["module::func1"] == 3


class TestLogSnapshotDecorator:
    """Test the log_snapshot decorator."""

    def test_successful_execution(self):
        """Test decorator with successful execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SnapshotCollector(Path(tmpdir))
            set_global_collector(collector)

            @log_snapshot
            def add(a, b):
                return a + b

            result = add(2, 3)

            assert result == 5

            # Check snapshot was recorded - use the actual function_id from collector
            assert len(collector.snapshots) == 1
            func_id = list(collector.snapshots.keys())[0]
            assert func_id.endswith("::add")
            snapshots = collector.load_snapshots(func_id)
            assert len(snapshots) == 1
            assert snapshots[0].args == [2, 3]
            assert snapshots[0].output == 5

    def test_exception_capture(self):
        """Test decorator captures exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SnapshotCollector(Path(tmpdir))
            set_global_collector(collector)

            @log_snapshot
            def divide(a, b):
                return a / b

            with pytest.raises(ZeroDivisionError):
                divide(10, 0)

            # Check exception was recorded - use the actual function_id from collector
            assert len(collector.snapshots) == 1
            func_id = list(collector.snapshots.keys())[0]
            assert func_id.endswith("::divide")
            snapshots = collector.load_snapshots(func_id)
            assert len(snapshots) == 1
            assert snapshots[0].exception_type == "ZeroDivisionError"


# =============================================================================
# Equivalence Checker Tests
# =============================================================================


class TestOutputComparator:
    """Test output comparison logic."""

    def test_primitives_equal(self):
        """Test primitive equality."""
        comparator = OutputComparator()

        equal, diff = comparator.are_equal(42, 42)
        assert equal
        assert diff is None

        equal, diff = comparator.are_equal("hello", "hello")
        assert equal

        equal, diff = comparator.are_equal(True, True)
        assert equal

    def test_primitives_not_equal(self):
        """Test primitive inequality."""
        comparator = OutputComparator()

        equal, diff = comparator.are_equal(42, 43)
        assert not equal
        assert "42 != 43" in diff

    def test_floats_with_tolerance(self):
        """Test float comparison with tolerance."""
        comparator = OutputComparator()

        equal, diff = comparator.are_equal(3.14159, 3.14159, tolerance=1e-9)
        assert equal

        equal, diff = comparator.are_equal(3.14, 3.14000001, tolerance=1e-5)
        assert equal

        equal, diff = comparator.are_equal(3.14, 3.15, tolerance=1e-5)
        assert not equal

    def test_lists_equal(self):
        """Test list equality."""
        comparator = OutputComparator()

        equal, diff = comparator.are_equal([1, 2, 3], [1, 2, 3])
        assert equal

        equal, diff = comparator.are_equal([1, 2, 3], [1, 2, 4])
        assert not equal

    def test_dicts_equal(self):
        """Test dictionary equality."""
        comparator = OutputComparator()

        equal, diff = comparator.are_equal(
            {"a": 1, "b": 2},
            {"a": 1, "b": 2}
        )
        assert equal

        equal, diff = comparator.are_equal(
            {"a": 1, "b": 2},
            {"a": 1, "b": 3}
        )
        assert not equal

    def test_nested_structures(self):
        """Test nested data structures."""
        comparator = OutputComparator()

        data1 = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        data2 = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}

        equal, diff = comparator.are_equal(data1, data2)
        assert equal


class TestExceptionMapper:
    """Test exception mapping between languages."""

    def test_equivalent_exceptions(self):
        """Test exception equivalence mapping."""
        assert ExceptionMapper.are_equivalent_exceptions(
            "KeyError",
            "java.util.NoSuchElementException"
        )

        assert ExceptionMapper.are_equivalent_exceptions(
            "ValueError",
            "java.lang.IllegalArgumentException"
        )

        assert ExceptionMapper.are_equivalent_exceptions(
            "ZeroDivisionError",
            "java.lang.ArithmeticException"
        )

    def test_non_equivalent_exceptions(self):
        """Test non-equivalent exceptions."""
        assert not ExceptionMapper.are_equivalent_exceptions(
            "KeyError",
            "java.lang.NullPointerException"
        )


class TestEquivalenceChecker:
    """Test the main equivalence checker."""

    def test_check_function_all_pass(self):
        """Test function with all snapshots passing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create snapshots
            collector = SnapshotCollector(tmpdir)

            snapshot1 = ExecutionSnapshot(
                function_name="add",
                function_id="module::add",
                timestamp="2024-01-01T00:00:00",
                args=[2, 3],
                kwargs={},
                output=5,
            )
            snapshot2 = ExecutionSnapshot(
                function_name="add",
                function_id="module::add",
                timestamp="2024-01-01T00:00:01",
                args=[10, 20],
                kwargs={},
                output=30,
            )

            collector.record_snapshot(snapshot1)
            collector.record_snapshot(snapshot2)

            # Create executor
            def add_executor(*args, **kwargs):
                return args[0] + args[1]

            # Check equivalence
            checker = EquivalenceChecker(tmpdir)
            report = checker.check_function("module::add", add_executor)

            assert report.total_snapshots == 2
            assert report.passed == 2
            assert report.failed == 0
            assert report.is_equivalent

    def test_check_function_with_failures(self):
        """Test function with some failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create snapshot
            collector = SnapshotCollector(tmpdir)

            snapshot = ExecutionSnapshot(
                function_name="multiply",
                function_id="module::multiply",
                timestamp="2024-01-01T00:00:00",
                args=[2, 3],
                kwargs={},
                output=6,
            )

            collector.record_snapshot(snapshot)

            # Create buggy executor (returns wrong result)
            def buggy_executor(*args, **kwargs):
                return args[0] + args[1]  # Should multiply, but adds

            # Check equivalence
            checker = EquivalenceChecker(tmpdir)
            report = checker.check_function("module::multiply", buggy_executor)

            assert report.total_snapshots == 1
            assert report.passed == 0
            assert report.failed == 1
            assert not report.is_equivalent


# =============================================================================
# Function Mocker Tests
# =============================================================================


class TestFunctionMocker:
    """Test function mocking generation."""

    def test_generate_stub_mock_java(self):
        """Test stub generation for Java."""
        mocker = FunctionMocker("python", "java")

        mock = mocker.generate_mock(
            function_id="module::my_func",
            function_name="myFunc",
            signature="public int myFunc(int x)",
            strategy=MockStrategy.STUB,
            reason="Test reason"
        )

        assert mock.strategy == MockStrategy.STUB
        assert "UnsupportedOperationException" in mock.generated_code
        assert "myFunc" in mock.generated_code

    def test_generate_stub_mock_python(self):
        """Test stub generation for Python."""
        mocker = FunctionMocker("java", "python")

        mock = mocker.generate_mock(
            function_id="com.example::myMethod",
            function_name="my_method",
            signature="def my_method(x: int) -> int",
            strategy=MockStrategy.STUB,
            reason="Test reason"
        )

        assert mock.strategy == MockStrategy.STUB
        assert "NotImplementedError" in mock.generated_code
        assert "my_method" in mock.generated_code


class TestMockRegistry:
    """Test mock registry."""

    def test_register_and_retrieve_mock(self):
        """Test registering and retrieving mocks."""
        registry = MockRegistry()
        mocker = FunctionMocker("python", "java")

        mock = mocker.generate_mock(
            function_id="module::func1",
            function_name="func1",
            signature="public void func1()",
            strategy=MockStrategy.STUB,
            reason="Test"
        )

        registry.register_mock(mock)

        assert registry.is_mocked("module::func1")
        retrieved = registry.get_mock("module::func1")
        assert retrieved.function_id == "module::func1"

    def test_get_mocks_by_strategy(self):
        """Test filtering mocks by strategy."""
        registry = MockRegistry()
        mocker = FunctionMocker("python", "java")

        stub_mock = mocker.generate_mock(
            "module::func1", "func1", "public void func1()",
            MockStrategy.STUB, "Test"
        )
        manual_mock = mocker.generate_mock(
            "module::func2", "func2", "public void func2()",
            MockStrategy.MANUAL, "Test"
        )

        registry.register_mock(stub_mock)
        registry.register_mock(manual_mock)

        stubs = registry.get_mocks_by_strategy(MockStrategy.STUB)
        assert len(stubs) == 1
        assert stubs[0].function_id == "module::func1"

        manuals = registry.get_mocks_by_strategy(MockStrategy.MANUAL)
        assert len(manuals) == 1
        assert manuals[0].function_id == "module::func2"

    def test_generate_report(self):
        """Test report generation."""
        registry = MockRegistry()
        mocker = FunctionMocker("python", "java")

        mock = mocker.generate_mock(
            "module::func1", "func1", "public void func1()",
            MockStrategy.MANUAL, "Complex algorithm"
        )

        registry.register_mock(mock)

        report = registry.generate_report()

        assert "MOCKED FUNCTIONS REPORT" in report
        assert "module::func1" in report
        assert "Complex algorithm" in report
        assert "ACTION REQUIRED" in report


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase3Integration:
    """Integration tests for Phase 3 workflow."""

    def test_end_to_end_snapshot_to_equivalence(self):
        """Test complete flow from snapshot capture to equivalence check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Capture snapshots
            collector = SnapshotCollector(tmpdir)
            set_global_collector(collector)

            @log_snapshot
            def calculate_total(items, tax_rate=0.05):
                total = sum(item['price'] for item in items)
                return total * (1 + tax_rate)

            # Execute with test data
            result1 = calculate_total([{'price': 100}], 0.05)
            result2 = calculate_total([{'price': 50}, {'price': 30}], 0.10)

            assert result1 == 105.0
            assert result2 == 88.0

            # Get the actual function_id from the collector
            assert len(collector.snapshots) == 1
            func_id = list(collector.snapshots.keys())[0]
            assert func_id.endswith("::calculate_total")

            # Step 2: Create Java-equivalent executor (simulated)
            def java_executor(*args, **kwargs):
                # Simulate Java implementation
                items = args[0]
                tax_rate = kwargs.get('tax_rate', 0.05) if kwargs else (args[1] if len(args) > 1 else 0.05)

                total = sum(item['price'] for item in items)
                return total * (1 + tax_rate)

            # Step 3: Check equivalence using the actual function_id
            checker = EquivalenceChecker(tmpdir)
            report = checker.check_function(func_id, java_executor)

            # Verify results
            assert report.total_snapshots == 2
            assert report.passed == 2
            assert report.is_equivalent
            assert report.success_rate == 1.0
