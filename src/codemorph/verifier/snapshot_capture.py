"""
Execution Snapshots System (Phase 3)

Captures "ground truth" I/O behavior from source code during test execution.
Uses instrumentation to record function inputs, outputs, and exceptions.

Based on Section 11 of the CodeMorph v2.0 plan.
"""

import functools
import hashlib
import inspect
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSnapshot:
    """Represents a single execution of a function."""

    function_name: str
    function_id: str  # module::function
    timestamp: str

    # Input data
    args: List[Any]
    kwargs: Dict[str, Any]

    # Output data
    output: Optional[Any] = None
    exception: Optional[str] = None
    exception_type: Optional[str] = None

    # Context
    call_depth: int = 0
    callee_calls: List[Dict[str, Any]] = None  # Calls to other functions

    def __post_init__(self):
        if self.callee_calls is None:
            self.callee_calls = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionSnapshot':
        """Create from dictionary."""
        return cls(**data)


class SnapshotSerializer:
    """Handles serialization of Python objects to JSON-compatible format."""

    @staticmethod
    def serialize(obj: Any) -> Any:
        """
        Serialize a Python object to JSON-compatible format.
        Handles common types, with fallback to string representation.
        """
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        if isinstance(obj, (list, tuple)):
            return [SnapshotSerializer.serialize(item) for item in obj]

        if isinstance(obj, dict):
            return {
                str(k): SnapshotSerializer.serialize(v)
                for k, v in obj.items()
            }

        if isinstance(obj, set):
            return {
                "_type": "set",
                "values": [SnapshotSerializer.serialize(item) for item in obj]
            }

        if hasattr(obj, '__dict__'):
            # Custom object - serialize its attributes
            return {
                "_type": type(obj).__name__,
                "_module": type(obj).__module__,
                "attributes": {
                    k: SnapshotSerializer.serialize(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith('_')
                }
            }

        # Fallback: use string representation
        return {
            "_type": type(obj).__name__,
            "_str": str(obj),
            "_repr": repr(obj)
        }

    @staticmethod
    def deserialize(data: Any) -> Any:
        """
        Deserialize JSON data back to Python objects.
        Best-effort reconstruction.
        """
        if data is None or isinstance(data, (bool, int, float, str)):
            return data

        if isinstance(data, list):
            return [SnapshotSerializer.deserialize(item) for item in data]

        if isinstance(data, dict):
            if "_type" in data:
                if data["_type"] == "set":
                    return set(SnapshotSerializer.deserialize(item)
                             for item in data.get("values", []))
                # For custom objects, just return the attributes dict
                return data.get("attributes", data)

            return {
                k: SnapshotSerializer.deserialize(v)
                for k, v in data.items()
            }

        return data


class SnapshotCollector:
    """
    Manages snapshot collection for a test session.
    Stores snapshots in JSONL format for streaming.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the snapshot collector.

        Args:
            output_dir: Directory to store snapshot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: Dict[str, List[ExecutionSnapshot]] = {}
        self.serializer = SnapshotSerializer()

        # Track call stack for nested function calls
        self.call_stack: List[str] = []

    def record_snapshot(self, snapshot: ExecutionSnapshot):
        """Record a snapshot."""
        func_id = snapshot.function_id

        if func_id not in self.snapshots:
            self.snapshots[func_id] = []

        self.snapshots[func_id].append(snapshot)

        # Also write immediately to disk (streaming)
        self._append_to_file(func_id, snapshot)

    def _append_to_file(self, func_id: str, snapshot: ExecutionSnapshot):
        """Append snapshot to JSONL file."""
        # Create safe filename from function ID
        safe_name = func_id.replace("::", "_").replace("/", "_")
        filepath = self.output_dir / f"{safe_name}.jsonl"

        try:
            with open(filepath, 'a') as f:
                json.dump(snapshot.to_dict(), f)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to write snapshot to {filepath}: {e}")

    def save_all(self):
        """Save all snapshots (already done incrementally via _append_to_file)."""
        logger.info(f"Collected {sum(len(snaps) for snaps in self.snapshots.values())} "
                   f"snapshots for {len(self.snapshots)} functions")

    def load_snapshots(self, func_id: str) -> List[ExecutionSnapshot]:
        """Load snapshots for a specific function."""
        safe_name = func_id.replace("::", "_").replace("/", "_")
        filepath = self.output_dir / f"{safe_name}.jsonl"

        if not filepath.exists():
            return []

        snapshots = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    snapshots.append(ExecutionSnapshot.from_dict(data))

        return snapshots

    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics."""
        return {
            func_id: len(snaps)
            for func_id, snaps in self.snapshots.items()
        }


# Global collector instance (can be set per test session)
_global_collector: Optional[SnapshotCollector] = None


def set_global_collector(collector: SnapshotCollector):
    """Set the global snapshot collector."""
    global _global_collector
    _global_collector = collector


def get_global_collector() -> Optional[SnapshotCollector]:
    """Get the global snapshot collector."""
    return _global_collector


def log_snapshot(func: Callable) -> Callable:
    """
    Decorator to instrument a function and capture execution snapshots.

    Usage:
        @log_snapshot
        def my_function(x, y):
            return x + y

    The decorator will:
    1. Capture input arguments (args, kwargs)
    2. Execute the function
    3. Capture output or exception
    4. Save snapshot to collector
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        collector = get_global_collector()

        # If no collector, just execute normally
        if collector is None:
            return func(*args, **kwargs)

        # Build function ID
        module = inspect.getmodule(func)
        module_name = module.__name__ if module else "__main__"
        func_id = f"{module_name}::{func.__name__}"

        # Serialize inputs
        serialized_args = collector.serializer.serialize(args)
        serialized_kwargs = collector.serializer.serialize(kwargs)

        # Track call depth
        call_depth = len(collector.call_stack)
        collector.call_stack.append(func_id)

        # Create snapshot
        snapshot = ExecutionSnapshot(
            function_name=func.__name__,
            function_id=func_id,
            timestamp=datetime.utcnow().isoformat(),
            args=serialized_args,
            kwargs=serialized_kwargs,
            call_depth=call_depth
        )

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Serialize output
            snapshot.output = collector.serializer.serialize(result)

            # Record snapshot
            collector.record_snapshot(snapshot)

            return result

        except Exception as e:
            # Capture exception details
            snapshot.exception = str(e)
            snapshot.exception_type = type(e).__name__

            # Record snapshot
            collector.record_snapshot(snapshot)

            # Re-raise
            raise

        finally:
            # Pop from call stack
            if collector.call_stack:
                collector.call_stack.pop()

    return wrapper


def instrument_module(module, function_names: Optional[List[str]] = None):
    """
    Instrument all functions in a module with snapshot logging.

    Args:
        module: The module to instrument
        function_names: Specific function names to instrument (None = all)

    Returns:
        Dict mapping original functions to instrumented versions
    """
    instrumented = {}

    for name in dir(module):
        obj = getattr(module, name)

        # Only instrument functions
        if not callable(obj) or name.startswith('_'):
            continue

        # Filter by name if specified
        if function_names is not None and name not in function_names:
            continue

        # Instrument
        instrumented_func = log_snapshot(obj)
        setattr(module, name, instrumented_func)
        instrumented[name] = instrumented_func

    return instrumented


def capture_snapshots_from_tests(
    test_module_path: Path,
    source_module_path: Path,
    output_dir: Path,
    test_runner: str = "pytest"
) -> SnapshotCollector:
    """
    Run tests and capture execution snapshots.

    Args:
        test_module_path: Path to test file/directory
        source_module_path: Path to source code to instrument
        output_dir: Where to save snapshots
        test_runner: Test runner to use ("pytest", "unittest")

    Returns:
        SnapshotCollector with captured snapshots
    """
    import subprocess
    import sys
    import importlib.util

    # Create collector
    collector = SnapshotCollector(output_dir)
    set_global_collector(collector)

    # Load source module
    spec = importlib.util.spec_from_file_location("source_module", source_module_path)
    source_module = importlib.util.module_from_spec(spec)

    # Instrument the source module
    logger.info(f"Instrumenting {source_module_path}")
    instrument_module(source_module)

    # Add to sys.modules so tests can import it
    sys.modules['source_module'] = source_module
    spec.loader.exec_module(source_module)

    # Run tests
    logger.info(f"Running tests from {test_module_path}")

    if test_runner == "pytest":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_module_path), "-v"],
            capture_output=True,
            text=True
        )
    else:  # unittest
        result = subprocess.run(
            [sys.executable, "-m", "unittest", str(test_module_path)],
            capture_output=True,
            text=True
        )

    logger.info(f"Test run completed with return code {result.returncode}")

    if result.returncode != 0:
        logger.warning("Some tests failed, but snapshots were captured")

    # Save all snapshots
    collector.save_all()

    return collector
