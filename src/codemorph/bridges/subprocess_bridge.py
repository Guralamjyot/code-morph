"""
Subprocess-based cross-language bridge using JSON for data exchange.

This is the DEFAULT bridge implementation as recommended by the simplified spec.
It's universal, portable, and works on any system with Python and Java installed.

Advantages over JPype/Py4J:
- No native dependencies
- Works everywhere
- Easy to debug (JSON is human-readable)
- Simple subprocess model
- No complex JVM integration

Usage:
    bridge = SubprocessBridge()
    result = bridge.call_java("com.example.Calculator", "add", {"a": 5, "b": 3})
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class BridgeError(Exception):
    """Raised when cross-language bridge execution fails."""

    pass


class SubprocessBridge:
    """
    Universal cross-language bridge using subprocess + JSON.

    This bridge:
    1. Serializes inputs to JSON
    2. Launches target language process
    3. Passes JSON via stdin
    4. Reads JSON result from stdout
    5. Deserializes and returns

    This approach is simpler and more debuggable than native bridges.
    """

    def __init__(self, java_classpath: str | Path | None = None):
        """
        Initialize the subprocess bridge.

        Args:
            java_classpath: Classpath for Java execution (optional)
        """
        self.java_classpath = str(java_classpath) if java_classpath else "."

    def call_java(
        self,
        class_name: str,
        method_name: str,
        args: dict[str, Any],
        timeout: int = 30,
    ) -> Any:
        """
        Call a Java method via subprocess with JSON I/O.

        Args:
            class_name: Fully qualified Java class name (e.g., "com.example.Calculator")
            method_name: Method name to call
            args: Method arguments as dict (will be serialized to JSON)
            timeout: Execution timeout in seconds

        Returns:
            The result from Java (deserialized from JSON)

        Raises:
            BridgeError: If execution fails
        """
        try:
            # Serialize inputs to JSON
            input_json = json.dumps(args)

            # Build Java command
            # Expects a BridgeRunner class that:
            # 1. Reads JSON from stdin
            # 2. Deserializes to Java objects
            # 3. Calls the target method
            # 4. Serializes result to JSON
            # 5. Writes to stdout
            java_cmd = [
                "java",
                "-cp",
                self.java_classpath,
                "codemorph.bridge.BridgeRunner",
                class_name,
                method_name,
            ]

            # Execute Java process
            result = subprocess.run(
                java_cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # Don't raise on non-zero exit
            )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown Java execution error"
                raise BridgeError(
                    f"Java execution failed (exit code {result.returncode}): {error_msg}"
                )

            # Deserialize result from stdout
            if not result.stdout.strip():
                return None

            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise BridgeError(
                    f"Failed to parse Java output as JSON: {result.stdout[:200]}\nError: {e}"
                )

        except subprocess.TimeoutExpired:
            raise BridgeError(f"Java execution timed out after {timeout} seconds")
        except FileNotFoundError:
            raise BridgeError(
                "Java executable not found. Ensure Java is installed and in PATH."
            )
        except Exception as e:
            raise BridgeError(f"Unexpected error during Java bridge call: {e}")

    def call_python(
        self,
        module_path: str,
        function_name: str,
        args: dict[str, Any],
        python_path: str | None = None,
        timeout: int = 30,
    ) -> Any:
        """
        Call a Python function via subprocess with JSON I/O.

        Useful for Java → Python calls (e.g., when mocking Python functions).

        Args:
            module_path: Python module path (e.g., "myapp.utils")
            function_name: Function name to call
            args: Function arguments as dict
            python_path: Path to Python executable (default: current Python)
            timeout: Execution timeout in seconds

        Returns:
            The result from Python (deserialized from JSON)

        Raises:
            BridgeError: If execution fails
        """
        try:
            # Serialize inputs
            input_json = json.dumps(
                {"module": module_path, "function": function_name, "args": args}
            )

            # Use current Python or specified
            python_exe = python_path or "python3"

            # Build Python command
            # Uses a bridge runner script
            python_cmd = [
                python_exe,
                "-m",
                "codemorph.bridges.python_runner",
            ]

            # Execute Python process
            result = subprocess.run(
                python_cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown Python execution error"
                raise BridgeError(
                    f"Python execution failed (exit code {result.returncode}): {error_msg}"
                )

            if not result.stdout.strip():
                return None

            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise BridgeError(
                    f"Failed to parse Python output as JSON: {result.stdout[:200]}\nError: {e}"
                )

        except subprocess.TimeoutExpired:
            raise BridgeError(f"Python execution timed out after {timeout} seconds")
        except FileNotFoundError:
            raise BridgeError(
                f"Python executable not found at: {python_exe}. "
                "Ensure Python is installed and in PATH."
            )
        except Exception as e:
            raise BridgeError(f"Unexpected error during Python bridge call: {e}")

    def check_java_available(self) -> tuple[bool, str]:
        """
        Check if Java is available and get version.

        Returns:
            Tuple of (is_available, version_string)
        """
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Java version goes to stderr for some reason
            version_info = result.stderr.split("\n")[0] if result.stderr else "Unknown"
            return True, version_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False, "Not found"

    def check_python_available(self, python_path: str = "python3") -> tuple[bool, str]:
        """
        Check if Python is available and get version.

        Args:
            python_path: Path to Python executable

        Returns:
            Tuple of (is_available, version_string)
        """
        try:
            result = subprocess.run(
                [python_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version_info = result.stdout.strip() or result.stderr.strip()
            return True, version_info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False, "Not found"

    def test_round_trip(self) -> bool:
        """
        Test JSON serialization round-trip.

        This is useful for type compatibility checking (Section 10 of spec).

        Returns:
            True if round-trip succeeds
        """
        test_data = {
            "int_val": 42,
            "float_val": 3.14,
            "string_val": "hello",
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
            "bool_val": True,
            "null_val": None,
        }

        try:
            # Test JSON encoding/decoding
            json_str = json.dumps(test_data)
            decoded = json.loads(json_str)

            # Verify data integrity
            return decoded == test_data

        except Exception:
            return False


# =============================================================================
# Helper: Type Compatibility via JSON Round-Trip (Spec Section 10.1)
# =============================================================================


def check_type_compatibility(
    python_value: Any, java_type_signature: str, bridge: SubprocessBridge
) -> bool:
    """
    Check if a Python value can be serialized to JSON and deserialized by Java.

    This is the core type compatibility check from the simplified spec (Section 10.1).

    Args:
        python_value: The Python value to check
        java_type_signature: Expected Java type (e.g., "List<Integer>")
        bridge: Bridge instance to use for checking

    Returns:
        True if the value can round-trip successfully
    """
    try:
        # Step 1: Serialize Python value to JSON
        json_val = json.dumps(python_value)

        # Step 2: Ask Java to deserialize this JSON to the target type
        # This calls a Java helper that attempts deserialization with Jackson/Gson
        result = bridge.call_java(
            "codemorph.bridge.TypeChecker",
            "canDeserialize",
            {"json_string": json_val, "type_signature": java_type_signature},
        )

        return result.get("compatible", False)

    except Exception:
        return False
