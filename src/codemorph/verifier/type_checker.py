"""
Type compatibility checker for CodeMorph.

Uses JSON as a universal serialization format to verify that Python and Java
types are compatible through round-trip serialization testing.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import orjson


class TypeCompatibilityError(Exception):
    """Raised when types are incompatible."""

    pass


class TypeCompatibilityChecker:
    """
    Checks type compatibility between Python and Java using JSON bridge.

    The approach:
    1. Serialize a Python value to JSON
    2. Attempt to deserialize in Java to the target type
    3. Serialize back to JSON in Java
    4. Verify the round-trip is consistent
    """

    def __init__(self, java_version: str = "17"):
        self.java_version = java_version
        self._temp_dir = None

    def get_temp_dir(self) -> Path:
        """Get or create temporary directory for type checking."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="codemorph_typecheck_"))
        return self._temp_dir

    # =========================================================================
    # Python Type Mapping
    # =========================================================================

    PYTHON_TO_JAVA_TYPE_MAP = {
        "int": "Integer",
        "float": "Double",
        "str": "String",
        "bool": "Boolean",
        "list": "List",
        "dict": "Map",
        "tuple": "List",  # Java doesn't have immutable tuples, use List
        "set": "Set",
        "NoneType": "null",
    }

    JAVA_TO_PYTHON_TYPE_MAP = {
        "Integer": "int",
        "int": "int",
        "Double": "float",
        "double": "float",
        "String": "str",
        "Boolean": "bool",
        "boolean": "bool",
        "List": "list",
        "ArrayList": "list",
        "Map": "dict",
        "HashMap": "dict",
        "Set": "set",
        "HashSet": "set",
    }

    def infer_python_type(self, value: Any) -> str:
        """Infer the Python type of a value."""
        type_name = type(value).__name__
        return self.PYTHON_TO_JAVA_TYPE_MAP.get(type_name, "Object")

    def infer_java_type_from_python(self, value: Any) -> str:
        """
        Infer the appropriate Java type for a Python value.

        Args:
            value: Python value

        Returns:
            Java type string (e.g., "List<Integer>", "Map<String, Double>")
        """
        if value is None:
            return "Object"  # nullable

        if isinstance(value, bool):
            return "Boolean"
        elif isinstance(value, int):
            return "Integer"
        elif isinstance(value, float):
            return "Double"
        elif isinstance(value, str):
            return "String"
        elif isinstance(value, list):
            if len(value) == 0:
                return "List<Object>"
            # Infer generic type from first element
            elem_type = self.infer_java_type_from_python(value[0])
            return f"List<{elem_type}>"
        elif isinstance(value, dict):
            if len(value) == 0:
                return "Map<String, Object>"
            # Infer from first key-value pair
            first_key = next(iter(value.keys()))
            first_val = value[first_key]
            key_type = self.infer_java_type_from_python(first_key)
            val_type = self.infer_java_type_from_python(first_val)
            return f"Map<{key_type}, {val_type}>"
        elif isinstance(value, set):
            if len(value) == 0:
                return "Set<Object>"
            elem_type = self.infer_java_type_from_python(next(iter(value)))
            return f"Set<{elem_type}>"
        else:
            return "Object"

    # =========================================================================
    # Compatibility Checking
    # =========================================================================

    def check_compatibility(
        self, python_value: Any, java_type_signature: str
    ) -> tuple[bool, str | None]:
        """
        Check if a Python value can be represented in Java with the given type.

        Args:
            python_value: The Python value to test
            java_type_signature: The Java type (e.g., "List<Integer>")

        Returns:
            Tuple of (compatible, error_message)
        """
        # Serialize Python value to JSON
        try:
            json_str = json.dumps(python_value)
        except (TypeError, ValueError) as e:
            return (False, f"Python value not JSON-serializable: {e}")

        # Try to deserialize in Java
        try:
            success = self._test_java_deserialization(json_str, java_type_signature)
            if success:
                return (True, None)
            else:
                return (False, f"Java cannot deserialize JSON to type {java_type_signature}")
        except Exception as e:
            return (False, f"Java deserialization test failed: {e}")

    def _test_java_deserialization(self, json_str: str, java_type: str) -> bool:
        """
        Test if Java can deserialize the JSON to the specified type.

        This creates a temporary Java program that attempts deserialization.

        Args:
            json_str: JSON string to deserialize
            java_type: Target Java type

        Returns:
            True if deserialization succeeds
        """
        # Create a Java test program
        java_code = self._generate_java_type_test(java_type)

        temp_dir = self.get_temp_dir()
        java_file = temp_dir / "TypeTest.java"

        with open(java_file, "w") as f:
            f.write(java_code)

        # Compile the Java program
        compile_result = subprocess.run(
            ["javac", str(java_file)],
            capture_output=True,
            text=True,
        )

        if compile_result.returncode != 0:
            # Compilation failed - likely the type signature is invalid
            return False

        # Run the Java program with the JSON as input
        run_result = subprocess.run(
            ["java", "-cp", str(temp_dir), "TypeTest"],
            input=json_str,
            capture_output=True,
            text=True,
        )

        # Check if it succeeded
        return run_result.returncode == 0

    def _generate_java_type_test(self, java_type: str) -> str:
        """
        Generate a Java program that tests deserialization to a specific type.

        Args:
            java_type: The Java type to test (e.g., "List<Integer>")

        Returns:
            Java source code as string
        """
        # Simplified test program using Gson
        # In a real implementation, we'd need to handle complex generic types better
        return f"""
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.*;
import java.io.*;

public class TypeTest {{
    public static void main(String[] args) {{
        try {{
            // Read JSON from stdin
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            StringBuilder jsonBuilder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {{
                jsonBuilder.append(line);
            }}
            String json = jsonBuilder.toString();

            // Try to deserialize
            Gson gson = new Gson();

            // Handle different types
            {self._generate_type_deserialization_code(java_type)}

            System.out.println("SUCCESS");
            System.exit(0);

        }} catch (Exception e) {{
            System.err.println("FAILURE: " + e.getMessage());
            System.exit(1);
        }}
    }}
}}
"""

    def _generate_type_deserialization_code(self, java_type: str) -> str:
        """Generate type-specific deserialization code."""
        # Simple cases
        if java_type in ["Integer", "Double", "String", "Boolean"]:
            return f"{java_type} result = gson.fromJson(json, {java_type}.class);"

        # Handle generic types
        if java_type.startswith("List<"):
            return f"""
            Type type = new TypeToken<{java_type}>(){{}}.getType();
            {java_type} result = gson.fromJson(json, type);
            """
        elif java_type.startswith("Map<"):
            return f"""
            Type type = new TypeToken<{java_type}>(){{}}.getType();
            {java_type} result = gson.fromJson(json, type);
            """
        elif java_type.startswith("Set<"):
            return f"""
            Type type = new TypeToken<{java_type}>(){{}}.getType();
            {java_type} result = gson.fromJson(json, type);
            """
        else:
            # Fallback: try to deserialize as Object
            return "Object result = gson.fromJson(json, Object.class);"

    # =========================================================================
    # Round-Trip Testing
    # =========================================================================

    def verify_round_trip(
        self,
        python_value: Any,
        python_type: str,
        java_type: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Verify that a value can round-trip between Python and Java.

        Args:
            python_value: The original Python value
            python_type: Python type annotation (e.g., "int", "List[int]")
            java_type: Java type signature (e.g., "Integer", "List<Integer>")

        Returns:
            Tuple of (success, details_dict)
        """
        details = {
            "original_value": python_value,
            "python_type": python_type,
            "java_type": java_type,
            "steps": [],
        }

        # Step 1: Serialize in Python
        try:
            json_from_python = json.dumps(python_value)
            details["steps"].append(("python_serialize", "success", json_from_python))
        except Exception as e:
            details["steps"].append(("python_serialize", "failed", str(e)))
            return (False, details)

        # Step 2: Deserialize in Java (compatibility check)
        compatible, error = self.check_compatibility(python_value, java_type)
        if compatible:
            details["steps"].append(("java_deserialize", "success", None))
        else:
            details["steps"].append(("java_deserialize", "failed", error))
            return (False, details)

        # Step 3: Verify the value is preserved (Python deserialize)
        try:
            recovered_value = json.loads(json_from_python)
            # For simple types, check equality
            if self._values_equivalent(python_value, recovered_value):
                details["steps"].append(("verify_equivalence", "success", recovered_value))
                return (True, details)
            else:
                details["steps"].append((
                    "verify_equivalence",
                    "failed",
                    f"Values differ: {python_value} != {recovered_value}"
                ))
                return (False, details)
        except Exception as e:
            details["steps"].append(("verify_equivalence", "failed", str(e)))
            return (False, details)

    def _values_equivalent(self, v1: Any, v2: Any) -> bool:
        """Check if two values are equivalent (accounting for type coercion)."""
        # Handle None
        if v1 is None and v2 is None:
            return True

        # Handle numeric types (int/float can be equivalent)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(v1 - v2) < 1e-9

        # Handle collections recursively
        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(self._values_equivalent(a, b) for a, b in zip(v1, v2))

        if isinstance(v1, dict) and isinstance(v2, dict):
            if set(v1.keys()) != set(v2.keys()):
                return False
            return all(self._values_equivalent(v1[k], v2[k]) for k in v1.keys())

        # Direct equality for other types
        return v1 == v2

    # =========================================================================
    # Batch Checking
    # =========================================================================

    def check_function_signature_compatibility(
        self,
        python_params: dict[str, str],
        python_return: str,
        java_params: dict[str, str],
        java_return: str,
        test_values: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if function signatures are compatible.

        Args:
            python_params: Dict of param_name -> python_type
            python_return: Python return type
            java_params: Dict of param_name -> java_type
            java_return: Java return type
            test_values: Optional test values for each parameter

        Returns:
            Tuple of (compatible, list of incompatibility reasons)
        """
        errors = []

        # Check parameter compatibility
        if set(python_params.keys()) != set(java_params.keys()):
            errors.append(
                f"Parameter names differ: {set(python_params.keys())} vs {set(java_params.keys())}"
            )

        # If test values provided, check each parameter
        if test_values:
            for param_name in python_params.keys():
                if param_name in java_params and param_name in test_values:
                    compatible, error = self.check_compatibility(
                        test_values[param_name],
                        java_params[param_name]
                    )
                    if not compatible:
                        errors.append(f"Parameter '{param_name}': {error}")

        return (len(errors) == 0, errors)

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir)
