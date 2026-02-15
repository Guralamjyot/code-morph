"""
Python bridge runner for subprocess-based cross-language calls.

This script is invoked as:
    python -m codemorph.bridges.python_runner

It:
1. Reads JSON from stdin
2. Imports the specified module
3. Calls the specified function with arguments
4. Serializes the result to JSON
5. Writes to stdout

Used by Java code to call back to Python functions (e.g., for mocking).
"""

import importlib
import json
import sys
import traceback
from typing import Any


def run_function(module_path: str, function_name: str, args: dict[str, Any]) -> Any:
    """
    Import a module and call a function with the given arguments.

    Args:
        module_path: Python module path (e.g., "myapp.utils")
        function_name: Function name to call
        args: Dictionary of keyword arguments

    Returns:
        The function result

    Raises:
        ImportError: If module can't be imported
        AttributeError: If function doesn't exist
        Exception: If function execution fails
    """
    # Import the module
    module = importlib.import_module(module_path)

    # Get the function
    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in module '{module_path}'"
        )

    func = getattr(module, function_name)

    # Call the function with arguments
    # Convert dict to kwargs
    result = func(**args)

    return result


def main():
    """Main entry point for the Python runner."""
    try:
        # Read JSON from stdin
        input_data = sys.stdin.read()

        if not input_data.strip():
            print(
                json.dumps({"error": "No input received"}), file=sys.stderr, flush=True
            )
            sys.exit(1)

        # Parse input JSON
        try:
            request = json.loads(input_data)
        except json.JSONDecodeError as e:
            print(
                json.dumps({"error": f"Invalid JSON input: {e}"}),
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        # Extract parameters
        module_path = request.get("module")
        function_name = request.get("function")
        args = request.get("args", {})

        if not module_path or not function_name:
            print(
                json.dumps({"error": "Missing 'module' or 'function' in request"}),
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        # Execute the function
        result = run_function(module_path, function_name, args)

        # Serialize result to JSON
        output = json.dumps({"success": True, "result": result})

        # Write to stdout
        print(output, flush=True)
        sys.exit(0)

    except ImportError as e:
        error_output = json.dumps(
            {
                "success": False,
                "error": f"ImportError: {e}",
                "traceback": traceback.format_exc(),
            }
        )
        print(error_output, file=sys.stderr, flush=True)
        sys.exit(1)

    except AttributeError as e:
        error_output = json.dumps(
            {
                "success": False,
                "error": f"AttributeError: {e}",
                "traceback": traceback.format_exc(),
            }
        )
        print(error_output, file=sys.stderr, flush=True)
        sys.exit(1)

    except Exception as e:
        error_output = json.dumps(
            {
                "success": False,
                "error": f"Execution error: {e}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        print(error_output, file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
