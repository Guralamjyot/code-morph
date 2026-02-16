"""
Python language plugin.

Handles Python-specific AST parsing, fragment extraction, and code generation.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

from codemorph.config.models import CodeFragment, FragmentType, ImportInfo
from codemorph.languages.base.plugin import LanguagePlugin


class PythonPlugin(LanguagePlugin):
    """Python language plugin using Python's built-in AST module."""

    def __init__(self, version: str = "3.10"):
        self.version = version

    # =========================================================================
    # Plugin Metadata
    # =========================================================================

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def supported_versions(self) -> list[str]:
        return ["2.7", "3.6", "3.7", "3.8", "3.9", "3.10", "3.11", "3.12"]

    @property
    def file_extensions(self) -> list[str]:
        return [".py"]

    # =========================================================================
    # AST Parsing
    # =========================================================================

    def parse_file(self, file_path: Path) -> ast.Module:
        """Parse a Python file into an AST."""
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        return self.parse_source(source)

    def parse_source(self, source_code: str) -> ast.Module:
        """Parse Python source code into an AST."""
        try:
            return ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Python syntax error: {e}")

    # =========================================================================
    # Fragment Extraction
    # =========================================================================

    def extract_fragments(self, file_path: Path, tree: ast.Module) -> list[CodeFragment]:
        """Extract code fragments from a Python AST."""
        fragments: list[CodeFragment] = []

        # Get source lines for extracting code
        with open(file_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()

        for node in tree.body:
            fragment = None

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Standalone function (top-level only)
                fragment = self._extract_function(node, file_path, source_lines, parent_class=None)

            elif isinstance(node, ast.ClassDef):
                # Class definition
                fragment = self._extract_class(node, file_path, source_lines)

                # Extract methods from the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_fragment = self._extract_function(
                            item, file_path, source_lines, parent_class=node.name
                        )
                        fragments.append(method_fragment)

            elif isinstance(node, ast.Assign) and hasattr(node, "lineno"):
                # Global variable assignment (already top-level since we iterate tree.body)
                fragment = self._extract_global_var(node, file_path, source_lines)

            if fragment:
                fragments.append(fragment)

        return fragments

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        source_lines: list[str],
        parent_class: str | None = None,
    ) -> CodeFragment:
        """Extract a function or method as a CodeFragment."""
        start_line = node.lineno - 1  # ast uses 1-based indexing
        end_line = node.end_lineno if node.end_lineno else start_line + 1

        source_code = "".join(source_lines[start_line:end_line])

        fragment_id = f"{file_path.stem}::{node.name}"
        if parent_class:
            fragment_id = f"{file_path.stem}::{parent_class}.{node.name}"

        fragment_type = FragmentType.METHOD if parent_class else FragmentType.FUNCTION

        return CodeFragment(
            id=fragment_id,
            name=node.name,
            fragment_type=fragment_type,
            source_file=file_path,
            start_line=start_line + 1,  # Convert back to 1-based for reporting
            end_line=end_line,
            source_code=source_code,
            parent_class=parent_class,
            docstring=ast.get_docstring(node),
        )

    def _extract_class(
        self, node: ast.ClassDef, file_path: Path, source_lines: list[str]
    ) -> CodeFragment:
        """Extract a class definition as a CodeFragment."""
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else start_line + 1

        source_code = "".join(source_lines[start_line:end_line])

        fragment_id = f"{file_path.stem}::{node.name}"

        return CodeFragment(
            id=fragment_id,
            name=node.name,
            fragment_type=FragmentType.CLASS,
            source_file=file_path,
            start_line=start_line + 1,
            end_line=end_line,
            source_code=source_code,
            docstring=ast.get_docstring(node),
        )

    def _extract_global_var(
        self, node: ast.Assign, file_path: Path, source_lines: list[str]
    ) -> CodeFragment | None:
        """Extract a global variable assignment."""
        if not node.targets:
            return None

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None

        var_name = target.id
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else start_line + 1

        source_code = "".join(source_lines[start_line:end_line])

        # Determine if it's a constant (ALL_CAPS convention)
        fragment_type = (
            FragmentType.CONSTANT if var_name.isupper() else FragmentType.GLOBAL_VAR
        )

        fragment_id = f"{file_path.stem}::{var_name}"

        return CodeFragment(
            id=fragment_id,
            name=var_name,
            fragment_type=fragment_type,
            source_file=file_path,
            start_line=start_line + 1,
            end_line=end_line,
            source_code=source_code,
        )

    def _is_top_level(self, node: ast.AST, tree: ast.Module) -> bool:
        """Check if a node is at the top level of the module."""
        return node in tree.body

    # =========================================================================
    # Dependency Analysis
    # =========================================================================

    def get_fragment_dependencies(self, fragment: CodeFragment, tree: ast.Module) -> list[str]:
        """Analyze a fragment to find its dependencies."""
        dependencies: set[str] = set()

        # Parse the fragment itself
        try:
            fragment_ast = ast.parse(fragment.source_code)
        except SyntaxError:
            return []

        # Walk the AST and find all name references
        for node in ast.walk(fragment_ast):
            if isinstance(node, ast.Name):
                # This references another symbol
                dependencies.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Method/attribute access
                if isinstance(node.value, ast.Name):
                    dependencies.add(node.value.id)

        # Convert to fragment IDs (this is simplified; real implementation needs more context)
        return list(dependencies)

    def extract_imports(
        self, tree: ast.Module, source_file: Path | None = None
    ) -> list[ImportInfo]:
        """
        Extract all import statements from the AST.

        Args:
            tree: Parsed AST module
            source_file: Path to source file (for metadata)

        Returns:
            List of ImportInfo objects representing each import
        """
        imports: list[ImportInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import module, import module as alias
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        is_from_import=False,
                        is_standard_library=self._is_stdlib(alias.name),
                        is_internal=False,  # Will be determined by analyzer
                        source_file=source_file,
                    ))

            elif isinstance(node, ast.ImportFrom):
                # Handle: from module import name, from module import name as alias
                if node.module:  # Skip relative imports like "from . import x"
                    names = [alias.name for alias in node.names]
                    imports.append(ImportInfo(
                        module=node.module,
                        names=names,
                        alias=None,
                        is_from_import=True,
                        is_standard_library=self._is_stdlib(node.module),
                        is_internal=False,  # Will be determined by analyzer
                        source_file=source_file,
                    ))

        return imports

    def _is_stdlib(self, module_name: str) -> bool:
        """
        Check if a module is part of Python's standard library.

        Args:
            module_name: Module name (may be dotted, e.g., 'os.path')

        Returns:
            True if the module is in stdlib
        """
        # Get top-level module name
        top_level = module_name.split(".")[0]

        # Common stdlib modules (not exhaustive, but covers most common ones)
        stdlib_modules = {
            "abc", "argparse", "ast", "asyncio", "base64", "bisect",
            "builtins", "collections", "contextlib", "copy", "csv",
            "dataclasses", "datetime", "decimal", "difflib", "enum",
            "functools", "gc", "glob", "gzip", "hashlib", "heapq",
            "html", "http", "inspect", "io", "itertools", "json",
            "logging", "math", "multiprocessing", "numbers", "operator",
            "os", "pathlib", "pickle", "platform", "pprint", "queue",
            "random", "re", "shutil", "signal", "socket", "sqlite3",
            "ssl", "statistics", "string", "struct", "subprocess", "sys",
            "tempfile", "threading", "time", "timeit", "traceback",
            "types", "typing", "unittest", "urllib", "uuid", "warnings",
            "weakref", "xml", "zipfile", "zlib",
        }

        return top_level in stdlib_modules

    def extract_signature(self, fragment: CodeFragment) -> str | None:
        """Extract the signature of a function/method."""
        if fragment.fragment_type not in (FragmentType.FUNCTION, FragmentType.METHOD):
            return None

        try:
            tree = ast.parse(fragment.source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Build signature string
                    args_list = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args_list.append(arg_str)

                    signature = f"def {node.name}({', '.join(args_list)})"

                    if node.returns:
                        signature += f" -> {ast.unparse(node.returns)}"

                    return signature
        except:
            pass

        return None

    def extract_docstring(self, fragment: CodeFragment) -> str | None:
        """Extract the docstring from a fragment."""
        return fragment.docstring

    # =========================================================================
    # Naming Conventions
    # =========================================================================

    def convert_name(
        self, name: str, fragment_type: FragmentType, target_convention: str | None = None
    ) -> str:
        """Convert a name to Python conventions (or target convention if specified)."""
        if target_convention == "camelCase":
            # Convert snake_case to camelCase (for Java target)
            parts = name.split("_")
            return parts[0] + "".join(p.capitalize() for p in parts[1:])
        elif target_convention == "PascalCase":
            # Convert to PascalCase
            return "".join(p.capitalize() for p in name.split("_"))
        elif target_convention == "SCREAMING_SNAKE":
            return name.upper()

        # Default: keep Python conventions
        return name

    def get_naming_convention(self, fragment_type: FragmentType) -> str:
        """Get the naming convention for a fragment type in Python."""
        if fragment_type in (FragmentType.FUNCTION, FragmentType.METHOD, FragmentType.GLOBAL_VAR):
            return "snake_case"
        elif fragment_type in (FragmentType.CLASS, FragmentType.INTERFACE):
            return "PascalCase"
        elif fragment_type == FragmentType.CONSTANT:
            return "SCREAMING_SNAKE"
        return "snake_case"

    # =========================================================================
    # Code Generation Helpers
    # =========================================================================

    def generate_stub(self, fragment: CodeFragment, signature: str) -> str:
        """Generate a stub implementation for a fragment."""
        if fragment.fragment_type in (FragmentType.FUNCTION, FragmentType.METHOD):
            return f"{signature}:\n    pass\n"
        elif fragment.fragment_type == FragmentType.CLASS:
            return f"class {fragment.name}:\n    pass\n"
        else:
            return f"{fragment.name} = None\n"

    def generate_mock(self, fragment: CodeFragment, bridge_call: str) -> str:
        """Generate a mock that calls back to the source language."""
        if fragment.fragment_type in (FragmentType.FUNCTION, FragmentType.METHOD):
            sig = self.extract_signature(fragment) or f"def {fragment.name}(*args, **kwargs)"
            return f"{sig}:\n    return {bridge_call}\n"
        else:
            return f"{fragment.name} = {bridge_call}\n"

    def wrap_in_class(self, code: str, class_name: str) -> str:
        """Python doesn't require wrapping in a class, return as-is."""
        return code

    # =========================================================================
    # Compilation & Execution
    # =========================================================================

    def compile_fragment(
        self, code: str, output_dir: Path, dependencies: list[Path] | None = None
    ) -> tuple[bool, list[str]]:
        """
        'Compile' Python code (syntax check + pyc generation).

        Python is interpreted, but we can check for syntax errors.

        Args:
            code: Python source code to check
            output_dir: Directory to write bytecode files
            dependencies: Optional list of dependency directories for sys.path

        Returns:
            Tuple of (success, error_messages)
        """
        errors: list[str] = []

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Write code to temporary file
            temp_file = output_dir / "temp_check.py"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            # First, do AST parse for basic syntax check
            try:
                ast.parse(code)
            except SyntaxError as e:
                error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
                if e.text:
                    error_msg += f"\n  {e.text.strip()}"
                    if e.offset:
                        error_msg += f"\n  {' ' * (e.offset - 1)}^"
                errors.append(error_msg)
                temp_file.unlink(missing_ok=True)
                return (False, errors)

            # Then compile to bytecode
            import py_compile

            try:
                py_compile.compile(str(temp_file), doraise=True)

                # Clean up temp files
                temp_file.unlink(missing_ok=True)
                pyc_file = temp_file.with_suffix(".pyc")
                if pyc_file.exists():
                    pyc_file.unlink()

                return (True, [])

            except py_compile.PyCompileError as e:
                error_msg = f"Compilation error: {e.msg}"
                errors.append(error_msg)
                temp_file.unlink(missing_ok=True)
                return (False, errors)

        except Exception as e:
            errors.append(f"Unexpected error during compilation: {str(e)}")
            return (False, errors)

    def execute_function(
        self,
        function_name: str,
        compiled_path: Path,
        inputs: dict[str, Any],
        timeout: int = 30,
    ) -> tuple[Any, str | None]:
        """Execute a Python function with given inputs."""
        try:
            # Import the module
            import importlib.util

            spec = importlib.util.spec_from_file_location("temp_module", compiled_path)
            if spec is None or spec.loader is None:
                return (None, "Failed to load module")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the function
            func = getattr(module, function_name, None)
            if func is None:
                return (None, f"Function '{function_name}' not found")

            # Execute with inputs
            result = func(**inputs)
            return (result, None)

        except Exception as e:
            return (None, str(e))

    # =========================================================================
    # Test Framework Integration
    # =========================================================================

    def get_test_framework(self) -> str:
        return "pytest"

    def run_tests(self, test_dir: Path, timeout: int = 60) -> tuple[bool, dict[str, Any]]:
        """Run pytest on the test directory."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            all_passed = result.returncode == 0
            return (all_passed, {"stdout": result.stdout, "stderr": result.stderr})

        except subprocess.TimeoutExpired:
            return (False, {"error": "Test execution timed out"})
        except Exception as e:
            return (False, {"error": str(e)})

    def instrument_for_snapshots(self, source_code: str) -> str:
        """
        Instrument source code to capture execution snapshots.

        This adds decorators to functions to log inputs/outputs.
        """
        # TODO: Implement instrumentation logic
        # For now, return unchanged
        return source_code

    # =========================================================================
    # Version-Specific Features
    # =========================================================================

    def get_version_features(self, version: str) -> set[str]:
        """Get features available in a specific Python version."""
        features = {
            "2.7": {"print_statement", "unicode_literals", "old_division"},
            "3.6": {"f_strings", "type_hints", "async_await"},
            "3.7": {"dataclasses", "postponed_annotations"},
            "3.8": {"walrus_operator", "positional_only_params"},
            "3.9": {"dict_union_operator", "generic_types"},
            "3.10": {"match_statement", "union_types"},
            "3.11": {"exception_groups", "tomllib"},
            "3.12": {"type_parameter_syntax"},
        }

        # Accumulate features up to the specified version
        available = set()
        version_nums = sorted(features.keys())

        for v in version_nums:
            available.update(features[v])
            if v == version:
                break

        return available

    def get_deprecation_warnings(
        self, fragment: CodeFragment, source_version: str, target_version: str
    ) -> list[str]:
        """Get deprecation warnings when upgrading Python versions."""
        warnings = []

        source_features = self.get_version_features(source_version)
        target_features = self.get_version_features(target_version)

        # Check for features that are deprecated/removed
        if "print_statement" in source_features and "print_statement" not in target_features:
            if "print " in fragment.source_code:
                warnings.append("print statement deprecated; use print() function")

        if "old_division" in source_features and "old_division" not in target_features:
            warnings.append("Division behavior changed; / is true division, use // for floor division")

        return warnings
