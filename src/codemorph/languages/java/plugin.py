"""
Java language plugin.

Handles Java-specific AST parsing, fragment extraction, and code generation.
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from codemorph.config.models import CodeFragment, FragmentType, ImportInfo
from codemorph.languages.base.plugin import LanguagePlugin


class JavaPlugin(LanguagePlugin):
    """Java language plugin using tree-sitter for parsing."""

    def __init__(self, version: str = "17"):
        self.version = version
        self._parser = None

    # =========================================================================
    # Plugin Metadata
    # =========================================================================

    @property
    def language_name(self) -> str:
        return "java"

    @property
    def supported_versions(self) -> list[str]:
        return ["11", "17", "21"]

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    # =========================================================================
    # AST Parsing (using tree-sitter)
    # =========================================================================

    def _get_parser(self):
        """Lazy initialization of tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_java as tsjava
                from tree_sitter import Language, Parser

                JAVA_LANGUAGE = Language(tsjava.language())
                self._parser = Parser(JAVA_LANGUAGE)
            except ImportError:
                raise RuntimeError(
                    "tree-sitter-java not installed. Run: pip install tree-sitter-java"
                )
        return self._parser

    def parse_file(self, file_path: Path) -> Any:
        """Parse a Java file into a tree-sitter AST."""
        with open(file_path, "rb") as f:
            source = f.read()
        return self.parse_source(source.decode("utf-8"))

    def parse_source(self, source_code: str) -> Any:
        """Parse Java source code into a tree-sitter AST."""
        parser = self._get_parser()
        tree = parser.parse(bytes(source_code, "utf-8"))
        return tree

    # =========================================================================
    # Fragment Extraction
    # =========================================================================

    def extract_fragments(self, file_path: Path, tree: Any) -> list[CodeFragment]:
        """Extract code fragments from a Java tree-sitter AST."""
        fragments: list[CodeFragment] = []

        # Get source code
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
            source_lines = source_code.split("\n")

        root_node = tree.root_node

        # Find package and class declarations
        current_class = None

        for node in self._traverse_tree(root_node):
            if node.type == "class_declaration":
                # Extract class
                class_fragment = self._extract_class(node, file_path, source_lines)
                fragments.append(class_fragment)
                current_class = class_fragment.name

            elif node.type == "method_declaration":
                # Extract method
                method_fragment = self._extract_method(node, file_path, source_lines, current_class)
                fragments.append(method_fragment)

            elif node.type == "field_declaration":
                # Extract field (global variable in class context)
                field_fragment = self._extract_field(node, file_path, source_lines, current_class)
                if field_fragment:
                    fragments.append(field_fragment)

            elif node.type == "interface_declaration":
                # Extract interface
                interface_fragment = self._extract_interface(node, file_path, source_lines)
                fragments.append(interface_fragment)

        return fragments

    def _traverse_tree(self, node: Any):
        """Traverse tree-sitter tree depth-first."""
        yield node
        for child in node.children:
            yield from self._traverse_tree(child)

    def _extract_class(self, node: Any, file_path: Path, source_lines: list[str]) -> CodeFragment:
        """Extract a class declaration as a CodeFragment."""
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        source_code = "\n".join(source_lines[start_line : end_line + 1])

        # Extract class name
        class_name = None
        for child in node.children:
            if child.type == "identifier":
                class_name = child.text.decode("utf-8")
                break

        if not class_name:
            class_name = "UnknownClass"

        fragment_id = f"{file_path.stem}::{class_name}"

        return CodeFragment(
            id=fragment_id,
            name=class_name,
            fragment_type=FragmentType.CLASS,
            source_file=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            source_code=source_code,
        )

    def _extract_param_types(self, node: Any) -> list[str]:
        """Extract simplified parameter type names from a method's formal_parameters.

        Strips generics (e.g., List<String> -> List) and returns simplified type
        names. Used to discriminate overloaded methods.
        """
        param_types = []
        for child in node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "formal_parameter":
                        # The type is usually the first non-modifier child
                        for pchild in param.children:
                            if pchild.type in (
                                "type_identifier",
                                "integral_type",
                                "floating_point_type",
                                "boolean_type",
                                "void_type",
                                "generic_type",
                                "array_type",
                            ):
                                type_name = pchild.text.decode("utf-8")
                                # Strip generics: List<String> -> List
                                type_name = re.sub(r"<[^>]*>", "", type_name)
                                # Strip array brackets
                                type_name = type_name.replace("[]", "")
                                param_types.append(type_name.strip())
                                break
        return param_types

    def _extract_method(
        self, node: Any, file_path: Path, source_lines: list[str], parent_class: str | None
    ) -> CodeFragment:
        """Extract a method declaration as a CodeFragment."""
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        source_code = "\n".join(source_lines[start_line : end_line + 1])

        # Extract method name
        method_name = None
        for child in node.children:
            if child.type == "identifier":
                method_name = child.text.decode("utf-8")
                break

        if not method_name:
            method_name = "unknownMethod"

        fragment_id = f"{file_path.stem}::{method_name}"
        if parent_class:
            fragment_id = f"{file_path.stem}::{parent_class}.{method_name}"

        # Append overload discriminator if this method has parameters
        # $TypeA_TypeB suffix for overload-aware IDs
        param_types = self._extract_param_types(node)
        if param_types:
            type_suffix = "_".join(param_types)
            fragment_id = f"{fragment_id}${type_suffix}"

        return CodeFragment(
            id=fragment_id,
            name=method_name,
            fragment_type=FragmentType.METHOD,
            source_file=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            source_code=source_code,
            parent_class=parent_class,
        )

    def _extract_field(
        self, node: Any, file_path: Path, source_lines: list[str], parent_class: str | None
    ) -> CodeFragment | None:
        """Extract a field declaration."""
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        source_code = "\n".join(source_lines[start_line : end_line + 1])

        # Extract field name (simplified)
        field_name = None
        for child in node.children:
            if child.type == "variable_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        field_name = subchild.text.decode("utf-8")
                        break

        if not field_name:
            return None

        # Determine if it's a constant (static final)
        is_constant = "static final" in source_code or "final static" in source_code
        fragment_type = FragmentType.CONSTANT if is_constant else FragmentType.GLOBAL_VAR

        fragment_id = f"{file_path.stem}::{field_name}"
        if parent_class:
            fragment_id = f"{file_path.stem}::{parent_class}.{field_name}"

        return CodeFragment(
            id=fragment_id,
            name=field_name,
            fragment_type=fragment_type,
            source_file=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            source_code=source_code,
            parent_class=parent_class,
        )

    def _extract_interface(
        self, node: Any, file_path: Path, source_lines: list[str]
    ) -> CodeFragment:
        """Extract an interface declaration."""
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        source_code = "\n".join(source_lines[start_line : end_line + 1])

        # Extract interface name
        interface_name = None
        for child in node.children:
            if child.type == "identifier":
                interface_name = child.text.decode("utf-8")
                break

        if not interface_name:
            interface_name = "UnknownInterface"

        fragment_id = f"{file_path.stem}::{interface_name}"

        return CodeFragment(
            id=fragment_id,
            name=interface_name,
            fragment_type=FragmentType.INTERFACE,
            source_file=file_path,
            start_line=start_line + 1,
            end_line=end_line + 1,
            source_code=source_code,
        )

    # =========================================================================
    # Dependency Analysis
    # =========================================================================

    def get_fragment_dependencies(self, fragment: CodeFragment, tree: Any) -> list[str]:
        """Analyze a fragment to find its dependencies."""
        dependencies: set[str] = set()

        # Simple regex-based dependency extraction
        # Look for method calls and class references
        code = fragment.source_code

        # Find method calls: someMethod(...)
        method_calls = re.findall(r"(\w+)\s*\(", code)
        dependencies.update(method_calls)

        # Find class instantiations: new SomeClass(...)
        class_instantiations = re.findall(r"new\s+(\w+)\s*\(", code)
        dependencies.update(class_instantiations)

        # Find extends clause: class Pair extends Tuple
        extends_matches = re.findall(r"\bextends\s+(\w+)", code)
        dependencies.update(extends_matches)

        # Find implements clause: class Pair implements IValue0<A>, IValue1<B>
        implements_match = re.search(r"\bimplements\s+([\w\s,<>?]+?)(?:\s*\{)", code)
        if implements_match:
            impl_str = implements_match.group(1)
            # Extract interface names (strip generic params)
            impl_names = re.findall(r"(\w+)\s*(?:<|,|$)", impl_str)
            dependencies.update(impl_names)

        return list(dependencies)

    def extract_imports(
        self, tree: Any, source_file: Path | None = None
    ) -> list[ImportInfo]:
        """
        Extract all import statements from the Java AST.

        Args:
            tree: Parsed tree-sitter AST
            source_file: Path to source file (for metadata)

        Returns:
            List of ImportInfo objects representing each import
        """
        imports: list[ImportInfo] = []

        root_node = tree.root_node

        for node in self._traverse_tree(root_node):
            if node.type == "import_declaration":
                import_info = self._parse_import_node(node, source_file)
                if import_info:
                    imports.append(import_info)

        return imports

    def _parse_import_node(
        self, node: Any, source_file: Path | None
    ) -> ImportInfo | None:
        """
        Parse a single import declaration node.

        Handles:
        - import java.util.List;
        - import java.util.*;
        - import static java.lang.Math.PI;
        """
        # Get the full import text
        import_text = node.text.decode("utf-8").strip()

        # Check if it's a static import
        is_static = "static" in import_text

        # Remove 'import', 'static', and ';'
        import_path = import_text.replace("import", "").replace("static", "").replace(";", "").strip()

        if not import_path:
            return None

        # Handle wildcard imports: java.util.*
        is_wildcard = import_path.endswith(".*")
        if is_wildcard:
            module = import_path[:-2]  # Remove .*
            names = ["*"]
        else:
            # Regular import: java.util.List or java.lang.Math.PI (static)
            parts = import_path.rsplit(".", 1)
            if len(parts) == 2:
                module = parts[0]
                names = [parts[1]]
            else:
                module = import_path
                names = []

        # Determine if it's a standard library import
        is_stdlib = self._is_stdlib(module)

        return ImportInfo(
            module=module,
            names=names,
            alias=None,  # Java doesn't have import aliases
            is_from_import=True,  # Java imports always specify what they're importing
            is_standard_library=is_stdlib,
            is_internal=False,  # Will be determined by analyzer
            source_file=source_file,
        )

    def _is_stdlib(self, module_name: str) -> bool:
        """
        Check if a module is part of Java's standard library.

        Args:
            module_name: Module name (e.g., 'java.util', 'javax.swing')

        Returns:
            True if the module is in the Java standard library
        """
        # Java standard library packages
        stdlib_prefixes = {
            "java.",
            "javax.",
            "jdk.",
            "sun.",
            "com.sun.",
        }

        for prefix in stdlib_prefixes:
            if module_name.startswith(prefix):
                return True

        return False

    def extract_signature(self, fragment: CodeFragment) -> str | None:
        """Extract the signature of a method."""
        if fragment.fragment_type != FragmentType.METHOD:
            return None

        # Extract signature (everything up to the opening brace)
        code = fragment.source_code
        match = re.search(r"(.*?)\s*\{", code, re.DOTALL)
        if match:
            signature = match.group(1).strip()
            # Remove modifiers and annotations for cleaner signature
            # (Keep this simple for now)
            return signature

        return None

    def extract_docstring(self, fragment: CodeFragment) -> str | None:
        """Extract JavaDoc from a fragment."""
        code = fragment.source_code
        match = re.search(r"/\*\*(.*?)\*/", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # =========================================================================
    # Naming Conventions
    # =========================================================================

    def convert_name(
        self, name: str, fragment_type: FragmentType, target_convention: str | None = None
    ) -> str:
        """Convert a name to Java conventions (or target convention if specified)."""
        if target_convention == "snake_case":
            # Convert camelCase to snake_case (for Python target)
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        # Default: keep Java conventions
        return name

    def get_naming_convention(self, fragment_type: FragmentType) -> str:
        """Get the naming convention for a fragment type in Java."""
        if fragment_type in (FragmentType.METHOD, FragmentType.GLOBAL_VAR):
            return "camelCase"
        elif fragment_type in (FragmentType.CLASS, FragmentType.INTERFACE):
            return "PascalCase"
        elif fragment_type == FragmentType.CONSTANT:
            return "SCREAMING_SNAKE"
        return "camelCase"

    # =========================================================================
    # Code Generation Helpers
    # =========================================================================

    def generate_stub(self, fragment: CodeFragment, signature: str) -> str:
        """Generate a stub implementation for a fragment."""
        if fragment.fragment_type == FragmentType.METHOD:
            # Determine return type from signature
            if "void" in signature:
                return f"{signature} {{\n    // TODO: Implement\n}}\n"
            else:
                return f"{signature} {{\n    return null; // TODO: Implement\n}}\n"
        elif fragment.fragment_type == FragmentType.CLASS:
            return f"public class {fragment.name} {{\n    // TODO: Implement\n}}\n"
        else:
            return f"private Object {fragment.name} = null;\n"

    def generate_mock(self, fragment: CodeFragment, bridge_call: str) -> str:
        """Generate a mock that calls back to Python."""
        if fragment.fragment_type == FragmentType.METHOD:
            sig = self.extract_signature(fragment) or f"public Object {fragment.name}()"
            return f"{sig} {{\n    return {bridge_call};\n}}\n"
        else:
            return f"private Object {fragment.name} = {bridge_call};\n"

    def wrap_in_class(self, code: str, class_name: str) -> str:
        """Wrap code in a Java class (required for Java)."""
        return f"public class {class_name} {{\n{code}\n}}\n"

    # =========================================================================
    # Compilation & Execution
    # =========================================================================

    def compile_fragment(
        self, code: str, output_dir: Path, dependencies: list[Path] | None = None
    ) -> tuple[bool, list[str]]:
        """
        Compile Java code using javac.

        Args:
            code: Java source code to compile
            output_dir: Directory to write compiled .class files
            dependencies: Optional list of dependency directories for classpath

        Returns:
            Tuple of (success, error_messages)
        """
        errors: list[str] = []

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract class name from code (class, interface, or enum)
            class_name_match = re.search(r"(?:public\s+)?class\s+(\w+)", code)
            if not class_name_match:
                class_name_match = re.search(r"(?:public\s+)?interface\s+(\w+)", code)
            if not class_name_match:
                class_name_match = re.search(r"(?:public\s+)?enum\s+(\w+)", code)

            if not class_name_match:
                return (False, ["Could not find class, interface, or enum declaration in code"])

            class_name = class_name_match.group(1)

            # Check if code has package declaration
            package_match = re.search(r"package\s+([\w.]+)\s*;", code)
            if package_match:
                package_name = package_match.group(1)
                # Create package directory structure
                package_dir = output_dir / package_name.replace(".", "/")
                package_dir.mkdir(parents=True, exist_ok=True)
                temp_file = package_dir / f"{class_name}.java"
            else:
                temp_file = output_dir / f"{class_name}.java"

            # Write code to file
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Build classpath
            classpath_parts = [str(output_dir)]
            if dependencies:
                classpath_parts.extend(str(d) for d in dependencies)
            classpath = ":".join(classpath_parts)

            # Compile with javac
            cmd = ["javac", "-cp", classpath, "-d", str(output_dir), str(temp_file)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return (True, [])
            else:
                # Parse compilation errors for better readability
                error_lines = result.stderr.strip().split("\n")
                parsed_errors = self._parse_javac_errors(error_lines, temp_file)
                return (False, parsed_errors)

        except FileNotFoundError:
            return (False, ["javac not found. Please install Java JDK."])
        except subprocess.TimeoutExpired:
            return (False, ["Compilation timed out after 30 seconds"])
        except Exception as e:
            return (False, [f"Compilation error: {str(e)}"])

    def _parse_javac_errors(self, error_lines: list[str], source_file: Path) -> list[str]:
        """
        Parse javac error output into readable error messages.

        Args:
            error_lines: Lines from javac stderr
            source_file: Path to the source file being compiled

        Returns:
            List of formatted error messages
        """
        errors = []
        current_error = []

        for line in error_lines:
            if not line.strip():
                if current_error:
                    errors.append(" ".join(current_error))
                    current_error = []
                continue

            # Remove absolute path from error messages
            line = line.replace(str(source_file), source_file.name)

            # Check if this is an error location line (contains file:line:)
            if re.match(r".+\.java:\d+:", line):
                if current_error:
                    errors.append(" ".join(current_error))
                current_error = [line]
            else:
                current_error.append(line)

        if current_error:
            errors.append(" ".join(current_error))

        # If no structured errors found, return raw stderr
        if not errors and error_lines:
            return [" ".join(error_lines)]

        return errors[:10]  # Limit to first 10 errors

    def execute_function(
        self,
        function_name: str,
        compiled_path: Path,
        inputs: dict[str, Any],
        timeout: int = 30,
    ) -> tuple[Any, str | None]:
        """Execute a Java method using reflection (via temporary harness)."""
        # This is complex - would need a Java harness
        # For POC, return not implemented
        return (None, "Java execution not yet implemented")

    # =========================================================================
    # Test Framework Integration
    # =========================================================================

    def get_test_framework(self) -> str:
        return "junit"

    def run_tests(self, test_dir: Path, timeout: int = 60) -> tuple[bool, dict[str, Any]]:
        """Run JUnit tests."""
        # This requires Maven/Gradle integration
        # For POC, return not implemented
        return (False, {"error": "Java test execution not yet implemented"})

    def instrument_for_snapshots(self, source_code: str) -> str:
        """Instrument Java code to capture execution snapshots."""
        # TODO: Implement instrumentation
        return source_code

    # =========================================================================
    # Version-Specific Features
    # =========================================================================

    def get_version_features(self, version: str) -> set[str]:
        """Get features available in a specific Java version."""
        features = {
            "11": {"var_keyword", "lambda_expressions", "streams", "optional"},
            "17": {"sealed_classes", "pattern_matching_instanceof", "records", "text_blocks"},
            "21": {"virtual_threads", "pattern_matching_switch", "record_patterns"},
        }

        available = set()
        version_nums = sorted(features.keys(), key=int)

        for v in version_nums:
            available.update(features[v])
            if v == version:
                break

        return available

    def get_deprecation_warnings(
        self, fragment: CodeFragment, source_version: str, target_version: str
    ) -> list[str]:
        """Get deprecation warnings when upgrading Java versions."""
        warnings = []

        # Check for removed/deprecated features
        # (Java is generally backward compatible, but newer features are preferred)

        if int(target_version) >= 17 and int(source_version) < 17:
            warnings.append("Consider using records instead of simple data classes")
            warnings.append("Consider using sealed classes for restricted hierarchies")

        return warnings
