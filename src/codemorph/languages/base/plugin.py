"""
Base language plugin interface.

All language plugins (Python, Java, future languages) must implement this interface.
This ensures consistent behavior across different source/target languages.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from codemorph.config.models import CodeFragment, FragmentType


class LanguagePlugin(ABC):
    """
    Abstract base class for language plugins.

    Each language plugin provides:
    - AST parsing capabilities
    - Fragment extraction
    - Code generation helpers
    - Naming convention transformations
    - Compilation/execution support
    """

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the language name (e.g., 'python', 'java')."""
        pass

    @property
    @abstractmethod
    def supported_versions(self) -> list[str]:
        """Return list of supported language versions."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """Return file extensions for this language (e.g., ['.py'], ['.java'])."""
        pass

    # =========================================================================
    # AST Parsing
    # =========================================================================

    @abstractmethod
    def parse_file(self, file_path: Path) -> Any:
        """
        Parse a source file into an AST.

        Args:
            file_path: Path to the source file

        Returns:
            Language-specific AST representation
        """
        pass

    @abstractmethod
    def parse_source(self, source_code: str) -> Any:
        """
        Parse source code string into an AST.

        Args:
            source_code: Source code as string

        Returns:
            Language-specific AST representation
        """
        pass

    # =========================================================================
    # Fragment Extraction
    # =========================================================================

    @abstractmethod
    def extract_fragments(self, file_path: Path, ast: Any) -> list[CodeFragment]:
        """
        Extract code fragments from an AST.

        Args:
            file_path: Path to the source file (for identification)
            ast: Parsed AST

        Returns:
            List of CodeFragment objects
        """
        pass

    @abstractmethod
    def get_fragment_dependencies(
        self, fragment: CodeFragment, ast: Any
    ) -> list[str]:
        """
        Analyze a fragment to find its dependencies.

        Args:
            fragment: The code fragment to analyze
            ast: The full file AST (for context)

        Returns:
            List of fragment IDs that this fragment depends on
        """
        pass

    @abstractmethod
    def extract_signature(self, fragment: CodeFragment) -> str | None:
        """
        Extract the signature of a function/method fragment.

        Args:
            fragment: A function or method fragment

        Returns:
            Signature string, or None if not applicable
        """
        pass

    @abstractmethod
    def extract_docstring(self, fragment: CodeFragment) -> str | None:
        """
        Extract the docstring from a fragment.

        Args:
            fragment: A code fragment

        Returns:
            Docstring content, or None if not present
        """
        pass

    # =========================================================================
    # Naming Conventions
    # =========================================================================

    @abstractmethod
    def convert_name(
        self, name: str, fragment_type: FragmentType, target_convention: str | None = None
    ) -> str:
        """
        Convert a name to follow language conventions.

        Args:
            name: Original name
            fragment_type: Type of the code element
            target_convention: Optional specific convention to use

        Returns:
            Converted name following language conventions
        """
        pass

    @abstractmethod
    def get_naming_convention(self, fragment_type: FragmentType) -> str:
        """
        Get the naming convention for a fragment type.

        Args:
            fragment_type: Type of the code element

        Returns:
            Convention name (e.g., 'camelCase', 'snake_case', 'PascalCase')
        """
        pass

    # =========================================================================
    # Code Generation Helpers
    # =========================================================================

    @abstractmethod
    def generate_stub(self, fragment: CodeFragment, signature: str) -> str:
        """
        Generate a stub implementation for a fragment.

        Used when mocking is needed.

        Args:
            fragment: The original fragment
            signature: The target signature

        Returns:
            Stub code that can be compiled
        """
        pass

    @abstractmethod
    def generate_mock(
        self, fragment: CodeFragment, bridge_call: str
    ) -> str:
        """
        Generate a mock that calls back to the source language.

        Args:
            fragment: The original fragment
            bridge_call: The bridge invocation code

        Returns:
            Mock implementation code
        """
        pass

    @abstractmethod
    def wrap_in_class(self, code: str, class_name: str) -> str:
        """
        Wrap standalone code in a class (for languages that require it).

        Args:
            code: The code to wrap
            class_name: Name for the wrapper class

        Returns:
            Code wrapped in a class, or original code if not needed
        """
        pass

    # =========================================================================
    # Compilation & Execution
    # =========================================================================

    @abstractmethod
    def compile_fragment(
        self, code: str, output_dir: Path, dependencies: list[Path] | None = None
    ) -> tuple[bool, list[str]]:
        """
        Attempt to compile a code fragment.

        Args:
            code: Source code to compile
            output_dir: Directory for compilation artifacts
            dependencies: Paths to dependency files/jars

        Returns:
            Tuple of (success: bool, errors: list[str])
        """
        pass

    @abstractmethod
    def execute_function(
        self,
        function_name: str,
        compiled_path: Path,
        inputs: dict[str, Any],
        timeout: int = 30,
    ) -> tuple[Any, str | None]:
        """
        Execute a compiled function with given inputs.

        Args:
            function_name: Name of the function to execute
            compiled_path: Path to compiled artifact
            inputs: Dictionary of input arguments
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (result, error_message)
        """
        pass

    # =========================================================================
    # Test Framework Integration
    # =========================================================================

    @abstractmethod
    def get_test_framework(self) -> str:
        """Return the test framework name for this language."""
        pass

    @abstractmethod
    def run_tests(
        self, test_dir: Path, timeout: int = 60
    ) -> tuple[bool, dict[str, Any]]:
        """
        Run tests and return results.

        Args:
            test_dir: Directory containing tests
            timeout: Execution timeout

        Returns:
            Tuple of (all_passed: bool, results: dict)
        """
        pass

    @abstractmethod
    def instrument_for_snapshots(self, source_code: str) -> str:
        """
        Instrument source code to capture execution snapshots.

        Args:
            source_code: Original source code

        Returns:
            Instrumented source code
        """
        pass

    # =========================================================================
    # Version-Specific Features
    # =========================================================================

    @abstractmethod
    def get_version_features(self, version: str) -> set[str]:
        """
        Get features available in a specific language version.

        Used for version upgrade detection.

        Args:
            version: Language version string

        Returns:
            Set of feature identifiers
        """
        pass

    @abstractmethod
    def get_deprecation_warnings(
        self, fragment: CodeFragment, source_version: str, target_version: str
    ) -> list[str]:
        """
        Get deprecation warnings when upgrading versions.

        Args:
            fragment: Code fragment to analyze
            source_version: Current version
            target_version: Target version

        Returns:
            List of deprecation warning messages
        """
        pass
