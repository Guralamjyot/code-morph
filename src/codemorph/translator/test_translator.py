"""
Test Translation Service for CodeMorph.

Translates source test files to target language using LLM.
E.g., converts pytest tests to JUnit 5 tests.
"""

from pathlib import Path

from codemorph.config.models import (
    CodeMorphConfig,
    LanguageType,
    TranslatedFragment,
)
from codemorph.state.symbol_registry import SymbolRegistry


class TestTranslator:
    """
    Translates source test files to target language using LLM.

    Uses the symbol registry to ensure correct naming in generated tests.

    Usage:
        translator = TestTranslator(config, llm_client, symbol_registry)

        java_test_code = translator.translate_test_file(
            source_test=Path("test_calculator.py"),
            translated_functions=translated_fragments
        )
    """

    def __init__(
        self,
        config: CodeMorphConfig,
        llm_client,  # LLMClient - type hint omitted to avoid circular import
        symbol_registry: SymbolRegistry,
    ):
        """
        Initialize the test translator.

        Args:
            config: CodeMorph configuration
            llm_client: LLM client for generating translations
            symbol_registry: Symbol registry with name mappings
        """
        self.config = config
        self.llm_client = llm_client
        self.symbol_registry = symbol_registry

    def translate_test_file(
        self,
        source_test: Path,
        translated_functions: dict[str, TranslatedFragment],
    ) -> str:
        """
        Translate a single test file from source to target language.

        Args:
            source_test: Path to source test file
            translated_functions: Dict of translated fragments for context

        Returns:
            Translated test code in target language
        """
        source_code = source_test.read_text(encoding="utf-8")

        # Build context with symbol mappings
        symbol_context = self._build_symbol_context(translated_functions)

        # Build the translation prompt
        prompt = self._build_test_translation_prompt(
            source_code=source_code,
            source_file=source_test,
            symbol_context=symbol_context,
        )

        # Generate translation
        java_test_code = self.llm_client.generate(prompt)

        # Clean up response (remove markdown code blocks if present)
        java_test_code = self._clean_llm_response(java_test_code)

        return java_test_code

    def _build_test_translation_prompt(
        self,
        source_code: str,
        source_file: Path,
        symbol_context: str,
    ) -> str:
        """Build the prompt for test file translation."""
        source_lang = self.config.project.source.language.value
        target_lang = self.config.project.target.language.value
        target_version = self.config.project.target.version

        # Determine target test framework and translation rules
        if self.config.project.target.language == LanguageType.JAVA:
            return self._build_pytest_to_junit_prompt(
                source_code, source_file, symbol_context, source_lang, target_lang, target_version
            )
        else:
            return self._build_junit_to_pytest_prompt(
                source_code, source_file, symbol_context, source_lang, target_lang, target_version
            )

    def _build_pytest_to_junit_prompt(
        self,
        source_code: str,
        source_file: Path,
        symbol_context: str,
        source_lang: str,
        target_lang: str,
        target_version: str,
    ) -> str:
        """Build prompt for pytest to JUnit translation."""
        test_framework = "JUnit 5"
        test_imports = """import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;"""

        # Get package name for Java
        package_declaration = ""
        package = self.config.project.target.package_name
        if package:
            package_declaration = f"package {package};\n\n"

        # Build class name from source file
        class_name = self._get_test_class_name(source_file)

        return f"""You are translating {source_lang} pytest tests to {target_lang} {target_version} {test_framework} tests.

SYMBOL MAPPINGS (use these EXACT names in your translation):
{symbol_context}

TRANSLATION RULES:
1. Convert test functions to @Test annotated methods
2. Convert pytest.raises(Exception) to assertThrows(Exception.class, () -> ...)
3. Convert assert statements to assertEquals(), assertTrue(), assertFalse(), assertNotNull(), etc.
4. Convert pytest fixtures (@pytest.fixture) to @BeforeEach methods
5. Use the EXACT Java class/method names from the symbol mappings above
6. Add proper imports for {test_framework} and the translated classes
7. Preserve test logic and assertions exactly
8. Handle edge cases and error conditions
9. Use descriptive @DisplayName annotations

REQUIRED IMPORTS:
{test_imports}

OUTPUT FORMAT:
- Start with package declaration (if applicable): {package_declaration}
- Include all necessary imports
- Create a public class named: {class_name}
- Include @Test methods for each test case

SOURCE TEST FILE ({source_lang}):
```{source_lang}
{source_code}
```

Generate the equivalent {target_lang} {test_framework} test class.
Return ONLY the {target_lang} code, no explanations or markdown.
"""

    def _build_junit_to_pytest_prompt(
        self,
        source_code: str,
        source_file: Path,
        symbol_context: str,
        source_lang: str,
        target_lang: str,
        target_version: str,
    ) -> str:
        """Build prompt for JUnit to pytest translation."""
        test_framework = "pytest"
        test_imports = """import pytest
from typing import Any"""

        # Build test file name from source file
        test_file_name = self._get_pytest_file_name(source_file)

        return f"""You are translating {source_lang} JUnit tests to {target_lang} {target_version} {test_framework} tests.

SYMBOL MAPPINGS (use these EXACT names in your translation):
{symbol_context}

TRANSLATION RULES:
1. Convert @Test methods to test_ prefixed functions
2. Convert assertThrows(Exception.class, () -> ...) to pytest.raises(Exception)
3. Convert JUnit assertions to Python assert statements:
   - assertEquals(expected, actual) -> assert actual == expected
   - assertTrue(condition) -> assert condition
   - assertFalse(condition) -> assert not condition
   - assertNull(obj) -> assert obj is None
   - assertNotNull(obj) -> assert obj is not None
   - assertThrows -> pytest.raises
4. Convert @BeforeEach methods to pytest fixtures with @pytest.fixture
5. Convert @AfterEach methods to pytest fixtures with yield
6. Use the EXACT Python function/class names from the symbol mappings above
7. Remove class wrapper (Python tests are typically module-level functions)
8. Preserve test logic and assertions exactly
9. Use snake_case for function names

REQUIRED IMPORTS:
{test_imports}

OUTPUT FORMAT:
- Start with imports
- Import the translated modules
- Create test_ prefixed functions for each test case
- Use pytest fixtures for setup/teardown

SOURCE TEST FILE ({source_lang}):
```{source_lang}
{source_code}
```

Generate the equivalent {target_lang} {test_framework} test module.
Return ONLY the {target_lang} code, no explanations or markdown.
"""

    def _build_symbol_context(
        self,
        translated_functions: dict[str, TranslatedFragment],
    ) -> str:
        """Build symbol mapping context for test translation."""
        lines = []

        for func_id, frag in translated_functions.items():
            mapping = self.symbol_registry.get_mapping(func_id)
            if mapping:
                lines.append(f"- {mapping.source_name} → {mapping.target_name}")
                if mapping.signature:
                    lines.append(f"  Signature: {mapping.signature}")
            elif frag.target_name:
                # Fall back to fragment info if not in registry
                lines.append(f"- {frag.fragment.name} → {frag.target_name}")
                if frag.target_signature:
                    lines.append(f"  Signature: {frag.target_signature}")

        if not lines:
            return "No symbol mappings available. Use standard naming conventions."

        return "\n".join(lines)

    def _get_test_class_name(self, source_file: Path) -> str:
        """
        Convert a source test filename to target test class name (for Java).

        Examples:
            test_calculator.py → CalculatorTest
            test_user_service.py → UserServiceTest
            CalculatorTest.java → CalculatorTest (unchanged)
        """
        name = source_file.stem  # e.g., "test_calculator" or "CalculatorTest"

        # Handle Python test file -> Java test class
        if name.startswith("test_"):
            name = name[5:]
            # Convert snake_case to PascalCase
            pascal = "".join(word.capitalize() for word in name.split("_"))
            return f"{pascal}Test"

        # Handle Java test file (already in correct format)
        if name.endswith("Test"):
            return name

        # Default: add Test suffix
        return f"{name}Test"

    def _get_pytest_file_name(self, source_file: Path) -> str:
        """
        Convert a source test filename to pytest file name.

        Examples:
            CalculatorTest.java → test_calculator.py
            UserServiceTest.java → test_user_service.py
            test_calculator.py → test_calculator.py (unchanged)
        """
        name = source_file.stem  # e.g., "CalculatorTest" or "test_calculator"

        # Handle Python test file (already in correct format)
        if name.startswith("test_"):
            return f"{name}.py"

        # Handle Java test file -> Python test file
        # Remove Test suffix
        if name.endswith("Test"):
            name = name[:-4]

        # Convert PascalCase/camelCase to snake_case
        import re
        snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

        return f"test_{snake_name}.py"

    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response by removing markdown formatting."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Find start of code
            start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                    # Skip language identifier if present
                    if lines[start_idx].strip() in ("java", "python", ""):
                        start_idx = 2
                    break

            # Find end of code
            end_idx = len(lines)
            for i in range(len(lines) - 1, start_idx, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break

            response = "\n".join(lines[start_idx:end_idx])

        return response.strip()

    def get_test_file_mapping(self, source_test: Path, output_dir: Path) -> Path:
        """
        Get the output path for a translated test file.

        Args:
            source_test: Path to source test file
            output_dir: Base output directory

        Returns:
            Path for the translated test file
        """
        if self.config.project.target.language == LanguageType.JAVA:
            # Java test structure: src/test/java/package/ClassTest.java
            class_name = self._get_test_class_name(source_test)
            package = self.config.project.target.package_name
            if package:
                package_path = package.replace(".", "/")
                return output_dir / "src" / "test" / "java" / package_path / f"{class_name}.java"
            return output_dir / "src" / "test" / "java" / f"{class_name}.java"
        else:
            # Python test structure: tests/test_name.py
            pytest_file_name = self._get_pytest_file_name(source_test)
            return output_dir / "tests" / pytest_file_name
