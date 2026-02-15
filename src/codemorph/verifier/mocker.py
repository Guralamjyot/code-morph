"""
Function Mocking System (Phase 3)

Generates mock/stub implementations for functions that fail translation.
Allows graceful degradation by calling back to original source code.

Based on Section 13 of the CodeMorph v2.0 plan.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MockStrategy(str, Enum):
    """Strategy for mocking untranslatable functions."""

    BRIDGE = "bridge"  # Call original code via language bridge
    STUB = "stub"  # Generate a stub that throws NotImplementedError
    MANUAL = "manual"  # Mark for manual implementation


@dataclass
class MockedFunction:
    """Information about a mocked function."""

    function_id: str
    function_name: str
    source_language: str
    target_language: str
    strategy: MockStrategy
    reason: str
    generated_code: str

    def __str__(self) -> str:
        return (
            f"{self.function_id}:\n"
            f"  Strategy: {self.strategy.value}\n"
            f"  Reason: {self.reason}"
        )


class FunctionMocker:
    """Generates mock implementations for failed translations."""

    def __init__(self, source_lang: str, target_lang: str):
        """
        Initialize function mocker.

        Args:
            source_lang: Source language (e.g., "python")
            target_lang: Target language (e.g., "java")
        """
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()

    def generate_mock(
        self,
        function_id: str,
        function_name: str,
        signature: str,
        strategy: MockStrategy,
        reason: str
    ) -> MockedFunction:
        """
        Generate a mock implementation.

        Args:
            function_id: Full function identifier (module::function)
            function_name: Function name
            signature: Function signature in target language
            strategy: Mocking strategy
            reason: Why the function needed mocking

        Returns:
            MockedFunction with generated code
        """
        if strategy == MockStrategy.BRIDGE:
            code = self._generate_bridge_mock(function_id, function_name, signature)
        elif strategy == MockStrategy.STUB:
            code = self._generate_stub_mock(function_name, signature)
        else:  # MANUAL
            code = self._generate_manual_placeholder(function_name, signature)

        return MockedFunction(
            function_id=function_id,
            function_name=function_name,
            source_language=self.source_lang,
            target_language=self.target_lang,
            strategy=strategy,
            reason=reason,
            generated_code=code
        )

    def _generate_bridge_mock(
        self,
        function_id: str,
        function_name: str,
        signature: str
    ) -> str:
        """Generate a mock that calls back to original code via bridge."""

        if self.target_lang == "java":
            return self._generate_java_bridge_mock(function_id, function_name, signature)
        elif self.target_lang == "python":
            return self._generate_python_bridge_mock(function_id, function_name, signature)
        else:
            raise NotImplementedError(f"Bridge mocking not implemented for {self.target_lang}")

    def _generate_stub_mock(self, function_name: str, signature: str) -> str:
        """Generate a stub that throws NotImplementedError."""

        if self.target_lang == "java":
            return f"""{signature} {{
    throw new UnsupportedOperationException(
        "{function_name}: Translation failed - manual implementation required"
    );
}}"""
        elif self.target_lang == "python":
            # Always generate a valid Python stub regardless of source signature
            return f"""def {function_name}(*args, **kwargs):
    raise NotImplementedError(
        "{function_name}: Translation failed - manual implementation required"
    )"""
        else:
            raise NotImplementedError(f"Stub mocking not implemented for {self.target_lang}")

    def _generate_manual_placeholder(self, function_name: str, signature: str) -> str:
        """Generate a placeholder for manual implementation."""

        if self.target_lang == "java":
            return f"""{signature} {{
    // TODO: Implement {function_name} manually
    // The automatic translation failed for this function.
    // Please implement the logic based on the original source code.
    throw new UnsupportedOperationException("Not yet implemented");
}}"""
        elif self.target_lang == "python":
            return f"""{signature}:
    # TODO: Implement {function_name} manually
    # The automatic translation failed for this function.
    # Please implement the logic based on the original source code.
    raise NotImplementedError("Not yet implemented")"""
        else:
            raise NotImplementedError(f"Manual placeholder not implemented for {self.target_lang}")

    def _generate_java_bridge_mock(
        self,
        function_id: str,
        function_name: str,
        signature: str
    ) -> str:
        """Generate Java code that calls Python via bridge."""

        # Extract module and function name
        if "::" in function_id:
            module, func = function_id.split("::", 1)
        else:
            module = "__main__"
            func = function_name

        # Parse signature to extract parameters
        # Simple parsing - assumes format: "modifier returnType name(params)"
        params = self._extract_java_params(signature)
        param_names = [p.split()[-1] for p in params]

        return f"""{signature} {{
    // MOCKED: Calls original Python implementation via bridge
    try {{
        PythonBridge bridge = PythonBridge.getInstance();
        Object result = bridge.callPythonFunction(
            "{module}",
            "{func}",
            {', '.join(param_names) if param_names else ''}
        );
        // Note: Type conversion may be needed
        return ({self._extract_java_return_type(signature)}) result;
    }} catch (Exception e) {{
        throw new RuntimeException("Failed to call Python bridge for {function_name}", e);
    }}
}}"""

    def _generate_python_bridge_mock(
        self,
        function_id: str,
        function_name: str,
        signature: str
    ) -> str:
        """Generate Python code that calls Java via bridge."""

        # Extract class and method
        if "::" in function_id:
            class_path, method = function_id.split("::", 1)
        else:
            class_path = "Main"
            method = function_name

        # Parse signature to extract parameters
        params = self._extract_python_params(signature)

        return f"""{signature}:
    # MOCKED: Calls original Java implementation via bridge
    from codemorph.bridges import JavaBridge

    bridge = JavaBridge.get_instance()
    result = bridge.call_java_method(
        "{class_path}",
        "{method}",
        {', '.join(params) if params else ''}
    )
    return result"""

    def _extract_java_params(self, signature: str) -> List[str]:
        """Extract parameter list from Java signature."""
        # Find parameter list in parentheses
        start = signature.find('(')
        end = signature.find(')')
        if start == -1 or end == -1:
            return []

        param_str = signature[start+1:end].strip()
        if not param_str:
            return []

        # Split by comma (simple parsing)
        return [p.strip() for p in param_str.split(',')]

    def _extract_python_params(self, signature: str) -> List[str]:
        """Extract parameter names from Python signature."""
        # Find parameter list in parentheses
        start = signature.find('(')
        end = signature.find(')')
        if start == -1 or end == -1:
            return []

        param_str = signature[start+1:end].strip()
        if not param_str:
            return []

        # Split by comma and extract names (skip 'self')
        params = []
        for p in param_str.split(','):
            p = p.strip()
            # Remove type hints
            if ':' in p:
                p = p.split(':')[0].strip()
            # Remove default values
            if '=' in p:
                p = p.split('=')[0].strip()
            if p and p != 'self':
                params.append(p)

        return params

    def _extract_java_return_type(self, signature: str) -> str:
        """Extract return type from Java signature."""
        # Assumes format: "modifier returnType name(params)"
        parts = signature.split('(')[0].strip().split()

        # Remove modifiers (public, private, static, final, etc.)
        modifiers = {'public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized'}
        non_modifier_parts = [p for p in parts if p.lower() not in modifiers]

        if len(non_modifier_parts) >= 2:
            return non_modifier_parts[-2]  # Second to last is return type

        return "Object"  # Fallback


class MockRegistry:
    """Tracks all mocked functions in a project."""

    def __init__(self):
        self.mocked_functions: Dict[str, MockedFunction] = {}

    def register_mock(self, mock: MockedFunction):
        """Register a mocked function."""
        self.mocked_functions[mock.function_id] = mock
        logger.info(f"Registered mock for {mock.function_id} (strategy: {mock.strategy})")

    def get_mock(self, function_id: str) -> Optional[MockedFunction]:
        """Get mock for a function."""
        return self.mocked_functions.get(function_id)

    def is_mocked(self, function_id: str) -> bool:
        """Check if function is mocked."""
        return function_id in self.mocked_functions

    def get_all_mocks(self) -> List[MockedFunction]:
        """Get all mocked functions."""
        return list(self.mocked_functions.values())

    def get_mocks_by_strategy(self, strategy: MockStrategy) -> List[MockedFunction]:
        """Get mocks using a specific strategy."""
        return [
            mock for mock in self.mocked_functions.values()
            if mock.strategy == strategy
        ]

    def generate_report(self) -> str:
        """Generate a report of all mocked functions."""
        lines = []
        lines.append("=" * 80)
        lines.append("MOCKED FUNCTIONS REPORT")
        lines.append("=" * 80)
        lines.append("")

        if not self.mocked_functions:
            lines.append("No functions were mocked.")
            return "\n".join(lines)

        lines.append(f"Total mocked functions: {len(self.mocked_functions)}")
        lines.append("")

        # Group by strategy
        for strategy in MockStrategy:
            mocks = self.get_mocks_by_strategy(strategy)
            if mocks:
                lines.append(f"\n{strategy.value.upper()} ({len(mocks)}):")
                lines.append("-" * 40)
                for mock in mocks:
                    lines.append(f"  • {mock.function_id}")
                    lines.append(f"    Reason: {mock.reason}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("ACTION REQUIRED")
        lines.append("=" * 80)
        lines.append("")

        manual_mocks = self.get_mocks_by_strategy(MockStrategy.MANUAL)
        if manual_mocks:
            lines.append("The following functions require manual implementation:")
            for mock in manual_mocks:
                lines.append(f"  • {mock.function_id}")
        else:
            lines.append("No manual implementation required.")

        return "\n".join(lines)

    def save_to_file(self, filepath: Path):
        """Save mock report to file."""
        report = self.generate_report()

        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Mock report saved to {filepath}")


def create_callee_mock(
    callee_function_id: str,
    return_value: any,
    target_lang: str = "java"
) -> str:
    """
    Create a mock for a callee function that returns a fixed value.

    This is used during Phase 3 testing to isolate function logic.

    Args:
        callee_function_id: ID of the function to mock
        return_value: Fixed value to return
        target_lang: Target language

    Returns:
        Mock code as string
    """
    if target_lang == "java":
        # Generate a simple mock that returns the fixed value
        return f"""// Mock of {callee_function_id}
// Returns: {return_value}
return {_format_java_literal(return_value)};"""
    elif target_lang == "python":
        return f"""# Mock of {callee_function_id}
# Returns: {return_value}
return {repr(return_value)}"""
    else:
        raise NotImplementedError(f"Callee mocking not implemented for {target_lang}")


def _format_java_literal(value: any) -> str:
    """Format a Python value as a Java literal."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        # Simple array literal
        items = ", ".join(_format_java_literal(item) for item in value)
        return f"new Object[]{{{items}}}"
    elif isinstance(value, dict):
        # Return as Map
        return "new HashMap<>()"  # Simplified
    else:
        return f'"{str(value)}"'  # Fallback to string
