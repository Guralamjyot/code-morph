"""
Cross-Language Bridge: Java Executor

Allows Python to execute Java code for verification purposes.
Uses subprocess-based JSON communication for simplicity and portability.

Based on Section 18 of the CodeMorph v2.0 plan.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JavaExecutor:
    """
    Executes Java code from Python using subprocess and JSON communication.

    This is used during Phase 3 to test translated Java code with Python-captured inputs.
    """

    def __init__(
        self,
        java_home: Optional[str] = None,
        classpath: Optional[List[Path]] = None,
        timeout: int = 30
    ):
        """
        Initialize Java executor.

        Args:
            java_home: Path to Java installation (uses system default if None)
            classpath: Additional classpath entries
            timeout: Execution timeout in seconds
        """
        self.java_home = java_home
        self.classpath = classpath or []
        self.timeout = timeout

        # Verify Java is available
        self._verify_java()

    def _verify_java(self):
        """Verify Java installation."""
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Java not found or not working")

            logger.info(f"Java found: {result.stderr.splitlines()[0]}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Java not available: {e}")

    def execute_function(
        self,
        class_name: str,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        source_files: List[Path],
        main_wrapper: Optional[str] = None
    ) -> Tuple[Any, Optional[str]]:
        """
        Execute a Java function with given inputs.

        Args:
            class_name: Fully qualified class name (e.g., "com.example.Calculator")
            method_name: Method name to execute
            args: Positional arguments
            kwargs: Keyword arguments (converted to builder pattern or map)
            source_files: Java source files to compile
            main_wrapper: Optional custom main method wrapper code

        Returns:
            Tuple of (result, exception_message)
            If exception occurred, result is None and exception_message is set
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Compile Java sources
            logger.debug(f"Compiling {len(source_files)} Java files")
            compiled = self._compile_sources(source_files, tmpdir)

            if not compiled:
                return None, "Compilation failed"

            # Step 2: Create test harness
            harness_file = self._create_test_harness(
                tmpdir,
                class_name,
                method_name,
                args,
                kwargs,
                main_wrapper
            )

            # Step 3: Compile test harness
            harness_compiled = self._compile_sources([harness_file], tmpdir)

            if not harness_compiled:
                return None, "Test harness compilation failed"

            # Step 4: Execute test harness
            result, exception = self._execute_harness(tmpdir)

            return result, exception

    def _compile_sources(self, source_files: List[Path], output_dir: Path) -> bool:
        """Compile Java source files."""
        # Build classpath
        classpath_str = str(output_dir)
        if self.classpath:
            classpath_str += ":" + ":".join(str(p) for p in self.classpath)

        # Compile command
        cmd = [
            "javac",
            "-d", str(output_dir),
            "-cp", classpath_str,
        ] + [str(f) for f in source_files]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                logger.error(f"Compilation failed:\n{result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("Compilation timeout")
            return False

    def _create_test_harness(
        self,
        output_dir: Path,
        class_name: str,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        main_wrapper: Optional[str] = None
    ) -> Path:
        """
        Create a Java test harness that:
        1. Reads JSON input from stdin
        2. Calls the target method
        3. Writes JSON output to stdout
        """

        # Serialize inputs to JSON
        inputs_json = json.dumps({
            "args": args,
            "kwargs": kwargs
        })

        # Generate harness code
        if main_wrapper:
            # Use custom wrapper
            harness_code = main_wrapper
        else:
            # Generate default wrapper
            harness_code = self._generate_default_harness(
                class_name,
                method_name,
                inputs_json
            )

        # Write to file
        harness_file = output_dir / "TestHarness.java"
        harness_file.write_text(harness_code)

        return harness_file

    def _generate_default_harness(
        self,
        class_name: str,
        method_name: str,
        inputs_json: str
    ) -> str:
        """Generate default test harness code."""

        # Extract package and simple class name
        if "." in class_name:
            package = ".".join(class_name.split(".")[:-1])
            simple_class = class_name.split(".")[-1]
        else:
            package = ""
            simple_class = class_name

        package_stmt = f"package {package};" if package else ""

        return f"""{package_stmt}

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.*;

public class TestHarness {{
    public static void main(String[] args) {{
        try {{
            // Parse input JSON
            String inputJson = {json.dumps(inputs_json)};

            ObjectMapper mapper = new ObjectMapper();
            JsonNode input = mapper.readTree(inputJson);

            // Extract arguments
            JsonNode argsNode = input.get("args");
            JsonNode kwargsNode = input.get("kwargs");

            // Create instance (assumes default constructor)
            {simple_class} instance = new {simple_class}();

            // Call method (simplified - assumes no args for now)
            Object result = instance.{method_name}();

            // Serialize result
            String resultJson = mapper.writeValueAsString(result);

            // Output result
            System.out.println("RESULT:" + resultJson);

        }} catch (Exception e) {{
            // Output exception
            System.out.println("EXCEPTION:" + e.getClass().getName() + ":" + e.getMessage());
            e.printStackTrace();
        }}
    }}
}}
"""

    def _execute_harness(self, classpath_dir: Path) -> Tuple[Any, Optional[str]]:
        """Execute the test harness and parse results."""

        cmd = [
            "java",
            "-cp", str(classpath_dir),
            "TestHarness"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=classpath_dir
            )

            # Parse output
            output = result.stdout.strip()

            if output.startswith("RESULT:"):
                # Successful execution
                result_json = output[7:]  # Remove "RESULT:" prefix
                result_data = json.loads(result_json)
                return result_data, None

            elif output.startswith("EXCEPTION:"):
                # Exception occurred
                exception_info = output[10:]  # Remove "EXCEPTION:" prefix
                return None, exception_info

            else:
                # Unexpected output
                logger.error(f"Unexpected output from test harness:\n{output}")
                return None, f"Unexpected output: {output[:100]}"

        except subprocess.TimeoutExpired:
            return None, "Execution timeout"

        except json.JSONDecodeError as e:
            return None, f"Failed to parse result JSON: {e}"


class SimplifiedJavaExecutor:
    """
    Simplified Java executor that doesn't require Jackson for JSON.
    Uses direct method invocation via reflection.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute_static_method(
        self,
        source_file: Path,
        class_name: str,
        method_name: str,
        *args
    ) -> Tuple[Any, Optional[str]]:
        """
        Execute a static Java method.

        This is simpler than the full executor but limited to static methods.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Compile source
            result = subprocess.run(
                ["javac", "-d", str(tmpdir), str(source_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return None, f"Compilation failed: {result.stderr}"

            # Create wrapper that calls the method
            wrapper_code = self._create_simple_wrapper(
                class_name,
                method_name,
                args
            )

            wrapper_file = tmpdir / "Wrapper.java"
            wrapper_file.write_text(wrapper_code)

            # Compile wrapper
            result = subprocess.run(
                ["javac", "-cp", str(tmpdir), "-d", str(tmpdir), str(wrapper_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return None, f"Wrapper compilation failed: {result.stderr}"

            # Execute wrapper
            result = subprocess.run(
                ["java", "-cp", str(tmpdir), "Wrapper"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                # Check if it's an exception
                if "Exception" in result.stderr:
                    return None, result.stderr.split('\n')[0]
                return None, f"Execution failed: {result.stderr}"

            # Parse output
            output = result.stdout.strip()

            try:
                # Try to parse as number
                if '.' in output:
                    return float(output), None
                else:
                    return int(output), None
            except ValueError:
                # Return as string
                return output, None

    def _create_simple_wrapper(
        self,
        class_name: str,
        method_name: str,
        args: tuple
    ) -> str:
        """Create a simple wrapper for static method."""

        # Format arguments
        formatted_args = ", ".join(self._format_arg(arg) for arg in args)

        return f"""public class Wrapper {{
    public static void main(String[] args) {{
        try {{
            Object result = {class_name}.{method_name}({formatted_args});
            System.out.println(result);
        }} catch (Exception e) {{
            e.printStackTrace();
            System.exit(1);
        }}
    }}
}}"""

    def _format_arg(self, arg: Any) -> str:
        """Format a Python argument as Java code."""
        if isinstance(arg, bool):
            return "true" if arg else "false"
        elif isinstance(arg, str):
            return f'"{arg}"'
        elif isinstance(arg, (int, float)):
            return str(arg)
        elif isinstance(arg, list):
            items = ", ".join(self._format_arg(item) for item in arg)
            return f"new Object[]{{{items}}}"
        else:
            return str(arg)
