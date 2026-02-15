"""
Compile Fixer for Phase 4 Assembly.

LLM-driven compile-fix loop that iterates until all assembled Python files
compile cleanly or max iterations are reached.
"""

import logging
from pathlib import Path
from typing import Any

from codemorph.assembler.tools import AgentToolRegistry, run_agent_loop

logger = logging.getLogger(__name__)

COMPILE_FIXER_SYSTEM_PROMPT = """\
You are fixing compilation errors in an assembled Python project.

Use the available tools to find and fix errors:
1. Call compile_check on each .py file to find syntax errors
2. Call read_file to see the problematic code
3. Call write_file to apply fixes
4. Call run_script to test fixes (e.g., try importing the module)
5. Call grep_translated or rag_search if you need to understand patterns

CRITICAL RULES:
1. Fix one file at a time.
2. Make minimal changes — only fix the actual errors.
3. Do NOT rewrite large sections of code.
4. After fixing, re-check with compile_check to verify.
5. Try importing the package after all files compile.

When all files compile cleanly and imports work, respond with:
"ALL_CLEAN: All files compile and import successfully."

If you cannot fix an error after trying, respond with:
"STUCK: <description of the remaining issue>"
"""


class CompileFixer:
    """LLM-driven compile-fix loop."""

    def __init__(self, llm_client: Any, verbose: bool = True):
        self.llm_client = llm_client
        self.verbose = verbose

    def run(
        self,
        output_dir: Path,
        tools: AgentToolRegistry,
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """Run the compile-fix loop.

        Args:
            output_dir: Directory containing assembled Python files
            tools: Tool registry
            max_iterations: Maximum fix iterations

        Returns:
            Summary dict with results
        """
        output_dir = Path(output_dir)
        results = {
            "iterations": 0,
            "files_checked": 0,
            "errors_found": 0,
            "errors_fixed": 0,
            "final_status": "unknown",
            "remaining_errors": [],
        }

        for iteration in range(max_iterations):
            results["iterations"] = iteration + 1
            logger.info(f"Compile-fix iteration {iteration + 1}/{max_iterations}")

            # First, do a quick mechanical check
            errors = self._check_all_files(output_dir)
            results["files_checked"] = len(list(output_dir.rglob("*.py")))

            if not errors:
                # All clean — try import check
                import_ok = self._check_imports(output_dir, tools)
                if import_ok:
                    results["final_status"] = "clean"
                    results["errors_fixed"] = results["errors_found"]
                    logger.info("All files compile and import successfully")
                    return results
                else:
                    # Import issues — let LLM fix them
                    errors = [("import_check", "Import check failed")]

            results["errors_found"] = len(errors)

            # Let the LLM fix the errors
            error_summary = "\n".join(
                f"  {fname}: {msg}" for fname, msg in errors
            )

            initial_message = (
                f"Compile-fix iteration {iteration + 1}.\n"
                f"Output directory: {output_dir}\n"
                f"Errors found:\n{error_summary}\n\n"
                f"Fix these errors using the available tools."
            )

            try:
                response = run_agent_loop(
                    llm_client=self.llm_client,
                    system_prompt=COMPILE_FIXER_SYSTEM_PROMPT,
                    tools=tools,
                    max_turns=20,
                    initial_message=initial_message,
                    verbose=self.verbose,
                )
            except Exception as e:
                logger.warning(f"LLM compile fixer failed ({e}), stopping")
                results["final_status"] = "llm_unavailable"
                results["remaining_errors"] = [msg for _, msg in errors]
                return results

            if "ALL_CLEAN" in response:
                results["final_status"] = "clean"
                results["errors_fixed"] = results["errors_found"]
                logger.info("Compile fixer reports all clean")
                return results

            if "STUCK" in response:
                logger.warning(f"Compile fixer is stuck: {response}")
                results["final_status"] = "stuck"
                results["remaining_errors"] = [e[1] for e in errors]
                return results

        # Exhausted iterations
        remaining_errors = self._check_all_files(output_dir)
        results["final_status"] = "max_iterations" if remaining_errors else "clean"
        results["remaining_errors"] = [msg for _, msg in remaining_errors]

        return results

    def _check_all_files(self, output_dir: Path) -> list[tuple[str, str]]:
        """Check all .py files for compilation errors.

        Returns:
            List of (filename, error_message) tuples
        """
        errors = []

        for py_file in sorted(output_dir.rglob("*.py")):
            try:
                source = py_file.read_text(encoding="utf-8")
                compile(source, str(py_file), "exec")
            except SyntaxError as e:
                rel_path = str(py_file.relative_to(output_dir))
                errors.append(
                    (rel_path, f"Line {e.lineno}: {e.msg}")
                )

        return errors

    def _check_imports(self, output_dir: Path, tools: AgentToolRegistry) -> bool:
        """Try importing the assembled package."""
        import subprocess

        # Find the package directory (first directory with __init__.py)
        init_files = list(output_dir.rglob("__init__.py"))
        if not init_files:
            return True  # No package to import

        package_dir = init_files[0].parent
        package_name = package_dir.name

        try:
            result = subprocess.run(
                ["python", "-c",
                 f"import sys; sys.path.insert(0, '{output_dir}'); "
                 f"import {package_name}; print('Import OK')"],
                capture_output=True, text=True, timeout=15,
                cwd=str(output_dir),
            )
            if result.returncode == 0:
                return True
            logger.info(f"Import check failed: {result.stderr.strip()}")
            return False
        except Exception as e:
            logger.info(f"Import check error: {e}")
            return False
