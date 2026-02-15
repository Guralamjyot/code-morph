"""
OpenCode Fixer for Phase 4 Assembly.

Replaces the internal CompileFixer with an external OpenCode agent that has
access to original Java source and translated fragments via MCP tools.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from codemorph.assembler.index_builder import ProjectIndex
from codemorph.config.models import CodeMorphConfig

logger = logging.getLogger(__name__)

OPENCODE_BIN = Path("/workspace/persistent/home/.opencode/bin/opencode")
MCP_PYTHON = Path("/workspace/persistent/venv311/bin/python")
MCP_SERVER = Path(__file__).resolve().parents[3] / "codemorph_mcp.py"


class OpenCodeFixer:
    """External OpenCode agent compile-fix loop.

    Unlike CompileFixer, this does NOT take an llm_client or tool registry.
    OpenCode gets tools via MCP, and uses its own LLM configuration.
    """

    def __init__(
        self,
        config: CodeMorphConfig,
        index: ProjectIndex,
        state_dir: Path,
        source_dir: Path,
        verbose: bool = True,
        timeout: int = 600,
        interactive: bool = False,
    ):
        self.config = config
        self.index = index
        self.state_dir = Path(state_dir).resolve()
        self.source_dir = Path(source_dir).resolve()
        self.verbose = verbose
        self.timeout = timeout
        self.interactive = interactive

    def run(
        self,
        output_dir: Path,
        max_iterations: int = 2,
    ) -> dict[str, Any]:
        """Run the OpenCode compile-fix loop.

        Returns:
            Summary dict compatible with CompileFixer output format.
        """
        output_dir = Path(output_dir).resolve()
        results: dict[str, Any] = {
            "iterations": 0,
            "files_checked": 0,
            "errors_found": 0,
            "errors_fixed": 0,
            "final_status": "unknown",
            "remaining_errors": [],
        }

        # Write config files that OpenCode needs
        self._write_opencode_config(output_dir)
        self._write_opencode_instructions(output_dir)

        for iteration in range(max_iterations):
            results["iterations"] = iteration + 1
            logger.info(f"OpenCode fix iteration {iteration + 1}/{max_iterations}")

            # Pre-flight: check for errors
            errors = self._check_all_files(output_dir)
            results["files_checked"] = len(list(output_dir.rglob("*.py")))

            if not errors:
                import_ok = self._check_imports(output_dir)
                if import_ok:
                    results["final_status"] = "clean"
                    results["errors_fixed"] = results["errors_found"]
                    logger.info("All files compile and import successfully — skipping OpenCode")
                    return results
                else:
                    errors = [("import_check", "Import check failed — see stderr")]

            results["errors_found"] = len(errors)
            if self.verbose:
                for fname, msg in errors:
                    logger.info(f"  Error: {fname}: {msg}")

            # Build prompt and run OpenCode
            prompt = self._build_prompt(errors, output_dir)

            if self.interactive:
                from rich.console import Console as RichConsole
                from rich.table import Table
                from rich.panel import Panel
                from rich.prompt import Prompt

                _console = RichConsole()

                # Display errors in a Rich table
                table = Table(
                    title=f"Iteration {iteration + 1}/{max_iterations} — {len(errors)} errors"
                )
                table.add_column("File", style="cyan")
                table.add_column("Error", style="red")
                for fname, msg in errors:
                    table.add_row(fname, msg)
                _console.print(table)

                # Ask user what to do
                choice = Prompt.ask(
                    "[bold]Action[/bold]: [R]etry / [E]dit prompt / [S]kip",
                    choices=["r", "e", "s"],
                    default="r",
                )

                if choice == "s":
                    logger.info("User chose to skip this iteration")
                    continue
                elif choice == "e":
                    _console.print(
                        Panel(prompt, title="Current prompt", border_style="dim")
                    )
                    extra = Prompt.ask(
                        "[bold]Additional guidance[/bold] (appended to prompt)"
                    )
                    if extra.strip():
                        prompt += f"\n\nAdditional user guidance:\n{extra}"

            success = self._run_opencode(output_dir, prompt)
            if not success:
                logger.warning("OpenCode process failed or timed out")

            # Post-hoc: check results
            remaining = self._check_all_files(output_dir)
            if not remaining:
                import_ok = self._check_imports(output_dir)
                if import_ok:
                    results["final_status"] = "clean"
                    results["errors_fixed"] = results["errors_found"]
                    logger.info("All files compile and import after OpenCode fix")
                    return results
                else:
                    remaining = [("import_check", "Import check still failing")]

            logger.info(f"  {len(remaining)} errors remain after iteration {iteration + 1}")

        # Exhausted iterations
        remaining = self._check_all_files(output_dir)
        if not remaining:
            import_ok = self._check_imports(output_dir)
            if import_ok:
                results["final_status"] = "clean"
            else:
                results["final_status"] = "import_errors"
                results["remaining_errors"] = ["Import check failed"]
        else:
            results["final_status"] = "max_iterations"
            results["remaining_errors"] = [msg for _, msg in remaining]

        return results

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _write_opencode_config(self, output_dir: Path) -> None:
        """Generate .opencode.json with MCP server pointing at state_dir and source_dir."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — OpenCode may fail")

        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "openai": {
                    "apiKey": "{env:OPENAI_API_KEY}",
                },
            },
            "model": "openai/gpt-4o-mini",
            "mcpServers": {
                "codemorph": {
                    "type": "stdio",
                    "command": str(MCP_PYTHON),
                    "args": [str(MCP_SERVER)],
                    "env": [
                        f"CODEMORPH_STATE_DIR={self.state_dir}",
                        f"CODEMORPH_SOURCE_DIR={self.source_dir}",
                        f"CODEMORPH_PROJECT_ROOT={MCP_SERVER.parent}",
                    ],
                },
            },
        }

        config_path = output_dir / ".opencode.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        logger.info(f"Wrote {config_path}")

    def _write_opencode_instructions(self, output_dir: Path) -> None:
        """Generate OPENCODE.md with project context from the index."""
        lines = [
            "# CodeMorph Assembly — Project Context",
            "",
            "This project was auto-assembled from Java→Python translation.",
            "You have MCP tools to read the original Java source and translated fragments.",
            "",
            "## Available MCP Tools",
            "",
            "- `read_source_file(filename)` — read original Java source (e.g. 'Pair.java')",
            "- `read_fragment(fragment_id)` — read translated Python for a fragment",
            "- `list_class_members(class_name)` — list all members of a class with status",
            "- `get_class_hierarchy(class_name)` — get inheritance info",
            "- `lookup_registry(java_name, python_name, class_name)` — name mappings",
            "- `get_project_summary(detail_level)` — overview ('brief' or 'full')",
            "",
            "## Classes in this project",
            "",
        ]

        for cls_name in sorted(self.index.classes.keys()):
            summary = self.index.classes[cls_name]
            bases = self.index.hierarchy.get(cls_name, [])
            base_str = f" (extends {', '.join(bases)})" if bases else ""
            lines.append(
                f"- **{cls_name}**{base_str}: "
                f"{summary.translated_count} translated, "
                f"{summary.mocked_count} mocked, "
                f"{len(summary.members)} total members"
            )

        lines.extend([
            "",
            "## Critical Rules",
            "",
            "1. Make SURGICAL, minimal fixes — only fix the actual errors.",
            "2. Do NOT rewrite entire files or large sections.",
            "3. Do NOT remove methods or simplify classes.",
            "4. Use MCP tools to look up original Java source when unsure.",
            "5. Each class must retain ALL its methods from the translation.",
            "6. Fix import errors by checking what symbols are actually defined.",
            "7. Fix syntax errors by looking at the surrounding context.",
            "",
        ])

        md_path = output_dir / "OPENCODE.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote {md_path}")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, errors: list[tuple[str, str]], output_dir: Path) -> str:
        """Build a comprehensive prompt for OpenCode."""
        error_lines = "\n".join(f"  {fname}: {msg}" for fname, msg in errors)

        # Gather file listing for context
        py_files = sorted(output_dir.rglob("*.py"))
        file_list = "\n".join(f"  {f.relative_to(output_dir)}" for f in py_files)

        prompt = (
            f"Fix the compilation errors in this Python project.\n\n"
            f"Read OPENCODE.md for project context and available MCP tools.\n\n"
            f"Current errors ({len(errors)}):\n{error_lines}\n\n"
            f"Project files:\n{file_list}\n\n"
            f"Instructions:\n"
            f"1. Read the files with errors to understand the context.\n"
            f"2. Use MCP tools (read_source_file, read_fragment, list_class_members) "
            f"to look up the original Java source and translated fragments.\n"
            f"3. Make minimal, surgical fixes to resolve each error.\n"
            f"4. Do NOT rewrite entire files — only change what's needed.\n"
            f"5. Do NOT remove methods or simplify classes.\n"
            f"6. After fixing, verify each file compiles by reading it again.\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # OpenCode invocation
    # ------------------------------------------------------------------

    def _run_opencode(self, output_dir: Path, prompt: str) -> bool:
        """Run OpenCode non-interactively with the given prompt."""
        if not OPENCODE_BIN.exists():
            logger.error(f"OpenCode binary not found at {OPENCODE_BIN}")
            return False

        cmd = [
            str(OPENCODE_BIN),
            "-p", prompt,
            "-q",
            "-c", str(output_dir),
        ]

        env = os.environ.copy()

        logger.info(f"Running OpenCode (timeout={self.timeout}s)...")
        if self.verbose:
            logger.info(f"  CWD: {output_dir}")
            logger.info(f"  Prompt length: {len(prompt)} chars")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(output_dir),
                env=env,
            )

            if self.verbose:
                if result.stdout:
                    # Log first/last bits of output
                    stdout_lines = result.stdout.strip().splitlines()
                    for line in stdout_lines[:20]:
                        logger.info(f"  [opencode] {line}")
                    if len(stdout_lines) > 20:
                        logger.info(f"  [opencode] ... ({len(stdout_lines) - 20} more lines)")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[:10]:
                        logger.info(f"  [opencode:err] {line}")

            if result.returncode != 0:
                logger.warning(f"OpenCode exited with code {result.returncode}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.warning(f"OpenCode timed out after {self.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Failed to run OpenCode: {e}")
            return False

    # ------------------------------------------------------------------
    # Compile checks (same logic as CompileFixer)
    # ------------------------------------------------------------------

    def _check_all_files(self, output_dir: Path) -> list[tuple[str, str]]:
        """Check all .py files for compilation errors."""
        errors = []
        for py_file in sorted(output_dir.rglob("*.py")):
            try:
                source = py_file.read_text(encoding="utf-8")
                compile(source, str(py_file), "exec")
            except SyntaxError as e:
                rel_path = str(py_file.relative_to(output_dir))
                errors.append((rel_path, f"Line {e.lineno}: {e.msg}"))
        return errors

    def _check_imports(self, output_dir: Path) -> bool:
        """Try importing the assembled package using python3.13."""
        init_files = list(output_dir.rglob("__init__.py"))
        if not init_files:
            return True

        package_dir = init_files[0].parent
        package_name = package_dir.name

        try:
            result = subprocess.run(
                [
                    "python3.13", "-c",
                    f"import sys; sys.path.insert(0, '{output_dir}'); "
                    f"import {package_name}; print('Import OK')",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(output_dir),
            )
            if result.returncode == 0:
                return True
            logger.info(f"Import check failed: {result.stderr.strip()}")
            return False
        except Exception as e:
            logger.info(f"Import check error: {e}")
            return False
