"""
QA Orchestrator for Phase 4 Assembly.

Replaces the simple compile-fix loop with a structured sub-agent system.
Spawns focused OpenCode processes that review translation fidelity, structure,
and logic.  Each sub-agent writes a structured JSON report; the Python
orchestrator collects reports and applies fixes mechanically.
"""

import ast
import json
import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from codemorph.assembler.index_builder import ProjectIndex
from codemorph.assembler.opencode_fixer import (
    OPENCODE_BIN,
    MCP_PYTHON,
    MCP_SERVER,
    OpenCodeFixer,
)
from codemorph.config.models import CodeMorphConfig

logger = logging.getLogger(__name__)


class QAOrchestrator:
    """OpenCode sub-agent QA system for assembled Python output.

    Pipeline:
      1. Compile check (mechanical)
      2. Spawn sub-agents (each writes a JSON report)
         a. Translation Review Agent (per-class batch)
         b. Structure & Import Agent (project-wide)
         c. Compile Fix Agent (conditional, if syntax errors remain)
      3. Collect & merge reports
      4. Apply fixes from reports (Python orchestrator)
      5. Final compile + import check
    """

    def __init__(
        self,
        config: CodeMorphConfig,
        index: ProjectIndex,
        state_dir: Path,
        source_dir: Path,
        verbose: bool = True,
        interactive: bool = False,
        timeout: int = 300,
    ):
        self.config = config
        self.index = index
        self.state_dir = Path(state_dir).resolve()
        self.source_dir = Path(source_dir).resolve()
        self.verbose = verbose
        self.interactive = interactive
        self.timeout = timeout

        # Reuse OpenCodeFixer utilities for compile checks & config writing
        self._fixer = OpenCodeFixer(
            config=config,
            index=index,
            state_dir=state_dir,
            source_dir=source_dir,
            verbose=verbose,
            timeout=timeout,
            interactive=interactive,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir: Path,
        max_iterations: int = 2,
    ) -> dict[str, Any]:
        """Run the full QA pipeline.

        Returns a dict compatible with ``OpenCodeFixer.run()``::

            {"iterations": N, "final_status": "clean"|"max_iterations",
             "remaining_errors": [...]}
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

        # Write config files for OpenCode (MCP, OPENCODE.md)
        self._write_opencode_config(output_dir)
        self._fixer._write_opencode_instructions(output_dir)

        for iteration in range(max_iterations):
            results["iterations"] = iteration + 1
            logger.info(f"QA iteration {iteration + 1}/{max_iterations}")

            # ----------------------------------------------------------
            # 1. Pre-flight compile check
            # ----------------------------------------------------------
            errors = self._fixer._check_all_files(output_dir)
            results["files_checked"] = len(list(output_dir.rglob("*.py")))
            results["errors_found"] = len(errors)

            if not errors:
                import_ok = self._fixer._check_imports(output_dir)
                if import_ok:
                    # Compile and imports pass — run runtime verification
                    runtime_issues = self._run_runtime_verification(output_dir)
                    if not runtime_issues:
                        results["final_status"] = "clean"
                        results["errors_fixed"] = results["errors_found"]
                        logger.info("All files compile, import, and pass runtime checks")
                        return results
                    # Write runtime issues as a report for collection
                    runtime_report = output_dir / "_qa_report_runtime.json"
                    runtime_report.write_text(
                        json.dumps({"agent": "runtime_verifier", "issues": runtime_issues}, indent=2),
                        encoding="utf-8",
                    )
                    logger.info(f"  Runtime verification found {len(runtime_issues)} issues")

                    # Apply mechanical runtime fixes before sub-agents
                    mechanical = [i for i in runtime_issues if i["type"] == "enum_str_error"]
                    if mechanical:
                        applied = self._apply_fixes(output_dir, mechanical)
                        logger.info(f"  Applied {applied} mechanical runtime fixes")

            # ----------------------------------------------------------
            # 2. Spawn sub-agents
            # ----------------------------------------------------------
            # Clean previous reports (keep runtime report)
            for old in output_dir.glob("_qa_report_*.json"):
                if old.name != "_qa_report_runtime.json":
                    old.unlink()

            # Collect runtime issues for sub-agent context
            runtime_report_path = output_dir / "_qa_report_runtime.json"
            runtime_context_issues: list[dict[str, Any]] = []
            if runtime_report_path.exists():
                try:
                    data = json.loads(runtime_report_path.read_text(encoding="utf-8"))
                    runtime_context_issues = data.get("issues", [])
                except (json.JSONDecodeError, OSError):
                    pass

            # 2a. Translation Review Agent (batch all classes)
            class_names = sorted(self.index.classes.keys())
            self._spawn_translation_review(output_dir, class_names, runtime_context_issues)

            # 2b. Structure & Import Agent
            self._spawn_structure_review(output_dir)

            # ----------------------------------------------------------
            # 3. Collect reports
            # ----------------------------------------------------------
            all_issues = self._collect_reports(output_dir)
            logger.info(f"  Collected {len(all_issues)} issues from sub-agents")

            # ----------------------------------------------------------
            # 4. Apply mechanical fixes
            # ----------------------------------------------------------
            if all_issues:
                applied = self._apply_fixes(output_dir, all_issues)
                logger.info(f"  Applied {applied} fixes")

            # ----------------------------------------------------------
            # 5. Post-fix compile check
            # ----------------------------------------------------------
            remaining = self._fixer._check_all_files(output_dir)
            if remaining:
                # 2c. Compile Fix Agent (conditional)
                logger.info(f"  {len(remaining)} compile errors remain — spawning fix agent")
                self._spawn_compile_fix(output_dir, remaining)

                remaining = self._fixer._check_all_files(output_dir)

            if not remaining:
                import_ok = self._fixer._check_imports(output_dir)
                if import_ok:
                    results["final_status"] = "clean"
                    results["errors_fixed"] = results["errors_found"]
                    logger.info("All files clean after QA iteration")
                    return results

            logger.info(f"  {len(remaining)} errors remain after iteration {iteration + 1}")

        # Exhausted iterations
        remaining = self._fixer._check_all_files(output_dir)
        if not remaining:
            import_ok = self._fixer._check_imports(output_dir)
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
    # Config (extends OpenCodeFixer config with OUTPUT_DIR)
    # ------------------------------------------------------------------

    def _write_opencode_config(self, output_dir: Path) -> None:
        """Write .opencode.json with MCP server including OUTPUT_DIR."""
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
                        f"CODEMORPH_OUTPUT_DIR={output_dir}",
                    ],
                },
            },
        }

        config_path = output_dir / ".opencode.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        logger.info(f"Wrote {config_path}")

    # ------------------------------------------------------------------
    # Sub-agent: Translation Review
    # ------------------------------------------------------------------

    def _spawn_translation_review(
        self,
        output_dir: Path,
        class_names: list[str],
        runtime_issues: list[dict[str, Any]] | None = None,
    ) -> None:
        """Spawn a Translation Review sub-agent for a batch of classes."""
        # Build class context block
        class_context_parts: list[str] = []
        for cls in class_names:
            summary = self.index.classes.get(cls)
            if not summary:
                continue
            members = [m.fragment_id for m in summary.members]
            class_context_parts.append(
                f"- {cls}: members={members}"
            )
        class_context = "\n".join(class_context_parts)

        # Find output files
        py_files = sorted(output_dir.rglob("*.py"))
        file_list = "\n".join(
            f"  {f.relative_to(output_dir)}" for f in py_files
            if f.name != "__init__.py"
        )

        # Build runtime issues context section
        runtime_section = ""
        if runtime_issues:
            issue_lines = []
            for ri in runtime_issues:
                issue_lines.append(
                    f"  - [{ri['type']}] {ri.get('class_name', '?')}.{ri.get('method', '?')}: "
                    f"{ri.get('description', '')[:150]}"
                )
            runtime_section = (
                "\n\n## Known Runtime Issues\n"
                "The following issues were detected by automated runtime testing.\n"
                "Use `run_test_script(script)` to verify fixes.\n\n"
                + "\n".join(issue_lines)
            )

        report_file = f"_qa_report_translate.json"

        prompt = textwrap.dedent(f"""\
            You are a Translation Review Agent for a Java-to-Python code conversion.

            Your job: review each assembled Python file, compare against the original
            Java source, and **run test scripts** to verify correctness at runtime.

            ## Classes to review
            {class_context}

            ## Output files
            {file_list}

            ## Steps

            ### Step 1: Per-module import test
            For EACH Python file, run `run_test_script` to verify it imports cleanly:
            ```python
            run_test_script("import <module_path>; print('OK')")
            ```
            If any module fails to import, report it as an `import_error` issue.

            ### Step 2: Compare against Java source
            For each class:
            1. Read the assembled Python file using `read_output_file(path)`.
            2. Use `read_source_file(filename)` to get the original Java.
            3. Use `list_class_members(class_name)` and `read_fragment(fragment_id)`
               to see individual translations.
            4. Compare: does each Python method correctly implement the Java logic?

            ### Step 3: Runtime smoke tests
            For key classes, write and run test scripts via `run_test_script(script)`:
            - Try instantiating classes with no-arg constructors
            - Call methods with simple arguments and check they don't crash
            - Verify enum members exist and `str()` returns just the name
            - Test equality/hash on simple objects
            Example:
            ```python
            run_test_script("from tictactoe.position import Position; p = Position(1,2); assert p.row == 1; print('OK')")
            ```

            ## Common Java-to-Python Translation Bugs (CHECK ALL OF THESE)

            1. **Bare class constants**: Java `ROWS` becomes Python `self.ROWS` or
               `ClassName.ROWS` — look for bare uppercase names in method bodies
            2. **Field/method collision**: Java allows `boolean hasWin` field AND
               `boolean hasWin(Player p)` method. Python `self.has_win = None` in
               `__init__` **overwrites** `def has_win(self, player)`. The field
               assignment must be removed or renamed to `_has_win`.
            3. **Missing self**: Instance methods must have `self` as first param.
               `def build_tree(state, ...)` should be `def build_tree(self, state, ...)`
            4. **Bare enum members**: Java `return O` inside enum → Python needs
               `return Player.O`. Check for bare single-letter names like O, X.
            5. **Missing imports**: `Position`, `Player`, `GameBoard`, `Final`, `Type`,
               `List` — each file must import what it uses.
            6. **Orphaned annotations**: `FIELD: Final = None` when the field is set
               in `__init__` or is actually a method — remove the annotation.
            7. **Duplicate methods**: Same method name appears twice in a class.
            8. **`YourClass` placeholder**: LLM sometimes uses `YourClass.method()`
               instead of `self.method()` — replace with correct reference.
            9. **Wrong static/instance**: Java static methods need `@staticmethod`,
               instance methods need `self`.
            10. **Validate method args**: `validate_position(board)` passing a 2D list
                instead of `validate_position(row, col)`.

            ## What to report
            - **duplicate_method**: Same method name defined twice in a class
            - **incorrect_reference**: Missing `self.` prefix, wrong class prefix,
              bare enum members, `YourClass` placeholders
            - **field_method_collision**: `self.X = ...` in __init__ shadows `def X()`
            - **orphaned_annotation**: `FIELD: Final = None` that should be removed
            - **missing_method**: A method from Java is entirely absent
            - **missing_import**: A name is used but not imported
            - **missing_self**: Instance method lacks `self` parameter
            - **logic_error**: Python code does not match Java logic
            - **runtime_error**: Code crashes at runtime (verify with run_test_script)
            - **signature_mismatch**: Wrong parameter count
            {runtime_section}
            ## Output
            Write a JSON file at `{report_file}` with this exact structure:
            ```json
            {{
              "agent": "translation_review",
              "issues": [
                {{
                  "type": "duplicate_method|incorrect_reference|field_method_collision|orphaned_annotation|missing_method|missing_import|missing_self|logic_error|runtime_error",
                  "file": "package/module.py",
                  "class_name": "ClassName",
                  "method": "method_name",
                  "line": 42,
                  "current": "the problematic code",
                  "suggested": "the fixed code",
                  "description": "human-readable explanation"
                }}
              ]
            }}
            ```

            IMPORTANT:
            - **Actually run test scripts** — do not just read code.
            - Write the JSON report file FIRST, then stop.
            - Do NOT modify any Python source files yourself.
            - Only report genuine issues you can verify against the Java source or
              by running test scripts.
            - If you find no issues, write `{{"agent": "translation_review", "issues": []}}`.
        """)

        logger.info("Spawning Translation Review sub-agent...")
        self._run_opencode(output_dir, prompt)

    # ------------------------------------------------------------------
    # Sub-agent: Structure & Import Review
    # ------------------------------------------------------------------

    def _spawn_structure_review(self, output_dir: Path) -> None:
        """Spawn a Structure & Import Review sub-agent."""
        py_files = sorted(output_dir.rglob("*.py"))
        file_list = "\n".join(
            f"  {f.relative_to(output_dir)}" for f in py_files
        )

        report_file = "_qa_report_structure.json"

        prompt = textwrap.dedent(f"""\
            You are a Structure & Import Review Agent for a Java-to-Python conversion.

            Your job: review the assembled Python project for structural issues,
            **test each module's imports at runtime**, and verify cross-file references.

            ## Project files
            {file_list}

            ## Steps

            ### Step 1: Runtime import test for EACH module
            For EVERY .py file (excluding __init__.py), run:
            ```python
            run_test_script("import <module.path>; print('OK')")
            ```
            This catches missing imports, undefined names at module scope, and
            circular import issues. Report any failures as `import_error`.

            ### Step 2: Structural analysis
            1. Read every `.py` file in the project.
            2. Use `get_class_hierarchy("")` to see the full hierarchy.
            3. Use `get_project_summary("full")` for an overview.
            4. Check each file for:
               - Classes or functions used but not imported (e.g., `Position` used
                 but no `from .position import Position`)
               - Type annotations referencing undefined names (`Final`, `Type`, `List`)
                 — these need `from typing import ...`
               - Imports that reference non-existent modules or symbols
               - Unused imports (imported but never referenced in code)
               - Class bases that don't match the Java hierarchy

            ### Step 3: Cross-file reference validation
            For each class that references another class (e.g., `TicTacToeGameState`
            using `GameBoard`, `Player`, `Position`):
            - Verify the import statement exists
            - Run a test script that imports both classes and verifies no ImportError

            ## Common structural bugs in Java-to-Python translations
            - Missing `from typing import Final, List, Type` when those annotations are used
            - Missing relative imports (`from .module import Class`)
            - `self.INSTRUCTION_TEXT` vs bare `INSTRUCTION_TEXT` (class constant
              referenced without `self.` or `ClassName.` prefix)
            - Inner classes that should have been extracted to top-level
            - `@abstractmethod` on classes that don't inherit from `ABC`

            ## Output
            Write a JSON file at `{report_file}` with this exact structure:
            ```json
            {{
              "agent": "structure_review",
              "issues": [
                {{
                  "type": "import_error|missing_import|unused_import|wrong_base_class|broken_reference|incorrect_reference",
                  "file": "package/module.py",
                  "symbol": "SymbolName",
                  "line": 10,
                  "current": "current code or import",
                  "suggested": "from .module import Symbol",
                  "description": "explanation"
                }}
              ]
            }}
            ```

            IMPORTANT:
            - **Actually run test scripts** — do not just read code.
            - Write the JSON report file FIRST, then stop.
            - Do NOT modify any Python source files yourself.
            - Only report genuine issues.
            - If you find no issues, write `{{"agent": "structure_review", "issues": []}}`.
        """)

        logger.info("Spawning Structure & Import Review sub-agent...")
        self._run_opencode(output_dir, prompt)

    # ------------------------------------------------------------------
    # Sub-agent: Compile Fix (conditional)
    # ------------------------------------------------------------------

    def _spawn_compile_fix(
        self, output_dir: Path, errors: list[tuple[str, str]]
    ) -> None:
        """Spawn a Compile Fix sub-agent to resolve syntax errors."""
        error_lines = "\n".join(f"  {fname}: {msg}" for fname, msg in errors)

        prompt = textwrap.dedent(f"""\
            Fix the following compilation errors in this Python project.
            Read OPENCODE.md for project context and available MCP tools.

            Errors ({len(errors)}):
            {error_lines}

            Instructions:
            1. Read the files with errors.
            2. Use MCP tools to look up original Java source and translated fragments.
            3. Make minimal, surgical fixes to resolve each error.
            4. Do NOT rewrite entire files or remove methods.
            5. Do NOT simplify logic — preserve the original translation intent.
        """)

        logger.info("Spawning Compile Fix sub-agent...")
        self._run_opencode(output_dir, prompt)

    # ------------------------------------------------------------------
    # Report collection
    # ------------------------------------------------------------------

    def _collect_reports(self, output_dir: Path) -> list[dict[str, Any]]:
        """Read all _qa_report_*.json files and merge their issues."""
        all_issues: list[dict[str, Any]] = []

        for report_path in sorted(output_dir.glob("_qa_report_*.json")):
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                issues = data.get("issues", [])
                logger.info(
                    f"  Report {report_path.name}: {len(issues)} issues "
                    f"(agent={data.get('agent', 'unknown')})"
                )
                all_issues.extend(issues)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"  Failed to parse {report_path.name}: {e}")

        return all_issues

    # ------------------------------------------------------------------
    # Fix application
    # ------------------------------------------------------------------

    def _apply_fixes(
        self, output_dir: Path, issues: list[dict[str, Any]]
    ) -> int:
        """Apply mechanical fixes from sub-agent reports. Returns count applied."""
        applied = 0

        # Group issues by file
        by_file: dict[str, list[dict[str, Any]]] = {}
        for issue in issues:
            fpath = issue.get("file", "")
            if fpath:
                by_file.setdefault(fpath, []).append(issue)

        for rel_path, file_issues in by_file.items():
            full_path = output_dir / rel_path
            if not full_path.exists():
                logger.warning(f"  File not found: {rel_path}")
                continue

            source = full_path.read_text(encoding="utf-8")
            modified = False

            for issue in file_issues:
                issue_type = issue.get("type", "")

                if issue_type == "duplicate_method":
                    new_source = self._fix_duplicate_method(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "orphaned_annotation":
                    new_source = self._fix_orphaned_annotation(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "missing_import":
                    new_source = self._fix_missing_import(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "incorrect_reference":
                    new_source = self._fix_incorrect_reference(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "unused_import":
                    new_source = self._fix_unused_import(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "enum_str_error":
                    new_source = self._fix_enum_str(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "field_method_collision":
                    new_source = self._fix_field_method_collision(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type == "missing_self":
                    new_source = self._fix_missing_self(source, issue)
                    if new_source and new_source != source:
                        source = new_source
                        modified = True
                        applied += 1

                elif issue_type in ("runtime_error", "signature_mismatch",
                                    "import_error", "logic_error"):
                    # Logged for sub-agent handling; no mechanical fix
                    logger.info(
                        f"  [{issue_type}] {issue.get('class_name', '?')}"
                        f".{issue.get('method', '?')}: {issue.get('description', '')[:100]}"
                    )

            if modified:
                full_path.write_text(source, encoding="utf-8")
                logger.info(f"  Updated {rel_path}")

        return applied

    def _fix_duplicate_method(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Remove duplicate method definitions using AST."""
        method_name = issue.get("method")
        if not method_name:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()
        class_name = issue.get("class_name")

        # Find all FunctionDef with this name inside the target class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if class_name and node.name != class_name:
                    continue

                occurrences: list[ast.FunctionDef] = []
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if stmt.name == method_name:
                            occurrences.append(stmt)

                if len(occurrences) < 2:
                    continue

                # Keep the first occurrence, remove the rest
                # (The plan says "keep" field may specify, default to first)
                keep = issue.get("keep", "first")
                to_remove = occurrences[1:] if keep == "first" else occurrences[:-1]

                # Build set of line indices to remove
                remove_lines: set[int] = set()
                for func in to_remove:
                    start = func.lineno - 1
                    end = func.end_lineno if func.end_lineno else start + 1
                    # Also remove blank lines immediately before the def
                    while start > 0 and not lines[start - 1].strip():
                        start -= 1
                    for i in range(start, end):
                        remove_lines.add(i)

                new_lines = [
                    line for i, line in enumerate(lines)
                    if i not in remove_lines
                ]
                return "\n".join(new_lines)

        return None

    def _fix_orphaned_annotation(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Remove an orphaned annotation line."""
        line_num = issue.get("line")
        current = issue.get("current", "")
        if not line_num and not current:
            return None

        lines = source.splitlines()

        if line_num and 1 <= line_num <= len(lines):
            # Verify the line matches what we expect
            actual = lines[line_num - 1].strip()
            if current and current.strip() not in actual:
                return None
            del lines[line_num - 1]
            return "\n".join(lines)

        # Fallback: search for the current text
        if current:
            for i, line in enumerate(lines):
                if current.strip() in line.strip():
                    del lines[i]
                    return "\n".join(lines)

        return None

    def _fix_missing_import(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Add a missing import line after existing imports."""
        suggested = issue.get("suggested", issue.get("suggested_import", ""))
        if not suggested:
            return None

        lines = source.splitlines()

        # Check if this import already exists
        for line in lines:
            if suggested.strip() in line.strip():
                return None

        # Find the last import line and insert after it
        last_import_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                last_import_idx = i

        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, suggested.strip())
        else:
            # No imports found — add at top (after any docstrings / future imports)
            insert_at = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    insert_at = i + 1
                elif stripped.startswith("from __future__"):
                    insert_at = i + 1
                elif stripped == "" and i < 3:
                    insert_at = i + 1
                else:
                    break
            lines.insert(insert_at, suggested.strip())

        return "\n".join(lines)

    def _fix_incorrect_reference(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Fix an incorrect reference via string replacement at the specified line.

        Uses word-boundary matching when ``current`` is a simple identifier
        to avoid replacing inside ``self.ROWS`` or other prefixed forms.

        IMPORTANT: Never replaces class-level constant definitions
        (``COLS = 3``) with ``self.COLS = 3`` — only replaces bare names
        that appear in expression/usage context inside method bodies.
        """
        current = issue.get("current", "")
        suggested = issue.get("suggested", "")
        line_num = issue.get("line")

        if not current or not suggested:
            return None

        lines = source.splitlines()

        # Build a word-boundary regex if current is a bare identifier
        is_bare_ident = re.fullmatch(r"[A-Za-z_]\w*", current) is not None
        if is_bare_ident:
            # Match bare name NOT preceded by . or alphanumeric (avoids self.X, cls.X)
            pattern = re.compile(r"(?<![.\w])" + re.escape(current) + r"(?!\w)")
        else:
            pattern = None

        if line_num and 1 <= line_num <= len(lines):
            old_line = lines[line_num - 1]
            # Guard: never change a class-level constant definition
            if self._is_class_level_definition(old_line, current):
                return None
            if pattern:
                new_line, n = pattern.subn(suggested, old_line, count=1)
                if n > 0:
                    lines[line_num - 1] = new_line
                    return "\n".join(lines)
            elif current in old_line:
                lines[line_num - 1] = old_line.replace(current, suggested, 1)
                return "\n".join(lines)

        # Fallback: find the first occurrence in method bodies (not class-level)
        if is_bare_ident and pattern:
            for i, line in enumerate(lines):
                if self._is_class_level_definition(line, current):
                    continue
                new_line, n = pattern.subn(suggested, line, count=1)
                if n > 0:
                    lines[i] = new_line
                    return "\n".join(lines)
        elif current in source:
            # Only use raw replacement if NOT a bare ident (multi-char expressions)
            for i, line in enumerate(lines):
                if self._is_class_level_definition(line, current):
                    continue
                if current in line:
                    lines[i] = line.replace(current, suggested, 1)
                    return "\n".join(lines)

        return None

    @staticmethod
    def _is_class_level_definition(line: str, name: str) -> bool:
        """Check if a line is a class-level constant definition for ``name``.

        Matches patterns like ``    COLS = 3``, ``    ROWS: Final = 3``.
        These must NOT be prefixed with ``self.`` — they're valid as bare
        class-level assignments.
        """
        stripped = line.lstrip()
        # Assignment: NAME = value
        if stripped.startswith(f"{name} =") or stripped.startswith(f"{name}="):
            return True
        # Annotated: NAME: Type = value
        if stripped.startswith(f"{name}:"):
            return True
        return False

    def _fix_unused_import(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Remove an unused import line."""
        current = issue.get("current", "")
        line_num = issue.get("line")

        lines = source.splitlines()

        if line_num and 1 <= line_num <= len(lines):
            actual = lines[line_num - 1].strip()
            if current and current.strip() not in actual:
                return None
            del lines[line_num - 1]
            return "\n".join(lines)

        if current:
            for i, line in enumerate(lines):
                if current.strip() == line.strip():
                    del lines[i]
                    return "\n".join(lines)

        return None

    # ------------------------------------------------------------------
    # Runtime verification
    # ------------------------------------------------------------------

    def _run_runtime_verification(self, output_dir: Path) -> list[dict[str, Any]]:
        """Run mechanical runtime smoke tests on the assembled output."""
        from codemorph.assembler.runtime_verifier import RuntimeVerifier

        # Discover package name from the first __init__.py
        init_files = list(output_dir.rglob("__init__.py"))
        if not init_files:
            logger.warning("No __init__.py found — skipping runtime verification")
            return []

        package_name = init_files[0].parent.name
        logger.info(f"Running runtime verification (package={package_name})...")
        try:
            return RuntimeVerifier(self.index, output_dir, package_name).run()
        except Exception as e:
            logger.warning(f"Runtime verification failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Mechanical fix: Enum __str__
    # ------------------------------------------------------------------

    def _fix_enum_str(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Inject ``def __str__(self): return self.name`` into an Enum class."""
        class_name = issue.get("class_name")
        if not class_name:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue

            # Find the indentation of the class body
            if not node.body:
                continue
            first_stmt = node.body[0]
            first_line = lines[first_stmt.lineno - 1]
            indent = first_line[: len(first_line) - len(first_line.lstrip())]

            # Insert __str__ method after the last member assignment (before methods)
            insert_after = node.body[-1].end_lineno
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    # Insert before the first method
                    insert_after = stmt.lineno - 1
                    # Back up past decorators
                    if stmt.decorator_list:
                        insert_after = stmt.decorator_list[0].lineno - 1
                    break

            str_method = [
                "",
                f"{indent}def __str__(self):",
                f"{indent}    return self.name",
            ]

            new_lines = lines[:insert_after] + str_method + lines[insert_after:]
            return "\n".join(new_lines)

        return None

    # ------------------------------------------------------------------
    # Mechanical fix: Field/method collision
    # ------------------------------------------------------------------

    def _fix_field_method_collision(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Remove ``self.X = ...`` from __init__ when it shadows ``def X()``."""
        method_name = issue.get("method")
        class_name = issue.get("class_name")
        if not method_name:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if class_name and node.name != class_name:
                continue

            for stmt in node.body:
                if not isinstance(stmt, ast.FunctionDef) or stmt.name != "__init__":
                    continue

                # Find the offending self.X = ... line
                for sub in ast.walk(stmt):
                    if not isinstance(sub, ast.Assign):
                        continue
                    for target in sub.targets:
                        if not isinstance(target, ast.Attribute):
                            continue
                        if not (isinstance(target.value, ast.Name)
                                and target.value.id == "self"):
                            continue
                        if target.attr == method_name:
                            # Remove this line
                            line_idx = sub.lineno - 1
                            if 0 <= line_idx < len(lines):
                                del lines[line_idx]
                                return "\n".join(lines)

        return None

    # ------------------------------------------------------------------
    # Mechanical fix: Missing self parameter
    # ------------------------------------------------------------------

    def _fix_missing_self(
        self, source: str, issue: dict[str, Any]
    ) -> str | None:
        """Add ``self`` as first parameter to an instance method."""
        method_name = issue.get("method")
        line_num = issue.get("line")
        if not method_name or not line_num:
            return None

        lines = source.splitlines()
        if line_num < 1 or line_num > len(lines):
            return None

        line = lines[line_num - 1]
        # Match def method_name(...) and insert self
        pattern = rf"(def\s+{re.escape(method_name)}\s*\()(.*?\))"
        m = re.search(pattern, line)
        if not m:
            return None

        prefix = m.group(1)
        params = m.group(2)

        if params.strip() == ")":
            # No params: def method() -> def method(self)
            new_line = line[:m.start()] + prefix + "self)" + line[m.end():]
        else:
            # Has params: def method(x, y) -> def method(self, x, y)
            new_line = line[:m.start()] + prefix + "self, " + params + line[m.end():]

        lines[line_num - 1] = new_line
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # OpenCode invocation (reuses same pattern as OpenCodeFixer)
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

        logger.info(f"Running OpenCode sub-agent (timeout={self.timeout}s)...")
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
                    stdout_lines = result.stdout.strip().splitlines()
                    for line in stdout_lines[:20]:
                        logger.info(f"  [opencode] {line}")
                    if len(stdout_lines) > 20:
                        logger.info(
                            f"  [opencode] ... ({len(stdout_lines) - 20} more lines)"
                        )
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[:10]:
                        logger.info(f"  [opencode:err] {line}")

            if result.returncode != 0:
                logger.warning(f"OpenCode exited with code {result.returncode}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.warning(f"OpenCode sub-agent timed out after {self.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Failed to run OpenCode sub-agent: {e}")
            return False
