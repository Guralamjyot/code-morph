"""
Phase 4 Orchestrator — Main Assembly Pipeline.

Coordinates the entire Phase 4 assembly process:
1. Build index from latest.json + symbol_registry
2. LLM decides project structure
3. LLM drafts each class with INJECT markers
4. Mechanical code injection
5. Semantic fixes (mechanical AST rewrites)
6. LLM fills mocked gaps
7. Write project to disk
8. Compile-fix loop
"""

import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from codemorph.assembler.class_drafter import ClassDrafter
from codemorph.assembler.code_injector import CodeInjector
from codemorph.assembler.compile_fixer import CompileFixer
from codemorph.assembler.gap_filler import GapFiller
from codemorph.assembler.index_builder import IndexBuilder, ProjectIndex
from codemorph.assembler.semantic_fixer import SemanticFixer
from codemorph.assembler.structure_agent import ProjectStructure, StructureAgent
from codemorph.assembler.tools import AgentToolRegistry
from codemorph.config.models import CodeMorphConfig

logger = logging.getLogger(__name__)
console = Console()


class Phase4Orchestrator:
    """Main Phase 4 pipeline: assemble translated fragments into a project."""

    def __init__(
        self,
        config: CodeMorphConfig,
        state_dir: Path,
        source_dir: Path | None = None,
        output_dir: Path | None = None,
        verbose: bool = True,
        fill_mocks: bool = True,
        max_fix_iterations: int = 5,
        use_opencode: bool = False,
        interactive: bool = False,
    ):
        self.config = config
        self.state_dir = Path(state_dir)
        self.source_dir = Path(source_dir) if source_dir else Path(config.project.source.root)
        self.output_dir = Path(output_dir) if output_dir else Path("./assembled")
        self.verbose = verbose
        self.fill_mocks = fill_mocks
        self.max_fix_iterations = max_fix_iterations
        self.use_opencode = use_opencode
        self.interactive = interactive

    def run(self) -> dict[str, Any]:
        """Run the full Phase 4 assembly pipeline.

        Returns:
            Summary dict with results from each step
        """
        start_time = time.time()
        results: dict[str, Any] = {}

        # Create LLM client
        from codemorph.translator.llm_client import create_llm_client

        llm_client = create_llm_client(self.config.llm)

        # =====================================================================
        # Step 1: Build index
        # =====================================================================
        console.print("\n[bold cyan]Step 1/8: Building project index...[/bold cyan]")

        index = IndexBuilder(self.state_dir, self.source_dir).build()

        total_translated = sum(c.translated_count for c in index.classes.values())
        total_mocked = sum(c.mocked_count for c in index.classes.values())

        console.print(f"  Found {len(index.classes)} classes, {len(index.fragments)} fragments")
        console.print(f"  {total_translated} translated, {total_mocked} mocked")

        results["index"] = {
            "classes": len(index.classes),
            "fragments": len(index.fragments),
            "translated": total_translated,
            "mocked": total_mocked,
        }

        if not index.classes:
            console.print("[bold red]No classes found in index. Aborting.[/bold red]")
            return results

        # Create tool registry (shared across all agents)
        tools = AgentToolRegistry(index, self.config, self.output_dir)

        # =====================================================================
        # Step 2: LLM decides project structure
        # =====================================================================
        console.print("\n[bold cyan]Step 2/8: LLM analyzing project structure...[/bold cyan]")

        structure_agent = StructureAgent(llm_client, verbose=self.verbose)
        structure = structure_agent.analyze(
            index, tools, project_name=self.config.project.name
        )

        # Validate: every class in the index must appear in a module
        all_structured = set()
        for classes in structure.modules.values():
            all_structured.update(classes)
        missing_classes = set(index.classes.keys()) - all_structured
        if missing_classes:
            console.print(
                f"  [yellow]Structure agent missed {len(missing_classes)} classes, "
                f"using fallback structure[/yellow]"
            )
            structure = structure_agent._fallback_structure(index)
            if self.config.project.name:
                structure.package_name = self.config.project.name

        console.print(f"  Package: {structure.package_name}")
        console.print(f"  Modules: {list(structure.modules.keys())}")

        results["structure"] = {
            "package_name": structure.package_name,
            "modules": structure.modules,
        }

        # Build module context map for imports
        module_context = {}
        for module_name, classes in structure.modules.items():
            for cls in classes:
                module_context[cls] = module_name

        # =====================================================================
        # Step 3: LLM drafts each class + Step 4: inject translated code
        # =====================================================================
        console.print("\n[bold cyan]Step 3/8: Drafting and assembling classes...[/bold cyan]")

        drafter = ClassDrafter(llm_client, verbose=self.verbose)
        injector = CodeInjector(index)
        skeletons: dict[str, str] = {}
        assembled: dict[str, str] = {}

        dep_order = index.get_dependency_order()
        console.print(f"  Dependency order: {dep_order}")

        for class_name in dep_order:
            console.print(f"  Drafting {class_name}...")
            skeleton = drafter.draft(class_name, index, tools, module_context)
            skeletons[class_name] = skeleton

            console.print(f"  Injecting code for {class_name}...")
            assembled_code = injector.inject(skeleton)
            assembled[class_name] = assembled_code

            inject_count = skeleton.count("# INJECT:")
            mocked_count = assembled_code.count("# MOCKED:") + assembled_code.count("# MISSING:")
            console.print(
                f"    {inject_count} fragments injected, "
                f"{mocked_count} mocked remaining"
            )

        results["drafting"] = {
            "classes_drafted": len(skeletons),
            "classes_assembled": len(assembled),
        }

        # =====================================================================
        # Step 5: Semantic fixes (mechanical AST rewrites)
        # =====================================================================
        console.print("\n[bold cyan]Step 5/8: Applying semantic fixes...[/bold cyan]")

        semantic_fixer = SemanticFixer(index)
        assembled, fix_reports = semantic_fixer.fix_all(assembled)

        applied_count = sum(1 for r in fix_reports.values() if r.any_applied)
        total_changes = sum(
            len(change)
            for r in fix_reports.values()
            for f in r.fixes
            for change in [f.changes]
        )
        console.print(f"  {applied_count}/{len(fix_reports)} classes had fixes applied")
        for cls_name, report in fix_reports.items():
            if report.any_applied:
                for fix in report.fixes:
                    if fix.applied:
                        for change in fix.changes:
                            console.print(f"    {cls_name}: {change}")

        results["semantic_fixes"] = {
            "classes_fixed": applied_count,
            "total_changes": total_changes,
            "details": {
                name: report.summary for name, report in fix_reports.items()
            },
        }

        # =====================================================================
        # Step 6: Fill mocked gaps
        # =====================================================================
        if self.fill_mocks:
            mocked_fragments = [
                f for f in index.fragments.values() if f.is_mocked
            ]
            if mocked_fragments:
                console.print(
                    f"\n[bold cyan]Step 6/8: Filling {len(mocked_fragments)} "
                    f"mocked fragments...[/bold cyan]"
                )
                gap_filler = GapFiller(llm_client, index=index, verbose=self.verbose)
                fills = gap_filler.fill_all(index, assembled, tools)

                if fills:
                    assembled = gap_filler.apply_fills(assembled, fills)
                    console.print(
                        f"  Filled {len(fills)}/{len(mocked_fragments)} mocked fragments"
                    )

                results["gap_filling"] = {
                    "mocked_total": len(mocked_fragments),
                    "filled": len(fills),
                }
            else:
                console.print("\n[bold cyan]Step 6/8: No mocked fragments to fill[/bold cyan]")
                results["gap_filling"] = {"mocked_total": 0, "filled": 0}

        else:
            console.print("\n[dim]Step 6/8: Skipped (--no-fill-mocks)[/dim]")
            results["gap_filling"] = {"skipped": True}

        # Post-gap-fill cleanup: reapply fragment ID reference fix
        post_fixer = SemanticFixer(index)
        for cls_name in assembled:
            code = assembled[cls_name]
            code, _ = post_fixer._fix_fragment_id_references(cls_name, code)
            code, _ = post_fixer._fix_broken_double_defs(cls_name, code)
            assembled[cls_name] = code

        # =====================================================================
        # Step 5: Write project to disk
        # =====================================================================
        console.print("\n[bold cyan]Step 7/8: Writing project to disk...[/bold cyan]")

        self._write_project(structure, assembled, index)

        console.print(f"  Written to {self.output_dir}")
        results["output_dir"] = str(self.output_dir)

        # =====================================================================
        # Step 6: Compile-fix loop
        # =====================================================================
        console.print("\n[bold cyan]Step 8/8: Compile & fix...[/bold cyan]")

        if self.use_opencode:
            from codemorph.assembler.qa_orchestrator import QAOrchestrator

            console.print("  [cyan]Using QA Orchestrator with sub-agents[/cyan]")
            qa = QAOrchestrator(
                config=self.config,
                index=index,
                state_dir=self.state_dir,
                source_dir=self.source_dir,
                verbose=self.verbose,
                interactive=self.interactive,
            )
            fix_results = qa.run(self.output_dir, max_iterations=self.max_fix_iterations)
        else:
            fixer = CompileFixer(llm_client, verbose=self.verbose)
            fix_results = fixer.run(
                self.output_dir, tools, max_iterations=self.max_fix_iterations
            )

        status_color = "green" if fix_results["final_status"] == "clean" else "yellow"
        console.print(
            f"  Status: [{status_color}]{fix_results['final_status']}[/{status_color}]"
        )
        console.print(f"  Iterations: {fix_results['iterations']}")

        if fix_results.get("remaining_errors"):
            console.print(f"  Remaining errors: {len(fix_results['remaining_errors'])}")
            for err in fix_results["remaining_errors"][:5]:
                console.print(f"    - {err}")

        results["compile_fix"] = fix_results

        # =====================================================================
        # Summary
        # =====================================================================
        elapsed = time.time() - start_time

        console.print(
            Panel(
                f"[bold green]Phase 4 Assembly Complete[/bold green]\n\n"
                f"Classes assembled: {len(assembled)}\n"
                f"Fragments injected: {total_translated}\n"
                f"Mocked filled: {results.get('gap_filling', {}).get('filled', 0)}\n"
                f"Compile status: {fix_results['final_status']}\n"
                f"Output: {self.output_dir}\n"
                f"Time: {elapsed:.1f}s",
                title="Phase 4 Summary",
                border_style="green",
            )
        )

        results["elapsed_seconds"] = elapsed
        return results

    def _fix_module_imports(
        self,
        module_content: str,
        module_name: str,
        structure: ProjectStructure,
    ) -> str:
        """Fix imports in assembled module content.

        - Rewrites bare 'from module import X' to 'from .module import X'
        - Removes self-imports (importing from the module's own name)
        - Removes dotted Java-style imports (from interfaces.IValue0 import ...)
        - Removes unused imports (names not referenced in the code body)
        - Deduplicates import lines
        - Consolidates imports to the top of the file
        """
        import re

        all_modules = set(structure.modules.keys())
        class_to_module = {}
        for mod, classes in structure.modules.items():
            for cls in classes:
                class_to_module[cls.lower()] = mod
                class_to_module[cls] = mod

        lines = module_content.splitlines()
        # Candidate imports: (fixed_line, set_of_imported_names)
        candidate_imports: list[tuple[str, set[str]]] = []
        code_lines: list[str] = []
        seen_imports: set[str] = set()

        for line in lines:
            stripped = line.strip()

            # Match 'from X import Y, Z, ...'
            from_match = re.match(
                r"^from\s+([a-zA-Z_][\w.]*)\s+import\s+(.+)$", stripped
            )
            if from_match:
                from_module = from_match.group(1)
                imports_str = from_match.group(2).strip()
                imported_names = {
                    n.strip().split(" as ")[-1].strip()
                    for n in imports_str.split(",")
                    if n.strip()
                }

                # Skip __future__ imports (we add our own)
                if from_module == "__future__":
                    continue

                # Remove Java-style dotted submodule imports
                if "." in from_module:
                    base = from_module.split(".")[0]
                    if base in all_modules or base.lower() in class_to_module:
                        continue

                # Determine the fixed import line
                if from_module in all_modules:
                    if from_module == module_name:
                        continue  # Self-import
                    fixed = f"from .{from_module} import {imports_str}"
                elif from_module in class_to_module:
                    target_mod = class_to_module[from_module]
                    if target_mod == module_name:
                        continue
                    fixed = f"from .{target_mod} import {imports_str}"
                else:
                    # Standard library / third-party — keep as-is
                    fixed = stripped

                if fixed not in seen_imports:
                    seen_imports.add(fixed)
                    candidate_imports.append((fixed, imported_names))
                continue

            # Match 'import X'
            import_match = re.match(r"^import\s+([a-zA-Z_][\w.]*)", stripped)
            if import_match:
                mod = import_match.group(1)
                name = mod.split(".")[-1]
                if mod in all_modules and mod != module_name:
                    fixed = f"from . import {mod}"
                else:
                    fixed = stripped
                if fixed not in seen_imports:
                    seen_imports.add(fixed)
                    candidate_imports.append((fixed, {name}))
                continue

            # Catch malformed import lines the strict regex missed
            if stripped.startswith("from ") and " import " in stripped:
                continue  # Skip broken imports
            if stripped.startswith("import ") and not stripped.startswith("import("):
                continue

            code_lines.append(line)

        # Build the code body text to check which names are actually used.
        # Strip string literals to avoid false positives from names in error messages.
        code_body = "\n".join(code_lines)
        code_body_no_strings = re.sub(r"(['\"]).*?\1", '""', code_body)

        # Filter imports: keep only those where at least one imported name
        # appears in the code body. Always keep stdlib/typing imports.
        stdlib_prefixes = (
            "from typing", "from abc",
            "from collections", "from dataclasses", "from enum",
            "from functools", "from itertools", "import abc",
            "import typing", "from types ",
        )
        import_lines: list[str] = []
        for fixed_line, names in candidate_imports:
            # Always keep stdlib/typing imports
            if any(fixed_line.startswith(p) for p in stdlib_prefixes):
                import_lines.append(fixed_line)
                continue
            # Keep if any imported name is referenced in code (outside strings)
            if any(re.search(r'\b' + re.escape(n) + r'\b', code_body_no_strings) for n in names):
                import_lines.append(fixed_line)

        # Add missing imports for class base classes
        # The base fixer may have added bases that aren't imported yet
        already_imported = set()
        for _, names in candidate_imports:
            already_imported.update(names)
        for line in code_lines:
            m = re.match(r"^\s*class\s+\w+\s*\(([^)]+)\)\s*:", line)
            if m:
                for base in m.group(1).split(","):
                    base = base.strip()
                    if base and base not in already_imported and base in class_to_module:
                        target_mod = class_to_module[base]
                        if target_mod != module_name:
                            imp = f"from .{target_mod} import {base}"
                            if imp not in seen_imports:
                                seen_imports.add(imp)
                                import_lines.append(imp)
                                already_imported.add(base)

        # Reassemble: future import first, then other imports, then code
        result_parts = ["from __future__ import annotations", ""]
        if import_lines:
            result_parts.extend(import_lines)
            result_parts.append("")
        # Strip leading blank lines from code
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        result_parts.extend(code_lines)
        return "\n".join(result_parts)

    def _fix_class_bases(
        self,
        module_content: str,
        index: ProjectIndex,
    ) -> str:
        """Fix class base classes to match actual hierarchy from the index.

        The LLM may invent inheritance relationships or omit them. This
        method rewrites class declarations to use the actual hierarchy
        from the Java source.
        """
        import re

        lines = module_content.splitlines()
        result = []
        for line in lines:
            # Match class declarations with bases: class Foo(Bar, Baz):
            m = re.match(r"^(\s*)class\s+(\w+)\s*\(([^)]*)\)\s*:", line)
            if not m:
                # Match class declarations without bases: class Foo:
                m = re.match(r"^(\s*)class\s+(\w+)\s*:", line)
            if m:
                indent = m.group(1)
                class_name = m.group(2)
                if class_name in index.hierarchy:
                    # Only include bases that are classes in our project index
                    actual_bases = [
                        b for b in index.hierarchy[class_name]
                        if b in index.classes
                    ]
                    if actual_bases:
                        bases_str = ", ".join(actual_bases)
                        result.append(f"{indent}class {class_name}({bases_str}):")
                    else:
                        result.append(f"{indent}class {class_name}:")
                    continue
            result.append(line)
        return "\n".join(result)

    def _write_project(
        self,
        structure: ProjectStructure,
        assembled: dict[str, str],
        index: ProjectIndex,
    ) -> None:
        """Write the assembled project to disk."""
        package_dir = self.output_dir / structure.package_name
        package_dir.mkdir(parents=True, exist_ok=True)

        # Write each module file
        for module_name, class_names in structure.modules.items():
            module_path = package_dir / f"{module_name}.py"

            # Collect assembled code for all classes in this module
            module_parts = []
            for class_name in class_names:
                if class_name in assembled:
                    module_parts.append(assembled[class_name])
                else:
                    module_parts.append(f"# TODO: {class_name} not assembled\n")

            module_content = "\n\n".join(module_parts)

            # Fix class base classes to match actual hierarchy
            module_content = self._fix_class_bases(
                module_content, index
            )

            # Fix imports: bare → relative, dedup, remove self-imports, remove unused
            module_content = self._fix_module_imports(
                module_content, module_name, structure
            )

            module_path.write_text(module_content, encoding="utf-8")
            logger.info(f"  Written {module_path}")

        # Write __init__.py
        init_lines = [
            f'"""',
            f"{structure.package_name} — auto-assembled by CodeMorph Phase 4.",
            f'"""',
            "",
        ]

        for module_name, class_names in structure.modules.items():
            class_imports = ", ".join(class_names)
            init_lines.append(
                f"from {structure.package_name}.{module_name} import {class_imports}"
            )

        init_lines.append("")
        init_lines.append("__all__ = [")
        for export in structure.init_exports:
            init_lines.append(f'    "{export}",')
        init_lines.append("]")
        init_lines.append("")

        init_path = package_dir / "__init__.py"
        init_path.write_text("\n".join(init_lines), encoding="utf-8")
        logger.info(f"  Written {init_path}")

        # Write a minimal pyproject.toml or setup marker
        setup_path = self.output_dir / "pyproject.toml"
        if not setup_path.exists():
            setup_path.write_text(
                f'[project]\nname = "{structure.package_name}"\nversion = "0.1.0"\n'
                f'requires-python = ">=3.10"\n',
                encoding="utf-8",
            )
