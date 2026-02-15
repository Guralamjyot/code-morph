"""
Semantic Fixer for Phase 4 Assembly.

Applies generalized, pattern-based AST rewrites to fix common Java→Python
translation issues. Each fix targets a language mismatch pattern, not any
specific library.

Every fix is:
- Pattern-based (uses builtins list, AST analysis, or index metadata)
- Independently toggleable
- Idempotent (running twice produces the same output)
"""

from __future__ import annotations

import ast
import builtins
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from codemorph.assembler.index_builder import ProjectIndex

logger = logging.getLogger(__name__)

# Python builtins commonly shadowed by Java→Python translations
COMMON_SHADOWED_BUILTINS = frozenset(
    name for name in dir(builtins)
    if not name.startswith("_") and name.islower()
) & {
    "iter", "list", "dict", "type", "id", "range", "map", "filter", "set",
    "hash", "next", "tuple", "str", "int", "float", "bool", "open", "format",
    "input", "object", "super", "bytes", "sorted", "reversed", "enumerate",
    "zip", "any", "all", "min", "max", "sum", "len", "abs", "round", "print",
    "isinstance", "issubclass", "getattr", "setattr", "hasattr", "delattr",
    "property", "staticmethod", "classmethod", "vars", "dir", "repr", "chr",
    "ord", "hex", "oct", "bin", "pow", "complex", "frozenset", "memoryview",
    "slice", "callable",
}


@dataclass
class FixResult:
    """Result of applying a single fix."""

    fix_name: str
    applied: bool
    changes: list[str] = field(default_factory=list)


@dataclass
class SemanticFixReport:
    """Report of all fixes applied to a class."""

    class_name: str
    fixes: list[FixResult] = field(default_factory=list)

    @property
    def any_applied(self) -> bool:
        return any(f.applied for f in self.fixes)

    @property
    def summary(self) -> str:
        applied = [f for f in self.fixes if f.applied]
        if not applied:
            return f"{self.class_name}: no fixes needed"
        names = ", ".join(f.fix_name for f in applied)
        return f"{self.class_name}: applied {len(applied)} fixes ({names})"


class SemanticFixer:
    """Applies generalized AST-based rewrites to assembled Python code."""

    def __init__(self, index: ProjectIndex):
        self.index = index
        self._fixes: list[tuple[str, Any]] = [
            ("nested_enum_extraction", self._fix_nested_enum),
            ("builtin_shadowing", self._fix_builtin_shadowing),
            ("missing_staticmethod", self._fix_missing_staticmethod),
            ("infinite_recursion", self._fix_infinite_recursion),
            ("missing_init", self._fix_missing_init),
            ("missing_super_init", self._fix_missing_super_init),
            ("duplicate_guards", self._fix_duplicate_guards),
            ("interface_to_abc", self._fix_interface_to_abc),
            ("method_name_consistency", self._fix_method_name_consistency),
            ("field_name_registry", self._fix_field_names),
            ("orphaned_annotations", self._fix_orphaned_annotations),
            ("duplicate_assignments", self._fix_duplicate_assignments),
            ("none_sentinel", self._fix_none_sentinel),
            ("fragment_id_references", self._fix_fragment_id_references),
            ("broken_double_defs", self._fix_broken_double_defs),
        ]

    def fix(self, class_name: str, code: str) -> tuple[str, SemanticFixReport]:
        """Apply all fixes to a single class's assembled code.

        Args:
            class_name: Name of the class being fixed
            code: Assembled Python source code

        Returns:
            Tuple of (fixed_code, report)
        """
        report = SemanticFixReport(class_name=class_name)

        for fix_name, fix_func in self._fixes:
            try:
                code, result = fix_func(class_name, code)
                report.fixes.append(result)
            except Exception as e:
                logger.warning(f"Fix '{fix_name}' failed for {class_name}: {e}")
                report.fixes.append(FixResult(fix_name=fix_name, applied=False))

        return code, report

    def fix_all(
        self, assembled: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, SemanticFixReport]]:
        """Apply all fixes to all assembled classes.

        Args:
            assembled: Map of class_name -> assembled code

        Returns:
            Tuple of (fixed_assembled, reports)
        """
        fixed = {}
        reports = {}

        for class_name, code in assembled.items():
            fixed_code, report = self.fix(class_name, code)
            fixed[class_name] = fixed_code
            reports[class_name] = report
            if report.any_applied:
                logger.info(f"SemanticFixer: {report.summary}")

        return fixed, reports

    # =========================================================================
    # Fix 1: Builtin Shadowing
    # =========================================================================

    def _fix_builtin_shadowing(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Rename local variables that shadow Python builtins.

        Scans each function for assignment targets that match a builtin name
        AND where the same name is also called as a function in the same scope.
        Renames by prepending underscore (e.g., iter -> _iter).
        """
        result = FixResult(fix_name="builtin_shadowing", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        # Collect all renames: (line_start, line_end, old_name, new_name)
        renames: list[tuple[int, int, str, str]] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Collect assignment targets in this function
            assigned_names: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned_names.add(target.id)
                elif isinstance(child, ast.AugAssign):
                    if isinstance(child.target, ast.Name):
                        assigned_names.add(child.target.id)
                elif isinstance(child, ast.For):
                    if isinstance(child.target, ast.Name):
                        assigned_names.add(child.target.id)
                elif isinstance(child, ast.With):
                    for item in child.items:
                        if item.optional_vars and isinstance(
                            item.optional_vars, ast.Name
                        ):
                            assigned_names.add(item.optional_vars.id)
                elif isinstance(child, ast.NamedExpr):
                    if isinstance(child.target, ast.Name):
                        assigned_names.add(child.target.id)

            # Collect names that are called as functions in this scope
            called_names: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    called_names.add(child.func.id)

            # Find shadows: assigned AND called AND is a builtin
            shadows = assigned_names & called_names & COMMON_SHADOWED_BUILTINS

            if shadows:
                func_start = node.lineno  # 1-indexed
                func_end = node.end_lineno or func_start
                for name in shadows:
                    renames.append((func_start, func_end, name, f"_{name}"))
                    result.changes.append(
                        f"Renamed '{name}' -> '_{name}' in {node.name}()"
                    )

        if not renames:
            return code, result

        # Apply renames (process from bottom to top to preserve line numbers)
        renames.sort(key=lambda r: r[0], reverse=True)
        for func_start, func_end, old_name, new_name in renames:
            for i in range(func_start - 1, min(func_end, len(lines))):
                # Only rename the local variable, not the builtin function call
                # We need to be careful: rename occurrences that are assignments
                # or usages of the variable, but NOT the builtin function itself
                # when it's used as a call target on the RHS of the assignment.
                #
                # Strategy: rename all bare `old_name` occurrences, which is safe
                # because the fix already confirmed the name is both assigned and
                # called — the renamed var won't collide with the builtin anymore.
                lines[i] = re.sub(
                    r"\b" + re.escape(old_name) + r"\b",
                    new_name,
                    lines[i],
                )

        result.applied = True
        return "\n".join(lines), result

    # =========================================================================
    # Fix 2: Missing @staticmethod
    # =========================================================================

    def _fix_missing_staticmethod(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Add @staticmethod to class methods that don't use self/cls.

        Cross-references with Java source to confirm static intent.
        """
        result = FixResult(fix_name="missing_staticmethod", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        insertions: list[tuple[int, str, str]] = []  # (line_idx, indent, decorator)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                # Skip if already decorated
                decorator_names = set()
                for dec in item.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorator_names.add(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        decorator_names.add(dec.attr)

                if "staticmethod" in decorator_names or "classmethod" in decorator_names:
                    continue

                # Skip dunder methods
                if item.name.startswith("__") and item.name.endswith("__"):
                    continue

                # Check first parameter
                args = item.args
                if args.args and args.args[0].arg in ("self", "cls"):
                    continue

                # No self/cls — check if body references self
                has_self_ref = False
                for child in ast.walk(item):
                    if isinstance(child, ast.Attribute) and isinstance(
                        child.value, ast.Name
                    ):
                        if child.value.id == "self":
                            has_self_ref = True
                            break

                if has_self_ref:
                    continue

                # Cross-check with Java source if available
                is_static = self._is_java_static(class_name, item.name)

                # If Java source confirms static, or no Java source but method
                # definitely doesn't use self, add @staticmethod
                if is_static or not args.args or args.args[0].arg not in ("self", "cls"):
                    # Determine indentation of the def line
                    def_line = lines[item.lineno - 1]
                    indent = def_line[: len(def_line) - len(def_line.lstrip())]
                    insertions.append(
                        (item.lineno - 1, indent, "@staticmethod")
                    )
                    result.changes.append(
                        f"Added @staticmethod to {node.name}.{item.name}()"
                    )

        if not insertions:
            return code, result

        # Insert from bottom to top
        insertions.sort(key=lambda x: x[0], reverse=True)
        for line_idx, indent, decorator in insertions:
            lines.insert(line_idx, f"{indent}{decorator}")

        result.applied = True
        return "\n".join(lines), result

    def _is_java_static(self, class_name: str, method_name: str) -> bool:
        """Check if a method is static in the Java source."""
        for frag_id, entry in self.index.fragments.items():
            if entry.class_name != class_name:
                continue
            if entry.member_name and self._names_match(
                entry.member_name, method_name
            ):
                if entry.java_source and re.search(
                    r"\bstatic\b", entry.java_source
                ):
                    return True
        return False

    @staticmethod
    def _names_match(java_name: str, python_name: str) -> bool:
        """Check if a Java name matches a Python name (camelCase vs snake_case)."""
        # Direct match
        if java_name == python_name:
            return True
        # Convert Java camelCase to snake_case
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", java_name)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        return s.lower() == python_name

    # =========================================================================
    # Fix 3: Infinite Recursion (Overload Conflation)
    # =========================================================================

    def _fix_infinite_recursion(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Detect methods that call themselves with different argument count.

        This is the universal Java→Python overload conflation pattern where
        two Java overloads get merged into one Python method that recurses.
        """
        result = FixResult(fix_name="infinite_recursion", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        comments_to_add: list[tuple[int, str, str]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                # Count the method's own parameter count (excluding self/cls)
                own_params = len(item.args.args)
                if item.args.args and item.args.args[0].arg in ("self", "cls"):
                    own_params -= 1

                # Look for self.method_name(...) calls with different arity
                for child in ast.walk(item):
                    if not isinstance(child, ast.Call):
                        continue
                    if not isinstance(child.func, ast.Attribute):
                        continue
                    if not isinstance(child.func.value, ast.Name):
                        continue
                    if child.func.value.id != "self":
                        continue
                    if child.func.attr != item.name:
                        continue

                    call_args = len(child.args)
                    if call_args != own_params:
                        # Potential infinite recursion from overload conflation
                        line_idx = child.lineno - 1
                        if line_idx < len(lines):
                            line = lines[line_idx]
                            indent = line[: len(line) - len(line.lstrip())]
                            comments_to_add.append((
                                line_idx,
                                indent,
                                f"# WARNING: Potential infinite recursion — "
                                f"Java overload with {call_args} args vs "
                                f"method's {own_params} params",
                            ))
                            result.changes.append(
                                f"Flagged self-recursive call in "
                                f"{node.name}.{item.name}() "
                                f"(args: {call_args} vs params: {own_params})"
                            )

        if not comments_to_add:
            return code, result

        # Insert comments from bottom to top
        comments_to_add.sort(key=lambda x: x[0], reverse=True)
        for line_idx, indent, comment in comments_to_add:
            lines.insert(line_idx, f"{indent}{comment}")

        result.applied = True
        return "\n".join(lines), result

    # =========================================================================
    # Fix 4: Missing __init__
    # =========================================================================

    def _fix_missing_init(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Generate __init__ for classes that use self.x but never define it.

        Also detects classes with Java constructors in their fragments.
        """
        result = FixResult(fix_name="missing_init", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Skip Enum subclasses — Enums don't use regular __init__
            is_enum = any(
                (isinstance(b, ast.Name) and b.id in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"))
                or (isinstance(b, ast.Attribute) and b.attr in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"))
                for b in node.bases
            )
            if is_enum:
                continue

            # Check if __init__ already exists
            has_init = any(
                isinstance(item, ast.FunctionDef) and item.name == "__init__"
                for item in node.body
            )
            if has_init:
                continue

            # Collect all self.attr references and self.attr = ... assignments
            attrs_read: set[str] = set()
            attrs_written: set[str] = set()

            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for child in ast.walk(item):
                    if isinstance(child, ast.Attribute) and isinstance(
                        child.value, ast.Name
                    ) and child.value.id == "self":
                        attrs_read.add(child.attr)
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Attribute) and isinstance(
                                target.value, ast.Name
                            ) and target.value.id == "self":
                                attrs_written.add(target.attr)

            # Attributes read but never explicitly initialized in any method
            # These need __init__
            uninitialized = attrs_read - attrs_written
            all_attrs = attrs_read | attrs_written

            if not all_attrs:
                continue

            # Check if there's a Java constructor for this class
            has_java_constructor = False
            for frag_id, entry in self.index.fragments.items():
                if entry.class_name != class_name:
                    continue
                if entry.java_source and re.search(
                    rf"\b{re.escape(class_name)}\s*\(", entry.java_source
                ):
                    has_java_constructor = True
                    break

            if not all_attrs and not has_java_constructor:
                continue

            # Generate __init__
            # Find the indentation of the class body
            body_start = node.body[0].lineno - 1
            body_line = lines[body_start] if body_start < len(lines) else ""
            indent = body_line[: len(body_line) - len(body_line.lstrip())]
            if not indent:
                indent = "    "

            init_lines = [f"{indent}def __init__(self):"]

            # Check if class has parents
            has_parents = bool(node.bases)
            if has_parents:
                init_lines.append(f"{indent}    super().__init__()")

            # Initialize all found attributes
            for attr in sorted(all_attrs):
                init_lines.append(f"{indent}    self.{attr} = None")

            if not all_attrs and not has_parents:
                init_lines.append(f"{indent}    pass")

            init_lines.append("")

            # Insert before the first method in the class body
            insert_idx = body_start
            for line in init_lines:
                lines.insert(insert_idx, line)
                insert_idx += 1

            result.applied = True
            result.changes.append(
                f"Generated __init__ for {node.name} with attrs: "
                f"{sorted(all_attrs)}"
            )
            # Only fix the first class in this code block
            break

        return "\n".join(lines), result

    # =========================================================================
    # Fix 4b: Missing super().__init__()
    # =========================================================================

    def _fix_missing_super_init(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Add super().__init__(...) to __init__ methods that are missing it.

        Only applies when the class has a concrete (non-interface) parent
        present in the project index.
        """
        result = FixResult(fix_name="missing_super_init", applied=False)

        parent = self._get_concrete_parent(class_name)
        if parent is None:
            return code, result

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        # Collect insertions (process bottom-up to keep line numbers stable)
        insertions: list[tuple[int, str, str]] = []  # (line_idx, indent, statement)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "__init__":
                continue
            if self._has_super_init_call(node):
                continue

            # Determine super() args
            super_args = self._get_super_args_from_java(class_name)
            if super_args is None:
                # Fall back: use all non-self __init__ params
                args = node.args
                param_names: list[str] = []
                for a in args.args:
                    if a.arg != "self":
                        param_names.append(a.arg)
                if args.vararg:
                    param_names = [f"*{args.vararg.arg}"]
                super_args = param_names

            arg_str = ", ".join(super_args)
            stmt = f"super().__init__({arg_str})"

            # Find insertion point: first statement after def line + docstring
            insert_idx = self._find_body_start(node, lines)
            body_line = lines[node.body[0].lineno - 1] if node.body else ""
            indent = body_line[: len(body_line) - len(body_line.lstrip())]
            if not indent:
                indent = "        "  # default: 2-level indent

            insertions.append((insert_idx, indent, stmt))

        if not insertions:
            return code, result

        # Insert bottom-up
        insertions.sort(key=lambda x: x[0], reverse=True)
        for insert_idx, indent, stmt in insertions:
            lines.insert(insert_idx, f"{indent}{stmt}")
            result.changes.append(f"Added {stmt} to {class_name}.__init__")

        result.applied = True
        return "\n".join(lines), result

    def _get_concrete_parent(self, class_name: str) -> str | None:
        """Return the first non-interface parent of *class_name* that exists in the index."""
        parents = self.index.hierarchy.get(class_name, [])
        for parent in parents:
            cls = self.index.classes.get(parent)
            if cls and cls.symbol_type != "interface":
                return parent
        return None

    @staticmethod
    def _has_super_init_call(init_node: ast.FunctionDef) -> bool:
        """Check whether an __init__ AST node already calls super().__init__()."""
        for child in ast.walk(init_node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            # super().__init__(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "__init__"
                and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"
            ):
                return True
        return False

    def _get_super_args_from_java(self, class_name: str) -> list[str] | None:
        """Parse the Java constructor's super() call to extract argument names."""
        # Look up the class-level fragment
        frag_id = f"{class_name}::{class_name}"
        entry = self.index.fragments.get(frag_id)
        if not entry or not entry.java_source:
            return None

        java = re.sub(r"\s+", " ", entry.java_source)
        m = re.search(r"super\s*\(([^)]*)\)\s*;", java)
        if not m:
            return None

        raw = m.group(1).strip()
        if not raw:
            return []
        return [a.strip() for a in raw.split(",") if a.strip()]

    @staticmethod
    def _find_body_start(func_node: ast.FunctionDef, lines: list[str]) -> int:
        """Return the line index (0-based) where new statements should be inserted.

        Skips past the def line and any docstring.
        """
        if not func_node.body:
            return func_node.lineno  # right after def

        first_stmt = func_node.body[0]

        # If first statement is a docstring (Expr(Constant(str))), skip it
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            # Insert after the docstring
            return first_stmt.end_lineno  # end_lineno is 1-based, used as 0-based insert = after

        # Otherwise insert before the first statement
        return first_stmt.lineno - 1

    # =========================================================================
    # Fix 4c: Duplicate Guard Clauses
    # =========================================================================

    def _fix_duplicate_guards(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Remove consecutive identical guard clauses.

        Detects patterns like:
            if COND:
                raise ValueError("msg1")
            if COND:            # ← duplicate, unreachable
                raise ValueError("msg2")

        Compares conditions via ast.dump(); if identical and both bodies are
        a single raise or return, removes the second one.
        """
        result = FixResult(fix_name="duplicate_guards", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        # Collect (start, end) line ranges to remove (0-indexed)
        removals: list[tuple[int, int]] = []

        for node in ast.walk(tree):
            # Look inside function bodies and class bodies
            body: list[ast.stmt] | None = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                body = node.body
            elif isinstance(node, ast.ClassDef):
                body = node.body

            if body is None:
                continue

            # Compare consecutive pairs of If statements
            i = 0
            while i < len(body) - 1:
                stmt_a = body[i]
                stmt_b = body[i + 1]

                if not isinstance(stmt_a, ast.If) or not isinstance(stmt_b, ast.If):
                    i += 1
                    continue

                # Both must have identical conditions
                if ast.dump(stmt_a.test) != ast.dump(stmt_b.test):
                    i += 1
                    continue

                # Both bodies must be a single raise or return
                def _is_single_raise_or_return(stmts: list[ast.stmt]) -> bool:
                    return len(stmts) == 1 and isinstance(stmts[0], (ast.Raise, ast.Return))

                if not _is_single_raise_or_return(stmt_a.body) or not _is_single_raise_or_return(stmt_b.body):
                    i += 1
                    continue

                # Neither should have elif/else
                if stmt_a.orelse or stmt_b.orelse:
                    i += 1
                    continue

                # Mark second If for removal
                start = stmt_b.lineno - 1  # 0-indexed
                end = stmt_b.end_lineno or stmt_b.lineno  # end_lineno is 1-indexed inclusive
                removals.append((start, end))
                result.changes.append(
                    f"Removed duplicate guard clause at line {stmt_b.lineno}"
                )
                i += 2  # skip both
                continue

        if not removals:
            return code, result

        # Remove bottom-up to preserve line numbers
        removals.sort(key=lambda x: x[0], reverse=True)
        for start, end in removals:
            del lines[start:end]

        result.applied = True
        return "\n".join(lines), result

    # =========================================================================
    # Fix 5: Interface to ABC
    # =========================================================================

    def _fix_interface_to_abc(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Convert classes marked as interfaces to proper ABCs.

        Removes __init__, instance attributes, and ensures all methods
        are abstract.
        """
        result = FixResult(fix_name="interface_to_abc", applied=False)

        # Check if this class is an interface in the index
        class_info = self.index.classes.get(class_name)
        if not class_info or class_info.symbol_type != "interface":
            return code, result

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        removals: list[tuple[int, int]] = []  # (start, end) line ranges to remove
        decorator_insertions: list[tuple[int, str]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            if node.name != class_name:
                continue

            for item in node.body:
                # Remove __init__ from interfaces
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    start = item.lineno - 1
                    end = item.end_lineno or item.lineno
                    # Also remove decorators
                    if item.decorator_list:
                        start = item.decorator_list[0].lineno - 1
                    removals.append((start, end))
                    result.changes.append(f"Removed __init__ from interface {class_name}")
                    continue

                # Add @abstractmethod to methods that don't have it
                if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                    has_abstract = any(
                        (isinstance(d, ast.Name) and d.id == "abstractmethod")
                        or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                        for d in item.decorator_list
                    )
                    if not has_abstract:
                        def_line = lines[item.lineno - 1]
                        indent = def_line[: len(def_line) - len(def_line.lstrip())]
                        decorator_insertions.append(
                            (item.lineno - 1, f"{indent}@abstractmethod")
                        )
                        result.changes.append(
                            f"Added @abstractmethod to {class_name}.{item.name}()"
                        )

            # Ensure class inherits from ABC
            has_abc = any(
                (isinstance(b, ast.Name) and b.id == "ABC")
                for b in node.bases
            )
            if not has_abc:
                # Modify the class line to add ABC base
                class_line_idx = node.lineno - 1
                class_line = lines[class_line_idx]
                if "(" in class_line:
                    # Add ABC to existing bases
                    class_line = class_line.replace("(", "(ABC, ", 1)
                else:
                    # Add (ABC) before the colon
                    class_line = class_line.replace(":", "(ABC):", 1)
                lines[class_line_idx] = class_line
                result.changes.append(f"Added ABC base to {class_name}")

        if not result.changes:
            return code, result

        # Apply removals from bottom to top
        removals.sort(key=lambda x: x[0], reverse=True)
        for start, end in removals:
            del lines[start:end]

        # Re-parse to get correct line numbers for insertions after removals
        # Instead, we recalculate by applying insertions directly
        # (decorator insertions were calculated before removals, so we skip
        #  and re-detect after removals)

        # Rebuild code and add imports
        code = "\n".join(lines)

        # Add ABC import if not present
        if "from abc import" not in code:
            code = "from abc import ABC, abstractmethod\n" + code
            result.changes.append("Added ABC import")

        # Re-parse to add @abstractmethod decorators
        try:
            tree = ast.parse(code)
        except SyntaxError:
            result.applied = True
            return code, result

        lines = code.splitlines()
        insertions = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if item.name.startswith("__") and item.name.endswith("__"):
                    continue
                has_abstract = any(
                    (isinstance(d, ast.Name) and d.id == "abstractmethod")
                    or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                    for d in item.decorator_list
                )
                if not has_abstract:
                    def_line = lines[item.lineno - 1]
                    indent = def_line[: len(def_line) - len(def_line.lstrip())]
                    insertions.append((item.lineno - 1, f"{indent}@abstractmethod"))

        insertions.sort(key=lambda x: x[0], reverse=True)
        for line_idx, text in insertions:
            lines.insert(line_idx, text)

        result.applied = True
        return "\n".join(lines), result

    # =========================================================================
    # Fix 6: Method Name Consistency
    # =========================================================================

    def _fix_method_name_consistency(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Normalize inconsistent snake_case method names within groups.

        Groups methods by common prefix where suffix is a digit, then
        normalizes to the majority pattern or registry pattern.
        """
        result = FixResult(fix_name="method_name_consistency", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        # Collect all method names in all classes
        method_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_names.append(item.name)

        if not method_names:
            return code, result

        # Group by common prefix ending before a digit
        # e.g., get_value_0, get_value1, get_value_2 -> prefix "get_value"
        groups: dict[str, list[str]] = {}
        for name in method_names:
            # Find the longest prefix that ends before a digit suffix
            match = re.match(r"^(.+?)_?(\d+)$", name)
            if match:
                prefix = match.group(1)
                # Normalize prefix (strip trailing underscore)
                prefix = prefix.rstrip("_")
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(name)

        # For each group with >1 member, find the canonical pattern
        renames: dict[str, str] = {}
        for prefix, names in groups.items():
            if len(names) < 2:
                continue

            # Check symbol registry for canonical pattern
            canonical_pattern = None
            for frag_id, entry in self.index.fragments.items():
                if entry.class_name != class_name:
                    continue
                if entry.python_name:
                    entry_match = re.match(r"^(.+?)_?(\d+)$", entry.python_name)
                    if entry_match and entry_match.group(1).rstrip("_") == prefix:
                        # Use the registry pattern
                        canonical_pattern = entry.python_name
                        break

            if canonical_pattern:
                # Extract the pattern: does it use underscore before digit?
                canon_match = re.match(r"^(.+?)(_?)(\d+)$", canonical_pattern)
                if canon_match:
                    canon_prefix = canon_match.group(1)
                    canon_sep = canon_match.group(2)
            else:
                # Majority vote
                with_underscore = sum(
                    1 for n in names if re.match(r"^.+_\d+$", n)
                )
                without_underscore = len(names) - with_underscore
                canon_prefix = prefix
                canon_sep = "_" if with_underscore >= without_underscore else ""

            # Rename outliers
            for name in names:
                match = re.match(r"^(.+?)_?(\d+)$", name)
                if match:
                    digit = match.group(2)
                    canonical = f"{canon_prefix}{canon_sep}{digit}"
                    if canonical != name:
                        renames[name] = canonical
                        result.changes.append(f"Renamed {name} -> {canonical}")

        if not renames:
            return code, result

        # Apply renames to the code (method defs + all references)
        for old_name, new_name in renames.items():
            code = re.sub(
                r"\b" + re.escape(old_name) + r"\b",
                new_name,
                code,
            )

        result.applied = True
        return code, result

    # =========================================================================
    # Fix 7: Field Name Registry
    # =========================================================================

    def _fix_field_names(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Rename fields/constants to match the symbol registry.

        For any field fragment where the assembled name doesn't match
        the registry's python_name, rename it.
        """
        result = FixResult(fix_name="field_name_registry", applied=False)

        renames: dict[str, str] = {}

        for frag_id, entry in self.index.fragments.items():
            if entry.class_name != class_name:
                continue
            if entry.symbol_type not in ("constant", "field", "global_var"):
                continue
            if not entry.python_name or not entry.member_name:
                continue
            # Skip entries where python_name is a fragment ID (no registry mapping)
            if "::" in entry.python_name:
                continue

            # The registry's expected name
            expected = entry.python_name
            # Also consider snake_case of the member name
            snake = self._to_snake_case(entry.member_name)

            # Search for common wrong patterns in the code
            # e.g., UPPER_CASE when it should be lower_case, or vice versa
            java_name = entry.member_name
            upper_name = java_name.upper()

            # Check if the expected name is already in the code
            if re.search(r"\b" + re.escape(expected) + r"\b", code):
                continue

            # Try to find what the LLM actually named it
            candidates = [upper_name, java_name, snake]
            for candidate in candidates:
                if candidate == expected:
                    continue
                if re.search(r"\b" + re.escape(candidate) + r"\b", code):
                    renames[candidate] = expected
                    result.changes.append(
                        f"Renamed field {candidate} -> {expected}"
                    )
                    break

        if not renames:
            return code, result

        for old_name, new_name in renames.items():
            code = re.sub(
                r"\b" + re.escape(old_name) + r"\b",
                new_name,
                code,
            )

        result.applied = True
        return code, result

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert camelCase or PascalCase to snake_case."""
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        return s.lower()

    # =========================================================================
    # Fix 8: None Sentinel
    # =========================================================================

    def _fix_none_sentinel(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Replace `next(x, None) is None` with a sentinel object.

        The pattern `next(x, None) is None` is a common Java→Python
        mispattern where `None` is used as an iterator exhaustion sentinel,
        but `None` could be a valid element.
        """
        result = FixResult(fix_name="none_sentinel", applied=False)

        # Detect the pattern: next(..., None) used with `is None` or `== None`
        # This can appear on the same line or across lines:
        #   val = next(it, None); if val is None: ...
        #   val = next(it, None)
        #   if val is None:
        same_line_pattern = re.compile(
            r"next\s*\(\s*\w+\s*,\s*None\s*\)\s*(?:is|==)\s*None"
        )
        # Multi-line: next(x, None) assigned to a var, then var is None check
        next_none_pattern = re.compile(
            r"(\w+)\s*=\s*next\s*\(\s*\w+\s*,\s*None\s*\)"
        )

        has_same_line = same_line_pattern.search(code)
        has_multi_line = False
        if not has_same_line:
            # Check for multi-line pattern
            match = next_none_pattern.search(code)
            if match:
                var_name = match.group(1)
                # Check if `var is None` or `var == None` appears later
                check_pattern = re.compile(
                    r"\b" + re.escape(var_name) + r"\s+is\s+None\b"
                    r"|\b" + re.escape(var_name) + r"\s*==\s*None\b"
                )
                if check_pattern.search(code):
                    has_multi_line = True

        if not has_same_line and not has_multi_line:
            return code, result

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        functions_to_fix: list[tuple[int, int, str]] = []  # (start, end, indent)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_start = node.lineno - 1
            func_end = (node.end_lineno or node.lineno) - 1

            # Check if this function contains the pattern (same-line or multi-line)
            func_code = "\n".join(lines[func_start:func_end + 1])
            has_pattern = same_line_pattern.search(func_code)
            if not has_pattern:
                match = next_none_pattern.search(func_code)
                if match:
                    var_name = match.group(1)
                    check_pattern = re.compile(
                        r"\b" + re.escape(var_name) + r"\s+is\s+None\b"
                        r"|\b" + re.escape(var_name) + r"\s*==\s*None\b"
                    )
                    has_pattern = check_pattern.search(func_code)

            if has_pattern:
                body_line = lines[node.body[0].lineno - 1] if node.body else ""
                indent = body_line[: len(body_line) - len(body_line.lstrip())]
                functions_to_fix.append((func_start, func_end, indent))

        if not functions_to_fix:
            return code, result

        # Process from bottom to top
        functions_to_fix.sort(key=lambda x: x[0], reverse=True)

        for func_start, func_end, indent in functions_to_fix:
            # First, find variables assigned from next(x, None)
            assigned_vars: set[str] = set()
            for i in range(func_start, min(func_end + 1, len(lines))):
                m = re.match(
                    r".*?(\w+)\s*=\s*next\s*\(\s*\w+\s*,\s*None\s*\)",
                    lines[i],
                )
                if m:
                    assigned_vars.add(m.group(1))

            # Replace next(x, None) with next(x, _SENTINEL) in this function
            for i in range(func_start, min(func_end + 1, len(lines))):
                lines[i] = re.sub(
                    r"next\s*\(\s*(\w+)\s*,\s*None\s*\)",
                    r"next(\1, _SENTINEL)",
                    lines[i],
                )
                # Same-line: ... _SENTINEL) is None -> ... _SENTINEL) is _SENTINEL
                lines[i] = re.sub(
                    r"_SENTINEL\s*\)\s*is\s*None",
                    "_SENTINEL) is _SENTINEL",
                    lines[i],
                )
                lines[i] = re.sub(
                    r"_SENTINEL\s*\)\s*==\s*None",
                    "_SENTINEL) is _SENTINEL",
                    lines[i],
                )
                # Multi-line: var is None -> var is _SENTINEL
                for var in assigned_vars:
                    lines[i] = re.sub(
                        r"\b" + re.escape(var) + r"\s+is\s+None\b",
                        f"{var} is _SENTINEL",
                        lines[i],
                    )
                    lines[i] = re.sub(
                        r"\b" + re.escape(var) + r"\s*==\s*None\b",
                        f"{var} is _SENTINEL",
                        lines[i],
                    )

            # Insert _SENTINEL = object() at the top of the function body
            # Find the first line of the function body
            body_insert_idx = func_start + 1  # After the def line
            # Skip decorators and docstrings
            for i in range(func_start + 1, min(func_end + 1, len(lines))):
                stripped = lines[i].strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Skip docstring — find the end
                    if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        body_insert_idx = i + 1
                    else:
                        for j in range(i + 1, min(func_end + 1, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                body_insert_idx = j + 1
                                break
                    break
                elif stripped and not stripped.startswith("#"):
                    body_insert_idx = i
                    break

            # Check if _SENTINEL is already defined
            sentinel_already = any(
                "_SENTINEL" in lines[i] and "object()" in lines[i]
                for i in range(func_start, min(func_end + 1, len(lines)))
            )
            if not sentinel_already:
                lines.insert(body_insert_idx, f"{indent}_SENTINEL = object()")

            result.changes.append(
                f"Replaced None sentinel with _SENTINEL object"
            )

        result.applied = True
        return "\n".join(lines), result

    # =========================================================================
    # Fix 9: Fragment ID References
    # =========================================================================

    def _fix_fragment_id_references(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Replace leaked fragment ID syntax with plain Python names.

        Fragment IDs like ClassName::ClassName.field appear in LLM-generated
        code (gap fills, skeletons) when the LLM copies the ID syntax used
        in its context prompt.

        Patterns handled:
        - self.ClassName::ClassName.field  -> self.field
        - ClassName::ClassName.field       -> field
        """
        result = FixResult(fix_name="fragment_id_references", applied=False)

        if "::" not in code:
            return code, result

        original = code

        # Process line-by-line to preserve # MOCKED: / # MISSING: markers
        # (used by gap filler's apply_fills to locate injection points)
        lines = code.splitlines()
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("# MOCKED:") or stripped.startswith("# MISSING:"):
                fixed_lines.append(line)
                continue
            # Pattern 1: self.ClassName::ClassName.field -> self.field
            line = re.sub(r"self\.\w+::\w+\.(\w+)", r"self.\1", line)
            # Pattern 2: ClassName::ClassName.field -> field (standalone)
            line = re.sub(r"\w+::\w+\.(\w+)", r"\1", line)
            fixed_lines.append(line)

        code = "\n".join(fixed_lines)

        if code != original:
            result.applied = True
            count = original.count("::") - code.count("::")
            result.changes.append(
                f"Replaced {count} fragment ID reference(s) with plain names"
            )

        return code, result

    # =========================================================================
    # Fix 10: Broken Double Defs
    # =========================================================================

    def _fix_broken_double_defs(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Fix adjacent duplicate method definitions.

        When the LLM generates two consecutive def lines for the same method
        (e.g. a bare `def foo(self):` followed by `def foo(self) -> int:`),
        keep only the second (more complete) definition.
        """
        result = FixResult(fix_name="broken_double_defs", applied=False)

        original = code

        # Pattern 1: indented bare def followed by same-name def with return type
        code = re.sub(
            r"(\s+)def (\w+)\(self\):\n\s*def \2\(self\)( -> [^:]+):",
            r"\1def \2(self)\3:",
            code,
        )

        # Pattern 2: indented bare def followed by same-name def with params
        code = re.sub(
            r"(\s+)def (\w+)\(self\):\n(\s*def \2\(self,\s*[^)]*\)[^:]*:)",
            r"\1\3",
            code,
        )

        if code != original:
            result.applied = True
            result.changes.append("Removed duplicate method definition(s)")

        return code, result

    # =========================================================================
    # Fix 11: Orphaned Annotations
    # =========================================================================

    def _fix_orphaned_annotations(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Remove orphaned ``X: Type = None`` lines when X already has a concrete assignment."""
        result = FixResult(fix_name="orphaned_annotations", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        remove_lines: set[int] = set()  # 1-indexed line numbers to remove

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Pass 1: collect names with concrete (non-None) assignments
            concrete_names: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            if not (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                                concrete_names.add(target.id)
                # Also count __init__ self.X = <non-None> as concrete
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                    for s in ast.walk(stmt):
                        if isinstance(s, ast.Assign):
                            for t in s.targets:
                                if (
                                    isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"
                                ):
                                    concrete_names.add(t.attr)

            # Pass 2: find annotated assignments with None value for already-concrete names
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    is_none = stmt.value is None or (
                        isinstance(stmt.value, ast.Constant) and stmt.value.value is None
                    )
                    if name in concrete_names and is_none:
                        for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
                            remove_lines.add(ln)

                # Also catch plain NAME = None when NAME = <value> exists
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    t = stmt.targets[0]
                    if isinstance(t, ast.Name) and t.id in concrete_names:
                        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                            for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
                                remove_lines.add(ln)

        if remove_lines:
            new_lines = [l for i, l in enumerate(lines, 1) if i not in remove_lines]
            result.applied = True
            result.changes.append(f"Removed {len(remove_lines)} orphaned annotation line(s)")
            return "\n".join(new_lines), result

        return code, result

    # =========================================================================
    # Fix 12: Duplicate Field Assignments
    # =========================================================================

    def _fix_duplicate_assignments(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Remove duplicate field assignments within each class body.

        Keeps the first occurrence of ``NAME = value``; removes later
        identical assignments (same target name AND same value repr).
        """
        result = FixResult(fix_name="duplicate_assignments", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        lines = code.splitlines()
        remove_lines: set[int] = set()  # 1-indexed

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Track (name, ast.dump(value)) → first occurrence
            seen: dict[tuple[str, str], int] = {}

            for stmt in node.body:
                names_and_values: list[tuple[str, str]] = []

                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    t = stmt.targets[0]
                    if isinstance(t, ast.Name):
                        names_and_values.append((t.id, ast.dump(stmt.value)))

                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    val_dump = ast.dump(stmt.value) if stmt.value else "<no_value>"
                    names_and_values.append((stmt.target.id, val_dump))

                for key in names_and_values:
                    if key in seen:
                        # Duplicate — mark for removal
                        for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
                            remove_lines.add(ln)
                        result.changes.append(
                            f"Removed duplicate assignment of {key[0]} at line {stmt.lineno}"
                        )
                    else:
                        seen[key] = stmt.lineno

        if remove_lines:
            new_lines = [l for i, l in enumerate(lines, 1) if i not in remove_lines]
            result.applied = True
            return "\n".join(new_lines), result

        return code, result

    # -----------------------------------------------------------------
    # Fix: nested_enum_extraction
    # -----------------------------------------------------------------

    def _fix_nested_enum(
        self, class_name: str, code: str
    ) -> tuple[str, FixResult]:
        """Extract flattened enum members into a proper nested Enum class.

        Java allows ``static enum Player { O, X; }`` nested inside a class.
        The LLM sometimes flattens the enum members into the outer class body
        as ``O = auto()`` / ``X = auto()`` without creating the nested Enum.

        This fix:
        1. Detects ``auto()`` assignments inside non-Enum classes
        2. Looks up the Java source for ``static enum Name { ... }``
        3. Wraps the members in a ``class Name(Enum):`` nested block
        4. Moves any related static methods (defined inside the Java enum) into it
        5. Adds ``from enum import Enum, auto`` import
        """
        result = FixResult(fix_name="nested_enum_extraction", applied=False)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, result

        # Find the outer class node
        outer_class: ast.ClassDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                outer_class = node
                break

        if outer_class is None:
            return code, result

        # Check if the outer class is already an Enum
        for base in outer_class.bases:
            bname = ""
            if isinstance(base, ast.Name):
                bname = base.id
            elif isinstance(base, ast.Attribute):
                bname = base.attr
            if bname in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"):
                return code, result  # Already an Enum, nothing to do

        # Collect auto() assignments in the class body
        auto_stmts: list[tuple[str, ast.stmt]] = []  # (name, stmt)
        for stmt in outer_class.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Name) and func.id == "auto":
                        auto_stmts.append((target.id, stmt))

        if not auto_stmts:
            return code, result

        # Find the nested enum name from Java source
        enum_name = self._find_nested_enum_name(class_name, [n for n, _ in auto_stmts])
        if not enum_name:
            # Fallback: can't determine name, still fix the import
            enum_name = class_name + "Enum"

        # Find static methods that belong to the nested enum (from Java source)
        enum_methods = self._find_nested_enum_methods(class_name, enum_name)

        lines = code.splitlines()

        # Top-level enum: no indent, members at 4-space indent
        enum_indent = ""
        member_indent = "    "

        # Collect line numbers to remove (auto stmts + enum methods)
        remove_lines: set[int] = set()  # 0-indexed
        for _, stmt in auto_stmts:
            for ln in range(stmt.lineno - 1, (stmt.end_lineno or stmt.lineno)):
                remove_lines.add(ln)

        # Find and collect enum method code blocks (re-indented for top-level)
        enum_method_code_blocks: list[str] = []
        for mname in enum_methods:
            for stmt in outer_class.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == mname:
                    start = stmt.lineno - 1
                    # Include decorators
                    if stmt.decorator_list:
                        start = stmt.decorator_list[0].lineno - 1
                    end = stmt.end_lineno or (start + 1)
                    # Determine the original indentation of this method's def line
                    orig_def_line = lines[stmt.lineno - 1]
                    orig_indent = orig_def_line[: len(orig_def_line) - len(orig_def_line.lstrip())]
                    for ln in range(start, end):
                        remove_lines.add(ln)
                        # Re-indent: strip original indent, add member_indent
                        stripped = lines[ln]
                        if stripped.startswith(orig_indent):
                            stripped = stripped[len(orig_indent):]
                        else:
                            stripped = stripped.lstrip()
                        enum_method_code_blocks.append(member_indent + stripped)
                    enum_method_code_blocks.append("")  # blank line after method

        # Build the top-level enum class
        enum_class_lines = [
            "",
            f"{enum_indent}class {enum_name}(Enum):",
        ]
        for name, _ in auto_stmts:
            enum_class_lines.append(f"{member_indent}{name} = auto()")
        enum_class_lines.append("")

        # Add __str__ so enum members print as just name (Java behavior)
        enum_class_lines.append(f"{member_indent}def __str__(self):")
        enum_class_lines.append(f"{member_indent}    return self.name")
        enum_class_lines.append("")

        # Add relocated methods
        if enum_method_code_blocks:
            enum_class_lines.extend(enum_method_code_blocks)
        enum_class_lines.append("")

        # Remove auto stmts and enum methods from the outer class,
        # then insert the enum class BEFORE the outer class definition.
        outer_class_line = outer_class.lineno - 1  # 0-indexed
        # Back up past decorators
        if outer_class.decorator_list:
            outer_class_line = outer_class.decorator_list[0].lineno - 1
        # Also include any comment/blank lines immediately before
        while outer_class_line > 0 and lines[outer_class_line - 1].strip().startswith("#"):
            outer_class_line -= 1

        new_lines = []
        for i, line in enumerate(lines):
            if i == outer_class_line:
                # Insert enum class before the outer class
                new_lines.extend(enum_class_lines)
            if i in remove_lines:
                continue
            new_lines.append(line)

        new_code = "\n".join(new_lines)

        # Ensure enum imports are present
        if "from enum import" not in new_code:
            # Add after __future__ import or at top
            import_line = "from enum import Enum, auto"
            code_lines = new_code.splitlines()
            insert_at = 0
            for i, ln in enumerate(code_lines):
                stripped = ln.strip()
                if stripped.startswith("from __future__"):
                    insert_at = i + 1
                elif stripped.startswith(("import ", "from ")) and i < 10:
                    insert_at = i + 1
            code_lines.insert(insert_at, import_line)
            new_code = "\n".join(code_lines)
        elif "auto" not in code.split("from enum import")[1].split("\n")[0]:
            # enum import exists but auto is missing — add it
            new_code = re.sub(
                r"from enum import (.+)",
                lambda m: f"from enum import {m.group(1).strip()}, auto"
                if "auto" not in m.group(1) else m.group(0),
                new_code,
                count=1,
            )

        result.applied = True
        result.changes.append(
            f"Extracted {len(auto_stmts)} enum members into top-level "
            f"class {enum_name}(Enum)"
        )
        return new_code, result

    def _find_nested_enum_name(
        self, class_name: str, member_names: list[str]
    ) -> str | None:
        """Find the name of a nested enum from Java source."""
        summary = self.index.classes.get(class_name)
        if not summary:
            return None

        # Read the Java source for the class-level fragment
        for member in summary.members:
            if member.member_name is None and member.java_source:
                # Class-level fragment — search for nested enum declaration
                m = re.search(
                    r"(?:public\s+)?(?:static\s+)?enum\s+(\w+)\s*\{",
                    member.java_source,
                )
                if m:
                    return m.group(1)

        # Fallback: search all fragments for this class
        for fid, entry in self.index.fragments.items():
            if entry.class_name != class_name:
                continue
            if entry.java_source:
                m = re.search(
                    r"(?:public\s+)?(?:static\s+)?enum\s+(\w+)\s*\{",
                    entry.java_source,
                )
                if m:
                    return m.group(1)

        return None

    def _find_nested_enum_methods(
        self, class_name: str, enum_name: str
    ) -> list[str]:
        """Find methods defined inside a nested Java enum.

        Parses the class-level fragment's full Java source to extract the
        enum block and find method names declared within it.
        """
        methods: list[str] = []

        summary = self.index.classes.get(class_name)
        if not summary:
            return methods

        # Find the class-level fragment (contains full Java class source)
        class_source = ""
        for member in summary.members:
            if member.member_name is None and member.java_source:
                class_source = member.java_source
                break

        if not class_source:
            return methods

        # Extract the enum block using brace matching
        enum_pattern = re.search(
            rf"enum\s+{re.escape(enum_name)}\s*\{{", class_source
        )
        if not enum_pattern:
            return methods

        # Find the matching closing brace
        start = enum_pattern.end()
        depth = 1
        pos = start
        while pos < len(class_source) and depth > 0:
            if class_source[pos] == "{":
                depth += 1
            elif class_source[pos] == "}":
                depth -= 1
            pos += 1

        enum_body = class_source[start : pos - 1]

        # Find method declarations inside the enum body
        for m in re.finditer(
            r"(?:public\s+)?(?:static\s+)?\w+\s+(\w+)\s*\(", enum_body
        ):
            java_name = m.group(1)
            # Convert camelCase to snake_case
            py_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", java_name)
            py_name = py_name.lower()
            methods.append(py_name)

        return methods
