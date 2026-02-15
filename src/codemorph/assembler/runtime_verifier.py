"""
Runtime Verifier for Phase 4 QA.

Generates and executes mechanical smoke tests against the assembled Python
output.  No LLM — all tests are derived from AST inspection and fragment
metadata.  Returns issues in the same JSON format used by QA sub-agent reports.
"""

import ast
import logging
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from codemorph.assembler.index_builder import ProjectIndex

logger = logging.getLogger(__name__)

PYTHON_BIN = sys.executable or "python3"

# Enum base class names we recognize
_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})

# Python builtins that are valid bare names
_BUILTINS = frozenset(dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__))


class RuntimeVerifier:
    """Run mechanical smoke tests against assembled output."""

    def __init__(
        self,
        index: ProjectIndex,
        output_dir: Path,
        package_name: str,
    ):
        self.index = index
        self.output_dir = Path(output_dir).resolve()
        self.package_name = package_name

    def run(self) -> list[dict[str, Any]]:
        """Execute all smoke tests.  Returns issues in QA report format."""
        issues: list[dict[str, Any]] = []
        issues.extend(self._test_per_module_import())
        issues.extend(self._test_enum_str())
        issues.extend(self._test_instantiation())
        issues.extend(self._test_method_callable())
        issues.extend(self._test_bare_class_constants())
        issues.extend(self._test_field_method_collision())
        issues.extend(self._test_missing_self_param())
        issues.extend(self._test_bare_enum_references())
        issues.extend(self._test_missing_class_imports())
        return issues

    # ------------------------------------------------------------------
    # 1. Per-module import test
    # ------------------------------------------------------------------

    def _test_per_module_import(self) -> list[dict[str, Any]]:
        """Test that each individual .py module can be imported without error.

        This goes beyond the package-level ``import tictactoe`` — it catches
        missing imports, undefined names at module scope, and circular import
        issues that only manifest when a specific module is loaded first.
        """
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue

            rel = py_file.relative_to(self.output_dir)
            module_path = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")

            script = textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, {str(self.output_dir)!r})
                try:
                    import {module_path}
                    print("OK")
                except Exception as e:
                    print(f"FAIL: {{type(e).__name__}}: {{e}}")
                    sys.exit(1)
            """)

            ok, output = self._run_script(script)
            if not ok:
                issues.append({
                    "type": "import_error",
                    "file": str(rel),
                    "class_name": "",
                    "method": "",
                    "line": 0,
                    "current": f"import {module_path} fails",
                    "suggested": f"Fix imports in {rel}",
                    "description": (
                        f"Importing module {module_path} failed: "
                        f"{output.strip()[:300]}"
                    ),
                })

        return issues

    # ------------------------------------------------------------------
    # 2. Enum __str__ verification
    # ------------------------------------------------------------------

    def _test_enum_str(self) -> list[dict[str, Any]]:
        """Check that Enum subclasses return just the member name from str().

        Java enums' toString() returns the name only.  Python's default
        Enum.__str__ returns 'ClassName.MEMBER'.  If __str__ is missing the
        output will be wrong at runtime.
        """
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                if not self._is_enum_class(node):
                    continue

                has_str = any(
                    isinstance(stmt, ast.FunctionDef) and stmt.name == "__str__"
                    for stmt in node.body
                )
                if has_str:
                    continue

                members = self._get_enum_members(node)
                if not members:
                    continue

                member = members[0]
                rel = py_file.relative_to(self.output_dir)
                module_path = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")

                script = textwrap.dedent(f"""\
                    import sys
                    sys.path.insert(0, {str(self.output_dir)!r})
                    from {module_path} import {node.name}
                    m = {node.name}[{member!r}]
                    result = str(m)
                    if result != {member!r}:
                        print(f"FAIL: str({{m!r}}) = {{result!r}}, expected {member!r}")
                        sys.exit(1)
                    else:
                        print("OK")
                """)

                ok, output = self._run_script(script)
                if not ok:
                    issues.append({
                        "type": "enum_str_error",
                        "file": str(rel),
                        "class_name": node.name,
                        "method": "__str__",
                        "line": node.lineno,
                        "current": f"str({node.name}.{member}) returns wrong value",
                        "suggested": f"Add __str__ returning self.name to {node.name}",
                        "description": (
                            f"Enum {node.name} lacks __str__; "
                            f"str() returns 'ClassName.MEMBER' instead of just the name. "
                            f"Output: {output.strip()[:200]}"
                        ),
                    })

        return issues

    # ------------------------------------------------------------------
    # 3. Instantiation test
    # ------------------------------------------------------------------

    def _test_instantiation(self) -> list[dict[str, Any]]:
        """Try to instantiate non-abstract classes with no required args."""
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if self._is_abstract(node):
                    continue
                if self._is_enum_class(node):
                    continue

                init_method = None
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                        init_method = stmt
                        break

                if init_method:
                    args = init_method.args
                    n_pos = len(args.args) - 1
                    n_defaults = len(args.defaults)
                    required = n_pos - n_defaults
                    if required > 0:
                        continue

                rel = py_file.relative_to(self.output_dir)
                module_path = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")

                script = textwrap.dedent(f"""\
                    import sys
                    sys.path.insert(0, {str(self.output_dir)!r})
                    from {module_path} import {node.name}
                    try:
                        obj = {node.name}()
                        print(f"OK: {{type(obj).__name__}}")
                    except Exception as e:
                        print(f"FAIL: {{type(e).__name__}}: {{e}}")
                        sys.exit(1)
                """)

                ok, output = self._run_script(script)
                if not ok:
                    issues.append({
                        "type": "runtime_error",
                        "file": str(rel),
                        "class_name": node.name,
                        "method": "__init__",
                        "line": node.lineno,
                        "current": f"{node.name}() raises at runtime",
                        "suggested": "Fix constructor or default arguments",
                        "description": (
                            f"Instantiating {node.name}() failed: "
                            f"{output.strip()[:200]}"
                        ),
                    })

        return issues

    # ------------------------------------------------------------------
    # 4. Method signature verification
    # ------------------------------------------------------------------

    _JAVA_TO_PYTHON_METHODS: dict[str, str] = {
        "equals": "__eq__",
        "hashCode": "__hash__",
        "toString": "__str__",
        "compareTo": "__lt__",
        "clone": "__copy__",
        "iterator": "__iter__",
        "hasNext": "__next__",
        "size": "__len__",
        "close": "__del__",
        "main": "main",
    }

    def _test_method_callable(self) -> list[dict[str, Any]]:
        """Verify methods exist with expected parameter counts from Java fragments."""
        issues: list[dict[str, Any]] = []

        for class_name, summary in self.index.classes.items():
            py_file = self._find_class_file(class_name)
            if not py_file:
                continue

            rel = py_file.relative_to(self.output_dir)
            actual_methods = self._get_class_methods(py_file, class_name)
            if actual_methods is None:
                continue

            for member in summary.members:
                if member.member_name is None:
                    continue
                java_method = member.member_name
                if not member.java_source or "(" not in member.java_source:
                    continue

                python_method = self._resolve_python_method_name(
                    member.fragment_id, java_method
                )
                candidates = [python_method]
                snake = self._camel_to_snake(java_method)
                if snake not in candidates:
                    candidates.append(snake)
                dunder = self._JAVA_TO_PYTHON_METHODS.get(java_method)
                if dunder and dunder not in candidates:
                    candidates.append(dunder)
                if java_method.startswith("get") and len(java_method) > 3:
                    attr_name = self._camel_to_snake(java_method[3:])
                    if attr_name not in candidates:
                        candidates.append(attr_name)
                elif java_method.startswith("is") and len(java_method) > 2:
                    attr_name = self._camel_to_snake(java_method[2:])
                    if attr_name not in candidates:
                        candidates.append(attr_name)

                found = any(c in actual_methods for c in candidates)

                if not found and self._count_java_params(member.java_source) == 0:
                    if java_method.startswith(("get", "is")):
                        continue

                if not found:
                    found = self._method_exists_anywhere(candidates, class_name)

                if found:
                    continue

                issues.append({
                    "type": "signature_mismatch",
                    "file": str(rel),
                    "class_name": class_name,
                    "method": python_method,
                    "line": 0,
                    "current": (
                        f"Method {java_method} (tried: {candidates}) "
                        f"not found on {class_name}"
                    ),
                    "suggested": f"Add {python_method} method to {class_name}",
                    "description": (
                        f"Java method {java_method} has no matching Python method. "
                        f"Tried: {candidates}. "
                        f"Available: {sorted(actual_methods)}"
                    ),
                })

        return issues

    # ------------------------------------------------------------------
    # 5. Bare class constant references (missing self. or ClassName.)
    # ------------------------------------------------------------------

    def _test_bare_class_constants(self) -> list[dict[str, Any]]:
        """Detect bare references to class-level constants inside methods.

        Java allows ``ROWS`` to refer to ``this.ROWS`` or ``ClassName.ROWS``
        implicitly.  Python requires ``self.ROWS`` or ``ClassName.ROWS``.
        LLM translators often miss this, producing NameError at runtime.
        """
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel = py_file.relative_to(self.output_dir)

            # Collect top-level imports/names that are legitimately in module scope
            module_names = self._collect_module_scope_names(tree)

            # Collect all enum member names in this file (handled by _test_bare_enum_references)
            file_enum_members: set[str] = set()
            for enode in ast.walk(tree):
                if isinstance(enode, ast.ClassDef) and self._is_enum_class(enode):
                    for m in self._get_enum_members(enode):
                        file_enum_members.add(m)

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                # Skip enum classes — their members are checked by _test_bare_enum_references
                if self._is_enum_class(node):
                    continue

                # Collect class-level constants and fields
                class_constants: set[str] = set()
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                class_constants.add(target.id)
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        class_constants.add(stmt.target.id)

                if not class_constants:
                    continue

                # Remove any names that are also enum members (to avoid duplicates)
                class_constants -= file_enum_members

                # Walk methods looking for bare Name references to class constants
                for stmt in node.body:
                    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue

                    # Determine if this is a static/classmethod
                    is_static_method = any(
                        (isinstance(d, ast.Name) and d.id == "staticmethod")
                        or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
                        for d in stmt.decorator_list
                    )

                    # Collect local names in this method (params + local assigns)
                    local_names = {a.arg for a in stmt.args.args}
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if isinstance(t, ast.Name):
                                    local_names.add(t.id)
                        elif isinstance(sub, (ast.For, ast.AsyncFor)):
                            if isinstance(sub.target, ast.Name):
                                local_names.add(sub.target.id)
                        elif isinstance(sub, ast.comprehension):
                            if isinstance(sub.target, ast.Name):
                                local_names.add(sub.target.id)

                    for sub in ast.walk(stmt):
                        if not isinstance(sub, ast.Name):
                            continue
                        name = sub.id
                        if name not in class_constants:
                            continue
                        if name in local_names:
                            continue
                        if name in module_names:
                            continue
                        if name in _BUILTINS:
                            continue
                        # Is this a bare reference (not self.X or Class.X)?
                        # Check parent — if parent is ast.Attribute and this is the value,
                        # it's fine (e.g., self.ROWS would have Name(id='self') as .value)
                        # But we need to check if it's used as a standalone Name load
                        if isinstance(sub.ctx, ast.Load):
                            prefix = f"{node.name}.{name}" if is_static_method else f"self.{name}"
                            issues.append({
                                "type": "incorrect_reference",
                                "file": str(rel),
                                "class_name": node.name,
                                "method": stmt.name,
                                "line": sub.lineno,
                                "current": name,
                                "suggested": prefix,
                                "description": (
                                    f"Bare reference to class constant '{name}' in "
                                    f"{node.name}.{stmt.name}() line {sub.lineno}. "
                                    f"Python requires '{prefix}'."
                                ),
                            })

        return issues

    # ------------------------------------------------------------------
    # 6. Field/method name collision
    # ------------------------------------------------------------------

    def _test_field_method_collision(self) -> list[dict[str, Any]]:
        """Detect __init__ field assignments that shadow method definitions.

        Java has separate namespaces for fields and methods (you can have
        ``boolean hasWin`` and ``boolean hasWin(Player p)`` in the same class).
        Python doesn't — if ``__init__`` does ``self.has_win = None``, it
        overwrites the ``def has_win(self, player)`` method at runtime.
        """
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel = py_file.relative_to(self.output_dir)

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                # Collect method names
                method_names: set[str] = set()
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_names.add(stmt.name)

                # Find __init__ and check self.X = ... assignments
                for stmt in node.body:
                    if not isinstance(stmt, ast.FunctionDef) or stmt.name != "__init__":
                        continue

                    for sub in ast.walk(stmt):
                        if not isinstance(sub, ast.Assign):
                            continue
                        for target in sub.targets:
                            if not isinstance(target, ast.Attribute):
                                continue
                            if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
                                continue
                            attr = target.attr
                            if attr in method_names and attr != "__init__":
                                issues.append({
                                    "type": "field_method_collision",
                                    "file": str(rel),
                                    "class_name": node.name,
                                    "method": attr,
                                    "line": sub.lineno,
                                    "current": f"self.{attr} = ...",
                                    "suggested": (
                                        f"Remove 'self.{attr} = ...' from __init__ "
                                        f"(it shadows def {attr}(self, ...)), or "
                                        f"rename the field to '_{attr}'"
                                    ),
                                    "description": (
                                        f"__init__ assigns self.{attr} which shadows the "
                                        f"method def {attr}() in class {node.name}. "
                                        f"At runtime, calling obj.{attr}(...) will fail "
                                        f"with 'NoneType is not callable' or similar."
                                    ),
                                })

        return issues

    # ------------------------------------------------------------------
    # 7. Missing self parameter on instance methods
    # ------------------------------------------------------------------

    def _test_missing_self_param(self) -> list[dict[str, Any]]:
        """Detect instance methods that are missing the ``self`` parameter.

        Java instance methods don't declare ``this`` as a parameter.  LLM
        translators sometimes forget to add ``self`` (especially for methods
        that were translated from static context or inner classes).
        """
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel = py_file.relative_to(self.output_dir)

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                for stmt in node.body:
                    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue

                    # Skip if it has @staticmethod or @classmethod decorator
                    is_static = False
                    is_classmethod = False
                    for decorator in stmt.decorator_list:
                        dname = ""
                        if isinstance(decorator, ast.Name):
                            dname = decorator.id
                        elif isinstance(decorator, ast.Attribute):
                            dname = decorator.attr
                        if dname == "staticmethod":
                            is_static = True
                        elif dname == "classmethod":
                            is_classmethod = True
                    if is_static or is_classmethod:
                        continue

                    # Check if first param is self
                    args = stmt.args.args
                    if not args:
                        issues.append({
                            "type": "missing_self",
                            "file": str(rel),
                            "class_name": node.name,
                            "method": stmt.name,
                            "line": stmt.lineno,
                            "current": f"def {stmt.name}(...) has no parameters",
                            "suggested": f"Add 'self' as first parameter",
                            "description": (
                                f"Instance method {node.name}.{stmt.name}() has no "
                                f"parameters — missing 'self'. Will crash with "
                                f"TypeError when called on an instance."
                            ),
                        })
                    elif args[0].arg not in ("self", "cls"):
                        # Check if method body references self — if it does,
                        # the first param is wrong. If it doesn't and uses no
                        # instance state, it might legitimately be static.
                        uses_self = False
                        for sub in ast.walk(stmt):
                            if isinstance(sub, ast.Name) and sub.id == "self":
                                uses_self = True
                                break
                            if isinstance(sub, ast.Attribute):
                                if isinstance(sub.value, ast.Name) and sub.value.id == "self":
                                    uses_self = True
                                    break
                        if uses_self:
                            issues.append({
                                "type": "missing_self",
                                "file": str(rel),
                                "class_name": node.name,
                                "method": stmt.name,
                                "line": stmt.lineno,
                                "current": f"def {stmt.name}({args[0].arg}, ...)",
                                "suggested": f"First parameter should be 'self'",
                                "description": (
                                    f"Method {node.name}.{stmt.name}() uses 'self' in "
                                    f"its body but first param is '{args[0].arg}', "
                                    f"not 'self'. Missing self parameter."
                                ),
                            })

        return issues

    # ------------------------------------------------------------------
    # 8. Bare enum member references
    # ------------------------------------------------------------------

    def _test_bare_enum_references(self) -> list[dict[str, Any]]:
        """Detect bare enum member names used without their class prefix.

        Java allows ``return O`` inside an enum method, Python requires
        ``return Player.O``.  Similarly, ``player == X`` must be
        ``player == Player.X``.
        """
        issues: list[dict[str, Any]] = []

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel = py_file.relative_to(self.output_dir)
            module_names = self._collect_module_scope_names(tree)

            # First pass: collect all enum classes and their members in this file
            enum_members: dict[str, list[str]] = {}  # enum_name -> [member_names]
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and self._is_enum_class(node):
                    members = self._get_enum_members(node)
                    if members:
                        enum_members[node.name] = members

            if not enum_members:
                continue

            # Build reverse map: member_name -> enum_class_name
            member_to_enum: dict[str, str] = {}
            for enum_name, members in enum_members.items():
                for member in members:
                    member_to_enum[member] = enum_name

            # Second pass: walk all functions looking for bare enum member references
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                # Which class is this method in?
                parent_class = self._find_parent_class(tree, node)

                # Collect local names
                local_names = {a.arg for a in node.args.args}
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                local_names.add(t.id)

                for sub in ast.walk(node):
                    if not isinstance(sub, ast.Name):
                        continue
                    if not isinstance(sub.ctx, ast.Load):
                        continue
                    name = sub.id
                    if name not in member_to_enum:
                        continue
                    if name in local_names:
                        continue
                    if name in module_names:
                        continue
                    if name in _BUILTINS:
                        continue

                    enum_cls = member_to_enum[name]
                    issues.append({
                        "type": "incorrect_reference",
                        "file": str(rel),
                        "class_name": parent_class or "",
                        "method": node.name,
                        "line": sub.lineno,
                        "current": name,
                        "suggested": f"{enum_cls}.{name}",
                        "description": (
                            f"Bare enum member reference '{name}' in "
                            f"{node.name}() line {sub.lineno}. "
                            f"Should be '{enum_cls}.{name}'."
                        ),
                    })

        return issues

    # ------------------------------------------------------------------
    # 9. Missing class imports
    # ------------------------------------------------------------------

    def _test_missing_class_imports(self) -> list[dict[str, Any]]:
        """Detect references to project classes that are not imported.

        Scans each file for bare Name references to known project class names
        (from the index) that don't appear in the file's import statements.
        """
        issues: list[dict[str, Any]] = []

        # Build set of all known project class names
        project_classes: set[str] = set(self.index.classes.keys())
        if not project_classes:
            return issues

        # Build map: class_name -> module_name by scanning output files
        class_to_module: dict[str, str] = {}
        for scan_file in self.output_dir.rglob("*.py"):
            if scan_file.name.startswith("_"):
                continue
            try:
                scan_tree = ast.parse(scan_file.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            mod_name = scan_file.stem  # e.g., "game_board"
            for scan_node in scan_tree.body:
                if isinstance(scan_node, ast.ClassDef) and scan_node.name in project_classes:
                    class_to_module[scan_node.name] = mod_name

        for py_file in sorted(self.output_dir.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel = py_file.relative_to(self.output_dir)

            # Collect all imported names in this file
            imported_names = self._collect_module_scope_names(tree)

            # Collect all class names defined IN this file (they're available without import)
            local_classes: set[str] = set()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    local_classes.add(node.name)

            # Scan all Name references in the file for project class names
            missing: dict[str, int] = {}  # class_name -> first line
            for node in ast.walk(tree):
                if not isinstance(node, ast.Name):
                    continue
                if not isinstance(node.ctx, ast.Load):
                    continue
                name = node.id
                if name not in project_classes:
                    continue
                if name in imported_names:
                    continue
                if name in local_classes:
                    continue
                if name in _BUILTINS:
                    continue
                if name not in missing:
                    missing[name] = node.lineno

            for cls_name, line in missing.items():
                module = class_to_module.get(cls_name, "")
                if module:
                    suggested = f"from .{module} import {cls_name}"
                else:
                    suggested = f"import {cls_name}"
                issues.append({
                    "type": "missing_import",
                    "file": str(rel),
                    "class_name": "",
                    "method": "",
                    "line": line,
                    "current": f"{cls_name} used but not imported",
                    "suggested": suggested,
                    "description": (
                        f"Class '{cls_name}' is referenced in {rel} "
                        f"(line {line}) but not imported. "
                        f"Add: {suggested}"
                    ),
                })

        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_script(self, script: str, timeout: int = 10) -> tuple[bool, str]:
        """Run a Python script in a subprocess. Returns (success, output)."""
        try:
            result = subprocess.run(
                [PYTHON_BIN, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.output_dir),
                env={**__import__("os").environ, "PYTHONPATH": str(self.output_dir)},
            )
            combined = (result.stdout + result.stderr).strip()
            return result.returncode == 0, combined
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _base_name(node: ast.expr) -> str:
        """Extract the simple name from an AST base class node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @staticmethod
    def _is_abstract(node: ast.ClassDef) -> bool:
        """Check if a class is abstract (ABC base or has abstractmethod)."""
        for base in node.bases:
            name = RuntimeVerifier._base_name(base)
            if name in ("ABC", "ABCMeta"):
                return True
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                for decorator in stmt.decorator_list:
                    dname = ""
                    if isinstance(decorator, ast.Name):
                        dname = decorator.id
                    elif isinstance(decorator, ast.Attribute):
                        dname = decorator.attr
                    if dname == "abstractmethod":
                        return True
        return False

    @staticmethod
    def _is_enum_class(node: ast.ClassDef) -> bool:
        """Check if a class inherits from Enum or related bases."""
        return any(
            RuntimeVerifier._base_name(b) in _ENUM_BASES
            for b in node.bases
        )

    @staticmethod
    def _get_enum_members(node: ast.ClassDef) -> list[str]:
        """Collect enum member names (class-level Assign targets)."""
        members: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        members.append(target.id)
        return members

    @staticmethod
    def _collect_module_scope_names(tree: ast.Module) -> set[str]:
        """Collect names defined at module scope (imports, classes, functions, assigns)."""
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        return names

    @staticmethod
    def _find_parent_class(tree: ast.Module, target_func: ast.FunctionDef) -> str | None:
        """Find the class name that contains a given function node."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if stmt is target_func:
                        return node.name
        return None

    def _method_exists_anywhere(
        self, candidates: list[str], exclude_class: str
    ) -> bool:
        """Check if any candidate method name exists in any class across output."""
        for py_file in self.output_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name != exclude_class:
                    for stmt in node.body:
                        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if stmt.name in candidates:
                                return True
        return False

    def _resolve_python_method_name(self, fragment_id: str, java_method: str) -> str:
        """Resolve a Java method name to its expected Python equivalent."""
        for key, val in self.index.name_map.items():
            if key == fragment_id or ("::" in key and key.endswith(f".{java_method}")):
                return val
        if java_method in self._JAVA_TO_PYTHON_METHODS:
            return self._JAVA_TO_PYTHON_METHODS[java_method]
        return self._camel_to_snake(java_method)

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """Convert camelCase to snake_case."""
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        return s.lower()

    def _get_class_methods(self, py_file: Path, class_name: str) -> set[str] | None:
        """Get all method names defined in a class via AST."""
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                methods = set()
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(stmt.name)
                return methods
        return None

    def _find_class_file(self, class_name: str) -> Path | None:
        """Find the .py file that defines a given class."""
        for py_file in self.output_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        return py_file
            except (SyntaxError, OSError):
                continue
        return None

    @staticmethod
    def _count_java_params(java_source: str | None) -> int | None:
        """Count the number of parameters in a Java method signature."""
        if not java_source:
            return None
        m = re.search(r"\w+\s+\w+\s*\(([^)]*)\)", java_source)
        if not m:
            return None
        params_str = m.group(1).strip()
        if not params_str:
            return 0
        depth = 0
        count = 1
        for ch in params_str:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
            elif ch == "," and depth == 0:
                count += 1
        return count
