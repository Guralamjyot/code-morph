"""
Code Injector for Phase 4 Assembly.

Mechanical (non-LLM) step that replaces # INJECT: fragment_id markers
with actual translated code from the index. Mocked fragments are left
as # MOCKED markers for the gap filler.
"""

import ast
import logging
import re
import textwrap
from typing import Any

from codemorph.assembler.index_builder import FragmentEntry, ProjectIndex

logger = logging.getLogger(__name__)

# Stub namespace for exec-testing field assignments.  Provides common typing
# names so that annotations like ``ROWS: Final = 3`` don't fail the safety check.
_TYPING_STUBS: dict[str, Any] = {
    "__builtins__": {},
    "Final": None, "Optional": None, "List": None, "Dict": None,
    "Set": None, "Tuple": None, "Type": None, "Any": None,
    "ClassVar": None, "Sequence": None, "Mapping": None,
    "Callable": None, "Union": None, "Literal": None,
}


def _get_assign_name(stmt: ast.stmt) -> str | None:
    """Extract the target name from a simple assignment or annotated assignment."""
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
        if isinstance(t, ast.Name):
            return t.id
    return None


class CodeInjector:
    """Plugs translated code from the index into class skeletons."""

    def __init__(self, index: ProjectIndex):
        self.index = index

    def inject(self, skeleton: str) -> str:
        """Inject translated code into a class skeleton.

        Replaces # INJECT: fragment_id markers with actual method bodies.
        Leaves # MOCKED: fragment_id markers for the gap filler.

        Args:
            skeleton: Class skeleton with markers

        Returns:
            Assembled Python source with code injected
        """
        lines = skeleton.splitlines()
        result_lines: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for INJECT marker
            inject_match = re.match(r"^(\s*)#\s*INJECT:\s*(.+)$", line)
            if inject_match:
                indent = inject_match.group(1)
                fragment_id = inject_match.group(2).strip()

                entry = self.index.fragments.get(fragment_id)
                if entry and entry.target_code and not entry.is_mocked:
                    injected = self._extract_and_indent(
                        entry, indent, lines, i
                    )
                    result_lines.extend(injected)
                else:
                    # Fragment not found or mocked — generate a stub
                    marker = "MOCKED" if (entry and entry.is_mocked) else "MISSING"
                    is_class_level = entry and entry.member_name is None
                    if is_class_level:
                        # Try deterministic constructor generation first
                        ctor_code = self._generate_constructor_from_java(entry) if entry else None
                        if ctor_code:
                            result_lines.extend(self._reindent(ctor_code, indent))
                        else:
                            # Class-level: just pass
                            result_lines.append(f"{indent}# {marker}: {fragment_id}")
                            result_lines.append(f"{indent}pass")
                    elif entry and entry.symbol_type in ("method", "function"):
                        # Try deterministic method generation first
                        method_code = self._generate_method_from_java(entry)
                        if method_code:
                            result_lines.extend(self._reindent(method_code, indent))
                        else:
                            # Method: generate a stub def
                            method_name = self._to_snake_case(entry.member_name or "unknown")
                            sig = entry.signature
                            if not sig or not sig.strip().startswith("def "):
                                sig = f"def {method_name}(self)"
                            result_lines.append(f"{indent}{sig}:")
                            result_lines.append(f"{indent}    # {marker}: {fragment_id}")
                            result_lines.append(f"{indent}    raise NotImplementedError('{fragment_id}')")
                    elif entry and entry.symbol_type == "constant":
                        # Constant: placeholder assignment
                        name = entry.member_name or "UNKNOWN"
                        result_lines.append(f"{indent}# {marker}: {fragment_id}")
                        result_lines.append(f"{indent}{name} = None  # TODO: implement")
                    else:
                        result_lines.append(f"{indent}# {marker}: {fragment_id}")
                        result_lines.append(f"{indent}pass")

                i += 1
                continue

            # Check for bare MOCKED/MISSING markers (from LLM-drafted skeletons
            # or from fallback skeletons). Generate appropriate stubs.
            mocked_match = re.match(r"^(\s*)#\s*(?:MOCKED|MISSING):\s*(.+)$", line)
            if mocked_match:
                indent = mocked_match.group(1)
                fragment_id = mocked_match.group(2).strip()
                entry = self.index.fragments.get(fragment_id)

                # Check if next line already has a body
                next_idx = i + 1
                has_body = False
                while next_idx < len(lines):
                    next_line = lines[next_idx].strip()
                    if next_line == "":
                        next_idx += 1
                        continue
                    if next_line.startswith(("raise ", "pass", "def ", "class ")):
                        has_body = True
                    break

                if has_body:
                    result_lines.append(line)
                elif entry and entry.member_name is None:
                    # Try deterministic constructor generation first
                    ctor_code = self._generate_constructor_from_java(entry)
                    if ctor_code:
                        result_lines.extend(self._reindent(ctor_code, indent))
                    else:
                        # Class-level: pass (raise would execute at class def)
                        result_lines.append(line)
                        result_lines.append(f"{indent}pass")
                elif entry and entry.symbol_type in ("method", "function"):
                    # Try deterministic method generation first
                    method_code = self._generate_method_from_java(entry)
                    if method_code:
                        result_lines.extend(self._reindent(method_code, indent))
                    else:
                        # Method: generate stub def
                        method_name = self._to_snake_case(entry.member_name or "unknown")
                        sig = entry.signature
                        if not sig or not sig.strip().startswith("def "):
                            sig = f"def {method_name}(self)"
                        result_lines.append(f"{indent}{sig}:")
                        result_lines.append(f"{indent}    # MOCKED: {fragment_id}")
                        result_lines.append(f"{indent}    raise NotImplementedError('{fragment_id}')")
                elif entry and entry.symbol_type == "constant":
                    name = entry.member_name or "UNKNOWN"
                    result_lines.append(f"{indent}# MOCKED: {fragment_id}")
                    result_lines.append(f"{indent}{name} = None  # TODO")
                else:
                    result_lines.append(line)
                    result_lines.append(f"{indent}pass")
                i += 1
                continue

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines)

    def _has_concrete_parent(self, class_name: str) -> bool:
        """Check if *class_name* has a concrete (non-interface) parent in the index."""
        parents = self.index.hierarchy.get(class_name, [])
        for parent in parents:
            cls = self.index.classes.get(parent)
            if cls and cls.symbol_type != "interface":
                return True
        return False

    def _generate_constructor_from_java(self, entry: FragmentEntry) -> str | None:
        """Generate a deterministic Python __init__ from a Java constructor.

        Parses the Java source for the constructor signature, super() call,
        and field assignments, then emits the equivalent Python __init__.
        Returns None if any parsing step fails (caller falls back to MOCKED).
        """
        if not entry or not entry.java_source:
            return None

        class_name = entry.class_name
        java = re.sub(r"\s+", " ", entry.java_source)

        # 1. Extract constructor params — find all constructors, pick the
        # non-deprecated, simplest one.  Use brace-aware extraction since
        # annotations like @SuppressWarnings("unused") contain parentheses.
        ctor_header_pat = re.compile(
            r"(?:public|protected)\s+" + re.escape(class_name) + r"\s*\("
        )
        candidates: list[str] = []
        for m in ctor_header_pat.finditer(java):
            start = m.end()
            depth = 1
            pos = start
            while pos < len(java) and depth > 0:
                if java[pos] == "(":
                    depth += 1
                elif java[pos] == ")":
                    depth -= 1
                pos += 1
            if depth == 0:
                raw = java[start:pos - 1].strip()
                # Skip deprecated constructors (have @Deprecated before them)
                prefix = java[max(0, m.start() - 80):m.start()]
                if "@Deprecated" in prefix:
                    continue
                candidates.append(raw)

        if not candidates:
            return None
        # Prefer the constructor with fewest params (simplest)
        raw_params = min(candidates, key=len)
        param_names: list[str] = []
        has_varargs = False
        if raw_params:
            for part in raw_params.split(","):
                part = part.strip()
                if not part:
                    continue
                # Detect varargs: "Type... name"
                vm = re.match(r"(?:final\s+)?(?:[\w.<>,\s\[\]?]+)\.\.\.\s+(\w+)$", part)
                if vm:
                    param_names.append(vm.group(1))
                    has_varargs = True
                    continue
                # Match "final Type<G> name" or "Type name" — take the last word
                pm = re.match(r"(?:final\s+)?(?:[\w.<>,\s\[\]?]+)\s+(\w+)$", part)
                if pm:
                    param_names.append(pm.group(1))
                else:
                    return None  # Can't parse param

        # 2. Extract super() arguments
        super_m = re.search(r"super\s*\(([^)]*)\)\s*;", java)
        super_args: list[str] = []
        if super_m:
            raw_super = super_m.group(1).strip()
            if raw_super:
                super_args = [a.strip() for a in raw_super.split(",") if a.strip()]

        # 3. Extract field assignments: this.field = param; or this.field = Arrays.asList(param);
        #    Deduplicate by field name (multiple constructors may have same assignments)
        field_assigns: list[tuple[str, str]] = []
        seen_fields: set[str] = set()
        for fm in re.finditer(
            r"this\.(\w+)\s*=\s*(?:Arrays\.asList\s*\(\s*(\w+)\s*\)|(\w+))\s*;", java
        ):
            field_name = fm.group(1)
            if field_name in seen_fields:
                continue
            seen_fields.add(field_name)
            # group(2) = param inside Arrays.asList(), group(3) = plain param
            value_name = fm.group(2) or fm.group(3)
            # Only keep assignments sourced from a constructor param
            if value_name in param_names:
                py_field = self._to_snake_case(field_name)
                # Arrays.asList(x) → list(x); also wrap varargs plain assigns in list()
                if fm.group(2):
                    field_assigns.append((py_field, f"list({value_name})"))
                elif has_varargs and value_name == param_names[-1]:
                    field_assigns.append((py_field, f"list({value_name})"))
                else:
                    field_assigns.append((py_field, value_name))

        # 4. Build Python __init__
        if has_varargs:
            # Last param becomes *param
            py_param_parts = param_names[:-1] + [f"*{param_names[-1]}"]
        else:
            py_param_parts = list(param_names)
        py_params = ", ".join(py_param_parts)
        lines: list[str] = [f"def __init__(self, {py_params}):" if py_params else "def __init__(self):"]

        has_parent = self._has_concrete_parent(class_name)
        if super_args and has_parent:
            lines.append(f"    super().__init__({', '.join(super_args)})")
        elif has_parent and not has_varargs:
            # Skip super().__init__() for varargs ctors with no concrete parent super() call
            lines.append("    super().__init__()")

        for py_field, value_name in field_assigns:
            lines.append(f"    self.{py_field} = {value_name}")

        # Must have at least one statement in the body
        if len(lines) == 1:
            lines.append("    pass")

        return "\n".join(lines)

    # Java→Python dunder method name mapping
    _DUNDER_MAP: dict[str, str] = {
        "toString": "__str__",
        "equals": "__eq__",
        "hashCode": "__hash__",
        "compareTo": "__lt__",
        "iterator": "__iter__",
    }

    # Known Java method → Python function mappings for return expressions
    _METHOD_CALL_MAP: dict[str, str] = {
        "toString": "str",
        "iterator": "iter",
        "clone": "list",
        "size": "len",
    }

    def _generate_method_from_java(self, entry: FragmentEntry) -> str | None:
        """Generate deterministic Python method from simple Java source patterns.

        Handles:
        1. Simple field/constant return: return SIZE; → return self.SIZE
        2. Method-call return: return this.field.toString() → return str(self.field)
        3. Standard equals(Object) → __eq__
        4. Standard compareTo(Type) → __lt__

        Returns None if the Java source doesn't match any known pattern.
        """
        if not entry or not entry.java_source or not entry.member_name:
            return None

        java = re.sub(r"\s+", " ", entry.java_source).strip()

        # Determine the Python method name
        base_name = entry.member_name.split("$")[0]  # Strip overload discriminator
        py_name = self._DUNDER_MAP.get(base_name, self._to_snake_case(base_name))

        # --- Pattern 3: Standard equals(Object) ---
        if base_name == "equals":
            return self._gen_equals(entry, java)

        # --- Pattern 4: Standard compareTo(Type) ---
        if base_name == "compareTo":
            return self._gen_compare_to(entry, java)

        # --- Pattern 1: Simple return of field/constant ---
        # Matches: return SIZE; or return this.field;
        m = re.search(r"\breturn\s+(this\.)?(\w+)\s*;", java)
        if m:
            has_this = m.group(1) is not None
            field = m.group(2)
            py_field = self._to_snake_case(field) if has_this else field
            # Constants (ALL_CAPS) keep their name
            if field.isupper() or (field.upper() == field and "_" in field):
                py_field = field
            return f"def {py_name}(self):\n    return self.{py_field}"

        # --- Pattern 2: Method-call return ---
        # Matches: return this.field.method(); or return this.field.method(args);
        m = re.search(r"\breturn\s+this\.(\w+)\.(\w+)\s*\(\s*\)\s*;", java)
        if m:
            field = m.group(1)
            method = m.group(2)
            py_field = self._to_snake_case(field)
            py_func = self._METHOD_CALL_MAP.get(method)
            if py_func:
                return f"def {py_name}(self):\n    return {py_func}(self.{py_field})"

        return None

    def _gen_equals(self, entry: FragmentEntry, java: str) -> str | None:
        """Generate __eq__ from a standard Java equals(Object) pattern."""
        # Extract the field comparison: look for this.FIELD.equals(other.FIELD)
        # or Arrays.equals(this.field, other.field)
        field_m = re.search(
            r"(?:this\.(\w+)\.equals\s*\(\s*\w+\.(\w+)\s*\)"
            r"|Arrays\.equals\s*\(\s*this\.(\w+)\s*,\s*\w+\.(\w+)\s*\))",
            java,
        )
        if not field_m:
            return None

        field1 = field_m.group(1) or field_m.group(3)
        py_field = self._to_snake_case(field1)

        lines = [
            "def __eq__(self, other):",
            "    if self is other:",
            "        return True",
            f"    if not isinstance(other, self.__class__):",
            "        return NotImplemented",
            f"    return self.{py_field} == other.{py_field}",
        ]
        return "\n".join(lines)

    def _gen_compare_to(self, entry: FragmentEntry, java: str) -> str | None:
        """Generate __lt__ from a standard Java compareTo pattern."""
        # Look for element-wise comparison of an array field
        field_m = re.search(
            r"this\.(\w+)\s*(?:\[|\.(?:get|length))", java
        )
        if not field_m:
            return None

        field = field_m.group(1)
        py_field = self._to_snake_case(field)

        lines = [
            "def __lt__(self, other):",
            f"    for a, b in zip(self.{py_field}, other.{py_field}):",
            "        if a < b:",
            "            return True",
            "        if a > b:",
            "            return False",
            f"    return len(self.{py_field}) < len(other.{py_field})",
        ]
        return "\n".join(lines)

    def _extract_and_indent(
        self,
        entry: FragmentEntry,
        target_indent: str,
        context_lines: list[str],
        marker_line_idx: int,
    ) -> list[str]:
        """Extract the method body from translated code and indent it.

        The fragment's target_code may contain a full standalone class with
        redundant imports and class wrapper. We extract just the relevant
        method body.

        Args:
            entry: The fragment entry
            target_indent: Indentation to apply
            context_lines: All lines of the skeleton
            marker_line_idx: Index of the marker line

        Returns:
            List of properly indented lines
        """
        code = entry.target_code
        if not code:
            return [f"{target_indent}pass  # Empty fragment: {entry.fragment_id}"]

        # Determine if this is a class-level or method-level injection
        # Some method fragments (e.g., interface methods) have member_name=None
        # but symbol_type="method" — route those through method extraction.
        is_class_level = entry.member_name is None and entry.symbol_type not in ("method", "function")

        if is_class_level:
            # For class-level fragments, extract the class body
            body = self._extract_class_body(code, entry)
        elif entry.symbol_type in ("global_var", "constant", "field"):
            # For field/variable fragments, extract the assignment
            body = self._extract_field(code, entry)
        else:
            # For method-level fragments, extract the method body
            body = self._extract_method_body(code, entry)

        if not body:
            if is_class_level:
                # All methods were filtered (they'll be individually injected).
                # Emit pass instead of dumping raw code with class wrappers.
                body = "pass  # class body (methods injected individually)"
            else:
                # Fallback: use the raw code, dedented
                body = textwrap.dedent(code).strip()

        # For field/constant fragments, validate the assignment executes safely.
        # Phase 2/3 translations often have broken initializers (A(), value, etc.)
        if entry.symbol_type in ("global_var", "constant", "field"):
            # Strip type annotations before exec-testing so that
            # `ROWS: Final[List[int]] = 3` becomes `ROWS = 3` for the test.
            # The original body (with annotations) is preserved if the test passes.
            test_body = re.sub(r':\s*[^=]+=', ' =', body, count=1)
            try:
                exec(compile(test_body, "<field>", "exec"), _TYPING_STUBS.copy())
            except Exception:
                # Replace the value part with None, keep the annotation
                body = re.sub(r'=\s*.+$', '= None', body, count=1)

        # Re-indent to match the target indentation
        return self._reindent(body, target_indent)

    def _extract_method_body(self, code: str, entry: FragmentEntry) -> str | None:
        """Extract the method body from translated code.

        Handles cases where the fragment contains:
        1. Just a method def
        2. A method inside a class wrapper (picks the right one by name)
        3. Raw code without def (constants, assignments)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Can't parse — return raw code
            return textwrap.dedent(code).strip()

        # Collect ALL function definitions
        func_nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if func_nodes:
            # Try to match by python_name, member_name, or snake_case variants
            target_names = set()
            if entry.python_name:
                target_names.add(entry.python_name)
            if entry.member_name:
                target_names.add(entry.member_name)
                target_names.add(self._to_snake_case(entry.member_name))

            # Prefer exact name match
            best = None
            for node in func_nodes:
                if node.name in target_names:
                    best = node
                    break

            # Fallback: if fragment has only one method (besides __init__
            # in class wrappers), use it
            if best is None:
                non_init = [n for n in func_nodes if n.name != "__init__"]
                if len(non_init) == 1:
                    best = non_init[0]
                elif func_nodes:
                    # Last resort: use the last function def (usually the
                    # target method, after class wrapper boilerplate)
                    best = func_nodes[-1]

            if best:
                func_lines = code.splitlines()
                start = best.lineno - 1
                end = best.end_lineno if best.end_lineno else len(func_lines)
                func_code = "\n".join(func_lines[start:end])
                return textwrap.dedent(func_code).strip()

        # No function def found — look for assignments (constants)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assign_lines = code.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if node.end_lineno else start + 1
                return "\n".join(assign_lines[start:end]).strip()

        return None

    def _extract_field(self, code: str, entry: FragmentEntry) -> str | None:
        """Extract a field/variable assignment from translated code.

        Handles class wrappers like 'class MyClass: VAL = ...' by
        extracting just the assignment.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return textwrap.dedent(code).strip()

        # Look for assignments or annotated assignments inside class bodies
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                        lines = code.splitlines()
                        start = stmt.lineno - 1
                        end = stmt.end_lineno if stmt.end_lineno else start + 1
                        return textwrap.dedent("\n".join(lines[start:end])).strip()

        # No class wrapper — look for top-level assignments
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                lines = code.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if node.end_lineno else start + 1
                return textwrap.dedent("\n".join(lines[start:end])).strip()

        return None

    def _extract_class_body(self, code: str, entry: FragmentEntry | None = None) -> str | None:
        """Extract the class body from translated code, filtering out methods.

        For class-level fragments, we want only:
        - Field assignments / annotated assignments / constants
        - ``__init__`` constructor
        - Inner class definitions
        - Docstrings (``Expr`` with string value)

        Methods that have their own fragment entry in the index are excluded
        because they will be injected individually at their own ``# INJECT:``
        markers.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return textwrap.dedent(code).strip()

        class_name = entry.class_name if entry else None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_lines = code.splitlines()
                if not node.body:
                    continue

                kept_ranges: list[tuple[int, int]] = []
                for stmt in node.body:
                    # Assignments / annotated assignments: skip if the field
                    # has its own fragment (it will be injected individually).
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                        field_name = _get_assign_name(stmt)
                        if field_name and class_name and self._field_has_fragment(class_name, field_name):
                            continue  # will be injected at its own # INJECT: marker
                        start = stmt.lineno - 1
                        end = stmt.end_lineno if stmt.end_lineno else start + 1
                        kept_ranges.append((start, end))
                        continue

                    # Inner class definitions: always keep
                    if isinstance(stmt, ast.ClassDef):
                        start = stmt.lineno - 1
                        end = stmt.end_lineno if stmt.end_lineno else start + 1
                        kept_ranges.append((start, end))
                        continue

                    # Keep docstrings (Expr nodes with a string constant)
                    if isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, (ast.Constant, ast.Str)
                    ):
                        start = stmt.lineno - 1
                        end = stmt.end_lineno if stmt.end_lineno else start + 1
                        kept_ranges.append((start, end))
                        continue

                    # For FunctionDef / AsyncFunctionDef: keep __init__, skip
                    # methods that have their own fragment in the index.
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if stmt.name == "__init__":
                            start = stmt.lineno - 1
                            end = stmt.end_lineno if stmt.end_lineno else start + 1
                            kept_ranges.append((start, end))
                            continue

                        # Check if this method has a matching fragment entry
                        if class_name and self._method_has_fragment(class_name, stmt.name):
                            # Skip — it will be injected at its own marker
                            continue

                        # No matching fragment — keep it (inner helper, etc.)
                        start = stmt.lineno - 1
                        end = stmt.end_lineno if stmt.end_lineno else start + 1
                        kept_ranges.append((start, end))
                        continue

                    # Keep anything else (pass, raise, etc.)
                    start = stmt.lineno - 1
                    end = stmt.end_lineno if stmt.end_lineno else start + 1
                    kept_ranges.append((start, end))

                if not kept_ranges:
                    return None

                # Reassemble kept statements by their line ranges
                parts: list[str] = []
                for start, end in kept_ranges:
                    parts.append("\n".join(class_lines[start:end]))
                body_code = "\n".join(parts)
                return textwrap.dedent(body_code).strip()

        # No class found — return the whole thing
        return textwrap.dedent(code).strip()

    def _method_has_fragment(self, class_name: str, method_name: str) -> bool:
        """Check if a method has a matching fragment entry in the index.

        Handles:
        - Standard: ``GameBoard::GameBoard.getBoard``
        - Interface (no dot): ``DiscreteGameState::availableStates``
        - Inner class: ``MinimaxAgent::Node.buildTree$...``
        - Overload discriminators (``$`` suffix)
        - snake_case ↔ camelCase and dunder name mappings
        """
        # Collect all Java-side name variants for this Python method
        java_names = [method_name]

        # snake_case → camelCase: "get_board" → "getBoard"
        parts = method_name.split("_")
        if len(parts) > 1:
            java_names.append(parts[0] + "".join(p.capitalize() for p in parts[1:]))

        # Dunder reverse lookup: __str__ → toString, __eq__ → equals, etc.
        _REVERSE_DUNDER = {v: k for k, v in self._DUNDER_MAP.items()}
        if method_name in _REVERSE_DUNDER:
            java_names.append(_REVERSE_DUNDER[method_name])

        # Build candidate fragment ID patterns for each name variant.
        # Three patterns per name:
        #   ClassName::ClassName.method  (standard)
        #   ClassName::method            (interface / flat)
        #   ClassName::*.method          (inner class — handled via prefix scan)
        candidates: list[str] = []
        for jname in java_names:
            candidates.append(f"{class_name}::{class_name}.{jname}")
            candidates.append(f"{class_name}::{jname}")

        for candidate in candidates:
            if candidate in self.index.fragments:
                return True
            # Check overloaded variants: fragment_id$Type1_Type2
            for fid in self.index.fragments:
                if fid.startswith(candidate + "$"):
                    return True

        # Scan for inner-class patterns: ClassName::InnerClass.method
        prefix = f"{class_name}::"
        for jname in java_names:
            suffix_dot = f".{jname}"
            for fid in self.index.fragments:
                if fid.startswith(prefix) and (
                    fid.endswith(suffix_dot)
                    or (suffix_dot + "$") in fid
                    or fid.split("::", 1)[-1] == jname
                ):
                    return True

        # Also check if any fragment's python_name matches this method
        for fid, fentry in self.index.fragments.items():
            if fid.startswith(prefix) and fentry.python_name == method_name:
                return True

        return False

    def _field_has_fragment(self, class_name: str, field_name: str) -> bool:
        """Check if a field has its own fragment entry in the index."""
        candidates = [
            f"{class_name}::{class_name}.{field_name}",
            f"{class_name}::{field_name}",
        ]
        return any(c in self.index.fragments for c in candidates)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert camelCase or PascalCase to snake_case."""
        import re as _re
        s = _re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        s = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        return s.lower()

    def _reindent(self, code: str, indent: str) -> list[str]:
        """Re-indent code lines to match a target indentation."""
        lines = textwrap.dedent(code).splitlines()
        result = []
        for line in lines:
            if line.strip():
                result.append(indent + line)
            else:
                result.append("")
        return result

    def inject_all(
        self,
        skeletons: dict[str, str],
    ) -> dict[str, str]:
        """Inject code into all class skeletons.

        Args:
            skeletons: Map of class_name -> skeleton code

        Returns:
            Map of class_name -> assembled code
        """
        results = {}
        total_injected = 0
        total_mocked = 0

        for class_name, skeleton in skeletons.items():
            assembled = self.inject(skeleton)
            results[class_name] = assembled

            # Count results
            total_injected += assembled.count("# INJECT:") == 0  # All injected if none remain
            total_mocked += assembled.count("# MOCKED:")
            total_mocked += assembled.count("# MISSING:")

        logger.info(f"Injected code into {len(skeletons)} classes, "
                     f"{total_mocked} mocked markers remaining")
        return results
