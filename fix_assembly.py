#!/usr/bin/env python3.13
"""Fix systematic assembly issues in javatuples-python output."""

import re
from pathlib import Path

OUTPUT_DIR = Path("/workspace/code-agent/javatuples-python/javatuples")


def fix_fragment_id_refs(code: str) -> str:
    """Replace ClassName::ClassName.field patterns with just field."""
    # self.ClassName::ClassName.field -> self.field
    code = re.sub(r'self\.(\w+)::(\w+)\.(\w+)', r'self.\3', code)
    # ClassName::ClassName.field at class level (as attribute/assignment)
    # e.g. "    Decade::Decade.SIZE = 10" -> "    SIZE = 10"
    code = re.sub(r'^(\s*)(\w+)::(\w+)\.(\w+)', r'\1\4', code, flags=re.MULTILINE)
    return code


def fix_duplicate_method_stubs(code: str) -> str:
    """Remove stub method lines that precede the real definition.

    Pattern:
        def method(self):
    def method(self) -> int:

    The stub (no return type, body is next def) should be removed.
    """
    lines = code.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is a def line followed by another def with the same name
        m = re.match(r'^(\s*)def (\w+)\(', line)
        if m and i + 1 < len(lines):
            indent = m.group(1)
            name = m.group(2)
            next_line = lines[i + 1]
            # Check if next line is a def with same name (duplicate stub)
            m2 = re.match(r'^def ' + re.escape(name) + r'\(', next_line)
            if m2:
                # Skip the stub line, keep the real definition (but indent it)
                result.append(indent + next_line.lstrip())
                i += 2
                continue
        result.append(line)
        i += 1
    return '\n'.join(result)


def fix_indentation(code: str) -> str:
    """Fix methods defined at module level that should be in the class."""
    lines = code.split('\n')
    result = []
    in_class = False
    class_indent = ""

    for i, line in enumerate(lines):
        # Track class definitions
        m = re.match(r'^(\s*)class \w+', line)
        if m:
            in_class = True
            class_indent = m.group(1)

        # If we're in a class and find a module-level def, indent it
        if in_class and re.match(r'^def \w+\(self', line):
            result.append(class_indent + "    " + line)
            continue

        result.append(line)

    return '\n'.join(result)


def fix_generic_subscripts(code: str) -> str:
    """Remove runtime generic subscript usage like Decade[A, B, C](...) -> Decade(...)."""
    # ClassName[TypeParams](args) -> ClassName(args)
    code = re.sub(r'(\b(?:Unit|Pair|Triplet|Quartet|Quintet|Sextet|Septet|Octet|Ennead|Decade)\b)\[[^\]]+\]\(', r'\1(', code)
    return code


def fix_mocked_comments(code: str) -> str:
    """Remove leftover MOCKED comments."""
    code = re.sub(r'\s*# MOCKED:.*$', '', code, flags=re.MULTILINE)
    return code


def fix_file(filepath: Path) -> bool:
    """Apply all fixes to a single file. Returns True if modified."""
    original = filepath.read_text()
    code = original

    code = fix_fragment_id_refs(code)
    code = fix_duplicate_method_stubs(code)
    code = fix_indentation(code)
    code = fix_generic_subscripts(code)
    code = fix_mocked_comments(code)

    if code != original:
        filepath.write_text(code)
        return True
    return False


def main():
    import py_compile

    files = sorted(OUTPUT_DIR.glob("*.py"))
    print(f"Fixing {len(files)} files...")

    for f in files:
        modified = fix_file(f)
        status = "MODIFIED" if modified else "unchanged"

        # Check compile status
        try:
            py_compile.compile(str(f), doraise=True)
            compile_status = "OK"
        except py_compile.PyCompileError as e:
            compile_status = f"ERROR: {e}"

        print(f"  {f.name}: {status} | {compile_status}")


if __name__ == "__main__":
    main()
