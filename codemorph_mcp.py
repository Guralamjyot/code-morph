#!/usr/bin/env python3
"""
CodeMorph MCP Server — exposes translation state tools to external agents.

Provides tools for reading original Java source, translated Python fragments,
class member listings, hierarchy info, and symbol registry lookups.
Designed to be used by OpenCode (or any MCP client) during Phase 4 fixing.
"""

import json
import re
import subprocess
from pathlib import Path

import orjson
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration — set via env or defaults
# ---------------------------------------------------------------------------

import os

STATE_DIR = Path(os.environ.get("CODEMORPH_STATE_DIR", ".codemorph"))
SOURCE_DIR = Path(os.environ.get("CODEMORPH_SOURCE_DIR", "examples/javatuples/src"))
PROJECT_ROOT = Path(os.environ.get("CODEMORPH_PROJECT_ROOT", "/workspace/code-agent/code-convert"))
OUTPUT_DIR = Path(os.environ.get("CODEMORPH_OUTPUT_DIR", "")) if os.environ.get("CODEMORPH_OUTPUT_DIR") else None

# Resolve relative paths against project root
if not STATE_DIR.is_absolute():
    STATE_DIR = PROJECT_ROOT / STATE_DIR
if not SOURCE_DIR.is_absolute():
    SOURCE_DIR = PROJECT_ROOT / SOURCE_DIR
if OUTPUT_DIR and not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR


# ---------------------------------------------------------------------------
# Load state once at startup
# ---------------------------------------------------------------------------

def _load_state():
    latest = STATE_DIR / "latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"No latest.json in {STATE_DIR}")
    with open(latest, "rb") as f:
        return orjson.loads(f.read())


def _load_registry():
    reg_file = STATE_DIR / "symbol_registry.json"
    if reg_file.exists():
        with open(reg_file, "rb") as f:
            return orjson.loads(f.read())
    return {}


STATE = _load_state()
REGISTRY = _load_registry()

# Pre-index fragments and classes
FRAGMENTS = STATE.get("translated_fragments", {})
ANALYSIS = STATE.get("analysis_result", {})
ALL_FRAGMENTS = ANALYSIS.get("fragments", {})

# Build class → fragment mapping
CLASS_MEMBERS: dict[str, list[str]] = {}
for fid in ALL_FRAGMENTS:
    cls = fid.split("::")[0] if "::" in fid else fid
    CLASS_MEMBERS.setdefault(cls, []).append(fid)

# Build hierarchy from analysis
HIERARCHY: dict[str, list[str]] = {}
for fid, frag in ALL_FRAGMENTS.items():
    cls = fid.split("::")[0]
    if cls not in HIERARCHY:
        source_code = frag.get("source_code", "")
        bases = []
        # Extract from Java source: class X extends Y implements Z, W
        m = re.search(r"class\s+\w+\s+extends\s+(\w+)", source_code)
        if m:
            bases.append(m.group(1))
        m2 = re.search(r"implements\s+([\w\s,]+)", source_code)
        if m2:
            bases.extend(b.strip() for b in m2.group(1).split(","))
        HIERARCHY[cls] = bases


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("codemorph")


@mcp.tool()
def read_source_file(filename: str) -> str:
    """Read an original Java source file.

    Args:
        filename: Source filename, e.g. 'Pair.java'
    """
    target = SOURCE_DIR / filename
    if not target.exists():
        # Search recursively
        matches = list(SOURCE_DIR.rglob(filename))
        if matches:
            target = matches[0]
        else:
            return f"File not found: {filename}. Available: {[f.name for f in SOURCE_DIR.rglob('*.java')]}"

    return target.read_text(encoding="utf-8")


@mcp.tool()
def read_fragment(fragment_id: str) -> str:
    """Read the translated Python code for a specific fragment.

    Shows both the original Java source and translated Python.

    Args:
        fragment_id: Fragment ID, e.g. 'Pair::Pair.fromArray' or 'Tuple::Tuple'
    """
    # Exact match
    translated = FRAGMENTS.get(fragment_id)
    source_frag = ALL_FRAGMENTS.get(fragment_id)

    if not translated and not source_frag:
        # Fuzzy match
        candidates = [fid for fid in {**FRAGMENTS, **ALL_FRAGMENTS}
                      if fragment_id.lower() in fid.lower()]
        if candidates:
            return f"Fragment '{fragment_id}' not found. Did you mean:\n" + "\n".join(candidates[:10])
        return f"Fragment '{fragment_id}' not found."

    parts = [f"Fragment: {fragment_id}"]

    if source_frag:
        parts.append(f"Type: {source_frag.get('fragment_type', 'unknown')}")
        src = source_frag.get("source_code", "")
        if src:
            parts.append(f"\n--- Java Source ---\n{src}")

    if translated:
        status = translated.get("status", "unknown")
        parts.append(f"Status: {status}")
        target_code = translated.get("target_code", "")
        if target_code:
            parts.append(f"\n--- Translated Python ---\n{target_code}")

    return "\n".join(parts)


@mcp.tool()
def list_class_members(class_name: str) -> str:
    """List all members of a class with their translation status.

    Args:
        class_name: Class name, e.g. 'Pair', 'Tuple', 'Decade'
    """
    members = CLASS_MEMBERS.get(class_name, [])
    if not members:
        available = sorted(CLASS_MEMBERS.keys())
        return f"Class '{class_name}' not found. Available classes: {available}"

    lines = [
        f"Class: {class_name}",
        f"Members: {len(members)}",
        f"Bases: {HIERARCHY.get(class_name, [])}",
        "",
    ]

    for fid in sorted(members):
        translated = FRAGMENTS.get(fid)
        if translated:
            status = translated.get("status", "unknown")
            target_code = translated.get("target_code", "")
            code_len = len(target_code) if target_code else 0
            lines.append(f"  [{status}] {fid} ({code_len} chars)")
        else:
            lines.append(f"  [NOT_TRANSLATED] {fid}")

    return "\n".join(lines)


@mcp.tool()
def get_class_hierarchy(class_name: str) -> str:
    """Get the class hierarchy. If class_name is non-empty, show just that class's bases. Pass empty string to show all.

    Args:
        class_name: Class name to look up, or empty string for all classes
    """
    if class_name:
        bases = HIERARCHY.get(class_name, [])
        if not bases:
            return f"Class '{class_name}' not found or has no bases."
        return f"{class_name} extends/implements: {bases}"

    lines = ["Class Hierarchy:", ""]
    for cls in sorted(HIERARCHY.keys()):
        bases = HIERARCHY[cls]
        if bases:
            lines.append(f"  {cls} -> {', '.join(bases)}")
        else:
            lines.append(f"  {cls} (no bases)")
    return "\n".join(lines)


@mcp.tool()
def lookup_registry(java_name: str, python_name: str, class_name: str) -> str:
    """Look up Java to Python name mappings in the symbol registry. Pass empty string for fields you don't want to filter by.

    Args:
        java_name: Search by Java name (or empty string to skip)
        python_name: Search by Python name (or empty string to skip)
        class_name: List all mappings for a class (or empty string to skip)
    """
    mappings = REGISTRY.get("mappings", {})
    if not mappings:
        return "Symbol registry is empty or not loaded."

    results = []

    if java_name:
        for key, val in mappings.items():
            if java_name.lower() in key.lower():
                results.append(f"  {key} -> {val}")

    if python_name:
        for key, val in mappings.items():
            if isinstance(val, dict):
                pname = val.get("python_name", "")
                if python_name.lower() in pname.lower():
                    results.append(f"  {key} -> {val}")
            elif isinstance(val, str) and python_name.lower() in val.lower():
                results.append(f"  {key} -> {val}")

    if class_name:
        for key, val in mappings.items():
            if key.startswith(class_name + "::") or key.startswith(class_name + "."):
                results.append(f"  {key} -> {val}")

    if not results:
        return f"No matches found. Registry has {len(mappings)} entries."

    return f"Found {len(results)} mappings:\n" + "\n".join(results[:50])


@mcp.tool()
def get_project_summary(detail_level: str) -> str:
    """Get a compact summary of the entire translation project.

    Args:
        detail_level: Use 'brief' for class names only, or 'full' for complete details
    """
    total_frags = len(ALL_FRAGMENTS)
    translated = len(FRAGMENTS)
    classes = sorted(CLASS_MEMBERS.keys())

    status_counts: dict[str, int] = {}
    for fid, tf in FRAGMENTS.items():
        s = tf.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    lines = [
        f"Project: Java -> Python",
        f"Classes: {len(classes)}",
        f"Total fragments: {total_frags}",
        f"Translated: {translated}",
        f"Status breakdown: {status_counts}",
    ]

    if detail_level != "brief":
        lines.append("")
        lines.append("Classes:")
        for cls in classes:
            members = CLASS_MEMBERS[cls]
            translated_count = sum(1 for m in members if m in FRAGMENTS)
            bases = HIERARCHY.get(cls, [])
            base_str = f" extends {', '.join(bases)}" if bases else ""
            lines.append(f"  {cls}{base_str} — {len(members)} members, {translated_count} translated")
    else:
        lines.append(f"Classes: {', '.join(classes)}")

    return "\n".join(lines)


@mcp.tool()
def read_output_file(file_path: str) -> str:
    """Read an assembled Python output file.

    Args:
        file_path: Relative path within the output directory, e.g. 'tictactoe/game_board.py'
    """
    if OUTPUT_DIR is None:
        return "OUTPUT_DIR not configured. Set CODEMORPH_OUTPUT_DIR env var."

    target = OUTPUT_DIR / file_path
    if not target.exists():
        # Try searching recursively
        name = Path(file_path).name
        matches = list(OUTPUT_DIR.rglob(name))
        if matches:
            target = matches[0]
        else:
            available = [str(f.relative_to(OUTPUT_DIR)) for f in OUTPUT_DIR.rglob("*.py")]
            return f"File not found: {file_path}. Available: {available}"

    return target.read_text(encoding="utf-8")


@mcp.tool()
def list_output_files(pattern: str) -> str:
    """List files in the assembled output directory.

    Args:
        pattern: Glob pattern to filter files, e.g. '*.py' for all Python files
    """
    if OUTPUT_DIR is None:
        return "OUTPUT_DIR not configured. Set CODEMORPH_OUTPUT_DIR env var."

    if not OUTPUT_DIR.exists():
        return f"Output directory does not exist: {OUTPUT_DIR}"

    matched_files = sorted(OUTPUT_DIR.rglob(pattern))
    if not matched_files:
        return f"No files matching '{pattern}' found in output directory."

    lines = [f"Output directory: {OUTPUT_DIR}", f"Files ({len(matched_files)}):"]
    for f in matched_files:
        rel = f.relative_to(OUTPUT_DIR)
        size = f.stat().st_size
        lines.append(f"  {rel} ({size} bytes)")

    return "\n".join(lines)


@mcp.tool()
def run_test_script(script: str, timeout: int = 15) -> str:
    """Execute a Python test script against the assembled output.
    The output directory is on sys.path so you can import the package directly.

    Args:
        script: Python code to execute
        timeout: Max execution time in seconds (default 15)
    """
    if OUTPUT_DIR is None:
        return "OUTPUT_DIR not configured. Set CODEMORPH_OUTPUT_DIR env var."

    if not OUTPUT_DIR.exists():
        return f"Output directory does not exist: {OUTPUT_DIR}"

    import sys
    python_bin = sys.executable or "python3"

    env = {**os.environ, "PYTHONPATH": str(OUTPUT_DIR)}

    try:
        result = subprocess.run(
            [python_bin, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(OUTPUT_DIR),
            env=env,
        )

        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        parts.append(f"EXIT CODE: {result.returncode}")
        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return f"TIMEOUT: Script exceeded {timeout}s limit."
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    mcp.run()
