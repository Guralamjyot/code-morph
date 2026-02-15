"""
Agent Tool Registry for Phase 4 Assembly.

Provides callable tools that the LLM agent can invoke to explore and understand
the source and target projects. This is what makes the agent autonomous.
"""

import ast
import json
import logging
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from codemorph.assembler.index_builder import ProjectIndex
from codemorph.config.models import CodeMorphConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool invocation."""

    success: bool
    output: str
    data: Any = None


@dataclass
class ToolDefinition:
    """Metadata for a registered tool."""

    name: str
    description: str
    parameters: dict[str, str]
    func: Callable[..., ToolResult]


class AgentToolRegistry:
    """Registry of tools the assembly agent can call."""

    def __init__(
        self,
        index: ProjectIndex,
        config: CodeMorphConfig,
        output_dir: Path | None = None,
    ):
        self.index = index
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("./assembled")
        self.tools: dict[str, ToolDefinition] = {}
        self._vector_store = None
        self._register_all()

    def _register_all(self) -> None:
        """Register all available tools."""
        self._register(
            "grep_source",
            "Search the source project files for a regex pattern.",
            {"pattern": "Regex pattern to search for", "file_filter": "Glob filter (default: *.java)"},
            self._grep_source,
        )
        self._register(
            "grep_translated",
            "Search across all translated Python code fragments.",
            {"pattern": "Regex pattern to search for", "class_filter": "Optional class name to limit search"},
            self._grep_translated,
        )
        self._register(
            "read_source_file",
            "Read the full content of a source file.",
            {"filename": "Source filename (e.g., 'Pair.java')"},
            self._read_source_file,
        )
        self._register(
            "read_fragment",
            "Read the translated Python code for a specific fragment.",
            {"fragment_id": "Fragment ID (e.g., 'Pair::Pair.fromArray')"},
            self._read_fragment,
        )
        self._register(
            "lookup_registry",
            "Look up name mappings in the symbol registry.",
            {"java_name": "Search by Java name", "python_name": "Search by Python name", "class_name": "List all mappings for a class"},
            self._lookup_registry,
        )
        self._register(
            "list_class_members",
            "List all members of a class with their status.",
            {"class_name": "Class name (e.g., 'Pair')"},
            self._list_class_members,
        )
        self._register(
            "get_class_hierarchy",
            "Get the class hierarchy from the source project.",
            {"class_name": "Optional specific class (default: full hierarchy)"},
            self._get_class_hierarchy,
        )
        self._register(
            "run_script",
            "Execute a Python script and return output.",
            {"code": "Python code to execute", "timeout": "Max execution time in seconds (default: 30)"},
            self._run_script,
        )
        self._register(
            "write_file",
            "Write content to a file in the output project directory.",
            {"path": "Relative path within the output dir", "content": "File content"},
            self._write_file,
        )
        self._register(
            "read_file",
            "Read content of any file in the project.",
            {"path": "Relative path from project root or output dir"},
            self._read_file,
        )
        self._register(
            "rag_search",
            "Search the knowledge base for similar code patterns or translation examples.",
            {"query": "Natural language or code query", "category": "Filter by category", "top_k": "Number of results (default: 3)"},
            self._rag_search,
        )
        self._register(
            "compile_check",
            "Check if a Python file compiles without syntax errors.",
            {"file_path": "Path to the .py file (relative to output dir or absolute)"},
            self._compile_check,
        )
        self._register(
            "get_project_summary",
            "Get a compact summary of the entire project index.",
            {},
            self._get_project_summary,
        )

    def _register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        func: Callable[..., ToolResult],
    ) -> None:
        """Register a tool."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )

    def get_tool_descriptions(self) -> str:
        """Return tool descriptions formatted for the LLM system prompt."""
        lines = ["## Available Tools", "", "Call tools using JSON: {\"tool\": \"name\", \"args\": {\"key\": \"value\"}}", ""]
        for tool in self.tools.values():
            lines.append(f"### {tool.name}")
            lines.append(tool.description)
            if tool.parameters:
                lines.append("Parameters:")
                for param, desc in tool.parameters.items():
                    lines.append(f"  - {param}: {desc}")
            lines.append("")
        return "\n".join(lines)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments."""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {tool_name}. Available: {', '.join(self.tools.keys())}",
            )
        try:
            return self.tools[tool_name].func(**kwargs)
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed")
            return ToolResult(success=False, output=f"Tool error: {e}")

    # =========================================================================
    # Tool implementations
    # =========================================================================

    def _grep_source(self, pattern: str, file_filter: str = "*.java") -> ToolResult:
        """Search source project files for a regex pattern."""
        source_root = Path(self.config.project.source.root)
        if not source_root.exists():
            return ToolResult(success=False, output=f"Source directory not found: {source_root}")

        compiled = re.compile(pattern, re.IGNORECASE)
        matches = []

        for source_file in source_root.rglob(file_filter):
            try:
                content = source_file.read_text(encoding="utf-8")
                for i, line in enumerate(content.splitlines(), 1):
                    if compiled.search(line):
                        rel = source_file.relative_to(source_root)
                        matches.append(f"{rel}:{i}: {line.strip()}")
            except Exception:
                continue

        if not matches:
            return ToolResult(success=True, output=f"No matches for pattern '{pattern}'")

        output = f"Found {len(matches)} matches:\n" + "\n".join(matches[:50])
        if len(matches) > 50:
            output += f"\n... and {len(matches) - 50} more"
        return ToolResult(success=True, output=output, data=matches)

    def _grep_translated(self, pattern: str, class_filter: str = None) -> ToolResult:
        """Search across all translated Python code fragments."""
        compiled = re.compile(pattern, re.IGNORECASE)
        matches = []

        for frag_id, entry in self.index.fragments.items():
            if class_filter and entry.class_name != class_filter:
                continue
            if entry.target_code and compiled.search(entry.target_code):
                # Find matching lines
                for i, line in enumerate(entry.target_code.splitlines(), 1):
                    if compiled.search(line):
                        matches.append(f"[{frag_id}]:{i}: {line.strip()}")

        if not matches:
            return ToolResult(success=True, output=f"No matches for pattern '{pattern}'")

        output = f"Found {len(matches)} matches:\n" + "\n".join(matches[:50])
        if len(matches) > 50:
            output += f"\n... and {len(matches) - 50} more"
        return ToolResult(success=True, output=output, data=matches)

    def _read_source_file(self, filename: str) -> ToolResult:
        """Read a source file."""
        source_root = Path(self.config.project.source.root)

        # Try direct match first
        target = source_root / filename
        if not target.exists():
            # Search recursively
            matches = list(source_root.rglob(filename))
            if matches:
                target = matches[0]
            else:
                return ToolResult(success=False, output=f"File not found: {filename}")

        try:
            content = target.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content, data={"path": str(target)})
        except Exception as e:
            return ToolResult(success=False, output=f"Error reading {filename}: {e}")

    def _read_fragment(self, fragment_id: str) -> ToolResult:
        """Read a specific translated fragment."""
        entry = self.index.fragments.get(fragment_id)
        if not entry:
            # Try fuzzy match
            candidates = [
                fid for fid in self.index.fragments
                if fragment_id.lower() in fid.lower()
            ]
            if candidates:
                return ToolResult(
                    success=False,
                    output=f"Fragment '{fragment_id}' not found. Did you mean: {', '.join(candidates[:5])}?",
                )
            return ToolResult(success=False, output=f"Fragment '{fragment_id}' not found")

        parts = [
            f"Fragment: {entry.fragment_id}",
            f"Status: {entry.status}",
            f"Mocked: {entry.is_mocked}",
            f"Python name: {entry.python_name}",
            f"Signature: {entry.signature or 'N/A'}",
        ]
        if entry.java_source:
            parts.append(f"\n--- Java Source ---\n{entry.java_source}")
        if entry.target_code:
            parts.append(f"\n--- Translated Python ---\n{entry.target_code}")

        return ToolResult(success=True, output="\n".join(parts), data=entry)

    def _lookup_registry(
        self,
        java_name: str = None,
        python_name: str = None,
        class_name: str = None,
    ) -> ToolResult:
        """Look up name mappings in the index."""
        results = []

        if class_name:
            for frag_id, entry in self.index.fragments.items():
                if entry.class_name == class_name:
                    results.append(
                        f"{entry.java_name} -> {entry.python_name} "
                        f"[{entry.symbol_type}] ({entry.status})"
                        + (f" sig: {entry.signature}" if entry.signature else "")
                    )

        if java_name:
            for source_q, target_name in self.index.name_map.items():
                if java_name.lower() in source_q.lower():
                    results.append(f"{source_q} -> {target_name}")

        if python_name:
            for source_q, target_name in self.index.name_map.items():
                if python_name.lower() in target_name.lower():
                    results.append(f"{source_q} -> {target_name}")

        if not results:
            return ToolResult(success=True, output="No matching registry entries found")

        return ToolResult(
            success=True,
            output=f"Found {len(results)} entries:\n" + "\n".join(results),
            data=results,
        )

    def _list_class_members(self, class_name: str) -> ToolResult:
        """List all members of a class with their status."""
        summary = self.index.classes.get(class_name)
        if not summary:
            available = ", ".join(sorted(self.index.classes.keys()))
            return ToolResult(
                success=False,
                output=f"Class '{class_name}' not found. Available: {available}",
            )

        lines = [
            f"Class: {summary.name} ({summary.symbol_type})",
            f"Source: {summary.java_source_file}",
            f"Translated: {summary.translated_count}, Mocked: {summary.mocked_count}",
            f"Dependencies: {', '.join(summary.dependencies) or 'none'}",
            "",
            "Members:",
        ]

        for member in summary.members:
            status_marker = "M" if member.is_mocked else "T"
            name = member.member_name or "(class-level)"
            sig = member.signature or ""
            lines.append(f"  [{status_marker}] {name} ({member.symbol_type}) {sig}")

        return ToolResult(
            success=True, output="\n".join(lines), data=summary
        )

    def _get_class_hierarchy(self, class_name: str = None) -> ToolResult:
        """Get class hierarchy info."""
        if class_name:
            bases = self.index.hierarchy.get(class_name, [])
            if not bases:
                return ToolResult(
                    success=True,
                    output=f"{class_name}: no known base classes/interfaces",
                )
            return ToolResult(
                success=True,
                output=f"{class_name} extends/implements: {', '.join(bases)}",
                data=bases,
            )

        # Full hierarchy
        lines = []
        for cls, bases in sorted(self.index.hierarchy.items()):
            if bases:
                lines.append(f"{cls} -> {', '.join(bases)}")
            else:
                lines.append(f"{cls} (root)")

        if not lines:
            return ToolResult(success=True, output="No hierarchy information available")

        return ToolResult(
            success=True,
            output="Class Hierarchy:\n" + "\n".join(lines),
            data=self.index.hierarchy,
        )

    def _run_script(self, code: str, timeout: int = 30) -> ToolResult:
        """Execute a Python script and return output."""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.output_dir),
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"stdout:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"stderr:\n{result.stderr}")
            output_parts.append(f"return_code: {result.returncode}")

            return ToolResult(
                success=result.returncode == 0,
                output="\n".join(output_parts) or "No output",
                data={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=f"Script timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output=f"Script execution error: {e}")

    def _write_file(self, path: str, content: str) -> ToolResult:
        """Write a file in the output directory."""
        full_path = self.output_dir / path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Written {len(content)} bytes to {full_path}",
                data={"path": str(full_path)},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Write error: {e}")

    def _read_file(self, path: str) -> ToolResult:
        """Read a file from the project."""
        # Try output dir first, then project root
        candidates = [
            self.output_dir / path,
            Path(self.config.project.source.root) / path,
            Path(path),
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8")
                    return ToolResult(
                        success=True, output=content, data={"path": str(candidate)}
                    )
                except Exception as e:
                    return ToolResult(success=False, output=f"Error reading {candidate}: {e}")

        return ToolResult(success=False, output=f"File not found: {path}")

    def _rag_search(self, query: str, category: str = None, top_k: int = 3) -> ToolResult:
        """Search the RAG knowledge base."""
        # Try to use vector store if available
        if self._vector_store is None:
            try:
                from codemorph.knowledge.vector_store import VectorStore
                storage_dir = Path(self.config.rag.chroma_persist_dir)
                if storage_dir.exists():
                    self._vector_store = VectorStore(storage_dir)
            except Exception:
                self._vector_store = False  # Mark as unavailable

        if self._vector_store and self._vector_store is not False:
            results = self._vector_store.query(
                query_code=query,
                language=self.index.target_lang,
                category=category,
                top_k=top_k,
            )
            if results:
                lines = []
                for example, score in results:
                    lines.append(
                        f"[{score:.2f}] {example.description}\n"
                        f"Category: {example.category}\n"
                        f"```\n{example.code}\n```\n"
                    )
                return ToolResult(
                    success=True,
                    output=f"Found {len(results)} RAG results:\n" + "\n".join(lines),
                    data=results,
                )

        # Fallback: keyword search over translated fragments
        query_lower = query.lower()
        matches = []
        for frag_id, entry in self.index.fragments.items():
            score = 0
            if entry.target_code:
                for word in query_lower.split():
                    if word in entry.target_code.lower():
                        score += 1
            if entry.java_source:
                for word in query_lower.split():
                    if word in entry.java_source.lower():
                        score += 1
            if score > 0:
                matches.append((frag_id, entry, score))

        matches.sort(key=lambda x: x[2], reverse=True)
        matches = matches[:top_k]

        if not matches:
            return ToolResult(success=True, output="No matching patterns found")

        lines = []
        for frag_id, entry, score in matches:
            code_preview = (entry.target_code or "")[:200]
            lines.append(f"[{score}] {frag_id}\n{code_preview}...")

        return ToolResult(
            success=True,
            output=f"Found {len(matches)} keyword matches:\n" + "\n".join(lines),
            data=matches,
        )

    def _compile_check(self, file_path: str) -> ToolResult:
        """Check if a Python file compiles."""
        # Resolve path
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = self.output_dir / file_path

        if not full_path.exists():
            return ToolResult(success=False, output=f"File not found: {full_path}")

        try:
            source = full_path.read_text(encoding="utf-8")
            compile(source, str(full_path), "exec")
            return ToolResult(success=True, output=f"{full_path.name}: OK — compiles cleanly")
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output=f"{full_path.name}: SYNTAX ERROR at line {e.lineno}: {e.msg}",
                data={"line": e.lineno, "msg": e.msg, "text": e.text},
            )

    def _get_project_summary(self) -> ToolResult:
        """Get a compact project summary."""
        lines = [
            f"Project: {self.index.source_lang} -> {self.index.target_lang}",
            f"Classes: {len(self.index.classes)}",
            f"Total fragments: {len(self.index.fragments)}",
            f"Total translated: {sum(c.translated_count for c in self.index.classes.values())}",
            f"Total mocked: {sum(c.mocked_count for c in self.index.classes.values())}",
            "",
            "Classes:",
        ]

        for name in sorted(self.index.classes.keys()):
            cs = self.index.classes[name]
            bases = self.index.hierarchy.get(name, [])
            base_str = f" extends {', '.join(bases)}" if bases else ""
            lines.append(
                f"  {name} ({cs.symbol_type}){base_str} "
                f"— {len(cs.members)} members, "
                f"{cs.translated_count} translated, {cs.mocked_count} mocked"
            )

        return ToolResult(success=True, output="\n".join(lines), data=None)


# =============================================================================
# Agent loop utilities
# =============================================================================


def parse_tool_calls(response: str) -> list[dict[str, Any]]:
    """Parse tool calls from an LLM response.

    Looks for JSON objects with "tool" and "args" keys.
    Supports multiple tool calls in a single response, including
    nested JSON objects within the "args" field.
    """
    calls = []

    # Find JSON objects by brace matching
    i = 0
    while i < len(response):
        if response[i] == "{":
            # Try to extract a balanced JSON object
            depth = 0
            in_string = False
            escape = False
            start = i
            for j in range(i, len(response)):
                ch = response[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = response[start : j + 1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict) and "tool" in data:
                                calls.append({
                                    "name": data["tool"],
                                    "args": data.get("args", {}),
                                })
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    return calls


def extract_final_answer(response: str) -> str:
    """Extract the final answer from an LLM response (text without tool calls).

    Removes any JSON tool-call blocks (balanced braces containing a "tool" key).
    """
    # Identify spans to remove by finding tool-call JSON objects
    spans_to_remove: list[tuple[int, int]] = []
    i = 0
    while i < len(response):
        if response[i] == "{":
            depth = 0
            in_string = False
            escape = False
            start = i
            for j in range(i, len(response)):
                ch = response[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = response[start : j + 1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict) and "tool" in data:
                                spans_to_remove.append((start, j + 1))
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    if not spans_to_remove:
        return response

    # Build result skipping tool-call spans
    parts: list[str] = []
    prev = 0
    for s, e in spans_to_remove:
        parts.append(response[prev:s])
        prev = e
    parts.append(response[prev:])

    cleaned = "".join(parts).strip()
    return cleaned or response


def run_agent_loop(
    llm_client: Any,
    system_prompt: str,
    tools: AgentToolRegistry,
    max_turns: int = 20,
    initial_message: str | None = None,
    verbose: bool = True,
) -> str:
    """Run an agentic loop where the LLM can call tools.

    Args:
        llm_client: LLM client with a _call_llm or generate method
        system_prompt: System prompt including tool descriptions
        tools: Tool registry
        max_turns: Maximum number of turns
        initial_message: Optional initial user message
        verbose: Whether to log tool calls

    Returns:
        The LLM's final answer
    """
    from codemorph.translator.llm_client import LLMConversation

    conversation = LLMConversation()
    full_system = system_prompt + "\n\n" + tools.get_tool_descriptions()
    full_system += (
        "\n\nWhen you are done and have your final answer, respond with plain text "
        "(no tool calls). Include your final answer clearly."
    )
    conversation.add_message("system", full_system)

    if initial_message:
        conversation.add_message("user", initial_message)

    for turn in range(max_turns):
        # Call LLM — always use _call_llm for proper multi-turn conversation
        response = llm_client._call_llm(conversation.messages)

        conversation.add_message("assistant", response)

        # Parse tool calls
        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            # No tool calls — LLM is done
            return extract_final_answer(response)

        # Execute tools and feed results back
        tool_results = []
        for call in tool_calls:
            result = tools.execute(call["name"], **call["args"])
            result_text = f"[{call['name']}]: {result.output}"
            tool_results.append(result_text)
            if verbose:
                preview = result.output[:150].replace("\n", " ")
                logger.info(f"  Tool: {call['name']}({call['args']}) -> {preview}...")

        conversation.add_message("user", "\n\n".join(tool_results))

    # Exhausted turns — return last response
    return extract_final_answer(conversation.messages[-1]["content"])
