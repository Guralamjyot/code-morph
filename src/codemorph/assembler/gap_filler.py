"""
Gap Filler for Phase 4 Assembly.

LLM-driven agent that fills in mocked (untranslated) fragments using tools
to understand the Java source, find similar translated patterns, and write
Python translations that fit the assembled class context.
"""

import logging
import re
from typing import Any

from codemorph.assembler.index_builder import FragmentEntry, ProjectIndex
from codemorph.assembler.tools import AgentToolRegistry, run_agent_loop
from codemorph.translator.llm_client import strip_markdown_code_blocks

logger = logging.getLogger(__name__)

GAP_FILLER_SYSTEM_PROMPT = """\
You are filling in a mocked (untranslated) Java method for an assembled Python class.

Use tools to understand the Java source, find similar translated patterns, and write
the Python translation. Suggested workflow:
1. Call read_fragment("{fragment_id}") to see the Java source and current mock
2. Call rag_search or grep_translated to find similar patterns already translated
3. Call read_file to see the current assembled class for context
4. Write the Python translation
5. Call run_script to validate it parses: import ast; ast.parse('''...''')

CRITICAL RULES:
1. Output ONLY the Python method/function body that replaces the mock.
2. Match the style and conventions of the already-translated methods in the class.
3. Use Python naming conventions (snake_case for methods).
4. Include proper type hints matching the rest of the class.
5. Do NOT include the class wrapper — just the method def.
6. Ensure the code is syntactically valid Python.

When done, output the final Python code for this method. No JSON, no tool calls,
just the raw Python code.
"""


class GapFiller:
    """LLM-driven agent that fills mocked fragments."""

    def __init__(
        self,
        llm_client: Any,
        index: "ProjectIndex | None" = None,
        verbose: bool = True,
    ):
        self.llm_client = llm_client
        self.index = index
        self.verbose = verbose

    def fill(
        self,
        fragment: FragmentEntry,
        assembled_classes: dict[str, str],
        tools: AgentToolRegistry,
        max_attempts: int = 5,
    ) -> str | None:
        """Fill a single mocked fragment.

        Args:
            fragment: The mocked fragment to fill
            assembled_classes: Map of class_name -> assembled code (for context)
            tools: Tool registry
            max_attempts: Max retry attempts

        Returns:
            The translated Python code, or None if all attempts fail
        """
        for attempt in range(max_attempts):
            logger.info(
                f"Filling {fragment.fragment_id} (attempt {attempt + 1}/{max_attempts})"
            )

            try:
                prompt = GAP_FILLER_SYSTEM_PROMPT.replace(
                    "{fragment_id}", fragment.fragment_id
                )

                # Provide context about the class
                context_parts = [
                    f"Fill the mocked fragment: {fragment.fragment_id}",
                    f"Class: {fragment.class_name}",
                    f"Member: {fragment.member_name or '(class-level)'}",
                    f"Type: {fragment.symbol_type}",
                ]

                if fragment.signature:
                    context_parts.append(f"Expected signature: {fragment.signature}")

                # Enriched context for class-level fragments
                is_class_level = fragment.member_name is None
                if is_class_level and self.index:
                    context_parts.append(
                        "\nThis is a CLASS-LEVEL fragment. Generate __init__, "
                        "class fields, and any abstract/concrete methods."
                    )
                    # Include full Java source if available
                    if fragment.java_source:
                        java_src = fragment.java_source
                        if len(java_src) > 3000:
                            java_src = java_src[:3000] + "\n// ... (truncated)"
                        context_parts.append(
                            f"\nFull Java source of the class:\n```java\n{java_src}\n```"
                        )
                    # Find and include subclass code for context
                    subclass_parts = self._get_subclass_context(
                        fragment.class_name, assembled_classes
                    )
                    if subclass_parts:
                        context_parts.append(
                            f"\nSubclass code (for understanding expectations):\n{subclass_parts}"
                        )
                    # Add generalized constructor/method translation rules
                    context_parts.append(
                        "\nGENERALIZED RULES:"
                        "\n- Java constructors → def __init__(self, ...): with super().__init__(...) for subclasses"
                        "\n- Java toString() → __str__(), equals() → __eq__(), hashCode() → __hash__()"
                        "\n- Java compareTo() → __lt__()/__eq__() or implement @functools.total_ordering"
                        "\n- Java static methods → @staticmethod"
                        "\n- Do NOT shadow Python builtins (iter, list, type, etc.) as variable names"
                    )

                # Include the current assembled class as context (truncated if large)
                class_code = assembled_classes.get(fragment.class_name, "")
                if class_code:
                    if len(class_code) > 4000:
                        class_code = class_code[:4000] + "\n# ... (truncated)"
                    context_parts.append(f"\nCurrent assembled class:\n```python\n{class_code}\n```")

                initial_message = "\n".join(context_parts)

                response = run_agent_loop(
                    llm_client=self.llm_client,
                    system_prompt=prompt,
                    tools=tools,
                    max_turns=10,
                    initial_message=initial_message,
                    verbose=self.verbose,
                )

                code = strip_markdown_code_blocks(response)

                # Validate the code parses
                if self._validate_code(code):
                    logger.info(f"Successfully filled {fragment.fragment_id}")
                    return code

                logger.warning(
                    f"Attempt {attempt + 1} for {fragment.fragment_id} produced invalid code"
                )
            except Exception as e:
                logger.warning(
                    f"LLM gap filling failed for {fragment.fragment_id} "
                    f"(attempt {attempt + 1}): {e}"
                )
                break  # Don't retry on connection errors

        logger.error(f"Failed to fill {fragment.fragment_id} after attempts")
        return None

    def fill_all(
        self,
        index: ProjectIndex,
        assembled_classes: dict[str, str],
        tools: AgentToolRegistry,
    ) -> dict[str, str]:
        """Fill all mocked fragments.

        Args:
            index: Project index
            assembled_classes: Map of class_name -> assembled code
            tools: Tool registry

        Returns:
            Map of fragment_id -> translated code for successfully filled fragments
        """
        mocked = [
            entry for entry in index.fragments.values() if entry.is_mocked
        ]

        if not mocked:
            logger.info("No mocked fragments to fill")
            return {}

        logger.info(f"Filling {len(mocked)} mocked fragments...")
        filled: dict[str, str] = {}

        for fragment in mocked:
            code = self.fill(fragment, assembled_classes, tools)
            if code:
                filled[fragment.fragment_id] = code

        logger.info(f"Filled {len(filled)}/{len(mocked)} mocked fragments")
        return filled

    def apply_fills(
        self,
        assembled_classes: dict[str, str],
        fills: dict[str, str],
    ) -> dict[str, str]:
        """Apply filled translations to assembled classes.

        Replaces # MOCKED: fragment_id + NotImplementedError with actual code.

        Args:
            assembled_classes: Map of class_name -> assembled code
            fills: Map of fragment_id -> translated code

        Returns:
            Updated assembled_classes with fills applied
        """
        updated = {}

        for class_name, code in assembled_classes.items():
            for frag_id, fill_code in fills.items():
                # Replace the mocked marker + NotImplementedError
                pattern = (
                    r"^(\s*)#\s*MOCKED:\s*"
                    + re.escape(frag_id)
                    + r"\s*\n\s*raise NotImplementedError\([^)]*\)"
                )
                replacement = self._indent_code(fill_code, r"\1")
                code = re.sub(pattern, replacement, code, flags=re.MULTILINE)

                # Also handle bare MOCKED markers without NotImplementedError
                pattern_bare = (
                    r"^(\s*)#\s*MOCKED:\s*" + re.escape(frag_id) + r"\s*$"
                )
                code = re.sub(
                    pattern_bare, replacement, code, flags=re.MULTILINE
                )

            updated[class_name] = code

        return updated

    def _validate_code(self, code: str) -> bool:
        """Check if code is valid Python."""
        import ast

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            # Try wrapping in a class to see if it's a method
            try:
                ast.parse(f"class _Wrapper:\n{self._indent(code, '    ')}")
                return True
            except SyntaxError:
                return False

    def _indent(self, code: str, indent: str) -> str:
        """Add indentation to all lines of code."""
        return "\n".join(
            indent + line if line.strip() else line
            for line in code.splitlines()
        )

    def _get_subclass_context(
        self,
        class_name: str,
        assembled_classes: dict[str, str],
    ) -> str:
        """Find subclasses of the given class and return their assembled code.

        Uses the index hierarchy to find classes that extend/implement this one.
        """
        if not self.index:
            return ""

        subclasses = []
        for child_name, parents in self.index.hierarchy.items():
            if class_name in parents and child_name in assembled_classes:
                subclasses.append(child_name)

        if not subclasses:
            return ""

        parts = []
        total_len = 0
        for sub in subclasses[:3]:  # Limit to 3 subclasses
            sub_code = assembled_classes[sub]
            if total_len + len(sub_code) > 3000:
                sub_code = sub_code[:1500] + "\n# ... (truncated)"
            parts.append(f"# --- {sub} ---\n```python\n{sub_code}\n```")
            total_len += len(sub_code)

        return "\n".join(parts)

    def _indent_code(self, code: str, indent_ref: str) -> str:
        """Create replacement text with proper indentation."""
        import textwrap
        dedented = textwrap.dedent(code).strip()
        lines = dedented.splitlines()
        # The first line gets the captured indent, subsequent lines get it too
        return "\n".join(lines)
