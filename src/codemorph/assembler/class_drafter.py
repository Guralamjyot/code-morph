"""
Class Drafter for Phase 4 Assembly.

LLM-driven agent that drafts one class at a time, producing a skeleton
with # INJECT: fragment_id markers for each translated method.
"""

import logging
from typing import Any

from codemorph.assembler.index_builder import ProjectIndex
from codemorph.assembler.tools import AgentToolRegistry, run_agent_loop

logger = logging.getLogger(__name__)

DRAFTER_SYSTEM_PROMPT = """\
You are drafting a Python class skeleton for a Java-to-Python translation project.

You have tools to explore the source and translated code. Start by calling
list_class_members("{class_name}") to see all available members, then use
read_fragment, get_class_hierarchy, grep_source, or read_source_file as needed.

CRITICAL RULES:
1. Output a complete Python class skeleton with proper imports, TypeVars, and class declaration.
2. For each translated method, include ONLY a comment marker: # INJECT: fragment_id
   The system will inject the actual translated code automatically.
3. For mocked (untranslated) methods, use: # MOCKED: fragment_id
4. The class-level fragment (constructor, class body) uses: # INJECT: ClassName::ClassName
5. Include proper Python imports at the top of the file.
6. Use Python conventions: snake_case methods, type hints, proper inheritance.
7. Add TypeVars if the class uses generics.
8. Keep method signatures matching the translated code's signatures from the registry.
9. Do NOT include method bodies — only the # INJECT or # MOCKED marker as the body.

When ready, output the complete Python file content. Do NOT wrap it in a JSON object.
Just output the raw Python code.
"""


class ClassDrafter:
    """LLM-driven agent that drafts one class at a time."""

    def __init__(self, llm_client: Any, verbose: bool = True):
        self.llm_client = llm_client
        self.verbose = verbose

    def draft(
        self,
        class_name: str,
        index: ProjectIndex,
        tools: AgentToolRegistry,
        module_context: dict[str, str] | None = None,
    ) -> str:
        """Draft a class skeleton with INJECT markers.

        Args:
            class_name: The class to draft
            index: Project index
            tools: Tool registry
            module_context: Map of class_name -> module_name for imports

        Returns:
            Python source code with # INJECT markers
        """
        prompt = DRAFTER_SYSTEM_PROMPT.replace("{class_name}", class_name)

        # Build additional context about what module this class is in
        context_parts = [f"Draft the Python class: {class_name}"]
        if module_context:
            context_parts.append(f"\nModule layout for imports:")
            for cls, mod in module_context.items():
                context_parts.append(f"  {cls} -> {mod}")

        initial_message = "\n".join(context_parts)

        logger.info(f"Drafting class: {class_name}")

        try:
            response = run_agent_loop(
                llm_client=self.llm_client,
                system_prompt=prompt,
                tools=tools,
                max_turns=15,
                initial_message=initial_message,
                verbose=self.verbose,
            )
            skeleton = self._clean_skeleton(response, class_name, index)
        except Exception as e:
            logger.warning(f"LLM drafting failed for {class_name} ({e}), using fallback")
            skeleton = self._generate_fallback_skeleton(class_name, index)

        logger.info(f"Drafted {class_name}: {skeleton.count('# INJECT')} inject markers, "
                     f"{skeleton.count('# MOCKED')} mocked markers")
        return skeleton

    def _clean_skeleton(self, response: str, class_name: str, index: ProjectIndex) -> str:
        """Clean and validate the LLM's skeleton output."""
        import re
        from codemorph.translator.llm_client import strip_markdown_code_blocks

        code = strip_markdown_code_blocks(response)

        # Check for INJECT/MOCKED markers with *real* fragment IDs
        # Real fragment IDs look like "ClassName::ClassName.method"
        real_markers = re.findall(
            r"#\s*(?:INJECT|MOCKED):\s*(\S+)", code
        )
        valid_markers = [
            m for m in real_markers
            if m in index.fragments
        ]

        # Use fallback if the LLM missed more than half of the class's fragments
        cls_summary = index.classes.get(class_name)
        expected_count = len(cls_summary.members) if cls_summary else 0
        if len(valid_markers) < max(2, expected_count // 2):
            logger.warning(
                f"Only {len(valid_markers)}/{expected_count} valid markers in "
                f"{class_name} skeleton, generating from index"
            )
            code = self._generate_fallback_skeleton(class_name, index)

        return code

    def _generate_fallback_skeleton(self, class_name: str, index: ProjectIndex) -> str:
        """Generate a fallback skeleton when the LLM doesn't produce markers."""
        summary = index.classes.get(class_name)
        if not summary:
            return f"# Could not generate skeleton for {class_name}\n"

        lines = [
            "from __future__ import annotations",
            "",
        ]

        # Add base class imports from hierarchy
        bases = index.hierarchy.get(class_name, [])
        if bases:
            lines.append(f"# Base classes: {', '.join(bases)}")
            lines.append("")

        # Class declaration
        base_str = ", ".join(bases) if bases else ""
        if base_str:
            lines.append(f"class {class_name}({base_str}):")
        else:
            lines.append(f"class {class_name}:")

        # Add members — each marker at class body indent (4 spaces).
        # The injector replaces each marker line with the full translated
        # code block (method def, constant assignment, or class body).
        has_members = False
        for member in summary.members:
            has_members = True
            marker_type = "MOCKED" if member.is_mocked else "INJECT"
            lines.append(f"    # {marker_type}: {member.fragment_id}")
            lines.append("")

        if not has_members:
            lines.append("    pass")

        lines.append("")
        return "\n".join(lines)
