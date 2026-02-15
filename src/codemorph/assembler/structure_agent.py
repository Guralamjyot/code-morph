"""
Structure Agent for Phase 4 Assembly.

LLM-driven agent that reads the source project and decides the target
Python package structure. Operates in an agentic tool-use loop.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from codemorph.assembler.index_builder import ProjectIndex
from codemorph.assembler.tools import AgentToolRegistry, run_agent_loop

logger = logging.getLogger(__name__)

STRUCTURE_SYSTEM_PROMPT = """\
You are a Python project architect. Your job is to design the target Python package
structure for a code translation project.

You have tools to explore the source project. Start by calling get_project_summary()
to see what's available, then use grep_source, read_source_file, get_class_hierarchy,
and rag_search as needed to understand the project structure.

IMPORTANT RULES:
1. Group related classes into modules. Interfaces can share a module if small.
2. Use snake_case for module names (Python convention).
3. Create an __init__.py that exports all public classes.
4. Consider the class hierarchy when grouping — base classes before subclasses.
5. Keep it simple — one class per module unless classes are tightly coupled.

When you have decided on the structure, output ONLY a JSON block (no tool calls)
with this exact format:

```json
{
  "package_name": "the_package_name",
  "modules": {
    "module_name": ["ClassName1", "ClassName2"],
    ...
  },
  "init_exports": ["ClassName1", "ClassName2", ...],
  "notes": "Brief explanation of your choices"
}
```
"""


@dataclass
class ProjectStructure:
    """The decided project structure."""

    package_name: str
    modules: dict[str, list[str]]
    init_exports: list[str]
    notes: str = ""

    def get_module_for_class(self, class_name: str) -> str | None:
        """Find which module a class belongs to."""
        for module_name, classes in self.modules.items():
            if class_name in classes:
                return module_name
        return None


class StructureAgent:
    """LLM-driven agent that decides the target project structure."""

    def __init__(self, llm_client: Any, verbose: bool = True):
        self.llm_client = llm_client
        self.verbose = verbose

    def analyze(
        self,
        index: ProjectIndex,
        tools: AgentToolRegistry,
        project_name: str | None = None,
    ) -> ProjectStructure:
        """Run the structure agent to determine project layout.

        Args:
            index: The project index
            tools: Tool registry for the agent
            project_name: Optional project name override

        Returns:
            ProjectStructure with the decided layout
        """
        logger.info("Structure agent: analyzing project...")

        try:
            response = run_agent_loop(
                llm_client=self.llm_client,
                system_prompt=STRUCTURE_SYSTEM_PROMPT,
                tools=tools,
                max_turns=15,
                initial_message="Analyze the project and decide the Python package structure.",
                verbose=self.verbose,
            )
            structure = self._parse_structure(response, index)
        except Exception as e:
            logger.warning(f"LLM structure agent failed ({e}), using fallback")
            structure = self._fallback_structure(index)

        # Override package name if provided
        if project_name:
            structure.package_name = project_name

        logger.info(f"Structure agent decided: package={structure.package_name}, "
                     f"modules={list(structure.modules.keys())}")
        return structure

    def _parse_structure(self, response: str, index: ProjectIndex) -> ProjectStructure:
        """Parse the LLM's structure decision from its response."""
        # Try to extract JSON from the response
        json_data = self._extract_json(response)

        if json_data:
            return ProjectStructure(
                package_name=json_data.get("package_name", "translated"),
                modules=json_data.get("modules", {}),
                init_exports=json_data.get("init_exports", []),
                notes=json_data.get("notes", ""),
            )

        # Fallback: generate a reasonable default structure
        logger.warning("Could not parse LLM structure response, using fallback")
        return self._fallback_structure(index)

    def _extract_json(self, text: str) -> dict | None:
        """Extract a JSON object from text (may be wrapped in markdown)."""
        import re

        # Try to find JSON in code blocks
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        brace_start = text.find("{")
        if brace_start >= 0:
            # Find matching closing brace
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start : i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    def _fallback_structure(self, index: ProjectIndex) -> ProjectStructure:
        """Generate a reasonable default structure from the index."""
        modules: dict[str, list[str]] = {}
        init_exports: list[str] = []

        # Separate interfaces from classes
        interfaces = []
        classes = []

        for name, summary in index.classes.items():
            if summary.symbol_type == "interface":
                interfaces.append(name)
            else:
                classes.append(name)

        # Group interfaces into one module
        if interfaces:
            modules["interfaces"] = sorted(interfaces)

        # One class per module
        for class_name in sorted(classes):
            module_name = self._class_to_module_name(class_name)
            modules[module_name] = [class_name]

        init_exports = sorted(classes + interfaces)

        # Derive package name from project name or first class
        package_name = (
            index.source_lang + "_translated"
            if not classes
            else self._class_to_module_name(classes[0]).split("_")[0]
        )

        return ProjectStructure(
            package_name=package_name,
            modules=modules,
            init_exports=init_exports,
            notes="Auto-generated fallback structure",
        )

    def _class_to_module_name(self, class_name: str) -> str:
        """Convert a CamelCase class name to a snake_case module name."""
        import re
        # Insert underscore before uppercase letters
        s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", class_name)
        return s.lower()
