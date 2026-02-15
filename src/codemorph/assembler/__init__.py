"""
Phase 4: Dynamic Project Assembly Agent.

LLM-driven assembler that reads translated fragments and symbol mappings,
structures a target project, drafts classes, injects translated code,
fills gaps, and iterates compile-fix loops.
"""

from codemorph.assembler.index_builder import (
    ClassSummary,
    FragmentEntry,
    IndexBuilder,
    ProjectIndex,
)
from codemorph.assembler.orchestrator import Phase4Orchestrator

__all__ = [
    "ClassSummary",
    "FragmentEntry",
    "IndexBuilder",
    "Phase4Orchestrator",
    "ProjectIndex",
]
