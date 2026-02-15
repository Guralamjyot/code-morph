"""
Index Builder for Phase 4 Assembly.

Reads latest.json + symbol_registry.json and builds a compact, queryable
ProjectIndex. This is pure data loading — no LLM, no decisions.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codemorph.config.models import TranslatedFragment, TranslationStatus
from codemorph.state.persistence import TranslationState
from codemorph.state.symbol_registry import SymbolRegistry

logger = logging.getLogger(__name__)


@dataclass
class FragmentEntry:
    """A single indexed fragment with all metadata needed for assembly."""

    fragment_id: str
    class_name: str
    member_name: str | None
    symbol_type: str
    status: str
    is_mocked: bool
    java_name: str
    python_name: str
    signature: str | None
    target_code: str | None
    java_source: str | None


@dataclass
class ClassSummary:
    """Summary of a class with all its members."""

    name: str
    java_source_file: str
    symbol_type: str
    members: list[FragmentEntry] = field(default_factory=list)
    translated_count: int = 0
    mocked_count: int = 0
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ProjectIndex:
    """The complete project index built from state files."""

    classes: dict[str, ClassSummary] = field(default_factory=dict)
    fragments: dict[str, FragmentEntry] = field(default_factory=dict)
    name_map: dict[str, str] = field(default_factory=dict)
    hierarchy: dict[str, list[str]] = field(default_factory=dict)
    overloads: dict[str, list[str]] = field(default_factory=dict)
    source_lang: str = "java"
    target_lang: str = "python"

    def get_dependency_order(self) -> list[str]:
        """Return class names in dependency order (topological sort)."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited or name not in self.classes:
                return
            visited.add(name)
            for base in self.hierarchy.get(name, []):
                visit(base)
            for dep in self.classes[name].dependencies:
                visit(dep)
            order.append(name)

        for class_name in self.classes:
            visit(class_name)

        return order


class IndexBuilder:
    """Builds a ProjectIndex from state files and source directory."""

    def __init__(self, state_dir: Path, source_dir: Path | None = None):
        self.state_dir = Path(state_dir)
        self.source_dir = Path(source_dir) if source_dir else None

    def build(self) -> ProjectIndex:
        """Build the complete project index."""
        index = ProjectIndex()

        # Step 1: Load translation state (latest.json)
        state = self._load_state()
        if state is None:
            logger.warning("No translation state found")
            return index

        # Detect languages from config
        index.source_lang = state.config.project.source.language.value
        index.target_lang = state.config.project.target.language.value

        # Step 2: Load symbol registry
        registry = self._load_registry()

        # Step 3: Build fragment entries from translated_fragments
        for frag_id, translated_frag in state.translated_fragments.items():
            entry = self._build_fragment_entry(frag_id, translated_frag, registry)
            index.fragments[frag_id] = entry

        # Step 4: Group fragments into classes
        # Track seen overload bases so we skip secondary variants whose
        # target_code is identical to the primary (merged overloads).
        seen_overload_bases: dict[str, str] = {}  # base_id -> first frag_id added
        for frag_id, entry in index.fragments.items():
            base = frag_id[: frag_id.index("$")] if "$" in frag_id else frag_id
            if base in seen_overload_bases:
                first_frag_id = seen_overload_bases[base]
                first = index.fragments.get(first_frag_id)
                if (
                    first
                    and first.target_code
                    and entry.target_code == first.target_code
                ):
                    # Same merged code already added via primary — skip
                    continue
            seen_overload_bases.setdefault(base, frag_id)

            class_name = entry.class_name
            if class_name not in index.classes:
                index.classes[class_name] = ClassSummary(
                    name=class_name,
                    java_source_file=self._find_source_file(class_name, state),
                    symbol_type=self._infer_class_type(class_name, index.fragments),
                )
            summary = index.classes[class_name]
            summary.members.append(entry)
            if entry.is_mocked:
                summary.mocked_count += 1
            else:
                summary.translated_count += 1

        # Step 4b: Detect overloaded methods (same base ID, multiple variants)
        base_id_groups: dict[str, list[str]] = {}
        for frag_id in index.fragments:
            if "$" in frag_id:
                base_id = frag_id[:frag_id.index("$")]
            else:
                base_id = frag_id
            if base_id not in base_id_groups:
                base_id_groups[base_id] = []
            base_id_groups[base_id].append(frag_id)

        for base_id, variants in base_id_groups.items():
            if len(variants) > 1:
                index.overloads[base_id] = sorted(variants)

        # Step 5: Build name map from registry
        if registry:
            for source_q, mapping in registry.mappings.items():
                index.name_map[source_q] = mapping.target_name

        # Step 6: Extract hierarchy from Java source files
        if self.source_dir and self.source_dir.exists():
            index.hierarchy = self._extract_hierarchy()

        # Step 7: Build dependency lists for classes
        self._build_class_dependencies(index)

        logger.info(
            f"Index built: {len(index.classes)} classes, "
            f"{len(index.fragments)} fragments"
        )

        return index

    def _load_state(self) -> TranslationState | None:
        """Load the latest translation state."""
        try:
            return TranslationState.load_latest(self.state_dir)
        except FileNotFoundError:
            # Try loading latest.json directly
            latest = self.state_dir / "latest.json"
            if latest.exists():
                return TranslationState.load(latest)
            return None

    def _load_registry(self) -> SymbolRegistry | None:
        """Load the symbol registry."""
        try:
            return SymbolRegistry.load(self.state_dir)
        except Exception:
            logger.warning("Could not load symbol registry")
            return None

    def _build_fragment_entry(
        self,
        frag_id: str,
        translated_frag: TranslatedFragment,
        registry: SymbolRegistry | None,
    ) -> FragmentEntry:
        """Build a FragmentEntry from a TranslatedFragment + registry data."""
        # Parse class_name and member_name from fragment_id (e.g. "Pair::Pair.fromArray")
        class_name, member_name = self._parse_fragment_id(frag_id)

        # Get registry info
        java_name = frag_id
        python_name = frag_id
        signature = None

        if registry and registry.has_symbol(frag_id):
            mapping = registry.get_mapping(frag_id)
            if mapping:
                python_name = mapping.target_name
                signature = mapping.signature
                java_name = mapping.source_name

        # Determine symbol type
        symbol_type = translated_frag.fragment.fragment_type.value

        return FragmentEntry(
            fragment_id=frag_id,
            class_name=class_name,
            member_name=member_name,
            symbol_type=symbol_type,
            status=translated_frag.status.value,
            is_mocked=(
                translated_frag.is_mocked
                or translated_frag.status == TranslationStatus.MOCKED
            ),
            java_name=java_name,
            python_name=python_name,
            signature=signature,
            target_code=translated_frag.target_code,
            java_source=translated_frag.fragment.source_code,
        )

    def _parse_fragment_id(self, frag_id: str) -> tuple[str, str | None]:
        """Parse a fragment ID into (class_name, member_name).

        Strips the overload discriminator ($TypeA_TypeB suffix) if present.

        Examples:
            "Pair::Pair" -> ("Pair", None)  # class-level
            "Pair::Pair.fromArray" -> ("Pair", "fromArray")
            "Pair::Pair.fromArray$Object" -> ("Pair", "fromArray")
            "Pair::Pair.SIZE" -> ("Pair", "SIZE")
        """
        if "::" not in frag_id:
            return frag_id, None

        parts = frag_id.split("::", 1)
        class_name = parts[0]
        rest = parts[1]

        # Strip overload discriminator ($...) if present
        if "$" in rest:
            rest = rest[:rest.index("$")]

        # Check if there's a member (after the first dot following the class name)
        # Pattern: ClassName.memberName
        if "." in rest:
            dot_idx = rest.index(".")
            member_name = rest[dot_idx + 1:]
            return class_name, member_name

        # No member - this is the class-level fragment
        return class_name, None

    def _find_source_file(self, class_name: str, state: TranslationState) -> str:
        """Find the Java source file for a class."""
        for frag_id, translated_frag in state.translated_fragments.items():
            if frag_id.startswith(f"{class_name}::"):
                return str(translated_frag.fragment.source_file)
        return f"{class_name}.java"

    def _infer_class_type(
        self, class_name: str, fragments: dict[str, FragmentEntry]
    ) -> str:
        """Infer whether this is a class or interface."""
        for frag_id, entry in fragments.items():
            if entry.class_name == class_name and entry.member_name is None:
                return entry.symbol_type
        # Check if name starts with I and has uppercase next char (interface convention)
        if (
            class_name.startswith("I")
            and len(class_name) > 1
            and class_name[1].isupper()
        ):
            return "interface"
        return "class"

    def _extract_hierarchy(self) -> dict[str, list[str]]:
        """Extract class hierarchy from Java source files."""
        hierarchy: dict[str, list[str]] = {}

        if not self.source_dir:
            return hierarchy

        for java_file in self.source_dir.rglob("*.java"):
            try:
                content = java_file.read_text(encoding="utf-8")
                # Collapse whitespace so extends/implements on separate lines
                # are treated as a single declaration
                # First, extract the class declaration up to the opening brace
                decl_match = re.search(
                    r"((?:public\s+)?(?:abstract\s+)?(?:final\s+)?"
                    r"(?:class|interface)\s+\w+.*?)\{",
                    content,
                    re.DOTALL,
                )
                if not decl_match:
                    continue
                # Normalize whitespace in the declaration
                decl = re.sub(r"\s+", " ", decl_match.group(1)).strip()

                # Now parse the normalized declaration
                pattern = (
                    r"(?:public\s+)?(?:abstract\s+)?(?:final\s+)?"
                    r"(?:class|interface)\s+(\w+)"
                    r"(?:\s*<[^>]*>)?"  # generics
                    r"(?:\s+extends\s+([\w.<>,\s]+?))?"
                    r"(?:\s+implements\s+([\w.<>,\s]+))?"
                    r"\s*$"
                )
                match = re.search(pattern, decl)
                if match:
                    class_name = match.group(1)
                    bases = []
                    if match.group(2):
                        for base in match.group(2).split(","):
                            base = base.strip()
                            base = re.sub(r"<[^>]*>", "", base).strip()
                            if base:
                                bases.append(base)
                    if match.group(3):
                        for iface in match.group(3).split(","):
                            iface = iface.strip()
                            iface = re.sub(r"<[^>]*>", "", iface).strip()
                            if iface:
                                bases.append(iface)
                    hierarchy[class_name] = bases
            except Exception as e:
                logger.debug(f"Could not parse {java_file}: {e}")

        return hierarchy

    def _build_class_dependencies(self, index: ProjectIndex) -> None:
        """Build dependency lists for each class based on hierarchy and fragments."""
        for class_name, summary in index.classes.items():
            deps = set()
            # Add hierarchy deps
            for base in index.hierarchy.get(class_name, []):
                if base in index.classes:
                    deps.add(base)
            # Add deps from fragment dependencies
            for member in summary.members:
                frag_id = member.fragment_id
                # Check translated fragment dependencies if available
                # (these are from the CodeFragment.dependencies field)
                for other_frag_id in index.fragments:
                    if other_frag_id == frag_id:
                        continue
                    other_entry = index.fragments[other_frag_id]
                    if other_entry.class_name != class_name and other_entry.class_name in index.classes:
                        # Check if this fragment's code references the other class
                        if member.target_code and other_entry.class_name in (member.target_code or ""):
                            deps.add(other_entry.class_name)
            summary.dependencies = sorted(deps)
