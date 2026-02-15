"""
Symbol Registry for CodeMorph.

Per-project registry of source→target symbol mappings.
Populated incrementally as each fragment is translated.
"""

import time
from pathlib import Path

import orjson

from codemorph.config.models import (
    SymbolConflict,
    SymbolMapping,
    TranslationStatus,
)


class SymbolRegistry:
    """
    Per-project registry of source→target symbol mappings.

    This registry:
    - Tracks all translated symbols with their source and target names
    - Handles naming conflicts automatically (appends suffix)
    - Persists to .codemorph/symbol_registry.json
    - Provides lookup for dependency context during translation

    Usage:
        registry = SymbolRegistry(state_dir)

        # Register a translated symbol
        mapping = registry.register_symbol(
            source_name="calculate_tax",
            source_qualified="billing.py::calculate_tax",
            target_name="calculateTax",
            symbol_type="function",
            signature="public double calculateTax(double amount)"
        )

        # Look up target name
        target = registry.get_target_name("billing.py::calculate_tax")

        # Get signature for dependency context
        sig = registry.get_signature("billing.py::calculate_tax")

        # Save to disk
        registry.save()
    """

    def __init__(self, state_dir: Path):
        """
        Initialize the symbol registry.

        Args:
            state_dir: Directory for persisting registry (.codemorph/)
        """
        self.state_dir = state_dir
        self.mappings: dict[str, SymbolMapping] = {}
        self.conflicts: list[SymbolConflict] = []
        self._target_names: set[str] = set()  # For conflict detection
        self._target_qualified_names: set[str] = set()

    def register_symbol(
        self,
        source_name: str,
        source_qualified: str,
        target_name: str,
        symbol_type: str,
        target_file: Path | None = None,
        signature: str | None = None,
        target_package: str | None = None,
    ) -> SymbolMapping:
        """
        Register a translated symbol in the registry.

        Automatically handles naming conflicts by appending numeric suffix.

        Args:
            source_name: Original name in source language
            source_qualified: Fully qualified source name (e.g., "module.py::func")
            target_name: Translated name in target language
            symbol_type: Type of symbol (function, class, method, variable, constant)
            target_file: Path to target file (optional)
            signature: Target language signature (optional)
            target_package: Target package name for Java (optional)

        Returns:
            The created SymbolMapping (may have modified target_name if conflict)
        """
        # Check for naming conflicts
        final_target_name = target_name
        if target_name in self._target_names:
            final_target_name = self._resolve_conflict(target_name)
            self.conflicts.append(SymbolConflict(
                original_target_name=target_name,
                resolved_target_name=final_target_name,
                source_qualified_name=source_qualified,
                reason="Name collision with existing symbol"
            ))

        # Build qualified target name
        target_qualified = self._build_qualified_name(
            final_target_name, target_file, target_package
        )

        mapping = SymbolMapping(
            source_name=source_name,
            source_qualified_name=source_qualified,
            target_name=final_target_name,
            target_qualified_name=target_qualified,
            symbol_type=symbol_type,
            target_file=target_file,
            signature=signature,
            status=TranslationStatus.TRANSLATED,
            created_at=time.time(),
        )

        self.mappings[source_qualified] = mapping
        self._target_names.add(final_target_name)
        self._target_qualified_names.add(target_qualified)

        return mapping

    def get_target_name(self, source_qualified: str) -> str | None:
        """
        Look up the target name for a source symbol.

        Args:
            source_qualified: Fully qualified source name

        Returns:
            Target name or None if not found
        """
        if source_qualified in self.mappings:
            return self.mappings[source_qualified].target_name
        return None

    def get_signature(self, source_qualified: str) -> str | None:
        """
        Get the target signature for a source symbol.

        Useful for providing dependency context during translation.

        Args:
            source_qualified: Fully qualified source name

        Returns:
            Target signature or None if not found
        """
        if source_qualified in self.mappings:
            return self.mappings[source_qualified].signature
        return None

    def get_mapping(self, source_qualified: str) -> SymbolMapping | None:
        """
        Get the full mapping for a source symbol.

        Args:
            source_qualified: Fully qualified source name

        Returns:
            SymbolMapping or None if not found
        """
        return self.mappings.get(source_qualified)

    def update_status(self, source_qualified: str, status: TranslationStatus):
        """
        Update the translation status of a symbol.

        Args:
            source_qualified: Fully qualified source name
            status: New status
        """
        if source_qualified in self.mappings:
            self.mappings[source_qualified].status = status

    def update_signature(self, source_qualified: str, signature: str):
        """
        Update the signature of a symbol.

        Args:
            source_qualified: Fully qualified source name
            signature: New signature
        """
        if source_qualified in self.mappings:
            self.mappings[source_qualified].signature = signature

    def get_all_signatures(self) -> dict[str, str]:
        """
        Get all registered signatures for dependency context.

        Returns:
            Dict mapping source_qualified -> signature
        """
        return {
            source_qualified: mapping.signature
            for source_qualified, mapping in self.mappings.items()
            if mapping.signature is not None
        }

    def get_symbols_by_status(self, status: TranslationStatus) -> list[SymbolMapping]:
        """
        Get all symbols with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of matching SymbolMappings
        """
        return [m for m in self.mappings.values() if m.status == status]

    def get_symbols_by_type(self, symbol_type: str) -> list[SymbolMapping]:
        """
        Get all symbols of a specific type.

        Args:
            symbol_type: Type to filter by (function, class, method, etc.)

        Returns:
            List of matching SymbolMappings
        """
        return [m for m in self.mappings.values() if m.symbol_type == symbol_type]

    def has_symbol(self, source_qualified: str) -> bool:
        """Check if a symbol is registered."""
        return source_qualified in self.mappings

    def _resolve_conflict(self, name: str) -> str:
        """
        Resolve a naming conflict by appending a numeric suffix.

        Args:
            name: The conflicting name

        Returns:
            A unique name with suffix (e.g., "calculateTax_1")
        """
        counter = 1
        while f"{name}_{counter}" in self._target_names:
            counter += 1
        return f"{name}_{counter}"

    def _build_qualified_name(
        self,
        name: str,
        target_file: Path | None,
        package: str | None = None,
    ) -> str:
        """
        Build a fully qualified target name.

        Args:
            name: Simple name
            target_file: Target file path
            package: Java package name

        Returns:
            Fully qualified name
        """
        if package:
            return f"{package}.{name}"
        if target_file:
            # Extract class name from file path for Java
            stem = target_file.stem  # e.g., "Calculator" from "Calculator.java"
            return f"{stem}.{name}"
        return name

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> Path:
        """
        Persist the registry to disk.

        Returns:
            Path to the saved registry file
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        registry_file = self.state_dir / "symbol_registry.json"

        data = {
            "mappings": {k: v.model_dump() for k, v in self.mappings.items()},
            "conflicts": [c.model_dump() for c in self.conflicts],
            "updated_at": time.time(),
        }

        # Convert Path objects to strings for serialization
        for mapping_data in data["mappings"].values():
            if mapping_data.get("target_file"):
                mapping_data["target_file"] = str(mapping_data["target_file"])

        with open(registry_file, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

        return registry_file

    @classmethod
    def load(cls, state_dir: Path) -> "SymbolRegistry":
        """
        Load a registry from disk.

        Args:
            state_dir: Directory containing the registry file

        Returns:
            Loaded SymbolRegistry instance (or empty if file doesn't exist)
        """
        registry = cls(state_dir)
        registry_file = state_dir / "symbol_registry.json"

        if not registry_file.exists():
            return registry

        with open(registry_file, "rb") as f:
            data = orjson.loads(f.read())

        # Restore mappings
        for source_qualified, mapping_data in data.get("mappings", {}).items():
            # Convert target_file back to Path
            if mapping_data.get("target_file"):
                mapping_data["target_file"] = Path(mapping_data["target_file"])
            mapping = SymbolMapping(**mapping_data)
            registry.mappings[source_qualified] = mapping
            registry._target_names.add(mapping.target_name)
            registry._target_qualified_names.add(mapping.target_qualified_name)

        # Restore conflicts
        for conflict_data in data.get("conflicts", []):
            registry.conflicts.append(SymbolConflict(**conflict_data))

        return registry

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_statistics(self) -> dict:
        """
        Get statistics about the registry.

        Returns:
            Dict with counts by status and type
        """
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for mapping in self.mappings.values():
            # Count by status
            status_key = mapping.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            # Count by type
            by_type[mapping.symbol_type] = by_type.get(mapping.symbol_type, 0) + 1

        return {
            "total_symbols": len(self.mappings),
            "total_conflicts": len(self.conflicts),
            "by_status": by_status,
            "by_type": by_type,
        }

    def export_mapping_table(self) -> str:
        """
        Export a human-readable mapping table in Markdown format.

        Returns:
            Markdown string with mapping table
        """
        lines = [
            "# Symbol Registry",
            "",
            f"**Total Symbols**: {len(self.mappings)}",
            f"**Naming Conflicts**: {len(self.conflicts)}",
            "",
            "## Mappings",
            "",
            "| Source Name | Target Name | Type | Status |",
            "|-------------|-------------|------|--------|",
        ]

        for mapping in sorted(self.mappings.values(), key=lambda m: m.source_name):
            lines.append(
                f"| `{mapping.source_name}` | `{mapping.target_name}` | "
                f"{mapping.symbol_type} | {mapping.status.value} |"
            )

        if self.conflicts:
            lines.extend([
                "",
                "## Naming Conflicts Resolved",
                "",
                "| Original | Resolved | Reason |",
                "|----------|----------|--------|",
            ])
            for conflict in self.conflicts:
                lines.append(
                    f"| `{conflict.original_target_name}` | "
                    f"`{conflict.resolved_target_name}` | {conflict.reason} |"
                )

        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of registered symbols."""
        return len(self.mappings)

    def __contains__(self, source_qualified: str) -> bool:
        """Check if a symbol is registered."""
        return source_qualified in self.mappings
