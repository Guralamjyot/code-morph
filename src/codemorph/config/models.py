"""
Core configuration models for CodeMorph.

Defines all configuration structures using Pydantic for validation.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LanguageType(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVA = "java"


class PythonVersion(str, Enum):
    """Supported Python versions."""

    PY27 = "2.7"
    PY36 = "3.6"
    PY37 = "3.7"
    PY38 = "3.8"
    PY39 = "3.9"
    PY310 = "3.10"
    PY311 = "3.11"
    PY312 = "3.12"


class JavaVersion(str, Enum):
    """Supported Java versions (LTS)."""

    JAVA11 = "11"
    JAVA17 = "17"
    JAVA21 = "21"


class JavaBuildSystem(str, Enum):
    """Java build system options."""

    MAVEN = "maven"
    GRADLE = "gradle"


class CheckpointMode(str, Enum):
    """Human-in-the-loop checkpoint modes."""

    INTERACTIVE = "interactive"  # Stop at every function
    BATCH = "batch"  # Stop only at phase completion
    AUTO = "auto"  # No stops, only report errors


class TranslationStatus(str, Enum):
    """Status of a translated fragment."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    TRANSLATED = "translated"  # LLM generated, not yet compiled
    COMPILED = "compiled"  # Compiles successfully
    TYPE_VERIFIED = "type_verified"  # Type compatibility verified
    IO_VERIFIED = "io_verified"  # I/O equivalence verified
    MOCKED = "mocked"  # Fallback to source language
    FAILED = "failed"  # Could not translate
    HUMAN_REVIEW = "human_review"  # Needs human intervention


class FragmentType(str, Enum):
    """Types of code fragments that can be translated."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    GLOBAL_VAR = "global_var"
    CONSTANT = "constant"
    IMPORT = "import"
    MODULE = "module"


# ============================================================================
# LLM Configuration
# ============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    """Configuration for the LLM client."""

    provider: LLMProvider = Field(default=LLMProvider.OPENROUTER, description="LLM provider")
    host: str = Field(default="http://localhost:11434", description="Ollama server URL (for Ollama)")
    api_key: str | None = Field(default=None, description="API key (for OpenRouter/OpenAI)")
    model: str = Field(default="openrouter/aurora-alpha", description="Model to use for translation")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    context_window: int = Field(default=16384, description="Maximum context window size")
    timeout: int = Field(default=300, description="Request timeout in seconds")


# ============================================================================
# RAG Configuration
# ============================================================================


class RAGConfig(BaseModel):
    """
    Configuration for optional RAG (Retrieval Augmented Generation).

    Implements the two-tier retrieval strategy from Section 17.1:
    - Bootstrap Layer: Golden reference examples from bootstrap_dir
    - Snowball Layer: Verified translations indexed during Phase 2
    """

    enabled: bool = Field(default=False, description="Enable RAG for context retrieval")
    top_k: int = Field(default=2, description="Number of style examples to include in prompts")
    bootstrap_dir: str | None = Field(
        default="./examples/bootstrap_examples",
        description="Directory with golden reference code examples (Bootstrap Layer)"
    )
    include_signatures: bool = Field(default=True, description="Include function signatures")
    include_docstrings: bool = Field(default=True, description="Include docstrings")
    chroma_persist_dir: str = Field(
        default=".codemorph/vectordb", description="ChromaDB persistence directory"
    )


# ============================================================================
# Translation Configuration
# ============================================================================


class TranslationConfig(BaseModel):
    """Configuration for the translation process."""

    max_retries_type_check: int = Field(
        default=15, description="Max retries for Phase 2 (type-driven)"
    )
    max_retries_semantics: int = Field(
        default=5, description="Max retries for Phase 3 (semantics-driven)"
    )
    requery_budget: int = Field(default=10, description="Budget for feature mapping re-queries")
    allow_mocking: bool = Field(
        default=True, description="Allow mocking when translation fails"
    )
    strict_naming: bool = Field(
        default=True, description="Enforce target language naming conventions"
    )


# ============================================================================
# Verification Configuration
# ============================================================================


class VerificationConfig(BaseModel):
    """Configuration for verification and testing."""

    runner: str = Field(default="pytest", description="Test runner to use (pytest, unittest)")
    generate_tests: bool = Field(
        default=True, description="Generate tests if source lacks them"
    )
    equivalence_check: bool = Field(default=True, description="Run I/O equivalence comparison")
    execution_timeout: int = Field(default=30, description="Timeout for test execution (seconds)")
    snapshot_dir: str = Field(
        default=".codemorph/snapshots", description="Directory for execution snapshots"
    )


# ============================================================================
# Project Configuration
# ============================================================================


class SourceConfig(BaseModel):
    """Source project configuration."""

    language: LanguageType
    version: str  # Will be validated against language-specific versions
    root: Path = Field(default=Path("."), description="Source code root directory")
    test_root: Path | None = Field(default=None, description="Test suite root directory")
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["__pycache__", "*.pyc", ".git", "node_modules", "venv", ".venv"],
        description="Patterns to exclude from analysis",
    )


class TargetConfig(BaseModel):
    """Target project configuration."""

    language: LanguageType
    version: str  # Will be validated against language-specific versions
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    build_system: JavaBuildSystem | None = Field(
        default=None, description="Build system for Java (required if target is Java)"
    )
    package_name: str | None = Field(
        default=None, description="Base package name for Java output"
    )


class ProjectConfig(BaseModel):
    """Overall project configuration."""

    name: str = Field(default="codemorph_project", description="Project name")
    source: SourceConfig
    target: TargetConfig
    state_dir: Path = Field(
        default=Path(".codemorph"), description="Directory for CodeMorph state files"
    )


# ============================================================================
# Main Configuration
# ============================================================================


class CodeMorphConfig(BaseModel):
    """Root configuration model for CodeMorph."""

    project: ProjectConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    checkpoint_mode: CheckpointMode = Field(default=CheckpointMode.BATCH)

    def is_version_upgrade(self) -> bool:
        """Check if this is a same-language version upgrade."""
        return self.project.source.language == self.project.target.language

    def get_translation_type(self) -> str:
        """Get a human-readable description of the translation type."""
        src = self.project.source
        tgt = self.project.target

        if self.is_version_upgrade():
            return f"{src.language.value} {src.version} → {tgt.version} (version upgrade)"
        return f"{src.language.value} {src.version} → {tgt.language.value} {tgt.version}"


# ============================================================================
# Code Fragment Models (used across the system)
# ============================================================================


class CodeFragment(BaseModel):
    """Represents an atomic unit of translation."""

    id: str = Field(description="Unique identifier (e.g., 'module.py::function_name')")
    name: str = Field(description="Fragment name")
    fragment_type: FragmentType
    source_file: Path
    start_line: int
    end_line: int
    source_code: str
    dependencies: list[str] = Field(default_factory=list, description="IDs of dependencies")
    dependents: list[str] = Field(default_factory=list, description="IDs of dependents")
    docstring: str | None = None
    signature: str | None = None  # For functions/methods
    parent_class: str | None = None  # For methods
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranslatedFragment(BaseModel):
    """A code fragment with its translation."""

    fragment: CodeFragment
    target_code: str | None = None
    target_name: str | None = None  # Renamed identifier in target language
    target_file: Path | None = None
    target_signature: str | None = None
    status: TranslationStatus = TranslationStatus.PENDING
    compilation_errors: list[str] = Field(default_factory=list)
    type_errors: list[str] = Field(default_factory=list)
    io_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    is_mocked: bool = False
    mock_reason: str | None = None
    llm_conversation_ids: list[str] = Field(
        default_factory=list, description="IDs of LLM conversations for this fragment"
    )


class AnalysisResult(BaseModel):
    """Result of Phase 1: Project Partitioning."""

    fragments: dict[str, CodeFragment] = Field(
        default_factory=dict, description="All extracted fragments"
    )
    translation_order: list[str] = Field(
        default_factory=list, description="Topologically sorted fragment IDs"
    )
    circular_dependencies: list[list[str]] = Field(
        default_factory=list, description="Groups of circular dependencies"
    )
    dependency_graph_path: Path | None = Field(
        default=None, description="Path to serialized dependency graph"
    )
    complexity_scores: dict[str, int] = Field(
        default_factory=dict, description="Complexity score per fragment"
    )
    detected_imports: list["ImportInfo"] = Field(
        default_factory=list, description="All detected imports in the project"
    )


# ============================================================================
# Import and Library Mapping Models
# ============================================================================


class ImportInfo(BaseModel):
    """Represents an import statement detected in source code."""

    module: str = Field(description="The module being imported (e.g., 'json', 'requests')")
    names: list[str] = Field(default_factory=list, description="Specific names imported (for 'from x import y')")
    alias: str | None = Field(default=None, description="Import alias (e.g., 'import numpy as np')")
    is_from_import: bool = Field(default=False, description="True if 'from x import y' style")
    is_standard_library: bool = Field(default=False, description="True if part of standard library")
    is_internal: bool = Field(default=False, description="True if internal to the project being translated")
    source_file: Path | None = Field(default=None, description="File where this import was found")


class LibraryMapping(BaseModel):
    """Mapping from a source language library to target language equivalent."""

    source_library: str = Field(description="Source library name (e.g., 'json', 'requests')")
    target_library: str = Field(description="Target library name (e.g., 'com.fasterxml.jackson.databind')")
    target_imports: list[str] = Field(
        default_factory=list,
        description="Java imports to add (e.g., ['com.fasterxml.jackson.databind.ObjectMapper'])"
    )
    maven_dependency: str | None = Field(
        default=None,
        description="Maven dependency string (groupId:artifactId:version)"
    )
    notes: str | None = Field(default=None, description="Usage notes or important differences")
    verified_by_user: bool = Field(default=False, description="True if user has verified this mapping")
    suggested_by_llm: bool = Field(default=False, description="True if this was suggested by LLM")


class LibraryAnalysisResult(BaseModel):
    """Result of analyzing project imports for library mapping."""

    known_mappings: list[LibraryMapping] = Field(
        default_factory=list, description="Libraries with known mappings"
    )
    unknown_imports: list["ImportInfo"] = Field(
        default_factory=list, description="Imports needing LLM suggestion"
    )
    internal_imports: list[str] = Field(
        default_factory=list, description="Internal project imports (no mapping needed)"
    )
    standard_library: list[str] = Field(
        default_factory=list, description="Standard library imports"
    )


# ============================================================================
# Symbol Registry Models
# ============================================================================


class SymbolMapping(BaseModel):
    """Mapping of a single symbol from source to target language.

    Populated incrementally as each fragment is translated.
    """

    source_name: str = Field(description="Original name in source language (e.g., 'calculate_tax')")
    source_qualified_name: str = Field(
        description="Fully qualified source name (e.g., 'billing.py::calculate_tax')"
    )
    target_name: str = Field(description="Translated name in target language (e.g., 'calculateTax')")
    target_qualified_name: str = Field(
        description="Fully qualified target name (e.g., 'com.app.Billing.calculateTax')"
    )
    symbol_type: str = Field(description="Type: function, class, method, variable, constant")
    target_file: Path | None = Field(default=None, description="Target file path")
    signature: str | None = Field(default=None, description="Target language signature")
    status: TranslationStatus = Field(default=TranslationStatus.PENDING)
    created_at: float = Field(default_factory=lambda: __import__("time").time())


class SymbolConflict(BaseModel):
    """Records a naming conflict and its resolution."""

    original_target_name: str = Field(description="The name that caused the conflict")
    resolved_target_name: str = Field(description="The resolved name (with suffix)")
    source_qualified_name: str = Field(description="Source symbol that had the conflict")
    reason: str = Field(description="Why the conflict occurred")
