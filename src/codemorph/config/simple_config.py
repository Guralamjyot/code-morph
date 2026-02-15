"""
Simplified configuration models matching the streamlined spec.

This is a cleaner, more focused configuration system with only essential fields.
The original config/models.py is kept for backward compatibility but this
represents the recommended configuration approach.

Example config YAML (from spec):

```yaml
project:
  name: "MyProject"
  source_language: "python"
  target_language: "java"
  source_root: "./src"
  test_root: "./tests"
  output_root: "./output"

llm:
  provider: "openrouter"  # ollama | openrouter
  model: "anthropic/claude-3.5-sonnet"
  api_key_env: "OPENROUTER_API_KEY"
  temperature: 0.2

translation:
  max_retries_phase2: 15
  max_retries_phase3: 5
  allow_mocking: true
  generate_tests_if_missing: true

checkpoints:
  mode: "interactive"  # interactive | batch | auto
```
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class LLMProvider(str, Enum):
    """Supported LLM providers (simplified)."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class CheckpointMode(str, Enum):
    """Human-in-the-loop checkpoint modes."""

    INTERACTIVE = "interactive"  # Stop at every function
    BATCH = "batch"  # Stop only at phase completion
    AUTO = "auto"  # No stops, only report errors


# =============================================================================
# Project Configuration
# =============================================================================


class ProjectConfig(BaseModel):
    """Essential project configuration."""

    name: str = Field(default="codemorph_project", description="Project name")

    # Source configuration
    source_language: str = Field(description="Source language (python or java)")
    source_version: str | None = Field(
        default=None, description="Source version (e.g., '3.10', '17')"
    )
    source_root: Path = Field(default=Path("./src"), description="Source code directory")
    test_root: Path | None = Field(default=None, description="Test directory (optional)")

    # Target configuration
    target_language: str = Field(description="Target language (python or java)")
    target_version: str | None = Field(
        default=None, description="Target version (e.g., '3.12', '21')"
    )
    output_root: Path = Field(default=Path("./output"), description="Output directory")

    # Optional Java-specific config
    package_name: str | None = Field(
        default=None, description="Java package name (e.g., 'com.example.app')"
    )

    # Internal state directory (hidden from user config)
    state_dir: Path = Field(
        default=Path(".codemorph"), description="Internal state directory"
    )

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language is supported."""
        if v.lower() not in ["python", "java"]:
            raise ValueError(f"Unsupported language: {v}. Must be 'python' or 'java'")
        return v.lower()

    def is_version_upgrade(self) -> bool:
        """Check if this is a same-language version upgrade."""
        return self.source_language == self.target_language


# =============================================================================
# LLM Configuration
# =============================================================================


class LLMConfig(BaseModel):
    """Essential LLM configuration."""

    provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM provider")

    model: str = Field(
        default="deepseek-coder:6.7b",
        description="Model name (e.g., 'deepseek-coder:6.7b', 'anthropic/claude-3.5-sonnet')",
    )

    # API configuration
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable name containing API key (e.g., 'OPENROUTER_API_KEY')",
    )

    host: str = Field(
        default="http://localhost:11434", description="Ollama host (for Ollama provider)"
    )

    # Generation parameters
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature (0 = deterministic)"
    )

    context_window: int = Field(
        default=16384, description="Maximum context window size in tokens"
    )

    @property
    def api_key(self) -> str | None:
        """Get API key from environment if configured."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


# =============================================================================
# Translation Configuration
# =============================================================================


class TranslationConfig(BaseModel):
    """Translation process configuration (simplified)."""

    # Retry budgets (from spec)
    max_retries_phase2: int = Field(
        default=15, ge=1, description="Max retries for Phase 2 (compilation + type-compat)"
    )

    max_retries_phase3: int = Field(
        default=5, ge=1, description="Max retries for Phase 3 (I/O equivalence)"
    )

    # Behavioral flags
    allow_mocking: bool = Field(
        default=True, description="Generate mocks when translation fails"
    )

    generate_tests_if_missing: bool = Field(
        default=True, description="Auto-generate tests if source has none"
    )

    strict_naming: bool = Field(
        default=True, description="Enforce target language naming conventions"
    )


# =============================================================================
# Checkpoint Configuration
# =============================================================================


class CheckpointConfig(BaseModel):
    """Human-in-the-loop checkpoint configuration."""

    mode: CheckpointMode = Field(
        default=CheckpointMode.BATCH, description="Checkpoint frequency"
    )


# =============================================================================
# Optional Advanced Features
# =============================================================================


class RAGConfig(BaseModel):
    """
    RAG (Retrieval Augmented Generation) configuration.

    OPTIONAL: Disabled by default. Enable for style-consistent translation.
    """

    enabled: bool = Field(default=False, description="Enable RAG")

    top_k: int = Field(default=2, ge=1, description="Number of examples to retrieve")

    bootstrap_dir: Path | None = Field(
        default=None, description="Directory with golden reference examples (Bootstrap Layer)"
    )


# =============================================================================
# Root Configuration
# =============================================================================


class SimpleCodeMorphConfig(BaseModel):
    """
    Simplified CodeMorph configuration (recommended).

    This is a streamlined version matching the spec with only essential options.
    """

    project: ProjectConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Optional advanced features
    rag: RAGConfig = Field(default_factory=RAGConfig)

    def get_translation_description(self) -> str:
        """Get human-readable translation description."""
        src_ver = f" {self.project.source_version}" if self.project.source_version else ""
        tgt_ver = f" {self.project.target_version}" if self.project.target_version else ""

        if self.project.is_version_upgrade():
            return f"{self.project.source_language}{src_ver} → {self.project.target_language}{tgt_ver} (version upgrade)"

        return f"{self.project.source_language}{src_ver} → {self.project.target_language}{tgt_ver}"

    @classmethod
    def from_minimal_dict(cls, config_dict: dict[str, Any]) -> "SimpleCodeMorphConfig":
        """
        Create config from a minimal dictionary.

        Allows users to specify only required fields; others use defaults.

        Example:
            config = SimpleCodeMorphConfig.from_minimal_dict({
                "project": {
                    "source_language": "python",
                    "target_language": "java",
                },
                "llm": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-3.5-sonnet",
                }
            })
        """
        return cls(**config_dict)


# =============================================================================
# Example Configurations
# =============================================================================


def get_example_python_to_java_config() -> SimpleCodeMorphConfig:
    """Get example configuration for Python → Java translation."""
    return SimpleCodeMorphConfig(
        project=ProjectConfig(
            name="PythonToJavaProject",
            source_language="python",
            source_version="3.10",
            source_root=Path("./src"),
            target_language="java",
            target_version="17",
            output_root=Path("./output"),
            package_name="com.example.app",
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-3.5-sonnet",
            api_key_env="OPENROUTER_API_KEY",
            temperature=0.2,
        ),
        translation=TranslationConfig(
            max_retries_phase2=15,
            max_retries_phase3=5,
            allow_mocking=True,
            generate_tests_if_missing=True,
        ),
        checkpoints=CheckpointConfig(mode=CheckpointMode.BATCH),
    )


def get_example_java_to_python_config() -> SimpleCodeMorphConfig:
    """Get example configuration for Java → Python translation."""
    return SimpleCodeMorphConfig(
        project=ProjectConfig(
            name="JavaToPythonProject",
            source_language="java",
            source_version="11",
            source_root=Path("./src/main/java"),
            target_language="python",
            target_version="3.12",
            output_root=Path("./output"),
        ),
        llm=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="deepseek-coder:33b",
            temperature=0.2,
        ),
        translation=TranslationConfig(
            max_retries_phase2=15,
            max_retries_phase3=5,
        ),
        checkpoints=CheckpointConfig(mode=CheckpointMode.INTERACTIVE),
    )


def get_example_python_upgrade_config() -> SimpleCodeMorphConfig:
    """Get example configuration for Python version upgrade (3.8 → 3.12)."""
    return SimpleCodeMorphConfig(
        project=ProjectConfig(
            name="PythonUpgrade",
            source_language="python",
            source_version="3.8",
            source_root=Path("./src"),
            target_language="python",
            target_version="3.12",
            output_root=Path("./output"),
        ),
        llm=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="codellama:13b",
        ),
        checkpoints=CheckpointConfig(mode=CheckpointMode.AUTO),
    )
