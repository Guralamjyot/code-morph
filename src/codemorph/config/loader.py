"""
Configuration loader for CodeMorph.

Handles loading configuration from YAML files and environment variables.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import (
    CodeMorphConfig,
    JavaBuildSystem,
    JavaVersion,
    LanguageType,
    ProjectConfig,
    PythonVersion,
    SourceConfig,
    TargetConfig,
)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def validate_language_version(language: LanguageType, version: str) -> str:
    """Validate that the version is valid for the given language."""
    if language == LanguageType.PYTHON:
        valid_versions = [v.value for v in PythonVersion]
        if version not in valid_versions:
            raise ConfigurationError(
                f"Invalid Python version '{version}'. Valid versions: {valid_versions}"
            )
    elif language == LanguageType.JAVA:
        valid_versions = [v.value for v in JavaVersion]
        if version not in valid_versions:
            raise ConfigurationError(
                f"Invalid Java version '{version}'. Valid versions: {valid_versions}"
            )
    return version


def load_config_from_yaml(config_path: Path) -> CodeMorphConfig:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")

    if raw_config is None:
        raise ConfigurationError("Configuration file is empty")

    try:
        config = CodeMorphConfig(**raw_config)
    except ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed:\n{e}")

    # Additional validation
    validate_language_version(config.project.source.language, config.project.source.version)
    validate_language_version(config.project.target.language, config.project.target.version)

    # Ensure Java target has a build system specified
    if (
        config.project.target.language == LanguageType.JAVA
        and config.project.target.build_system is None
    ):
        raise ConfigurationError(
            "Java target requires 'build_system' to be specified (maven or gradle)"
        )

    return config


def create_config_from_args(
    source_dir: Path,
    source_lang: str,
    source_version: str,
    target_lang: str,
    target_version: str,
    output_dir: Path,
    build_system: str | None = None,
    package_name: str | None = None,
    test_dir: Path | None = None,
    project_name: str | None = None,
    **kwargs: Any,
) -> CodeMorphConfig:
    """Create configuration from CLI arguments."""
    src_language = LanguageType(source_lang.lower())
    tgt_language = LanguageType(target_lang.lower())

    validate_language_version(src_language, source_version)
    validate_language_version(tgt_language, target_version)

    # Parse build system if provided
    java_build_system = None
    if build_system:
        java_build_system = JavaBuildSystem(build_system.lower())
    elif tgt_language == LanguageType.JAVA:
        raise ConfigurationError(
            "Java target requires '--build-system' to be specified (maven or gradle)"
        )

    source_config = SourceConfig(
        language=src_language,
        version=source_version,
        root=source_dir,
        test_root=test_dir,
    )

    target_config = TargetConfig(
        language=tgt_language,
        version=target_version,
        output_dir=output_dir,
        build_system=java_build_system,
        package_name=package_name,
    )

    project_config = ProjectConfig(
        name=project_name or source_dir.name or "codemorph_project",
        source=source_config,
        target=target_config,
        state_dir=output_dir / ".codemorph",
    )

    # Build full config with optional overrides from kwargs
    config_dict: dict[str, Any] = {"project": project_config}

    if "llm" in kwargs:
        config_dict["llm"] = kwargs["llm"]
    if "rag" in kwargs:
        config_dict["rag"] = kwargs["rag"]
    if "translation" in kwargs:
        config_dict["translation"] = kwargs["translation"]
    if "verification" in kwargs:
        config_dict["verification"] = kwargs["verification"]
    if "checkpoint_mode" in kwargs:
        config_dict["checkpoint_mode"] = kwargs["checkpoint_mode"]

    return CodeMorphConfig(**config_dict)


def generate_default_config(output_path: Path) -> None:
    """Generate a default configuration file."""
    default_config = {
        "project": {
            "name": "my_project",
            "source": {
                "language": "python",
                "version": "3.10",
                "root": "./src",
                "test_root": "./tests",
            },
            "target": {
                "language": "java",
                "version": "17",
                "output_dir": "./output",
                "build_system": "gradle",
                "package_name": "com.example.myproject",
            },
        },
        "llm": {
            "host": "http://localhost:11434",
            "model": "deepseek-coder:6.7b",
            "temperature": 0.2,
            "context_window": 16384,
        },
        "rag": {
            "enabled": False,
            "top_k": 10,
            "include_signatures": True,
            "include_docstrings": True,
        },
        "translation": {
            "max_retries_type_check": 15,
            "max_retries_semantics": 5,
            "requery_budget": 10,
            "allow_mocking": True,
            "strict_naming": True,
        },
        "verification": {
            "generate_tests": True,
            "equivalence_check": True,
            "execution_timeout": 30,
        },
        "checkpoint_mode": "batch",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
