"""
Language plugin registry.

Central registry for all language plugins. Handles plugin discovery and instantiation.
"""

from typing import Type

from codemorph.config.models import LanguageType
from codemorph.languages.base.plugin import LanguagePlugin
from codemorph.languages.java.plugin import JavaPlugin
from codemorph.languages.python.plugin import PythonPlugin


class LanguagePluginRegistry:
    """Registry for language plugins."""

    _plugins: dict[LanguageType, Type[LanguagePlugin]] = {
        LanguageType.PYTHON: PythonPlugin,
        LanguageType.JAVA: JavaPlugin,
    }

    @classmethod
    def get_plugin(cls, language: LanguageType, version: str) -> LanguagePlugin:
        """
        Get a language plugin instance.

        Args:
            language: The language type
            version: The language version

        Returns:
            Instantiated language plugin

        Raises:
            ValueError: If language is not supported
        """
        if language not in cls._plugins:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {list(cls._plugins.keys())}"
            )

        plugin_class = cls._plugins[language]
        return plugin_class(version=version)

    @classmethod
    def register_plugin(cls, language: LanguageType, plugin_class: Type[LanguagePlugin]):
        """
        Register a new language plugin.

        This allows for extending CodeMorph with additional languages.

        Args:
            language: The language type
            plugin_class: The plugin class (must extend LanguagePlugin)
        """
        if not issubclass(plugin_class, LanguagePlugin):
            raise TypeError(f"{plugin_class} must extend LanguagePlugin")

        cls._plugins[language] = plugin_class

    @classmethod
    def list_supported_languages(cls) -> list[str]:
        """Get a list of supported language names."""
        return [lang.value for lang in cls._plugins.keys()]

    @classmethod
    def is_supported(cls, language: LanguageType) -> bool:
        """Check if a language is supported."""
        return language in cls._plugins


def get_plugin(language: LanguageType, version: str = "3.10") -> LanguagePlugin:
    """
    Convenience function to get a language plugin.

    Args:
        language: The language type
        version: The language version (default: "3.10" for Python, will use appropriate default for others)

    Returns:
        Instantiated language plugin
    """
    # Use sensible defaults for versions
    if version == "3.10" and language == LanguageType.JAVA:
        version = "17"
    return LanguagePluginRegistry.get_plugin(language, version)
