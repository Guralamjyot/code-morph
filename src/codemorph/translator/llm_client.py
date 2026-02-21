"""
LLM client for code translation.

Handles communication with Ollama, OpenRouter, and OpenAI for generating code translations.
"""

import json
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import logging
import ollama
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from codemorph.config.models import CodeFragment, LLMConfig, LLMProvider

load_dotenv()


def strip_markdown_code_blocks(text: str) -> str:
    """
    Strip markdown code blocks from LLM response.

    Handles formats like:
    - ```java\ncode\n```
    - ```\ncode\n```
    - ``` code ```

    Args:
        text: Raw LLM response that may contain markdown formatting

    Returns:
        Clean code without markdown wrappers
    """
    text = text.strip()

    # Pattern to match code blocks with optional language identifier
    # Matches: ```java\n...\n``` or ```\n...\n```
    pattern = r'^```(?:\w+)?\s*\n?(.*?)\n?```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Handle inline code blocks on single lines: ```code```
    if text.startswith('```') and text.endswith('```'):
        # Remove opening ```lang or ``` and closing ```
        text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        # Remove language identifier if present on first line
        lines = text.split('\n', 1)
        if len(lines) > 1 and lines[0].strip().isalpha():
            text = lines[1]
        return text.strip()

    return text


class LLMConversation:
    """Represents a single conversation with the LLM."""

    def __init__(self, conversation_id: str | None = None):
        self.id = conversation_id or str(uuid4())
        self.messages: list[dict[str, str]] = []
        self.metadata: dict[str, Any] = {}
        self.created_at = time.time()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "id": self.id,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": time.time(),
        }

    def save(self, output_dir: Path):
        """Save conversation to a JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.id}.json"

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class OllamaClient:
    """Client for Ollama LLM API."""

    def __init__(self, config: LLMConfig, conversation_log_dir: Path | None = None):
        self.config = config
        self.conversation_log_dir = conversation_log_dir
        self.client = ollama.Client(host=config.host)

        # Test connection
        try:
            self.client.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {config.host}: {e}")

    def _create_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Create the system prompt for the LLM."""
        prompt = f"""You are an expert software engineer specializing in code translation.
Your task is to translate code from {source_lang} to {target_lang}.

CRITICAL REQUIREMENTS:
1. Maintain exact same behavior - the translated code must produce identical outputs for identical inputs
2. Preserve function signatures and types as much as possible
3. Follow {target_lang} naming conventions and best practices
4. Include all necessary imports and dependencies
5. Add comments only where the translation is non-obvious
6. Return ONLY raw {target_lang} code - NO markdown, NO backticks, NO ```{target_lang.lower()} blocks
7. Ensure the code compiles/runs without errors
8. Start your response directly with the code (e.g., "import" or "public class")

NEVER wrap your response in markdown code blocks like ```java or ```. Output raw code only.

When you cannot translate something directly:
- Document the limitation in a comment
- Provide the closest equivalent
"""
        # Add Java→Python-specific rules
        if source_lang.lower() == "java" and target_lang.lower() == "python":
            prompt += """
JAVA-TO-PYTHON RULES:
9. NEVER shadow Python builtins as variable names. Common trap: `iter = iter(x)`. Use `_iter` or `it` instead. Full list: iter, list, dict, type, id, range, map, filter, set, hash, next, tuple, str, int, float, bool, open, format, input, object, super, bytes, sorted, reversed, enumerate, zip, any, all, min, max, sum.
10. Java `static` methods MUST get `@staticmethod` decorator in Python.
11. Java constructors → `def __init__(self, ...):` with `super().__init__(...)` for subclasses.
12. Java `toString()` → `__str__()`, `equals()` → `__eq__()`, `hashCode()` → `__hash__()`, `compareTo()` → `__lt__()/__eq__()` or implement `@functools.total_ordering`.
13. For consistent snake_case conversion: camelCase digits stay attached — `getValue0()` → `get_value0()`, NOT `get_value_0()`.
"""
        # Add Python→Java-specific rules
        elif source_lang.lower() == "python" and target_lang.lower() == "java":
            prompt += """
PYTHON-TO-JAVA RULES:
9. Python `__init__` → Java constructor with the class name. Remove `self` parameter.
10. Python `__str__` → Java `toString()`, `__eq__` → `equals(Object)`, `__hash__` → `hashCode()`, `__lt__` → `compareTo()` or `Comparable<T>` interface.
11. Python `self.field` → Java instance fields declared at class level. Infer types from usage.
12. Python `Enum` class → Java `enum` with proper value field, constructor, and getter.
13. Python `list[T]` → Java `List<T>` (use `ArrayList<>` for instantiation). Import `java.util.*`.
14. Python `dict[K,V]` → Java `Map<K,V>` (use `HashMap<>` for instantiation).
15. Python `Optional[T]` → Java nullable `T` (use `@Nullable` or null checks).
16. Python `raise ValueError(msg)` → Java `throw new IllegalArgumentException(msg)`.
17. Python `isinstance(x, T)` → Java `x instanceof T`.
18. Python snake_case methods → Java camelCase. e.g., `get_total_value` → `getTotalValue`.
19. Every Java class/enum MUST be wrapped in a `public class ClassName { ... }` with proper access modifiers.
20. Python `for x in collection` → Java enhanced for-loop or Stream API.
21. Python `None` → Java `null`, `True`/`False` → `true`/`false`.
22. When translating a complete Python class, produce a COMPLETE Java class with ALL methods, fields, constructors, and imports. Do NOT omit any methods.
23. CRITICAL: Output ONLY the class/enum being translated. Do NOT include stub, placeholder, or helper definitions for other classes. Assume referenced classes exist in separate files. Use imports instead.
24. For standalone functions (like `main`), wrap in a `public class Main { public static void main(String[] args) { ... } }`.
25. Python `Enum` with string values: use Java `enum` with a `private final String value` field, a constructor, and a `getValue()` method. Include the `isPerishable()` type methods if present.
26. CRITICAL: Python uses direct attribute access (self.name, self.price, etc.). When translating a class to Java, generate public getter methods for ALL private instance fields. For example: `self.name` → `private String name;` + `public String getName() { return name; }`. This is essential because other classes will need to access these fields via getters.
27. For mutable fields (like quantity), also generate a setter method.
28. Do NOT use Java reflection (java.lang.reflect) to access fields. Always use proper getter/setter methods.
"""
        return prompt

    def translate_fragment(
        self,
        fragment: CodeFragment,
        source_lang: str,
        source_version: str,
        target_lang: str,
        target_version: str,
        feature_mapping_instructions: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, LLMConversation]:
        """
        Translate a code fragment using the LLM.

        Args:
            fragment: The code fragment to translate
            source_lang: Source language name
            source_version: Source language version
            target_lang: Target language name
            target_version: Target language version
            feature_mapping_instructions: Additional instructions from feature mapping
            context: Additional context (dependencies, etc.)

        Returns:
            Tuple of (translated_code, conversation)
        """
        conversation = LLMConversation()
        conversation.metadata = {
            "fragment_id": fragment.id,
            "fragment_type": fragment.fragment_type.value,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        # Build the prompt
        system_prompt = self._create_system_prompt(source_lang, target_lang)
        conversation.add_message("system", system_prompt)

        # Build user prompt
        user_prompt = self._build_translation_prompt(
            fragment,
            source_lang,
            source_version,
            target_lang,
            target_version,
            feature_mapping_instructions,
            context,
        )
        conversation.add_message("user", user_prompt)

        # Call LLM
        response = self._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        # Save conversation if logging enabled
        if self.conversation_log_dir:
            conversation.save(self.conversation_log_dir)

        return response, conversation

    def refine_translation(
        self,
        original_code: str,
        translated_code: str,
        errors: list[str],
        conversation: LLMConversation,
        instruction: str = "Fix the compilation errors",
    ) -> tuple[str, LLMConversation]:
        """
        Refine a translation based on errors.

        Args:
            original_code: Original source code
            translated_code: Current translation
            errors: List of error messages
            conversation: Existing conversation to continue
            instruction: What to fix (compilation, I/O mismatch, etc.)

        Returns:
            Tuple of (refined_code, updated_conversation)
        """
        # Build refinement prompt
        user_prompt = f"""{instruction}

Original code:
```
{original_code}
```

Current translation:
```
{translated_code}
```

Errors:
{self._format_errors(errors)}

Please provide a corrected version of the translation.
Return ONLY the corrected code, no explanations.
"""

        conversation.add_message("user", user_prompt)

        # Call LLM
        response = self._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        # Save conversation
        if self.conversation_log_dir:
            conversation.save(self.conversation_log_dir)

        return response, conversation

    def fix_io_mismatch(
        self,
        fragment: CodeFragment,
        translated_code: str,
        expected_output: Any,
        actual_output: Any,
        test_input: dict[str, Any],
        conversation: LLMConversation,
    ) -> tuple[str, LLMConversation]:
        """
        Fix I/O mismatch (Phase 3 refinement).

        Args:
            fragment: Original fragment
            translated_code: Current translation
            expected_output: Expected output from source
            actual_output: Actual output from translation
            test_input: The test input that produced the mismatch
            conversation: Existing conversation

        Returns:
            Tuple of (fixed_code, updated_conversation)
        """
        user_prompt = f"""The translated function produces incorrect output.

Original function: {fragment.name}
Test input: {json.dumps(test_input, indent=2)}
Expected output: {json.dumps(expected_output, indent=2)}
Actual output: {json.dumps(actual_output, indent=2)}

Current translation:
```
{translated_code}
```

IMPORTANT: You must ONLY modify the function body. DO NOT change the signature.

Please fix the logic to produce the correct output.
Return ONLY the corrected code.
"""

        conversation.add_message("user", user_prompt)

        # Call LLM
        response = self._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        # Save conversation
        if self.conversation_log_dir:
            conversation.save(self.conversation_log_dir)

        return response, conversation

    def _build_translation_prompt(
        self,
        fragment: CodeFragment,
        source_lang: str,
        source_version: str,
        target_lang: str,
        target_version: str,
        feature_mapping_instructions: list[str] | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Build the main translation prompt."""
        prompt_parts = []

        # Basic info
        prompt_parts.append(f"Translate this {source_lang} {source_version} code to {target_lang} {target_version}:")

        # Feature mapping instructions
        if feature_mapping_instructions:
            prompt_parts.append("\nIMPORTANT TRANSLATION RULES:")
            for instruction in feature_mapping_instructions:
                prompt_parts.append(f"- {instruction}")

        # Context (dependency signatures)
        if context and "dependency_signatures" in context:
            prompt_parts.append("\nAvailable dependencies:")
            for sig in context["dependency_signatures"]:
                prompt_parts.append(f"  {sig}")

        # Full overload source code for merged translation
        if context and "overload_sources" in context:
            prompt_parts.append(
                "\nOVERLOAD MERGE: This method has sibling Java overloads (shown below). "
                "You MUST produce a SINGLE Python function that handles ALL variants "
                "using default parameters or *args dispatch. "
                "NEVER produce multiple def statements with the same name."
            )
            for i, src in enumerate(context["overload_sources"], 1):
                prompt_parts.append(f"\n--- Sibling overload {i} ---")
                prompt_parts.append(f"```java\n{src}\n```")
        # Fallback: signature-only overload context
        elif context and "overloads" in context:
            overload_sigs = context["overloads"]
            prompt_parts.append(
                "\nOVERLOAD WARNING: This method has multiple Java overloads:"
            )
            for sig in overload_sigs:
                prompt_parts.append(f"  - {sig}")
            prompt_parts.append(
                "Python lacks method overloading. Combine overloads into a "
                "single method using `*args` + `isinstance` dispatch or optional "
                "parameters with defaults. NEVER create self-recursive delegation "
                "between overloads."
            )

        # Per-method instruction: when translating a single method, tell the LLM
        # to output ONLY the method body (no enclosing class).
        if fragment.fragment_type.value == "method":
            if target_lang.lower() == "java":
                prompt_parts.append(
                    "\nIMPORTANT: You are translating a SINGLE Python method to Java. "
                    "Output ONLY the Java method (access modifier + return type + name + body). "
                    "Do NOT wrap in a class declaration. "
                    "Do NOT include field declarations or import statements. "
                    "The method will be integrated into a complete class in a subsequent step."
                )
            elif target_lang.lower() == "python":
                prompt_parts.append(
                    "\nIMPORTANT: You are translating a SINGLE Java method to Python. "
                    "Output ONLY the Python method (def statement + body). "
                    "Do NOT wrap in a class definition. "
                    "The method will be integrated into a complete Python class in a subsequent step."
                )

        # For class fragments translated with pre-translated method context
        if context and "translated_methods" in context:
            methods = context["translated_methods"]
            prompt_parts.append(
                f"\nCONTEXT — Pre-translated {target_lang} methods for this class "
                f"({len(methods)} method(s)). Integrate these into the complete class:"
            )
            for m in methods:
                prompt_parts.append(f"\n// {m['name']}:")
                prompt_parts.append(f"```{target_lang}\n{m['target_method'].strip()}\n```")
            prompt_parts.append(
                "\nIntegrate all of the above translated methods into the complete "
                f"{target_lang} class. Keep their logic intact; adjust for cohesion as needed."
            )

        # The code itself
        prompt_parts.append(f"\nCode to translate:")
        if fragment.docstring:
            prompt_parts.append(f"# Documentation: {fragment.docstring}")

        prompt_parts.append(f"```{source_lang}")
        prompt_parts.append(fragment.source_code)
        prompt_parts.append("```")

        # Signature hint if available
        if fragment.signature:
            prompt_parts.append(f"\nOriginal signature: {fragment.signature}")

        prompt_parts.append("\nTranslated code:")

        return "\n".join(prompt_parts)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """
        Call the Ollama LLM API.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            The assistant's response content (with markdown stripped)
        """
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.context_window,
                },
            )

            content = response["message"]["content"]
            # Strip markdown code blocks that LLMs often add despite instructions
            return strip_markdown_code_blocks(content)

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

    def _format_errors(self, errors: list[str]) -> str:
        """Format errors for display in prompt."""
        if not errors:
            return "No errors"

        formatted = []
        for i, error in enumerate(errors, 1):
            formatted.append(f"{i}. {error}")

        return "\n".join(formatted)

    def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        try:
            models = self.client.list()
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}")

    def check_model_availability(self) -> bool:
        """Check if the configured model is available."""
        try:
            available = self.get_available_models()
            return self.config.model in available
        except:
            return False


class OpenRouterClient:
    """Client for OpenAI-compatible APIs (OpenRouter, OpenAI, etc.)."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: LLMConfig, conversation_log_dir: Path | None = None, base_url: str | None = "openrouter"):
        self.config = config
        self.conversation_log_dir = conversation_log_dir

        if not config.api_key:
            import os
            config.api_key = os.environ.get("OPENAI_API_KEY")
            if not config.api_key:
                raise ValueError("API key required (set api_key in config or OPENAI_API_KEY env var)")

        # base_url="openrouter" → use OpenRouter URL
        # base_url=None → use default OpenAI URL
        # base_url="https://..." → use custom URL
        if base_url == "openrouter":
            effective_url = self.OPENROUTER_BASE_URL
        else:
            effective_url = base_url  # None = OpenAI default

        client_kwargs = {"api_key": config.api_key}
        if effective_url is not None:
            client_kwargs["base_url"] = effective_url
        self.client = OpenAI(**client_kwargs)

    def _create_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Create the system prompt for the LLM."""
        prompt = f"""You are an expert software engineer specializing in code translation.
Your task is to translate code from {source_lang} to {target_lang}.

CRITICAL REQUIREMENTS:
1. Maintain exact same behavior - the translated code must produce identical outputs for identical inputs
2. Preserve function signatures and types as much as possible
3. Follow {target_lang} naming conventions and best practices
4. Include all necessary imports and dependencies
5. Add comments only where the translation is non-obvious
6. Return ONLY raw {target_lang} code - NO markdown, NO backticks, NO ```{target_lang.lower()} blocks
7. Ensure the code compiles/runs without errors
8. Start your response directly with the code (e.g., "import" or "public class")

NEVER wrap your response in markdown code blocks like ```java or ```. Output raw code only.

When you cannot translate something directly:
- Document the limitation in a comment
- Provide the closest equivalent
"""
        # Add Java→Python-specific rules
        if source_lang.lower() == "java" and target_lang.lower() == "python":
            prompt += """
JAVA-TO-PYTHON RULES:
9. NEVER shadow Python builtins as variable names. Common trap: `iter = iter(x)`. Use `_iter` or `it` instead. Full list: iter, list, dict, type, id, range, map, filter, set, hash, next, tuple, str, int, float, bool, open, format, input, object, super, bytes, sorted, reversed, enumerate, zip, any, all, min, max, sum.
10. Java `static` methods MUST get `@staticmethod` decorator in Python.
11. Java constructors → `def __init__(self, ...):` with `super().__init__(...)` for subclasses.
12. Java `toString()` → `__str__()`, `equals()` → `__eq__()`, `hashCode()` → `__hash__()`, `compareTo()` → `__lt__()/__eq__()` or implement `@functools.total_ordering`.
13. For consistent snake_case conversion: camelCase digits stay attached — `getValue0()` → `get_value0()`, NOT `get_value_0()`.
"""
        # Add Python→Java-specific rules
        elif source_lang.lower() == "python" and target_lang.lower() == "java":
            prompt += """
PYTHON-TO-JAVA RULES:
9. Python `__init__` → Java constructor with the class name. Remove `self` parameter.
10. Python `__str__` → Java `toString()`, `__eq__` → `equals(Object)`, `__hash__` → `hashCode()`, `__lt__` → `compareTo()` or `Comparable<T>` interface.
11. Python `self.field` → Java instance fields declared at class level. Infer types from usage.
12. Python `Enum` class → Java `enum` with proper value field, constructor, and getter.
13. Python `list[T]` → Java `List<T>` (use `ArrayList<>` for instantiation). Import `java.util.*`.
14. Python `dict[K,V]` → Java `Map<K,V>` (use `HashMap<>` for instantiation).
15. Python `Optional[T]` → Java nullable `T` (use `@Nullable` or null checks).
16. Python `raise ValueError(msg)` → Java `throw new IllegalArgumentException(msg)`.
17. Python `isinstance(x, T)` → Java `x instanceof T`.
18. Python snake_case methods → Java camelCase. e.g., `get_total_value` → `getTotalValue`.
19. Every Java class/enum MUST be wrapped in a `public class ClassName { ... }` with proper access modifiers.
20. Python `for x in collection` → Java enhanced for-loop or Stream API.
21. Python `None` → Java `null`, `True`/`False` → `true`/`false`.
22. When translating a complete Python class, produce a COMPLETE Java class with ALL methods, fields, constructors, and imports. Do NOT omit any methods.
23. CRITICAL: Output ONLY the class/enum being translated. Do NOT include stub, placeholder, or helper definitions for other classes. Assume referenced classes exist in separate files. Use imports instead.
24. For standalone functions (like `main`), wrap in a `public class Main { public static void main(String[] args) { ... } }`.
25. Python `Enum` with string values: use Java `enum` with a `private final String value` field, a constructor, and a `getValue()` method. Include the `isPerishable()` type methods if present.
26. CRITICAL: Python uses direct attribute access (self.name, self.price, etc.). When translating a class to Java, generate public getter methods for ALL private instance fields. For example: `self.name` → `private String name;` + `public String getName() { return name; }`. This is essential because other classes will need to access these fields via getters.
27. For mutable fields (like quantity), also generate a setter method.
28. Do NOT use Java reflection (java.lang.reflect) to access fields. Always use proper getter/setter methods.
"""
        return prompt

    def translate_fragment(
        self,
        fragment: CodeFragment,
        source_lang: str,
        target_lang: str,
        source_version: str = "",
        target_version: str = "",
        dependency_context: dict[str, Any] | None = None,
        feature_mapping_instructions: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, LLMConversation]:
        """Translate a code fragment using OpenRouter."""
        # Merge context and dependency_context
        effective_context = dependency_context or context

        conversation = LLMConversation()
        conversation.metadata = {
            "fragment_id": fragment.id,
            "fragment_type": fragment.fragment_type.value,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        system_prompt = self._create_system_prompt(source_lang, target_lang)
        conversation.add_message("system", system_prompt)

        user_prompt = self._build_translation_prompt(
            fragment,
            source_lang,
            target_lang,
            feature_mapping_instructions,
            effective_context,
        )
        conversation.add_message("user", user_prompt)

        response = self._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        if self.conversation_log_dir:
            conversation.save(self.conversation_log_dir)

        return response, conversation

    def refine_translation(
        self,
        original_code: str,
        translated_code: str,
        errors: list[str],
        conversation: LLMConversation | None = None,
        dependency_context: dict[str, Any] | None = None,
        feature_mapping_instructions: list[str] | None = None,
        instruction: str = "Fix the compilation errors",
    ) -> tuple[str, LLMConversation]:
        """Refine a translation based on errors."""
        if conversation is None:
            conversation = LLMConversation()

        user_prompt = f"""{instruction}

Original code:
```
{original_code}
```

Current translation:
```
{translated_code}
```

Errors:
{self._format_errors(errors)}

Please provide a corrected version of the translation.
Return ONLY the corrected code, no explanations.
"""

        conversation.add_message("user", user_prompt)
        response = self._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        if self.conversation_log_dir:
            conversation.save(self.conversation_log_dir)

        return response, conversation

    def fix_io_mismatch(
        self,
        fragment: CodeFragment,
        translated_code: str,
        expected_output: Any,
        actual_output: Any,
        test_input: dict[str, Any],
        conversation: LLMConversation,
    ) -> tuple[str, LLMConversation]:
        """Fix I/O mismatch (Phase 3 refinement)."""
        user_prompt = f"""The translated function produces incorrect output.

Original function: {fragment.name}
Test input: {json.dumps(test_input, indent=2)}
Expected output: {json.dumps(expected_output, indent=2)}
Actual output: {json.dumps(actual_output, indent=2)}

Current translation:
```
{translated_code}
```

IMPORTANT: You must ONLY modify the function body. DO NOT change the signature.

Please fix the logic to produce the correct output.
Return ONLY the corrected code.
"""

        conversation.add_message("user", user_prompt)
        response = self._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        if self.conversation_log_dir:
            conversation.save(self.conversation_log_dir)

        return response, conversation

    def _build_translation_prompt(
        self,
        fragment: CodeFragment,
        source_lang: str,
        target_lang: str,
        feature_mapping_instructions: list[str] | None,
        dependency_context: dict[str, Any] | None,
    ) -> str:
        """Build the main translation prompt."""
        prompt_parts = []

        prompt_parts.append(f"Translate this {source_lang} code to {target_lang}:")

        if feature_mapping_instructions:
            prompt_parts.append("\nIMPORTANT TRANSLATION RULES:")
            for instruction in feature_mapping_instructions:
                prompt_parts.append(f"- {instruction}")

        if dependency_context and "dependency_signatures" in dependency_context:
            prompt_parts.append("\nAvailable dependencies:")
            for sig in dependency_context["dependency_signatures"]:
                prompt_parts.append(f"  {sig}")

        # Full overload source code for merged translation
        if dependency_context and "overload_sources" in dependency_context:
            prompt_parts.append(
                "\nOVERLOAD MERGE: This method has sibling Java overloads (shown below). "
                "You MUST produce a SINGLE Python function that handles ALL variants "
                "using default parameters or *args dispatch. "
                "NEVER produce multiple def statements with the same name."
            )
            for i, src in enumerate(dependency_context["overload_sources"], 1):
                prompt_parts.append(f"\n--- Sibling overload {i} ---")
                prompt_parts.append(f"```java\n{src}\n```")
        # Fallback: signature-only overload context
        elif dependency_context and "overloads" in dependency_context:
            overload_sigs = dependency_context["overloads"]
            prompt_parts.append(
                "\nOVERLOAD WARNING: This method has multiple Java overloads:"
            )
            for sig in overload_sigs:
                prompt_parts.append(f"  - {sig}")
            prompt_parts.append(
                "Python lacks method overloading. Combine overloads into a "
                "single method using `*args` + `isinstance` dispatch or optional "
                "parameters with defaults. NEVER create self-recursive delegation "
                "between overloads."
            )

        # Per-method instruction: when translating a single method, output only
        # the method body (no enclosing class).
        if fragment.fragment_type.value == "method":
            if target_lang.lower() == "java":
                prompt_parts.append(
                    "\nIMPORTANT: You are translating a SINGLE Python method to Java. "
                    "Output ONLY the Java method (access modifier + return type + name + body). "
                    "Do NOT wrap in a class declaration. "
                    "Do NOT include field declarations or import statements. "
                    "The method will be integrated into a complete class in a subsequent step."
                )
            elif target_lang.lower() == "python":
                prompt_parts.append(
                    "\nIMPORTANT: You are translating a SINGLE Java method to Python. "
                    "Output ONLY the Python method (def statement + body). "
                    "Do NOT wrap in a class definition. "
                    "The method will be integrated into a complete Python class in a subsequent step."
                )

        # For class fragments translated with pre-translated method context
        if dependency_context and "translated_methods" in dependency_context:
            methods = dependency_context["translated_methods"]
            prompt_parts.append(
                f"\nCONTEXT — Pre-translated {target_lang} methods for this class "
                f"({len(methods)} method(s)). Integrate these into the complete class:"
            )
            for m in methods:
                prompt_parts.append(f"\n// {m['name']}:")
                prompt_parts.append(f"```{target_lang}\n{m['target_method'].strip()}\n```")
            prompt_parts.append(
                "\nIntegrate all of the above translated methods into the complete "
                f"{target_lang} class. Keep their logic intact; adjust for cohesion as needed."
            )

        prompt_parts.append(f"\nCode to translate:")
        if fragment.docstring:
            prompt_parts.append(f"# Documentation: {fragment.docstring}")

        prompt_parts.append(f"```{source_lang}")
        prompt_parts.append(fragment.source_code)
        prompt_parts.append("```")

        if fragment.signature:
            prompt_parts.append(f"\nOriginal signature: {fragment.signature}")

        prompt_parts.append("\nTranslated code:")

        return "\n".join(prompt_parts)

    def generate(self, prompt: str) -> str:
        """Generate a completion from a freeform text prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_llm(messages)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the OpenAI-compatible API with retry-backoff for rate limits."""
        logger = logging.getLogger("codemorph.translator.llm_client")
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                )

                if not response.choices:
                    raise RuntimeError("LLM API returned empty choices")
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM API returned None content")
                # Strip markdown code blocks that LLMs often add despite instructions
                return strip_markdown_code_blocks(content)

            except RateLimitError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16, 32 seconds
                    logger.info(f"Rate limited (429), retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"LLM API call failed after {max_retries} rate-limit retries: {e}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"LLM API call failed: {e}")

    def _format_errors(self, errors: list[str]) -> str:
        """Format errors for display in prompt."""
        if not errors:
            return "No errors"

        formatted = []
        for i, error in enumerate(errors, 1):
            formatted.append(f"{i}. {error}")

        return "\n".join(formatted)


def create_llm_client(config: LLMConfig, conversation_log_dir: Path | None = None):
    """Factory function to create the appropriate LLM client based on provider."""
    if config.provider == LLMProvider.OPENROUTER:
        return OpenRouterClient(config, conversation_log_dir, base_url="openrouter")
    elif config.provider == LLMProvider.OPENAI:
        return OpenRouterClient(config, conversation_log_dir, base_url=None)
    elif config.provider == LLMProvider.OLLAMA:
        return OllamaClient(config, conversation_log_dir)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
