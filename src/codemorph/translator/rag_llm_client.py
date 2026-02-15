"""
RAG-Enhanced LLM Client

Implements Section 17.1 and 17.2 of the CodeMorph plan:
- Context Management with RAG
- Two-tier retrieval (Bootstrap + Snowball)
- Expert System prompt engineering
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config.models import CodeFragment
from ..knowledge.vector_store import VectorStore
from .llm_client import LLMConversation, OllamaClient

logger = logging.getLogger(__name__)


class RAGEnhancedLLMClient:
    """
    LLM Client with RAG-based style retrieval.

    Implements the two-tier retrieval strategy:
    - Hard Context: Dependency signatures
    - Soft Context: Style examples from vector store
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_store: Optional[VectorStore] = None,
        enable_rag: bool = True,
    ):
        """
        Initialize RAG-enhanced LLM client.

        Args:
            ollama_client: Base Ollama client
            vector_store: Vector store for style examples
            enable_rag: Whether to use RAG retrieval
        """
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.enable_rag = enable_rag

    def translate_fragment(
        self,
        fragment: CodeFragment,
        source_lang: str,
        source_version: str,
        target_lang: str,
        target_version: str,
        feature_mapping_instructions: Optional[List[str]] = None,
        dependency_signatures: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, LLMConversation]:
        """
        Translate fragment with RAG-enhanced prompting.

        Implements Section 17.2 prompt structure:
        1. Role Definition
        2. Task & Constraints
        3. Context Injection (Hard Context - dependencies)
        4. Style Examples (Soft Context - RAG)
        5. Source Input
        """
        conversation = LLMConversation()
        conversation.metadata = {
            "fragment_id": fragment.id,
            "fragment_type": fragment.fragment_type.value,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "rag_enabled": self.enable_rag,
        }

        # 1. Role Definition (System Prompt)
        system_prompt = self._create_expert_system_prompt(
            source_lang, source_version, target_lang, target_version
        )
        conversation.add_message("system", system_prompt)

        # 2-5. Build enhanced user prompt with RAG
        user_prompt = self._build_rag_enhanced_prompt(
            fragment,
            source_lang,
            target_lang,
            feature_mapping_instructions,
            dependency_signatures,
            context,
        )
        conversation.add_message("user", user_prompt)

        # Call LLM
        response = self.ollama_client._call_llm(conversation.messages)
        conversation.add_message("assistant", response)

        # Save conversation
        if self.ollama_client.conversation_log_dir:
            conversation.save(self.ollama_client.conversation_log_dir)

        return response, conversation

    def _create_expert_system_prompt(
        self,
        source_lang: str,
        source_version: str,
        target_lang: str,
        target_version: str,
    ) -> str:
        """
        Create expert system prompt (Section 17.2 - Role Definition).

        This establishes the LLM's role as a specialized migration architect.
        """
        return f"""You are a Senior {target_lang} Migration Architect with deep expertise in:
- Code translation from {source_lang} {source_version} to {target_lang} {target_version}
- Type safety and compile-time verification
- Maintaining semantic equivalence across languages
- Following {target_lang} best practices and idioms

Your core competencies:
1. EXACT BEHAVIOR PRESERVATION: Translated code must produce identical I/O for all inputs
2. TYPE SAFETY: All types must be correctly mapped and verified
3. IDIOMATIC CODE: Follow {target_lang} naming conventions and patterns
4. COMPILE CORRECTNESS: Generated code must compile without errors
5. MAINTAINABILITY: Produce clean, readable code that matches project style

When you encounter untranslatable constructs:
- Document the limitation clearly in comments
- Provide the closest semantic equivalent

CRITICAL OUTPUT FORMAT:
- Return ONLY raw {target_lang} code
- NO markdown, NO backticks, NO ```{target_lang.lower()} blocks
- Start directly with code (e.g., "import" or "public class")
- The code will be automatically compiled - markdown would cause errors
"""

    def _build_rag_enhanced_prompt(
        self,
        fragment: CodeFragment,
        source_lang: str,
        target_lang: str,
        feature_mapping_instructions: Optional[List[str]],
        dependency_signatures: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build the main translation prompt with RAG enhancements.

        Implements Section 17.2 prompt structure:
        2. Task & Constraints
        3. Context Injection (dependency signatures)
        4. Style Examples (RAG retrieval)
        5. Source Input
        """
        prompt_parts = []

        # 2. Task & Constraints
        prompt_parts.append(f"TASK: Translate this {source_lang} code to {target_lang}")

        # Feature mapping constraints
        if feature_mapping_instructions:
            prompt_parts.append("\nTRANSLATION CONSTRAINTS:")
            for instruction in feature_mapping_instructions:
                prompt_parts.append(f"  • {instruction}")

        # 3. Context Injection (Hard Context - Dependency Signatures)
        if dependency_signatures:
            prompt_parts.append("\nAVAILABLE DEPENDENCIES:")
            prompt_parts.append("The following functions/classes are already translated and available:")
            for sig in dependency_signatures:
                prompt_parts.append(f"  {sig}")

        # 4. Style Examples (Soft Context - RAG)
        if self.enable_rag and self.vector_store:
            style_examples = self._retrieve_style_examples(
                fragment, target_lang
            )

            if style_examples:
                prompt_parts.append("\nSTYLE REFERENCE EXAMPLES:")
                prompt_parts.append("Use these examples from the current project as style guides:")

                for i, example in enumerate(style_examples, 1):
                    prompt_parts.append(f"\nExample {i} ({example.category}):")
                    prompt_parts.append(f"```{example.language}")
                    prompt_parts.append(example.code)
                    prompt_parts.append("```")

                    if example.verified:
                        prompt_parts.append("  ✓ This is verified, high-quality code")

        # 5. Source Input
        prompt_parts.append("\nSOURCE CODE TO TRANSLATE:")

        if fragment.docstring:
            prompt_parts.append(f"// Original documentation:")
            prompt_parts.append(f"// {fragment.docstring}")

        if fragment.signature:
            prompt_parts.append(f"// Original signature: {fragment.signature}")

        prompt_parts.append(f"\n```{source_lang}")
        prompt_parts.append(fragment.source_code)
        prompt_parts.append("```")

        # Output instruction
        prompt_parts.append(f"\nProduce the {target_lang} translation:")

        return "\n".join(prompt_parts)

    def _retrieve_style_examples(
        self,
        fragment: CodeFragment,
        target_lang: str,
    ) -> List[Any]:
        """
        Retrieve style examples from vector store (RAG).

        Implements the Snowball Layer logic:
        - Prefer verified translations from Phase 2
        - Fall back to bootstrap golden examples
        """
        if not self.vector_store:
            return []

        # Infer category from fragment
        category = self._infer_category(fragment)

        # Query vector store
        try:
            # First, try to get verified snowball examples
            style_examples = self.vector_store.get_style_examples(
                language=target_lang,
                category=category,
                prefer_verified=True,
            )

            # Limit to top 2 examples to avoid context bloat
            return style_examples[:2]

        except Exception as e:
            logger.warning(f"Failed to retrieve style examples: {e}")
            return []

    def _infer_category(self, fragment: CodeFragment) -> str:
        """Infer category from fragment for RAG retrieval."""
        code = fragment.source_code.lower()

        # Map common patterns to categories
        if "error" in code or "exception" in code or "try" in code:
            return "error_handling"
        elif "class" in code:
            return "class_definition"
        elif "def " in code or "function" in code:
            return "function"
        elif "for " in code or "while " in code:
            return "iteration"
        elif "if " in code:
            return "conditional"
        else:
            return fragment.fragment_type.value

    def refine_translation(
        self,
        original_code: str,
        translated_code: str,
        error_message: str,
        conversation: LLMConversation,
        instruction: str = "Fix the compilation errors",
    ) -> str:
        """
        Refine translation with error feedback.

        Uses the existing conversation for context continuity.
        """
        # Use the base client's refinement logic
        response, _ = self.ollama_client.refine_translation(
            original_code=original_code,
            translated_code=translated_code,
            errors=[error_message],
            conversation=conversation,
            instruction=instruction,
        )

        return response

    def index_verified_translation(
        self,
        fragment_id: str,
        target_lang: str,
        translated_code: str,
        fragment_type: str,
    ):
        """
        Index a verified translation in the vector store (Snowball Layer).

        This should be called after Phase 2 successful verification.
        """
        if not self.vector_store:
            return

        try:
            self.vector_store.index_verified_translation(
                fragment_id=fragment_id,
                language=target_lang,
                code=translated_code,
                fragment_type=fragment_type,
            )

            logger.info(f"Indexed verified translation: {fragment_id}")

        except Exception as e:
            logger.warning(f"Failed to index verified translation: {e}")
