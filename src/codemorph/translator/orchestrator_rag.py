"""
RAG-Enhanced Phase 2 Orchestrator

Extends the Phase 2 orchestrator with RAG support:
- Initializes vector store (Bootstrap + Snowball layers)
- Uses RAG-enhanced LLM client
- Indexes verified translations for future style retrieval
"""

import logging
from pathlib import Path
from typing import Dict

from rich.console import Console

from ..config.models import (
    CodeFragment,
    CodeMorphConfig,
    TranslatedFragment,
    TranslationStatus,
)
from ..knowledge.vector_store import VectorStore
from ..state.persistence import StatePersistence
from .llm_client import create_llm_client, LLMConfig
from .rag_llm_client import RAGEnhancedLLMClient
from .orchestrator import Phase2Orchestrator

logger = logging.getLogger(__name__)
console = Console()


class RAGEnhancedPhase2Orchestrator(Phase2Orchestrator):
    """
    Phase 2 Orchestrator with RAG support.

    Extends the base orchestrator to:
    1. Initialize vector store with bootstrap examples
    2. Use RAG-enhanced LLM client for style-aware translations
    3. Index verified translations (Snowball Layer)
    """

    def __init__(self, config: CodeMorphConfig, state: StatePersistence):
        """Initialize orchestrator with RAG support."""
        # Call parent init
        super().__init__(config, state)

        # Initialize vector store if RAG is enabled
        if config.rag.enabled:
            logger.info("Initializing RAG vector store...")

            # Create vector store directory
            vector_store_dir = state.state_dir / "vector_store"
            vector_store_dir.mkdir(parents=True, exist_ok=True)

            # Initialize vector store
            self.vector_store = VectorStore(storage_dir=vector_store_dir)

            # Load bootstrap examples if available
            if config.rag.bootstrap_dir:
                bootstrap_dir = Path(config.rag.bootstrap_dir)
                if bootstrap_dir.exists():
                    logger.info(f"Loading bootstrap examples from {bootstrap_dir}")
                    self.vector_store.load_bootstrap_examples(bootstrap_dir)
                else:
                    logger.warning(f"Bootstrap directory not found: {bootstrap_dir}")

            # Create RAG-enhanced LLM client
            self.rag_llm_client = RAGEnhancedLLMClient(
                ollama_client=self.llm_client,
                vector_store=self.vector_store,
                enable_rag=True,
            )

            # Log vector store statistics
            stats = self.vector_store.get_statistics()
            logger.info(f"Vector store initialized: {stats}")

        else:
            logger.info("RAG disabled - using standard LLM client")
            self.vector_store = None
            self.rag_llm_client = None

    def _translate_fragment(self, fragment: CodeFragment) -> TranslatedFragment:
        """
        Translate fragment with optional RAG enhancement.

        Overrides parent method to:
        1. Use RAG-enhanced client if available
        2. Index verified translations in vector store
        """
        # Get dependency signatures for context
        dep_signatures = self._get_dependency_signatures(fragment)

        # Get feature mapping instructions
        feature_instructions = self.feature_mapper.get_instructions_for_fragment(
            fragment,
            self.config.project.source.language,
            self.config.project.target.language,
        )

        # Initialize translated fragment
        translated = TranslatedFragment(
            fragment=fragment,
            status=TranslationStatus.IN_PROGRESS,
        )

        console.print(
            f"[cyan]Translating:[/cyan] {fragment.name} ({fragment.fragment_type.value})"
        )

        # Use RAG-enhanced client if available
        if self.rag_llm_client and self.config.rag.enabled:
            translated = self._translate_with_rag(
                fragment,
                translated,
                dep_signatures,
                feature_instructions,
            )
        else:
            # Fall back to standard translation
            translated = super()._translate_fragment(fragment)

        # If translation succeeded and was verified, index it (Snowball Layer)
        if (
            self.vector_store
            and translated.status == TranslationStatus.TYPE_VERIFIED
            and translated.target_code
        ):
            self._index_verified_translation(translated)

        return translated

    def _translate_with_rag(
        self,
        fragment: CodeFragment,
        translated: TranslatedFragment,
        dep_signatures: list[str],
        feature_instructions: list[str],
    ) -> TranslatedFragment:
        """
        Translate using RAG-enhanced LLM client.

        This uses the two-tier retrieval strategy:
        - Hard Context: Dependency signatures
        - Soft Context: Style examples from vector store
        """
        max_retries = self.config.translation.max_retries_type_check
        retry_count = 0

        while retry_count <= max_retries:
            try:
                if retry_count == 0:
                    # Initial translation with RAG
                    target_code, conversation = self.rag_llm_client.translate_fragment(
                        fragment=fragment,
                        source_lang=self.config.project.source.language.value,
                        source_version=self.config.project.source.version,
                        target_lang=self.config.project.target.language.value,
                        target_version=self.config.project.target.version,
                        feature_mapping_instructions=feature_instructions,
                        dependency_signatures=dep_signatures,
                    )
                else:
                    # Refinement
                    target_code = self.rag_llm_client.refine_translation(
                        original_code=fragment.source_code,
                        translated_code=translated.target_code or "",
                        error_message="\n".join(translated.compilation_errors or []),
                        conversation=translated.llm_conversation,
                    )

                translated.target_code = target_code
                if retry_count == 0:
                    translated.llm_conversation = conversation

                # Validate feature mapping
                if not self._validate_feature_mapping(fragment, target_code):
                    console.print(
                        f"  [yellow]⚠[/yellow] Feature mapping failed (retry {retry_count + 1}/{max_retries})"
                    )
                    retry_count += 1
                    continue

                # Try to compile
                compilation_success, errors = self._compile_fragment(
                    fragment, target_code
                )

                if not compilation_success:
                    translated.compilation_errors = errors
                    console.print(
                        f"  [yellow]⚠[/yellow] Compilation failed (retry {retry_count + 1}/{max_retries})"
                    )
                    for error in errors[:3]:
                        console.print(f"    {error}")
                    retry_count += 1
                    continue

                # Compilation succeeded
                translated.status = TranslationStatus.COMPILED
                translated.compilation_errors = []
                console.print(f"  [green]✓[/green] Compiled successfully")

                # Try type compatibility check
                if self.config.verification.equivalence_check:
                    type_compatible = self._check_type_compatibility(
                        fragment, target_code
                    )
                    if type_compatible:
                        translated.status = TranslationStatus.TYPE_VERIFIED
                        console.print(f"  [green]✓[/green] Type compatibility verified")
                    else:
                        console.print(
                            f"  [yellow]⚠[/yellow] Type compatibility check failed"
                        )

                # Success!
                break

            except Exception as e:
                console.print(f"  [red]✗[/red] Error: {e}")
                if not translated.compilation_errors:
                    translated.compilation_errors = []
                translated.compilation_errors.append(str(e))
                retry_count += 1

        # Handle exhausted retries
        if retry_count > max_retries:
            if self.config.translation.allow_mocking:
                translated.status = TranslationStatus.MOCKED
                translated.is_mocked = True
                translated.mock_reason = "Exceeded retry limit"
                translated.target_code = self._generate_mock(fragment)
                console.print(f"  [yellow]⚠[/yellow] Mocked (exceeded retries)")
            else:
                translated.status = TranslationStatus.FAILED
                console.print(f"  [red]✗[/red] Translation failed")

        return translated

    def _index_verified_translation(self, translated: TranslatedFragment):
        """
        Index a verified translation in the vector store (Snowball Layer).

        This allows later translations to learn from earlier successful ones.
        """
        try:
            self.vector_store.index_verified_translation(
                fragment_id=translated.fragment.id,
                language=self.config.project.target.language.value,
                code=translated.target_code,
                fragment_type=translated.fragment.fragment_type.value,
            )

            logger.debug(
                f"Indexed verified translation: {translated.fragment.name}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to index {translated.fragment.name}: {e}"
            )

    def _get_dependency_signatures(self, fragment: CodeFragment) -> list[str]:
        """
        Get dependency signatures for context injection (Hard Context).

        This is the "Dependency Injection" part of Section 17.1.
        """
        signatures = []

        for dep_id in fragment.dependencies:
            # Check if dependency has been translated
            if dep_id in self.state.translated_fragments:
                translated_dep = self.state.translated_fragments[dep_id]

                if translated_dep.target_code and translated_dep.fragment.signature:
                    # Extract just the signature, not the full implementation
                    signatures.append(translated_dep.fragment.signature)

        return signatures


def run_phase2_translation_with_rag(
    config: CodeMorphConfig,
    state: StatePersistence,
) -> Dict[str, TranslatedFragment]:
    """
    Run Phase 2 with RAG enhancement.

    This is the entry point for RAG-enhanced translation.
    Falls back to standard translation if RAG is disabled.
    """
    if config.rag.enabled:
        logger.info("Running Phase 2 with RAG enhancement")
        orchestrator = RAGEnhancedPhase2Orchestrator(config, state)
    else:
        logger.info("Running Phase 2 without RAG (standard mode)")
        orchestrator = Phase2Orchestrator(config, state)

    return orchestrator.run()
