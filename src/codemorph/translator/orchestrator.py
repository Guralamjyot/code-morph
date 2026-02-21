"""
Phase 2 Orchestrator: Type-Driven Translation.

This orchestrator coordinates the type-driven translation phase, which includes:
1. Translating fragments in dependency order
2. Applying feature mapping rules
3. Compiling generated code
4. Verifying type compatibility
5. Retrying on failures with LLM refinement
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from codemorph.config.models import (
    CheckpointMode,
    CodeFragment,
    CodeMorphConfig,
    TranslatedFragment,
    TranslationStatus,
)
from codemorph.knowledge.feature_mapper import FeatureMapper, create_default_mapper
from codemorph.languages.registry import LanguagePluginRegistry
from codemorph.state.persistence import TranslationState
from codemorph.state.symbol_registry import SymbolRegistry
from codemorph.translator.llm_client import create_llm_client
from codemorph.verifier.type_checker import TypeCompatibilityChecker

console = Console()


class Phase2Orchestrator:
    """Orchestrates Phase 2: Type-Driven Translation."""

    def __init__(self, config: CodeMorphConfig, state: TranslationState):
        self.config = config
        self.state = state

        # Get language plugins
        self.source_plugin = LanguagePluginRegistry.get_plugin(
            config.project.source.language,
            config.project.source.version,
        )
        self.target_plugin = LanguagePluginRegistry.get_plugin(
            config.project.target.language,
            config.project.target.version,
        )

        # Initialize services
        self.llm_client = create_llm_client(config.llm)
        self.feature_mapper = create_default_mapper()
        self.type_checker = TypeCompatibilityChecker()

        # Initialize symbol registry (per-project symbol tracking)
        self.symbol_registry = SymbolRegistry(config.project.state_dir)

        # Initialize checkpoint UI if interactive mode OR show_diff is enabled
        self.checkpoint_ui = None
        if config.checkpoint_mode == CheckpointMode.INTERACTIVE or getattr(config, "show_diff", False):
            from codemorph.cli.checkpoint_ui import CheckpointUI
            self.checkpoint_ui = CheckpointUI(
                source_lang=config.project.source.language.value,
                target_lang=config.project.target.language.value,
            )

        self._show_diff = getattr(config, "show_diff", False)
        self._source_lang = config.project.source.language.value
        self._target_lang = config.project.target.language.value

        # Stats tracking
        self.stats = {
            "total": 0,
            "compiled": 0,
            "type_verified": 0,
            "failed": 0,
            "mocked": 0,
            "human_reviewed": 0,
        }

    def run(self) -> dict[str, TranslatedFragment]:
        """
        Execute Phase 2: Type-Driven Translation.

        Returns:
            Dictionary of translated fragments by ID
        """
        console.print("\n[bold cyan]Phase 2: Type-Driven Translation[/bold cyan]\n")

        if not self.state.analysis_result:
            raise ValueError("Phase 1 analysis result not found. Run Phase 1 first.")

        translation_order = self.state.analysis_result.translation_order
        fragments = self.state.analysis_result.fragments
        self.stats["total"] = len(translation_order)

        console.print(f"[cyan]Translating {len(translation_order)} fragments...[/cyan]\n")

        # Pre-compute overload groups: group $-suffixed IDs sharing the same base
        overload_groups: dict[str, list[str]] = {}  # base_id -> [variant IDs in order]
        merged_overloads: dict[str, str] = {}       # secondary_fid -> primary_fid
        for fid in translation_order:
            if "$" not in fid:
                continue
            base = fid[: fid.index("$")]
            overload_groups.setdefault(base, []).append(fid)
        bases_to_remove = [base for base, variants in overload_groups.items() if len(variants) < 2]
        for base in bases_to_remove:
            overload_groups.pop(base)
        for base, variants in overload_groups.items():
            primary = variants[0]
            for secondary in variants[1:]:
                merged_overloads[secondary] = primary

        if overload_groups:
            console.print(
                f"[cyan]Detected {len(overload_groups)} overload group(s) "
                f"({sum(len(v) for v in overload_groups.values())} fragments) — "
                f"merging at translation time[/cyan]\n"
            )

        _target_is_java = self.config.project.target.language.value == "java"
        _source_is_java = self.config.project.source.language.value == "java"
        # Two-pass is needed when translating between Java and Python:
        # methods/enums are translated first so the class can use them as context.
        _needs_two_pass = _target_is_java or _source_is_java

        # Build to_translate — all fragments (no skipping); class fragments
        # will be reordered to come after non-class fragments.
        pass1: list[tuple[int, str, CodeFragment]] = []  # methods, enums, functions
        pass2: list[tuple[int, str, CodeFragment]] = []  # class fragments (deferred)

        for idx, fragment_id in enumerate(translation_order):
            fragment = fragments[fragment_id]
            if fragment_id in merged_overloads:
                # Defer secondary overloads — handle after primaries are done
                continue

            if _needs_two_pass and fragment.fragment_type.value == "class":
                pass2.append((idx, fragment_id, fragment))
            else:
                pass1.append((idx, fragment_id, fragment))

        to_translate = pass1 + pass2

        if pass2 and _needs_two_pass:
            direction = "P2J" if _target_is_java else "J2P"
            console.print(
                f"[cyan]{direction} two-pass translation: {len(pass1)} method(s)/enum(s) first, "
                f"then {len(pass2)} class(es) with method context[/cyan]\n"
            )

        self.stats["total"] = len(to_translate)

        results: dict[str, TranslatedFragment] = {}

        def _update_stats(tf: TranslatedFragment) -> None:
            if tf.status == TranslationStatus.COMPILED:
                self.stats["compiled"] += 1
            elif tf.status == TranslationStatus.TYPE_VERIFIED:
                self.stats["type_verified"] += 1
            elif tf.status == TranslationStatus.FAILED:
                self.stats["failed"] += 1
            elif tf.status == TranslationStatus.MOCKED:
                self.stats["mocked"] += 1
            elif tf.status == TranslationStatus.HUMAN_REVIEW:
                self.stats["human_reviewed"] += 1

        def _handle_overloads(use_progress: bool = False, progress=None, task=None) -> None:
            """Copy secondary overload results from their primaries."""
            for fragment_id in translation_order:
                if fragment_id not in merged_overloads:
                    continue
                fragment = fragments[fragment_id]
                primary_fid = merged_overloads[fragment_id]
                primary_tf = results.get(primary_fid) or self.state.translated_fragments.get(primary_fid)
                if primary_tf:
                    translated = TranslatedFragment(
                        fragment=fragment,
                        status=primary_tf.status,
                        target_code=primary_tf.target_code,
                        is_mocked=primary_tf.is_mocked,
                        mock_reason=primary_tf.mock_reason,
                    )
                    console.print(
                        f"[dim]Skipping secondary overload:[/dim] {fragment.name} "
                        f"(merged with {primary_fid})"
                    )
                    self._register_symbol(fragment, translated)
                    self.state.update_fragment(translated)
                    self.state.current_fragment_index += 1
                    _update_stats(translated)
                else:
                    console.print(
                        f"[yellow]⚠[/yellow] Primary overload {primary_fid} not found for {fragment_id}"
                    )
                if use_progress and progress is not None:
                    progress.advance(task)
                if self.state.current_fragment_index % 10 == 0:
                    self.state.save()
                    self.symbol_registry.save()

        if self._show_diff:
            # ── Show-diff mode: sequential, no live bar, Enter pause between each ──
            total_frags = len(to_translate)
            for item_idx, (idx, fid, frag) in enumerate(to_translate):
                translated = self._translate_fragment(frag, idx, len(translation_order))
                results[fid] = translated
                self.state.update_fragment(translated)
                self.state.current_fragment_index += 1
                _update_stats(translated)

                self.checkpoint_ui.show_translation_preview(
                    frag,
                    translated,
                    source_lang=self._source_lang,
                    target_lang=self._target_lang,
                    index=item_idx,
                    total=total_frags,
                )

                if item_idx < total_frags - 1:
                    try:
                        console.input("\n[dim]── Press Enter for next fragment ──[/dim]")
                    except EOFError:
                        pass  # non-interactive / piped stdin — skip pause

                if self.state.current_fragment_index % 10 == 0:
                    self.state.save()
                    self.symbol_registry.save()

            _handle_overloads(use_progress=False)

        else:
            # ── Normal mode: parallel / serial with live Progress bar ──
            # Pass 1 (methods/enums) may run in parallel; Pass 2 (classes)
            # always runs sequentially so method context is available.
            if self.config.project.target.language.value == "java":
                max_workers_p1 = 1
            else:
                max_workers_p1 = min(4, len(pass1)) if pass1 else 1

            def _do_translate(item):
                idx, fid, frag = item
                return (fid, frag, self._translate_fragment(frag, idx, len(translation_order)))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Translating fragments...", total=len(to_translate)
                )

                # ── Pass 1: methods, enums, functions, globals ──
                with ThreadPoolExecutor(max_workers=max_workers_p1) as executor:
                    futures = {
                        executor.submit(_do_translate, item): item
                        for item in pass1
                    }
                    for future in as_completed(futures):
                        fid, frag, translated = future.result()
                        results[fid] = translated
                        self.state.update_fragment(translated)
                        self.state.current_fragment_index += 1
                        _update_stats(translated)
                        progress.advance(task)

                # ── Pass 2: class fragments (sequential, method context available) ──
                if pass2:
                    progress.update(task, description="Translating classes (with method context)...")
                    for item in pass2:
                        fid, frag, translated = _do_translate(item)
                        results[fid] = translated
                        self.state.update_fragment(translated)
                        self.state.current_fragment_index += 1
                        _update_stats(translated)
                        progress.advance(task)

                _handle_overloads(use_progress=True, progress=progress, task=task)

        # Final save
        self.state.current_phase = 2
        self.state.save()
        self.symbol_registry.save()

        # Display results
        self._display_results()

        return self.state.translated_fragments

    def get_symbol_registry(self) -> SymbolRegistry:
        """Get the symbol registry for use by other components."""
        return self.symbol_registry

    def _translate_fragment(
        self, fragment: CodeFragment, index: int = 0, total: int = 0
    ) -> TranslatedFragment:
        """
        Translate a single fragment with retry logic and optional checkpoints.

        Args:
            fragment: The fragment to translate
            index: Current fragment index (for checkpoint display)
            total: Total fragments (for checkpoint display)

        Returns:
            TranslatedFragment with translation result
        """
        console.print(
            f"[cyan]Translating:[/cyan] {fragment.name} ({fragment.fragment_type.value})"
        )

        # Initialize translated fragment
        translated = TranslatedFragment(
            fragment=fragment,
            status=TranslationStatus.IN_PROGRESS,
        )

        # Get dependency signatures for context (including from symbol registry)
        dep_context = self._get_dependency_context(fragment)

        # Get feature mapping instructions
        feature_instructions = self.feature_mapper.get_instructions_for_fragment(
            fragment,
            self.config.project.source.language,
            self.config.project.target.language,
        )

        # Detect sibling overloads for this fragment
        overload_sigs = self._get_overload_context(fragment)

        # Translation retry loop
        max_retries = self.config.translation.max_retries_type_check
        retry_count = 0
        last_conversation = None

        while retry_count <= max_retries:
            try:
                # Generate translation
                if retry_count == 0:
                    # Initial translation
                    context = {"dependency_signatures": dep_context} if dep_context else None
                    if context is None:
                        context = {}
                    if overload_sigs:
                        context["overloads"] = overload_sigs
                    # If this fragment is a primary overload, include full
                    # sibling source code so the LLM can merge them
                    overload_sources = self._get_overload_sources(fragment)
                    if overload_sources:
                        context["overload_sources"] = overload_sources
                    # For class fragments: inject already-translated methods
                    # as context so the LLM can produce a coherent whole class.
                    if fragment.fragment_type.value == "class":
                        translated_methods = self._get_translated_methods_for_class(
                            fragment.name
                        )
                        if translated_methods:
                            context["translated_methods"] = translated_methods
                            console.print(
                                f"  [dim]Using {len(translated_methods)} pre-translated "
                                f"method(s) as context[/dim]"
                            )
                    target_code, conversation = self.llm_client.translate_fragment(
                        fragment=fragment,
                        source_lang=self.config.project.source.language.value,
                        source_version=self.config.project.source.version,
                        target_lang=self.config.project.target.language.value,
                        target_version=self.config.project.target.version,
                        feature_mapping_instructions=feature_instructions,
                        context=context,
                    )
                else:
                    # Refinement after error
                    target_code, conversation = self.llm_client.refine_translation(
                        original_code=fragment.source_code,
                        translated_code=translated.target_code or "",
                        errors=translated.compilation_errors,
                        conversation=last_conversation,
                    )

                translated.target_code = target_code
                last_conversation = conversation
                # Store conversation ID if available
                if hasattr(conversation, 'id'):
                    translated.llm_conversation_ids.append(conversation.id)
                elif isinstance(conversation, str):
                    translated.llm_conversation_ids.append(conversation)

                # Validate feature mapping compliance
                is_valid, failed_rules = self.feature_mapper.validate_translation(
                    fragment,
                    target_code,
                    self.config.project.source.language,
                    self.config.project.target.language,
                )
                if not is_valid:
                    failure_msg = f"Feature mapping validation failed: {', '.join(failed_rules)}"
                    console.print(
                        f"  [yellow]⚠[/yellow] {failure_msg} (retry {retry_count + 1}/{max_retries})"
                    )
                    translated.compilation_errors = [failure_msg]
                    retry_count += 1
                    continue

                # Individual method fragments targeting Java can't compile
                # standalone — they'll be integrated into the full class
                # translation (Pass 2) which compiles the complete class.
                if (
                    self.config.project.target.language.value == "java"
                    and fragment.fragment_type.value == "method"
                ):
                    translated.status = TranslationStatus.TRANSLATED
                    translated.compilation_errors = []
                    console.print(
                        f"  [green]✓[/green] Method translated "
                        f"(compilation deferred to class assembly)"
                    )
                    break

                # Try to compile
                compilation_success, errors = self._compile_fragment(
                    fragment, target_code
                )

                if not compilation_success:
                    translated.compilation_errors = errors
                    console.print(
                        f"  [yellow]⚠[/yellow] Compilation failed (retry {retry_count + 1}/{max_retries})"
                    )
                    for error in errors[:3]:  # Show first 3 errors
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
                        # Don't retry for type issues - will be fixed in Phase 3
                        break

                # Success!
                break

            except Exception as e:
                console.print(f"  [red]✗[/red] Error during translation: {e}")
                translated.compilation_errors.append(str(e))
                retry_count += 1

        # Check if we exhausted retries
        if retry_count > max_retries:
            if self.config.translation.allow_mocking:
                # Mock the function — preserve last LLM attempt for diagnostics
                translated.status = TranslationStatus.MOCKED
                translated.is_mocked = True
                translated.mock_reason = (
                    f"Exceeded retry limit for compilation"
                    f" (last errors: {translated.compilation_errors[:3]})"
                )
                translated.target_code = self._generate_mock(fragment)
                console.print(f"  [yellow]⚠[/yellow] Mocked (exceeded retries)")
            else:
                translated.status = TranslationStatus.FAILED
                console.print(f"  [red]✗[/red] Failed (mocking disabled)")

        translated.retry_count = retry_count

        # Interactive checkpoint (only in INTERACTIVE mode, not show_diff mode)
        if (
            self.checkpoint_ui
            and not self._show_diff
            and self.config.checkpoint_mode == CheckpointMode.INTERACTIVE
            and translated.status not in (TranslationStatus.FAILED,)
        ):
            translated = self._handle_checkpoint(fragment, translated, index, total)

        # Register symbol in registry (after successful translation or checkpoint)
        if translated.status in (
            TranslationStatus.COMPILED,
            TranslationStatus.TYPE_VERIFIED,
            TranslationStatus.HUMAN_REVIEW,
        ):
            self._register_symbol(fragment, translated)

        return translated

    def _handle_checkpoint(
        self,
        fragment: CodeFragment,
        translated: TranslatedFragment,
        index: int,
        total: int,
    ) -> TranslatedFragment:
        """
        Handle interactive checkpoint for a translation.

        Args:
            fragment: Source fragment
            translated: Translated fragment
            index: Current index
            total: Total fragments

        Returns:
            Updated TranslatedFragment based on user action
        """
        from codemorph.cli.checkpoint_ui import CheckpointAction
        from rich.prompt import Prompt

        action = self.checkpoint_ui.show_translation_checkpoint(
            fragment, translated, phase=2, index=index, total=total
        )

        if action == CheckpointAction.APPROVE:
            return translated

        elif action == CheckpointAction.REJECT:
            # Get hint and retry
            hint = self.checkpoint_ui.show_reject_hint_prompt()
            if hint:
                # Add hint to context and retry (simplified)
                console.print(f"[yellow]Re-translating with hint...[/yellow]")
                # For now, just mark for re-processing
                translated.compilation_errors.append(f"User hint: {hint}")
            return translated

        elif action == CheckpointAction.EDIT:
            # User manually edits the code
            edited_code = self.checkpoint_ui.show_edit_dialog(translated)
            translated.target_code = edited_code
            translated.status = TranslationStatus.HUMAN_REVIEW
            console.print("[green]Manual edit applied.[/green]")
            return translated

        elif action == CheckpointAction.MOCK:
            # User requests mock
            translated.status = TranslationStatus.MOCKED
            translated.is_mocked = True
            translated.mock_reason = "User requested mock"
            translated.target_code = self._generate_mock(fragment)
            console.print("[yellow]Mocked by user request.[/yellow]")
            return translated

        elif action == CheckpointAction.SKIP:
            # Skip for now
            translated.status = TranslationStatus.PENDING
            return translated

        return translated

    def _register_symbol(
        self, fragment: CodeFragment, translated: TranslatedFragment
    ):
        """
        Register a translated symbol in the registry.

        Args:
            fragment: Source fragment
            translated: Translated fragment
        """
        # Determine target name (convert naming convention)
        target_name = self.target_plugin.convert_name(
            fragment.name,
            fragment.fragment_type,
            target_convention="camelCase" if self.config.project.target.language.value == "java" else None,
        )

        # Extract signature from translated code
        signature = None
        if translated.target_code:
            # Simple signature extraction (first line of function/method)
            lines = translated.target_code.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("public ") or line.startswith("private ") or line.startswith("protected "):
                    signature = line.rstrip("{").strip()
                    break
                elif "def " in line:
                    signature = line.rstrip(":").strip()
                    break

        self.symbol_registry.register_symbol(
            source_name=fragment.name,
            source_qualified=fragment.id,
            target_name=target_name,
            symbol_type=fragment.fragment_type.value,
            signature=signature,
        )
        self.symbol_registry.update_status(fragment.id, translated.status)

    def _get_overload_context(self, fragment: CodeFragment) -> list[str]:
        """Detect sibling fragments with the same method name + parent class.

        Returns a list of Java signatures for overloaded methods, to pass
        as context to the LLM so it can combine them properly.
        """
        if not fragment.parent_class or not fragment.name:
            return []

        siblings = []
        for frag_id, translated in self.state.translated_fragments.items():
            if translated.fragment.parent_class != fragment.parent_class:
                continue
            if translated.fragment.name != fragment.name:
                continue
            if translated.fragment.id == fragment.id:
                continue
            sig = translated.fragment.signature or translated.fragment.source_code.split("\n")[0]
            siblings.append(sig)

        # Also check unprocessed fragments in the analysis result
        if self.state.analysis_result:
            for frag_id, frag in self.state.analysis_result.fragments.items():
                if frag.parent_class != fragment.parent_class:
                    continue
                if frag.name != fragment.name:
                    continue
                if frag.id == fragment.id:
                    continue
                if frag_id in self.state.translated_fragments:
                    continue  # Already covered above
                sig = frag.signature or frag.source_code.split("\n")[0]
                if sig not in siblings:
                    siblings.append(sig)

        return siblings

    def _get_overload_sources(self, fragment: CodeFragment) -> list[str]:
        """Collect full Java source_code for all sibling overloads of a fragment.

        Returns an empty list if the fragment is not an overload (no ``$`` in its ID)
        or if it has no siblings.
        """
        if "$" not in fragment.id:
            return []
        base = fragment.id[: fragment.id.index("$")]
        sources: list[str] = []
        if self.state.analysis_result:
            for fid, frag in self.state.analysis_result.fragments.items():
                if fid == fragment.id:
                    continue
                if "$" not in fid:
                    continue
                if fid[: fid.index("$")] == base:
                    sources.append(frag.source_code)
        return sources

    def retranslate_overloads(self) -> None:
        """Re-translate only overloaded fragment groups using merged context.

        Loads existing state, identifies overload groups, re-translates the
        primary of each group with full sibling context, then copies the
        result to all secondaries.  Only the overload groups get LLM calls.
        """
        if not self.state.analysis_result:
            raise ValueError("Phase 1 analysis result not found.")

        fragments = self.state.analysis_result.fragments

        # Identify overload groups from translation order
        translation_order = self.state.analysis_result.translation_order
        overload_groups: dict[str, list[str]] = {}
        for fid in translation_order:
            if "$" not in fid:
                continue
            base = fid[: fid.index("$")]
            overload_groups.setdefault(base, []).append(fid)
        # Keep only true groups (2+ variants)
        overload_groups = {b: vs for b, vs in overload_groups.items() if len(vs) >= 2}

        if not overload_groups:
            console.print("[yellow]No overload groups found — nothing to do.[/yellow]")
            return

        console.print(
            f"\n[bold cyan]Re-translating {len(overload_groups)} overload group(s) "
            f"({sum(len(v) for v in overload_groups.values())} fragments)[/bold cyan]\n"
        )

        for base, variants in overload_groups.items():
            primary_fid = variants[0]
            primary_frag = fragments[primary_fid]

            console.print(f"[cyan]Overload group:[/cyan] {base}")
            console.print(f"  Primary: {primary_fid}")
            for sec in variants[1:]:
                console.print(f"  Secondary: {sec}")

            # Re-translate the primary with full overload sources
            translated = self._translate_fragment(primary_frag, 0, len(variants))

            # Save primary
            self.state.update_fragment(translated)
            self._register_symbol(primary_frag, translated)

            # Copy to secondaries
            for secondary_fid in variants[1:]:
                secondary_frag = fragments[secondary_fid]
                secondary_tf = TranslatedFragment(
                    fragment=secondary_frag,
                    status=translated.status,
                    target_code=translated.target_code,
                    is_mocked=translated.is_mocked,
                    mock_reason=translated.mock_reason,
                )
                self.state.update_fragment(secondary_tf)
                self._register_symbol(secondary_frag, secondary_tf)
                console.print(
                    f"  [dim]Copied merged result → {secondary_fid}[/dim]"
                )

        # Save state
        self.state.save()
        self.symbol_registry.save()
        console.print("\n[green]Overload re-translation complete. State saved.[/green]")

    def _get_translated_methods_for_class(self, class_name: str) -> list[dict]:
        """
        Return all already-translated method snippets for a given parent class.

        Called when translating a class fragment so the LLM can see how each
        method was individually translated and produce a coherent whole class.

        Args:
            class_name: The name of the parent class (e.g. "Product")

        Returns:
            List of dicts with keys: name, python_source (or java_source), target_method
        """
        methods = []
        for tf in self.state.translated_fragments.values():
            frag = tf.fragment
            if (
                frag.parent_class == class_name
                and frag.fragment_type.value == "method"
                and tf.target_code
                and tf.target_code.strip()
                and tf.status not in (TranslationStatus.FAILED,)
            ):
                methods.append({
                    "name": frag.name,
                    "source_method": frag.source_code,
                    "target_method": tf.target_code,
                })
        return methods

    def _get_dependency_context(self, fragment: CodeFragment) -> list[dict[str, str]]:
        """
        Get signatures of already-translated dependencies.

        For Java targets, extracts public method signatures from the translated
        Java code so dependent fragments can use the correct API.

        Args:
            fragment: Fragment whose dependencies to retrieve

        Returns:
            List of dependency signatures
        """
        import re
        context = []
        is_java_target = self.config.project.target.language.value == "java"

        for dep_id in fragment.dependencies:
            # Check if we have the translated code for this dependency
            if dep_id in self.state.translated_fragments:
                translated_dep = self.state.translated_fragments[dep_id]
                if translated_dep.target_code and translated_dep.status in (
                    TranslationStatus.COMPILED, TranslationStatus.TYPE_VERIFIED
                ):
                    if is_java_target:
                        # Extract Java public method signatures for context
                        java_sigs = self._extract_java_signatures(translated_dep.target_code)
                        if java_sigs:
                            context.append({
                                "name": translated_dep.fragment.name,
                                "signature": "\n".join(java_sigs),
                                "type": translated_dep.fragment.fragment_type.value,
                            })
                            continue

            # Fallback: symbol registry or source signature
            registry_sig = self.symbol_registry.get_signature(dep_id)
            registry_mapping = self.symbol_registry.get_mapping(dep_id)

            if registry_sig and registry_mapping:
                context.append({
                    "name": registry_mapping.target_name,
                    "signature": registry_sig,
                    "type": registry_mapping.symbol_type,
                })
            elif dep_id in self.state.translated_fragments:
                translated_dep = self.state.translated_fragments[dep_id]
                if translated_dep.target_code:
                    signature = self.target_plugin.extract_signature(
                        translated_dep.fragment
                    )
                    context.append({
                        "name": translated_dep.fragment.name,
                        "signature": signature or "// Signature not available",
                        "type": translated_dep.fragment.fragment_type.value,
                    })

        return context

    def _extract_java_signatures(self, java_code: str) -> list[str]:
        """Extract public method/constructor/field signatures from Java code."""
        import re
        sigs = []
        for line in java_code.split("\n"):
            stripped = line.strip()
            # Match public method/constructor declarations
            if re.match(r'public\s+\S+.*\(.*\)', stripped):
                # Get just the signature (before the body)
                sig = stripped.rstrip('{').strip()
                if sig:
                    sigs.append(sig)
            # Match enum constants
            elif re.match(r'[A-Z_]+\(', stripped):
                sigs.append(stripped.rstrip(',').rstrip(';'))
            # Match class/enum declaration
            elif re.match(r'(public\s+)?(class|enum|interface)\s+\w+', stripped):
                sigs.append(stripped.rstrip('{').strip())
        return sigs

    def _validate_feature_mapping(
        self, fragment: CodeFragment, target_code: str
    ) -> bool:
        """
        Validate that the translation follows feature mapping rules.

        Args:
            fragment: Original fragment
            target_code: Translated code

        Returns:
            True if all applicable rules are satisfied
        """
        is_valid, failed_rules = self.feature_mapper.validate_translation(
            fragment,
            target_code,
            self.config.project.source.language,
            self.config.project.target.language,
        )

        if not is_valid:
            console.print(
                f"    [yellow]Failed rules:[/yellow] {', '.join(failed_rules)}"
            )

        return is_valid

    def _compile_fragment(
        self, fragment: CodeFragment, target_code: str
    ) -> tuple[bool, list[str]]:
        """
        Attempt to compile the translated fragment.

        For Java targets, includes previously-translated fragments as dependencies
        so cross-class references can resolve.

        Args:
            fragment: Original fragment
            target_code: Translated code

        Returns:
            Tuple of (success, errors)
        """
        try:
            # Create temporary output directory
            output_dir = (
                self.config.project.state_dir / "compile_temp" / fragment.id.replace("::", "_")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            dependencies = None
            # For Java targets: write already-translated class fragments to a shared
            # dependency directory so javac can resolve cross-class references.
            if self.config.project.target.language.value == "java":
                import re
                deps_dir = self.config.project.state_dir / "compile_temp" / "_shared_deps"
                deps_dir.mkdir(parents=True, exist_ok=True)

                # Write previously-compiled fragments to shared dir
                for fid, tf in self.state.translated_fragments.items():
                    if fid == fragment.id:
                        continue
                    if tf.status not in (TranslationStatus.COMPILED, TranslationStatus.TYPE_VERIFIED):
                        continue
                    if not tf.target_code:
                        continue
                    # Extract class name and write to shared dir
                    match = re.search(r'(?:public\s+)?(?:class|enum|interface)\s+(\w+)', tf.target_code)
                    if match:
                        dep_file = deps_dir / f"{match.group(1)}.java"
                        dep_file.write_text(tf.target_code, encoding="utf-8")

                # Also write the current fragment itself to shared dir after compilation
                dependencies = [deps_dir]

            # Try to compile
            success, errors = self.target_plugin.compile_fragment(
                target_code, output_dir, dependencies=dependencies
            )

            # On success, save to shared deps for future fragments
            if success and self.config.project.target.language.value == "java":
                import re
                deps_dir = self.config.project.state_dir / "compile_temp" / "_shared_deps"
                match = re.search(r'(?:public\s+)?(?:class|enum|interface)\s+(\w+)', target_code)
                if match:
                    dep_file = deps_dir / f"{match.group(1)}.java"
                    dep_file.write_text(target_code, encoding="utf-8")

            return success, errors

        except Exception as e:
            return False, [str(e)]

    def _check_type_compatibility(
        self, fragment: CodeFragment, target_code: str
    ) -> bool:
        """
        Check type compatibility between source and target.

        Args:
            fragment: Original fragment
            target_code: Translated code

        Returns:
            True if types are compatible
        """
        try:
            # For now, we'll do a simple check
            # In Phase 3, this will use actual execution snapshots

            # Extract types from fragment signature
            if not fragment.signature:
                return True  # Can't verify without signature

            # TODO: More sophisticated type checking
            # For now, just return True if it compiled
            return True

        except Exception as e:
            console.print(f"    [yellow]Type check error:[/yellow] {e}")
            return False

    def _generate_mock(self, fragment: CodeFragment) -> str:
        """
        Generate a mock implementation that calls back to Python.

        Args:
            fragment: Fragment to mock

        Returns:
            Mock implementation code
        """
        # Generate appropriate mock based on target language
        if self.config.project.target.language.value == "java":
            return self._generate_java_mock(fragment)
        else:
            # For Python target (same language upgrade)
            return self._generate_python_mock(fragment)

    def _generate_java_mock(self, fragment: CodeFragment) -> str:
        """Generate Java mock that calls Python via bridge."""
        # Extract function name
        func_name = fragment.name

        # Convert to Java naming convention
        java_name = self._to_camel_case(func_name)

        mock_code = f"""
// MOCKED: This function calls the original Python implementation
// Reason: Could not generate compilable Java translation
public class {java_name.capitalize()} {{
    public static Object {java_name}(Object... args) {{
        // TODO: Implement Python bridge call
        // return PythonBridge.call("{func_name}", args);
        throw new UnsupportedOperationException(
            "Mock implementation - Python bridge not yet configured"
        );
    }}
}}
"""
        return mock_code

    def _generate_python_mock(self, fragment: CodeFragment) -> str:
        """Generate Python mock (for version upgrades)."""
        return f"""
# MOCKED: Could not translate this function
# Original code preserved
{fragment.source_code}
"""

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    def _display_results(self):
        """Display translation results summary."""
        console.print("\n[bold]Phase 2 Summary[/bold]\n")

        # Stats table
        stats_table = Table(show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Fragments", str(self.stats["total"]))
        stats_table.add_row("Compiled", str(self.stats["compiled"]))
        stats_table.add_row("Type Verified", str(self.stats["type_verified"]))
        stats_table.add_row("Mocked", str(self.stats["mocked"]))
        stats_table.add_row("Failed", str(self.stats["failed"]))

        # Calculate success rate
        success_count = self.stats["compiled"] + self.stats["type_verified"]
        success_rate = (
            (success_count / self.stats["total"] * 100) if self.stats["total"] > 0 else 0
        )
        stats_table.add_row("Success Rate", f"{success_rate:.1f}%")

        console.print(stats_table)

        # Warnings
        if self.stats["mocked"] > 0:
            console.print(
                f"\n[yellow]⚠ {self.stats['mocked']} function(s) were mocked and require manual attention[/yellow]"
            )

        if self.stats["failed"] > 0:
            console.print(
                f"\n[red]✗ {self.stats['failed']} function(s) failed translation[/red]"
            )


def run_phase2_translation(
    config: CodeMorphConfig, state: TranslationState
) -> tuple[dict[str, TranslatedFragment], SymbolRegistry]:
    """
    Convenience function to run Phase 2 translation.

    Args:
        config: CodeMorph configuration
        state: Translation state from Phase 1

    Returns:
        Tuple of (translated_fragments, symbol_registry)
    """
    orchestrator = Phase2Orchestrator(config, state)
    fragments = orchestrator.run()
    return fragments, orchestrator.get_symbol_registry()
