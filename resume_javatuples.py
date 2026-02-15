#!/usr/bin/env python3.13
"""
Resume script for javatuples translation.
Picks up Phase 2 from where it stopped (870/905 fragments),
runs Phase 3 verification, then Phase 4 assembly.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Ensure we're in the right directory
os.chdir("/workspace/code-agent/code-convert")

# Set up Java
java_path = Path("/workspace/persistent/java/jdk-17.0.2")
if java_path.exists():
    os.environ["JAVA_HOME"] = str(java_path)
    os.environ["PATH"] = f"{java_path / 'bin'}:{os.environ.get('PATH', '')}"

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("resume_javatuples")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from codemorph.config.loader import load_config_from_yaml
from codemorph.config.models import TranslationStatus, CheckpointMode
from codemorph.state.persistence import TranslationState
from codemorph.state.symbol_registry import SymbolRegistry
from codemorph.translator.orchestrator import Phase2Orchestrator

console = Console()


def resume_phase2(state: TranslationState, cfg):
    """Resume Phase 2 by translating only the remaining fragments."""
    already_translated = set(state.translated_fragments.keys())
    translation_order = state.analysis_result.translation_order
    all_fragment_ids = set(state.analysis_result.fragments.keys())

    remaining = [fid for fid in translation_order if fid not in already_translated]
    # Also catch any fragments not in the translation_order but in fragments dict
    extra = all_fragment_ids - set(translation_order) - already_translated
    remaining.extend(sorted(extra))

    console.print(f"\n[bold cyan]Phase 2 Resume[/bold cyan]")
    console.print(f"  Already translated: {len(already_translated)}")
    console.print(f"  Remaining: {len(remaining)}")
    console.print(f"  Total fragments: {len(all_fragment_ids)}")

    if not remaining:
        console.print("[green]Phase 2 already complete! All fragments translated.[/green]")
        return state.translated_fragments

    # Create the orchestrator (initializes LLM client, plugins, etc.)
    orchestrator = Phase2Orchestrator(cfg, state)

    fragments = state.analysis_result.fragments

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Translating remaining fragments...", total=len(remaining))

        for idx, fragment_id in enumerate(remaining):
            if fragment_id not in fragments:
                console.print(f"[yellow]Warning: fragment {fragment_id} not in analysis, skipping[/yellow]")
                progress.advance(task)
                continue

            fragment = fragments[fragment_id]
            progress.update(task, description=f"Translating: {fragment.name}")

            try:
                translated = orchestrator._translate_fragment(fragment, idx, len(remaining))
                state.update_fragment(translated)
                state.current_fragment_index += 1

                status_color = "green" if translated.status == TranslationStatus.TYPE_VERIFIED else "yellow"
                console.print(
                    f"  [{status_color}]{translated.status.value}[/{status_color}] {fragment_id}"
                )
            except Exception as e:
                console.print(f"  [red]Error translating {fragment_id}: {e}[/red]")
                logger.exception(f"Error translating {fragment_id}")

            progress.advance(task)

            # Save every 5 fragments
            if (idx + 1) % 5 == 0:
                state.save()
                orchestrator.symbol_registry.save()

    # Final save
    state.current_phase = 2
    state.save()
    orchestrator.symbol_registry.save()

    # Summary
    statuses = {}
    for fid, fdata in state.translated_fragments.items():
        s = fdata.status.value
        statuses[s] = statuses.get(s, 0) + 1
    console.print(f"\n[bold]Phase 2 Complete:[/bold] {statuses}")

    return state.translated_fragments


def run_phase3(state: TranslationState, cfg, translated_fragments):
    """Run Phase 3: Semantics-Driven Verification."""
    console.print("\n[bold cyan]Phase 3: Semantics-Driven Verification[/bold cyan]")
    from codemorph.verifier.orchestrator import run_phase3_semantics

    equivalence_reports = run_phase3_semantics(cfg, state, translated_fragments)
    console.print(f"[green]Phase 3 complete[/green]")
    return equivalence_reports


def run_phase4(cfg, state_dir: Path, output_dir: Path):
    """Run Phase 4: Assembly."""
    console.print(f"\n[bold cyan]Phase 4: Assembly[/bold cyan]")
    console.print(f"  State: {state_dir}")
    console.print(f"  Output: {output_dir}")

    from codemorph.assembler.orchestrator import Phase4Orchestrator

    source_dir = Path(cfg.project.source.root)

    orchestrator = Phase4Orchestrator(
        config=cfg,
        state_dir=state_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        verbose=True,
        fill_mocks=True,
        max_fix_iterations=5,
    )

    results = orchestrator.run()

    if results.get("compile_fix", {}).get("final_status") == "clean":
        console.print("\n[bold green]Assembly complete! All files compile cleanly.[/bold green]")
    else:
        console.print("\n[bold yellow]Assembly complete with some issues.[/bold yellow]")

    return results


def main():
    start = time.time()

    # Load config
    cfg = load_config_from_yaml(Path("javatuples_config.yaml"))
    # Force auto checkpoint mode (no interactive prompts)
    cfg.checkpoint_mode = CheckpointMode.AUTO

    state_dir = Path(cfg.project.state_dir)

    # Load existing state
    console.print(f"[cyan]Loading state from {state_dir / 'latest.json'}...[/cyan]")
    state = TranslationState.load_latest(state_dir)
    console.print(f"  Session: {state.session_id}")
    console.print(f"  Phase: {state.current_phase}")
    console.print(f"  Fragments translated: {len(state.translated_fragments)}")

    # Phase 2: Resume
    translated_fragments = resume_phase2(state, cfg)

    # Phase 3: Semantics verification
    run_phase3(state, cfg, translated_fragments)

    # Phase 4: Assembly - output to /workspace/code-agent/javatuples-python/
    output_dir = Path("/workspace/code-agent/javatuples-python")
    run_phase4(cfg, state_dir, output_dir)

    elapsed = time.time() - start
    console.print(f"\n[bold green]All phases complete in {elapsed:.1f}s[/bold green]")
    console.print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
