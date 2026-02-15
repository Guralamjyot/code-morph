# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

CodeMorph is a bidirectional Python ↔ Java code translator that uses LLMs (Ollama/OpenRouter) with a three-phase Oxidizer-inspired pipeline: (1) project partitioning, (2) type-driven translation, (3) semantics-driven verification. It also supports within-language version upgrades (e.g., Python 2.7 → 3.10, Java 11 → 21).

## Build & Install

```bash
pip install -e .              # Minimal install (subprocess bridge, no native deps)
pip install -e ".[dev]"       # + black, ruff, mypy, pytest-cov
pip install -e ".[bridge]"    # + JPype/Py4J (optional native bridges)
pip install -e ".[rag]"       # + ChromaDB, sentence-transformers (optional RAG)
pip install -e ".[all]"       # Everything
```

## Testing

```bash
pytest tests/ -v                                         # All tests
pytest tests/unit/test_feature_mapper.py -v              # Single file
pytest tests/unit/test_phase2_orchestrator.py::test_name -v  # Single test
pytest tests/ --cov=codemorph --cov-report=html          # With coverage
```

Test config is in `pyproject.toml` under `[tool.pytest.ini_options]`: testpaths=`tests/`, addopts=`-v --tb=short`.

## Linting & Formatting

```bash
black src/ tests/           # Format (line-length=100, target py310+)
ruff check src/ tests/      # Lint (rules: E, F, I, N, W, UP)
mypy src/codemorph/         # Type check (strict mode, ignore_missing_imports)
```

All tool configs are in `pyproject.toml`.

## CLI

Entry point: `codemorph` → `src/codemorph/cli/main.py:app` (Typer).

Key commands:
- `codemorph translate <source> --target-lang java --output <dir>` — full pipeline
- `codemorph analyze <source>` — Phase 1 only
- `codemorph init --output codemorph.yaml` — generate default config
- `codemorph doctor` — check system dependencies

## Architecture: Three-Phase Pipeline

### Phase 1: Project Partitioning (`analyzer/`)
- `orchestrator.py` — runs Phase 1 analysis
- `graph_builder.py` — builds dependency DAG with NetworkX
- Language plugins (`languages/{python,java}/plugin.py`) parse source via tree-sitter, extract `CodeFragment` objects (functions, classes, methods, globals)
- Topological sort determines translation order; circular deps flagged for interface extraction

### Phase 2: Type-Driven Translation (`translator/`)
- `orchestrator.py` — main Phase 2 loop: for each fragment in dependency order, apply feature mapping → LLM translation → compile check → type compatibility check → retry or mock
- `orchestrator_rag.py` — RAG-enhanced variant with vector store style retrieval
- `llm_client.py` — `OllamaClient` and `OpenRouterClient` with conversation history tracking; methods: `translate_fragment()`, `refine_translation()`, `fix_io_mismatch()`
- `rag_llm_client.py` — RAG-enhanced LLM client with two-tier retrieval (Bootstrap + Snowball)
- `test_orchestrator.py` / `test_translator.py` — test file translation logic
- Retry budget: `max_retries_type_check` (default 15); falls back to mocking on exhaustion

### Phase 3: Semantics-Driven Translation (`verifier/`)
- `orchestrator.py` — runs I/O equivalence verification
- `snapshot_capture.py` — captures execution I/O from test runs
- `equivalence_checker.py` — compares Python vs Java outputs
- `type_checker.py` — JSON round-trip type compatibility
- `mocker.py` — generates mocks for untranslatable functions

### Cross-Language Bridges (`bridges/`)
- `subprocess_bridge.py` — **default bridge**: universal JSON-over-subprocess, no native deps. Serialize inputs → launch target language process → pass JSON via stdin → read JSON from stdout
- `python_runner.py` — Python subprocess execution
- `java_executor.py` — legacy Java execution (prefer subprocess_bridge)
- `bridges/java/BridgeRunner.java`, `TypeChecker.java` — Java-side bridge components

### Knowledge System (`knowledge/`)
- `feature_mapper.py` — rule-based language construct mappings (e.g., list comprehension → Stream API)
- `library_mapper.py` — Python ↔ Java library equivalents
- `vector_store.py` — ChromaDB RAG store (optional, install with `[rag]`)
- YAML rule files: `feature_rules/{python_to_java,java_to_python}.yaml`, `library_maps/{python_to_java,java_to_python}.yaml`

### State & Config
- `state/persistence.py` — session save/restore to `.codemorph/*.json`
- `state/symbol_registry.py` — cross-language name mapping (snake_case ↔ camelCase), tracks translation status per symbol
- `config/models.py` — comprehensive Pydantic config models (~40 fields), core enums (`LanguageType`, `FragmentType`, `TranslationStatus`, `CheckpointMode`, `LLMProvider`)
- `config/simple_config.py` — simplified config (15 fields), backward compatible
- `config/loader.py` — YAML loading, CLI arg parsing, validation

### Human-in-the-Loop
- `cli/checkpoint_ui.py` — interactive checkpoint prompts
- Three modes: `interactive` (every function), `batch` (phase boundaries), `auto` (no stops)

## Key Data Models (in `config/models.py`)

- `CodeFragment` — parsed source unit (id, source_code, ast_type, dependencies, language metadata)
- `TranslatedFragment` — translation result with compilation/verification status
- `AnalysisResult` — Phase 1 output (fragments dict, dependency graph, translation queue)
- `SymbolMapping` — maps source symbol to target name/file/signature/status

## Config File Format

YAML config (`codemorph.yaml`) with sections: `project` (source/target language, versions, paths), `llm` (provider, model, temperature), `translation` (retry budgets, mocking), `checkpoint_mode`. See `openrouter_config.yaml` or `java_to_python_config.yaml` for examples.

## Important Patterns

- All source code is under `src/codemorph/` (setuptools `packages.find` with `where=["src"]`)
- Language support is plugin-based: extend `languages/base/plugin.py` for new languages
- LLM interaction uses multi-turn conversation with error feedback for iterative refinement
- The subprocess bridge uses JSON as the universal interchange format for cross-language type checking
- Session state persists to `.codemorph/` directory for resume capability
