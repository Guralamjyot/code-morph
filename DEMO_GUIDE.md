# CodeMorph Demo Guide

How to run the Phase 1 (Analysis) and Phase 2 (Translation) demos, and how to
configure an LLM backend.

---

## Quick Start

```bash
# Install (once)
pip install -e .

# Phase 1 only — no LLM required
python demo_phase1.py

# Phase 1 + 2 — uses OpenRouter (default)
python demo_phase2.py --config my_config.yaml
```

Both scripts default to the **tictactoe** example (Java 17 → Python 3.11).
You can point them at any source directory in either direction.

---

## CLI Usage

The main entry point is `codemorph translate`:

```bash
# Java → Python (small example)
codemorph translate ./examples/test_j2p --config test_j2p_config.yaml -v

# Python → Java (small example)
codemorph translate ./examples/test_p2j --config test_p2j_config.yaml -v

# Tictactoe (larger, 79 fragments)
codemorph translate ./examples/tictactoe/src --config tictactoe_config.yaml -v
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source-dir` | `examples/tictactoe/src` | Directory containing source files |
| `--source-lang` | `java` | Source language: `java` or `python` |
| `--target-lang` | `python` | Target language: `java` or `python` |
| `--test-dir` | auto-detected | Test directory (looks for `../tests` relative to source) |
| `--output-dir` | `./demo_output/<project>` | Where to write output and state |
| `--config` | none | YAML config file — overrides all other flags |

### Demo Script Examples

```bash
# Default: tictactoe Java → Python
python demo_phase1.py
python demo_phase2.py --config tictactoe_config.yaml

# Python → Java (calculator example)
python demo_phase1.py \
    --source-dir examples/python_project \
    --source-lang python \
    --target-lang java

# Your own project
python demo_phase1.py \
    --source-dir /path/to/your/java/src \
    --source-lang java \
    --target-lang python \
    --test-dir /path/to/your/java/tests
```

---

## What Each Phase Does

### Phase 1: Analysis (no LLM needed)

1. **File discovery** — recursively finds `.java` or `.py` files in the source dir
2. **AST parsing** — parses each file using tree-sitter
3. **Fragment extraction** — breaks code into translatable units (classes, methods, functions, constants)
4. **Dependency analysis** — determines which fragments depend on which
5. **Graph building** — creates a directed dependency graph (saved as GraphML)
6. **Translation ordering** — topological sort to determine the order fragments should be translated

**Output artifacts:**
- `<output-dir>/.codemorph/latest.json` — full state (all fragments, dependencies, metadata)
- `<output-dir>/.codemorph/dependency_graph.graphml` — dependency graph

### Phase 2: Translation (requires LLM)

Runs Phase 1 first, then for each fragment in dependency order:

1. **Feature mapping** — applies language-construct rules (e.g. Java `Stream` → Python list comprehension)
2. **LLM translation** — sends the fragment + dependency context to the LLM
3. **Compilation check** — verifies the translated code parses/compiles
4. **Feature validation** — checks that required language patterns are present in the output
5. **Type verification** — validates type compatibility between source and target
6. **Retry loop** — on failure, sends errors back to the LLM for refinement (up to `max_retries_type_check` attempts)
7. **Mocking** — if retries are exhausted, generates a stub so the rest of the pipeline can continue
8. **Overload merging** — detects Java method overloads and merges them into a single Python function

**Output artifacts:**
- Updated `latest.json` with translated code for each fragment
- `<output-dir>/.codemorph/symbol_registry.json` — maps source names to target names

---

## LLM Setup

Phase 2 needs an LLM. Three providers are supported: **OpenRouter** (default,
recommended), **OpenAI**, and **Ollama** (local).

### Option A: OpenRouter (default)

OpenRouter gives you access to a wide range of models via a single API key.
The default model is **`openrouter/aurora-alpha`**.

1. **Get an API key** at [openrouter.ai/keys](https://openrouter.ai/keys)

2. **Create a config file** (e.g. `my_config.yaml`):

   ```yaml
   project:
     name: tictactoe
     source:
       language: java
       version: "17"
       root: ./examples/tictactoe/src
       test_root: ./examples/tictactoe/tests
     target:
       language: python
       version: "3.11"
       output_dir: ./output/tictactoe

   llm:
     provider: openrouter
     api_key: "sk-or-v1-your-key-here"
     model: "openrouter/aurora-alpha"
     temperature: 0.2

   translation:
     max_retries_type_check: 5
     allow_mocking: true

   checkpoint_mode: auto
   ```

   Alternatively, you can omit `api_key` from the YAML and set the
   environment variable instead:

   ```bash
   export OPENAI_API_KEY="sk-or-v1-your-key-here"
   ```

3. **Run it:**

   ```bash
   # Using the CLI
   codemorph translate ./examples/tictactoe/src --config my_config.yaml -v

   # Or using the demo script
   python demo_phase2.py --config my_config.yaml
   ```

#### Other Models on OpenRouter

The default `openrouter/aurora-alpha` works well for code translation. If you
want to try alternatives:

| Model ID | Context | Notes |
|----------|---------|-------|
| `openrouter/aurora-alpha` | 128k | **Default** — good all-around performance |
| `deepseek/deepseek-coder-v2` | 128k | Strong code translation |
| `deepseek/deepseek-v3-0324` | 128k | Complex logic, large classes |
| `qwen/qwen-2.5-coder-32b-instruct` | 32k | Fast, good at Java/Python |
| `meta-llama/llama-3.1-70b-instruct` | 128k | General purpose, reliable |

### Option B: OpenAI

Same config structure but uses OpenAI's API directly.

```yaml
llm:
  provider: openai
  api_key: "sk-proj-your-openai-key"   # or set OPENAI_API_KEY env var
  model: gpt-4o-mini
  temperature: 0.2
```

### Option C: Ollama (local, no API key)

Run models locally. Requires a machine with a GPU (or patience with CPU inference).

1. **Install Ollama:** [ollama.com/download](https://ollama.com/download)

2. **Pull a code model:**

   ```bash
   ollama pull deepseek-coder:6.7b      # 4 GB, fast
   ollama pull deepseek-coder-v2:16b    # 10 GB, better quality
   ollama pull qwen2.5-coder:14b        # 9 GB, good balance
   ```

3. **Start the server** (if not already running):

   ```bash
   ollama serve
   ```

4. **Config:**

   ```yaml
   llm:
     provider: ollama
     host: http://localhost:11434
     model: qwen2.5-coder:14b
     temperature: 0.2
   ```

---

## Config File Reference

A complete config file with all options and their defaults:

```yaml
project:
  name: my_project                     # Project name (used in logs)
  source:
    language: java                     # java | python
    version: "17"                      # Java: 11, 17, 21  Python: 2.7, 3.6-3.12
    root: ./src                        # Path to source files
    test_root: ./tests                 # Path to test files (optional)
  target:
    language: python                   # java | python
    version: "3.11"                    # Target language version
    output_dir: ./output               # Where translated code goes
    build_system: gradle               # gradle | maven (required when target is Java)
    package_name: com.example.app      # Java package name (required when target is Java)

llm:
  provider: openrouter                 # openrouter | openai | ollama
  api_key: "sk-or-v1-..."             # Required for openrouter/openai (or set OPENAI_API_KEY)
  host: http://localhost:11434         # Ollama server URL (only used with ollama provider)
  model: "openrouter/aurora-alpha"     # Model identifier
  temperature: 0.2                     # 0.0 = deterministic, higher = more creative
  context_window: 16384                # Max tokens in context
  timeout: 300                         # Request timeout in seconds

translation:
  max_retries_type_check: 15           # Retry budget per fragment (Phase 2)
  max_retries_semantics: 5             # Retry budget per fragment (Phase 3)
  requery_budget: 10                   # Feature mapping re-query budget
  allow_mocking: true                  # Generate stubs when retries exhausted
  strict_naming: true                  # Enforce target naming conventions

verification:
  generate_tests: true                 # Generate tests if source lacks them
  equivalence_check: true              # Run type compatibility checks
  execution_timeout: 30                # Timeout per test execution (seconds)

# How often to pause for human review
# interactive = every fragment, batch = phase boundaries, auto = never pause
checkpoint_mode: auto
```

---

## Included Examples

| Example | Direction | Source | Files | Fragments | Config |
|---------|-----------|--------|-------|-----------|--------|
| **test_j2p** | Java → Python | `examples/test_j2p/` | 1 | 2 | `test_j2p_config.yaml` |
| **test_p2j** | Python → Java | `examples/test_p2j/` | 1 | 1 | `test_p2j_config.yaml` |
| **tictactoe** | Java → Python | `examples/tictactoe/src/` | 11 | 79 | `tictactoe_config.yaml` |
| **calculator** | Python → Java | `examples/python_project/` | 2 | ~74 | `openrouter_config.yaml` |
| **javatuples** | Java → Python | `examples/javatuples/src/` | — | — | `javatuples_config.yaml` |

All configs are pre-configured with `openrouter/aurora-alpha`. Just add your API key if not already set.

---

## Troubleshooting

**"API key required"**
Set `api_key` in the YAML config or export `OPENAI_API_KEY` in your shell.

**"Failed to connect to Ollama at http://localhost:11434"**
Ollama isn't running. Either start it (`ollama serve`) or use `--config` with
an OpenRouter config (recommended).

**"Invalid Python/Java version"**
Supported versions — Python: `2.7`, `3.6`-`3.12`. Java: `11`, `17`, `21`.

**"Java target requires build_system"**
When translating to Java, the config must include `build_system: gradle` (or `maven`)
and `package_name`.

**Phase 2 is slow**
Each fragment makes one or more LLM calls. For 79 fragments with retries, expect
a few minutes with a fast API model, longer with local Ollama. Reduce
`max_retries_type_check` (e.g. to 3) for faster iteration during demos.
