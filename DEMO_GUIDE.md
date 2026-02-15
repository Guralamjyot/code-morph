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

# Phase 1 + 2 — requires an LLM (see "LLM Setup" below)
python demo_phase2.py --config my_config.yaml
```

Both scripts default to the **tictactoe** example (Java 17 → Python 3.11).
You can point them at any source directory in either direction.

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source-dir` | `examples/tictactoe/src` | Directory containing source files |
| `--source-lang` | `java` | Source language: `java` or `python` |
| `--target-lang` | `python` | Target language: `java` or `python` |
| `--test-dir` | auto-detected | Test directory (looks for `../tests` relative to source) |
| `--output-dir` | `./demo_output/<project>` | Where to write output and state |
| `--config` | none | YAML config file — overrides all other flags |

### Examples

```bash
# Default: tictactoe Java → Python
python demo_phase1.py
python demo_phase2.py --config openrouter_tictactoe.yaml

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

Phase 2 needs an LLM. Three providers are supported: **OpenRouter** (recommended
for open-source models), **OpenAI**, and **Ollama** (local).

### Option A: OpenRouter (recommended)

OpenRouter gives you access to open-source models (DeepSeek, Qwen, Llama, etc.)
via a single API key. Models run on their infrastructure — no GPU needed locally.

1. **Get an API key** at [openrouter.ai/keys](https://openrouter.ai/keys)

2. **Set the environment variable** (so you don't put keys in config files):

   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
   ```

3. **Create a config file** (e.g. `my_config.yaml`):

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
     # api_key is read from OPENAI_API_KEY env var (see step 2)
     model: deepseek/deepseek-coder-v2
     temperature: 0.2

   translation:
     max_retries_type_check: 5
     allow_mocking: true

   checkpoint_mode: auto
   ```

   You can also put the key directly in the YAML (`api_key: "sk-or-v1-..."`),
   but the env var approach keeps secrets out of config files.

4. **Run it:**

   ```bash
   python demo_phase2.py --config my_config.yaml
   ```

#### Recommended Open-Source Models on OpenRouter

| Model ID | Context | Good For | Cost |
|----------|---------|----------|------|
| `deepseek/deepseek-coder-v2` | 128k | Best overall code translation | ~$0.14/M tokens |
| `deepseek/deepseek-v3-0324` | 128k | Complex logic, large classes | ~$0.50/M tokens |
| `qwen/qwen-2.5-coder-32b-instruct` | 32k | Fast, good at Java/Python | ~$0.20/M tokens |
| `meta-llama/llama-3.1-70b-instruct` | 128k | General purpose, reliable | ~$0.40/M tokens |
| `mistralai/codestral-latest` | 32k | Code-specific, fast | ~$0.30/M tokens |

For the tictactoe example (~79 fragments), `deepseek/deepseek-coder-v2` or
`qwen/qwen-2.5-coder-32b-instruct` work well and are inexpensive.

### Option B: OpenAI

Same as OpenRouter but uses OpenAI's API directly.

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

4. **Run the demo** (no config file needed — Ollama is the default):

   ```bash
   python demo_phase2.py
   ```

   Or with a specific model:

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
    version: "17"                      # Java: 11, 17, 21  Python: 2.7, 3.6–3.12
    root: ./src                        # Path to source files
    test_root: ./tests                 # Path to test files (optional)
  target:
    language: python                   # java | python
    version: "3.11"                    # Target language version
    output_dir: ./output               # Where translated code goes
    build_system: gradle               # gradle | maven (required when target is Java)
    package_name: com.example.app      # Java package name (required when target is Java)

llm:
  provider: openrouter                 # ollama | openrouter | openai
  api_key: "sk-or-v1-..."             # Required for openrouter/openai (or set OPENAI_API_KEY)
  host: http://localhost:11434         # Ollama server URL (only used with ollama provider)
  model: deepseek/deepseek-coder-v2   # Model identifier
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

| Example | Source | Files | Fragments | Path |
|---------|--------|-------|-----------|------|
| **tictactoe** (default) | Java 17 | 11 | 79 | `examples/tictactoe/src/` |
| calculator | Python 3.10 | 2 | 74 | `examples/python_project/` |

---

## Troubleshooting

**"Failed to connect to Ollama at http://localhost:11434"**
Ollama isn't running. Either start it (`ollama serve`) or use `--config` with
an OpenRouter/OpenAI config.

**"API key required"**
Set `api_key` in the YAML config or export `OPENAI_API_KEY` in your shell.

**"Invalid Python/Java version"**
Supported versions — Python: `2.7`, `3.6`–`3.12`. Java: `11`, `17`, `21`.

**"Java target requires build_system"**
When translating to Java, the config must include `build_system: gradle` (or `maven`)
and `package_name`.

**Phase 2 is slow**
Each fragment makes one or more LLM calls. For 79 fragments with retries, expect
a few minutes with a fast API model, longer with local Ollama. Reduce
`max_retries_type_check` (e.g. to 3) for faster iteration during demos.
