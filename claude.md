# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ IMPORTANT: Simplified Implementation (January 2026)

CodeMorph has been simplified to align with the streamlined specification (`SPEC.md`). Key changes:

### 1. Subprocess+JSON Bridge (Default)
- **NEW DEFAULT**: [bridges/subprocess_bridge.py](src/codemorph/bridges/subprocess_bridge.py) - Universal, no native dependencies
- **OPTIONAL**: JPype/Py4J bridges (install with `pip install codemorph[bridge]`)
- **Why**: Simpler, portable, easier to debug. Works on any system with Python + Java.

### 2. Simplified Configuration
- **NEW**: [config/simple_config.py](src/codemorph/config/simple_config.py) - Streamlined config with only essentials
- **OLD**: [config/models.py](src/codemorph/config/models.py) - Kept for backward compatibility
- **Example Config**: See [SPEC.md](SPEC.md) Section 8 or [SIMPLIFICATION_PLAN.md](SIMPLIFICATION_PLAN.md)

### 3. Optional Dependencies
```bash
# Minimal install (recommended) - uses subprocess bridge
pip install -e .

# With native bridges (advanced, requires Java runtime)
pip install -e ".[bridge]"

# With RAG (optional, disabled by default)
pip install -e ".[rag]"

# Everything
pip install -e ".[all]"
```

### 4. Key Documents
- **[SPEC.md](SPEC.md)**: Simplified specification (authoritative reference)
- **[SIMPLIFICATION_PLAN.md](SIMPLIFICATION_PLAN.md)**: Detailed simplification roadmap
- **Below**: Original comprehensive architecture (still valid but being simplified)

---

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run CLI
codemorph --help
codemorph doctor          # Check all dependencies
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_feature_mapper.py -v

# Run single test
pytest tests/unit/test_phase2_orchestrator.py::test_function_name -v

# Run with coverage
pytest tests/ --cov=codemorph --cov-report=html
```

## Linting & Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/codemorph/
```

Configuration: `pyproject.toml` (line-length=100, Python 3.10+)

## Architecture Overview

CodeMorph is a bidirectional Python ↔ Java code translator using a three-phase Oxidizer-inspired approach:

```
Phase 1: Partitioning     Phase 2: Type-Driven       Phase 3: Semantics-Driven
─────────────────────     ───────────────────────    ─────────────────────────
AST Parse → Fragments     Feature Map → LLM Gen      Snapshots → Execute
     ↓                         ↓                          ↓
Dependency Graph          Compile Check              I/O Equivalence
     ↓                         ↓                          ↓
Topological Sort          Type Compat Verify         Refine or Mock
```

### Key Entry Points

- **CLI**: [main.py](src/codemorph/cli/main.py) - Typer-based commands (`translate`, `analyze`, `verify`, `init`, `doctor`, `resume`)
- **Phase 1 Orchestrator**: [analyzer/orchestrator.py](src/codemorph/analyzer/orchestrator.py) - Project partitioning
- **Phase 2 Orchestrator**: [translator/orchestrator.py](src/codemorph/translator/orchestrator.py) - Type-driven translation
- **Phase 3 Orchestrator**: [verifier/orchestrator.py](src/codemorph/verifier/orchestrator.py) - Semantics verification

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Language Plugins | `languages/{python,java}/plugin.py` | AST parsing, fragment extraction, compilation |
| LLM Client | `translator/llm_client.py` | Ollama/OpenRouter integration with conversation history |
| Feature Mapper | `knowledge/feature_mapper.py` | Rules for language construct translation |
| Library Mapper | `knowledge/library_mapper.py` | Python ↔ Java library equivalents |
| Symbol Registry | `state/symbol_registry.py` | Cross-language name mapping (snake_case ↔ camelCase) |
| **Subprocess Bridge** | **`bridges/subprocess_bridge.py`** | **Universal cross-language bridge via JSON (DEFAULT)** |
| Vector Store | `knowledge/vector_store.py` | ChromaDB RAG for style-consistent translation (optional) |
| Java Executor | `bridges/java_executor.py` | Legacy Java execution (use subprocess_bridge instead) |

### Data Flow

1. **Config**: `codemorph.yaml` → `config/loader.py` → `config/models.py` (Pydantic)
2. **Parsing**: Source files → Language plugin → `CodeFragment` objects
3. **Translation**: Fragments → Feature rules + LLM → `TranslatedFragment`
4. **Verification**: Snapshots from tests → Cross-language bridge → I/O comparison
5. **State**: Session saved to `.codemorph/latest.json` for resume capability

### Knowledge Bases

- Feature rules: `knowledge/feature_rules/python_to_java.yaml` (20+ mappings)
- Library maps: `knowledge/library_maps/{python_to_java,java_to_python}.yaml`
- Bootstrap examples: `examples/bootstrap_examples/*.json`

### Example Project

Test translations using `examples/python_project/calculator.py` which has comprehensive functions, classes, and edge cases with full test coverage.

---

# Detailed Architecture Reference

The sections below contain the complete CodeMorph specification for reference.

---

# CodeMorph v2.0 — Complete Project Plan
## Bidirectional Python ↔ Java Code Translation Agent

**Document Version:** 2.0
**Last Updated:** January 2025
**Status:** Ready for Implementation

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Agent Spawning & Execution](#4-agent-spawning--execution)
5. [Configuration System](#5-configuration-system)
6. [Phase 1: Project Partitioning](#6-phase-1-project-partitioning)
7. [Phase 2: Type-Driven Translation](#7-phase-2-type-driven-translation)
8. [Phase 3: Semantics-Driven Translation](#8-phase-3-semantics-driven-translation)
9. [Feature Mapping System](#9-feature-mapping-system)
10. [Type Compatibility System](#10-type-compatibility-system)
11. [Execution Snapshots System](#11-execution-snapshots-system)
12. [I/O Equivalence Verification](#12-io-equivalence-verification)
13. [Function Mocking System](#13-function-mocking-system)
14. [Symbol Registry & Name Mapping](#14-symbol-registry--name-mapping)
15. [Library Mapping System](#15-library-mapping-system)
16. [Human-in-the-Loop Interface](#16-human-in-the-loop-interface)
17. [LLM Integration](#17-llm-integration)
18. [Cross-Language Bridges](#18-cross-language-bridges)
19. [Error Handling & Recovery](#19-error-handling--recovery)
20. [Project Structure](#20-project-structure)
21. [Data Models](#21-data-models)
22. [CLI Reference](#22-cli-reference)
23. [Implementation Phases](#23-implementation-phases)
24. [Testing Strategy](#24-testing-strategy)
25. [Example Workflows](#25-example-workflows)

---

# 1. Executive Summary

## 1.1 Project Overview

| Attribute | Value |
|-----------|-------|
| **Project Name** | CodeMorph v2.0 |
| **Purpose** | Bidirectional code translation between Python and Java |
| **Approach** | Oxidizer-inspired: Feature Mapping + Type-Compatibility + I/O Equivalence |
| **Translation Unit** | Function-by-function with dependency awareness |
| **Verification** | Compile testing + I/O equivalence validation |
| **Human Interaction** | Checkpoint-based approval with configurable granularity |

## 1.2 Supported Languages

| Language | Versions | Direction |
|----------|----------|-----------|
| Python | 2.7, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12 | Source & Target |
| Java | 11, 17, 21 (LTS versions) | Source & Target |

## 1.3 Key Innovations (From Oxidizer Paper)

1. **Feature Mapping**: Predefined rules that guide LLM translation of language-specific constructs
2. **Type-Compatibility**: Early detection of type mismatches via serialization round-trip
3. **Two-Phase Translation**: Type-driven phase followed by semantics-driven refinement
4. **Execution Snapshots**: Test-derived I/O pairs for validation
5. **Function Mocking**: Graceful degradation when translation fails

## 1.4 Expected Outcomes

| Metric | Target |
|--------|--------|
| Compilation Success Rate | ≥95% |
| I/O Equivalence Rate | ≥70% |
| Runtime Errors During Validation | <5% |
| Human Intervention Required | <20% of functions |

---

# 2. System Overview

## 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CODEMORPH TRANSLATION FLOW                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   USER INPUT                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  • Source Directory (e.g., ./my_python_project)                         │  │
│   │  • Target Language (e.g., java)                                         │  │
│   │  • Output Directory (e.g., ./my_java_project)                           │  │
│   │  • Configuration (optional, e.g., ./codemorph.yaml)                     │  │
│   │  • Test Suite Location (optional, e.g., ./tests)                        │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│   PHASE 1: PROJECT PARTITIONING                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  • Parse all source files into AST                                      │  │
│   │  • Extract code fragments (functions, classes, globals)                 │  │
│   │  • Build dependency graph                                               │  │
│   │  • Determine translation order (topological sort)                       │  │
│   │  • CHECKPOINT #1: Human reviews analysis                                │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│   PHASE 2: TYPE-DRIVEN TRANSLATION                                              │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  FOR EACH fragment in dependency order:                                 │  │
│   │    • Apply feature mapping rules                                        │  │
│   │    • Query LLM for translation                                          │  │
│   │    • Check feature mapping compliance                                   │  │
│   │    • Attempt compilation                                                │  │
│   │    • Verify type-compatibility                                          │  │
│   │    • If fail: retry or mock                                             │  │
│   │  • CHECKPOINT #2: Human reviews type-compatible translation             │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│   PHASE 3: SEMANTICS-DRIVEN TRANSLATION                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  FOR EACH function:                                                     │  │
│   │    • Collect execution snapshots from tests                             │  │
│   │    • Check I/O equivalence (with callee mocking)                        │  │
│   │    • If fail: re-translate body (frozen signature)                      │  │
│   │    • Retry up to max_tries                                              │  │
│   │  • CHECKPOINT #3: Human reviews final translation                       │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│   OUTPUT                                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │  • Translated project in target language                                │  │
│   │  • Verification report (JSON + Markdown)                                │  │
│   │  • Symbol mapping file                                                  │  │
│   │  • List of mocked functions (requiring manual attention)                │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Core Principles

1. **Dependency-Aware Translation**: Translate dependencies before dependents
2. **Incremental Validation**: Verify each fragment before proceeding
3. **Early Error Detection**: Type-compatibility catches issues before they cascade
4. **Graceful Degradation**: Mock functions that can't be translated
5. **Human Oversight**: Checkpoints for critical decisions
6. **Reproducibility**: Deterministic results with logged LLM interactions

---

# 3. Architecture

## 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CODEMORPH ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                              AGENT ORCHESTRATOR                              │   │
│  │  • Manages translation workflow                                              │   │
│  │  • Coordinates phases                                                        │   │
│  │  • Handles checkpoints                                                       │   │
│  │  • Tracks progress                                                           │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                │                    │                    │                          │
│                ▼                    ▼                    ▼                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                  │
│  │   PHASE 1:       │  │   PHASE 2:       │  │   PHASE 3:       │                  │
│  │   PARTITIONING   │  │   TYPE-DRIVEN    │  │   SEMANTICS      │                  │
│  │                  │  │                  │  │                  │                  │
│  │  • AST Parser    │  │  • Feature       │  │  • Snapshot      │                  │
│  │  • Fragment      │  │    Mapping       │  │    Collector     │                  │
│  │    Extractor     │  │  • Type-Compat   │  │  • I/O Equiv     │                  │
│  │  • Dependency    │  │    Checker       │  │    Checker       │                  │
│  │    Graph         │  │  • Compiler      │  │  • Callee        │                  │
│  │                  │  │  • Mocker        │  │    Mocker        │                  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘                  │
│           │                    │                    │                              │
│           └────────────────────┼────────────────────┘                              │
│                                ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           SHARED SERVICES                                    │   │
│  │                                                                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│  │  │    LLM      │  │   Symbol    │  │  Feature    │  │   Vector    │        │   │
│  │  │   Client    │  │  Registry   │  │   Rules     │  │   Store     │        │   │
│  │  │             │  │             │  │             │  │             │        │   │
│  │  │  • Ollama   │  │  • Name     │  │  • Py→Java  │  │  • Nomic    │        │   │
│  │  │  • OpenAI   │  │    Mapping  │  │  • Java→Py  │  │    Embed    │        │   │
│  │  │  • Anthropic│  │  • Status   │  │  • Checks   │  │  • ChromaDB │        │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │   │
│  │                                                                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│  │  │  Library    │  │   Type      │  │  Execution  │  │  Cross-Lang │        │   │
│  │  │  Mapper     │  │   Mapper    │  │  Snapshots  │  │   Bridge    │        │   │
│  │  │             │  │             │  │             │  │             │        │   │
│  │  │  • Std Lib  │  │  • Py↔Java  │  │  • Cache    │  │  • Py4J     │        │   │
│  │  │  • 3rd Party│  │  • Custom   │  │  • Persist  │  │  • JPype    │        │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │   │
│  │                                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                              │
│                                      ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                         HUMAN-IN-THE-LOOP INTERFACE                          │   │
│  │                                                                              │   │
│  │  • Checkpoint prompts          • Progress display                           │   │
│  │  • Multi-choice selection      • Error explanations                         │   │
│  │  • Override capability         • Report generation                          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 4. Agent Spawning & Execution

## 4.1 Installation

### 4.1.1 Prerequisites

```bash
# System requirements
- Python 3.10+
- Java 11+ (for Java compilation/execution)
- Ollama (for local LLM) OR API keys for cloud LLMs
- 8GB+ RAM recommended
- 10GB+ disk space for models

# Optional but recommended
- Docker (for isolated compilation environments)
- pytest (for Python test execution)
- Maven/Gradle (for Java build)
```

### 4.1.2 Installation Steps

```bash
# Clone repository
git clone https://github.com/your-org/codemorph.git
cd codemorph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install Ollama and pull model (for local LLM)
# See: https://ollama.ai/download
ollama pull codellama:13b  # Or your preferred model

# Verify installation
codemorph --version
codemorph doctor  # Check all dependencies
```

### 4.1.3 Environment Setup

```bash
# Required environment variables
export CODEMORPH_HOME=~/.codemorph
export CODEMORPH_CACHE=$CODEMORPH_HOME/cache
export CODEMORPH_LOGS=$CODEMORPH_HOME/logs

# For Ollama (local)
export OLLAMA_HOST=http://localhost:11434

# For OpenAI (optional)
export OPENAI_API_KEY=sk-...

# For Anthropic (optional)
export ANTHROPIC_API_KEY=sk-ant-...

# For HuggingFace embeddings
export HF_TOKEN=hf_...
```

## 4.2 Running the Agent

### 4.2.1 Basic Usage

```bash
# Minimal invocation
codemorph translate \
    --source ./my_python_project \
    --target-lang java \
    --output ./my_java_project

# With all options
codemorph translate \
    --source ./my_python_project \
    --source-lang python \
    --source-version 3.10 \
    --target-lang java \
    --target-version 17 \
    --output ./my_java_project \
    --config ./codemorph.yaml \
    --test-suite ./tests \
    --checkpoint-mode interactive \
    --max-retries 5 \
    --verbose
```

### 4.2.2 Directory Structure Requirements

**Source Directory (Python):**
```
my_python_project/
├── src/                    # Source code (optional nesting)
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_main.py
│   └── test_models.py
├── requirements.txt        # Dependencies
├── setup.py               # Package config (optional)
└── pyproject.toml         # Modern package config (optional)
```

**Source Directory (Java):**
```
my_java_project/
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── example/
│                   ├── Main.java
│                   ├── models/
│                   │   └── User.java
│                   └── utils/
│                       └── Helpers.java
├── src/
│   └── test/
│       └── java/
│           └── com/
│               └── example/
│                   ├── MainTest.java
│                   └── models/
│                       └── UserTest.java
├── pom.xml                # Maven config
└── build.gradle           # Gradle config (alternative)
```

### 4.2.3 Output Directory Structure

The agent generates a structured output directory that isolates generated code, verification reports, and internal state.

```
my_java_project/
├── .codemorph/                # Internal Agent State
│   ├── symbol_registry.json   # Map of Py->Java symbols
│   ├── dependency_graph.json  # DAG of project structure
│   ├── snapshots/             # Recorded I/O from Python tests
│   └── logs/                  # LLM conversation logs
├── src/                       # Translated Source
│   └── main/java/...
├── tests/                     # Translated Tests
│   └── test/java/...
├── verification_report.md     # Human-readable summary
└── build.gradle               # Generated build file
```

---

# 5. Configuration System

The agent is driven by a YAML configuration file (`codemorph.yaml`) that controls strictness, paths, and model parameters.

```yaml
project:
  name: "LegacyBillingSystem"
  source_language: "python"
  target_language: "java"
  source_root: "./src"
  test_root: "./tests"

llm:
  provider: "ollama"           # ollama | openai | anthropic
  model: "codellama:13b"       # or gpt-4-turbo, claude-3-opus
  temperature: 0.2
  context_window: 16384
  embedding_model: "nomic-ai/nomic-embed-code"

translation:
  max_retries_type_check: 15   # Phase 2 budget
  max_retries_semantics: 5     # Phase 3 budget
  requery_budget: 10           # Feature mapping budget
  allow_mocking: true          # If false, fail hard on untranslatable code
  strict_naming: true          # Enforce CamelCase in Java

verification:
  runner: "pytest"             # Source test runner
  generate_tests: true         # Generate new tests if source lacks them
  equivalence_check: true      # Run I/O comparison
```

---

# 6. Phase 1: Project Partitioning

**Goal:** Break the codebase into manageable units (fragments) and determine the order of translation to satisfy dependencies.

## 6.1 Logic Flow
1.  **AST Parsing:** Use `tree-sitter` to parse Python files into Abstract Syntax Trees.
2.  **Fragment Extraction:** Identify "Atomic Units of Translation":
    *   Global Variables / Constants
    *   Classes (Data Classes / DTOs)
    *   Interfaces / Abstract Base Classes
    *   Standalone Functions
    *   Class Methods
3.  **Dependency Analysis:**
    *   Scan imports and usage within ASTs.
    *   Build a Directed Acyclic Graph (DAG) using `networkx`.
    *   Handle circular dependencies by flagging them for "Interface Extraction" (creating a shared interface file).
4.  **Ordering:** Perform a topological sort. Bottom-level dependencies (utils, constants) are translated first.

## 6.2 Output Artifact
`analysis_manifest.json`:
```json
{
  "translation_order": ["utils.py::helper", "models.py::User", "main.py::process"],
  "circular_deps": [],
  "complexity_scores": {"main.py::process": 85}
}
```

---

# 7. Phase 2: Type-Driven Translation

**Goal:** Produce code that *compiles* and creates valid types, ensuring dependencies link correctly.

## 7.1 The Algorithm (Per Fragment)
1.  **Context Retrieval:** Fetch the signatures of already-translated dependencies from the `SymbolRegistry`.
2.  **Feature Mapping Application:**
    *   Analyze AST against `FeatureMappingRules`.
    *   *Example:* Detect `[x for x in list]` → Inject prompt instruction: "Use Java Stream API".
3.  **LLM Generation:** Generate Java code.
4.  **Static Analysis & Compilation:**
    *   Run `javac` on the fragment (using a virtual file system or temporary directory).
    *   If compile error: Feed error back to LLM (Retry Budget: `max_retries_type_check`).
5.  **Type Compatibility Check:**
    *   Use the **Type Bridge**. Verify that the input/output types of the generated Java method map reversibly to the Python types used in the test suite.
    *   *Fail condition:* Python sends `List[int]`, Java expects `ArrayList<String>`.

---

# 8. Phase 3: Semantics-Driven Translation

**Goal:** Ensure the translated code behaves exactly like the source (I/O Equivalence).

## 8.1 The Algorithm
1.  **Snapshot Replay:**
    *   Load recorded inputs from Python execution snapshots.
    *   Serialize inputs to JSON.
    *   Deserialize into Java objects.
2.  **Execution:**
    *   Run the compiled Java method.
    *   **Callee Mocking:** If the function calls *other* functions, mock them to return the exact values recorded in the snapshot. This isolates the logic of the current function.
3.  **Comparison:**
    *   Capture Java output and Side Effects (e.g., file writes).
    *   Compare against Python snapshot.
4.  **Refinement Loop:**
    *   If behavior differs (e.g., Python `div` is floor, Java `/` is truncate):
    *   **Freeze Signature:** The LLM is instructed to fix the *body* only; it cannot change arguments or return types (to preserve Phase 2 success).

---

# 9. Feature Mapping System

A rule-based guardrail system to handle language idioms.

## 9.1 Data Structure
```python
class FeatureMappingRule:
    id: str                  # e.g., "PY_LIST_COMP"
    premise: Callable        # Returns True if AST contains list comprehension
    prompt_instruction: str  # "Convert list comprehension to .stream().map().collect()"
    validation: Callable     # Returns True if output contains "stream()" or loop
```

## 9.2 Key Mappings (Python -> Java)
| Rule ID | Python Construct | Java Instruction |
|:---|:---|:---|
| `PY_DECORATOR` | `@decorator` | "Use Aspect Oriented Programming (AOP) or a wrapper pattern." |
| `PY_KWARGS` | `**kwargs` | "Use `Map<String, Object>` or a Builder pattern." |
| `PY_MULT_INHERIT` | `class A(B, C)` | "Refactor to use Interfaces or Composition (Java does not support multiple class inheritance)." |
| `PY_CONTEXT_MGR` | `with open(...)` | "Use `try-with-resources` statement." |
| `PY_DUCK_TYPE` | Implicit interface | "Extract an explicit Interface defining the used methods." |

---

# 10. Type Compatibility System

## 10.1 The JSON Bridge
We use JSON as the universal data interchange format to verify type alignment without writing custom JNI code for every type.

**Verification Logic:**
```python
def check_type_compat(py_val, java_type_signature):
    # 1. Serialize Python Value
    json_val = json.dumps(py_val)

    # 2. Attempt Java Deserialization (using Jackson/Gson)
    # The Java harness tries to map this JSON to the target signature (e.g., List<Integer>)
    valid = java_bridge.can_deserialize(json_val, java_type_signature)

    return valid
```

---

# 11. Execution Snapshots System

**Goal:** Capture "Ground Truth" behavior from the source code.

## 11.1 Instrumentation (Python Side)
The agent uses a Python decorator/profiler to wrap functions during test execution.

```python
# Internal Instrumentation Decorator
def log_snapshot(func):
    def wrapper(*args, **kwargs):
        input_snap = serialize(args, kwargs)
        try:
            result = func(*args, **kwargs)
            output_snap = serialize(result)
            save_snapshot(func.__name__, input_snap, output_snap, error=None)
            return result
        except Exception as e:
            save_snapshot(func.__name__, input_snap, None, error=str(e))
            raise e
    return wrapper
```

## 11.2 Storage
Stored in `.codemorph/snapshots/function_name.jsonlines` to allow streaming of large datasets.

---

# 12. I/O Equivalence Verification

This module is the "Judge".

1.  **Input:** A specific `snapshot_id` (inputs + expected output).
2.  **Action:** Call the Java function via the **Bridge** (Section 18).
3.  **Assert:**
    *   `Java_Result == Python_Result`
    *   `Java_Exception == Python_Exception` (mapped, e.g., `KeyError` -> `MissingResourceException`)

---

# 13. Function Mocking System

**Goal:** Allow the project to build even if specific functions fail translation (e.g., specific Python C-extensions).

**Strategy:**
If a function fails Phase 2 or Phase 3 repeatedly:
1.  Generate a **Java Native Interface (JNI)** or **Shell Wrapper** stub.
2.  The Java code calls the *original Python code* for that specific function.
3.  Mark function as `MOCKED` in the report.

**Generated Java Mock:**
```java
public int complexAlgorithm(Data d) {
    // Fallback to Python for this specific function
    return PythonBridge.call("complex_algorithm", d);
}
```

---

# 14. Symbol Registry & Name Mapping

**Goal:** Maintain referential integrity while respecting naming conventions.

## 14.1 Registry Schema
```json
{
  "py::calculate_tax": {
    "target_name": "calculateTax",
    "target_file": "src/main/java/com/app/Billing.java",
    "signature": "public double calculateTax(double amount)",
    "status": "VERIFIED"
  },
  "py::user_id": {
    "target_name": "userId",
    "status": "TRANSLATED"
  }
}
```

## 14.2 Renaming Rules
*   `snake_case` (Python functions/vars) -> `camelCase` (Java)
*   `PascalCase` (Python classes) -> `PascalCase` (Java)
*   `SCREAMING_SNAKE` (Constants) -> `SCREAMING_SNAKE` (Java)
*   **Conflict Resolution:** If `new_name` exists, append suffix (e.g., `calculateTax_1`).

---

# 15. Library Mapping System

## 15.1 Built-in Maps (`library_mappings.yaml`)
A curated list of high-confidence equivalents.

| Python Library | Java Equivalent |
|:---|:---|
| `json` | `com.fasterxml.jackson.databind` |
| `requests` | `java.net.http.HttpClient` or `OkHttp` |
| `unittest` / `pytest` | `org.junit.jupiter` (JUnit 5) |
| `logging` | `org.slf4j` / `Logback` |
| `sqlalchemy` | `Hibernate` / `JPA` (Complex, requires schema analysis) |

## 15.2 Unknown Libraries
If a library is not in the map:
1.  Agent prompts LLM: "What is the standard Java alternative for Python's `numpy`?"
2.  LLM suggests library (e.g., `ND4J`).
3.  **Checkpoint:** Human confirms adding `ND4J` to `pom.xml`.

---

# 16. Human-in-the-Loop Interface

The agent is designed to stop at critical junctures.

**Modes:**
*   `--interactive`: Stop at every function (high control).
*   `--batch`: Stop only at Phase completion (high speed).
*   `--auto`: No stops, only report errors (high risk).

**Checkpoint UI (CLI):**
```text
[CHECKPOINT] Function: compute_metrics (utils.py)
---------------------------------------------------
Source:  def compute_metrics(data): ...
Target:  public Metrics computeMetrics(List<Double> data) { ... }
Status:  COMPILED | I/O VERIFIED
---------------------------------------------------
1. Approve
2. Reject (Retry with hint)
3. Edit Java manually
4. Mark for Mocking
> _
```

---

# 17. LLM Integration

## 17.1 Context Management & RAG Strategy
To fit code into limited context windows while ensuring architectural consistency, we employ a two-tier retrieval strategy:

1.  **Dependency Injection (Hard Context):**
    *   The prompt includes the *signatures* (interfaces) of immediate dependencies, but not their full implementations. This provides strict type boundaries without consuming token budget.
2.  **Dynamic Style Retrieval (Soft Context via RAG):**
    *   *The "Bootstrap" Layer:* (Optional) The Vector Store is pre-seeded with a "Golden Reference" repository (e.g., an existing compliant Java service) to establish baseline patterns (e.g., "Use Lombok for DTOs").
    *   *The "Snowball" Layer:* As files are successfully translated and verified in Phase 2, they are immediately embedded and indexed.
    *   *Runtime Query:* When translating `Main.java`, the agent queries the Vector Store for "error handling patterns" found in the already-translated `Utils.java`. This ensures that as the project grows, later files match the style of earlier files.

## 17.2 Prompt Engineering
Prompts are structured as "Expert System Instructions" rather than open-ended queries:

1.  **Role Definition:** "You are a Senior Java Migration Architect specialized in strict type safety."
2.  **Task & Constraints:** "Translate the following Python function. You must use the `java.nio` library for file I/O. Do not use raw `File` objects."
3.  **Context Injection:** "Here are the interfaces for the 3 functions called by this code: [Sig1, Sig2, Sig3]."
4.  **Style Examples (RAG):** "Reference the following 2 examples from the current project for how to handle `KeyErrors`: [Example Snippet A, Example Snippet B]."
5.  **Source Input:** The raw Python AST/Code.

---

# 18. Cross-Language Bridges

Used solely for verification (checking equivalence).

1.  **Python calling Java:** Use `JPype` or `subprocess`.
    *   *Why:* To test if generated Java code returns correct values for Python inputs.
2.  **Java calling Python:** Use `ProcessBuilder` (subprocess) passing JSON via `stdin/stdout`.
    *   *Why:* For "Mocking" fallback—letting Java call the original Python implementation.

---

# 19. Error Handling & Recovery

| Error Type | Strategy |
|:---|:---|
| **Context Limit Exceeded** | Summarize dependencies; split file into smaller chunks. |
| **Compilation Error** | Feed compiler output to LLM; retry (Decrement Phase 2 budget). |
| **I/O Mismatch** | Feed inputs/actual/expected to LLM; retry body (Decrement Phase 3 budget). |
| **Infinite Loop in Logic** | Timeout execution; revert to previous version or Mock. |
| **Dependency Missing** | Pause; Ask human to add library to `pom.xml`. |

---

# 20. Project Structure (Detailed)

```text
codemorph/
├── codemorph.yaml             # Default config
├── src/
│   ├── analyzer/              # Phase 1: AST & Graph
│   │   ├── parser.py
│   │   └── graph_builder.py
│   ├── translator/            # LLM Logic
│   │   ├── prompt_engine.py
│   │   └── llm_client.py
│   ├── verifier/              # Phases 2 & 3
│   │   ├── compiler.py        # javac wrapper
│   │   ├── bridge.py          # Py<->Java Interop
│   │   └── io_checker.py      # Equivalence logic
│   ├── knowledge/             # Static Data
│   │   ├── library_maps/
│   │   └── feature_rules/
│   └── main.py                # Entry point
└── tests/                     # Meta-tests (testing the translator)
```

---

# 21. Data Models

```python
# pydantic models

class CodeFragment(BaseModel):
    id: str                  # unique path "module::func"
    source_code: str
    ast_type: str            # "function", "class", "method"
    dependencies: List[str]  # IDs of other fragments
    is_translated: bool

class AnalysisResult(BaseModel):
    fragments: Dict[str, CodeFragment]
    dependency_graph: Any    # NetworkX object serialized
    translation_queue: List[str]

class TranslationCandidate(BaseModel):
    fragment_id: str
    target_code: str
    compilation_success: bool
    io_equivalence_success: bool
    error_log: Optional[str]
```

---

# 22. CLI Reference

```bash
# 1. Analyze Only (Generates Plan)
codemorph analyze --source ./py-app --output ./plan.json

# 2. Execute Plan
codemorph run --plan ./plan.json --target-dir ./java-app

# 3. Verify Specific File
codemorph verify --target-file ./java-app/src/Utils.java --source-file ./py-app/utils.py

# 4. Interactive Mode (Standard)
codemorph translate . --interactive
```

---

# 23. Implementation Phases

1.  **Weeks 1-2 (Core):** AST Parsing, Dependency Graph, simple "Hello World" function translation.
2.  **Weeks 3-4 (Type-Driven):** Java Compilation harness, Feature Mapping rules, Type Bridge.
3.  **Weeks 5-6 (Semantics):** Test instrumentation (Snapshots), I/O Equivalence Checker.
4.  **Weeks 7-8 (Scale):** Handling full files, imports, package structures, circular dependency resolution.
5.  **Weeks 9-10 (Refinement):** UI/CLI, Retry logic, Library mapping expansion.

---

# 24. Testing Strategy

We are building a tool to test code, so we must test the tool rigorously.

1.  **Golden Set:** Create a repository of 50 common Python algorithms (Sorting, API calls, String manipulation) with known perfect Java translations.
2.  **Regression Testing:** Ensure the agent achieves 100% equivalence on the Golden Set.
3.  **Integration Testing:** Run the agent on a real-world open-source library (e.g., a small utility lib like `python-slugify`) and verify the output Java compiles and passes generated tests.

---

# 25. Example Workflows

## Scenario: Translating a utility function

**Source (Python):**
```python
def calculate_total(items, tax_rate=0.05):
    total = sum(item['price'] for item in items)
    return total * (1 + tax_rate)
```

**Agent Steps:**
1.  **Analyze:** Identifies `calculate_total`. Input: List of Dicts. Output: Float.
2.  **Feature Map:** Detects `item['price']` (Map access) and `sum(...)` (Aggregation).
3.  **Draft (LLM):** Generates Java method using `List<Map<String, Double>>`.
4.  **Type Check:** Compiles. Checks if Python `[{'price': 10.0}]` is compatible with Java input. **Pass.**
5.  **Semantics:**
    *   Python Run: Input `[{'price':100}]` -> Output `105.0`.
    *   Java Run: Input `[{'price':100}]` -> Output `105.0`.
6.  **Result:**
```java
public double calculateTotal(List<Map<String, Double>> items, double taxRate) {
    double total = items.stream()
        .mapToDouble(item -> item.get("price"))
        .sum();
    return total * (1 + taxRate);
}
```
