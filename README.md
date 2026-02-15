# CodeMorph v1.0

Bidirectional code translation and version upgrades for Python and Java, powered by LLMs.

## Overview

CodeMorph is an intelligent code translation system that can:

- **Cross-Language Translation**: Convert code between Python and Java
- **Version Upgrades**: Upgrade code within the same language (e.g., Python 2.7 → Python 3.10, Java 11 → Java 21)
- **Test-Driven Verification**: Use unit tests as ground truth for semantic equivalence
- **Human-in-the-Loop**: Checkpoints for review and approval at critical stages

## Architecture

CodeMorph uses a three-phase approach inspired by the Oxidizer paper:

### Phase 1: Project Partitioning
- Parse source code into AST
- Extract code fragments (functions, classes, methods)
- Build dependency graph
- Determine translation order

### Phase 2: Type-Driven Translation
- Apply feature mapping rules
- Generate initial translation using LLM
- Verify compilation
- Check type compatibility

### Phase 3: Semantics-Driven Translation
- Collect execution snapshots from tests
- Verify I/O equivalence
- Refine translation based on mismatches
- Mock untranslatable functions

## Project Structure

```
code-convert/
├── src/codemorph/
│   ├── cli/                    # Command-line interface
│   │   └── main.py             # Entry point with typer
│   ├── config/                 # Configuration system
│   │   ├── models.py           # Pydantic models
│   │   └── loader.py           # Config loading/validation
│   ├── languages/              # Language plugin system
│   │   ├── base/
│   │   │   └── plugin.py       # Abstract base class
│   │   ├── python/
│   │   │   └── plugin.py       # Python language plugin
│   │   ├── java/
│   │   │   └── plugin.py       # Java language plugin
│   │   └── registry.py         # Plugin registry
│   ├── analyzer/               # Phase 1: Analysis
│   │   └── graph_builder.py   # Dependency graph builder
│   ├── translator/             # Phase 2 & 3: Translation
│   │   └── llm_client.py       # Ollama LLM client
│   ├── verifier/               # Verification (TODO)
│   ├── bridges/                # Cross-language bridges (TODO)
│   ├── state/                  # State persistence (TODO)
│   └── knowledge/              # Feature maps & library maps (TODO)
├── examples/
│   └── python_project/
│       ├── calculator.py       # Sample Python project
│       └── test_calculator.py  # Unit tests
├── tests/                      # CodeMorph's own tests
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Java JDK 11+ (for Java compilation/execution)
- Ollama (for local LLM)

### Steps

1. **Install Ollama**:
   ```bash
   # Visit https://ollama.ai and follow installation instructions
   # Then pull a code model:
   ollama pull deepseek-coder:6.7b
   ```

2. **Install CodeMorph**:
   ```bash
   cd code-convert
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. **Verify Installation**:
   ```bash
   codemorph --help
   codemorph doctor  # Check dependencies
   ```

## Usage

### Generate Default Configuration

```bash
codemorph init --output codemorph.yaml
```

### Translate Python → Java

```bash
codemorph translate \
    ./examples/python_project \
    --target-lang java \
    --target-version 17 \
    --build-system gradle \
    --output ./output/java_calculator
```

### Upgrade Python 2.7 → Python 3.10

```bash
codemorph translate \
    ./legacy_project \
    --source-lang python \
    --source-version 2.7 \
    --target-lang python \
    --target-version 3.10 \
    --output ./upgraded_project
```

### Use Custom Configuration

```bash
codemorph translate ./my_project --config codemorph.yaml
```

## Configuration

Example `codemorph.yaml`:

```yaml
project:
  name: "MyProject"
  source:
    language: "python"
    version: "3.10"
    root: "./src"
    test_root: "./tests"
  target:
    language: "java"
    version: "17"
    output_dir: "./output"
    build_system: "gradle"

llm:
  host: "http://localhost:11434"
  model: "deepseek-coder:6.7b"
  temperature: 0.2

translation:
  max_retries_type_check: 15
  max_retries_semantics: 5
  allow_mocking: true

checkpoint_mode: "batch"  # interactive, batch, or auto
```

## Development Status

### ✅ Completed

- [x] Project structure with modular plugin architecture
- [x] Configuration system (YAML + Pydantic)
- [x] CLI skeleton with Typer
- [x] Language plugin abstraction
- [x] Python language plugin (AST parsing, fragment extraction)
- [x] Java language plugin (tree-sitter parsing, fragment extraction)
- [x] Dependency graph builder
- [x] Ollama LLM client
- [x] Example Python project for testing

### 🚧 In Progress

- [ ] State persistence layer
- [ ] Feature mapping system
- [ ] Type compatibility checker
- [ ] Cross-language bridges (JPype, Py4J)
- [ ] Execution snapshot capture
- [ ] I/O equivalence verification
- [ ] Human-in-the-loop checkpoints
- [ ] Main orchestrator (connects all phases)

### 📋 Planned

- [ ] Library mapping system
- [ ] Symbol registry
- [ ] RAG integration (ChromaDB + embeddings)
- [ ] Comprehensive test suite
- [ ] Documentation & tutorials
- [ ] Support for additional languages (Go, Rust, etc.)

## Testing the Current Implementation

While the full translation pipeline is not yet complete, you can test individual components:

```python
# Test Python AST parsing
from pathlib import Path
from codemorph.languages.python.plugin import PythonPlugin

plugin = PythonPlugin(version="3.10")
ast = plugin.parse_file(Path("examples/python_project/calculator.py"))
fragments = plugin.extract_fragments(Path("examples/python_project/calculator.py"), ast)

for fragment in fragments:
    print(f"{fragment.id}: {fragment.fragment_type} ({fragment.start_line}-{fragment.end_line})")
```

## Contributing

This is an internal tool under active development. Key areas that need work:

1. **Cross-Language Bridges**: Implementing JPype/Py4J for runtime verification
2. **Feature Mapping Rules**: Creating comprehensive rule sets for Python ↔ Java
3. **Execution Snapshots**: Instrumentation for capturing test I/O
4. **State Persistence**: Resumable translation sessions
5. **Testing**: Unit and integration tests for all components

## License

Internal use only.

## References

- [Oxidizer Paper](https://arxiv.org/abs/2306.03894) - Inspiration for the approach
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Tree-sitter](https://tree-sitter.github.io) - AST parsing
- [Pydantic](https://docs.pydantic.dev) - Configuration validation

## Contact

For questions or issues, please contact the development team.
