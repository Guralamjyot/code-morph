# CodeMorph - Quick Start Guide

This guide will help you get CodeMorph up and running in 5 minutes.

## Prerequisites Check

Before starting, ensure you have:

- [ ] Python 3.10 or higher
- [ ] Java JDK 11+ (only if translating to/from Java)
- [ ] Ollama installed and running

## Step 1: Install Ollama

If you haven't installed Ollama yet:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai for other platforms
```

Start Ollama and pull a code model:

```bash
ollama serve  # Start Ollama server (in a separate terminal)
ollama pull deepseek-coder:6.7b  # Pull the default model (~3.8GB)
```

Verify Ollama is running:

```bash
ollama list  # Should show deepseek-coder:6.7b
```

## Step 2: Install CodeMorph

```bash
cd /Users/alamvirk/git_clones/code-convert

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CodeMorph
pip install -e .
```

Verify installation:

```bash
codemorph --version
```

## Step 3: Test the Components

Run the integration tests to verify everything works:

```bash
pytest tests/integration/test_basic_workflow.py -v
```

You should see all tests passing.

## Step 4: Try the Example

### Parse and Analyze a Python Project

```python
from pathlib import Path
from codemorph.languages.registry import LanguagePluginRegistry
from codemorph.config.models import LanguageType
from codemorph.analyzer.graph_builder import DependencyGraphBuilder

# Get the Python plugin
plugin = LanguagePluginRegistry.get_plugin(LanguageType.PYTHON, "3.10")

# Parse the example calculator
example_file = Path("examples/python_project/calculator.py")
ast = plugin.parse_file(example_file)

# Extract code fragments
fragments = plugin.extract_fragments(example_file, ast)

print(f"Found {len(fragments)} code fragments:")
for fragment in fragments:
    print(f"  - {fragment.name} ({fragment.fragment_type.value})")

# Build dependency graph
fragment_dict = {f.id: f for f in fragments}
graph_builder = DependencyGraphBuilder()
graph = graph_builder.build_graph(fragment_dict)

print(f"\nDependency graph has {graph.number_of_nodes()} nodes")
```

Save this as `test_parsing.py` and run it:

```bash
python test_parsing.py
```

### Generate a Configuration File

```bash
codemorph init --output my_config.yaml
```

Edit the generated `my_config.yaml` to customize settings.

## Step 5: Next Steps

The core translation engine is still under development. Current working components:

✅ **Working Now:**
- AST parsing (Python and Java)
- Fragment extraction
- Dependency graph building
- LLM client (Ollama integration)
- Configuration system
- CLI framework

🚧 **In Progress:**
- State persistence
- Feature mapping rules
- Type compatibility checking
- I/O equivalence verification
- Full translation orchestrator

## Troubleshooting

### "ollama not found"

Ensure Ollama is installed and in your PATH:

```bash
which ollama  # Should show path to ollama binary
```

### "tree-sitter-java not found"

Install the missing dependency:

```bash
pip install tree-sitter-java
```

### "Cannot connect to Ollama"

Ensure Ollama server is running:

```bash
# In a separate terminal
ollama serve
```

### Tests fail with import errors

Ensure you've installed CodeMorph in development mode:

```bash
pip install -e .
```

## Development Workflow

To work on CodeMorph:

1. **Make changes** to the source code in `src/codemorph/`
2. **Run tests** to verify your changes:
   ```bash
   pytest tests/ -v
   ```
3. **Format code** with black:
   ```bash
   black src/
   ```
4. **Lint** with ruff:
   ```bash
   ruff check src/
   ```

## Useful Commands

```bash
# Check system dependencies
codemorph doctor

# Generate default config
codemorph init

# See all available commands
codemorph --help

# Run with verbose output
codemorph translate ./my_project --verbose

# Use custom config
codemorph translate ./my_project --config custom.yaml
```

## What's Missing (Contributions Welcome!)

The following components need implementation:

1. **State Persistence Layer** - Save/restore translation state
2. **Feature Mapping Rules** - Language-specific transformation rules
3. **Type Compatibility Checker** - Verify type mappings work
4. **Cross-Language Bridges** - JPype/Py4J integration for verification
5. **Execution Snapshot System** - Capture test I/O for equivalence checking
6. **Main Orchestrator** - Tie all phases together
7. **Human-in-the-Loop UI** - Interactive checkpoint system

## Getting Help

For issues or questions:
1. Check the [README.md](README.md) for detailed architecture
2. Review the [CLAUDE.md](claude.md) for the complete implementation plan
3. Examine test files in `tests/` for usage examples

Happy translating! 🚀
