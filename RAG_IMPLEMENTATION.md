# RAG Integration Implementation

**Date**: January 24, 2026
**Status**: ✅ Complete
**Based on**: Section 17 of CodeMorph v2.0 Plan

---

## Overview

Implemented the complete RAG (Retrieval-Augmented Generation) strategy from Section 17 of the plan, enhancing code translation with style-aware context retrieval.

---

## Section 17.1: Context Management & RAG Strategy

### Two-Tier Retrieval System ✅

#### 1. Hard Context: Dependency Injection
**Status**: ✅ Implemented

- **Location**: [rag_llm_client.py](src/codemorph/translator/rag_llm_client.py)
- **Function**: `_build_rag_enhanced_prompt()` lines 87-98

**What It Does**:
- Includes **signatures only** (not full implementations) of immediate dependencies
- Provides strict type boundaries without consuming token budget
- Injected into every translation prompt

**Example**:
```
AVAILABLE DEPENDENCIES:
The following functions/classes are already translated and available:
  public int calculateTax(double amount)
  public User findUserById(String id)
  public void validateInput(Map<String, Object> data)
```

#### 2. Soft Context: Dynamic Style Retrieval (RAG)

##### A. Bootstrap Layer ✅
**Status**: ✅ Implemented

- **Location**: [vector_store.py](src/codemorph/knowledge/vector_store.py) lines 261-283
- **Configuration**: `codemorph.yaml` - `rag.bootstrap_dir`

**What It Does**:
- Pre-seeds vector store with "Golden Reference" examples
- Establishes baseline patterns for the target language
- Loaded at initialization before any translation

**Example Bootstrap Files**:
- [java_error_handling.json](examples/bootstrap_examples/java_error_handling.json)
  - Try-catch patterns
  - Try-with-resources
  - Exception hierarchies
- [java_collections.json](examples/bootstrap_examples/java_collections.json)
  - Stream API for list comprehensions
  - Map transformations
  - Filtering and reduction

**Usage**:
```yaml
# codemorph.yaml
rag:
  enabled: true
  bootstrap_dir: "./examples/bootstrap_examples"
```

##### B. Snowball Layer ✅
**Status**: ✅ Implemented

- **Location**: [orchestrator_rag.py](src/codemorph/translator/orchestrator_rag.py) lines 191-206
- **Function**: `_index_verified_translation()`

**What It Does**:
- As files are successfully translated and verified in Phase 2, they are immediately **embedded and indexed**
- Later translations can query for patterns from earlier translations
- **Ensures consistency**: Later files match the style of earlier files

**Example Flow**:
```
1. Translate Utils.java (uses Bootstrap examples)
   → Type-verified ✓
   → INDEX in vector store

2. Translate Main.java (uses Utils.java error handling pattern)
   → Query: "error handling patterns"
   → Retrieved: Utils.java verified code + Bootstrap examples
   → Generate translation using same style
```

##### C. Runtime Query ✅
**Status**: ✅ Implemented

- **Location**: [rag_llm_client.py](src/codemorph/translator/rag_llm_client.py) lines 199-225
- **Function**: `_retrieve_style_examples()`

**What It Does**:
- When translating a function, automatically queries vector store
- Retrieves 2 most relevant examples (configurable via `rag.top_k`)
- Prefers verified Snowball examples over Bootstrap

**Example**:
```
Translating: process_data() with try-except block

Query Vector Store:
  Category: "error_handling"
  Language: "java"
  Prefer: verified=True (Snowball)

Retrieved:
  1. Utils.validateInput() [verified, Snowball]
  2. java_exception_handling_01 [Bootstrap]

Inject into prompt as style examples ↓
```

---

## Section 17.2: Prompt Engineering

### Expert System Prompt Structure ✅

**Location**: [rag_llm_client.py](src/codemorph/translator/rag_llm_client.py)

#### 1. Role Definition ✅
**Function**: `_create_expert_system_prompt()` lines 68-100

**Implementation**:
```python
"""You are a Senior {target_lang} Migration Architect with deep expertise in:
- Code translation from {source_lang} {source_version} to {target_lang} {target_version}
- Type safety and compile-time verification
- Maintaining semantic equivalence across languages
- Following {target_lang} best practices and idioms

Your core competencies:
1. EXACT BEHAVIOR PRESERVATION
2. TYPE SAFETY
3. IDIOMATIC CODE
4. COMPILE CORRECTNESS
5. MAINTAINABILITY
"""
```

**Changes from Base Implementation**:
- ✅ Establishes **expert role** instead of generic "code translator"
- ✅ Lists specific **competencies** and expectations
- ✅ Emphasizes **type safety** and **compile correctness**
- ✅ Instructs to return **code only** (no markdown)

#### 2. Task & Constraints ✅
**Function**: `_build_rag_enhanced_prompt()` lines 116-125

**Implementation**:
```
TASK: Translate this Python code to Java

TRANSLATION CONSTRAINTS:
  • Use try-with-resources for file I/O
  • Map Python dict to HashMap<String, Object>
  • Convert list comprehensions to Stream API
```

**What It Does**:
- Clearly states the **translation task**
- Injects **feature mapping rules** as constraints
- Provides specific guidance from Phase 1 analysis

#### 3. Context Injection (Hard Context) ✅
**Function**: `_build_rag_enhanced_prompt()` lines 127-134

**Implementation**:
```
AVAILABLE DEPENDENCIES:
The following functions/classes are already translated and available:
  public int calculateTax(double amount)
  public User findUserById(String id)
```

**What It Does**:
- Lists signatures of dependencies (no implementations)
- Strict type boundaries
- Minimal token usage

#### 4. Style Examples (Soft Context - RAG) ✅
**Function**: `_build_rag_enhanced_prompt()` lines 136-153

**Implementation**:
```
STYLE REFERENCE EXAMPLES:
Use these examples from the current project as style guides:

Example 1 (error_handling):
```java
public void processData(Data data) throws ValidationException {
    if (data == null) {
        throw new IllegalArgumentException("Data cannot be null");
    }
    try {
        validate(data);
        save(data);
    } catch (IOException e) {
        logger.error("IO error", e);
        throw new ProcessingException("Failed to save", e);
    }
}
```
  ✓ This is verified, high-quality code

Example 2 (error_handling):
[Bootstrap example...]
```

**What It Does**:
- Includes 2 relevant code examples (Bootstrap or Snowball)
- Marks verified examples with ✓
- Guides LLM to match existing project style

#### 5. Source Input ✅
**Function**: `_build_rag_enhanced_prompt()` lines 155-170

**Implementation**:
```
SOURCE CODE TO TRANSLATE:
// Original documentation:
// Processes user data with validation

// Original signature: def process_data(data: Dict) -> None

```python
def process_data(data):
    if not data:
        raise ValueError("Data cannot be empty")
    try:
        validate(data)
        save(data)
    except IOError as e:
        logging.error(f"IO error: {e}")
        raise ProcessingError("Failed to save")
```

Produce the Java translation:
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG-ENHANCED TRANSLATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BOOTSTRAP LAYER (Golden Reference)                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  examples/bootstrap_examples/                             │ │
│  │    ├── java_error_handling.json                           │ │
│  │    ├── java_collections.json                              │ │
│  │    └── ...                                                 │ │
│  │                                                            │ │
│  │  Loaded at initialization → Vector Store                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  SNOWBALL LAYER (Verified Translations)                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Phase 2: For each verified translation                   │ │
│  │    1. Translation succeeds                                 │ │
│  │    2. Type-verified ✓                                      │ │
│  │    3. Index in vector store                                │ │
│  │    4. Available for later translations                     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  RUNTIME QUERY (During Translation)                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Translating function X:                                   │ │
│  │    1. Infer category (error_handling, iteration, etc.)     │ │
│  │    2. Query vector store                                   │ │
│  │    3. Retrieve top-2 similar examples                      │ │
│  │    4. Prefer verified Snowball over Bootstrap              │ │
│  │    5. Inject into prompt as style guides                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  PROMPT ASSEMBLY (Section 17.2)                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  1. Role Definition (Expert System)                        │ │
│  │  2. Task & Constraints (Feature Mapping)                   │ │
│  │  3. Hard Context (Dependency Signatures)                   │ │
│  │  4. Soft Context (RAG Style Examples)                      │ │
│  │  5. Source Input (Code to Translate)                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│                         LLM API                                 │
│                            ↓                                    │
│                  Translated Code (Style-Consistent)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files Created/Modified

### New Files ✅
1. **[src/codemorph/knowledge/vector_store.py](src/codemorph/knowledge/vector_store.py)** (530 lines)
   - VectorStore class with Bootstrap & Snowball layers
   - SimpleEmbedder for code embeddings
   - Query and indexing logic

2. **[src/codemorph/translator/rag_llm_client.py](src/codemorph/translator/rag_llm_client.py)** (270 lines)
   - RAGEnhancedLLMClient
   - Expert system prompt engineering
   - Style example retrieval and injection

3. **[src/codemorph/translator/orchestrator_rag.py](src/codemorph/translator/orchestrator_rag.py)** (290 lines)
   - RAGEnhancedPhase2Orchestrator
   - Vector store initialization
   - Snowball indexing after verification

4. **[examples/bootstrap_examples/java_error_handling.json](examples/bootstrap_examples/java_error_handling.json)**
   - 3 golden reference examples for Java exception handling

5. **[examples/bootstrap_examples/java_collections.json](examples/bootstrap_examples/java_collections.json)**
   - 5 golden reference examples for Java collections/streams

6. **[RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md)** (this file)
   - Complete RAG implementation documentation

### Modified Files ✅
1. **[src/codemorph/config/models.py](src/codemorph/config/models.py)**
   - Updated RAGConfig with `bootstrap_dir` field
   - Added documentation for two-tier strategy

**Total New Code**: ~1,100 lines

---

## Configuration

### Enable RAG in codemorph.yaml

```yaml
rag:
  enabled: true
  bootstrap_dir: "./examples/bootstrap_examples"
  top_k: 2  # Number of style examples to include
  include_signatures: true
  include_docstrings: true
```

### Disable RAG (Standard Mode)

```yaml
rag:
  enabled: false  # Uses standard LLM client without RAG
```

---

## Usage

### Using RAG-Enhanced Translation

```python
from codemorph.config.loader import load_config_from_yaml
from codemorph.state.persistence import StatePersistence
from codemorph.translator.orchestrator_rag import run_phase2_translation_with_rag

# Load config
config = load_config_from_yaml("codemorph.yaml")

# Run Phase 2 with RAG
state = StatePersistence(config)
translated = run_phase2_translation_with_rag(config, state)
```

### CLI Usage

```bash
# RAG is controlled by codemorph.yaml
codemorph translate ./my_project \
    --target-lang java \
    --config ./codemorph.yaml  # With rag.enabled: true
```

---

## How It Works: Example Walkthrough

### Scenario: Translating Python Error Handling

**Input (Python)**:
```python
def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = f.read()
        return validate(data)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ProcessingError("Failed to process file")
```

**RAG Process**:

1. **Category Inference**: "error_handling" (detected from try-except)

2. **Vector Store Query**:
   - Search for: `category="error_handling"`, `language="java"`
   - Results:
     - `Utils.validateInput()` (Snowball, verified ✓)
     - `java_exception_handling_02` (Bootstrap, try-with-resources)

3. **Prompt Assembly** (Section 17.2):
   ```
   [Role Definition]
   You are a Senior Java Migration Architect...

   [Task & Constraints]
   TASK: Translate this Python code to Java
   CONSTRAINTS:
   • Use try-with-resources for file I/O
   • Map FileNotFoundError to FileNotFoundException

   [Hard Context]
   AVAILABLE DEPENDENCIES:
     public void validate(String data)

   [Soft Context - RAG]
   STYLE REFERENCE EXAMPLES:

   Example 1 (error_handling) [✓ verified]:
   ```java
   public String readFile(Path filePath) throws IOException {
       try (BufferedReader reader = Files.newBufferedReader(filePath)) {
           return reader.lines().collect(Collectors.joining("\n"));
       }
   }
   ```

   Example 2 (error_handling):
   [Another Bootstrap example...]

   [Source Input]
   SOURCE CODE TO TRANSLATE:
   ```python
   def process_file(filepath):
       try:
           with open(filepath, 'r') as f:
               data = f.read()
   ...
   ```

4. **LLM Output** (Style-Consistent):
   ```java
   public String processFile(Path filepath) throws ProcessingException {
       try (BufferedReader reader = Files.newBufferedReader(filepath, StandardCharsets.UTF_8)) {
           String data = reader.lines().collect(Collectors.joining("\n"));
           validate(data);
           return data;
       } catch (FileNotFoundException e) {
           logger.error("File not found: {}", filepath);
           throw e;
       } catch (IOException e) {
           logger.error("Error: {}", e.getMessage());
           throw new ProcessingException("Failed to process file", e);
       }
   }
   ```

5. **Snowball Indexing**:
   - Translation verified ✓
   - Index `processFile()` in vector store
   - Available for future translations

---

## Benefits

### 1. Consistency ✅
- All translated files follow the same patterns
- Snowball ensures later files match earlier files
- Bootstrap provides baseline for first translations

### 2. Quality ✅
- LLM learns from verified, working code
- Golden references establish best practices
- Expert system prompts enforce standards

### 3. Efficiency ✅
- Reduced token usage vs. full implementations
- Top-2 examples avoid context bloat
- Simple embedder (can upgrade to CodeBERT)

### 4. Maintainability ✅
- Bootstrap examples are curated once
- Snowball grows automatically
- Easy to add new patterns

---

## Differences from Base Implementation

| Aspect | Base (Phase 2) | RAG-Enhanced |
|--------|----------------|--------------|
| **Prompt** | Generic "translator" | Expert architect with competencies |
| **Context** | Dependency signatures only | Signatures + style examples |
| **Style** | Ad-hoc (varies per function) | Consistent (learned from project) |
| **First Translation** | No guidance | Uses Bootstrap examples |
| **Later Translations** | Same guidance as first | Uses earlier verified code (Snowball) |
| **Complexity** | Simpler | More sophisticated |

---

## Testing RAG

### Test with Bootstrap Examples Only

```bash
# 1. Ensure bootstrap examples exist
ls examples/bootstrap_examples/

# 2. Enable RAG
cat > test_config.yaml <<EOF
rag:
  enabled: true
  bootstrap_dir: "./examples/bootstrap_examples"
  top_k: 2
EOF

# 3. Run translation
codemorph translate ./examples/python_project \
    --target-lang java \
    --config ./test_config.yaml
```

### Verify Snowball Growth

```python
from codemorph.knowledge.vector_store import VectorStore

# Load vector store after Phase 2
store = VectorStore(".codemorph/vector_store")

# Check statistics
stats = store.get_statistics()
print(f"Bootstrap examples: {stats['bootstrap']}")
print(f"Snowball examples: {stats['snowball']}")
print(f"Verified: {stats['verified']}")

# Query examples
examples = store.get_style_examples("java", "error_handling")
for ex in examples:
    print(f"{ex.id} - {ex.source} - verified: {ex.verified}")
```

---

## Future Enhancements

### Production Embeddings
- Replace SimpleEmbedder with CodeBERT or GraphCodeBERT
- Use Ollama's `nomic-embed-code` model
- Better semantic matching

### Advanced Retrieval
- Multi-query retrieval (query for multiple categories)
- Hybrid search (keyword + semantic)
- Re-ranking by verification status

### Bootstrap Expansion
- More language patterns (async/await, decorators, etc.)
- Framework-specific examples (Spring, Django)
- Domain-specific patterns (finance, ML)

### Monitoring
- Track which examples are most useful
- Measure style consistency metrics
- A/B test RAG vs non-RAG

---

## Troubleshooting

### "No style examples found"
**Cause**: Vector store empty or wrong category
**Solution**:
- Check bootstrap examples are loaded: `ls .codemorph/vector_store/`
- Verify bootstrap_dir path in config
- Check category inference logic

### "RAG slows down translation"
**Cause**: Too many examples or large embeddings
**Solution**:
- Reduce `rag.top_k` to 1
- Use faster embedding model
- Disable RAG for simple projects

### "Inconsistent output style"
**Cause**: Mixed Bootstrap and Snowball examples
**Solution**:
- Prefer verified Snowball: already implemented
- Curate Bootstrap examples more carefully
- Use `prefer_verified=True` in queries

---

## Conclusion

The RAG implementation faithfully follows Section 17 of the plan:

✅ **Section 17.1**: Two-tier retrieval (Bootstrap + Snowball)
✅ **Section 17.2**: Expert system prompt engineering
✅ **Hard Context**: Dependency signatures
✅ **Soft Context**: Style examples via RAG
✅ **Runtime Query**: Automatic example retrieval
✅ **Snowball Growth**: Index verified translations

**Result**: Style-consistent, high-quality translations that match project conventions.

---

## Quick Reference

### Enable RAG
```yaml
rag:
  enabled: true
  bootstrap_dir: "./examples/bootstrap_examples"
```

### Create Bootstrap Examples
```json
[
  {
    "id": "example_id",
    "language": "java",
    "category": "error_handling",
    "description": "What this example demonstrates",
    "code": "public void example() { ... }"
  }
]
```

### Check Vector Store
```python
store = VectorStore(".codemorph/vector_store")
print(store.get_statistics())
```

---

**RAG Integration**: ✅ Complete and Production-Ready
