# CodeMorph Simplification Plan
## Aligning Implementation with Simplified Spec

**Date:** 2026-01-27
**Status:** In Progress

---

## Overview

This document outlines the simplifications needed to align the current CodeMorph implementation with the streamlined specification from SPEC.md.

## Key Simplifications

### 1. Configuration System ✓

**Current State:**
- Complex Pydantic models with many optional fields
- Multiple config classes (RAGConfig, VerificationConfig, TranslationConfig, etc.)
- Over 400 lines in config/models.py

**Simplified Approach (from spec):**
```yaml
project:
  name: "MyProject"
  source_language: "python"
  target_language: "java"
  source_root: "./src"
  test_root: "./tests"
  output_root: "./output"

llm:
  provider: "openrouter"
  model: "anthropic/claude-3.5-sonnet"
  api_key_env: "OPENROUTER_API_KEY"
  temperature: 0.2

translation:
  max_retries_phase2: 15
  max_retries_phase3: 5
  allow_mocking: true
  generate_tests_if_missing: true

checkpoints:
  mode: "interactive"  # interactive | batch | auto
```

**Actions:**
- [ ] Remove unnecessary config fields
- [ ] Consolidate related configs
- [ ] Keep only essential options from spec

---

### 2. LLM Client Unification ✓

**Current State:**
- Separate `OllamaClient` and `OpenRouterClient` classes
- Duplicate code between clients
- Conversation history maintained per-method call

**Simplified Approach:**
- **Spec Emphasis:** "Conversational agent maintains full history across retries"
- Single unified client with pluggable backends
- Persistent conversation object passed through retry loops

**Actions:**
- [ ] Create unified `LLMClient` base class
- [ ] Move common logic to base
- [ ] Ensure conversation history persists across ALL retries in Phase 2 and Phase 3
- [ ] Simplify client factory

---

### 3. Cross-Language Bridge Simplification 🔴 **PRIORITY**

**Current State:**
```python
dependencies = [
    "jpype1>=1.4.0",  # Heavy native dependency
    "py4j>=0.10.0",   # Complex bidirectional bridge
]
```

**Spec Recommendation:**
> "Default: `subprocess + JSON` (works everywhere, easiest to debug)"

**Why Simplify:**
- JPype/Py4J require Java runtime and complex setup
- subprocess + JSON is universal, portable, easier to debug
- Spec explicitly recommends this as default

**Simplified Approach:**
```python
# bridges/subprocess_bridge.py
class SubprocessBridge:
    """Universal cross-language bridge using subprocess + JSON."""

    def execute_java(self, class_name: str, method: str, args: dict) -> Any:
        """Execute Java code via subprocess, passing JSON via stdin."""
        json_input = json.dumps(args)
        result = subprocess.run(
            ["java", "-cp", classpath, "BridgeMain", class_name, method],
            input=json_input,
            capture_output=True,
            text=True
        )
        return json.loads(result.stdout)
```

**Actions:**
- [ ] Create `bridges/subprocess_bridge.py` as default
- [ ] Move JPype/Py4J to optional dependencies
- [ ] Update Phase 3 I/O verification to use subprocess bridge
- [ ] Update mocking system to use subprocess bridge

---

### 4. Type Compatibility Checker 🔴 **PRIORITY**

**Current State:**
- `verifier/type_checker.py` exists but may not implement JSON round-trip
- Spec is very specific about this approach

**Spec Approach (Section 10.1):**
```python
def check_type_compat(py_val, java_type_signature):
    # 1. Serialize Python Value
    json_val = json.dumps(py_val)

    # 2. Attempt Java Deserialization (using Jackson/Gson)
    valid = java_bridge.can_deserialize(json_val, java_type_signature)

    return valid
```

**Actions:**
- [ ] Review current type_checker.py implementation
- [ ] Ensure it uses JSON serialization round-trip
- [ ] Create Java harness for deserialization testing
- [ ] Integrate with subprocess bridge

---

### 5. Incremental File Compilation

**Current Approach:**
- Phase 2 compiles individual fragments in temp directories
- Accumulation into final files happens later

**Spec Approach (Section 4.2):**
> "Functions accumulate into the target file. After each addition:
> - **Java:** Run `javac` on the complete file (functions wrapped in class)
> - **Python:** Run `py_compile` or `ast.parse` for syntax validation"

**Why This Matters:**
- Ensures imports and cross-references remain valid
- Catches integration issues early
- Matches the "incremental validation" principle

**Actions:**
- [ ] Modify Phase 2 to accumulate code in target files
- [ ] Compile entire file after each fragment addition
- [ ] Feed cumulative errors back to agent

---

### 6. Feature Mapping Simplification

**Current State:**
- `knowledge/feature_mapper.py` exists
- Feature rules in `knowledge/feature_rules/python_to_java.yaml`

**Spec Lists Only Essential Mappings:**
- List comprehension → Stream API
- Context manager → try-with-resources
- Multiple inheritance → interfaces + composition
- Decorators → wrapper pattern or AOP
- **kwargs → Map<String, Object> or Builder
- Duck typing → explicit interface

**Actions:**
- [ ] Review existing feature rules
- [ ] Remove overly complex rules
- [ ] Keep only the 6 core mappings from spec
- [ ] Simplify validation logic

---

### 7. RAG System Alignment

**Current State:**
- RAG implemented with ChromaDB
- Bootstrap and snowball layers may exist

**Spec Approach (Section 17.1):**
- **Bootstrap Layer:** Pre-seed with golden reference examples
- **Snowball Layer:** Index verified translations from Phase 2
- Query for style-consistent examples (top_k=2)

**Actions:**
- [ ] Verify RAG matches two-tier approach
- [ ] Simplify embedding logic if over-engineered
- [ ] Make RAG truly optional (disabled by default)

---

### 8. Dependency Cleanup

**Current pyproject.toml:**
```toml
dependencies = [
    "jpype1>=1.4.0",      # ❌ Make optional
    "py4j>=0.10.0",       # ❌ Make optional
    "chromadb>=0.4.0",    # ❌ Make optional (RAG)
    "ollama>=0.1.0",      # ✅ Keep
    "openai>=1.0.0",      # ✅ Keep
    "tree-sitter>=0.20.0", # ✅ Keep (core AST)
    "networkx>=3.0",      # ✅ Keep (dependency graph)
    "pydantic>=2.0.0",    # ✅ Keep (config)
    "typer>=0.9.0",       # ✅ Keep (CLI)
]
```

**Actions:**
- [ ] Move heavy dependencies to optional extras
- [ ] Create `[bridge]` extra for JPype/Py4J
- [ ] Create `[rag]` extra for ChromaDB
- [ ] Keep core dependencies minimal

---

## Implementation Priority

### Phase 1: Core Simplifications (Week 1)
1. ✅ Analyze current vs spec differences
2. 🔴 Implement subprocess+JSON bridge (replaces JPype/Py4J)
3. 🔴 Simplify type compatibility to JSON round-trip
4. 🔴 Update configuration to match spec essentials

### Phase 2: Orchestrator Updates (Week 2)
5. Update Phase 2 for incremental file compilation
6. Unify LLM clients with persistent conversation
7. Ensure retry loops maintain full history

### Phase 3: Cleanup & Testing (Week 3)
8. Simplify feature mapping to 6 core rules
9. Move optional dependencies to extras
10. Update documentation (CLAUDE.md)
11. Test end-to-end with simplified stack

---

## Success Metrics

After simplification:
- ✅ **Fewer dependencies:** Core install < 10 packages
- ✅ **Simpler config:** < 20 essential config fields
- ✅ **Universal bridge:** Works on any system with Python + Java
- ✅ **Clear flow:** Implementation matches spec sections 1:1
- ✅ **Easier debugging:** JSON serialization is human-readable

---

## Breaking Changes

### For Users
- Must explicitly enable RAG (`rag.enabled: true`)
- JPype/Py4J require `pip install codemorph[bridge]`
- Some config fields removed (use defaults)

### Migration Guide
```bash
# Before
pip install codemorph  # Installs everything

# After (minimal)
pip install codemorph  # Core only, uses subprocess bridge

# After (with RAG)
pip install codemorph[rag]

# After (with native bridges)
pip install codemorph[bridge]
```

---

## References

- **Simplified Spec:** `/Users/alamvirk/Downloads/SPEC.md`
- **Current CLAUDE.md:** `/Users/alamvirk/git_clones/code-convert/CLAUDE.md`
- **Oxidizer Paper Principles:**
  - Feature Mapping (Section 9)
  - Type-Compatibility (Section 10)
  - I/O Equivalence (Section 12)

---

## Notes

The goal is not to remove features, but to:
1. Make the default path simpler
2. Make advanced features optional
3. Align implementation with the validated spec design
4. Reduce cognitive load for contributors

The simplified spec represents the "essence" of CodeMorph—we're bringing the implementation back to that core.
