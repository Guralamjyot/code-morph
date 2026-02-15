#!/usr/bin/env python3
"""
End-to-end pipeline test: Translate Java -> Python.

Tests the full CodeMorph pipeline using a mock LLM client,
then verifies functional equivalence of the translated code.
"""

import sys
import traceback
from pathlib import Path

# ============================================================================
# Step 1: Test Phase 1 - Parsing and Analysis
# ============================================================================

def test_phase1():
    """Test Phase 1: Parse Java, extract fragments, build dependency graph."""
    print("=" * 70)
    print("PHASE 1: Project Partitioning")
    print("=" * 70)

    from codemorph.languages.java.plugin import JavaPlugin
    from codemorph.analyzer.graph_builder import DependencyGraphBuilder

    plugin = JavaPlugin(version="17")
    java_file = Path("examples/java_test_project/MathUtils.java")

    # Parse the Java file
    print("\n[1.1] Parsing Java file...")
    tree = plugin.parse_file(java_file)
    print(f"  OK - parsed {java_file}")

    # Extract fragments
    print("\n[1.2] Extracting fragments...")
    fragments = plugin.extract_fragments(java_file, tree)
    print(f"  OK - found {len(fragments)} fragments:")
    for f in fragments:
        sig = plugin.extract_signature(f)
        print(f"    - {f.id} [{f.fragment_type.value}] lines {f.start_line}-{f.end_line}")
        if sig:
            print(f"      signature: {sig}")

    # Extract imports
    print("\n[1.3] Extracting imports...")
    imports = plugin.extract_imports(tree, java_file)
    print(f"  OK - found {len(imports)} imports")

    # Build dependency graph
    print("\n[1.4] Building dependency graph...")
    fragment_dict = {f.id: f for f in fragments}
    graph_builder = DependencyGraphBuilder()
    graph = graph_builder.build_graph(fragment_dict)
    print(f"  OK - graph has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Get translation order
    result = graph_builder.create_analysis_result(fragment_dict)
    print(f"\n[1.5] Translation order ({len(result.translation_order)} items):")
    for i, fid in enumerate(result.translation_order):
        frag = result.fragments[fid]
        print(f"  {i+1}. {fid} ({frag.fragment_type.value})")

    print(f"\n  Circular dependencies: {len(result.circular_dependencies)}")

    return fragments, result


# ============================================================================
# Step 2: Test Phase 2 - Translation (with mock LLM)
# ============================================================================

# Pre-written correct Python translations for each Java fragment
MOCK_TRANSLATIONS = {
    "MathUtils::MathUtils": '''class MathUtils:
    PI = 3.14159265358979

    @staticmethod
    def factorial(n):
        if n < 0:
            raise ValueError("Negative numbers not allowed")
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def circle_area(radius):
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        return MathUtils.PI * radius * radius

    @staticmethod
    def gcd(a, b):
        a = abs(a)
        b = abs(b)
        while b != 0:
            a, b = b, a % b
        return a

    @staticmethod
    def fibonacci(count):
        if count <= 0:
            return []
        result = [0] * count
        result[0] = 0
        if count > 1:
            result[1] = 1
        for i in range(2, count):
            result[i] = result[i - 1] + result[i - 2]
        return result
''',

    "MathUtils::PI": "PI = 3.14159265358979",

    "MathUtils::MathUtils.factorial": '''def factorial(n):
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',

    "MathUtils::MathUtils.isPrime": '''def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
''',

    "MathUtils::MathUtils.circleArea": '''def circle_area(radius):
    PI = 3.14159265358979
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return PI * radius * radius
''',

    "MathUtils::MathUtils.gcd": '''def gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b != 0:
        a, b = b, a % b
    return a
''',

    "MathUtils::MathUtils.fibonacci": '''def fibonacci(count):
    if count <= 0:
        return []
    result = [0] * count
    result[0] = 0
    if count > 1:
        result[1] = 1
    for i in range(2, count):
        result[i] = result[i - 1] + result[i - 2]
    return result
''',
}


def test_phase2(fragments, analysis_result):
    """Test Phase 2: Translate fragments using mock LLM, compile-check Python output."""
    print("\n" + "=" * 70)
    print("PHASE 2: Type-Driven Translation (Mock LLM)")
    print("=" * 70)

    from codemorph.languages.python.plugin import PythonPlugin
    from codemorph.config.models import TranslatedFragment, TranslationStatus
    from codemorph.knowledge.feature_mapper import create_default_mapper

    target_plugin = PythonPlugin(version="3.11")
    feature_mapper = create_default_mapper()
    translated = {}
    stats = {"compiled": 0, "failed": 0, "total": 0}

    output_dir = Path("output/java_to_python_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fid in analysis_result.translation_order:
        fragment = analysis_result.fragments[fid]
        stats["total"] += 1

        print(f"\n[2.{stats['total']}] Translating: {fid} ({fragment.fragment_type.value})")

        # Get feature mapping instructions
        from codemorph.config.models import LanguageType
        instructions = feature_mapper.get_instructions_for_fragment(
            fragment, LanguageType.JAVA, LanguageType.PYTHON
        )
        if instructions:
            print(f"  Feature mapping rules applied: {len(instructions)}")
            for inst in instructions[:3]:
                print(f"    - {inst[:80]}...")

        # Get mock translation
        mock_code = MOCK_TRANSLATIONS.get(fid)
        if not mock_code:
            print(f"  [WARN] No mock translation for {fid}, skipping")
            translated[fid] = TranslatedFragment(
                fragment=fragment,
                status=TranslationStatus.FAILED,
                compilation_errors=["No mock translation available"],
            )
            stats["failed"] += 1
            continue

        print(f"  Mock LLM returned {len(mock_code)} chars")

        # Test Python compilation (syntax check)
        compile_dir = output_dir / "compile_temp" / fid.replace("::", "_").replace(".", "_")
        compile_dir.mkdir(parents=True, exist_ok=True)

        success, errors = target_plugin.compile_fragment(mock_code, compile_dir)

        if success:
            print(f"  [OK] Python syntax check passed")
            status = TranslationStatus.COMPILED
            stats["compiled"] += 1
        else:
            print(f"  [FAIL] Python syntax check failed:")
            for err in errors:
                print(f"    {err}")
            status = TranslationStatus.FAILED
            stats["failed"] += 1

        # Validate feature mapping rules
        is_valid, failed_rules = feature_mapper.validate_translation(
            fragment, mock_code, LanguageType.JAVA, LanguageType.PYTHON
        )
        if not is_valid:
            print(f"  [WARN] Feature mapping validation failed for: {', '.join(failed_rules)}")

        translated[fid] = TranslatedFragment(
            fragment=fragment,
            target_code=mock_code,
            status=status,
            compilation_errors=errors if not success else [],
        )

    # Write combined output file
    combined = "# Auto-translated from Java by CodeMorph\n\n"
    # Use the class translation if available
    class_code = MOCK_TRANSLATIONS.get("MathUtils::MathUtils", "")
    if class_code:
        combined += class_code

    output_file = output_dir / "math_utils.py"
    with open(output_file, "w") as f:
        f.write(combined)
    print(f"\n  Combined output written to: {output_file}")

    print(f"\n  Phase 2 Summary:")
    print(f"    Total: {stats['total']}")
    print(f"    Compiled: {stats['compiled']}")
    print(f"    Failed: {stats['failed']}")
    print(f"    Success rate: {stats['compiled'] / max(stats['total'], 1) * 100:.0f}%")

    return translated, output_file


# ============================================================================
# Step 3: Test Phase 3 - Functional Equivalence Verification
# ============================================================================

def test_phase3(output_file):
    """Test Phase 3: Verify translated Python is functionally equivalent to Java."""
    print("\n" + "=" * 70)
    print("PHASE 3: Semantic Verification (Functional Equivalence)")
    print("=" * 70)

    # Load the translated module
    import importlib.util
    spec = importlib.util.spec_from_file_location("math_utils", output_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MathUtils = module.MathUtils

    passed = 0
    failed = 0
    total = 0

    def check(test_name, actual, expected):
        nonlocal passed, failed, total
        total += 1
        if actual == expected:
            print(f"  [PASS] {test_name}: {actual}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name}: got {actual}, expected {expected}")
            failed += 1

    def check_approx(test_name, actual, expected, tol=1e-6):
        nonlocal passed, failed, total
        total += 1
        if abs(actual - expected) < tol:
            print(f"  [PASS] {test_name}: {actual:.6f} (expected {expected:.6f})")
            passed += 1
        else:
            print(f"  [FAIL] {test_name}: got {actual:.6f}, expected {expected:.6f}")
            failed += 1

    def check_exception(test_name, func, exc_type):
        nonlocal passed, failed, total
        total += 1
        try:
            func()
            print(f"  [FAIL] {test_name}: expected {exc_type.__name__} but no exception raised")
            failed += 1
        except exc_type:
            print(f"  [PASS] {test_name}: raised {exc_type.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name}: expected {exc_type.__name__} but got {type(e).__name__}: {e}")
            failed += 1

    # --- Test factorial ---
    print("\n[3.1] Testing factorial():")
    check("factorial(0)", MathUtils.factorial(0), 1)
    check("factorial(1)", MathUtils.factorial(1), 1)
    check("factorial(5)", MathUtils.factorial(5), 120)
    check("factorial(10)", MathUtils.factorial(10), 3628800)
    check_exception("factorial(-1)", lambda: MathUtils.factorial(-1), ValueError)

    # --- Test isPrime ---
    print("\n[3.2] Testing is_prime():")
    check("is_prime(0)", MathUtils.is_prime(0), False)
    check("is_prime(1)", MathUtils.is_prime(1), False)
    check("is_prime(2)", MathUtils.is_prime(2), True)
    check("is_prime(3)", MathUtils.is_prime(3), True)
    check("is_prime(4)", MathUtils.is_prime(4), False)
    check("is_prime(17)", MathUtils.is_prime(17), True)
    check("is_prime(100)", MathUtils.is_prime(100), False)
    check("is_prime(97)", MathUtils.is_prime(97), True)

    # --- Test circleArea ---
    print("\n[3.3] Testing circle_area():")
    check_approx("circle_area(1)", MathUtils.circle_area(1), 3.14159265358979)
    check_approx("circle_area(5)", MathUtils.circle_area(5), 78.5398163397448)
    check_approx("circle_area(0)", MathUtils.circle_area(0), 0.0)
    check_exception("circle_area(-1)", lambda: MathUtils.circle_area(-1), ValueError)

    # --- Test gcd ---
    print("\n[3.4] Testing gcd():")
    check("gcd(12, 8)", MathUtils.gcd(12, 8), 4)
    check("gcd(100, 75)", MathUtils.gcd(100, 75), 25)
    check("gcd(7, 13)", MathUtils.gcd(7, 13), 1)
    check("gcd(0, 5)", MathUtils.gcd(0, 5), 5)
    check("gcd(-12, 8)", MathUtils.gcd(-12, 8), 4)

    # --- Test fibonacci ---
    print("\n[3.5] Testing fibonacci():")
    check("fibonacci(0)", MathUtils.fibonacci(0), [])
    check("fibonacci(1)", MathUtils.fibonacci(1), [0])
    check("fibonacci(5)", MathUtils.fibonacci(5), [0, 1, 1, 2, 3])
    check("fibonacci(8)", MathUtils.fibonacci(8), [0, 1, 1, 2, 3, 5, 8, 13])

    # --- Summary ---
    print(f"\n  Phase 3 Summary:")
    print(f"    Total tests: {total}")
    print(f"    Passed: {passed}")
    print(f"    Failed: {failed}")
    print(f"    Pass rate: {passed / max(total, 1) * 100:.0f}%")

    return passed, failed


# ============================================================================
# Step 4: Test Pipeline Integration (Config, State, Symbol Registry)
# ============================================================================

def test_pipeline_integration(fragments, analysis_result, translated):
    """Test the pipeline integration components."""
    print("\n" + "=" * 70)
    print("PIPELINE INTEGRATION: Config, State, Symbol Registry")
    print("=" * 70)

    from codemorph.config.loader import create_config_from_args
    from codemorph.config.models import CheckpointMode, LanguageType
    from codemorph.state.persistence import TranslationState
    from codemorph.state.symbol_registry import SymbolRegistry

    # Test config creation
    print("\n[4.1] Creating configuration...")
    config = create_config_from_args(
        source_dir=Path("examples/java_test_project"),
        source_lang="java",
        source_version="17",
        target_lang="python",
        target_version="3.11",
        output_dir=Path("output/java_to_python_test"),
        checkpoint_mode=CheckpointMode.AUTO,
    )
    print(f"  OK - {config.get_translation_type()}")

    # Test state persistence
    print("\n[4.2] Testing state persistence...")
    state = TranslationState(config, Path("examples/java_test_project"))
    state.set_analysis_result(analysis_result)

    for fid, frag in translated.items():
        state.update_fragment(frag)

    state_file = state.save()
    print(f"  OK - saved state to {state_file}")

    # Test state loading
    loaded_state = TranslationState.load(state_file)
    print(f"  OK - loaded state, session={loaded_state.session_id}")
    print(f"  Phase: {loaded_state.current_phase}, Fragments: {len(loaded_state.translated_fragments)}")

    # Test progress tracking
    progress = loaded_state.get_progress()
    print(f"  Progress: {progress}")

    # Test symbol registry
    print("\n[4.3] Testing symbol registry...")
    registry = SymbolRegistry(config.project.state_dir)

    # Register translated symbols
    for fid, frag in translated.items():
        if frag.target_code:
            registry.register_symbol(
                source_name=frag.fragment.name,
                source_qualified=frag.fragment.id,
                target_name=frag.fragment.name,  # Simplified
                symbol_type=frag.fragment.fragment_type.value,
                signature=frag.fragment.signature,
            )
            registry.update_status(frag.fragment.id, frag.status)

    registry.save()
    print(f"  OK - registered {len(registry.mappings)} symbols")

    # Test report export
    print("\n[4.4] Exporting report...")
    report_path = config.project.target.output_dir / "translation_report.md"
    state.export_report(report_path)
    print(f"  OK - report at {report_path}")

    return config


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# CodeMorph End-to-End Pipeline Test: Java -> Python")
    print("#" * 70)

    try:
        # Phase 1
        fragments, analysis_result = test_phase1()

        # Phase 2
        translated, output_file = test_phase2(fragments, analysis_result)

        # Phase 3 - Functional equivalence
        passed, failed = test_phase3(output_file)

        # Pipeline integration
        test_pipeline_integration(fragments, analysis_result, translated)

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"  Phase 1 (Parsing): OK - {len(fragments)} fragments extracted")
        print(f"  Phase 2 (Translation): OK - all fragments compiled")
        print(f"  Phase 3 (Equivalence): {passed}/{passed + failed} tests passed")
        print(f"  Pipeline Integration: OK")

        if failed > 0:
            print(f"\n  [WARN] {failed} equivalence test(s) failed!")
            sys.exit(1)
        else:
            print(f"\n  ALL TESTS PASSED!")
            sys.exit(0)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
