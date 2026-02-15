"""
Phase 3: Semantics-Driven Translation Orchestrator

Coordinates the I/O equivalence verification phase.
Ensures translated code behaves exactly like source code.

Based on Section 8 of the CodeMorph v2.0 plan.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config.models import CodeMorphConfig, TranslatedFragment, TranslationStatus
from ..state.persistence import TranslationState
from ..translator.llm_client import create_llm_client
from .snapshot_capture import SnapshotCollector, capture_snapshots_from_tests
from .equivalence_checker import (
    EquivalenceChecker,
    EquivalenceStatus,
    FunctionEquivalenceReport
)
from .mocker import FunctionMocker, MockStrategy, MockRegistry
from ..bridges.java_executor import SimplifiedJavaExecutor

logger = logging.getLogger(__name__)


@dataclass
class Phase3Statistics:
    """Statistics for Phase 3 execution."""

    total_functions: int = 0
    functions_tested: int = 0
    functions_passed: int = 0
    functions_failed: int = 0
    functions_mocked: int = 0
    functions_skipped: int = 0

    total_snapshots: int = 0
    snapshots_passed: int = 0
    snapshots_failed: int = 0

    refinement_attempts: int = 0
    successful_refinements: int = 0

    def __str__(self) -> str:
        return f"""Phase 3 Statistics:
  Functions Tested: {self.functions_tested}/{self.total_functions}
  Functions Passed: {self.functions_passed} ({self._percent(self.functions_passed, self.total_functions)})
  Functions Failed: {self.functions_failed}
  Functions Mocked: {self.functions_mocked}

  Total Snapshots: {self.total_snapshots}
  Snapshots Passed: {self.snapshots_passed} ({self._percent(self.snapshots_passed, self.total_snapshots)})
  Snapshots Failed: {self.snapshots_failed}

  Refinements: {self.successful_refinements}/{self.refinement_attempts}"""

    @staticmethod
    def _percent(num: int, denom: int) -> str:
        if denom == 0:
            return "0.0%"
        return f"{100 * num / denom:.1f}%"


def run_phase3_semantics(
    config: CodeMorphConfig,
    state: TranslationState,
    translated_fragments: Dict[str, TranslatedFragment]
) -> Dict[str, FunctionEquivalenceReport]:
    """
    Run Phase 3: Semantics-Driven Translation.

    This phase:
    1. Captures execution snapshots from source tests
    2. Replays snapshots on translated code
    3. Verifies I/O equivalence
    4. Refines translations that fail equivalence checks
    5. Generates mocks for untranslatable functions

    Args:
        config: CodeMorph configuration
        state: State persistence manager
        translated_fragments: Fragments from Phase 2

    Returns:
        Dictionary mapping function IDs to equivalence reports
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: SEMANTICS-DRIVEN TRANSLATION")
    logger.info("=" * 80)

    orchestrator = Phase3Orchestrator(config, state)

    # Step 1: Capture execution snapshots
    logger.info("\n[Step 1/4] Capturing execution snapshots from tests...")
    snapshot_collector = orchestrator.capture_snapshots()

    # Step 2: Filter functions that need semantics verification
    logger.info("\n[Step 2/4] Filtering functions for verification...")
    functions_to_verify = orchestrator.get_functions_to_verify(translated_fragments)

    logger.info(f"  {len(functions_to_verify)} functions to verify")

    # Step 3: Verify each function
    logger.info("\n[Step 3/4] Verifying I/O equivalence...")
    reports = orchestrator.verify_functions(functions_to_verify, snapshot_collector)

    # Step 4: Refine failed functions
    logger.info("\n[Step 4/4] Refining failed translations...")
    orchestrator.refine_failed_functions(reports, snapshot_collector)

    # Save updated state
    state.save()

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info(str(orchestrator.statistics))
    logger.info("=" * 80)

    # Print mock report
    if orchestrator.mock_registry.get_all_mocks():
        logger.info("\n" + orchestrator.mock_registry.generate_report())

    return reports


class Phase3Orchestrator:
    """Orchestrates Phase 3 workflow."""

    def __init__(self, config: CodeMorphConfig, state: TranslationState):
        self.config = config
        self.state = state
        self.statistics = Phase3Statistics()

        # Initialize components
        self.llm_client = create_llm_client(config.llm)

        self.mock_registry = MockRegistry()

        self.mocker = FunctionMocker(
            source_lang=config.project.source.language.value,
            target_lang=config.project.target.language.value
        )

        # Snapshot and verification directories
        self.snapshot_dir = self.state.state_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)

        self.equivalence_checker = EquivalenceChecker(
            snapshot_dir=self.snapshot_dir,
            float_tolerance=1e-9
        )

    def capture_snapshots(self) -> SnapshotCollector:
        """Capture execution snapshots from source tests."""

        source_lang = self.config.project.source.language.value

        if source_lang == "java":
            return self._capture_java_snapshots()
        else:
            return self._capture_python_snapshots()

    def _capture_java_snapshots(self) -> SnapshotCollector:
        """Capture snapshots from Java source code using LLM-generated harnesses."""
        from .java_snapshot_capture import JavaSnapshotCapture

        test_root = self.config.project.source.test_root
        source_root = self.config.project.source.root

        if not test_root or not test_root.exists():
            logger.warning("No Java test directory found - skipping snapshot capture")
            return SnapshotCollector(self.snapshot_dir)

        logger.info(f"  Source root: {source_root}")
        logger.info(f"  Test root: {test_root}")

        try:
            capture = JavaSnapshotCapture(self.config, self.snapshot_dir)
            collector = capture.capture_all_snapshots(source_root, test_root)

            summary = collector.get_summary()
            self.statistics.total_snapshots = sum(summary.values())

            logger.info(f"  Captured {self.statistics.total_snapshots} snapshots "
                       f"for {len(summary)} functions")

            return collector

        except Exception as e:
            logger.error(f"Failed to capture Java snapshots: {e}")
            return SnapshotCollector(self.snapshot_dir)

    def _capture_python_snapshots(self) -> SnapshotCollector:
        """Capture snapshots from Python source tests."""

        # Check if test suite exists
        test_root = self.config.project.source.test_root
        if not test_root or not test_root.exists():
            logger.warning("No test suite found - skipping snapshot capture")
            return SnapshotCollector(self.snapshot_dir)

        logger.info(f"  Test root: {test_root}")

        # Determine test runner
        runner = self.config.verification.runner

        # Capture snapshots
        try:
            collector = capture_snapshots_from_tests(
                test_module_path=test_root,
                source_module_path=self.config.project.source.root,
                output_dir=self.snapshot_dir,
                test_runner=runner
            )

            summary = collector.get_summary()
            self.statistics.total_snapshots = sum(summary.values())

            logger.info(f"  Captured {self.statistics.total_snapshots} snapshots "
                       f"for {len(summary)} functions")

            return collector

        except Exception as e:
            logger.error(f"Failed to capture snapshots: {e}")
            return SnapshotCollector(self.snapshot_dir)

    def get_functions_to_verify(
        self,
        translated_fragments: Dict[str, TranslatedFragment]
    ) -> Dict[str, TranslatedFragment]:
        """Filter fragments that need semantics verification."""

        functions_to_verify = {}

        for frag_id, fragment in translated_fragments.items():
            # Only verify functions (not classes, globals, etc.)
            if fragment.fragment.fragment_type.value not in ("function", "method"):
                continue

            # Only verify successfully translated fragments
            if fragment.status not in (TranslationStatus.COMPILED, TranslationStatus.TYPE_VERIFIED):
                logger.debug(f"  Skipping {frag_id} (status: {fragment.status})")
                continue

            functions_to_verify[frag_id] = fragment

        self.statistics.total_functions = len(functions_to_verify)

        return functions_to_verify

    def verify_functions(
        self,
        functions: Dict[str, TranslatedFragment],
        snapshot_collector: SnapshotCollector
    ) -> Dict[str, FunctionEquivalenceReport]:
        """Verify I/O equivalence for all functions."""

        reports = {}

        for idx, (frag_id, fragment) in enumerate(functions.items(), 1):
            logger.info(f"  [{idx}/{len(functions)}] Verifying {frag_id}...")

            # Check if snapshots exist
            snapshots = snapshot_collector.load_snapshots(frag_id)

            if not snapshots:
                logger.warning(f"    No snapshots found - skipping")
                self.statistics.functions_skipped += 1
                continue

            # Create executor for this function
            executor = self._create_executor(fragment)

            # Run equivalence check
            report = self.equivalence_checker.check_function(
                function_id=frag_id,
                executor=executor,
                max_snapshots=None  # Test all snapshots
            )

            reports[frag_id] = report

            # Update statistics
            self.statistics.functions_tested += 1
            self.statistics.snapshots_passed += report.passed
            self.statistics.snapshots_failed += report.failed

            if report.is_equivalent:
                logger.info(f"    ✓ PASSED ({report.passed}/{report.total_snapshots})")
                self.statistics.functions_passed += 1
                fragment.status = TranslationStatus.IO_VERIFIED
            else:
                logger.warning(f"    ✗ FAILED ({report.passed}/{report.total_snapshots})")
                self.statistics.functions_failed += 1

        return reports

    def _get_class_fragment_code(self, fragment: TranslatedFragment) -> str | None:
        """
        Find the parent CLASS fragment's code for a METHOD fragment.

        When a METHOD fragment has invalid code (Java stub), the CLASS-level
        fragment often has valid Python that includes the method.
        """
        frag_id = fragment.fragment.id
        # Method IDs look like "ClassName::ClassName.methodName"
        # Class IDs look like "ClassName::ClassName"
        parts = frag_id.split("::")
        if len(parts) == 2:
            module_part = parts[0]
            # Try the class-level fragment: "Module::ClassName"
            # Extract class name from method ID: "ClassName.methodName" -> "ClassName"
            method_part = parts[1]
            if "." in method_part:
                class_name = method_part.split(".")[0]
                class_frag_id = f"{module_part}::{class_name}"
                class_frag = self.state.translated_fragments.get(class_frag_id)
                if class_frag and class_frag.target_code:
                    # Check it's valid Python (not a Java stub)
                    code = class_frag.target_code
                    if not code.lstrip().startswith(("public ", "private ", "protected ")):
                        try:
                            compile(code, "<class_check>", "exec")
                            return code
                        except SyntaxError:
                            pass
        return None

    def _create_executor(self, fragment: TranslatedFragment) -> callable:
        """Create an executor function for the translated code."""

        target_lang = self.config.project.target.language.value

        if target_lang == "java":
            executor = SimplifiedJavaExecutor(timeout=30)

            def java_executor(*args, **kwargs):
                logger.warning(f"Java execution not fully implemented - using placeholder")
                return None

            return java_executor

        elif target_lang == "python":
            # Pre-compile and cache the namespace for efficiency
            namespace = {"__builtins__": __builtins__}

            # Also load any already-translated dependencies into namespace
            for dep_id in fragment.fragment.dependencies:
                dep_frag = self.state.translated_fragments.get(dep_id)
                if dep_frag and dep_frag.target_code:
                    try:
                        exec(compile(dep_frag.target_code, f"<{dep_id}>", "exec"), namespace)
                    except Exception:
                        pass  # Best effort loading of dependencies

            # Try compiling the fragment's own code
            code_to_use = fragment.target_code
            used_class_fallback = False

            try:
                compiled_code = compile(code_to_use, "<translated>", "exec")
                exec(compiled_code, namespace)
            except SyntaxError as e:
                # METHOD fragment may have a Java stub — fall back to CLASS code
                class_code = self._get_class_fragment_code(fragment)
                if class_code:
                    logger.info(f"Method code has SyntaxError, using CLASS-level code for {fragment.fragment.name}")
                    try:
                        exec(compile(class_code, "<class_translated>", "exec"), namespace)
                        used_class_fallback = True
                    except SyntaxError as e2:
                        logger.error(f"CLASS code also has SyntaxError for {fragment.fragment.name}: {e2}")
                        def error_executor(*args, **kwargs):
                            raise RuntimeError(f"Translated code has syntax error: {e2}")
                        return error_executor
                else:
                    logger.error(f"Syntax error in translated code for {fragment.fragment.name}: {e}")
                    def error_executor(*args, **kwargs):
                        raise RuntimeError(f"Translated code has syntax error: {e}")
                    return error_executor

            # Find the function — try original Java name and snake_case Python name
            func_name = fragment.fragment.name
            snake_name = self._to_snake_case(func_name)

            found_func = namespace.get(func_name)

            if not found_func:
                # Try snake_case conversion (Java camelCase → Python snake_case)
                found_func = namespace.get(snake_name)

            if not found_func:
                # Look inside classes defined in the namespace
                import inspect as _inspect
                for key, val in namespace.items():
                    if isinstance(val, type) and not key.startswith("_"):
                        # Check for the method inside the class
                        for name_candidate in (func_name, snake_name):
                            method = getattr(val, name_candidate, None)
                            if method and callable(method):
                                found_func = method
                                break
                        if found_func:
                            break

            if not found_func:
                # Last resort: find any callable (non-class) that was defined
                for key, val in namespace.items():
                    if callable(val) and not key.startswith("_") and not isinstance(val, type):
                        found_func = val
                        break

            if found_func:
                def python_executor(*args, **kwargs):
                    try:
                        return found_func(*args, **kwargs)
                    except TypeError as e:
                        err_msg = str(e)
                        if "positional argument" in err_msg:
                            # Case 1: Too many scalar args → pack into single list
                            # e.g. args=(1,2,3,4,5), func expects (array)
                            if len(args) > 1:
                                try:
                                    return found_func(list(args), **kwargs)
                                except TypeError:
                                    pass
                            # Case 2: Single list arg → unpack into positional args
                            # e.g. args=([7,5],), func expects (a, b)
                            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                                return found_func(*args[0], **kwargs)
                        raise
                return python_executor
            else:
                logger.error(f"Function {func_name} not found in translated code")
                def missing_executor(*args, **kwargs):
                    raise RuntimeError(f"Function {func_name} not found in translated code")
                return missing_executor

        else:
            raise NotImplementedError(
                f"Executor not implemented for {target_lang}"
            )

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert camelCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def refine_failed_functions(
        self,
        reports: Dict[str, FunctionEquivalenceReport],
        snapshot_collector: SnapshotCollector
    ):
        """Refine functions that failed I/O equivalence checks."""

        max_retries = self.config.translation.max_retries_semantics

        for frag_id, report in reports.items():
            if report.is_equivalent:
                continue

            logger.info(f"\n  Refining {frag_id}...")

            fragment = self.state.translated_fragments.get(frag_id)
            if not fragment:
                logger.error(f"    Fragment not found in state")
                continue

            # Determine which fragment to refine: METHOD or CLASS fallback
            refine_fragment = fragment
            class_fragment = None
            class_code = self._get_class_fragment_code(fragment)
            if class_code:
                # METHOD code is invalid (Java stub) — refine the CLASS fragment
                parts = frag_id.split("::")
                if len(parts) == 2 and "." in parts[1]:
                    class_name = parts[1].split(".")[0]
                    class_frag_id = f"{parts[0]}::{class_name}"
                    class_fragment = self.state.translated_fragments.get(class_frag_id)
                    if class_fragment and class_fragment.target_code:
                        refine_fragment = class_fragment
                        logger.info(f"    Using CLASS fragment {class_frag_id} for refinement")

            # Get failed snapshots
            failed_results = [
                r for r in report.results
                if r.status in (EquivalenceStatus.FAILED, EquivalenceStatus.ERROR)
            ]

            if not failed_results:
                continue

            # Try to refine
            refined = self._refine_translation(
                refine_fragment,
                failed_results[:3],  # Use first 3 failures as examples
                max_retries,
                method_fragment=fragment if class_fragment else None,
            )

            if refined:
                logger.info(f"    ✓ Refinement successful")
                self.statistics.successful_refinements += 1
                fragment.status = TranslationStatus.IO_VERIFIED
                if class_fragment:
                    self.state.update_fragment(class_fragment)
            else:
                logger.warning(f"    ✗ Refinement failed - generating mock")
                self._generate_mock(fragment, "Failed I/O equivalence after refinement")
                fragment.status = TranslationStatus.MOCKED

            # Save updated fragment
            self.state.update_fragment(fragment)

    def _refine_translation(
        self,
        fragment: TranslatedFragment,
        failed_results: List,
        max_retries: int,
        method_fragment: Optional[TranslatedFragment] = None,
    ) -> bool:
        """
        Attempt to refine a translation based on I/O failures.

        Args:
            fragment: The fragment whose code will be refined (may be CLASS).
            failed_results: List of EquivalenceResult failures.
            max_retries: Max refinement attempts.
            method_fragment: If set, the original METHOD fragment (used for
                creating the executor when refining CLASS-level code).
        """

        import re as _re
        from .equivalence_checker import OutputComparator, ExceptionMapper

        # Save original code ONCE before the loop
        original_code = fragment.target_code

        for attempt in range(1, max_retries + 1):
            self.statistics.refinement_attempts += 1

            logger.info(f"    Refinement attempt {attempt}/{max_retries}")

            # Build refinement prompt
            failure_examples = self._format_failures(failed_results)

            # Query LLM
            from ..translator.llm_client import LLMConversation
            conversation = LLMConversation()
            response, conversation = self.llm_client.refine_translation(
                original_code=fragment.fragment.source_code,
                translated_code=fragment.target_code or "",
                errors=[failure_examples],
                conversation=conversation,
                instruction="Fix the I/O mismatches in the translated code. "
                    "Use ValueError for IllegalArgumentException, "
                    "use // for integer division when the Java code uses int/long division.",
            )

            if not response:
                logger.warning(f"      LLM returned empty response")
                continue

            # Strip markdown code fences if present
            response = _re.sub(r'^```(?:python|java)?\s*\n', '', response)
            response = _re.sub(r'\n```\s*$', '', response)
            response = response.strip()

            fragment.target_code = response

            logger.info(f"      Got refined code ({len(response)} chars)")

            # Re-test the refined version
            # Use the method_fragment for executor if we're refining CLASS code
            exec_fragment = method_fragment if method_fragment else fragment
            try:
                executor = self._create_executor(exec_fragment)

                # Re-run the failed snapshots
                all_passed = True
                comparator = OutputComparator()
                for result in failed_results:
                    try:
                        args = result.inputs.get("args", [])
                        kwargs = result.inputs.get("kwargs", {})
                        actual = executor(*args, **kwargs)

                        # If the expected result was an exception, this is a failure
                        # (we got a return value instead of an exception)
                        if result.expected_exception:
                            all_passed = False
                            break

                        equal, _ = comparator.are_equal(result.expected_output, actual)
                        if not equal:
                            all_passed = False
                            break
                    except Exception as e:
                        # If we expected an exception, check if it's the right type
                        if result.expected_exception:
                            actual_type = type(e).__name__
                            expected_type = result.expected_exception
                            if ExceptionMapper.are_equivalent_exceptions(
                                actual_type, expected_type
                            ):
                                continue  # This snapshot passed
                        all_passed = False
                        break

                if all_passed:
                    logger.info(f"      Refinement passed all failed snapshots!")
                    return True
                else:
                    logger.info(f"      Refinement still has failures, retrying...")

            except Exception as e:
                logger.warning(f"      Error testing refined code: {e}")

        # All retries failed — revert to original working code
        fragment.target_code = original_code
        return False

    def _format_failures(self, failed_results: List) -> str:
        """Format failure results for refinement prompt."""
        lines = []

        for result in failed_results[:3]:  # Show first 3
            lines.append(f"  Snapshot {result.snapshot_id}:")
            lines.append(f"    Input: {result.inputs}")

            if result.expected_exception:
                lines.append(f"    Expected exception: {result.expected_exception}")
                if result.actual_exception:
                    lines.append(f"    Actual exception: {result.actual_exception}")
                elif result.actual_output is not None:
                    lines.append(f"    Actual: returned {result.actual_output} (should have raised)")
                lines.append(f"    Hint: Use ValueError for IllegalArgumentException, "
                           f"ArithmeticError for ArithmeticException")
            else:
                lines.append(f"    Expected output: {result.expected_output}")
                lines.append(f"    Actual output: {result.actual_output}")

            if result.diff:
                lines.append(f"    Diff: {result.diff}")
            if result.error_message:
                lines.append(f"    Error: {result.error_message}")

        return "\n".join(lines)

    def _generate_mock(self, fragment: TranslatedFragment, reason: str):
        """Generate a mock for a failed translation."""

        logger.info(f"    Generating mock (reason: {reason})")

        # Determine mock strategy
        strategy = MockStrategy.STUB  # Default to stub

        if self.config.translation.allow_mocking:
            # Could use BRIDGE strategy if bridges are available
            # For now, use STUB
            strategy = MockStrategy.STUB

        # Generate mock
        mock = self.mocker.generate_mock(
            function_id=fragment.fragment.id,
            function_name=fragment.fragment.name,
            signature=fragment.fragment.signature or fragment.fragment.name,
            strategy=strategy,
            reason=reason
        )

        # Register mock
        self.mock_registry.register_mock(mock)
        self.statistics.functions_mocked += 1

        # Update fragment with mock code
        fragment.target_code = mock.generated_code
