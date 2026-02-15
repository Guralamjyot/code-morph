"""
Java Snapshot Capture - Generates I/O snapshots from Java code.

Uses the LLM to generate a snapshot harness program that calls each Java
function with test inputs and outputs the results as JSON lines.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..config.models import CodeMorphConfig
from ..translator.llm_client import create_llm_client
from .snapshot_capture import ExecutionSnapshot, SnapshotCollector

logger = logging.getLogger(__name__)


HARNESS_PROMPT = """Given the following Java source file and its JUnit test file, generate a standalone Java program called SnapshotHarness that:

1. Calls each tested public static method with the same inputs used in the test cases
2. For each call, prints one JSON line to stdout in this exact format:
   {{"function_id": "ClassName::methodName", "args": [arg1, arg2, ...], "output": result}}
   If the method throws an exception, print:
   {{"function_id": "ClassName::methodName", "args": [arg1, arg2, ...], "exception_type": "ExceptionName", "exception": "message"}}

IMPORTANT:
- The harness class MUST be called SnapshotHarness
- Only include calls for public static methods
- Use try-catch for each call so one failure doesn't stop other captures
- Output must be valid JSON on each line (JSONL format)
- Do NOT use any external libraries (only java.util, java.lang, etc.)
- Make sure to escape strings properly in JSON output
- For array results, format as JSON arrays: [1, 2, 3]
- For numeric results, use the number directly (no quotes)
- For String results, use quotes and escape special characters
- For boolean results, use true/false (no quotes)
- For List results, format as JSON arrays
- Include multiple test inputs per method (at least 3-5 if the test file has them)
- CRITICAL: Put the main method inside the class

SOURCE FILE ({source_name}):
```java
{source_code}
```

TEST FILE ({test_name}):
```java
{test_code}
```

Return ONLY the Java code for SnapshotHarness.java, nothing else."""


class JavaSnapshotCapture:
    """Captures I/O snapshots from Java code using LLM-generated harnesses."""

    def __init__(self, config: CodeMorphConfig, snapshot_dir: Path):
        self.config = config
        self.snapshot_dir = snapshot_dir
        self.llm_client = create_llm_client(config.llm)
        self.java_home = self._find_java_home()
        self.collector = SnapshotCollector(snapshot_dir)

    def _find_java_home(self) -> Optional[Path]:
        """Find Java home directory."""
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            return Path(java_home)
        for path in [Path("/workspace/persistent/java/jdk-17.0.2")]:
            if path.exists():
                return path
        return None

    def _get_javac(self) -> str:
        if self.java_home:
            return str(self.java_home / "bin" / "javac")
        return "javac"

    def _get_java(self) -> str:
        if self.java_home:
            return str(self.java_home / "bin" / "java")
        return "java"

    def compile_sources(self, source_dir: Path, output_dir: Path) -> bool:
        """Compile all Java source files."""
        source_files = list(source_dir.glob("*.java"))
        if not source_files:
            logger.warning(f"No Java source files found in {source_dir}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._get_javac(),
            "-d", str(output_dir),
        ] + [str(f) for f in source_files]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Java compilation failed:\n{result.stderr}")
            return False

        logger.info(f"Compiled {len(source_files)} Java source files")
        return True

    def capture_all_snapshots(self, source_dir: Path, test_dir: Path) -> SnapshotCollector:
        """Capture snapshots for all source files that have tests."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            classes_dir = tmpdir / "classes"

            # Compile sources
            if not self.compile_sources(source_dir, classes_dir):
                logger.error("Failed to compile Java sources")
                return self.collector

            # Process each source file
            source_files = sorted(source_dir.glob("*.java"))

            for source_file in source_files:
                # Find corresponding test file
                test_file = self._find_test_file(source_file, test_dir)

                if not test_file:
                    logger.info(f"No test file found for {source_file.name}, skipping")
                    continue

                logger.info(f"Capturing snapshots for {source_file.name}")

                try:
                    self._capture_snapshots_for_file(
                        source_file, test_file, classes_dir, tmpdir
                    )
                except Exception as e:
                    logger.error(f"Error capturing snapshots for {source_file.name}: {e}")

        return self.collector

    def _find_test_file(self, source_file: Path, test_dir: Path) -> Optional[Path]:
        """Find the test file corresponding to a source file."""
        name = source_file.stem

        candidates = [
            test_dir / f"{name}Test.java",
            test_dir / f"Test{name}.java",
            test_dir / f"{name}Tests.java",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fallback: look for any test file that references this class
        for test_file in test_dir.glob("*.java"):
            if test_file.stem.startswith("Test") or test_file.stem.endswith("Test"):
                content = test_file.read_text()
                if name in content:
                    return test_file

        return None

    def _capture_snapshots_for_file(
        self,
        source_file: Path,
        test_file: Path,
        classes_dir: Path,
        work_dir: Path,
    ):
        """Capture snapshots for a single source file using LLM-generated harness."""

        source_code = source_file.read_text()
        test_code = test_file.read_text()

        # Generate harness using LLM
        prompt = HARNESS_PROMPT.format(
            source_code=source_code,
            test_code=test_code,
            source_name=source_file.name,
            test_name=test_file.name,
        )

        logger.info(f"  Generating snapshot harness via LLM...")
        harness_code = self.llm_client.generate(prompt)

        if not harness_code:
            logger.error(f"  LLM returned empty harness for {source_file.name}")
            return

        # Write harness
        harness_file = work_dir / "SnapshotHarness.java"
        harness_file.write_text(harness_code)

        # Compile harness (with source classes on classpath)
        harness_classes_dir = work_dir / "harness_classes"
        harness_classes_dir.mkdir(exist_ok=True)

        cmd = [
            self._get_javac(),
            "-cp", str(classes_dir),
            "-d", str(harness_classes_dir),
            str(harness_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.warning(f"  Harness compilation failed, attempting LLM fix...")
            logger.debug(f"  Error: {result.stderr[:500]}")

            # Try to fix with LLM refinement
            refined = self._refine_harness(harness_code, result.stderr, source_code, test_code)
            if refined:
                harness_file.write_text(refined)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    logger.error(f"  Refined harness also failed to compile: {result.stderr[:200]}")
                    return
                logger.info(f"  Refined harness compiled successfully")
            else:
                return

        # Run harness
        cmd = [
            self._get_java(),
            "-cp", f"{classes_dir}:{harness_classes_dir}",
            "SnapshotHarness",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0 and not result.stdout.strip():
            logger.error(f"  Harness execution failed: {result.stderr[:200]}")
            return

        # Parse output (JSONL) — even if exit code is non-zero, parse whatever output we got
        snapshot_count = 0
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                data = json.loads(line)

                # Map function_id to match fragment IDs from Phase 1
                # Phase 1 uses format: "FileStem::ClassName.methodName" (no .java extension)
                raw_func_id = data["function_id"]
                class_name = raw_func_id.split("::")[0] if "::" in raw_func_id else source_file.stem
                method_name = raw_func_id.split("::")[-1]

                # Use the fragment ID format: "FileStem::ClassName.methodName"
                fragment_func_id = f"{source_file.stem}::{class_name}.{method_name}"

                snapshot = ExecutionSnapshot(
                    function_name=method_name,
                    function_id=fragment_func_id,
                    timestamp="",
                    args=data.get("args", []),
                    kwargs=data.get("kwargs", {}),
                    output=data.get("output"),
                    exception_type=data.get("exception_type"),
                    exception=data.get("exception"),
                )

                self.collector.record_snapshot(snapshot)
                snapshot_count += 1

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"  Failed to parse snapshot line: {line[:100]}... ({e})")

        logger.info(f"  Captured {snapshot_count} snapshots for {source_file.stem}")

    def _refine_harness(
        self, harness_code: str, error: str, source_code: str, test_code: str
    ) -> Optional[str]:
        """Use LLM to fix compilation errors in the harness."""

        refine_prompt = f"""The following Java snapshot harness failed to compile.

HARNESS CODE:
```java
{harness_code}
```

COMPILATION ERROR:
```
{error}
```

ORIGINAL SOURCE CODE (this is what the harness should call):
```java
{source_code}
```

Fix the compilation errors and return the corrected Java code.
The class MUST be called SnapshotHarness.
Each output line must be valid JSON in this format:
{{"function_id": "ClassName::methodName", "args": [...], "output": result}}

Return ONLY the Java code, nothing else."""

        try:
            return self.llm_client.generate(refine_prompt)
        except Exception as e:
            logger.error(f"  LLM refinement failed: {e}")
            return None
