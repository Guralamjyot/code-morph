#!/usr/bin/env python3.13
"""
Run Phase 4 assembly for javatuples with a stronger model (gpt-4o).
gpt-4o-mini struggles with tool-use agentic loops in the compile fixer.
"""

import logging
import os
from pathlib import Path

os.chdir("/workspace/code-agent/code-convert")

java_path = Path("/workspace/persistent/java/jdk-17.0.2")
if java_path.exists():
    os.environ["JAVA_HOME"] = str(java_path)
    os.environ["PATH"] = f"{java_path / 'bin'}:{os.environ.get('PATH', '')}"

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from codemorph.config.loader import load_config_from_yaml
from codemorph.assembler.orchestrator import Phase4Orchestrator

cfg = load_config_from_yaml(Path("javatuples_config.yaml"))

# Override model to gpt-4o for the agentic compile-fix loop
cfg.llm.model = "gpt-4o"

output_dir = Path("/workspace/code-agent/javatuples-python")

orchestrator = Phase4Orchestrator(
    config=cfg,
    state_dir=Path(".codemorph"),
    source_dir=Path(cfg.project.source.root),
    output_dir=output_dir,
    verbose=True,
    fill_mocks=True,
    max_fix_iterations=10,
)

results = orchestrator.run()

status = results.get("compile_fix", {}).get("final_status")
if status == "clean":
    print("\n=== Assembly complete! All files compile cleanly. ===")
else:
    print(f"\n=== Assembly finished with status: {status} ===")
    remaining = results.get("compile_fix", {}).get("remaining_errors", [])
    if remaining:
        print(f"Remaining errors ({len(remaining)}):")
        for e in remaining:
            print(f"  - {e}")
