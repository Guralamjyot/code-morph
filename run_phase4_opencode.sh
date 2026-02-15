#!/bin/bash
# Phase 4 Assembly with OpenCode agent for compile-fixing
#
# Usage:
#   bash run_phase4_opencode.sh [--skip-assembly]
#
# Without --skip-assembly: runs steps 1-7 (assemble), then launches opencode
# With --skip-assembly: skips to opencode directly (for re-runs)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="/workspace/code-agent/javatuples-python"
OPENCODE="/workspace/persistent/home/.opencode/bin/opencode"

# Source environment
source /workspace/persistent/claude-init.sh 2>/dev/null || true

# Export OpenAI key from .env
export OPENAI_API_KEY=$(grep OPENAI_API_KEY "$SCRIPT_DIR/.env" | cut -d= -f2-)

# Java
export JAVA_HOME="/workspace/persistent/java/jdk-17.0.2"
export PATH="$JAVA_HOME/bin:$PATH"

if [ "$1" != "--skip-assembly" ]; then
    echo "=== Phase 4: Steps 1-7 (Assembly) ==="
    cd "$SCRIPT_DIR"

    # Clean previous output
    rm -rf "$OUTPUT_DIR"

    python3.13 -c "
import logging, os
from pathlib import Path

os.chdir('$SCRIPT_DIR')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from codemorph.config.loader import load_config_from_yaml
from codemorph.assembler.orchestrator import Phase4Orchestrator

cfg = load_config_from_yaml(Path('javatuples_config.yaml'))
cfg.llm.model = 'gpt-4o'

orchestrator = Phase4Orchestrator(
    config=cfg,
    state_dir=Path('.codemorph'),
    source_dir=Path(cfg.project.source.root),
    output_dir=Path('$OUTPUT_DIR'),
    verbose=True,
    fill_mocks=True,
    max_fix_iterations=0,  # Skip compile-fix — OpenCode will do it
)

results = orchestrator.run()
print(f\"Assembly done: {results.get('index', {}).get('classes', 0)} classes\")
"
    echo "=== Assembly complete, output in $OUTPUT_DIR ==="
else
    echo "=== Skipping assembly, using existing $OUTPUT_DIR ==="
fi

echo ""
echo "=== Launching OpenCode for Phase 4 fixing ==="
echo "  Working dir: $OUTPUT_DIR"
echo "  MCP server: codemorph (translation state tools)"
echo "  Model: gpt-4o"
echo ""

cd "$OUTPUT_DIR"
exec "$OPENCODE"
