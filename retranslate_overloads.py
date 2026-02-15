#!/usr/bin/env python3.13
"""Re-translate only overloaded fragment groups from existing tictactoe state."""

from pathlib import Path

from codemorph.config.loader import load_config_from_yaml
from codemorph.state.persistence import TranslationState
from codemorph.translator.orchestrator import Phase2Orchestrator

STATE_DIR = Path(".codemorph-tictactoe")

config = load_config_from_yaml(Path("tictactoe_config.yaml"))
# Override state_dir to point at the tictactoe-specific state directory
config.project.state_dir = STATE_DIR
state = TranslationState.load_latest(STATE_DIR)
# Ensure the loaded state also writes back to the same directory
state.state_dir = STATE_DIR

orchestrator = Phase2Orchestrator(config, state)
orchestrator.retranslate_overloads()
