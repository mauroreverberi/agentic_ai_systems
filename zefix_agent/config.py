"""Settings of the register tool from config/agent_config.json.

The environment variable ZEFIX_MODE sets the mode, snapshot (default), live or record.
"""

import json
import os
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "agent_config.json"
SNAPSHOT_DIR = PROJECT_DIR / "snapshots" / "responses"
VALID_MODES = ("snapshot", "live", "record")


def load_config(path=CONFIG_PATH):
    """Read the settings from the JSON file and apply ZEFIX_MODE from the environment."""
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    mode = os.getenv("ZEFIX_MODE") or "snapshot"
    if mode not in VALID_MODES:
        raise ValueError(f"ZEFIX_MODE must be one of {VALID_MODES}, got {mode!r}")
    config["register"]["mode"] = mode
    return config
