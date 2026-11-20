"""Settings of the agent from config/agent_config.json.

OPENAI_API_KEY is needed for every agent run and comes from the environment or the file .env, which
is never committed. OPENAI_BASE_URL sets the LLM endpoint, OPENAI_MODEL the model and ZEFIX_MODE the
register mode.
"""

import json
import os
import pathlib

from dotenv import load_dotenv

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "agent_config.json"
PROMPT_PATH = PROJECT_DIR / "prompts" / "system_prompt.md"
SNAPSHOT_DIR = PROJECT_DIR / "snapshots" / "responses"
VALID_MODES = ("snapshot", "live", "record")


def load_config(path=CONFIG_PATH):
    """Read the settings from the JSON file and apply the environment variables."""
    # the .env file of this project only fills environment variables,
    # the key is never read from code
    load_dotenv(PROJECT_DIR / ".env")
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    config["llm"]["model"] = os.getenv("OPENAI_MODEL") or config["llm"]["model"]
    config["llm"]["base_url"] = os.getenv("OPENAI_BASE_URL") or None
    mode = os.getenv("ZEFIX_MODE") or "snapshot"
    if mode not in VALID_MODES:
        raise ValueError(f"ZEFIX_MODE must be one of {VALID_MODES}, got {mode!r}")
    config["register"]["mode"] = mode
    return config


def has_api_key():
    """Return True if OPENAI_API_KEY is set and not empty, without showing the value."""
    load_dotenv(PROJECT_DIR / ".env")
    return bool(os.getenv("OPENAI_API_KEY"))


def public_settings(config):
    """Return the settings that are safe to print, the key only appears as set or not set."""
    settings = {
        "model": config["llm"]["model"],
        "llm_api": "Chat Completions",
        "reasoning_effort": config["llm"]["reasoning_effort"],
        "temperature": config["llm"]["temperature"],
        "seed": config["llm"]["seed"],
        "llm_endpoint": "OpenAI default",
        "api_key": "not set",
        "register_mode": config["register"]["mode"],
        "register_endpoint": config["register"]["endpoint"],
        "limits": config["agent"],
    }
    if config["llm"]["use_responses_api"]:
        settings["llm_api"] = "Responses"
    if config["llm"]["base_url"]:
        settings["llm_endpoint"] = config["llm"]["base_url"]
    if has_api_key():
        settings["api_key"] = "set"
    return settings


def load_system_prompt(path=PROMPT_PATH):
    """Read the system prompt template."""
    return path.read_text(encoding="utf-8")
