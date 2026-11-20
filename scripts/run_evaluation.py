"""Run the evaluation cases with the real LLM and store one log per split in evaluation/results.

The register tool runs in record mode, so a query without a stored response is sent once and stored
in snapshots/responses. Each split also adds one line to evaluation/run_history.json.

Usage: python -m scripts.run_evaluation [dev] [final] [holdout]
"""

import json
import sys
from datetime import datetime, timezone

from langchain_core.callbacks import get_usage_metadata_callback

from zefix_agent.config import PROJECT_DIR, has_api_key, load_config, public_settings
from zefix_agent.evaluation import evaluate, load_cases
from zefix_agent.graph import build_llm

RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
HISTORY_PATH = PROJECT_DIR / "evaluation" / "run_history.json"
SPLITS = ["dev", "final", "holdout"]
# list price of gpt-6-luna in USD per million tokens
INPUT_PRICE = 0.10
OUTPUT_PRICE = 0.50


def now_utc():
    """Return the current time in UTC as text."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def count_tokens(usage):
    """Add up the input and output tokens of all LLM calls."""
    tokens = {"input": 0, "output": 0}
    for values in usage.usage_metadata.values():
        tokens["input"] += values["input_tokens"]
        tokens["output"] += values["output_tokens"]
    return tokens


def next_log_path(split):
    """Return the next free log file of a split, for example final_run_2.json."""
    number = 1
    while (RESULTS_DIR / f"{split}_run_{number}.json").exists():
        number += 1
    return RESULTS_DIR / f"{split}_run_{number}.json"


def run_split(split, llm, config):
    """Run all cases of one split, store the log and return the line for the run history."""
    path = next_log_path(split)
    started = now_utc()
    with get_usage_metadata_callback() as usage:
        rows = evaluate(load_cases(split), llm, config, path.stem)
    finished = now_utc()
    tokens = count_tokens(usage)
    cost = round((tokens["input"] * INPUT_PRICE + tokens["output"] * OUTPUT_PRICE) / 1_000_000, 4)

    log_rows = []
    for row in rows:
        log_rows.append({"case_id": row["case"]["id"], "passed": row["passed"],
                         "checks": row["checks"], "run": row["run"]})
    log = {"started_utc": started, "finished_utc": finished, "settings": public_settings(config),
           "mode": config["register"]["mode"], "tokens": tokens, "cost_usd_at_list_price": cost,
           "rows": log_rows}
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    failed = [row["case"]["id"] for row in rows if not row["passed"]]
    misses = sum(row["run"]["snapshot_misses"] for row in rows)
    print(f"{split}: passed {len(rows) - len(failed)} of {len(rows)}, snapshot misses {misses}, "
          f"tokens {tokens['input']:,} in and {tokens['output']:,} out, ${cost:.4f}, "
          f"saved {path.name}")
    return {"run": path.stem, "started_utc": started, "finished_utc": finished,
            "mode": config["register"]["mode"],
            "passed": len(rows) - len(failed), "cases": len(rows), "failed": ", ".join(failed),
            "snapshot_misses": misses, "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "cost_usd": cost}


def main():
    """Run the chosen splits, all three if none is given, and update the run history."""
    splits = sys.argv[1:] or SPLITS
    for split in splits:
        if split not in SPLITS:
            sys.exit(f"unknown split {split!r}, use {SPLITS}")
    if not has_api_key():
        sys.exit("OPENAI_API_KEY is not set, see .env.example")
    config = load_config()
    config["register"]["mode"] = "record"
    llm = build_llm(config)
    history = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    for split in splits:
        history.append(run_split(split, llm, config))
        HISTORY_PATH.write_text(json.dumps(history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
