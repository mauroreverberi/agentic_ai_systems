"""Evaluation of the agent on cases with expected results that were fixed before the first run.

The scoring uses the three levels of the course module on evaluating agents, the final result, the
single steps and the trajectory. For the trajectory it checks which tools a case used and how many
calls it made, not their order. I did not use Ragas, because each case here has a fixed expected
result from the register and no retrieved text that a model would have to judge.
"""

import json
import time

from .config import PROJECT_DIR
from .graph import ZefixAgent
from .register_tool import FaultPlan, RegisterTool
from .verification import STATES_THE_LIMIT, status_claims

CASES_PATH = PROJECT_DIR / "evaluation" / "cases.json"
MAX_TURNS = 4


def load_cases(split=None, path=CASES_PATH):
    """Read the evaluation cases of one split (dev, final or holdout), None means all."""
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    if split is None:
        return cases
    return [case for case in cases if case["split"] == split]


def tools_used(trace):
    """Return the names of the tools the agent called, in order."""
    return [entry["tool"] for entry in trace if entry["event"] == "tool_result"]


def unhedged_claims(text):
    """Return the status statements that the output filter should have removed."""
    claims = []
    for sentence in text.split(". "):
        if status_claims(sentence) and not STATES_THE_LIMIT.search(sentence.replace("’", "'")):
            claims.append(sentence)
    return claims


def score_case(case, run):
    """Compare one run with the expected result and return one check per expectation."""
    expected = case["expected"]
    result = run.get("result") or {}
    outcome = result.get("outcome")
    company = (result.get("company") or {}).get("company_id")
    candidates = [candidate["company_id"] for candidate in result.get("candidates") or []]
    used = tools_used(run["trace"])
    checks = []

    def check(name, passed, detail):
        """Add one named check with its result and a short detail."""
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if "outcome" in expected:
        check("outcome", outcome == expected["outcome"],
              f"expected {expected['outcome']}, got {outcome}")
    if "outcome_in" in expected:
        check("outcome_in", outcome in expected["outcome_in"],
              f"expected one of {expected['outcome_in']}, got {outcome}")
    if "must_not_outcome" in expected:
        check("must_not_outcome", outcome not in expected["must_not_outcome"], f"got {outcome}")
    if "first_outcome" in expected:
        check("asked_first", run["asked"] >= 1, f"questions asked {run['asked']}")
    if "company_id" in expected:
        check("company", company == expected["company_id"],
              f"expected {expected['company_id']}, got {company}")
    if "must_not_company" in expected:
        check("not_wrong_company", company not in expected["must_not_company"], f"got {company}")
    if "candidates_include" in expected:
        missing = set(expected["candidates_include"]) - set(candidates)
        check("candidates", not missing, f"missing {sorted(missing)}, got {candidates}")
    if "must_use" in expected:
        missing = [tool for tool in expected["must_use"] if tool not in used]
        check("must_use", not missing, f"missing {missing}, used {used}")
    if "max_tool_calls" in expected:
        check("max_tool_calls", len(used) <= expected["max_tool_calls"], f"used {len(used)}")
    if expected.get("follow_up_no_register_call"):
        calls = run.get("follow_up_tool_calls")
        check("follow_up_no_register_call", calls == 0, f"tool calls {calls}")
    if expected.get("no_status_claim"):
        parts = [result.get("rationale"), result.get("answer_to_user")]
        text = " ".join(part for part in parts if part)
        check("no_status_claim_in_final_text", not unhedged_claims(text), f"text {text!r}")
    return checks


def run_case(case, llm, config, thread_prefix="eval"):
    """Run one case with the real graph and answer its questions with the prepared answers."""
    faults = case.get("faults")
    register = RegisterTool(config["register"], fault_plan=FaultPlan(faults))
    agent = ZefixAgent(llm, register, config)
    thread_id = f"{thread_prefix}-{case['id']}"
    answers = list(case.get("user_answers") or [])
    started = time.perf_counter()
    turn = agent.identify(case["input"], thread_id)
    asked = 0
    questions = []
    while turn["status"] == "needs_input" and asked < MAX_TURNS:
        asked += 1
        questions.append(turn["question"])
        answer = "I do not know"
        if answers:
            answer = answers.pop(0)
        turn = agent.answer(answer, thread_id)
    run = {"case_id": case["id"], "result": turn.get("result"), "asked": asked,
           "questions": questions}
    if case.get("follow_up") and turn["status"] == "done":
        run["first_result"] = turn.get("result")
        followed = agent.follow_up(case["follow_up"], thread_id)
        run["result"] = followed.get("result")
        run["follow_up_tool_calls"] = (followed.get("result") or {}).get("tool_calls_used")
    run["trace"] = agent.trace(thread_id)
    run["seconds"] = round(time.perf_counter() - started, 2)
    run["simulated_faults"] = bool(faults)
    searches = agent.state(thread_id).get("searches") or []
    misses = [search for search in searches if search.get("error_type") == "snapshot_miss"]
    run["snapshot_misses"] = len(misses)
    return run


def evaluate(cases, llm, config, thread_prefix="eval"):
    """Run and score a list of cases, with one result per case."""
    rows = []
    for case in cases:
        run = run_case(case, llm, config, thread_prefix)
        checks = score_case(case, run)
        passed = all(check["passed"] for check in checks)
        rows.append({"case": case, "run": run, "checks": checks, "passed": passed})
        verdict = "FAIL"
        if passed:
            verdict = "pass"
        outcome = (run["result"] or {}).get("outcome")
        print(f"{case['id']:<4} {case['category']:<26} {verdict}  outcome={outcome}  "
              f"tools={tools_used(run['trace'])}  asked={run['asked']}")
    return rows
