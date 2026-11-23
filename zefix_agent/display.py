"""Readable output of decision logs and results for the notebook."""

import json

import pandas as pd

from .evaluation import tools_used

SHOWN_COMPANY_FIELDS = ("legal_name", "uid_display", "chid", "ehraid", "legal_form", "street",
                        "postal_code", "locality", "canton")
SHOWN_RESULT_KEYS = ("status", "error_type", "hits", "truncated", "attempts", "source", "simulated",
                     "refused", "statuses", "field", "value")


def short_result(result):
    """Return the shown fields of a tool result, without the empty ones and without False."""
    summary = {}
    for key in SHOWN_RESULT_KEYS:
        value = result.get(key)
        if value is not None and value is not False:
            summary[key] = value
    return summary


def show_trace(trace, start=0):
    """Print the decision log, so each tool call and its reason can be read in order."""
    for entry in trace[start:]:
        kind = entry["event"]
        if kind == "llm_action":
            for action in entry["actions"]:
                if action["reason"]:
                    print(f"LLM chose {action['tool']}: {action['reason']}")
                else:
                    print(f"LLM chose {action['tool']}")
        elif kind == "tool_result":
            print(f"  tool {entry['tool']} {json.dumps(entry['args'], ensure_ascii=False)}")
            summary = short_result(entry["result"] or {})
            print(f"    result {json.dumps(summary, ensure_ascii=False)}")
        elif kind == "checked":
            print(f"CHECK proposed {entry['proposed']}, verdict {entry['verdict']}, outcome "
                  f"{entry['resulting_outcome']}: {entry['message']}")
            if entry.get("reason"):
                print(f"  LLM rationale: {entry['reason']}")
        elif kind == "user_answered":
            print(f"ASKED: {entry['question']}")
            print(f"USER: {entry['answer']}")
        elif kind == "input_rejected":
            print(f"INPUT REJECTED: {entry['reason']}")
        elif kind == "stopped":
            print(f"CONTROLLED STOP: {entry['reason']}")
        elif kind == "llm_error":
            print(f"LLM FAILED: {entry['error']}")
        elif kind == "follow_up":
            print(f"FOLLOW-UP: {entry['text']}")
        elif kind == "result":
            print(f"RESULT {entry['outcome']} after {entry['tool_calls_used']} "
                  "tool calls in this turn")


def show_result(result):
    """Print the structured result in a compact form."""
    print(f"outcome: {result['outcome']}")
    print(f"request: {json.dumps(result.get('request'), ensure_ascii=False)}")
    company = result.get("company")
    if company:
        for field in SHOWN_COMPANY_FIELDS:
            print(f"  {field}: {company.get(field)}")
    for candidate in result.get("candidates") or []:
        print(f"  candidate {candidate['company_id']}: {candidate['legal_name']}, "
              f"{candidate['locality']}")
    print(f"rationale: {result.get('rationale')}")
    if result.get("answer_to_user"):
        print(f"answer: {result['answer_to_user']}")
    for note in result.get("uncertainties") or []:
        print(f"uncertainty: {note}")
    for removed in result.get("removed_statements") or []:
        print(f"removed by the status filter: {removed}")
    for removed in result.get("removed_identifiers") or []:
        print(f"removed by the identifier check: {removed}")
    print(f"retrieved at: {result.get('retrieved_at')}, register mode: "
          f"{result.get('register_mode')}")
    print(f"note: {result.get('disclaimer')}")


def show_turn(turn):
    """Print the question of the agent if it waits for the user, otherwise the result."""
    if turn["status"] == "needs_input":
        question = turn["question"]
        print(f"QUESTION: {question['question']}")
        for candidate in question.get("candidates") or []:
            print(f"  candidate {candidate['company_id']}: {candidate['legal_name']}, "
                  f"{candidate.get('locality')}")
        return
    show_result(turn["result"])


def results_table(rows):
    """Return a DataFrame with one line per evaluation case."""
    lines = []
    for row in rows:
        run = row["run"]
        case = row["case"]
        result = run["result"] or {}
        company = result.get("company") or {}
        failed = [check["detail"] for check in row["checks"] if not check["passed"]]
        lines.append({"case": case["id"], "category": case["category"],
                      "outcome": result.get("outcome"),
                      "company": company.get("company_id"), "passed": row["passed"],
                      "tool_calls": len(tools_used(run["trace"])), "questions": run["asked"],
                      "snapshot_misses": run["snapshot_misses"],
                      "simulated_faults": run["simulated_faults"],
                      "seconds": run["seconds"], "failed_checks": "; ".join(failed)})
    return pd.DataFrame(lines)
