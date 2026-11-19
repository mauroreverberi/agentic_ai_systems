"""The agent as an explicit LangGraph StateGraph.

It uses interrupt() for questions to the user and its own tools node instead of the prebuilt ToolNode.
"""

import json
import logging
import operator
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt

from . import schemas
from .config import load_system_prompt
from .register_tool import utc_now
from .validation import InvalidInput, clean_text, format_uid, name_score, name_tokens, normalize_uid, says_unknown, validate_request
from .verification import (clean_free_text, compare_with_request, known_identifiers, liquidation_note,
                           remove_unknown_identifiers, verify_decision)

LOG = logging.getLogger("zefix_agent.graph")
SUBMIT = "submit_decision"
MAX_FACT_CANDIDATES = 15
# fields of a fetched record and of a search that the LLM sees as known facts
RECORD_FACT_FIELDS = ("legal_name", "names", "uid", "chid", "ehraid", "legal_form", "street",
                      "postal_code", "locality", "canton", "municipality")
SEARCH_FACT_FIELDS = ("op", "params", "status", "error_type", "hits", "truncated")
EVIDENCE_FIELDS = ("legal_name", "uid", "chid", "ehraid", "legal_form", "street", "postal_code", "locality", "canton")
FINALIZE_NODES = {"MATCH": "finalize_match", "AMBIGUOUS": "finalize_ambiguous", "NO_MATCH": "finalize_no_match",
                  "TECHNICAL_ERROR": "finalize_technical_error"}


class AgentState(TypedDict, total=False):
    """Everything the agent remembers in a conversation. The checkpointer stores it per thread_id."""
    messages: Annotated[list, add_messages]
    incoming: Optional[dict]
    request: dict
    candidates: dict
    records: dict
    record_meta: dict
    searches: list
    tool_calls_used: int
    scan_calls_used: int
    clarifications_used: int
    reworks_used: int
    user_cannot_help: bool
    turn_started_at: float
    decision: Optional[dict]
    verdict: Optional[dict]
    stop_reason: Optional[str]
    result: Optional[dict]
    trace: Annotated[list, operator.add]


def event(node, what, **details):
    """Build one entry of the decision log with the time and the given values."""
    return {"at": utc_now(), "node": node, "event": what, **details}


def known_facts(state, limits):
    """Return the memory as JSON for the prompt, so it survives when old messages are trimmed."""
    candidates = state.get("candidates") or {}
    records = {}
    for company_id, record in (state.get("records") or {}).items():
        records[company_id] = {field: record.get(field) for field in RECORD_FACT_FIELDS}
    searches = []
    for search in state.get("searches") or []:
        searches.append({field: search.get(field) for field in SEARCH_FACT_FIELDS})
    return json.dumps({
        "request": state.get("request"),
        "candidates_seen": list(candidates.values())[-MAX_FACT_CANDIDATES:],
        "records_fetched": records,
        "searches_this_turn": searches,
        "budget": {
            "tool_calls_left": limits["max_tool_calls_per_turn"] - state.get("tool_calls_used", 0),
            "scans_left": limits["max_scan_calls_per_turn"] - state.get("scan_calls_used", 0),
            "clarifications_left": limits["max_clarifications"] - state.get("clarifications_used", 0),
        },
    }, ensure_ascii=False, indent=1)


def limit_context(messages, max_tokens):
    """Return the newest messages that fit the budget, starting with a human message."""
    trimmed = trim_messages(messages, max_tokens=max_tokens, strategy="last",
                            token_counter=count_tokens_approximately, start_on="human", allow_partial=False)
    if trimmed:
        return trimmed
    # one long turn can exceed the budget, then the current turn stays complete
    last_human = 0
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = i
    return messages[last_human:]


def latest_user_text(messages):
    """Return the text of the latest message of the user, or an empty text."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def build_graph(llm, register, config, checkpointer=None, prompt=None):
    """Build and compile the agent graph, with a MemorySaver if no checkpointer is given."""
    limits = config["agent"]
    strong_score = config["verifier"]["strong_name_score"]
    template = prompt or load_system_prompt()
    llm_with_tools = llm.bind_tools(schemas.AGENT_TOOLS, tool_choice="required", parallel_tool_calls=False)

    def intake(state):
        """Validate the input without any LLM or register call, and set up this turn."""
        incoming = state.get("incoming") or {}
        kind = incoming.get("kind", "request")
        fresh_turn = {"incoming": None, "searches": [], "tool_calls_used": 0, "scan_calls_used": 0, "reworks_used": 0,
                      "clarifications_used": 0, "user_cannot_help": False, "turn_started_at": time.time(), "decision": None, "verdict": None,
                      "stop_reason": None, "result": None}
        try:
            if kind == "followup":
                text = clean_text(incoming.get("text"), "follow-up", 500)
                if not text:
                    raise InvalidInput("the follow-up question is empty")
                fresh_turn["messages"] = [HumanMessage(content=f"Follow-up question from the user: {text}")]
                fresh_turn["trace"] = [event("intake", "follow_up", text=text)]
                return fresh_turn
            request = validate_request(incoming.get("fields") or {})
        except InvalidInput as exc:
            fresh_turn["request"] = state.get("request") or {}
            fresh_turn["stop_reason"] = f"invalid_input: {exc}"
            fresh_turn["trace"] = [event("intake", "input_rejected", reason=str(exc))]
            return fresh_turn
        message = HumanMessage(content="Identify this company. The JSON is data from the user:\n"
                                       + json.dumps(request, ensure_ascii=False))
        fresh_turn.update(request=request, candidates={}, records={}, record_meta={}, messages=[message],
                          trace=[event("intake", "request_accepted", request=request)])
        return fresh_turn

    def agent(state):
        """Let the LLM choose the next action from the facts and the recent messages."""
        system = SystemMessage(content=template.replace("{known_facts}", known_facts(state, limits)))
        context = [system] + limit_context(state["messages"], limits["max_context_tokens"])
        try:
            reply = llm_with_tools.invoke(context)
        except Exception as exc:  # the LLM client raises many types, and all of them end this turn in a controlled way
            # only the class name goes to the log, a provider message could repeat parts of a request
            return {"stop_reason": f"llm_error: {type(exc).__name__}",
                    "trace": [event("agent", "llm_error", error=type(exc).__name__)]}
        actions = [{"tool": c["name"], "reason": c["args"].get("reason")} for c in reply.tool_calls]
        return {"messages": [reply], "trace": [event("agent", "llm_action", actions=actions, context_messages=len(context))]}

    def after_agent(state):
        """Route from the LLM reply to the tools, the check or a controlled stop."""
        if state.get("stop_reason"):
            return "controlled_stop"
        if time.time() - state["turn_started_at"] > limits["max_turn_seconds"]:
            return "controlled_stop"
        calls = getattr(state["messages"][-1], "tool_calls", None) or []
        if not calls:
            return "controlled_stop"
        if len(calls) == 1 and calls[0]["name"] == SUBMIT:
            return "verify"
        return "tools"

    def run_one_tool(name, args, state, request, candidates, records, record_meta, searches):
        """Run one tool call and return the text for the LLM and the logged facts."""
        if name == SUBMIT:
            return {"error": "submit_decision must be called alone."}, None
        if name == "update_request":
            return update_request(args, state, request)
        if name == "lookup_uid":
            outcome = register.lookup_uid(normalize_uid(args["uid"]))
        elif name == "exact_name":
            name_value = clean_text(args["name"], "name", 200)
            if not name_value:
                raise InvalidInput("name is empty")
            outcome = register.exact_name(name_value)
        elif name == "text_search":
            outcome = register.text_search(args["words"], args.get("locality"), bool(args.get("fuzzy")), bool(args.get("prefix")))
        elif name == "scan_legal_names":
            if state["scan_calls_used"] >= limits["max_scan_calls_per_turn"]:
                return {"error": "The scan budget of this turn is used up, decide with what you have."}, {"op": name, "refused": "scan_budget"}
            state["scan_calls_used"] += 1
            outcome = register.scan_legal_names(args["fragment"], args.get("locality"))
        elif name == "get_company_records":
            return fetch_records(args, request, records, record_meta, searches, candidates)
        else:
            return {"error": f"Unknown tool {name}."}, {"op": name, "refused": "unknown_tool"}
        entry = search_entry(outcome)
        searches.append(entry)
        for found in outcome.get("candidates", []):
            candidates[found["company_id"]] = found
        return tool_reply(outcome), entry

    def update_request(args, state, request):
        """Change the request only if the value really stands in the latest user message."""
        user_text = latest_user_text(state["messages"])
        value = clean_text(args.get("value"), "value")
        field = args.get("field")
        if state.get("user_cannot_help") or says_unknown(value):
            # "I do not know" stands in the user message, but it is no detail about the company
            return ({"error": "The user gave no new detail, so the request stays as it is. Decide with what you have."},
                    {"op": "update_request", "refused": "user_gave_no_detail", "field": field})
        if not value or not set(name_tokens(value)) <= set(name_tokens(user_text)):
            return ({"error": f"The value {value!r} does not appear in the latest user message, so it was not stored."},
                    {"op": "update_request", "refused": "value_not_in_user_message", "field": field})
        if field == "uid":
            value = normalize_uid(value)
        request[field] = value
        return {"status": "ok", "request": request}, {"op": "update_request", "field": field, "value": value}

    def fetch_records(args, request, records, record_meta, searches, candidates):
        """Fetch up to the allowed number of records and compare each with the request."""
        ids = [str(i) for i in args.get("company_ids", [])]
        if not ids or len(ids) > register.config["max_records_per_call"]:
            raise InvalidInput(f"give 1 to {register.config['max_records_per_call']} company ids")
        replies, statuses = [], []
        for company_id in ids:
            outcome = register.company_record(company_id)
            entry = search_entry(outcome)
            searches.append(entry)
            statuses.append(entry)
            record = outcome.get("record")
            if outcome["status"] != "ok":
                reply = {"company_id": company_id}
                reply.update(tool_reply(outcome))
                replies.append(reply)
            elif not record:
                replies.append({"company_id": company_id, "status": "ok", "record": None})
            else:
                records[company_id] = record
                record_meta[company_id] = {k: outcome.get(k) for k in ("retrieved_at", "source", "simulated")}
                candidates.setdefault(company_id, {"company_id": company_id, "legal_name": record.get("legal_name") or "",
                                                   "locality": record.get("locality")})
                shown = {k: v for k, v in record.items() if k != "purpose"}
                replies.append({"company_id": company_id, "status": "ok", "record": shown,
                                "comparison_with_request": compare_with_request(request, record)})
        return {"results": replies}, {"op": "get_company_records", "ids": ids, "statuses": [s["status"] for s in statuses]}

    def tools(state):
        """Run the tool calls of the last LLM reply, count them and collect the evidence."""
        working = {"scan_calls_used": state.get("scan_calls_used", 0), "messages": state["messages"], "user_cannot_help": state.get("user_cannot_help", False)}
        request = dict(state.get("request") or {})
        candidates = dict(state.get("candidates") or {})
        records = dict(state.get("records") or {})
        record_meta = dict(state.get("record_meta") or {})
        searches = list(state.get("searches") or [])
        used = state.get("tool_calls_used", 0)
        replies, trace = [], []
        for call in state["messages"][-1].tool_calls:
            used += 1
            args = {k: v for k, v in call["args"].items() if k != "reason"}
            try:
                body, facts = run_one_tool(call["name"], call["args"], working, request, candidates, records, record_meta, searches)
            except InvalidInput as exc:
                body, facts = {"error": f"Input rejected: {exc}"}, {"op": call["name"], "refused": "invalid_argument"}
            except KeyError as exc:
                body, facts = {"error": f"Missing argument {exc}"}, {"op": call["name"], "refused": "missing_argument"}
            replies.append(ToolMessage(content=json.dumps(body, ensure_ascii=False), tool_call_id=call["id"], name=call["name"]))
            trace.append(event("tools", "tool_result", tool=call["name"], args=args, result=facts))
        return {"messages": replies, "request": request, "candidates": candidates, "records": records,
                "record_meta": record_meta, "searches": searches, "tool_calls_used": used,
                "scan_calls_used": working["scan_calls_used"], "trace": trace}

    def after_tools(state):
        """Stop in a controlled way when the budget or the time of this turn is used up."""
        if state["tool_calls_used"] >= limits["max_tool_calls_per_turn"]:
            return "controlled_stop"
        if time.time() - state["turn_started_at"] > limits["max_turn_seconds"]:
            return "controlled_stop"
        return "agent"

    def controlled_stop(state):
        """End the turn after an LLM failure or a used up limit. Nothing is concluded about the company."""
        over_time = time.time() - state["turn_started_at"] > limits["max_turn_seconds"]
        over_budget = state.get("tool_calls_used", 0) >= limits["max_tool_calls_per_turn"]
        reason = state.get("stop_reason")
        if not reason and over_time:
            reason = "time_limit"
        elif not reason and over_budget:
            reason = "tool_budget"
        elif not reason:
            reason = "no_action"
        last = state["messages"][-1] if state.get("messages") else None
        closing = []
        # the provider expects an answer to every open tool call in the stored conversation
        for call in getattr(last, "tool_calls", None) or []:
            closing.append(ToolMessage(content=json.dumps({"error": f"Stopped by {reason}."}), tool_call_id=call["id"], name=call["name"]))
        return {"messages": closing, "stop_reason": reason, "trace": [event("controlled_stop", "stopped", reason=reason)]}

    def verify(state):
        """Check the proposed decision against the evidence in the state."""
        call = state["messages"][-1].tool_calls[0]
        decision = call["args"]
        verdict = verify_decision(decision, state, strong_score)
        reworks = state.get("reworks_used", 0)
        if verdict["verdict"] == "rework":
            reworks += 1
            if reworks > limits["max_verifier_reworks"]:
                # with seen candidates the user can still choose, without them the turn failed
                outcome = "TECHNICAL_ERROR"
                if state.get("candidates"):
                    outcome = "AMBIGUOUS"
                verdict = {"verdict": "override", "outcome": outcome,
                           "message": f"The rework limit was reached. Last check said {verdict['message']}", "notes": []}
        reply = ToolMessage(content=json.dumps({"check": verdict["verdict"], "outcome": verdict["outcome"], "message": verdict["message"],
                                                "notes": verdict["notes"]}, ensure_ascii=False),
                            tool_call_id=call["id"], name=SUBMIT)
        return {"messages": [reply], "decision": decision, "verdict": verdict, "reworks_used": reworks,
                "trace": [event("verify", "checked", proposed=decision.get("outcome"), verdict=verdict["verdict"],
                                resulting_outcome=verdict["outcome"], message=verdict["message"], reason=decision.get("rationale"))]}

    def after_verify(state):
        """Route on the checked outcome. The four outcomes take separate paths."""
        verdict = state["verdict"]
        if verdict["verdict"] == "rework":
            return "agent"
        outcome = verdict["outcome"]
        if outcome == "AMBIGUOUS":
            can_ask = state.get("clarifications_used", 0) < limits["max_clarifications"] and not state.get("user_cannot_help")
            if can_ask and (state["decision"].get("clarification_question") or verdict["verdict"] == "override"):
                return "ask_user"
            return "finalize_ambiguous"
        return FINALIZE_NODES[outcome]

    def ask_user(state):
        """Pause the graph and wait for the user. The state stays in the checkpointer until the answer arrives."""
        # LangGraph runs this node again from the start on resume, so nothing before interrupt() may change anything
        decision = state.get("decision") or {}
        candidates = state.get("candidates") or {}
        question = decision.get("clarification_question") or "Can you give a further detail, such as the place or the UID?"
        ids = decision.get("candidate_ids") or list(candidates.keys())
        shown = [candidates[company_id] for company_id in ids if candidates.get(company_id)]
        answer = interrupt({"question": question, "candidates": shown, "notes": state["verdict"]["notes"]})
        cannot_help = says_unknown(answer)
        note = ""
        if cannot_help:
            note = " The user cannot add a detail. Do not call update_request, decide with what you have."
        return {"messages": [HumanMessage(content=f"Answer of the user to your question: {answer}.{note}")],
                "clarifications_used": state.get("clarifications_used", 0) + 1, "turn_started_at": time.time(), "user_cannot_help": cannot_help,
                "searches": state.get("searches") or [], "trace": [event("ask_user", "user_answered", question=question, answer=answer)]}

    def finalize(state, fixed_outcome=None):
        """Build the structured result. Identifiers come only from register records, never from the LLM."""
        verdict = state.get("verdict") or {}
        decision = state.get("decision") or {}
        stop = state.get("stop_reason")
        request = state.get("request") or {}
        records = state.get("records") or {}
        meta = state.get("record_meta") or {}
        outcome = fixed_outcome or verdict.get("outcome") or "TECHNICAL_ERROR"
        texts = checked_texts(decision, request, records)
        rationale = texts["rationale"]
        uncertainties = texts["uncertainties"] + list(verdict.get("notes") or [])
        if stop and stop.startswith("invalid_input"):
            outcome = "INVALID_INPUT"
            rationale = f"The input was rejected before any search. {stop.split(': ', 1)[1]}"
        elif stop:
            outcome = "TECHNICAL_ERROR"
            rationale = f"The turn stopped in a controlled way ({stop}). Nothing is concluded about the company."
        company, evidence = None, []
        chosen = decision.get("company_id")
        if outcome == "MATCH" and chosen and chosen in records:
            record = records[chosen]
            company = company_from_record(chosen, record)
            evidence = evidence_from_record(chosen, record, meta.get(chosen, {}))
            note = liquidation_note(record)
            if note:
                uncertainties.append(note)
        shown = []
        if outcome == "AMBIGUOUS":
            shown = shown_candidates(decision, request, state.get("candidates") or {})
        verifier = {}
        if verdict:
            verifier = {"verdict": verdict.get("verdict"), "message": verdict.get("message")}
        result = schemas.IdentificationResult(
            outcome=outcome, request=request, company=company, candidates=shown, evidence=evidence,
            retrieved_at=latest_retrieval(state.get("searches") or [], meta), register_mode=register.mode,
            rationale=rationale, answer_to_user=texts["answer"] or None, uncertainties=uncertainties,
            verifier=verifier, removed_statements=texts["removed"], removed_identifiers=texts["unknown"],
            tool_calls_used=state.get("tool_calls_used", 0))
        summary = {"outcome": outcome, "company_id": None, "legal_name": None}
        if company:
            summary["company_id"] = company.company_id
            summary["legal_name"] = company.legal_name
        message = AIMessage(content="Result of this turn: " + json.dumps(summary, ensure_ascii=False))
        return {"result": result.model_dump(), "messages": [message],
                "trace": [event("finalize", "result", outcome=outcome, tool_calls_used=state.get("tool_calls_used", 0),
                                removed_statements=texts["removed"], removed_identifiers=texts["unknown"])]}

    def after_intake(state):
        """Rejected input goes straight to the end, without LLM or register."""
        if state.get("stop_reason"):
            return "finalize_invalid_input"
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("intake", intake)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools)
    workflow.add_node("verify", verify)
    workflow.add_node("ask_user", ask_user)
    workflow.add_node("controlled_stop", controlled_stop)
    workflow.add_node("finalize_invalid_input", lambda s: finalize(s, "INVALID_INPUT"))
    workflow.add_node("finalize_match", lambda s: finalize(s, "MATCH"))
    workflow.add_node("finalize_ambiguous", lambda s: finalize(s, "AMBIGUOUS"))
    workflow.add_node("finalize_no_match", lambda s: finalize(s, "NO_MATCH"))
    workflow.add_node("finalize_technical_error", lambda s: finalize(s, "TECHNICAL_ERROR"))

    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges("intake", after_intake, ["agent", "finalize_invalid_input"])
    workflow.add_conditional_edges("agent", after_agent, ["tools", "verify", "controlled_stop"])
    workflow.add_conditional_edges("tools", after_tools, ["agent", "controlled_stop"])
    workflow.add_conditional_edges("verify", after_verify, ["agent", "ask_user", "finalize_match", "finalize_ambiguous",
                                                            "finalize_no_match", "finalize_technical_error"])
    workflow.add_edge("ask_user", "agent")
    workflow.add_edge("controlled_stop", "finalize_technical_error")
    for name in ("finalize_invalid_input", "finalize_match", "finalize_ambiguous", "finalize_no_match", "finalize_technical_error"):
        workflow.add_edge(name, END)
    return workflow.compile(checkpointer=checkpointer or MemorySaver())


def checked_texts(decision, request, records):
    """Return the free texts of the LLM without status statements and without unknown identifiers."""
    rationale, removed = clean_free_text(decision.get("rationale") or "")
    answer, removed_answer = clean_free_text(decision.get("answer_to_user") or "")
    # the LLM may repeat an identifier in its own words, but only one that a fetched record or the request holds
    known = known_identifiers(request, records)
    rationale, unknown = remove_unknown_identifiers(rationale, known)
    answer, unknown_answer = remove_unknown_identifiers(answer, known)
    unknown = unknown + unknown_answer
    uncertainties = []
    for note in decision.get("uncertainties") or []:
        kept, dropped = remove_unknown_identifiers(note, known)
        unknown += dropped
        if kept:
            uncertainties.append(kept)
    return {"rationale": rationale, "answer": answer, "uncertainties": uncertainties,
            "removed": removed + removed_answer, "unknown": unknown}


def company_from_record(company_id, record):
    """Return the company of the result, every value comes from the register record."""
    other_names = [name for name in record.get("names") or [] if name != record.get("legal_name")]
    return schemas.Company(
        company_id=company_id, company_uri=record["company_uri"], legal_name=record.get("legal_name"),
        other_names=other_names, uid=record.get("uid"), uid_display=format_uid(record.get("uid")),
        chid=record.get("chid"), ehraid=record.get("ehraid"), legal_form=record.get("legal_form"),
        legal_form_code=record.get("legal_form_code"), street=record.get("street"),
        postal_code=record.get("postal_code"), locality=record.get("locality"), canton=record.get("canton"),
        municipality=record.get("municipality"))


def evidence_from_record(company_id, record, meta):
    """Return one piece of evidence per field, with the retrieval time and the source of the record."""
    evidence = []
    for field in EVIDENCE_FIELDS:
        evidence.append(schemas.Evidence(field=field, value=record.get(field), company_id=company_id,
                                         retrieved_at=meta.get("retrieved_at"), source=meta.get("source"),
                                         simulated=bool(meta.get("simulated"))))
    return evidence


def shown_candidates(decision, request, candidates):
    """Return the candidates of an AMBIGUOUS result, the ones the LLM named or the 10 best names."""
    ids = decision.get("candidate_ids") or []
    if not ids:
        ordered = sorted(candidates.values(), key=lambda c: name_score(request.get("name") or "", c["legal_name"]), reverse=True)
        ids = [candidate["company_id"] for candidate in ordered[:10]]
    return [schemas.Candidate.model_validate(candidates[company_id]) for company_id in ids if company_id in candidates]


def latest_retrieval(searches, meta):
    """Return the latest retrieval time of the searches and fetched records, or None."""
    # a follow-up answered from memory makes no new search, so the stored records also count
    times = [search["retrieved_at"] for search in searches if search.get("retrieved_at")]
    times += [record_meta["retrieved_at"] for record_meta in meta.values() if record_meta.get("retrieved_at")]
    if not times:
        return None
    return max(times)


def search_entry(outcome):
    """Return the facts of one register operation for the state, without the response body."""
    entry = {
        "op": outcome.get("op"), "params": outcome.get("params"), "status": outcome["status"],
        "error_type": outcome.get("error_type"), "hits": None, "truncated": outcome.get("truncated", False),
        "attempts": outcome.get("attempts"), "source": outcome.get("source"),
        "simulated": outcome.get("simulated", False), "retrieved_at": outcome.get("retrieved_at"),
    }
    if "candidates" in outcome:
        entry["hits"] = len(outcome["candidates"])
    return entry


def tool_reply(outcome):
    """Return the reply for the LLM, a failed request must not look like an empty result."""
    if outcome["status"] != "ok":
        return {"status": "tool_error", "error_type": outcome.get("error_type"), "message": outcome.get("message"),
                "note": "The request failed. This says nothing about the company."}
    reply = {"status": "ok", "hits": len(outcome.get("candidates", [])), "candidates": outcome.get("candidates", [])}
    if outcome.get("truncated"):
        reply["note"] = "The row limit was reached, more companies may exist. Use more specific words."
    return reply


class ZefixAgent:
    """Small wrapper around the graph, so that every caller uses the same calls."""

    def __init__(self, llm, register, config, prompt=None):
        """Create the agent with its own memory."""
        self.config = config
        self.register = register
        self.checkpointer = MemorySaver()
        self.graph = build_graph(llm, register, config, self.checkpointer, prompt)

    def _thread(self, thread_id):
        """Return the settings of one conversation, recursion_limit is only a second safety net."""
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.config["agent"]["graph_recursion_limit"]}

    def _run(self, payload, thread_id):
        """Run the graph until it ends (done) or waits for the user (needs_input)."""
        settings = self._thread(thread_id)
        try:
            self.graph.invoke(payload, settings)
        except GraphRecursionError:
            LOG.error("recursion limit reached in thread %s", thread_id)
            return {"status": "done", "result": {"outcome": "TECHNICAL_ERROR", "rationale": "The graph reached its recursion limit."}}
        snapshot = self.graph.get_state(settings)
        for task in snapshot.tasks:
            for paused in task.interrupts:
                return {"status": "needs_input", "question": paused.value}
        return {"status": "done", "result": snapshot.values.get("result")}

    def identify(self, fields, thread_id):
        """Start a new identification in a conversation."""
        return self._run({"incoming": {"kind": "request", "fields": fields}}, thread_id)

    def answer(self, text, thread_id):
        """Continue a paused conversation with the answer of the user."""
        return self._run(Command(resume=text), thread_id)

    def follow_up(self, text, thread_id):
        """Ask a further question in a finished conversation, the stored facts stay available."""
        return self._run({"incoming": {"kind": "followup", "text": text}}, thread_id)

    def trace(self, thread_id):
        """Return the decision log of a conversation."""
        return self.graph.get_state(self._thread(thread_id)).values.get("trace", [])

    def state(self, thread_id):
        """Return the stored state of a conversation."""
        return self.graph.get_state(self._thread(thread_id)).values


def build_llm(config):
    """Create the chat model, the key comes only from the environment variable OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI  # imported here so that the deterministic tests do not need a key

    settings = config["llm"]
    # without this setting the client moves gpt-6 models to the Responses API on its own
    return ChatOpenAI(model=settings["model"], use_responses_api=settings["use_responses_api"],
                      reasoning_effort=settings["reasoning_effort"], temperature=settings["temperature"],
                      seed=settings["seed"], timeout=settings["request_timeout_seconds"],
                      max_retries=settings["max_retries"], base_url=settings["base_url"])
