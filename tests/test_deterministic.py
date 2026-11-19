"""Deterministic tests of validation, register tool, checks, graph structure and evaluation cases.

These tests need neither a key nor the network. They read stored real register responses, and they use
a scripted test double instead of the LLM, so they say nothing about how well the real agent decides.

Usage: python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from unittest import mock

import requests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scripted_llm import ScriptedLLM
from zefix_agent.config import load_config
from zefix_agent.evaluation import load_cases
from zefix_agent.graph import ZefixAgent, build_llm, limit_context
from zefix_agent.register_tool import FaultPlan, RegisterTool, SnapshotStore, build_lucene_query, clean_register_text
from zefix_agent.schemas import AGENT_TOOLS
from zefix_agent.validation import (InvalidInput, escape_sparql_literal, format_uid, name_score, normalize_uid,
                                    says_unknown, search_words, split_legal_form, validate_request)
from zefix_agent.verification import (clean_free_text, find_contradictions, known_identifiers, remove_unknown_identifiers,
                                      status_claims, verify_decision)


def snapshot_register(faults=None):
    """Build a register tool in snapshot mode that never waits."""
    config = load_config()
    config["register"]["mode"] = "snapshot"
    return RegisterTool(config["register"], fault_plan=FaultPlan(faults), sleep=lambda seconds: None)


def agent_with(script, faults=None):
    """Build the real graph with the scripted test double."""
    config = load_config()
    config["register"]["mode"] = "snapshot"
    llm = ScriptedLLM(script)
    return ZefixAgent(llm, snapshot_register(faults), config), llm


class ValidationTests(unittest.TestCase):
    def test_uid_formats_and_check_digit(self):
        self.assertEqual(normalize_uid("CHE-242.294.601"), "CHE242294601")
        self.assertEqual(normalize_uid("che242294601"), "CHE242294601")
        self.assertEqual(format_uid("CHE242294601"), "CHE-242.294.601")
        with self.assertRaises(InvalidInput):
            normalize_uid("CHE-242.294.602")
        with self.assertRaises(InvalidInput):
            normalize_uid("not a uid")

    def test_request_rules(self):
        with self.assertRaises(InvalidInput):
            validate_request({})
        with self.assertRaises(InvalidInput):
            validate_request({"name": "x" * 201})
        with self.assertRaises(InvalidInput):
            validate_request({"name": "ab\x00cd"})
        with self.assertRaises(InvalidInput):
            validate_request({"name": "?!"})
        with self.assertRaises(InvalidInput):
            validate_request({"name": "Zazuko", "sql": "drop"})
        self.assertEqual(validate_request({"name": "  Zazuko   GmbH "})["name"], "Zazuko GmbH")

    def test_query_values_cannot_break_out(self):
        attack = 'x" } } DROP GRAPH <https://a> #\n'
        escaped = escape_sparql_literal(attack)
        self.assertNotIn('x" }', escaped)
        self.assertNotIn("\n", escaped)
        self.assertEqual(search_words('(muster AND "x") OR label:*'), ["muster", "and", "x", "or", "label"])
        self.assertEqual(build_lucene_query(["mueller", "ag"], True, False), "label:(mueller~ AND ag)")

    def test_name_comparison(self):
        self.assertEqual(name_score("Mueller Holzbau", "MÜLLER HOLZBAU"), 1.0)
        self.assertEqual(name_score("Nestle S.A.", "Nestlé AG"), 1.0)
        self.assertEqual(split_legal_form("Nestlé S.A.")[1], "AG")
        self.assertEqual(split_legal_form("Zazuko GmbH")[1], "GmbH")

    def test_similar_name_alone_cannot_become_a_match(self):
        # this real pair reaches 0.867, so the score alone would let the wrong company through
        requested, registered = "Mueller Holzbau", "Gfeller Holzbau GmbH"
        self.assertGreater(name_score(requested, registered), 0.85)
        record = {"company_id": "350030", "legal_name": registered, "names": [registered], "uid": "CHE108197220", "locality": "Worb"}
        state = {"request": {"name": requested}, "records": {"350030": record}, "searches": [],
                 "candidates": {"350030": {"company_id": "350030", "legal_name": registered, "locality": "Worb"}}}
        self.assertEqual(verify_decision({"outcome": "MATCH", "company_id": "350030"}, state, 0.85)["outcome"], "AMBIGUOUS")

    def test_unknown_answers(self):
        for text in ("I do not know", "i don't know.", "keine Ahnung", "weiss nicht", "Ich weiß es nicht", "je ne sais pas", "no idea"):
            self.assertTrue(says_unknown(text), text)
        for text in ("The one in Obfelden", "Churwalden", "I do not know the street, but it is in Bern"):
            self.assertFalse(says_unknown(text), text)

    def test_register_text_cleaning(self):
        self.assertEqual(clean_register_text("null\nKäsereistrasse 8"), "Käsereistrasse 8")
        self.assertEqual(clean_register_text("null"), None)


class RegisterToolTests(unittest.TestCase):
    def test_reads_real_snapshot(self):
        register = snapshot_register()
        hits = register.lookup_uid("CHE242294601")
        self.assertEqual(hits["candidates"][0]["company_id"], "1198554")
        self.assertTrue(hits["source"].startswith("snapshot:"))
        record = register.company_record("1198554")["record"]
        self.assertEqual(record["uid"], "CHE242294601")
        self.assertEqual(record["locality"], "Biel/Bienne")
        self.assertEqual(register.http_requests, 0)

    def test_snapshot_miss_is_an_error_and_not_an_empty_result(self):
        outcome = snapshot_register().text_search("words that were never stored")
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_type"], "snapshot_miss")

    def test_timeout_is_retried_and_then_reported(self):
        register = snapshot_register([{"op": "*", "kind": "timeout", "times": 99}])
        outcome = register.exact_name("Zazuko GmbH")
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_type"], "timeout")
        self.assertEqual(outcome["attempts"], 3)
        self.assertTrue(outcome["simulated"])

    def test_retry_after_simulated_failure_stays_on_snapshots(self):
        register = snapshot_register([{"op": "exact_name", "kind": "http_503", "times": 2}])
        outcome = register.exact_name("Keller AG")
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(outcome["attempts"], 3)
        self.assertTrue(outcome["simulated"])
        self.assertTrue(outcome["source"].startswith("snapshot:"))
        self.assertEqual(register.http_requests, 0)

    def test_retry_after_simulated_failure_without_snapshot_is_a_miss(self):
        register = snapshot_register([{"op": "exact_name", "kind": "http_503", "times": 1}])
        outcome = register.exact_name("A name that was never stored")
        self.assertEqual(outcome["error_type"], "snapshot_miss")
        self.assertEqual(register.http_requests, 0)

    def test_count_without_stored_response_is_an_error_and_not_zero(self):
        config = load_config()
        config["register"]["mode"] = "snapshot"
        with tempfile.TemporaryDirectory() as empty:
            outcome = RegisterTool(config["register"], store=SnapshotStore(empty)).count_companies()
        self.assertEqual(outcome["status"], "error")
        self.assertIsNone(outcome["count"])

    def test_malformed_body_is_reported(self):
        register = snapshot_register([{"op": "*", "kind": "malformed", "times": 99}])
        self.assertEqual(register.exact_name("Zazuko GmbH")["error_type"], "malformed_response")

    def test_client_error_is_not_retried(self):
        class Fake:
            headers = {}
            calls = 0

            def post(self, *args, **kwargs):
                Fake.calls += 1
                response = requests.Response()
                response.status_code = 400
                response._content = b"bad"
                return response

        config = load_config()
        config["register"]["mode"] = "live"
        register = RegisterTool(config["register"], session=Fake(), sleep=lambda s: None)
        self.assertEqual(register.exact_name("Zazuko GmbH")["error_type"], "http_400")
        self.assertEqual(Fake.calls, 1)


class CheckTests(unittest.TestCase):
    RECORD = {"company_id": "1", "legal_name": "Zazuko GmbH", "names": [], "uid": "CHE242294601",
              "locality": "Biel/Bienne", "street": "Winkelstrasse 20", "postal_code": "2502", "municipality": "Biel/Bienne"}

    def state(self, request, searches=None, candidates=None, records=None):
        return {"request": request, "records": {"1": self.RECORD} if records is None else records,
                "candidates": candidates or {"1": {"company_id": "1", "legal_name": "Zazuko GmbH", "locality": "Biel/Bienne"}},
                "searches": searches or []}

    def test_match_needs_a_fetched_record(self):
        verdict = verify_decision({"outcome": "MATCH", "company_id": "1"}, self.state({"name": "Zazuko GmbH"}, records={}), 0.85)
        self.assertEqual(verdict["verdict"], "rework")

    def test_contradictions_block_a_match(self):
        request = {"name": "Zazuko GmbH", "locality": "Zürich"}
        self.assertTrue(find_contradictions(request, self.RECORD))
        self.assertEqual(verify_decision({"outcome": "MATCH", "company_id": "1"}, self.state(request), 0.85)["outcome"], "AMBIGUOUS")

    def test_legal_form_conflict_blocks_a_match(self):
        verdict = verify_decision({"outcome": "MATCH", "company_id": "1"}, self.state({"name": "Zazuko AG"}), 0.85)
        self.assertEqual(verdict["outcome"], "AMBIGUOUS")

    def test_other_spelling_needs_confirmation(self):
        verdict = verify_decision({"outcome": "MATCH", "company_id": "1"}, self.state({"name": "Zazukko GmbH"}), 0.85)
        self.assertEqual(verdict["outcome"], "AMBIGUOUS")

    def test_same_name_elsewhere_blocks_a_match(self):
        candidates = {"1": {"company_id": "1", "legal_name": "Zazuko GmbH", "locality": "Biel/Bienne"},
                      "2": {"company_id": "2", "legal_name": "Zazuko GmbH", "locality": "Bern"}}
        verdict = verify_decision({"outcome": "MATCH", "company_id": "1"}, self.state({"name": "Zazuko GmbH"}, candidates=candidates), 0.85)
        self.assertEqual(verdict["outcome"], "AMBIGUOUS")

    def test_no_match_needs_a_scan(self):
        searches = [{"op": "text_search", "status": "ok", "hits": 0}]
        state = {"request": {"name": "Nothing"}, "records": {}, "candidates": {}, "searches": searches}
        self.assertEqual(verify_decision({"outcome": "NO_MATCH"}, state, 0.85)["verdict"], "rework")

    def test_failed_scan_cannot_become_no_match(self):
        searches = [{"op": "text_search", "status": "ok", "hits": 0}, {"op": "scan_legal_names", "status": "error", "error_type": "timeout"}]
        state = {"request": {"name": "Nothing"}, "records": {}, "candidates": {}, "searches": searches}
        self.assertEqual(verify_decision({"outcome": "NO_MATCH"}, state, 0.85)["outcome"], "TECHNICAL_ERROR")

    def test_cut_off_scan_cannot_support_no_match(self):
        searches = [{"op": "text_search", "status": "ok", "hits": 0, "params": {"fuzzy": True}},
                    {"op": "scan_legal_names", "status": "ok", "hits": 25, "truncated": True}]
        state = {"request": {"name": "Bäckerei Meier GmbH"}, "records": {}, "candidates": {}, "searches": searches}
        self.assertEqual(verify_decision({"outcome": "NO_MATCH"}, state, 0.85)["verdict"], "rework")

    def test_status_statements(self):
        self.assertTrue(status_claims("The company is active and solvent."))
        kept, removed = clean_free_text("The UID fits. The company is solvent. The register cannot show whether it is solvent.")
        self.assertEqual(removed, ["The company is solvent."])
        self.assertIn("cannot show", kept)
        # sentences from a real run that say what the data cannot show, they must stay
        for sentence in ("The record does not establish whether it is active or solvent.",
                         "I can\u2019t determine whether the company is currently active or solvent."):
            self.assertEqual(clean_free_text(sentence)[1], [], sentence)

    def test_unknown_identifier_is_removed_from_free_text(self):
        known = known_identifiers({"uid": "CHE105909036"}, {"1198554": {"uid": "CHE242294601", "chid": "CH03640617915"}})
        text = "The UID is CHE-242.294.601. The CH-ID is CH03640617915. The request said CHE105909036. Another UID is CHE-999.999.999."
        kept, removed = remove_unknown_identifiers(text, known)
        self.assertEqual(removed, ["Another UID is CHE-999.999.999."])
        self.assertIn("CHE-242.294.601", kept)
        self.assertIn("CHE105909036", kept)
        # a postal code with a country prefix is no identifier
        self.assertEqual(remove_unknown_identifiers("The seat is in CH-8001 Zürich.", set())[1], [])


class GraphTests(unittest.TestCase):
    def test_invalid_input_ends_without_llm_or_register(self):
        agent, llm = agent_with([])
        turn = agent.identify({"uid": "CHE-242.294.602"}, "t1")
        self.assertEqual(turn["result"]["outcome"], "INVALID_INPUT")
        self.assertEqual(llm.seen_contexts, [])

    def test_match_path(self):
        agent, _ = agent_with([
            ("exact_name", {"name": "Zazuko GmbH"}),
            ("get_company_records", {"company_ids": ["1198554"]}),
            ("submit_decision", {"outcome": "MATCH", "company_id": "1198554", "rationale": "One entry fits."}),
        ])
        result = agent.identify({"name": "Zazuko GmbH"}, "t2")["result"]
        self.assertEqual(result["outcome"], "MATCH")
        self.assertEqual(result["company"]["uid"], "CHE242294601")
        self.assertTrue(result["evidence"][0]["source"].startswith("snapshot:"))

    def test_invented_identifier_never_reaches_the_result(self):
        agent, _ = agent_with([
            ("get_company_records", {"company_ids": ["1198554"]}),
            ("submit_decision", {"outcome": "MATCH", "company_id": "1198554", "uid": "CHE999999999",
                                 "rationale": "Fits. The UID is CHE-999.999.999.",
                                 "answer_to_user": "The UID is CHE242294601.", "uncertainties": ["CH99999999999 could be meant."]}),
        ])
        result = agent.identify({"name": "Zazuko GmbH"}, "t3")["result"]
        self.assertEqual(result["company"]["uid"], "CHE242294601")
        # the free text may repeat the identifier of the fetched record, but no other one
        self.assertEqual(result["rationale"], "Fits.")
        self.assertEqual(result["answer_to_user"], "The UID is CHE242294601.")
        self.assertEqual(result["removed_identifiers"], ["The UID is CHE-999.999.999.", "CH99999999999 could be meant."])
        self.assertNotIn("CH99999999999 could be meant.", result["uncertainties"])

    def test_clarification_keeps_the_state_and_resumes(self):
        agent, _ = agent_with([
            ("exact_name", {"name": "Peter Meier"}),
            ("submit_decision", {"outcome": "AMBIGUOUS", "candidate_ids": ["586855", "641452"], "rationale": "Two entries.",
                                 "clarification_question": "In which place?"}),
            ("update_request", {"field": "locality", "value": "Churwalden"}),
            ("get_company_records", {"company_ids": ["586855"]}),
            ("submit_decision", {"outcome": "MATCH", "company_id": "586855", "rationale": "The place separates them."}),
        ])
        turn = agent.identify({"name": "Peter Meier"}, "t4")
        self.assertEqual(turn["status"], "needs_input")
        self.assertEqual(len(turn["question"]["candidates"]), 2)
        self.assertEqual(len(agent.state("t4")["candidates"]), 2)
        turn = agent.answer("The one in Churwalden please", "t4")
        self.assertEqual(turn["result"]["outcome"], "MATCH")
        self.assertEqual(turn["result"]["company"]["company_id"], "586855")
        self.assertEqual(turn["result"]["request"]["locality"], "Churwalden")

    def test_user_detail_must_come_from_the_user(self):
        agent, _ = agent_with([
            ("exact_name", {"name": "Peter Meier"}),
            ("submit_decision", {"outcome": "AMBIGUOUS", "candidate_ids": ["586855", "641452"], "rationale": "Two.", "clarification_question": "Where?"}),
            ("update_request", {"field": "locality", "value": "Churwalden"}),
            ("submit_decision", {"outcome": "AMBIGUOUS", "candidate_ids": ["586855", "641452"], "rationale": "Still two."}),
        ])
        agent.identify({"name": "Peter Meier"}, "t5")
        result = agent.answer("I do not know", "t5")["result"]
        self.assertEqual(result["outcome"], "AMBIGUOUS")
        self.assertIsNone(result["request"].get("locality"))

    def test_unknown_answer_cannot_change_the_request(self):
        agent, _ = agent_with([
            ("exact_name", {"name": "Peter Meier"}),
            ("submit_decision", {"outcome": "AMBIGUOUS", "candidate_ids": ["586855", "641452"], "rationale": "Two.", "clarification_question": "Where?"}),
            ("update_request", {"field": "name", "value": "I do not know"}),
            ("submit_decision", {"outcome": "AMBIGUOUS", "candidate_ids": ["586855", "641452"], "rationale": "Still two.", "clarification_question": "Where again?"}),
        ])
        agent.identify({"name": "Peter Meier"}, "t11")
        turn = agent.answer("I do not know", "t11")
        self.assertEqual(turn["status"], "done")
        self.assertEqual(turn["result"]["request"]["name"], "Peter Meier")

    def test_giving_up_needs_a_fuzzy_search(self):
        agent, _ = agent_with([
            ("text_search", {"words": "qxzjvkw"}),
            ("scan_legal_names", {"fragment": "qxzjvkw"}),
            ("submit_decision", {"outcome": "NO_MATCH", "rationale": "No hit."}),
            ("text_search", {"words": "qxzjvkw", "fuzzy": True}),
            ("submit_decision", {"outcome": "NO_MATCH", "rationale": "No hit after the fuzzy search."}),
        ])
        result = agent.identify({"name": "Qxzjvkw Consulting"}, "t12")["result"]
        self.assertEqual(result["outcome"], "NO_MATCH")
        self.assertEqual(result["tool_calls_used"], 3)

    def test_follow_up_uses_stored_context_without_register_call(self):
        agent, llm = agent_with([
            ("get_company_records", {"company_ids": ["1198554"]}),
            ("submit_decision", {"outcome": "MATCH", "company_id": "1198554", "rationale": "Fits."}),
            ("submit_decision", {"outcome": "MATCH", "company_id": "1198554", "rationale": "From memory of this thread.",
                                 "answer_to_user": "The company is solvent. The UID is in the record."}),
        ])
        agent.identify({"name": "Zazuko GmbH"}, "t6")
        result = agent.follow_up("Is it solvent?", "t6")["result"]
        self.assertEqual(result["tool_calls_used"], 0)
        self.assertIsNotNone(result["retrieved_at"])
        self.assertEqual(result["removed_statements"], ["The company is solvent."])
        self.assertNotIn("solvent", result["answer_to_user"])
        self.assertIn("1198554", llm.seen_contexts[-1][0].content)

    def test_tool_failure_is_a_technical_error(self):
        agent, _ = agent_with([
            ("exact_name", {"name": "Zazuko GmbH"}),
            ("submit_decision", {"outcome": "NO_MATCH", "rationale": "Nothing was found."}),
        ], faults=[{"op": "*", "kind": "timeout", "times": 99}])
        result = agent.identify({"name": "Zazuko GmbH"}, "t7")["result"]
        self.assertEqual(result["outcome"], "TECHNICAL_ERROR")
        self.assertIsNone(result["company"])

    def test_budget_stops_the_loop(self):
        agent, _ = agent_with([("text_search", {"words": "zazuko"})] * 8)
        result = agent.identify({"name": "Zazuko GmbH"}, "t8")["result"]
        self.assertEqual(result["outcome"], "TECHNICAL_ERROR")
        self.assertEqual(result["tool_calls_used"], 8)

    def test_llm_failure_is_controlled(self):
        agent, _ = agent_with([RuntimeError("provider failed")])
        result = agent.identify({"name": "Zazuko GmbH"}, "t9")["result"]
        self.assertEqual(result["outcome"], "TECHNICAL_ERROR")
        self.assertIn("llm_error", result["rationale"])

    def test_verifier_sends_no_match_back_and_scan_budget_holds(self):
        agent, _ = agent_with([
            ("text_search", {"words": "qxzjvkw"}),
            ("submit_decision", {"outcome": "NO_MATCH", "rationale": "No hit."}),
            ("scan_legal_names", {"fragment": "qxzjvkw"}),
            ("text_search", {"words": "qxzjvkw", "fuzzy": True}),
            ("submit_decision", {"outcome": "NO_MATCH", "rationale": "No hit after the scan."}),
        ])
        result = agent.identify({"name": "Qxzjvkw Consulting"}, "t10")["result"]
        self.assertEqual(result["outcome"], "NO_MATCH")
        self.assertEqual(len(agent.state("t10")["searches"]), 3)

    def test_context_limit_keeps_tool_pairs(self):
        messages = []
        for i in range(30):
            messages += [HumanMessage(content="question " * 40),
                         AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": f"c{i}", "type": "tool_call"}]),
                         ToolMessage(content="answer " * 40, tool_call_id=f"c{i}")]
        trimmed = limit_context(messages, 400)
        self.assertLess(len(trimmed), len(messages))
        self.assertIsInstance(trimmed[0], HumanMessage)

    def test_real_chat_model_accepts_the_tools_without_network(self):
        # a placeholder value, this object is never called, the test only reads the prepared request
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder-not-a-key"}):
            model = build_llm(load_config())
        bound = model.bind_tools(AGENT_TOOLS, tool_choice="required", parallel_tool_calls=False)
        # langchain_openai builds the request without sending it, so the settings can be checked offline
        payload = bound.bound._get_request_payload([HumanMessage(content="x")], **bound.kwargs)
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertFalse(bound.bound._use_responses_api(payload))
        # OpenAI allows function calling for this model in Chat Completions only with this value
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertIn("submit_decision", [t["function"]["name"] for t in payload["tools"]])


class CaseDataTests(unittest.TestCase):
    def test_expected_companies_have_stored_records(self):
        register = snapshot_register()
        cases = load_cases()
        splits = [case["split"] for case in cases]
        self.assertEqual(splits.count("dev"), 6)
        self.assertEqual(splits.count("final"), 22)
        self.assertEqual(splits.count("holdout"), 10)
        for case in cases:
            expected = case["expected"]
            ids = expected.get("candidates_include", []) + expected.get("must_not_company", [])
            if "company_id" in expected:
                ids.append(expected["company_id"])
            for company_id in ids:
                record = register.company_record(company_id).get("record")
                self.assertIsNotNone(record, f"{case['id']} {company_id}")


if __name__ == "__main__":
    unittest.main()
