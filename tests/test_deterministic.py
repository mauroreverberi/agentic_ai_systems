"""Deterministic tests of validation, register tool and checks.

These tests do not need the network. They read stored real register responses.

Usage: python -m unittest discover -s tests -v
"""

import pathlib
import sys
import tempfile
import unittest

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from zefix_agent.config import load_config  # noqa: E402
from zefix_agent.register_tool import FaultPlan, RegisterTool, SnapshotStore, build_lucene_query, clean_register_text  # noqa: E402
from zefix_agent.validation import (InvalidInput, escape_sparql_literal, format_uid, name_score, normalize_uid,  # noqa: E402
                                    search_words, split_legal_form, validate_request)
from zefix_agent.verification import (clean_free_text, find_contradictions, known_identifiers, remove_unknown_identifiers,  # noqa: E402
                                      status_claims, verify_decision)


def snapshot_register(faults=None):
    """Build a register tool in snapshot mode that never waits."""
    config = load_config()
    config["register"]["mode"] = "snapshot"
    return RegisterTool(config["register"], fault_plan=FaultPlan(faults), sleep=lambda seconds: None)


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


if __name__ == "__main__":
    unittest.main()
