"""Deterministic tests of validation and register tool.

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


if __name__ == "__main__":
    unittest.main()
