"""Deterministic tests of the input checks and the name comparison.

Usage: .venv/bin/python -m unittest discover -s tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from zefix_agent.validation import InvalidInput, format_uid, name_score, normalize_uid, split_legal_form, validate_request  # noqa: E402


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

    def test_name_comparison(self):
        self.assertEqual(name_score("Mueller Holzbau", "MÜLLER HOLZBAU"), 1.0)
        self.assertEqual(name_score("Nestle S.A.", "Nestlé AG"), 1.0)
        self.assertEqual(split_legal_form("Nestlé S.A.")[1], "AG")
        self.assertEqual(split_legal_form("Zazuko GmbH")[1], "GmbH")


if __name__ == "__main__":
    unittest.main()
