"""Store the planned register responses in snapshots/responses.

It uses the register tool in record mode, so a query with a stored response is not sent again.

Usage: python scripts/record_snapshots.py
"""

import json
import pathlib
import sys

# the package is one folder above this script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from zefix_agent.config import load_config  # noqa: E402
from zefix_agent.register_tool import RegisterTool  # noqa: E402

# legal names for the exact lookup, some with a spelling that is not in the register
EXACT_NAMES = [
    "Zazuko GmbH", "Peter Meier", "Nestlé S.A.", "Nestle S.A.", "Keller AG", "Garage Müller",
    "Müller Holzbau", "Müller Holzbau GmbH", "Mueller Holzbau GmbH", "Schreinerei Müller",
    "Bäckerei Meier GmbH", "Bäckerei Meier AG", "Schweizerische Bundesbahnen",
    "Schweizerische Bundesbahnen SBB", "Swiss Federal Railways", "Migros Bank AG",
    "Migross Bank AG", "Schmid GmbH", "Alpenglanz Zeitreisen AG", "Qxzjvkw Consulting", "Holding",
    'Bachmann "zur Metzg" AG in Liquidation', "Bachmann zur Metzg AG",
    "Bachmann zur Metzg AG in Liquidation", "Bäckerei Maier AG", "Migros Bank",
]
# words for the text index, each with the options of text_search
TEXT_SEARCHES = [
    ("zazuko", {}), ("zazuko gmbh", {}), ("peter meier", {}),
    ("peter meier", {"locality": "churwalden"}), ("nestle", {}), ("nestle s a", {}),
    ("nestlé", {}), ("nestle", {"fuzzy": True}), ("nestlé s a", {}),
    ("nestlé", {"locality": "vevey"}), ("keller ag", {}), ("garage müller", {}),
    ("garage mueller", {}), ("müller holzbau", {}), ("müller holzbau gmbh", {}),
    ("mueller holzbau", {}), ("mueller holzbau gmbh", {}),
    ("mueller holzbau", {"fuzzy": True}), ("müller holzbau", {"locality": "gränichen"}),
    ("mueller holzbau", {"locality": "gränichen"}),
    ("schreinerei müller", {}), ("bäckerei meier", {}), ("bäckerei meier gmbh", {}),
    ("schweizerische bundesbahnen", {}), ("swiss federal railways", {}), ("migros bank", {}),
    ("migross bank", {}), ("migross bank", {"fuzzy": True}), ("schmid gmbh", {}), ("schmid", {}),
    ("alpenglanz zeitreisen", {}), ("alpenglanz", {}), ("alpenglanz zeitreisen", {"fuzzy": True}),
    ("qxzjvkw", {}), ("qxzjvkw consulting", {}), ("qxzjvkw", {"fuzzy": True}), ("holding", {}),
    ("bachmann metzg", {}), ("bachmann zur metzg", {}), ("bäckerei maier", {}),
    ("bäckerei maier", {"fuzzy": True}), ("maier bäckerei", {}),
    ("keller", {"locality": "basel"}), ("keller ag", {"locality": "basel"}),
    ("schreinerei müller", {"locality": "bern"}),
    ("migros bank", {"fuzzy": True}),
]
# name parts for the scan over all legal names, the second value is an optional place
SCANS = [
    ("schweizerische bundesbahnen", None), ("qxzjvkw", None), ("alpenglanz zeitreisen", None),
    ("alpenglanz", None), ("mueller holzbau", None), ("müller holzbau", None),
    ("garage müller", None), ("peter meier", None), ("bäckerei maier", None),
    ("migross bank", None), ("migross", None), ("migros bank", None),
]
# the second UID has a correct check digit, but no company has it
UIDS = ["CHE242294601", "CHE000000000", "CHE105909036"]
# company ids for the full records
RECORDS = [
    "1198554", "586855", "641452", "126288", "126286", "97692", "1151594", "1541422", "1665191",
    "774070", "570202", "1594098", "462042", "528525", "119271", "12164", "350030", "661574",
    "390344", "1337570", "1672987", "1752157", "1775150",
]
# fields of a record that are printed
RECORD_FIELDS = ("legal_name", "uid", "chid", "ehraid", "legal_form", "street", "postal_code",
                 "locality", "canton", "municipality", "names")


def run_searches(register):
    """Run the planned searches and return their outcomes."""
    outcomes = [register.exact_name(name) for name in EXACT_NAMES]
    outcomes += [register.text_search(words, **options) for words, options in TEXT_SEARCHES]
    outcomes += [register.scan_legal_names(fragment, locality) for fragment, locality in SCANS]
    outcomes += [register.lookup_uid(uid) for uid in UIDS]
    return outcomes


def print_search(outcome):
    """Print one search with its status, the number of hits and the first six hits."""
    hits = [f"{hit['company_id']}:{hit['legal_name']}@{hit['locality']}"
            for hit in outcome["candidates"][:6]]
    params = json.dumps(outcome["params"], ensure_ascii=False)
    print(outcome["op"], params, outcome["status"], len(outcome["candidates"]), hits)


def print_record(company_id, outcome):
    """Print the main fields of one company record."""
    record = outcome.get("record") or {}
    print(company_id, outcome["status"], {field: record.get(field) for field in RECORD_FIELDS})


def main():
    """Send the planned operations once and report how many hits each one has."""
    config = load_config()
    config["register"]["mode"] = "record"
    register = RegisterTool(config["register"])
    for outcome in run_searches(register):
        print_search(outcome)
    for company_id in RECORDS:
        print_record(company_id, register.company_record(company_id))
    print(f"sent {register.http_requests} HTTP requests")


if __name__ == "__main__":
    main()
