"""Deterministic checks of the decision the LLM proposes.

The LLM may propose an outcome, but this code decides whether the register answers in the state
support it. A check accepts the outcome, sends the LLM back for a missing step (rework) or replaces
the outcome with the one the evidence requires (override).
"""

import re

from .validation import (legal_forms_conflict, locality_matches, name_score, name_tokens,
                         split_legal_form)

OUTCOMES = ("MATCH", "AMBIGUOUS", "NO_MATCH", "TECHNICAL_ERROR")
# a different legal form only counts as a contradiction if the names are otherwise this close
CLOSE_NAME_SCORE = 0.85
# legal form codes of the register (eCH-0097) for the forms in a requested name, only codes that I
# saw in the stored label responses, other codes like branches are never a contradiction
LEGAL_FORM_CODES = {"AG": "0106", "GmbH": "0107", "Genossenschaft": "0108", "Stiftung": "0110"}
SOLE_PROPRIETORSHIP_CODE = "0101"
COMPARED_FORM_CODES = set(LEGAL_FORM_CODES.values()) | {SOLE_PROPRIETORSHIP_CODE}

# statements that the register data cannot support, in English and German,
# a * stands for the rest of a word, so bonit* also finds Bonität
UNSUPPORTED_PHRASES = [
    "solvent", "insolvent", "creditworthy", "creditworth*", "credit worth*",
    "financially sound", "financially healthy", "financially stable",
    "operational", "operationally", "operational active", "operationally active",
    "still active", "still operating", "still in business",
    "is active", "is operating", "is inactive", "is dissolved", "is bankrupt",
    "currently active", "currently operating", "in good standing", "legally capable",
    "reliable company", "trustworthy", "does not exist", "no longer exist", "no longer exists",
    "never existed", "bonit*", "zahlungsf*", "kreditw*", "handlungsf*",
]
# a sentence with one of these phrases says that the data cannot show a status,
# which is what it should say
LIMIT_PHRASES = [
    "cannot", "can't", "can not", "unable to",
    "does not show", "does not prove", "does not say", "does not state", "does not establish",
    "do not show", "do not prove", "do not say", "do not state", "do not establish",
    "no field", "no information", "no data", "not shown", "not stated", "not available",
    "kann nicht", "keine angabe", "keine aussage", "keine information",
]
STATUS_WORD_IN_NAME = re.compile(
    r"\b(in\s+liquidation|in\s+liquidazione|en\s+liquidation|in\s+liq\.?)\b", re.IGNORECASE)
# a UID like CHE-242.294.601 or a CH-ID like CH03640617915, with or without separators
IDENTIFIER = re.compile(r"\bCHE[-\s]?\d{3}[.\s]?\d{3}[.\s]?\d{3}\b"
                        r"|\bCH[-\s]?\d{3}[-.\s]?\d[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d\b",
                        re.IGNORECASE)
# a new sentence starts after . ! or ? and a space
SENTENCE_END = r"(?<=[.!?])\s+"


def phrase_regex(phrases):
    """Build one regex from plain phrases. A space is any space, * is the rest of a word."""
    parts = []
    # the longest phrase first, so that "operationally active" wins over "operational"
    for phrase in sorted(phrases, key=len, reverse=True):
        parts.append(re.escape(phrase).replace(r"\ ", r"\s").replace(r"\*", r"\w*"))
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


UNSUPPORTED_STATUS = phrase_regex(UNSUPPORTED_PHRASES)
STATES_THE_LIMIT = phrase_regex(LIMIT_PHRASES)


def registered_names(record):
    """Return the legal name and all translated names of a record."""
    names = [record.get("legal_name")] + list(record.get("names") or [])
    return [name for name in names if name]


def locality_fits(locality, record):
    """Return True if the requested place fits the place or the municipality of a record."""
    in_locality = locality_matches(locality, record.get("locality"))
    in_municipality = locality_matches(locality, record.get("municipality"))
    return in_locality or in_municipality


def find_contradictions(request, record):
    """Compare the request with one register record and return one sentence per contradiction."""
    found = []
    uid = request.get("uid")
    if uid and record.get("uid") and uid != record["uid"]:
        found.append(f"The requested UID {uid} differs from the register UID {record['uid']}.")
    locality = request.get("locality")
    if locality and not locality_fits(locality, record):
        found.append(f"The requested place {locality} differs from the register place "
                     f"{record.get('locality')}.")
    address = request.get("address")
    if address and record.get("street"):
        wanted = set(name_tokens(address))
        parts = [record.get("street"), record.get("postal_code"), record.get("locality")]
        have = set(name_tokens(" ".join(part for part in parts if part)))
        if not wanted <= have:
            found.append(f"The requested address {address} differs from the register address "
                         f"{record.get('street')}, {record.get('postal_code')} "
                         f"{record.get('locality')}.")
    name = request.get("name")
    if name and record.get("legal_name"):
        close_name = best_name_score(name, record) >= CLOSE_NAME_SCORE
        if close_name and legal_forms_conflict(name, record["legal_name"]):
            found.append(f"The legal form in the requested name {name} differs from the register "
                         f"name {record['legal_name']}.")
        elif close_name and legal_form_code_conflict(name, record):
            found.append(f"The legal form in the requested name {name} differs from the register "
                         f"legal form {record.get('legal_form') or record['legal_form_code']}.")
    return found


def legal_form_code_conflict(requested, record):
    """Return True if the requested name states a legal form that the code of the record denies."""
    _, form = split_legal_form(requested)
    code = record.get("legal_form_code")
    if form not in LEGAL_FORM_CODES or code not in COMPARED_FORM_CODES:
        return False
    return code != LEGAL_FORM_CODES[form]


def best_name_score(requested, record):
    """Return the best name score against the legal name and all translated names of a record."""
    return max((name_score(requested, name) for name in registered_names(record)), default=0.0)


def words_contained(requested, record):
    """Return True if every folded word of the requested name stands in a registered name."""
    wanted = set(split_legal_form(requested)[0])
    if not wanted:
        return False
    return any(wanted <= set(name_tokens(name)) for name in registered_names(record))


def compare_with_request(request, record):
    """Summarize how well one record fits the request, for the LLM and the logs."""
    comparison = {"name_score": None, "uid_equal": None, "locality_equal": None,
                  "contradictions": find_contradictions(request, record)}
    if request.get("name"):
        comparison["name_score"] = round(best_name_score(request["name"], record), 3)
    if request.get("uid"):
        comparison["uid_equal"] = request["uid"] == record.get("uid")
    if request.get("locality"):
        comparison["locality_equal"] = locality_fits(request["locality"], record)
    return comparison


def same_name_competitors(request, chosen_id, candidates, strong_score):
    """Return the other seen companies that fit the requested name as well as the chosen one."""
    if not request.get("name"):
        return []
    chosen_score = 0.0
    if chosen_id in candidates:
        chosen_score = name_score(request["name"], candidates[chosen_id]["legal_name"])
    competitors = []
    for company_id, candidate in candidates.items():
        if company_id == chosen_id:
            continue
        score = name_score(request["name"], candidate["legal_name"])
        if score < max(strong_score, chosen_score):
            continue
        # a place in the request separates companies with the same name
        wanted = request.get("locality")
        if wanted and not locality_matches(wanted, candidate.get("locality")):
            continue
        competitors.append(company_id)
    return competitors


def status_claims(text):
    """Return the phrases about status or ability that the register data cannot support."""
    return [match.group(0) for match in UNSUPPORTED_STATUS.finditer(text or "")]


def clean_free_text(text):
    """Remove the sentences with a status statement, return the text and what was removed."""
    removed, kept = [], []
    for sentence in re.split(SENTENCE_END, text or ""):
        # typographic apostrophes, so that "can’t" counts like "can't"
        states_the_limit = STATES_THE_LIMIT.search(sentence.replace("’", "'"))
        if status_claims(sentence) and not states_the_limit:
            removed.append(sentence)
        else:
            kept.append(sentence)
    return " ".join(kept).strip(), removed


def known_identifiers(request, records):
    """Return the identifiers the LLM may repeat, from the request and the fetched records."""
    known = set()
    if records:
        for record in records.values():
            known.add(record.get("uid"))
            known.add(record.get("chid"))
    if request:
        known.add(request.get("uid"))
    return {value.upper() for value in known if value}


def identifiers_in(sentence):
    """Return the UIDs and CH-IDs of a sentence in compact form, like CHE242294601."""
    found = set()
    for match in IDENTIFIER.finditer(sentence):
        found.add(re.sub(r"[^A-Z0-9]", "", match.group(0).upper()))
    return found


def remove_unknown_identifiers(text, known):
    """Remove the sentences with an unknown UID or CH-ID, return the text and what was removed."""
    removed, kept = [], []
    for sentence in re.split(SENTENCE_END, text or ""):
        if identifiers_in(sentence) - known:
            removed.append(sentence)
        else:
            kept.append(sentence)
    return " ".join(kept).strip(), removed


def liquidation_note(record):
    """Return a note on liquidation wording in a legal name, as text only and never as a status."""
    if not record or not record.get("legal_name"):
        return None
    match = STATUS_WORD_IN_NAME.search(record["legal_name"])
    if not match:
        return None
    return (f"The legal name contains the words '{match.group(0)}'. "
            "The dataset has no status field, so I report this only as text in the name.")


def _searches_ok(searches):
    """Split the searches of this turn into successful and failed ones."""
    ok = [search for search in searches if search["status"] == "ok"]
    failed = [search for search in searches if search["status"] != "ok"]
    return ok, failed


def _verdict(kind, outcome, message, notes=None):
    """Build the result of a check with verdict (accept, rework or override), outcome and reason."""
    return {"verdict": kind, "outcome": outcome, "message": message, "notes": notes or []}


def verify_match(decision, request, records, candidates, searches, strong_score,
                 uid_entries=None):
    """Check a proposed MATCH against the fetched record."""
    company_id = decision.get("company_id")
    if not company_id:
        return _verdict("rework", "MATCH",
                        "A MATCH needs company_id. Name the company or choose another outcome.")
    if company_id not in records:
        return _verdict("rework", "MATCH",
                        f"Company {company_id} has no fetched record. "
                        "Call get_company_records first.")
    record = records[company_id]
    contradictions = find_contradictions(request, record)
    if contradictions:
        return _verdict("override", "AMBIGUOUS", "The request contradicts the register record.",
                        contradictions)
    if request.get("uid"):
        return _verify_uid_match(request, record, searches, strong_score, uid_entries or {})
    score = best_name_score(request["name"], record)
    if score < strong_score:
        return _verdict("override", "AMBIGUOUS",
                        f"The name score {score} is below {strong_score}, so this is only a "
                        "possible candidate.",
                        [f"The name of {record['legal_name']} is not close enough to "
                         f"{request['name']}."])
    if score < 1.0 and not words_contained(request["name"], record):
        return _verdict("override", "AMBIGUOUS",
                        f"The name score {score} is high, but the spelling differs, "
                        "so the user has to confirm it.",
                        [f"{request['name']} and {record['legal_name']} can be two different "
                         "companies with similar names."])
    competitors = same_name_competitors(request, company_id, candidates, strong_score)
    if competitors:
        return _verdict("override", "AMBIGUOUS",
                        f"Other seen companies fit the request as well: {competitors}.",
                        ["The request does not separate these companies, "
                         "a further detail is needed."])
    notes = []
    if not request.get("locality") and not request.get("address"):
        notes.append("The match rests on the name alone, no place or address was given.")
    if any(search.get("truncated") for search in searches):
        notes.append("At least one search reached the row limit, so other companies may exist.")
    return _verdict("accept", "MATCH",
                    f"Name score {score}, no contradiction, no competing company seen.", notes)


def _verify_uid_match(request, record, searches, strong_score, uid_entries):
    """Check a proposed MATCH for a request with a UID."""
    if record.get("uid") != request["uid"]:
        return _verdict("override", "AMBIGUOUS", "The record has another UID than the request.")
    if request.get("name") and best_name_score(request["name"], record) < strong_score:
        return _verdict("override", "AMBIGUOUS",
                        "The UID exists, but its register name does not fit the requested name.",
                        [f"The UID belongs to {record['legal_name']}, the request says "
                         f"{request['name']}."])
    entries = uid_lookup_entries(request, searches, uid_entries)
    if entries is None:
        # without the lookup the check cannot know whether the UID has more than one entry
        return _verdict("rework", "MATCH", "Run lookup_uid for the requested UID before MATCH.")
    if entries > 1 and not request.get("locality") and not request.get("address"):
        return _verdict("override", "AMBIGUOUS",
                        f"This UID stands on {entries} register entries, "
                        "so the UID alone does not select one.",
                        ["The register holds more than one entry with this UID, for example for a "
                         "company with two seats."])
    return _verdict("accept", "MATCH", "The UID is equal and nothing contradicts the record.")


def asked_for_fuzzy(search):
    """Return True for a text search that was called with the fuzzy mode."""
    if search["op"] != "text_search":
        return False
    params = search.get("params")
    return bool(params and params.get("fuzzy"))


def is_fuzzy_search(search):
    """Return True for a completed text search that allowed small spelling differences."""
    return asked_for_fuzzy(search) and search["status"] == "ok"


def uid_lookup_entries(request, searches, uid_entries):
    """Return how many entries the lookup of the requested UID returned, None without a lookup."""
    uid = request["uid"]
    counts = [search["hits"] or 0 for search in searches
              if search["op"] == "lookup_uid" and search["status"] == "ok"
              and (search.get("params") or {}).get("uid") == uid]
    # a follow-up makes no new search, so the count stored with the request also counts
    if uid in uid_entries:
        counts.append(uid_entries[uid])
    if not counts:
        return None
    return max(counts)


def fuzzy_step_missing(request, candidates, searches, strong_score):
    """Return True if a name was given, no seen company fits it and no fuzzy search completed."""
    if not request.get("name"):
        return False
    names = [candidate["legal_name"] for candidate in candidates.values()]
    if any(name_score(request["name"], name) >= strong_score for name in names):
        return False
    return not any(is_fuzzy_search(search) for search in searches)


def verify_no_match(request, candidates, searches, strong_score, uid_entries=None):
    """Check a proposed NO_MATCH, a failed search or a skipped scan is not an empty register."""
    ok, failed = _searches_ok(searches)
    if failed and not ok:
        return _verdict("override", "TECHNICAL_ERROR",
                        "All searches failed, so nothing can be said about the register.")
    if request.get("uid"):
        entries = uid_lookup_entries(request, searches, uid_entries or {})
        if entries is None:
            if any(search["op"] == "lookup_uid" for search in failed):
                return _verdict("override", "TECHNICAL_ERROR",
                                "The lookup of the requested UID did not complete, "
                                "so a missing hit proves nothing.")
            return _verdict("rework", "NO_MATCH",
                            "Run lookup_uid for the requested UID before NO_MATCH.")
        if entries:
            # a UID that the register holds is a hit, whatever the other searches returned
            return _verdict("rework", "NO_MATCH",
                            f"The lookup of the requested UID returned {entries} register "
                            "entries, so NO_MATCH does not fit. Fetch them and decide MATCH or "
                            "AMBIGUOUS.")
    if request.get("name"):
        strong = [company_id for company_id, candidate in candidates.items()
                  if name_score(request["name"], candidate["legal_name"]) >= strong_score]
        if strong:
            return _verdict("rework", "NO_MATCH",
                            f"Seen companies fit the name: {strong}. "
                            "Fetch them or decide AMBIGUOUS.")
        scans = [search for search in searches
                 if search["op"] == "scan_legal_names" and search["status"] == "ok"]
        if not scans:
            if failed:
                return _verdict("override", "TECHNICAL_ERROR",
                                "The scan of the legal names did not complete, "
                                "so a missing hit proves nothing.")
            return _verdict("rework", "NO_MATCH",
                            "The text index does not cover all legal names. "
                            "Run scan_legal_names before NO_MATCH.")
        if all(scan.get("truncated") for scan in scans):
            # a cut off scan may have stopped before the company, so it is not a completed search
            return _verdict("rework", "NO_MATCH",
                            "The scan reached the row limit, so companies may be missing. Scan "
                            "again with a more specific part of the name.")
        if fuzzy_step_missing(request, candidates, searches, strong_score):
            if any(asked_for_fuzzy(search) for search in failed):
                # a fuzzy search that failed is not a completed search
                return _verdict("override", "TECHNICAL_ERROR",
                                "The fuzzy search did not complete, "
                                "so a missing hit proves nothing.")
            return _verdict("rework", "NO_MATCH",
                            "The searches are not tolerant of spelling. Run text_search with "
                            "fuzzy=true before NO_MATCH.")
    notes = []
    if failed:
        notes.append("At least one search failed during this turn, so the result is less certain.")
    return _verdict("accept", "NO_MATCH", "The completed searches found no fitting company.", notes)


def verify_decision(decision, state, strong_score):
    """Check the proposed decision against the evidence in the state."""
    outcome = decision.get("outcome")
    request = state["request"]
    records = state.get("records") or {}
    candidates = state.get("candidates") or {}
    searches = state.get("searches") or []
    if outcome not in OUTCOMES:
        return _verdict("rework", "AMBIGUOUS", f"Unknown outcome {outcome!r}.")
    if outcome == "MATCH":
        return verify_match(decision, request, records, candidates, searches, strong_score,
                            state.get("uid_entries"))
    if outcome == "NO_MATCH":
        return verify_no_match(request, candidates, searches, strong_score,
                               state.get("uid_entries"))
    if outcome == "AMBIGUOUS":
        if not candidates:
            return _verdict("rework", "AMBIGUOUS",
                            "AMBIGUOUS needs seen candidates. Search first or decide NO_MATCH.")
        if fuzzy_step_missing(request, candidates, searches, strong_score):
            return _verdict("rework", "AMBIGUOUS",
                            "No seen company fits the name. "
                            "Run text_search with fuzzy=true before you decide.")
        return _verdict("accept", "AMBIGUOUS",
                        "The agent reported more than one possibility or a conflict.")
    _, failed = _searches_ok(searches)
    if not failed:
        return _verdict("rework", "TECHNICAL_ERROR",
                        "No tool call failed, so TECHNICAL_ERROR does not fit.")
    return _verdict("accept", "TECHNICAL_ERROR", "A tool call failed.")
