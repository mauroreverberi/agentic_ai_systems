"""Register tool, the only code that talks to the Zefix graph on LINDAS.

A caller chooses one of the fixed operations and passes plain values. This module checks and
escapes them before they go into a prepared query, so no caller writes SPARQL. In snapshot mode
(default) the answers come only from stored real responses, live asks the endpoint, and record
asks it only for queries without a stored response and stores the answer.
"""

import csv
import hashlib
import io
import json
import logging
import pathlib
import time
from datetime import datetime, timezone

import requests

from .config import SNAPSHOT_DIR
from .validation import InvalidInput, escape_sparql_literal, search_words

# the retry messages go to this logger, so a caller can show or hide them
LOG = logging.getLogger("zefix_agent.register")
ORG_TYPE = "https://schema.ld.admin.ch/ZefixOrganisation"
COMPANY_BASE = "https://register.ld.admin.ch/zefix/company/"
LEGAL_FORM_BASE = "https://ld.admin.ch/ech/97/legalforms/"
MUNICIPALITY_BASE = "https://ld.admin.ch/municipality/"
SCHEMA = "http://schema.org/"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MISS_MESSAGE = "No stored register response for this exact query. This says nothing about the register."
PREFIXES = """PREFIX schema: <http://schema.org/>
PREFIX luc: <http://www.ontotext.com/connectors/lucene#>
PREFIX luc-index: <http://www.ontotext.com/connectors/lucene/instance#>
"""
# answers of the simulated failures, as status code, body and Retry-After header
SIMULATED_ANSWERS = {
    "http_503": (503, "simulated service unavailable", None),
    "http_429": (429, "simulated rate limit", "1"),
    "malformed": (200, "<html>simulated unexpected body</html>", None),
}


def utc_now():
    """Return the current UTC time in ISO 8601 format with seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def error(error_type, message, attempts, simulated):
    """Return the result of a failed register operation."""
    return {"status": "error", "error_type": error_type, "message": message,
            "attempts": attempts, "simulated": simulated}


class SnapshotMiss(Exception):
    """Raised in snapshot mode when no stored response exists for a query."""


class FaultPlan:
    """Simulated register failures for tests, never active in a normal run."""

    def __init__(self, faults=None):
        """Store the faults, each with op ("*" for all), kind and times (failed attempts)."""
        # the kinds are timeout, http_503, http_429, connection and malformed
        self.faults = []
        for fault in faults or []:
            fault = dict(fault)
            fault.setdefault("times", 1)
            self.faults.append(fault)

    def has_fault(self, op):
        """Return True if a failure is still planned for this operation."""
        return any(fault["op"] in (op, "*") and fault["times"] > 0 for fault in self.faults)

    def next_fault(self, op):
        """Return the kind of failure for this attempt and use it up, or None."""
        for fault in self.faults:
            if fault["op"] in (op, "*") and fault["times"] > 0:
                fault["times"] -= 1
                return fault["kind"]
        return None


class SnapshotStore:
    """Stored real register responses, one JSON file per canonical operation."""

    def __init__(self, directory=SNAPSHOT_DIR):
        """Remember the folder with the snapshot files."""
        self.directory = pathlib.Path(directory)

    @staticmethod
    def key(op, params):
        """Return a stable name for an operation, like text_search_1a2b3c4d5e6f."""
        canonical = json.dumps({"op": op, "params": params}, sort_keys=True, ensure_ascii=False)
        return f"{op}_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}"

    def load(self, op, params):
        """Return a stored response, or None if there is no snapshot."""
        path = self.directory / f"{self.key(op, params)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, op, params, query, body, retrieved_at, seconds):
        """Store a real response word for word with its origin and return the file path."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.key(op, params)}.json"
        snapshot = {"op": op, "params": params, "query": query, "retrieved_at": retrieved_at,
                    "response_seconds": round(seconds, 3), "body": body}
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
        return path


def locality_filter(locality):
    """Return the SPARQL filter for a place, or an empty text if no place is given."""
    if not locality:
        return ""
    place = escape_sparql_literal(locality.lower())
    return f'\n  FILTER(CONTAINS(LCASE(?locality), "{place}"))'


def query_exact_name(graph, limit, name):
    """Return the SPARQL query for one exact legal name (fast, case sensitive)."""
    return PREFIXES + f"""SELECT ?company ?name ?locality WHERE {{ GRAPH <{graph}> {{
  ?company schema:legalName "{escape_sparql_literal(name)}" ; schema:legalName ?name ;
           schema:address ?a . ?a schema:addressLocality ?locality }} }} LIMIT {limit}"""


def query_text_search(graph, limit, candidate_limit, lucene_query, locality):
    """Return the SPARQL query for a search on the Lucene text index, which covers schema:name."""
    return PREFIXES + f"""SELECT ?company ?name ?locality WHERE {{
  ?search a luc-index:lindas_label ; luc:query "{escape_sparql_literal(lucene_query)}" ;
          luc:limit "{candidate_limit}" ; luc:entities ?company .
  GRAPH <{graph}> {{ ?company a <{ORG_TYPE}> ; schema:legalName ?name ;
          schema:address ?a . ?a schema:addressLocality ?locality }}{locality_filter(locality)}
}} LIMIT {limit}"""


def query_scan(graph, limit, fragment, locality):
    """Return the SPARQL query that scans all legal names, it finds what the index misses."""
    return PREFIXES + f"""SELECT ?company ?name ?locality WHERE {{ GRAPH <{graph}> {{
  ?company a <{ORG_TYPE}> ; schema:legalName ?name ; schema:address ?a . ?a schema:addressLocality ?locality .
  FILTER(CONTAINS(LCASE(?name), "{escape_sparql_literal(fragment)}")){locality_filter(locality)}
}} }} LIMIT {limit}"""


def query_uid(graph, uid):
    """Return the SPARQL query for one UID."""
    return PREFIXES + f"""SELECT ?company ?name ?locality WHERE {{ GRAPH <{graph}> {{
  ?id schema:name "CompanyUID" ; schema:value "{uid}" .
  ?company schema:identifier ?id ; schema:legalName ?name ; schema:address ?a . ?a schema:addressLocality ?locality
}} }} LIMIT 5"""


def query_record(graph, company_id):
    """Return the SPARQL query for the full record of a company, one step into its linked nodes."""
    return f"""SELECT ?p ?o ?p2 ?o2 WHERE {{ GRAPH <{graph}> {{
  <{COMPANY_BASE}{company_id}> ?p ?o . OPTIONAL {{ ?o ?p2 ?o2 }}
}} }} LIMIT 200"""


def query_label(iri):
    """Return the SPARQL query for the labels of a legal form or a municipality."""
    return PREFIXES + f"""SELECT ?label (LANG(?label) AS ?lang) WHERE {{ GRAPH ?g {{ <{iri}> schema:name ?label }} }} LIMIT 10"""


def query_count(graph):
    """Return the SPARQL query that counts all organizations in the Zefix graph."""
    return f"""SELECT (COUNT(?company) AS ?n) WHERE {{ GRAPH <{graph}> {{ ?company a <{ORG_TYPE}> }} }}"""


def parse_csv(body, expected_columns):
    """Read a CSV answer as a list of rows and check that it has the expected columns."""
    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames != expected_columns:
        raise ValueError(f"unexpected columns {reader.fieldnames}")
    return list(reader)


def build_lucene_query(words, fuzzy, prefix):
    """Build the Lucene query from plain words, like label:(mueller~ AND holzbau*)."""
    parts = []
    for i, word in enumerate(words):
        if prefix and i == len(words) - 1:
            parts.append(word + "*")
        elif fuzzy and len(word) >= 4:
            parts.append(word + "~")
        else:
            parts.append(word)
    return "label:(" + " AND ".join(parts) + ")"


def clean_register_text(value):
    """Remove the word null from a register value, return None if nothing is left."""
    words = [word for word in (value or "").split() if word != "null"]
    return " ".join(words) or None


def clean_locality(locality):
    """Return the place without outer spaces, or None if no place is given."""
    if not locality:
        return None
    return locality.strip()


def parse_record(company_id, rows):
    """Turn the raw triples of a company into a flat record, no value is added by this code."""
    record = {
        "company_id": company_id, "company_uri": COMPANY_BASE + company_id, "legal_name": None,
        "names": [], "purpose": None, "uid": None, "chid": None, "ehraid": None, "street": None,
        "postal_code": None, "locality": None, "canton": None, "legal_form_code": None, "municipality_id": None,
    }
    address_fields = {SCHEMA + "streetAddress": "street", SCHEMA + "postalCode": "postal_code",
                      SCHEMA + "addressLocality": "locality", SCHEMA + "addressRegion": "canton"}
    id_fields = {"CompanyUID": "uid", "CompanyCHID": "chid", "CompanyEHRAID": "ehraid"}
    id_nodes = {}
    for row in rows:
        # p and o are the property and the value, p2 and o2 the same one step further
        p, o, p2, o2 = row["p"], row["o"], row.get("p2"), row.get("o2")
        if p == SCHEMA + "legalName":
            record["legal_name"] = o
        elif p == SCHEMA + "name" and o not in record["names"]:
            record["names"].append(o)
        elif p == SCHEMA + "description":
            # the register holds the text "null" for some branch offices
            record["purpose"] = None
            if o not in ("", "null"):
                record["purpose"] = o
        elif p == SCHEMA + "additionalType" and o.startswith(LEGAL_FORM_BASE):
            record["legal_form_code"] = o[len(LEGAL_FORM_BASE):]
        elif p == "https://schema.ld.admin.ch/municipality" and o.startswith(MUNICIPALITY_BASE):
            record["municipality_id"] = o[len(MUNICIPALITY_BASE):]
        elif p == SCHEMA + "address" and p2 in address_fields:
            record[address_fields[p2]] = clean_register_text(o2)
        elif p == SCHEMA + "identifier" and p2:
            node = id_nodes.setdefault(o, {})
            node[p2] = o2
    for node in id_nodes.values():
        field = id_fields.get(node.get(SCHEMA + "name"))
        if field:
            record[field] = node.get(SCHEMA + "value")
    return record


class RegisterTool:
    """Prepared queries against the Zefix graph, with timeouts, retries, rate limit and snapshots."""

    def __init__(self, register_config, store=None, fault_plan=None, session=None, sleep=time.sleep):
        """Set up the tool, tests replace sleep so that they do not wait."""
        self.config = register_config
        self.mode = register_config.get("mode", "snapshot")
        self.store = store or SnapshotStore()
        self.fault_plan = fault_plan or FaultPlan()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": register_config["user_agent"], "Accept": "text/csv"})
        self.sleep = sleep
        self.memory_cache = {}
        self.last_request_at = 0.0
        self.http_requests = 0

    def _wait_for_rate_limit(self):
        """Keep a minimum time between requests to the public endpoint."""
        pause = self.config["min_seconds_between_requests"] - (time.monotonic() - self.last_request_at)
        if pause > 0 and self.http_requests:
            self.sleep(pause)
        self.last_request_at = time.monotonic()

    def _stored(self, op, params):
        """Return the stored response in snapshot and record mode, or None."""
        if self.mode not in ("snapshot", "record"):
            return None
        return self.store.load(op, params)

    def _simulate(self, fault):
        """Return the answer of a simulated failure, or raise it like the requests library."""
        if fault == "timeout":
            raise requests.Timeout("simulated timeout")
        if fault == "connection":
            raise requests.ConnectionError("simulated connection failure")
        status_code, body, retry_after = SIMULATED_ANSWERS[fault]
        return {"status_code": status_code, "body": body, "retry_after": retry_after, "stored": None}

    def _send(self, op, params, query):
        """Return the stored response if there is one, otherwise ask the endpoint."""
        stored = self._stored(op, params)
        if stored:
            return {"status_code": 200, "body": stored["body"], "retry_after": None, "stored": stored}
        if self.mode == "snapshot":
            # a retry after a simulated failure must not reach the real endpoint
            raise SnapshotMiss()
        self._wait_for_rate_limit()
        self.http_requests += 1
        timeout = (self.config["connect_timeout_seconds"], self.config["read_timeout_seconds"])
        response = self.session.post(self.config["endpoint"], data={"query": query}, timeout=timeout)
        return {"status_code": response.status_code, "body": response.text,
                "retry_after": response.headers.get("Retry-After"), "stored": None}

    def _fetch(self, op, params, query):
        """Return the body for an operation from the cache, the snapshots or the endpoint."""
        cache_key = SnapshotStore.key(op, params)
        if self.fault_plan.has_fault(op):
            return self._fetch_with_retries(op, params, query, cache_key)
        if cache_key in self.memory_cache:
            cached = dict(self.memory_cache[cache_key])
            cached["source"] = "memory_cache"
            return cached
        stored = self._stored(op, params)
        if stored:
            result = {"status": "ok", "body": stored["body"], "retrieved_at": stored["retrieved_at"],
                      "source": f"snapshot:{cache_key}", "attempts": 0, "simulated": False}
            self.memory_cache[cache_key] = result
            return result
        if self.mode == "snapshot":
            return error("snapshot_miss", MISS_MESSAGE, 0, False)
        return self._fetch_with_retries(op, params, query, cache_key)

    def _fetch_with_retries(self, op, params, query, cache_key):
        """Ask the endpoint with a bounded number of attempts and exponential backoff."""
        max_attempts = self.config["max_attempts"]
        error_type, message, simulated = "unknown", "", False
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            retry_after = None
            fault = self.fault_plan.next_fault(op)
            if fault:
                simulated = True
            try:
                if fault:
                    answer = self._simulate(fault)
                else:
                    answer = self._send(op, params, query)
            except SnapshotMiss:
                return error("snapshot_miss", MISS_MESSAGE, attempt, simulated)
            except requests.Timeout as exc:
                error_type, message = "timeout", str(exc)
            except requests.ConnectionError as exc:
                error_type, message = "connection", str(exc)
            except requests.RequestException as exc:
                return error("request_failed", str(exc), attempt, simulated)
            else:
                retry_after = answer["retry_after"]
                if answer["status_code"] == 200:
                    return self._success(op, params, query, cache_key, answer, fault, attempt,
                                         simulated, time.perf_counter() - started)
                error_type, message = f"http_{answer['status_code']}", answer["body"][:200]
                if answer["status_code"] not in RETRYABLE_STATUS:
                    return error(error_type, message, attempt, simulated)
            LOG.warning("register attempt %s of %s failed for %s: %s", attempt, max_attempts, op, error_type)
            if attempt < max_attempts:
                self.sleep(self._backoff_seconds(attempt, retry_after))
        return error(error_type, message[:200], max_attempts, simulated)

    def _success(self, op, params, query, cache_key, answer, fault, attempt, simulated, seconds):
        """Build the result of an answered attempt, cache it and store a new real response."""
        stored = answer["stored"]
        if stored:
            retrieved_at = stored["retrieved_at"]
            source = f"snapshot:{cache_key}"
        else:
            retrieved_at = utc_now()
            source = "simulated" if fault else "live"
        result = {"status": "ok", "body": answer["body"], "retrieved_at": retrieved_at, "source": source,
                  "attempts": attempt, "simulated": simulated, "seconds": round(seconds, 3)}
        if not fault:
            cached = dict(result)
            cached["simulated"] = False
            self.memory_cache[cache_key] = cached
            if self.mode == "record" and not stored:
                self.store.save(op, params, query, answer["body"], retrieved_at, seconds)
        return result

    def _backoff_seconds(self, attempt, retry_after):
        """Return the seconds to wait before the next attempt, with the Retry-After header."""
        wait = self.config["backoff_base_seconds"] * (2 ** (attempt - 1))
        if retry_after and str(retry_after).isdigit():
            wait = max(wait, min(float(retry_after), self.config["max_retry_after_seconds"]))
        return wait

    def _candidates(self, op, params, query):
        """Run a search operation and return its rows as candidates."""
        fetched = self._fetch(op, params, query)
        if fetched["status"] != "ok":
            outcome = dict(fetched)
            outcome.update(op=op, params=params, candidates=[])
            return outcome
        try:
            rows = parse_csv(fetched["body"], ["company", "name", "locality"])
            if any(not row["company"].startswith(COMPANY_BASE) for row in rows):
                raise ValueError("unexpected company address")
        except (ValueError, csv.Error) as exc:
            outcome = error("malformed_response", str(exc), fetched["attempts"], fetched["simulated"])
            outcome.update(op=op, params=params, candidates=[])
            return outcome
        candidates = [{"company_id": row["company"][len(COMPANY_BASE):], "legal_name": row["name"],
                       "locality": row["locality"]} for row in rows]
        return {"status": "ok", "op": op, "params": params, "candidates": candidates,
                "retrieved_at": fetched["retrieved_at"], "source": fetched["source"],
                "attempts": fetched["attempts"], "simulated": fetched["simulated"],
                "truncated": len(candidates) >= self.config["result_limit"]}

    def lookup_uid(self, uid):
        """Find the company with one UID, given in the compact form."""
        return self._candidates("lookup_uid", {"uid": uid}, query_uid(self.config["graph"], uid))

    def exact_name(self, name):
        """Find companies with exactly this legal name."""
        limit = self.config["result_limit"]
        return self._candidates("exact_name", {"name": name}, query_exact_name(self.config["graph"], limit, name))

    def text_search(self, text, locality=None, fuzzy=False, prefix=False):
        """Search the text index by words, which also finds translated names."""
        words = search_words(text)
        if not words:
            raise InvalidInput("the search text has no usable word")
        place = clean_locality(locality)
        params = {"words": sorted(words), "locality": None, "fuzzy": fuzzy, "prefix": prefix}
        if prefix:
            # the word order only matters when the last word is a prefix
            params["words"] = words
        if place:
            params["locality"] = place.lower()
        lucene = build_lucene_query(words, fuzzy, prefix)
        query = query_text_search(self.config["graph"], self.config["result_limit"],
                                  self.config["lucene_candidate_limit"], lucene, place)
        return self._candidates("text_search", params, query)

    def scan_legal_names(self, fragment, locality=None):
        """Scan all legal names for a part of a name, this is the slow way."""
        cleaned = " ".join(fragment.lower().split())
        if len(cleaned) < 3:
            raise InvalidInput("the name part for a scan needs at least 3 characters")
        place = clean_locality(locality)
        params = {"fragment": cleaned, "locality": None}
        if place:
            params["locality"] = place.lower()
        query = query_scan(self.config["graph"], self.config["result_limit"], cleaned, place)
        return self._candidates("scan_legal_names", params, query)

    def count_companies(self):
        """Count the organizations in the Zefix graph."""
        fetched = self._fetch("count_companies", {}, query_count(self.config["graph"]))
        if fetched["status"] != "ok":
            outcome = dict(fetched)
            outcome["count"] = None
            return outcome
        try:
            count = int(parse_csv(fetched["body"], ["n"])[0]["n"])
        except (ValueError, IndexError, csv.Error) as exc:
            return {"status": "error", "error_type": "malformed_response", "message": str(exc), "count": None}
        return {"status": "ok", "count": count, "retrieved_at": fetched["retrieved_at"], "source": fetched["source"]}

    def _label(self, kind, code, iri):
        """Read the labels of a legal form or a municipality, a missing label is no error."""
        fetched = self._fetch(kind, {"code": code}, query_label(iri))
        if fetched["status"] != "ok":
            return {}
        try:
            rows = parse_csv(fetched["body"], ["label", "lang"])
        except (ValueError, csv.Error):
            return {}
        return {row["lang"] or "none": row["label"] for row in rows}

    def company_record(self, company_id):
        """Read the full record of a company with the labels of legal form and municipality."""
        if not str(company_id).isdigit():
            raise InvalidInput("company_id must contain only digits")
        company_id = str(company_id)
        params = {"company_id": company_id}
        fetched = self._fetch("company_record", params, query_record(self.config["graph"], company_id))
        if fetched["status"] != "ok":
            outcome = dict(fetched)
            outcome.update(op="company_record", params=params, record=None)
            return outcome
        try:
            rows = parse_csv(fetched["body"], ["p", "o", "p2", "o2"])
        except (ValueError, csv.Error) as exc:
            outcome = error("malformed_response", str(exc), fetched["attempts"], fetched["simulated"])
            outcome.update(op="company_record", params=params, record=None)
            return outcome
        outcome = {"status": "ok", "op": "company_record", "params": params, "attempts": fetched["attempts"],
                   "simulated": fetched["simulated"], "record": None,
                   "retrieved_at": fetched["retrieved_at"], "source": fetched["source"]}
        if rows:
            outcome["record"] = self._with_labels(parse_record(company_id, rows))
        return outcome

    def _with_labels(self, record):
        """Add the names of the legal form and the municipality to a record."""
        record["legal_form"] = None
        record["municipality"] = None
        code = record["legal_form_code"]
        if code:
            labels = self._label("legal_form_label", code, LEGAL_FORM_BASE + code)
            record["legal_form"] = labels.get("de") or labels.get("fr") or next(iter(labels.values()), None)
        municipality_id = record["municipality_id"]
        if municipality_id:
            labels = self._label("municipality_label", municipality_id, MUNICIPALITY_BASE + municipality_id)
            record["municipality"] = next(iter(labels.values()), None)
        return record
