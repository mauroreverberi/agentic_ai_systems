"""Deterministic input checks and name comparison.

Nothing in this module calls the network, so every rule can be tested alone.
"""

import re
import unicodedata
from difflib import SequenceMatcher

MAX_NAME_LENGTH = 200
MAX_FIELD_LENGTH = 120
# weights of the first eight digits for the check digit of a UID (modulo 11)
UID_WEIGHTS = (5, 4, 3, 2, 7, 6, 5, 4)

# only the common Swiss forms, a word in this list says nothing about who the company is
LEGAL_FORM_WORDS = {
    "ag": "AG", "sa": "AG", "ltd": "AG", "gmbh": "GmbH", "sarl": "GmbH", "sagl": "GmbH",
    "llc": "GmbH", "genossenschaft": "Genossenschaft", "cooperative": "Genossenschaft",
    "stiftung": "Stiftung", "verein": "Verein", "kg": "KG", "kollektivgesellschaft": "KG",
}
LEGAL_FORM_PHRASES = (("s", "a"), ("s", "a", "r", "l"), ("s", "a", "g", "l"))
ASCII_FOLD = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "æ": "ae", "œ": "oe"}


class InvalidInput(ValueError):
    """Raised when a request cannot be accepted."""


def clean_text(value, field, max_length=MAX_FIELD_LENGTH):
    """Return one text field with single spaces, or None if the field is empty."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidInput(f"{field} must be text")
    text = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in text):
        raise InvalidInput(f"{field} contains control characters")
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > max_length:
        raise InvalidInput(f"{field} is longer than {max_length} characters")
    return text


def normalize_uid(value):
    """Check a Swiss UID with its check digit and return the compact form, like CHE242294601."""
    pattern = r"CHE[-\s]?(\d{3})[.\s]?(\d{3})[.\s]?(\d{3})(\s?(MWST|TVA|IVA|HR|RC|RI))?"
    match = re.fullmatch(pattern, value.strip().upper())
    if not match:
        raise InvalidInput("UID does not have the format CHE-123.456.789")
    digits = match.group(1) + match.group(2) + match.group(3)
    remainder = sum(int(d) * w for d, w in zip(digits[:8], UID_WEIGHTS)) % 11
    expected = (11 - remainder) % 11
    if expected == 10 or expected != int(digits[8]):
        raise InvalidInput("UID check digit is wrong")
    return "CHE" + digits


def format_uid(uid):
    """Return a compact UID in the display form, like CHE-242.294.601."""
    if uid and re.fullmatch(r"CHE\d{9}", uid):
        return f"CHE-{uid[3:6]}.{uid[6:9]}.{uid[9:12]}"
    return uid


def validate_request(raw):
    """Return the request with clean values, a name or a UID must be set."""
    if not isinstance(raw, dict):
        raise InvalidInput("request must be a dictionary")
    unknown = set(raw) - {"name", "locality", "address", "uid"}
    if unknown:
        raise InvalidInput(f"unknown fields: {sorted(unknown)}")
    request = {
        "name": clean_text(raw.get("name"), "name", MAX_NAME_LENGTH),
        "locality": clean_text(raw.get("locality"), "locality"),
        "address": clean_text(raw.get("address"), "address"),
        "uid": clean_text(raw.get("uid"), "uid", 30),
    }
    if request["uid"]:
        request["uid"] = normalize_uid(request["uid"])
    if not request["name"] and not request["uid"]:
        raise InvalidInput("a company name or a UID is required")
    if request["name"] and not re.search(r"\w{2}", request["name"]):
        raise InvalidInput("name does not contain a usable word")
    return request


def fold(text):
    """Return the text in lowercase without accents, Müller gives mueller."""
    lowered = unicodedata.normalize("NFC", text).casefold()
    written_out = "".join(ASCII_FOLD.get(ch, ch) for ch in lowered)
    decomposed = unicodedata.normalize("NFKD", written_out)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def name_tokens(name):
    """Return the folded words of a name without punctuation."""
    return re.findall(r"[a-z0-9]+", fold(name))


def split_legal_form(name):
    """Return the words of a name without its legal form, and the form like AG or None."""
    tokens = name_tokens(name)
    form = None
    # forms written with dots like S.A. or S.a.r.l. arrive as single letters
    for phrase in sorted(LEGAL_FORM_PHRASES, key=len, reverse=True):
        size = len(phrase)
        for i in range(len(tokens) - size + 1):
            if tuple(tokens[i:i + size]) == phrase:
                form = "AG" if size == 2 else "GmbH"
                tokens = tokens[:i] + tokens[i + size:]
                break
        if form:
            break
    core = []
    for token in tokens:
        if token in LEGAL_FORM_WORDS and len(tokens) > 1:
            form = form or LEGAL_FORM_WORDS[token]
        else:
            core.append(token)
    return core, form


def name_score(requested, registered):
    """Return how similar two names are, from 0 to 1. The legal form is ignored."""
    core_a, _ = split_legal_form(requested)
    core_b, _ = split_legal_form(registered)
    if not core_a or not core_b:
        return 0.0
    if sorted(core_a) == sorted(core_b):
        return 1.0
    set_a, set_b = set(core_a), set(core_b)
    word_overlap = len(set_a & set_b) / len(set_a | set_b)
    char_similarity = SequenceMatcher(None, " ".join(core_a), " ".join(core_b)).ratio()
    return round(max(word_overlap, char_similarity), 3)


def legal_forms_conflict(requested, registered):
    """Return True if both names state a legal form and the two forms differ."""
    _, form_a = split_legal_form(requested)
    _, form_b = split_legal_form(registered)
    return bool(form_a and form_b and form_a != form_b)


def locality_matches(requested, registered):
    """Return True if all words of the requested place appear in the registered place."""
    if not requested or not registered:
        return False
    wanted = set(name_tokens(requested))
    return bool(wanted) and wanted <= set(name_tokens(registered))
