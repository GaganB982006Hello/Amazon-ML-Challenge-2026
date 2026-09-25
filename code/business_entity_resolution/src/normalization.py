"""Text normalization module for Amazon ML Challenge 2026.

Provides multi-representation normalization for business names and addresses.
Preserves non-Latin scripts (Devanagari, Tamil, etc.) while stripping accents
from Latin scripts. Extracts numeric identifiers (house numbers, postal codes).
"""

from collections import Counter
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple


# Regex for cleaning leading/trailing decorative noise
LEADING_DECOR_RE = re.compile(r"^[^a-zA-Z0-9\u0900-\u0D7F]+", re.UNICODE)
TRAILING_DECOR_RE = re.compile(r"[^a-zA-Z0-9\u0900-\u0D7F]+$", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

# Legal entity suffixes across English, French, Hindi, and Tamil
LEGAL_SUFFIXES_EN = [
    r"\bprivate\s+limited\b",
    r"\bpvt\s+ltd\b",
    r"\bpvt\s+limited\b",
    r"\bprivate\s+ltd\b",
    r"\blimited\b",
    r"\bltd\b",
    r"\bincorporated\b",
    r"\binc\b",
    r"\bcorporation\b",
    r"\bcorp\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bcompany\b",
    r"\bco\b",
    r"\bgmbh\b",
]

LEGAL_SUFFIXES_FR = [
    r"\bsarl\b",
    r"\bs\.a\.r\.l\b",
    r"\bsas\b",
    r"\bs\.a\.s\b",
    r"\bsci\b",
    r"\bs\.c\.i\b",
    r"\bsa\b",
    r"\beurl\b",
    r"\bsnc\b",
]

LEGAL_SUFFIXES_INDIC = [
    r"प्राइवेट\s+लिमिटेड",
    r"लिमिटेड",
    r"एलएलपी",
    r"कंपनी",
    r"எல்எல்பி",
]

ALL_LEGAL_SUFFIXES_RE = re.compile(
    "|".join(LEGAL_SUFFIXES_EN + LEGAL_SUFFIXES_FR + LEGAL_SUFFIXES_INDIC),
    re.IGNORECASE | re.UNICODE,
)

# Address abbreviations dictionary
STREET_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "ter": "terrace",
    "pkwy": "parkway",
    "pl": "place",
    "hwy": "highway",
    "sq": "square",
    "cir": "circle",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "no": "number",
}

# Common state abbreviations
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}

INDIA_STATES = {
    "dl": "delhi", "hr": "haryana", "up": "uttar pradesh", "mh": "maharashtra",
    "ka": "karnataka", "rj": "rajasthan", "wb": "west bengal", "gj": "gujarat",
    "ap": "andhra pradesh", "ts": "telangana", "kl": "kerala", "mp": "madhya pradesh",
    "pb": "punjab", "or": "odisha", "od": "odisha", "tn": "tamil nadu",
}


def strip_accents_latin(text: str) -> str:
    """Strip combining diacritics only from Latin characters while preserving Indic scripts."""
    result = []
    # Decompose into base char and diacritics
    nfkd = unicodedata.normalize("NFKD", text)
    for char in nfkd:
        # Check if character is a combining diacritic
        if unicodedata.combining(char):
            # Check if this diacritic is in the standard Latin combining range (0x0300 - 0x036F)
            if 0x0300 <= ord(char) <= 0x036F:
                continue  # Drop Latin accent
        result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def clean_text_base(text: str) -> str:
    """Basic text cleaning: strip decoration, lowercase, normalize whitespace."""
    if not text:
        return ""
    # Strip Latin accents (e.g. é -> e) while preserving Indic scripts
    cleaned = strip_accents_latin(text.strip())
    # Replace ampersands with 'and'
    cleaned = re.sub(r"&", " and ", cleaned)
    # Strip domain suffixes if present (e.g., .com, .co.in, .org, .net, .in, .us)
    cleaned = re.sub(r"\.(com|org|net|co\.in|in|biz|info|us|fr|edu|gov)\b", "", cleaned, flags=re.IGNORECASE)
    # Remove leading decorative noise like '-- ', '<< ', '** '
    cleaned = LEADING_DECOR_RE.sub("", cleaned)
    cleaned = TRAILING_DECOR_RE.sub("", cleaned)
    # Normalize internal whitespace
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned


def normalize_business_name(name: str) -> Tuple[str, str, List[str]]:
    """Normalize business name into multiple representations.

    Returns:
        (raw_clean, stripped_legal, token_list)
    """
    raw_clean = clean_text_base(name)
    if not raw_clean:
        return "", "", []

    # Strip legal entity suffixes
    stripped_legal = ALL_LEGAL_SUFFIXES_RE.sub("", raw_clean)
    # Clean whitespace and trailing punctuation after legal suffix removal
    stripped_legal = WHITESPACE_RE.sub(" ", stripped_legal).strip(" ,.-/")

    # Generate alphanumeric tokens (preserving all unicode letters and digits)
    tokens = re.findall(r"[\w\u0900-\u0D7F]+", stripped_legal if stripped_legal else raw_clean)

    return raw_clean, stripped_legal, tokens


def normalize_address(address: str, country: Optional[str] = None) -> Tuple[str, List[str], Set[str]]:
    """Normalize business address into multiple representations.

    Returns:
        (normalized_address_str, token_list, numeric_tokens_set)
    """
    cleaned = clean_text_base(address)
    if not cleaned:
        return "", [], set()

    # Extract all numeric tokens (house numbers, PIN codes, phone numbers)
    numeric_tokens = set(re.findall(r"\b\d+\b", cleaned))

    # Tokenize words
    raw_tokens = re.findall(r"[\w\u0900-\u0D7F]+", cleaned)
    normalized_tokens = []

    for t in raw_tokens:
        # Expand street abbreviations
        if t in STREET_ABBREVIATIONS:
            normalized_tokens.append(STREET_ABBREVIATIONS[t])
        elif country == "US" and t in US_STATES:
            normalized_tokens.append(US_STATES[t])
        elif country == "India" and t in INDIA_STATES:
            normalized_tokens.append(INDIA_STATES[t])
        else:
            normalized_tokens.append(t)

    normalized_str = " ".join(normalized_tokens)
    return normalized_str, normalized_tokens, numeric_tokens


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text:
        return set()
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}
