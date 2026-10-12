"""Text normalization and tokenization for business name / address matching.

No external lookups, no geocoding, no transliteration services — everything here is a
static, local, deterministic string transform (allowed under the challenge's fair-play
rules, which only ban external data augmentation / API lookups of business identity).
"""
import re

LEGAL_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "llc", "llp", "lp", "pvt", "private", "pc", "plc",
    "group", "holdings", "enterprises", "enterprise", "industries", "industry",
    "the", "and", "of", "for", "a", "an",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")


def normalize_text(s) -> str:
    if s is None or not isinstance(s, str):
        return ""
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s.lower())
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s, drop_legal_stopwords: bool = False, min_len: int = 2):
    norm = normalize_text(s)
    if not norm:
        return []
    toks = norm.split(" ")
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if drop_legal_stopwords and t in LEGAL_STOPWORDS:
            continue
        out.append(t)
    return out


def numeric_tokens(s, min_digits: int = 2):
    if s is None or not isinstance(s, str):
        return []
    return [n for n in _NUM_RE.findall(s) if len(n) >= min_digits]


def non_ascii_ratio(s) -> float:
    if not s or not isinstance(s, str):
        return 0.0
    non_ascii = sum(1 for ch in s if ord(ch) > 127)
    return non_ascii / max(len(s), 1)
