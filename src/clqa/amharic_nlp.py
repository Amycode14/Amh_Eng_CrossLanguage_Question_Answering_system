"""
Amharic document/query analysis pipeline.

Thesis reference: Section 4.2.2.3 "Amharic document analysis":
tokenization, normalization, stop-word removal and stemming, adopted from
Tessema Mindaye & Solomon Atnafu (normalizer, stop word list) and
Alemayehu Nega & Willett (stemmer). This module re-implements light
versions of each step directly (no external service dependency).
"""
import os
import re

# Characters that are used interchangeably in Amharic (Section 4.2.2.3),
# each mapped to one canonical representative character.
_NORMALIZE_MAP = {
    "ሀ": "ሀ", "ኀ": "ሀ", "ሐ": "ሀ", "ኻ": "ሀ",
    "ሰ": "ሰ", "ሠ": "ሰ",
    "ዐ": "አ", "አ": "አ", "ኣ": "አ",
    "ጸ": "ፀ", "ፀ": "ፀ",
    "ሁ": "ሁ", "ኁ": "ሁ", "ሑ": "ሁ",
    "ሂ": "ሂ", "ኂ": "ሂ", "ሒ": "ሂ",
    "ሄ": "ሄ", "ኄ": "ሄ", "ሔ": "ሄ",
    "ህ": "ህ", "ኅ": "ህ", "ሕ": "ህ",
    "ሆ": "ሆ", "ኆ": "ሆ", "ሖ": "ሆ",
}

# Common Amharic noun/verb suffixes and prefixes, ordered longest-first,
# approximating the affix list used by the Alemayehu Nega & Willett
# stemming algorithm referenced in the thesis.
_SUFFIXES = [
    "ዎቻቸው", "ዎቻችን", "ዎቻችሁ", "ኣቸው", "ኦች", "ዎች", "አቸው", "ችው",
    "ኣት", "ኡት", "ኦት", "ኣል", "ኧል", "ኡ", "ኦ", "ኣ", "ች", "ት", "ን", "ም",
]
_PREFIXES = ["እንደ", "በአ", "ስለ", "ከአ", "የ", "በ", "ለ", "ከ", "እ", "አ", "ን"]

_STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "stopwords_am.txt")


def load_stopwords(path=None):
    path = path or _STOPWORDS_PATH
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


_STOPWORDS = None


def get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = load_stopwords()
    return _STOPWORDS


def normalize(text):
    """Map interchangeable Amharic characters to a canonical form."""
    return "".join(_NORMALIZE_MAP.get(ch, ch) for ch in text)


_TOKEN_RE = re.compile(r"[\u1200-\u137F]+|[0-9]+(?:\.[0-9]+)?|[A-Za-z]+")


def tokenize(text):
    """Split Amharic (and embedded Latin/number) text into tokens."""
    return _TOKEN_RE.findall(text)


def sentence_split(text):
    """Split on Amharic full stop (። ), question mark and exclamation."""
    text = text.strip()
    parts = re.split(r"(?<=[።፧፨!?])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def remove_stopwords(tokens):
    sw = get_stopwords()
    return [t for t in tokens if t not in sw]


def stem_word(word):
    """Light suffix/prefix stripping stemmer for a single Amharic word."""
    if len(word) <= 3 or not re.match(r"^[\u1200-\u137F]+$", word):
        return word
    w = word
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            w = w[: -len(suf)]
            break
    for pre in _PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= 2:
            w = w[len(pre):]
            break
    return w if w else word


def stem_tokens(tokens):
    return [stem_word(t) for t in tokens]


def analyze(text, keep_stopwords_for_ner=False):
    """
    Full Amharic document/query analysis pipeline, mirroring Algorithm 1's
    'Amharic analyzer' call: normalization -> tokenization -> stop word
    removal -> stemming. Returns a dict with the intermediate artifacts
    (sentences, raw tokens, normalized/stemmed tokens) since later stages
    (NER, indexing) need different views of the same text.
    """
    normalized = normalize(text)
    sentences = sentence_split(normalized)
    raw_tokens = tokenize(normalized)
    tokens_no_stop = raw_tokens if keep_stopwords_for_ner else remove_stopwords(raw_tokens)
    stemmed = stem_tokens(tokens_no_stop)
    return {
        "normalized_text": normalized,
        "sentences": sentences,
        "raw_tokens": raw_tokens,
        "tokens": tokens_no_stop,
        "stemmed_tokens": stemmed,
    }
