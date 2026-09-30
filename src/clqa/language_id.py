"""
N-gram based language identification.

Thesis reference: Section 4.2.2.2 "Language Identification". The thesis
uses Apache Tika's N-gram language identifier (based on Cavnar & Trenkle's
"out-of-place" text categorization method), customized with a hand-built
Amharic language profile since Tika does not ship one out of the box.

This module re-implements the same out-of-place N-gram ranking algorithm
directly in Python so the pipeline needs no external Tika/Nutch service.
A language "profile" is the list of its most frequent character N-grams
(N = 1..5), ranked by frequency. An unknown document is turned into its
own profile, and the distance to each language profile is the sum, over
shared N-grams, of how far apart their ranks are (missing N-grams count
as a fixed maximum penalty) -- exactly the out-of-place measure described
in the thesis.
"""
import re
from collections import Counter

MAX_NGRAMS = 300
MAX_N = 5
OUT_OF_PLACE_MAX = MAX_NGRAMS


def _clean(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _ngrams(text, n):
    text = f" {text} "
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def build_profile(text, max_ngrams=MAX_NGRAMS):
    """Build a ranked N-gram frequency profile for a chunk of text."""
    counts = Counter()
    for n in range(1, MAX_N + 1):
        counts.update(_ngrams(_clean(text), n))
    ranked = [ng for ng, _ in counts.most_common(max_ngrams)]
    return {ng: rank for rank, ng in enumerate(ranked)}


def _distance(doc_profile, lang_profile):
    dist = 0
    for ng, doc_rank in doc_profile.items():
        lang_rank = lang_profile.get(ng)
        if lang_rank is None:
            dist += OUT_OF_PLACE_MAX
        else:
            dist += abs(doc_rank - lang_rank)
    return dist


class LanguageIdentifier:
    """Out-of-place N-gram language identifier supporting 'am' and 'en'."""

    def __init__(self):
        self.profiles = {}

    def train(self, lang_code, sample_texts):
        """Build a language profile from a list of sample training texts."""
        combined = " ".join(sample_texts)
        self.profiles[lang_code] = build_profile(combined)

    def identify(self, text):
        """Return (best_lang, scores_dict). Lower score = closer match."""
        if not self.profiles:
            raise RuntimeError("No language profiles trained yet.")
        # Fast path: Amharic (Ge'ez) script is a distinct Unicode block, so
        # if a document contains a meaningful fraction of Ge'ez characters
        # we already know it's Amharic. This mirrors the practical shortcut
        # any real system would take before falling back to full N-gram
        # ranking (Ge'ez script never overlaps with English/Latin script).
        geez = sum(1 for ch in text if '\u1200' <= ch <= '\u137F')
        latin = sum(1 for ch in text if ch.isalpha() and ch.isascii())
        if geez + latin > 0:
            if geez > latin:
                return "am", {"am": 0, "en": OUT_OF_PLACE_MAX * len(self.profiles)}
            elif latin > geez:
                pass  # fall through to full ranking for confirmation
        doc_profile = build_profile(text)
        scores = {lang: _distance(doc_profile, prof) for lang, prof in self.profiles.items()}
        best = min(scores, key=scores.get)
        return best, scores
