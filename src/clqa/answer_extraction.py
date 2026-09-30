"""
Answer Extraction component.

Thesis reference: Section 4.2.6 "Answer Extraction" and Algorithm 4
(pseudo code of answer extraction component):

  1. Identify & score candidate answer: find query-term match positions,
     build windows (main window + left/right neighbor windows) around
     each match, weight terms by their importance (inverse document
     frequency proxy here), give a bigram bonus, and combine into a
     window score.
  2. Select top-scoring answer: rank windows by score.
  3. Extract exact answer: from the top window, pick the substring whose
     named-entity type matches the expected answer type.

This module operates on the already NER-tagged passages returned by
passage_retrieval.PassageRetriever, so step 3 is a lookup against spans
already computed there.
"""
from collections import defaultdict

WINDOW_SIZE = 30  # Algorithm 4: "Windowsize: 30" (kept small here since
                   # our demo sentences are far shorter than 30 tokens --
                   # effectively this makes the whole sentence one window)


def _term_weight(term, doc_freq_map, total_docs):
    """Inverse-document-frequency-style term weight -- the thesis's
    "relative importance of terms in document" (Algorithm 4, step 1)."""
    df = doc_freq_map.get(term, 1)
    import math
    return math.log((total_docs + 1) / (df + 1)) + 1.0


def _windows_around(tokens, center, size):
    """Construct the main window plus left/right neighbor windows,
    matching Algorithm 4's comment: 'construct 5 windows around query
    term (main window, first previous window, first follow window,
    second previous window and second follow window)'."""
    n = len(tokens)
    half = max(size // 2, 3)
    windows = []
    offsets = [(0, "main"), (-1, "prev1"), (1, "next1"), (-2, "prev2"), (2, "next2")]
    for mult, name in offsets:
        start = max(0, center + mult * half - half // 2)
        end = min(n, start + half)
        start = max(0, end - half)
        if start >= end:
            continue
        windows.append({"name": name, "start": start, "end": end, "tokens": tokens[start:end]})
    return windows


def _has_bigram_match(window_tokens, query_terms, stemmer):
    stemmed = [stemmer(t) for t in window_tokens]
    for i in range(len(stemmed) - 1):
        if stemmed[i] in query_terms and stemmed[i + 1] in query_terms:
            return True
    return False


class AnswerExtractor:
    def __init__(self, stemmer_fn):
        """stemmer_fn: language-appropriate single-word stemmer, used to
        match query terms against passage tokens (so scoring works on the
        same normalized form the index uses)."""
        self.stemmer_fn = stemmer_fn

    def extract(self, passage, query, doc_freq_map=None, total_docs=1):
        """passage: one candidate from PassageRetriever.retrieve()
        (has raw_tokens, entities, score). query: dict with 'terms' and
        'answer_type'. Returns {answer, score, window} or None."""
        doc_freq_map = doc_freq_map or {}
        tokens = passage["raw_tokens"]
        query_terms = set(query["terms"])
        stemmed_tokens = [self.stemmer_fn(t) for t in tokens]

        match_positions = [i for i, st in enumerate(stemmed_tokens) if st in query_terms]
        if not match_positions:
            return None

        best_window, best_score = None, float("-inf")
        for center in match_positions:
            for window in _windows_around(tokens, center, WINDOW_SIZE):
                w_terms = [self.stemmer_fn(t) for t in window["tokens"]]
                weight_sum = sum(
                    _term_weight(t, doc_freq_map, total_docs)
                    for t in w_terms if t in query_terms
                )
                bigram_bonus = 0.5 if _has_bigram_match(window["tokens"], query_terms, self.stemmer_fn) else 0.0
                score = weight_sum + bigram_bonus
                if score > best_score:
                    best_score = score
                    best_window = window

        if best_window is None:
            return None

        # Step 3: extract exact answer -- the entity of the expected type
        # whose span falls inside (or overlaps) the winning window.
        answer_type = query["answer_type"]
        candidates_in_window = [
            e for e in passage["entities"]
            if e["label"] == answer_type and e["start"] < best_window["end"] and e["end"] > best_window["start"]
        ]
        if not candidates_in_window:
            # Fall back to any entity of the expected type anywhere in the
            # passage (keeps the system useful even on very short demo
            # sentences where the whole sentence is effectively one window).
            candidates_in_window = [e for e in passage["entities"] if e["label"] == answer_type]
        if not candidates_in_window:
            return None

        best_entity = min(
            candidates_in_window,
            key=lambda e: abs(((e["start"] + e["end"]) / 2) - ((best_window["start"] + best_window["end"]) / 2)),
        )
        return {
            "answer": best_entity["text"],
            "window": best_window,
            "score": best_score,
            "doc_id": passage["doc_id"],
            "sentence": passage["sentence"],
        }


def build_doc_freq_map(indexer, lang):
    """Reads term document-frequency straight from the indexer's inverted
    index, for use as the IDF-style term weight in Algorithm 4."""
    return dict(indexer.doc_freq[lang])
