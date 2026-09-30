"""
Amharic Named Entity Recognition.

Thesis reference: Section 4.2.2.3, "Amharic named entity recognition ...
We prepared a training data set or annotated document for training
classifier model. Once annotating documents is prepared in the specified
format, we generate a model for each class of named-entity (person,
location, organization, time)."

This module builds weakly-labeled training sentences from the bundled
gazetteer (data/gazetteer_am.json) matched against the bundled corpus
(data/corpus_am/), then trains a maximum-entropy tagger (see
maxent_ner.MaxEntTagger) with contextual features -- the same
"one model per named-entity class" idea, here implemented as a single
multi-class MaxEnt model (person/location/organization/date/O) since a
multinomial MaxEnt naturally handles multiple classes at once.
"""
import glob
import json
import os
import re

from . import amharic_nlp
from .maxent_ner import MaxEntTagger, merge_bio_spans

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_GAZETTEER_PATH = os.path.join(_DATA_DIR, "gazetteer_am.json")
_CORPUS_DIR = os.path.join(_DATA_DIR, "corpus_am")

_DATE_RE = re.compile(r"^(1[89]\d{2}|20\d{2})$")


def load_gazetteer(path=None):
    path = path or _GAZETTEER_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Amharic prepositions/conjunctions cliticize onto the following word with
# no space (e.g. "የኢትዮጵያ" = "የ" + "ኢትዮጵያ"). A pure exact-match gazetteer
# lookup misses every cliticized occurrence of a location/person/org name,
# which would be most of them in running text. We therefore also match a
# single-token gazetteer entry against the *suffix* of a longer token when
# the leftover prefix is a known clitic -- a lightweight stand-in for real
# morphological segmentation (which is its own research problem the thesis
# does not solve either; see the stemmer's prefix list in amharic_nlp.py).
_CLITICS = ["እንደ", "በአ", "ስለ", "ከአ", "የ", "በ", "ለ", "ከ", "እ", "አ", "ን"]


def _clitic_suffix_match(token, single_token_entries):
    for entry in single_token_entries:
        if token == entry:
            return entry, ""
        for clitic in _CLITICS:
            if token == clitic + entry:
                return entry, clitic
    return None, None


def _weak_label_sentence(tokens, gazetteer):
    """BIO-style (but flat, same-label-run) weak labeling of one tokenized
    sentence using longest-match gazetteer lookup (with clitic-aware
    fallback for single-token entries), plus a regex rule for the DATE
    class (years), matching Table 4.1's 'Time' answer category.
    """
    labels = ["O"] * len(tokens)
    # sort phrases by token length, longest first, to prefer longest match
    phrases = []
    single_token = {}
    for label, entries in gazetteer.items():
        for phrase in entries:
            ptoks = amharic_nlp.tokenize(phrase)
            if not ptoks:
                continue
            if len(ptoks) == 1:
                single_token.setdefault(label, []).append(ptoks[0])
            phrases.append((ptoks, label))
    phrases.sort(key=lambda x: -len(x[0]))

    i = 0
    n = len(tokens)
    while i < n:
        matched_len = 0
        for ptoks, label in phrases:
            L = len(ptoks)
            if tokens[i:i + L] == ptoks:
                for k in range(L):
                    labels[i + k] = label
                matched_len = L
                break
        if matched_len == 0:
            for label, entries in single_token.items():
                entry, clitic = _clitic_suffix_match(tokens[i], entries)
                if entry is not None:
                    labels[i] = label
                    matched_len = 1
                    break
        if matched_len == 0 and _DATE_RE.match(tokens[i]):
            labels[i] = "DATE"
        i += max(matched_len, 1)
    return labels


def build_training_sentences(gazetteer=None, corpus_dir=None):
    gazetteer = gazetteer or load_gazetteer()
    corpus_dir = corpus_dir or _CORPUS_DIR
    tagged_sentences = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for sent in amharic_nlp.sentence_split(amharic_nlp.normalize(text)):
            tokens = amharic_nlp.tokenize(sent)
            if not tokens:
                continue
            labels = _weak_label_sentence(tokens, gazetteer)
            tagged_sentences.append(list(zip(tokens, labels)))
    return tagged_sentences


_flat_gazetteer_cache = None


def _flat_gazetteer():
    global _flat_gazetteer_cache
    if _flat_gazetteer_cache is None:
        gaz = load_gazetteer()
        flat = {}
        for label, entries in gaz.items():
            for phrase in entries:
                for tok in amharic_nlp.tokenize(phrase):
                    flat[tok] = label
        _flat_gazetteer_cache = flat
    return _flat_gazetteer_cache


def _gazetteer_hint(tok):
    """Exact or clitic-stripped gazetteer lookup, used as a classifier
    feature so the trained model generalizes to unseen sentences (not just
    the weakly-labeled training corpus)."""
    flat = _flat_gazetteer()
    if tok in flat:
        return flat[tok]
    for clitic in _CLITICS:
        if tok.startswith(clitic) and tok[len(clitic):] in flat:
            return flat[tok[len(clitic):]]
    return "NONE"


def _features(tokens, i):
    tok = tokens[i]
    feats = {
        "word": tok,
        "is_date_like": bool(_DATE_RE.match(tok)),
        "prev_word": tokens[i - 1] if i > 0 else "<START>",
        "next_word": tokens[i + 1] if i < len(tokens) - 1 else "<END>",
        "suffix3": tok[-3:] if len(tok) >= 3 else tok,
        "gazetteer_hint": _gazetteer_hint(tok),
        "len_bucket": "short" if len(tok) <= 3 else ("mid" if len(tok) <= 6 else "long"),
        "position_ratio_bucket": round(i / max(len(tokens) - 1, 1), 1),
    }
    return feats


class AmharicNER:
    """Trains once on the bundled gazetteer + corpus; tags new sentences."""

    def __init__(self):
        self.gazetteer = load_gazetteer()
        self.tagger = MaxEntTagger(_features)
        training = build_training_sentences(self.gazetteer)
        self.tagger.train(training)
        self.num_training_sentences = len(training)

    def tag_tokens(self, tokens):
        return self.tagger.tag(tokens)

    def extract_entities(self, tokens):
        """Hybrid tagging: the MaxEnt classifier's predictions are
        overlaid with exact (clitic-aware) gazetteer matches, which take
        priority where they fire. Combining a gazetteer with a
        statistically trained tagger is standard practice precisely
        because a gazetteer trained/curated from a small corpus (as here)
        will not generalize as well as a production-scale annotated
        corpus would -- the classifier fills the gaps the gazetteer can't
        cover (unseen names, dates by pattern, etc.)."""
        labels = list(self.tag_tokens(tokens))
        gaz_labels = _weak_label_sentence(tokens, self.gazetteer)
        for i, gl in enumerate(gaz_labels):
            if gl != "O":
                labels[i] = gl
        return merge_bio_spans(tokens, labels)

    def has_entity_of_type(self, tokens, answer_type):
        return any(e["label"] == answer_type for e in self.extract_entities(tokens))
