"""
English Named Entity Recognition.

Thesis reference: Section 4.2.2.3, "English named entity recognition ... We
have used an Apache OpenNLP tools for sentence detection and named-entity
recognition." OpenNLP's NER models are themselves maximum-entropy models
trained on annotated corpora, so this module mirrors ner_amharic.py: a
MaxEnt tagger trained on weakly-labeled sentences built from the bundled
English gazetteer + corpus, using capitalization as an additional feature
that is unavailable for the Ge'ez script.
"""
import glob
import json
import os
import re

from . import english_nlp
from .maxent_ner import MaxEntTagger, merge_bio_spans

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_GAZETTEER_PATH = os.path.join(_DATA_DIR, "gazetteer_en.json")
_CORPUS_DIR = os.path.join(_DATA_DIR, "corpus_en")

_DATE_RE = re.compile(r"^(1[89]\d{2}|20\d{2})$")


def load_gazetteer(path=None):
    path = path or _GAZETTEER_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _weak_label_sentence(tokens, gazetteer):
    labels = ["O"] * len(tokens)
    lowered = [t.lower() for t in tokens]
    phrases = []
    for label, entries in gazetteer.items():
        for phrase in entries:
            ptoks = [t.lower() for t in english_nlp.tokenize(phrase)]
            if ptoks:
                phrases.append((ptoks, label))
    phrases.sort(key=lambda x: -len(x[0]))

    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        for ptoks, label in phrases:
            L = len(ptoks)
            if lowered[i:i + L] == ptoks:
                for k in range(L):
                    labels[i + k] = label
                i += L
                matched = True
                break
        if not matched:
            if _DATE_RE.match(tokens[i]):
                labels[i] = "DATE"
            i += 1
    return labels


def build_training_sentences(gazetteer=None, corpus_dir=None):
    gazetteer = gazetteer or load_gazetteer()
    corpus_dir = corpus_dir or _CORPUS_DIR
    tagged_sentences = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for sent in english_nlp.sentence_split(text):
            tokens = english_nlp.tokenize(sent)
            if not tokens:
                continue
            labels = _weak_label_sentence(tokens, gazetteer)
            tagged_sentences.append(list(zip(tokens, labels)))
    return tagged_sentences


def _features(tokens, i):
    tok = tokens[i]
    feats = {
        "word_lower": tok.lower(),
        "is_capitalized": tok[:1].isupper(),
        "is_date_like": bool(_DATE_RE.match(tok)),
        "prev_word_lower": tokens[i - 1].lower() if i > 0 else "<START>",
        "next_word_lower": tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>",
        "prev_capitalized": tokens[i - 1][:1].isupper() if i > 0 else False,
        "suffix3": tok[-3:].lower() if len(tok) >= 3 else tok.lower(),
        "position_ratio_bucket": round(i / max(len(tokens) - 1, 1), 1),
    }
    return feats


class EnglishNER:
    def __init__(self):
        self.gazetteer = load_gazetteer()
        self.tagger = MaxEntTagger(_features)
        training = build_training_sentences(self.gazetteer)
        self.tagger.train(training)
        self.num_training_sentences = len(training)

    def tag_tokens(self, tokens):
        return self.tagger.tag(tokens)

    def extract_entities(self, tokens):
        """Hybrid tagging: overlay exact gazetteer matches on top of the
        MaxEnt classifier's predictions (see ner_amharic.AmharicNER for
        the rationale)."""
        labels = list(self.tag_tokens(tokens))
        gaz_labels = _weak_label_sentence(tokens, self.gazetteer)
        for i, gl in enumerate(gaz_labels):
            if gl != "O":
                labels[i] = gl
        return merge_bio_spans(tokens, labels)

    def has_entity_of_type(self, tokens, answer_type):
        return any(e["label"] == answer_type for e in self.extract_entities(tokens))
