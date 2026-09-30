"""
English document/query analysis pipeline.

Thesis reference: Section 4.2.2.3 "English document analysis": sentence
detection, tokenization, normalization (lower-casing), stop word removal
and stemming, originally implemented via Apache OpenNLP (sentence
detector, NER) and the Apache Solr English analyzer (tokenizer, stop
filter, Porter stemmer). This module re-implements each step with plain
Python + the bundled Porter stemmer, so no external NLP server is needed.
"""
import re

from . import porter_stemmer

# A standard "Solr-style" English stop word list (Section 4.2.2.3 states
# "We adopted Apache Solr stop filter and word list").
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with", "was", "were", "been", "being", "have",
    "has", "had", "do", "does", "did", "can", "could", "would", "should",
    "so", "than", "too", "very", "s", "t", "just", "don", "now",
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?")


def sentence_split(text):
    text = text.strip()
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text):
    return _TOKEN_RE.findall(text)


def normalize(tokens):
    """Lower-case normalization (Section 4.2.2.3: normalization sub-process)."""
    return [t.lower() for t in tokens]


def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in STOPWORDS]


def stem_tokens(tokens):
    return porter_stemmer.stem_tokens(tokens)


def analyze(text, keep_stopwords_for_ner=False):
    """
    Full English document/query analysis pipeline, mirroring Algorithm 1's
    'English analyzer' call: sentence detection -> tokenization ->
    normalization -> stop word removal -> stemming.
    """
    sentences = sentence_split(text)
    raw_tokens = tokenize(text)
    normalized_tokens = normalize(raw_tokens)
    tokens_no_stop = normalized_tokens if keep_stopwords_for_ner else remove_stopwords(normalized_tokens)
    stemmed = stem_tokens(tokens_no_stop)
    return {
        "sentences": sentences,
        "raw_tokens": raw_tokens,
        "tokens": tokens_no_stop,
        "stemmed_tokens": stemmed,
    }
