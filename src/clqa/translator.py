"""
Statistical Machine Translation for question translation.

Thesis reference: Section 4.2.3 "Question Translation Component". The
thesis trains a full Moses phrase-based SMT system (translation model +
language model, word-aligned with GIZA++, decoded via the Moses decoder
server) on a bilingual corpus assembled from bible/history/constitution
text. That pipeline needs external binaries (Moses, GIZA++, KenLM) and a
large parallel corpus, neither of which is available in this sandbox.

This module keeps the same statistical approach -- a translation model
estimated from a bilingual corpus, no hand-written translation rules --
but implements it directly in Python:

  1. IBM Model 1 (Brown et al., 1993) trained by Expectation-Maximization
     on data/parallel_corpus.tsv. This is exactly the algorithm GIZA++
     itself runs as its *first* stage before more complex alignment
     models -- so this is a genuine (if simplified, single-stage) SMT
     word-alignment / lexical-translation model, not a dictionary.
  2. A trivial "language model" re-ordering step: since Amharic is
     subject-object-verb and English is subject-verb-object, and Amharic
     question words trail the clause while English question words lead
     it, a small set of question-word fronting rules is applied after
     word translation -- standing in for the target-language model +
     decoder search that Moses performs jointly with translation scoring.

The corpus is intentionally small (demo-scale, not web-scale), so
translation quality is naturally lower than production SMT -- this
mirrors the thesis's own finding (Chapter 5) that translation quality is
the main bottleneck of cross-lingual retrieval accuracy.
"""
import os
import re
from collections import defaultdict

from . import amharic_nlp, english_nlp

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_PARALLEL_CORPUS_PATH = os.path.join(_DATA_DIR, "parallel_corpus.tsv")

_AM_QUESTION_WORDS = {"ማን": "who", "የት": "where", "መቼ": "when", "ምንድን": "what", "ስንት": "how many"}
_EN_QUESTION_WORDS = {"who": "ማን", "where": "የት", "when": "መቼ", "what": "ምንድን", "how": "ስንት"}


def _tokenize(text, lang):
    if lang == "am":
        return amharic_nlp.tokenize(amharic_nlp.normalize(text))
    return [t.lower() for t in english_nlp.tokenize(text)]


def load_parallel_corpus(path=None):
    path = path or _PARALLEL_CORPUS_PATH
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            am, en = line.split("\t")
            pairs.append((am.strip(), en.strip()))
    return pairs


class IBMModel1:
    """IBM Model 1 word-translation model, trained by EM.

    p(target_word | source_word) is estimated so as to maximize the
    likelihood of the parallel corpus under a bag-of-words alignment
    model, exactly as described in Brown et al. (1993) and used inside
    GIZA++/Moses (Section 4.2.3's "word aligner tools").
    """

    NULL = "<NULL>"

    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.t = {}  # t[source_word][target_word] = probability

    def train(self, sentence_pairs, iterations=25):
        src_sents, tgt_sents = [], []
        for src_text, tgt_text in sentence_pairs:
            src = _tokenize(src_text, self.source_lang)
            tgt = _tokenize(tgt_text, self.target_lang)
            src_sents.append([self.NULL] + src)
            tgt_sents.append(tgt)

        vocab_src = set(w for s in src_sents for w in s)
        vocab_tgt = set(w for t in tgt_sents for w in t)
        uniform = 1.0 / len(vocab_tgt) if vocab_tgt else 0.0
        t = {s: defaultdict(lambda: uniform) for s in vocab_src}

        for _ in range(iterations):
            count = {s: defaultdict(float) for s in vocab_src}
            total_s = {s: 0.0 for s in vocab_src}
            for src, tgt in zip(src_sents, tgt_sents):
                for tw in tgt:
                    denom = sum(t[sw][tw] for sw in src)
                    if denom == 0:
                        continue
                    for sw in src:
                        delta = t[sw][tw] / denom
                        count[sw][tw] += delta
                        total_s[sw] += delta
            for sw in vocab_src:
                if total_s[sw] == 0:
                    continue
                for tw, c in count[sw].items():
                    t[sw][tw] = c / total_s[sw]

        self.t = {s: dict(d) for s, d in t.items()}
        self.vocab_tgt = vocab_tgt

    def best_translation(self, source_word, top_k=1):
        dist = self.t.get(source_word)
        if not dist:
            return []
        ranked = sorted(dist.items(), key=lambda x: -x[1])
        return ranked[:top_k]


def _apply_reordering(tokens, direction):
    """Front the question word, mirroring Amharic (SOV, question word near
    verb) -> English (SVO, question word first) reordering, and the
    reverse. This is a rule-based stand-in for the target-language model
    scoring step of a real decoder (Section 4.2.3)."""
    if direction == "am2en":
        qwords = set(_AM_QUESTION_WORDS.values()) | {"how many"}
    else:
        qwords = set(_EN_QUESTION_WORDS.values())
    for i, tok in enumerate(tokens):
        if tok in qwords and i != 0:
            return [tokens[i]] + tokens[:i] + tokens[i + 1:]
    return tokens


class Translator:
    """Bidirectional Amharic<->English SMT question translator."""

    def __init__(self, parallel_corpus_path=None):
        pairs = load_parallel_corpus(parallel_corpus_path)
        self.am2en_model = IBMModel1("am", "en")
        self.am2en_model.train(pairs)
        # reverse pairs for the opposite translation direction
        self.en2am_model = IBMModel1("en", "am")
        self.en2am_model.train([(en, am) for am, en in pairs])

    def translate(self, text, source_lang, target_lang):
        """Word-for-word SMT decoding: for each source token pick the
        highest-probability target word (argmax_t p(t|s)), matching a
        greedy Model-1 decoder, then apply question-word fronting."""
        if source_lang == target_lang:
            return text
        model = self.am2en_model if source_lang == "am" else self.en2am_model
        src_tokens = _tokenize(text, source_lang)
        out_tokens = []
        for sw in src_tokens:
            candidates = model.best_translation(sw, top_k=1)
            if candidates:
                out_tokens.append(candidates[0][0])
            else:
                out_tokens.append(sw)  # untranslatable (OOV) -> pass through
        direction = "am2en" if source_lang == "am" else "en2am"
        out_tokens = _apply_reordering(out_tokens, direction)
        return " ".join(out_tokens)
