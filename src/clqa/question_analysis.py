"""
Question Processing components.

Thesis reference: Section 4.2.4 "Question Processing", with sub-sections
4.2.4.1 (Amharic) and 4.2.4.2 (English question processing), each made of
three sub-components: question parsing, answer type determination, and
query generation (Algorithms 2 and 3).

Question parsing here: a lightweight shallow parse that pulls out the
question word, the verb next to it, and the remaining content words as
the "noun phrase" -- the same four feature slots Table 4.1 lists
(question word, focus, support, verb) -- implemented with rules rather
than the thesis's Amharic-TreebankChunker / OpenNLP chunker (neither of
which is installable here without a model-download step this sandbox
cannot perform).

Answer type determination: a maximum-entropy (multinomial logistic
regression) classifier, trained on data/qa_training_en.tsv and
data/qa_training_am.tsv, using a bag-of-words feature representation of
the question -- i.e. exactly the "classifier uses the words found in the
question as features" design the thesis states in Section 4.2.4.1(2).

Query generation: builds the query object consumed by passage retrieval
-- content terms (stemmed) + expected answer type + a window size --
which stands in for the thesis's Lucene SpanNearQuery (Section 4.2.4.1(3)).
"""
import csv
import os

from . import amharic_nlp, english_nlp
from .maxent_ner import MaxEntClassifier

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

_AM_QUESTION_WORDS = ["ማን", "የት", "መቼ", "ምንድን", "ምን", "ስንት"]
_EN_QUESTION_WORDS = ["who", "where", "when", "what", "which", "how"]

ANSWER_TYPES = ["PERSON", "LOCATION", "DATE", "ORGANIZATION", "QUANTITY"]

DEFAULT_WINDOW_SIZE = 30  # matches Algorithm 4's "Windowsize: 30"


def _load_training_tsv(path):
    questions, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            questions.append(row["question"])
            labels.append(row["answer_type"])
    return questions, labels


def _bow_features(tokens):
    """Bag-of-words feature dict -- 'the classifier uses the words found
    in the question as features' (Section 4.2.4.1(2))."""
    return {f"has:{t}": True for t in tokens}


class QuestionParser:
    """Shallow question parse: question word / verb / support (noun
    phrase) extraction, mirroring Table 4.1 and Algorithm 3."""

    def __init__(self, lang):
        self.lang = lang
        self.qwords = _AM_QUESTION_WORDS if lang == "am" else _EN_QUESTION_WORDS

    def parse(self, question_text):
        if self.lang == "am":
            tokens = amharic_nlp.tokenize(amharic_nlp.normalize(question_text))
        else:
            tokens = [t.lower() for t in english_nlp.tokenize(question_text)]

        qword, qword_idx = None, None
        for i, tok in enumerate(tokens):
            if tok in self.qwords:
                qword, qword_idx = tok, i
                break

        # "verb next to question word" (Algorithm 3): take the token
        # immediately after the question word as a naive verb proxy.
        verb = None
        if qword_idx is not None and qword_idx + 1 < len(tokens):
            verb = tokens[qword_idx + 1]

        support_tokens = [t for i, t in enumerate(tokens) if i != qword_idx and t != verb]
        return {
            "tokens": tokens,
            "question_word": qword,
            "verb": verb,
            "support_tokens": support_tokens,
        }


class AnswerTypeClassifier:
    """MaxEnt (multinomial logistic regression) answer-type classifier,
    one instance per language, trained on the bundled labeled question
    sets (Section 4.2.4.1(2) / 4.2.4.2)."""

    def __init__(self, lang):
        self.lang = lang
        path = os.path.join(_DATA_DIR, f"qa_training_{lang}.tsv")
        questions, labels = _load_training_tsv(path)
        self.classifier = MaxEntClassifier(self._featurize)
        self.classifier.train(questions, labels)

    def _featurize(self, question_text):
        if self.lang == "am":
            tokens = amharic_nlp.tokenize(amharic_nlp.normalize(question_text))
        else:
            tokens = [t.lower() for t in english_nlp.tokenize(question_text)]
        return _bow_features(tokens)

    def predict(self, question_text):
        return self.classifier.predict(question_text)

    def predict_proba(self, question_text):
        return self.classifier.predict_proba_dict(question_text)


class QueryGenerator:
    """Builds a SpanNearQuery-equivalent: stemmed content terms + expected
    answer type + window size (Section 4.2.4.1(3))."""

    def __init__(self, lang):
        self.lang = lang

    def generate(self, parsed_question, answer_type):
        content_tokens = parsed_question["support_tokens"]
        if self.lang == "am":
            content_tokens = amharic_nlp.remove_stopwords(content_tokens)
            stemmed = amharic_nlp.stem_tokens(content_tokens)
        else:
            content_tokens = english_nlp.remove_stopwords(content_tokens)
            stemmed = english_nlp.stem_tokens(content_tokens)
        return {
            "lang": self.lang,
            "terms": stemmed,
            "raw_terms": content_tokens,
            "answer_type": answer_type,
            "window_size": DEFAULT_WINDOW_SIZE,
        }


class QuestionAnalyzer:
    """Ties parsing + answer-type determination + query generation
    together for one language (Algorithm 2)."""

    def __init__(self, lang):
        self.lang = lang
        self.parser = QuestionParser(lang)
        self.type_classifier = AnswerTypeClassifier(lang)
        self.query_gen = QueryGenerator(lang)

    def analyze(self, question_text):
        parsed = self.parser.parse(question_text)
        answer_type = self.type_classifier.predict(question_text)
        query = self.query_gen.generate(parsed, answer_type)
        return {
            "question": question_text,
            "lang": self.lang,
            "parsed": parsed,
            "answer_type": answer_type,
            "query": query,
        }
