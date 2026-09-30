"""
Web Crawler + Indexer components.

Thesis reference:
  * Section 4.2.1 "Web Crawler" -- Nutch-based crawler seeded from
    bbc.com, zehabesha.com, mereja.com, ethiopianreporter.com.
  * Section 4.2.2 "Indexer Component" -- Apache Tika (text parsing),
    N-gram language identification, per-language document analysis, and
    Apache Lucene indexing with term-position information (Algorithm 1).

`Crawler` here is a light stand-in: this sandbox has no outbound network
access to arbitrary websites (see the tool's allowed-domain list), so
instead of a Nutch link-follower we load documents from a bundled local
corpus (data/corpus_en/, data/corpus_am/) that plays the role of "already
crawled and downloaded" documents. `fetch_url()` is also provided for
users who *do* have network access and want to point the crawler at real
pages -- it does the same Tika-style job (HTML -> plain text) using
Python's standard library + a simple tag stripper.

`Indexer` builds an in-memory inverted index with per-document term
positions -- the same information a Lucene `TermVector` stores -- so the
answer-extraction stage (Section 4.2.6) can do position-based window
search without needing an actual Lucene/Solr server.
"""
import glob
import os
import re
from collections import defaultdict

from . import amharic_nlp, english_nlp
from .language_id import LanguageIdentifier

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_CORPUS_EN_DIR = os.path.join(_DATA_DIR, "corpus_en")
_CORPUS_AM_DIR = os.path.join(_DATA_DIR, "corpus_am")

_TAG_RE = re.compile(r"<[^>]+>")


class Crawler:
    """Stand-in for the thesis's Nutch-based web crawler (Section 4.2.1)."""

    def __init__(self, corpus_en_dir=None, corpus_am_dir=None):
        self.corpus_en_dir = corpus_en_dir or _CORPUS_EN_DIR
        self.corpus_am_dir = corpus_am_dir or _CORPUS_AM_DIR

    def crawl_local_corpus(self):
        """Load the bundled 'already crawled' documents (demo data)."""
        docs = []
        for lang, folder in (("en", self.corpus_en_dir), ("am", self.corpus_am_dir)):
            for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                doc_id = f"{lang}:{os.path.basename(path)}"
                with open(path, encoding="utf-8") as f:
                    text = f.read().strip()
                docs.append({"doc_id": doc_id, "text": text, "known_lang": lang, "source": path})
        return docs

    @staticmethod
    def fetch_url(url, timeout=10):
        """Fetch + strip HTML tags from a live URL (Apache Tika stand-in).
        Only usable where outbound network access to the target host is
        permitted -- this sandbox is restricted to package registries, so
        this method is provided for completeness / external use rather
        than exercised by the bundled demo."""
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        text = _TAG_RE.sub(" ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class Indexer:
    """In-memory inverted index with term positions, replicating what
    Algorithm 1 (pseudo code of indexer component) describes: language
    identification -> per-language document analysis -> index."""

    def __init__(self, ner_en=None, ner_am=None):
        self.lang_id = LanguageIdentifier()
        self.documents = {}       # doc_id -> {lang, text, sentences, entities}
        # inverted_index[lang][stemmed_term] -> {doc_id: [positions]}
        self.inverted_index = {"en": defaultdict(lambda: defaultdict(list)),
                                "am": defaultdict(lambda: defaultdict(list))}
        self.doc_freq = {"en": defaultdict(int), "am": defaultdict(int)}
        self.ner_en = ner_en
        self.ner_am = ner_am
        self._train_language_identifier()

    def _train_language_identifier(self):
        crawler = Crawler()
        docs = crawler.crawl_local_corpus()
        en_texts = [d["text"] for d in docs if d["known_lang"] == "en"]
        am_texts = [d["text"] for d in docs if d["known_lang"] == "am"]
        self.lang_id.train("en", en_texts)
        self.lang_id.train("am", am_texts)

    def add_document(self, doc_id, text, known_lang=None):
        """Algorithm 1: identify language -> run the matching analyzer ->
        add to the index with term positions."""
        lang = known_lang or self.lang_id.identify(text)[0]
        if lang == "en":
            analysis = english_nlp.analyze(text)
            raw_tokens_for_ner = english_nlp.tokenize(text)
            entities = self.ner_en.extract_entities(raw_tokens_for_ner) if self.ner_en else []
        else:
            analysis = amharic_nlp.analyze(text)
            raw_tokens_for_ner = amharic_nlp.tokenize(amharic_nlp.normalize(text))
            entities = self.ner_am.extract_entities(raw_tokens_for_ner) if self.ner_am else []

        self.documents[doc_id] = {
            "lang": lang,
            "text": text,
            "sentences": analysis["sentences"],
            "raw_tokens": raw_tokens_for_ner,
            "entities": entities,
        }

        # Index stemmed terms with their token positions (position-based
        # search / Lucene TermVector equivalent, Section 4.2.4.1(3)).
        for pos, term in enumerate(analysis["stemmed_tokens"]):
            self.inverted_index[lang][term][doc_id].append(pos)

        seen_terms = set(analysis["stemmed_tokens"])
        for term in seen_terms:
            self.doc_freq[lang][term] += 1

    def index_corpus(self, documents):
        for doc in documents:
            self.add_document(doc["doc_id"], doc["text"], known_lang=doc.get("known_lang"))

    def num_docs(self, lang):
        return sum(1 for d in self.documents.values() if d["lang"] == lang)

    def get_document(self, doc_id):
        return self.documents.get(doc_id)

    def documents_in_lang(self, lang):
        return {doc_id: d for doc_id, d in self.documents.items() if d["lang"] == lang}
