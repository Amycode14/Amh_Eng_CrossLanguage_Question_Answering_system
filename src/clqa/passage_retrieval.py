"""
Passage Retrieval component.

Thesis reference: Section 4.2.5 "Passage Retrieval": "the passage
Retrieval component is responsible to rank a passage based on term
frequency and looking for named entity of answer ... A passage that
contains both query term and expected answer type is selected as the
most relevant passage ... if the expected answer type of question is
'person' then a passage that does not contain [a] named entity of
'person' in its content [is] discarded."

The thesis retrieves at the Apache Solr search-engine level; here
"passage" = one sentence of an indexed document (the natural passage
granularity for a small demo corpus), scored by stemmed term overlap and
filtered by the presence of a same-typed named entity, exactly as
described above.
"""
from . import amharic_nlp, english_nlp


def _sentence_stemmed_tokens(sentence, lang):
    if lang == "am":
        tokens = amharic_nlp.tokenize(sentence)
        tokens = amharic_nlp.remove_stopwords(tokens)
        return amharic_nlp.stem_tokens(tokens)
    tokens = english_nlp.tokenize(sentence)
    tokens = english_nlp.remove_stopwords(english_nlp.normalize(tokens))
    return english_nlp.stem_tokens(tokens)


def _sentence_raw_tokens(sentence, lang):
    if lang == "am":
        return amharic_nlp.tokenize(sentence)
    return english_nlp.tokenize(sentence)


class PassageRetriever:
    def __init__(self, indexer, ner_en, ner_am):
        self.indexer = indexer
        self.ner_en = ner_en
        self.ner_am = ner_am

    def _ner_for(self, lang):
        return self.ner_en if lang == "en" else self.ner_am

    def retrieve(self, query, top_k=5):
        """query: dict from QueryGenerator.generate() -- {lang, terms,
        answer_type, window_size}. Returns ranked passages, each carrying
        its own extracted entities (needed later by answer extraction)."""
        lang = query["lang"]
        terms = set(query["terms"])
        answer_type = query["answer_type"]
        ner = self._ner_for(lang)

        candidates = []
        for doc_id, doc in self.indexer.documents_in_lang(lang).items():
            for sent in doc["sentences"]:
                stemmed = _sentence_stemmed_tokens(sent, lang)
                overlap = sum(1 for t in stemmed if t in terms)
                if overlap == 0:
                    continue
                raw_tokens = _sentence_raw_tokens(sent, lang)
                entities = ner.extract_entities(raw_tokens) if ner else []
                has_expected_type = any(e["label"] == answer_type for e in entities)
                if not has_expected_type:
                    # "a passage that does not contain [the] named entity
                    # of [expected] answer type ... [is] discarded"
                    continue
                candidates.append({
                    "doc_id": doc_id,
                    "sentence": sent,
                    "raw_tokens": raw_tokens,
                    "score": overlap,
                    "entities": entities,
                })

        candidates.sort(key=lambda c: -c["score"])
        return candidates[:top_k]
