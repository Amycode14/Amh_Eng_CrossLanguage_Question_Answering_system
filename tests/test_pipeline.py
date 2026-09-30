import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clqa import amharic_nlp, english_nlp, porter_stemmer  # noqa: E402
from clqa.language_id import LanguageIdentifier  # noqa: E402
from clqa.ner_english import EnglishNER  # noqa: E402
from clqa.ner_amharic import AmharicNER  # noqa: E402
from clqa.translator import Translator  # noqa: E402
from clqa.question_analysis import QuestionAnalyzer  # noqa: E402
from clqa.indexer import Crawler, Indexer  # noqa: E402
from clqa.passage_retrieval import PassageRetriever  # noqa: E402
from clqa.answer_extraction import AnswerExtractor, build_doc_freq_map  # noqa: E402


def test_porter_stemmer_basic():
    assert porter_stemmer.stem("running") == "run"
    assert porter_stemmer.stem("founded") == "found"
    assert porter_stemmer.stem("universities") == "univers"


def test_english_analyze_pipeline():
    result = english_nlp.analyze("Addis Ababa is the capital city of Ethiopia.")
    assert "addis" in [t.lower() for t in result["tokens"]] or "Addis" in result["raw_tokens"]
    assert len(result["sentences"]) == 1


def test_amharic_normalize_and_tokenize():
    text = "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"
    tokens = amharic_nlp.tokenize(amharic_nlp.normalize(text))
    assert "አዲስ" in tokens
    assert "አበባ" in tokens


def test_amharic_stopword_removal():
    tokens = ["አዲስ", "የ", "አበባ"]
    filtered = amharic_nlp.remove_stopwords(tokens)
    assert "የ" not in filtered


def test_language_identifier_distinguishes_scripts():
    lang_id = LanguageIdentifier()
    lang_id.train("en", ["This is an English sentence about Ethiopia and its cities."])
    lang_id.train("am", ["ይህ ስለ ኢትዮጵያ እና ከተሞቿ የሚናገር የአማርኛ ዓረፍተ ነገር ነው።"])
    assert lang_id.identify("This is a test sentence in English.")[0] == "en"
    assert lang_id.identify("ይህ የአማርኛ ዓረፍተ ነገር ነው።")[0] == "am"


def test_english_ner_finds_person_location_date():
    ner = EnglishNER()
    tokens = english_nlp.tokenize("Meles Zenawi was the Prime Minister of Ethiopia from 1995 to 2012.")
    entities = ner.extract_entities(tokens)
    labels = {e["label"] for e in entities}
    assert "PERSON" in labels
    assert "LOCATION" in labels
    assert "DATE" in labels


def test_amharic_ner_finds_person_and_date():
    ner = AmharicNER()
    tokens = amharic_nlp.tokenize(amharic_nlp.normalize(
        "መለስ ዜናዊ ከ1995 እስከ 2012 የኢትዮጵያ ጠቅላይ ሚኒስትር ነበሩ።"))
    entities = ner.extract_entities(tokens)
    labels = {e["label"] for e in entities}
    assert "PERSON" in labels
    assert "DATE" in labels


def test_translator_transfers_key_content_words():
    tr = Translator()
    out = tr.translate("What is the capital city of Ethiopia?", "en", "am")
    # exact grammar isn't expected to be perfect (IBM Model 1, tiny corpus)
    # but core content words must transfer
    assert "ኢትዮጵያ" in out or "ከተማ" in out


def test_question_analysis_answer_type():
    qa_en = QuestionAnalyzer("en")
    result = qa_en.analyze("Who founded Addis Ababa?")
    assert result["answer_type"] == "PERSON"

    qa_am = QuestionAnalyzer("am")
    result_am = qa_am.analyze("የኢትዮጵያ ዋና ከተማ ምንድን ናት?")
    assert result_am["answer_type"] == "LOCATION"


def test_indexer_and_retrieval_end_to_end():
    ner_en = EnglishNER()
    ner_am = AmharicNER()
    indexer = Indexer(ner_en=ner_en, ner_am=ner_am)
    indexer.index_corpus(Crawler().crawl_local_corpus())
    assert indexer.num_docs("en") == 5
    assert indexer.num_docs("am") == 5

    qa_en = QuestionAnalyzer("en")
    analysis = qa_en.analyze("What is the capital city of Ethiopia?")
    retriever = PassageRetriever(indexer, ner_en, ner_am)
    candidates = retriever.retrieve(analysis["query"])
    assert len(candidates) > 0

    extractor = AnswerExtractor(lambda w: english_nlp.stem_tokens([w])[0])
    doc_freq = build_doc_freq_map(indexer, "en")
    extracted = extractor.extract(candidates[0], analysis["query"], doc_freq, indexer.num_docs("en"))
    assert extracted is not None
    assert extracted["answer"] == "Addis Ababa"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
