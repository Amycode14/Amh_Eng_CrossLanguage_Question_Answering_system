"""
End-to-end Amharic-English Cross Language Question Answering pipeline.

Thesis reference: Figure 4.1 "General Architecture of the Proposed
System" -- User Question -> Language Identification -> {Web Crawler ->
Indexer} in parallel, Question Translation (SMT), Amharic/English
Question Analysis (run on both the original and the translated
question), Passage Retrieval, Answer Extraction -> Answers.

This module wires together every component built in the sibling files:
  indexer.Crawler / indexer.Indexer      (4.2.1, 4.2.2)
  translator.Translator                   (4.2.3)
  question_analysis.QuestionAnalyzer      (4.2.4)
  passage_retrieval.PassageRetriever      (4.2.5)
  answer_extraction.AnswerExtractor       (4.2.6)

Exactly as Figure 4.1 shows, a single user question produces *two* runs:
one against same-language documents (monolingual retrieval) and one
against the other-language documents using the SMT-translated question
(cross-lingual retrieval). Both sets of candidate answers are returned so
callers (CLI, evaluation script) can inspect either the combined best
answer or each run separately -- matching how Chapter 5 evaluates
monolingual and cross-lingual runs as distinct experiments.
"""
from . import amharic_nlp, english_nlp
from .answer_extraction import AnswerExtractor, build_doc_freq_map
from .indexer import Crawler, Indexer
from .ner_amharic import AmharicNER
from .ner_english import EnglishNER
from .passage_retrieval import PassageRetriever
from .question_analysis import QuestionAnalyzer
from .translator import Translator

OTHER_LANG = {"am": "en", "en": "am"}


class CrossLingualQA:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._log("Training Amharic NER (MaxEnt)...")
        self.ner_am = AmharicNER()
        self._log(f"  {self.ner_am.num_training_sentences} weakly-labeled training sentences, "
                   f"train accuracy={self.ner_am.tagger.train_accuracy:.3f}")

        self._log("Training English NER (MaxEnt)...")
        self.ner_en = EnglishNER()
        self._log(f"  {self.ner_en.num_training_sentences} weakly-labeled training sentences, "
                   f"train accuracy={self.ner_en.tagger.train_accuracy:.3f}")

        self._log("Crawling + indexing local corpus...")
        self.indexer = Indexer(ner_en=self.ner_en, ner_am=self.ner_am)
        docs = Crawler().crawl_local_corpus()
        self.indexer.index_corpus(docs)
        self._log(f"  Indexed {self.indexer.num_docs('en')} English docs, "
                   f"{self.indexer.num_docs('am')} Amharic docs.")

        self._log("Training SMT (IBM Model 1, EM) Amharic<->English...")
        self.translator = Translator()

        self._log("Training answer-type MaxEnt classifiers...")
        self.analyzers = {"am": QuestionAnalyzer("am"), "en": QuestionAnalyzer("en")}
        for lang, qa in self.analyzers.items():
            self._log(f"  [{lang}] train accuracy="
                       f"{qa.type_classifier.classifier.train_accuracy:.3f}")

        self.retriever = PassageRetriever(self.indexer, self.ner_en, self.ner_am)
        self.extractors = {
            "am": AnswerExtractor(amharic_nlp.stem_word),
            "en": AnswerExtractor(lambda w: english_nlp.stem_tokens([w])[0]),
        }
        self._log("System ready.\n")

    def _log(self, msg):
        if self.verbose:
            print(f"[CrossLingualQA] {msg}")

    def _run_language(self, question_text, analysis_lang, retrieval_lang, mode):
        analyzer = self.analyzers[analysis_lang]
        analysis = analyzer.analyze(question_text)
        candidates = self.retriever.retrieve({**analysis["query"], "lang": retrieval_lang})
        doc_freq_map = build_doc_freq_map(self.indexer, retrieval_lang)
        total_docs = self.indexer.num_docs(retrieval_lang)
        extractor = self.extractors[retrieval_lang]

        results = []
        for cand in candidates:
            extracted = extractor.extract(cand, analysis["query"], doc_freq_map, total_docs)
            if extracted:
                results.append({**extracted, "mode": mode, "analysis_lang": analysis_lang,
                                 "retrieval_lang": retrieval_lang, "answer_type": analysis["answer_type"]})
        results.sort(key=lambda r: -r["score"])
        return analysis, results

    def ask(self, question_text, source_lang=None, top_k=3):
        """Full pipeline run for one user question.

        Returns a dict with the detected language, the SMT-translated
        question, the answer-type analysis for each language, and ranked
        answers from BOTH the monolingual run (question language vs. same
        language documents) and the cross-lingual run (translated
        question vs. the other language's documents) -- matching
        Figure 4.1 and the Chapter 5 monolingual/cross-lingual evaluation
        split.
        """
        lang = source_lang or self.indexer.lang_id.identify(question_text)[0]
        other = OTHER_LANG[lang]
        translated = self.translator.translate(question_text, lang, other)

        mono_analysis, mono_results = self._run_language(question_text, lang, lang, "monolingual")
        cross_analysis, cross_results = self._run_language(translated, other, other, "cross-lingual")

        return {
            "question": question_text,
            "detected_lang": lang,
            "translated_question": translated,
            "monolingual": {"answer_type": mono_analysis["answer_type"], "results": mono_results[:top_k]},
            "cross_lingual": {"answer_type": cross_analysis["answer_type"], "results": cross_results[:top_k]},
        }

    def best_answer(self, question_text, source_lang=None):
        result = self.ask(question_text, source_lang=source_lang)
        pooled = result["monolingual"]["results"] + result["cross_lingual"]["results"]
        pooled.sort(key=lambda r: -r["score"])
        return pooled[0] if pooled else None, result
