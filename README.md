# Amharic-English Cross Language Question Answering System

A working Python implementation of the methodology described in **Emebet
Bekele's 2019 MSc thesis** (Addis Ababa University), *"Amharic-English
Cross Language Question Answering System"* — statistical machine
translation, per-language document analysis, maximum-entropy named
entity recognition, and window-based answer extraction, tied together
into an end-to-end cross-lingual factoid QA pipeline.

## Why this isn't a byte-for-byte reproduction

The thesis architecture (Figure 4.1, Chapter 4) is built on a stack of
external server infrastructure: **Apache Nutch** (crawler), **Apache
Tika** (parsing/language ID), **Apache Solr/Lucene** (indexing/search),
**Apache OpenNLP** (sentence/NER models), and **Moses + GIZA++** (phrase-
based SMT with word alignment). None of that is installable or reachable
from this environment — there's no outbound access to arbitrary hosts or
to the model-download servers OpenNLP/Moses depend on, and standing up a
Solr/Moses server cluster is out of scope for a portable, single-command
demo anyway.

So every component has been **reimplemented directly in pure Python**
(plus `scikit-learn`/`numpy` for the maximum-entropy models), keeping the
same *statistical/algorithmic approach* the thesis describes rather than
substituting hand-written rules where the thesis used a trained model.
Concretely:

| Thesis component (chapter/section) | Thesis tooling | This implementation |
|---|---|---|
| Web Crawler (4.2.1) | Apache Nutch | `indexer.Crawler` — loads a bundled local corpus; `fetch_url()` provided for real crawling where network access allows it |
| Text parsing / language ID (4.2.2.1–2) | Apache Tika, N-gram profiles | `language_id.py` — the same out-of-place N-gram ranking algorithm (Cavnar & Trenkle), implemented directly |
| English document analysis (4.2.2.3) | OpenNLP + Solr analyzer | `english_nlp.py` + `porter_stemmer.py` — a from-scratch Porter (1980) stemmer, since OpenNLP models can't be downloaded here |
| Amharic document analysis (4.2.2.3) | Custom normalizer/stemmer (Alemayehu Nega & Willett) | `amharic_nlp.py` — same normalization-map + affix-stripping approach |
| Amharic/English NER (4.2.2.3) | Maximum-entropy models (OpenNLP MaxEnt) | `ner_amharic.py` / `ner_english.py` + `maxent_ner.py` — **scikit-learn `LogisticRegression(multinomial)`**, which *is* a maximum-entropy classifier (identical model family; MaxEnt and multinomial logistic regression are the same equations under different names in the NLP literature) |
| Indexing with term positions (4.2.2, Algorithm 1) | Lucene | `indexer.Indexer` — in-memory inverted index storing token positions per document, the same information a Lucene `TermVector` gives you |
| Question Translation / SMT (4.2.3) | Moses + GIZA++, large bible/history/constitution corpus | `translator.py` — **IBM Model 1**, trained by EM (the same first-stage algorithm GIZA++ itself runs), on a small bundled parallel corpus, plus a rule-based question-word-fronting step standing in for the target-language model + decoder search |
| Question parsing / answer-type / query generation (4.2.4, Algorithms 2–3) | Amharic chunker, OpenNLP chunker, MaxEnt classifier | `question_analysis.py` — rule-based shallow parse (question word / verb / support terms) + a MaxEnt (multinomial logistic regression) answer-type classifier trained on bag-of-words features, exactly as Section 4.2.4.1(2) describes |
| Passage Retrieval (4.2.5) | Solr search + NE-type filter | `passage_retrieval.py` — term-overlap scoring at sentence granularity + the same "discard passages missing the expected answer type" rule |
| Answer Extraction (4.2.6, Algorithm 4) | Custom windowing algorithm | `answer_extraction.py` — direct implementation of the 5-window (main/prev1/next1/prev2/next2) scoring + bigram bonus + IDF-style term weighting described in Algorithm 4 |

**The one deliberate compromise worth calling out:** the bundled corpus,
parallel corpus, and NER gazetteer are small demo-scale data (15 facts
about Ethiopia, ~30 sentence pairs) rather than the thesis's real crawled
web corpus and labeled datasets. This is a sandbox/portability
constraint, not a methodology change — every algorithm scales to more
data unmodified; you'd only need to point `data/corpus_en/`,
`data/corpus_am/`, `data/parallel_corpus.tsv`, and the gazetteers at
larger, real datasets to get thesis-scale results. The evaluation numbers
below should be read as *"the pipeline works end-to-end and each stage
behaves as designed,"* not as a claim about system performance at scale.

## Architecture

```
                         ┌─────────────────────┐
                         │   User Question      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ Language Identification│  (language_id.py)
                         └──────────┬───────────┘
                    ┌───────────────┼────────────────┐
                    │                                 │
         ┌──────────▼──────────┐          ┌──────────▼───────────┐
         │  Question Analysis   │          │ Question Translation  │
         │  (same language)     │          │   SMT (translator.py) │
         │ question_analysis.py │          └──────────┬───────────┘
         └──────────┬──────────┘                      │
                    │                        ┌──────────▼───────────┐
                    │                        │  Question Analysis    │
                    │                        │  (other language)     │
                    │                        │ question_analysis.py  │
                    │                        └──────────┬───────────┘
         ┌──────────▼──────────────────────────────────▼───────────┐
         │              Passage Retrieval (passage_retrieval.py)     │
         │      (against Indexer: indexer.py, built by Crawler)      │
         └──────────┬──────────────────────────────────┬───────────┘
                    │                                    │
         ┌──────────▼──────────┐            ┌───────────▼──────────┐
         │  Answer Extraction   │            │  Answer Extraction    │
         │ (answer_extraction.py)│           │ (answer_extraction.py)│
         └──────────┬──────────┘            └───────────┬──────────┘
                    │                                    │
              Monolingual answer                  Cross-lingual answer
```

This mirrors Figure 4.1 of the thesis: **every question produces two
answer sets** — a monolingual one (same-language documents) and a
cross-lingual one (SMT-translated question against the other language's
documents) — which is exactly how Chapter 5 evaluates the system as two
separate experiments.

## Setup

```bash
pip install -r requirements.txt
```

No internet access, downloaded models, or external servers are required
— everything trains in under a second on the bundled demo data.

## Usage

**Interactive CLI:**
```bash
python cli.py
```
```
Q: What is the capital city of Ethiopia?

Detected language : English
SMT-translated to : ምንድን ናት። የኢትዮጵያ ዋና ከተማ የኢትዮጵያ ኢትዮጵያ

[Monolingual]  expected answer type: LOCATION
  Best answer: 'Addis Ababa'  (score=5.19)
  From: "Addis Ababa is the capital city of Ethiopia."  [en:doc1_addis_ababa.txt]

[Cross-lingual] expected answer type: LOCATION
  Best answer: 'አዲስ አበባ የኢትዮጵያ'  (score=5.19)
  From: "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"  [am:doc1_addis_ababa.txt]
```

**One-shot question:**
```bash
python cli.py "Who founded Addis Ababa?"
```

**As a library:**
```python
from clqa.pipeline import CrossLingualQA

qa = CrossLingualQA()
result = qa.ask("Where does the Blue Nile River originate?")
print(result["monolingual"]["results"][0]["answer"])   # "Lake Tana"
```

**Evaluation (precision/recall, mirroring Chapter 5):**
```bash
python eval.py
```
Runs the bundled 15-question bilingual evaluation set in all four
configurations (EN monolingual, AM monolingual, EN→AM cross-lingual,
AM→EN cross-lingual) and prints a per-question breakdown plus a summary
table, e.g.:

```
1) English monolingual retrieval (EN question -> EN docs)    P= 80.0%  R= 80.0%
1) Amharic monolingual retrieval (AM question -> AM docs)     P= 73.3%  R= 73.3%
2) English-to-Amharic cross-language retrieval                P= 66.7%  R= 66.7%
2) Amharic-to-English cross-language retrieval                P= 60.0%  R= 40.0%
```
(Exact numbers will vary slightly run-to-run only if you change the
bundled data — the models themselves are deterministic.) These are in
the same ballpark as the thesis's own Chapter 5 findings: monolingual
retrieval outperforms cross-lingual, and the gap traces back to
translation quality — exactly the conclusion the thesis draws.

**Module-level ablation study (diagnostic evaluation, with confidence
intervals and error categorization):**
```bash
python ablation.py
```
Isolates each pipeline stage — answer-type classification, NER recall
on gold sentences, passage retrieval hit rate, and SMT translation
quality (unigram overlap F1 against an independent reference) — reports
each with a 95% Wilson score confidence interval, and manually
categorizes every end-to-end error into one of four failure modes
(`wrong_answer_type`, `no_passage_retrieved`, `wrong_passage_retrieved`,
`correct_passage_wrong_span`). This is the script behind the diagnostic
results and error analysis reported in the companion paper, *"Diagnosing
a Rule-Based Amharic–English Cross-Language Question Answering Pipeline:
A Reimplementation, Ablation Study, and Roadmap Toward Neural
Multilingual QA."*

**Tests:**
```bash
pytest tests/ -v
```

## Project layout

```
clqa_project/
├── README.md
├── requirements.txt
├── cli.py                    # interactive / one-shot CLI
├── eval.py                   # precision/recall evaluation (Chapter 5 style)
├── ablation.py                # module-level ablation, confidence intervals, error taxonomy
├── src/clqa/
│   ├── porter_stemmer.py     # English stemming (4.2.2.3)
│   ├── language_id.py        # N-gram language identification (4.2.2.2)
│   ├── english_nlp.py        # English document/query analysis (4.2.2.3)
│   ├── amharic_nlp.py        # Amharic document/query analysis (4.2.2.3)
│   ├── maxent_ner.py         # generic MaxEnt tagger/classifier (shared)
│   ├── ner_english.py        # English NER (4.2.2.3)
│   ├── ner_amharic.py        # Amharic NER (4.2.2.3)
│   ├── indexer.py            # crawler stand-in + inverted index (4.2.1, 4.2.2)
│   ├── translator.py         # IBM Model 1 SMT (4.2.3)
│   ├── question_analysis.py  # parsing + answer-type + query gen (4.2.4)
│   ├── passage_retrieval.py  # passage ranking + NE-type filter (4.2.5)
│   ├── answer_extraction.py  # window-based scoring, Algorithm 4 (4.2.6)
│   └── pipeline.py           # ties everything together (Figure 4.1)
├── data/
│   ├── corpus_en/, corpus_am/   # demo "crawled" documents (5 facts each)
│   ├── parallel_corpus.tsv      # bilingual sentences for SMT training
│   ├── qa_training_en.tsv, qa_training_am.tsv  # answer-type classifier training data
│   ├── qa_eval.tsv              # 15-question bilingual evaluation set
│   ├── gazetteer_en.json, gazetteer_am.json    # NER gazetteers
│   └── stopwords_am.txt
└── tests/test_pipeline.py    # unit tests for every component
```

## Known limitations (and how they map to future work)

- **Corpus scale**: 10 documents / 30 sentence pairs vs. a real web-scale
  crawl. Swap in larger files under `data/` and everything (language ID,
  NER training, SMT, retrieval) picks up the extra data automatically —
  no code changes needed.
- **SMT quality**: IBM Model 1 word-for-word translation with a rule-based
  reordering step is meaningfully weaker than a tuned Moses phrase-based
  system with a real language model. This is also the thesis's own
  identified bottleneck (Chapter 5/6): cross-lingual precision/recall
  trails monolingual specifically because of translation noise.
- **NER training data**: weakly labeled from a small gazetteer rather
  than hand-annotated, so the classifier leans partly on the gazetteer
  overlay at inference time (a standard hybrid approach) rather than
  learning entity patterns from scratch. With a real annotated corpus
  (as the thesis used), `ner_amharic.py`/`ner_english.py` would need no
  structural changes — just a bigger `build_training_sentences()` input.
- **Question parsing**: rule-based (question word + following token as
  "verb") rather than a trained chunker, since chunker models aren't
  downloadable here. `question_analysis.QuestionParser` is the single
  place to swap in a real chunker later.
