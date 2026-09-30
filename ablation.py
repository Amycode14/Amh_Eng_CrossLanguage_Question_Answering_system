"""
Module-level ablation study for the reimplemented baseline pipeline.

Produces the numbers reported in the "Diagnostic Evaluation" section of
the follow-up paper: per-module accuracy (answer-type classification,
NER recall on gold entities, passage retrieval hit rate, translation
token-overlap quality, end-to-end answer accuracy), each with a 95%
Wilson score confidence interval (appropriate for small-sample
proportions, unlike the normal approximation), plus a manual error
category breakdown for every incorrect end-to-end answer.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clqa import amharic_nlp, english_nlp  # noqa: E402
from clqa.pipeline import CrossLingualQA, OTHER_LANG  # noqa: E402
from eval import load_eval_set, is_correct  # noqa: E402

Z = 1.96  # 95% confidence


def wilson_ci(successes, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + Z ** 2 / n
    center = (p + Z ** 2 / (2 * n)) / denom
    margin = (Z * math.sqrt((p * (1 - p) / n) + (Z ** 2 / (4 * n ** 2)))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def fmt_ci(successes, n):
    p, lo, hi = wilson_ci(successes, n)
    return f"{p*100:5.1f}%  (95% CI [{lo*100:4.1f}%, {hi*100:4.1f}%], n={n})"


def word_overlap_f1(hyp_tokens, ref_tokens):
    """Unigram overlap F1 between a translation hypothesis and a genuine
    independent reference (the parallel eval question in the target
    language) -- a simplified BLEU-1 / ROUGE-1-style score, adequate for
    the small evaluation set used here."""
    from collections import Counter
    hyp_counts, ref_counts = Counter(hyp_tokens), Counter(ref_tokens)
    overlap = sum((hyp_counts & ref_counts).values())
    precision = overlap / len(hyp_tokens) if hyp_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def eval_answer_type_classifier(qa_system, eval_set):
    """Module 1: does the MaxEnt answer-type classifier alone predict the
    correct answer type for each eval question, in both languages?"""
    results = {"en": [], "am": []}
    for row in eval_set:
        for lang in ("en", "am"):
            question = row[f"question_{lang}"]
            analysis = qa_system.analyzers[lang].analyze(question)
            results[lang].append(analysis["answer_type"] == row["answer_type"])
    return results


def eval_ner_recall(qa_system, eval_set):
    """Module 2: given the GOLD sentence that contains the answer (found
    by locating the gold answer string in the indexed corpus), does NER
    correctly extract an entity of the expected type that matches the
    gold answer? This isolates NER quality from retrieval/translation
    noise."""
    results = {"en": [], "am": []}
    for lang, ner, tokenize_fn in (
        ("en", qa_system.ner_en, english_nlp.tokenize),
        ("am", qa_system.ner_am, lambda t: amharic_nlp.tokenize(amharic_nlp.normalize(t))),
    ):
        docs = qa_system.indexer.documents_in_lang(lang)
        for row in eval_set:
            gold = row[f"gold_{lang}"]
            found = False
            for doc in docs.values():
                for sent in doc["sentences"]:
                    if gold.lower() not in sent.lower() and gold not in sent:
                        continue
                    tokens = tokenize_fn(sent)
                    entities = ner.extract_entities(tokens)
                    for e in entities:
                        if e["label"] == row["answer_type"] and (
                            gold.lower() in e["text"].lower() or e["text"].lower() in gold.lower()
                        ):
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            results[lang].append(found)
    return results


def eval_retrieval_hit_rate(qa_system, eval_set, mode):
    """Module 3: does passage retrieval return AT LEAST ONE candidate
    passage that actually contains the gold answer string, regardless of
    whether answer extraction later picks the right span? mode:
    'monolingual' or 'cross_lingual'."""
    results = {"en": [], "am": []}
    for question_lang in ("en", "am"):
        answer_lang = question_lang if mode == "monolingual" else OTHER_LANG[question_lang]
        for row in eval_set:
            question = row[f"question_{question_lang}"]
            gold = row[f"gold_{answer_lang}"]
            result = qa_system.ask(question, source_lang=question_lang)
            candidates = result[mode]["results"]
            hit = any(gold.lower() in c["sentence"].lower() or gold in c["sentence"] for c in candidates)
            results[question_lang].append(hit)
    return results


def eval_translation_quality(qa_system, eval_set):
    """Module 4: unigram overlap F1 between the SMT output and the
    genuine independent reference question in the target language
    (the human-written parallel question in data/qa_eval.tsv)."""
    scores = {"en2am": [], "am2en": []}
    for row in eval_set:
        translated = qa_system.translator.translate(row["question_en"], "en", "am")
        hyp = amharic_nlp.tokenize(amharic_nlp.normalize(translated))
        ref = amharic_nlp.tokenize(amharic_nlp.normalize(row["question_am"]))
        scores["en2am"].append(word_overlap_f1(hyp, ref)[2])

        translated = qa_system.translator.translate(row["question_am"], "am", "en")
        hyp = [t.lower() for t in english_nlp.tokenize(translated)]
        ref = [t.lower() for t in english_nlp.tokenize(row["question_en"])]
        scores["am2en"].append(word_overlap_f1(hyp, ref)[2])
    return scores


def categorize_errors(qa_system, eval_set):
    """For every incorrect end-to-end answer (monolingual OR
    cross-lingual), assign an error category using the diagnostic signals
    already available: predicted answer type, whether any candidate
    passage was retrieved, and whether the gold sentence was among them."""
    categories = []
    for question_lang in ("en", "am"):
        for mode in ("monolingual", "cross_lingual"):
            answer_lang = question_lang if mode == "monolingual" else OTHER_LANG[question_lang]
            for row in eval_set:
                question = row[f"question_{question_lang}"]
                gold = row[f"gold_{answer_lang}"]
                result = qa_system.ask(question, source_lang=question_lang)
                candidates = result[mode]["results"]
                predicted = candidates[0]["answer"] if candidates else None
                if is_correct(predicted, gold):
                    continue
                predicted_type = result[mode]["answer_type"]
                expected_type = row["answer_type"]
                if predicted_type != expected_type:
                    cat = "wrong_answer_type"
                elif not candidates:
                    cat = "no_passage_retrieved"
                elif any(gold.lower() in c["sentence"].lower() for c in candidates):
                    cat = "correct_passage_wrong_span"
                else:
                    cat = "wrong_passage_retrieved"
                categories.append({
                    "question_lang": question_lang, "mode": mode, "question": question,
                    "gold": gold, "predicted": predicted, "category": cat,
                })
    return categories


def main():
    print("Building system...")
    qa_system = CrossLingualQA(verbose=False)
    eval_set = load_eval_set()
    n = len(eval_set)

    print("\n" + "=" * 72)
    print("MODULE-LEVEL ABLATION STUDY (n=%d bilingual eval questions)" % n)
    print("=" * 72)

    print("\n[Module 1] Answer-type classification accuracy (isolated)")
    at = eval_answer_type_classifier(qa_system, eval_set)
    for lang in ("en", "am"):
        correct = sum(at[lang])
        print(f"  {lang}: {fmt_ci(correct, n)}")

    print("\n[Module 2] NER recall on gold-sentence entities (isolated)")
    ner = eval_ner_recall(qa_system, eval_set)
    for lang in ("en", "am"):
        correct = sum(ner[lang])
        print(f"  {lang}: {fmt_ci(correct, n)}")

    print("\n[Module 3] Passage retrieval hit rate (gold sentence in top-k)")
    for mode in ("monolingual", "cross_lingual"):
        hits = eval_retrieval_hit_rate(qa_system, eval_set, mode)
        for lang in ("en", "am"):
            correct = sum(hits[lang])
            print(f"  {mode:14s} question_lang={lang}: {fmt_ci(correct, n)}")

    print("\n[Module 4] SMT translation quality (unigram overlap F1 vs. independent reference)")
    tq = eval_translation_quality(qa_system, eval_set)
    for direction, scores in tq.items():
        mean_f1 = sum(scores) / len(scores)
        print(f"  {direction}: mean F1 = {mean_f1*100:5.1f}%  (min={min(scores)*100:.1f}%, max={max(scores)*100:.1f}%)")

    print("\n[End-to-end] Full pipeline accuracy (from eval.py, reproduced here for reference)")
    from eval import run_eval
    for question_lang in ("en", "am"):
        for mode in ("monolingual", "cross_lingual"):
            report = run_eval(qa_system, question_lang, mode)
            print(f"  {question_lang} {mode:14s}: precision={fmt_ci(report['n_correct'], report['n_answered'] or 1)}"
                  f"  recall={fmt_ci(report['n_correct'], report['n_total'])}")

    print("\n" + "=" * 72)
    print("ERROR CATEGORY BREAKDOWN (all incorrect end-to-end answers, both runs)")
    print("=" * 72)
    errors = categorize_errors(qa_system, eval_set)
    from collections import Counter
    counts = Counter(e["category"] for e in errors)
    total_runs = n * 4  # 2 question langs x 2 modes
    for cat, count in counts.most_common():
        print(f"  {cat:28s}: {count:2d} / {total_runs} runs ({count/total_runs*100:.1f}%)")
    print(f"  {'TOTAL ERRORS':28s}: {len(errors):2d} / {total_runs} runs")

    print("\nPer-error detail:")
    for e in errors:
        print(f"  [{e['question_lang']}/{e['mode']:14s}] {e['category']:26s}  "
              f"Q: {e['question'][:50]:50s}  gold={e['gold']!r}  pred={e['predicted']!r}")


if __name__ == "__main__":
    main()
