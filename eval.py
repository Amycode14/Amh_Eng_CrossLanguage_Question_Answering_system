"""
Evaluation script -- reproduces the Chapter 5 experiment structure:

  1) Monolingual evaluation: Amharic questions against Amharic documents,
     English questions against English documents.
  2) Cross-language evaluation: Amharic questions (SMT-translated to
     English) against English documents, and vice versa.

For each run we compute (Section 5.3):
  Precision = correctly_answered / total_answers_given
  Recall    = correctly_answered / total_expected

"Correct" means the extracted answer text contains (or is contained in)
the gold answer string for that run, case/whitespace-insensitively. Note
that in a cross-lingual run the system retrieves from, and therefore
answers in, the *other* language's documents -- so gold answers are
looked up in that language, not the question's own language (a bilingual
eval set with gold answers in both languages is bundled specifically so
this comparison is done correctly; see data/qa_eval.tsv).

Usage:
    python eval.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clqa.pipeline import CrossLingualQA  # noqa: E402

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_EVAL_PATH = os.path.join(_DATA_DIR, "qa_eval.tsv")

OTHER_LANG = {"am": "en", "en": "am"}


def load_eval_set():
    with open(_EVAL_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _normalize_for_match(s):
    return " ".join(s.strip().lower().split())


def is_correct(predicted, gold):
    if predicted is None:
        return False
    p, g = _normalize_for_match(predicted), _normalize_for_match(gold)
    return g in p or p in g


def run_eval(qa_system, question_lang, mode):
    """question_lang: 'en' or 'am' -- the language the question is asked
    in. mode: 'monolingual' (answer expected in question_lang) or
    'cross_lingual' (answer expected in the OTHER language, since that's
    which documents get searched)."""
    eval_set = load_eval_set()
    answer_lang = question_lang if mode == "monolingual" else OTHER_LANG[question_lang]
    n_correct, n_answered, n_total = 0, 0, len(eval_set)
    rows = []
    for row in eval_set:
        question = row[f"question_{question_lang}"]
        gold = row[f"gold_{answer_lang}"]
        result = qa_system.ask(question, source_lang=question_lang)
        candidates = result[mode]["results"]
        predicted = candidates[0]["answer"] if candidates else None
        correct = is_correct(predicted, gold)
        n_answered += predicted is not None
        n_correct += correct
        rows.append({
            "question": question, "gold": gold, "predicted": predicted,
            "correct": correct, "expected_type": row["answer_type"],
            "predicted_type": result[mode]["answer_type"],
        })
    precision = n_correct / n_answered if n_answered else 0.0
    recall = n_correct / n_total if n_total else 0.0
    return {"question_lang": question_lang, "answer_lang": answer_lang, "mode": mode,
            "precision": precision, "recall": recall, "n_correct": n_correct,
            "n_answered": n_answered, "n_total": n_total, "rows": rows}


def print_report(report, title):
    print(f"\n{title}")
    print("-" * len(title))
    for row in report["rows"]:
        mark = "\u2713" if row["correct"] else "\u2717"
        print(f"  [{mark}] Q: {row['question']}")
        print(f"        expected_type={row['expected_type']}  predicted_type={row['predicted_type']}")
        print(f"        gold={row['gold']!r}  predicted={row['predicted']!r}")
    print(f"  Precision: {report['precision']*100:.1f}%  "
          f"Recall: {report['recall']*100:.1f}%  "
          f"({report['n_correct']}/{report['n_total']} correct, "
          f"{report['n_answered']}/{report['n_total']} answered)")


def main():
    print("Building the Cross Language QA system (training all models on the bundled demo data)...")
    qa_system = CrossLingualQA(verbose=True)

    reports = [
        run_eval(qa_system, "en", "monolingual"),
        run_eval(qa_system, "am", "monolingual"),
        run_eval(qa_system, "en", "cross_lingual"),  # EN question -> AM docs
        run_eval(qa_system, "am", "cross_lingual"),  # AM question -> EN docs
    ]

    titles = {
        ("en", "monolingual"): "1) English monolingual retrieval (EN question -> EN docs)",
        ("am", "monolingual"): "1) Amharic monolingual retrieval (AM question -> AM docs)",
        ("en", "cross_lingual"): "2) English-to-Amharic cross-language retrieval (EN question -> AM docs)",
        ("am", "cross_lingual"): "2) Amharic-to-English cross-language retrieval (AM question -> EN docs)",
    }
    for report in reports:
        print_report(report, titles[(report["question_lang"], report["mode"])])

    print("\n" + "=" * 70)
    print("Summary (compare with thesis Table 5.1 / 5.2):")
    print("=" * 70)
    for report in reports:
        label = titles[(report["question_lang"], report["mode"])]
        print(f"  {label:58s}  P={report['precision']*100:5.1f}%  R={report['recall']*100:5.1f}%")


if __name__ == "__main__":
    main()
