#!/usr/bin/env python3
"""
Interactive command-line interface for the Amharic-English Cross Language
Question Answering system (implementation of Emebet Bekele's 2019 MSc
thesis methodology). Ask a question in either Amharic or English; the
system identifies the language, retrieves monolingual and cross-lingual
answers, and prints both.

Usage:
    python cli.py
    python cli.py "What is the capital city of Ethiopia?"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clqa.pipeline import CrossLingualQA  # noqa: E402


def print_result(result):
    print(f"\nDetected language : {'Amharic' if result['detected_lang'] == 'am' else 'English'}")
    print(f"SMT-translated to : {result['translated_question']}")

    mono = result["monolingual"]
    print(f"\n[Monolingual]  expected answer type: {mono['answer_type']}")
    if mono["results"]:
        top = mono["results"][0]
        print(f"  Best answer: {top['answer']!r}  (score={top['score']:.2f})")
        print(f"  From: \"{top['sentence']}\"  [{top['doc_id']}]")
    else:
        print("  No answer found.")

    cross = result["cross_lingual"]
    print(f"\n[Cross-lingual] expected answer type: {cross['answer_type']}")
    if cross["results"]:
        top = cross["results"][0]
        print(f"  Best answer: {top['answer']!r}  (score={top['score']:.2f})")
        print(f"  From: \"{top['sentence']}\"  [{top['doc_id']}]")
    else:
        print("  No answer found.")


def main():
    print("Building the Amharic-English Cross Language QA system...")
    qa = CrossLingualQA(verbose=True)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print_result(qa.ask(question))
        return

    print("\nType a question in Amharic or English (or 'quit' to exit).")
    print("Example: What is the capital city of Ethiopia?")
    print("Example: የኢትዮጵያ ዋና ከተማ ምንድን ናት?\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        print_result(qa.ask(question))
        print()


if __name__ == "__main__":
    main()
