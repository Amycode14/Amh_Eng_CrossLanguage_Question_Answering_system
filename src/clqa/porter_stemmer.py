"""
Pure-Python implementation of the classic Porter (1980) stemming algorithm.

Thesis reference: Section 4.2.2.3 (English document analysis -> Stemming),
which states "Stemming is the process of reducing inflected ... words to
their word stem". The thesis relies on Apache Solr's English analyzer,
which internally runs a Porter-style stemmer. This module re-implements
that algorithm directly so the pipeline has no external NLP-server
dependency.
"""
import re

VOWELS = "aeiou"


def _is_consonant(word, i):
    ch = word[i]
    if ch in VOWELS:
        return False
    if ch == 'y':
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


def _measure(stem):
    """Count VC repetitions (the 'm' measure in Porter's paper)."""
    form = ''.join('C' if _is_consonant(stem, i) else 'V' for i in range(len(stem)))
    form = re.sub(r'C+', 'C', form)
    form = re.sub(r'V+', 'V', form)
    return form.count('VC')


def _contains_vowel(stem):
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word):
    return (len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1))


def _ends_cvc(word):
    if len(word) < 3:
        return False
    a, b, c = word[-3], word[-2], word[-1]
    if not _is_consonant(word, len(word) - 3):
        return False
    if _is_consonant(word, len(word) - 2):
        return False
    if not _is_consonant(word, len(word) - 1):
        return False
    return c not in ('w', 'x', 'y')


def _replace_suffix(word, suffixes_replacements, cond=None):
    for suf, rep in suffixes_replacements:
        if word.endswith(suf):
            stem = word[: len(word) - len(suf)]
            if cond is None or cond(stem):
                return stem + rep
    return None


def stem(word):
    """Return the Porter stem of a single lowercase English word."""
    word = word.lower()
    if len(word) <= 2:
        return word

    # Step 1a
    r = _replace_suffix(word, [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")])
    if r is not None:
        word = r

    # Step 1b
    if word.endswith("eed"):
        stem_ = word[:-3]
        if _measure(stem_) > 0:
            word = stem_ + "ee"
    else:
        flag = False
        for suf in ("ed", "ing"):
            if word.endswith(suf):
                stem_ = word[: -len(suf)]
                if _contains_vowel(stem_):
                    word = stem_
                    flag = True
                break
        if flag:
            if word.endswith(("at", "bl", "iz")):
                word = word + "e"
            elif _ends_double_consonant(word) and word[-1] not in ("l", "s", "z"):
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word = word + "e"

    # Step 1c
    if word.endswith("y") and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Step 2
    step2 = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ]
    r = _replace_suffix(word, step2, cond=lambda s: _measure(s) > 0)
    if r is not None:
        word = r

    # Step 3
    step3 = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    r = _replace_suffix(word, step3, cond=lambda s: _measure(s) > 0)
    if r is not None:
        word = r

    # Step 4
    step4 = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ]
    for suf in step4:
        if word.endswith(suf):
            stem_ = word[: -len(suf)]
            if suf == "ion":
                continue
            if _measure(stem_) > 1:
                word = stem_
            break
    if word.endswith("ion"):
        stem_ = word[:-3]
        if _measure(stem_) > 1 and stem_.endswith(("s", "t")):
            word = stem_

    # Step 5a
    if word.endswith("e"):
        stem_ = word[:-1]
        if _measure(stem_) > 1 or (_measure(stem_) == 1 and not _ends_cvc(stem_)):
            word = stem_

    # Step 5b
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]

    return word


def stem_tokens(tokens):
    return [stem(t) for t in tokens]
