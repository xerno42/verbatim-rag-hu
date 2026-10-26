"""
Accent handling for Hungarian text typed without diacritics ("szerzodes" for "szerződés").

- strip_accents / fold: the accent-insensitive form, for the folded BM25 channel and for
  simulating unaccented queries in the eval.
- restore_accents: puts the accents back into a question from the word forms of the chunks
  a component is about to score, so the reranker and the sentence judge compare accented text
  with accented text. Retrieval itself does not need it: the folded channel and the dense
  model take the question as typed.

Pure and stdlib-only.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable

_WORD_RE = re.compile(r"\w+")


def strip_accents(text: str) -> str:
    """Drop combining marks (á -> a, ő -> o), keeping case."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if not unicodedata.combining(c)))


def fold(text: str) -> str:
    """Lowercased and accent-free: the key that accented and unaccented forms share."""
    return strip_accents(text.lower())


def has_diacritics(text: str) -> bool:
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text))


def _match_case(form: str, word: str) -> str:
    if len(word) > 1 and word.isupper():
        return form.upper()
    if word[0].isupper():
        return form[0].upper() + form[1:]
    return form


def restore_accents(question: str, texts: Iterable[str]) -> str:
    """The question with each unaccented word replaced by its most frequent form in `texts`.

    A question with any diacritic is returned unchanged: its author types accents, so a
    missing one is deliberate or a typo, not something to guess. Plain forms are counted too, and a word is only replaced when one accented form occurs
    more often than every other form, the plain one included ("2020-ban" keeps "ban" when
    "ban" outnumbers "bán"); ties keep the word as typed.
    """
    if has_diacritics(question):
        return question
    forms: Dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for word in _WORD_RE.findall(unicodedata.normalize("NFC", text or "")):
            lower = word.lower()
            forms[fold(lower)][lower] += 1

    def replace(match: re.Match) -> str:
        word = match[0]
        ranked = forms[word.lower()].most_common(2) if word.lower() in forms else []
        if not ranked:
            return word
        best, n = ranked[0]
        if best == word.lower() or (len(ranked) > 1 and ranked[1][1] == n):
            return word  # already the most frequent form, or a tie
        return _match_case(best, word)

    return _WORD_RE.sub(replace, question)
