"""
Repair of HuSpaCy sentence boundaries for the Hungarian legal corpus.

`hu_core_news_lg` already keeps ordinals ("1990. évi") and common abbreviations ("kft.",
"stb.") inside their sentence. A probe of 3,000 corpus chunks (7,841 sentences) found three
defects that remain, and `repair_offsets` fixes them:

1. An enumerator split off as a sentence of its own: "(3)", "2.", "c)" (4% of sentences).
   It is merged into the sentence it introduces.
2. Enumerators on their own lines at the end of the previous sentence ("... és\\nii." |
   "15 órányi ..."). They are moved to the start of the next sentence.
3. A break after an abbreviation or ordinal inside a sentence ("az 5405 vtsz." | "alá
   tartozó ..."). The halves are merged when the second starts with a lowercase word.

Repairs only move boundaries within the text, so every sentence is still text[start:end] and
spans stay verbatim by construction.
"""

import re
from typing import List, Tuple

# Well-formed Roman numerals (1-399 upper, 1-39 lower), so "CIVIL." is not one.
_ROMAN = r"(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_ROMAN_LOWER = r"(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})"
# A whole enumerator: "(3)", "(4a)", "(aa)", "(iv)", "2.", "118.", "c)", "aa)", "A)", "ii.",
# "IV.". At most three digits, so a year ("2020.") is not one. Two letters only before ")",
# since "is.", "be.", "le." are words. Known, harmless false positives: a lone letter ("a.")
# and a lone 3-digit number ("350.") on their own line.
_MARKER_RE = re.compile(
    rf"\((?:\d{{1,3}}[a-z]?|[a-z]{{1,2}}|{_ROMAN_LOWER})\)"
    rf"|(?:\d{{1,3}}[a-z]?|[a-z]|{_ROMAN_LOWER}|{_ROMAN})[.)]"
    r"|[a-z]{2}\)|[A-Z]\)"
)
# Abbreviations that end a sentence-internal token, lowercased. Words that often end a real
# sentence ("is.", "kell.", "meg.") are deliberately absent.
_ABBREVIATIONS = frozenset(
    "vtsz. pl. stb. ill. kb. ld. vö. ún. sz. u. dr. kft. zrt. nyrt. bt. bek. tv. ptk. btk. "
    "pp. mt. hrsz.".split()
)
# "1990.", "6:519.", "IV." (a lowercase Roman numeral is more often an enumerator).
_ORDINAL_RE = re.compile(rf"\d+(?:[:/]\d+)*\.|{_ROMAN}\.")
_LEADING_PUNCT = "(„\"'["


def _is_markers(sentence: str) -> bool:
    """True when the sentence consists of enumerators only ("(3)", "(3)\\n(a)")."""
    tokens = sentence.split()
    return bool(tokens) and all(_MARKER_RE.fullmatch(t) for t in tokens)


def _continues(text: str, prev: Tuple[int, int], start: int) -> bool:
    """Rule 3: prev ends in an abbreviation or ordinal and the next sentence (at start)
    continues it with a lowercase word, or with "§" after an ordinal. Never into an
    enumerator. Blank lines are no barrier: the corpus has them inside sentences
    ("a 2006 vtsz.\n\nalá tartozó")."""
    last = text[prev[0] : prev[1]].split()[-1].lstrip(_LEADING_PUNCT)
    first = text[start:].split(None, 1)[0]
    if _MARKER_RE.fullmatch(first):
        return False
    lowercase = first[0].isalpha() and first[0].islower()
    if last.lower() in _ABBREVIATIONS:
        return lowercase
    if _ORDINAL_RE.fullmatch(last):
        return lowercase or first[0] == "§"
    return False


def repair_offsets(text: str, offsets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Repaired (start, end) sentence offsets, for trimmed, ordered, non-overlapping input
    offsets into text. Output offsets have the same properties."""
    out: List[Tuple[int, int]] = []
    carry = None  # start of enumerators moved from the previous sentence (rule 2)
    for i, (start, end) in enumerate(offsets):
        if carry is not None:
            start, carry = carry, None
        if out and (_is_markers(text[out[-1][0] : out[-1][1]]) or _continues(text, out[-1], start)):
            out[-1] = (out[-1][0], end)  # rules 1 and 3
        else:
            out.append((start, end))

        s, e = out[-1]
        sentence = text[s:e]
        if i + 1 < len(offsets) and not _is_markers(sentence):
            body = sentence  # strip trailing marker lines, keeping some text before them
            while (
                (cut := body.rfind("\n")) > 0
                and body[:cut].strip()
                and _MARKER_RE.fullmatch(body[cut + 1 :].strip())
            ):
                body = body[:cut].rstrip()
            if len(body) < len(sentence):
                rest = sentence[len(body) :]
                out[-1] = (s, s + len(body))
                carry = s + len(body) + len(rest) - len(rest.lstrip())
    return out
