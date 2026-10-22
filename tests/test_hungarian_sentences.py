"""Tests for the HuSpaCy sentence-boundary repair (pure, network-free). The cases follow the
shapes a probe of the Hungarian legal corpus found."""

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
hungarian_sentences = pytest.importorskip("verbatim_rag.hungarian_sentences")
repair_offsets = hungarian_sentences.repair_offsets


def _offsets(text, sentences):
    """Trimmed offsets of the given raw splits, located in order."""
    out, pos = [], 0
    for sentence in sentences:
        start = text.index(sentence, pos)
        out.append((start, start + len(sentence)))
        pos = start + len(sentence)
    return out


def _repair(text, sentences):
    repaired = repair_offsets(text, _offsets(text, sentences))
    for (s, e), nxt in zip(repaired, repaired[1:] + [(len(text), len(text))]):
        assert 0 <= s < e <= nxt[0]
        assert text[s:e] == text[s:e].strip()
    return [text[s:e] for s, e in repaired]


class TestMarkers:
    def test_marker_only_sentence_joins_the_next(self):
        text = "(3)\n\nAz (1) bekezdés nem alkalmazandó."
        assert _repair(text, ["(3)", "Az (1) bekezdés nem alkalmazandó."]) == [text]

    def test_marker_chain(self):
        text = "Előző mondat. (3)\n(a)\nA szöveg."
        assert _repair(text, ["Előző mondat.", "(3)", "(a)", "A szöveg."]) == [
            "Előző mondat.",
            "(3)\n(a)\nA szöveg.",
        ]

    @pytest.mark.parametrize(
        "marker", ["2.", "118.", "c)", "aa)", "A)", "ii.", "IV.", "(4a)", "(iv)", "(vii)"]
    )
    def test_marker_forms(self, marker):
        text = f"{marker} Szöveg."
        assert _repair(text, [marker, "Szöveg."]) == [text]

    def test_trailing_marker_moves_to_the_next_sentence(self):
        text = "Képzést és gyakorlást; és\nii.\n15 órányi gyakorlati képzést."
        assert _repair(
            text, ["Képzést és gyakorlást; és\nii.", "15 órányi gyakorlati képzést."]
        ) == [
            "Képzést és gyakorlást; és",
            "ii.\n15 órányi gyakorlati képzést.",
        ]

    def test_trailing_marker_lines_move_together(self):
        text = "Szöveg; és\n(b)\nii.\nKövetkező."
        assert _repair(text, ["Szöveg; és\n(b)\nii.", "Következő."]) == [
            "Szöveg; és",
            "(b)\nii.\nKövetkező.",
        ]

    def test_moved_marker_joins_a_following_marker(self):
        text = "Szöveg\nii. (a) Tétel."
        assert _repair(text, ["Szöveg\nii.", "(a)", "Tétel."]) == ["Szöveg", "ii. (a) Tétel."]

    def test_markers_in_the_last_sentence_stay(self):
        assert _repair("Szöveg. (4)", ["Szöveg.", "(4)"]) == ["Szöveg.", "(4)"]
        assert _repair("Szöveg.\n(4)", ["Szöveg.\n(4)"]) == ["Szöveg.\n(4)"]

    @pytest.mark.parametrize("word", ["2020.", "CIVIL.", "is."])
    def test_not_a_marker(self, word):
        assert _repair(f"{word} Az év.", [word, "Az év."]) == [word, "Az év."]


class TestContinuation:
    def test_abbreviation_followed_by_lowercase(self):
        text = "Az 5405 vtsz. alá tartozó szalag."
        assert _repair(text, ["Az 5405 vtsz.", "alá tartozó szalag."]) == [text]

    def test_ordinal_followed_by_lowercase_or_paragraph_sign(self):
        text = "Az 1990. évi törvény."
        assert _repair(text, ["Az 1990.", "évi törvény."]) == [text]
        text = "A Ptk. 6:519. § szerint felel."
        assert _repair(text, ["A Ptk. 6:519.", "§ szerint felel."]) == [text]

    def test_leading_punctuation_before_the_abbreviation(self):
        text = "Lásd („pl. a számlákat”)."
        assert _repair(text, ["Lásd („pl.", "a számlákat”)."]) == [text]

    def test_across_a_blank_line(self):
        text = "Bab, a 2006 vtsz.\n\nalá tartozó termékek kivételével."
        assert _repair(text, ["Bab, a 2006 vtsz.", "alá tartozó termékek kivételével."]) == [text]

    @pytest.mark.parametrize(
        "sentences",
        [
            ["Ezt is figyelembe kell.", "de nem mindig."],  # a word, not an abbreviation
            ["A Példa Kft.", "A vezető dr. Kovács."],  # capitalised next sentence
            ["A határidő 2.", "15 nap a határidő."],  # digit: a list boundary
            ["A pont 2.", "a) pont szerint."],  # enumerator next
            ["A pont 2.", "aa) alpont szerint."],  # two-letter sub-point next
        ],
    )
    def test_no_merge(self, sentences):
        assert _repair(" ".join(sentences), sentences) == sentences

    def test_empty(self):
        assert repair_offsets("", []) == []
