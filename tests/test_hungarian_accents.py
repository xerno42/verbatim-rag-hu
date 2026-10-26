"""Tests for the Hungarian accent helpers (pure, network-free)."""

import unicodedata

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
accents = pytest.importorskip("verbatim_rag.hungarian_accents")


class TestFolding:
    def test_strip_accents_keeps_case(self):
        assert accents.strip_accents("Szerződés ÉS Ügyfél") == "Szerzodes ES Ugyfel"

    def test_fold_lowercases(self):
        assert accents.fold("ŐRZÉS") == "orzes"

    def test_nfd_input_folds_like_nfc(self):
        assert accents.fold(unicodedata.normalize("NFD", "díj")) == "dij"

    def test_has_diacritics(self):
        assert accents.has_diacritics("díj")
        assert not accents.has_diacritics("dij 2020. (3)")


class TestRestoreAccents:
    texts = [
        "A szerződés felmondása esetén a munkáltató díjat fizet.",
        "A szerződés megszűnik, és a díj esedékes.",
    ]

    def test_restores_from_chunk_word_forms(self):
        out = accents.restore_accents("Mikor szunik meg a szerzodes es mi a dij?", self.texts)
        assert out == "Mikor szunik meg a szerződés és mi a díj?"

    def test_accented_question_is_unchanged(self):
        question = "Mi a dij a szerződés szerint?"
        assert accents.restore_accents(question, self.texts) == question

    def test_more_frequent_plain_form_wins(self):
        texts = ["2020-ban és 2021-ben", "a ban szó", "egy ban", "a bán"]
        assert accents.restore_accents("ban", texts) == "ban"

    def test_tie_between_accented_forms_keeps_the_word(self):
        assert accents.restore_accents("ot", ["öt alma", "őt látta"]) == "ot"

    def test_case_is_kept(self):
        assert accents.restore_accents("Szerzodes", self.texts) == "Szerződés"
        assert accents.restore_accents("DIJ", self.texts) == "DÍJ"

    def test_nfd_texts(self):
        texts = [unicodedata.normalize("NFD", t) for t in self.texts]
        assert accents.restore_accents("dij", texts) == "díj"

    def test_no_texts(self):
        assert accents.restore_accents("szerzodes", []) == "szerzodes"
