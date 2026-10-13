"""Tests for LayaSentenceExtractor with a mocked laya agent and spaCy pipeline (network-free)."""

import re
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
laya_extractor = pytest.importorskip("verbatim_rag.laya_extractor")
LayaSentenceExtractor = laya_extractor.LayaSentenceExtractor


class FakeNLP:
    """Splits after '.', '!' or '?'; sentences keep leading whitespace like HuSpaCy does."""

    pipe_names = ["tok2vec", "senter"]

    def pipe(self, texts, batch_size=64):
        for text in texts:
            sents = [
                SimpleNamespace(start_char=m.start(), end_char=m.end())
                for m in re.finditer(r"\s*[^.!?]+[.!?]?", text)
            ]
            yield SimpleNamespace(sents=sents)


def _result(p, truncated=False):
    return {
        "answers": {"relevant": {"type": "choice", "probabilities": {"A": p, "B": 1 - p}}},
        "usage": {"truncated": truncated},
    }


@pytest.fixture
def fake_laya():
    agent = MagicMock()
    module = types.ModuleType("laya")
    module.load = MagicMock(return_value=agent)
    with patch.dict(sys.modules, {"laya": module}):
        yield module, agent


def _scores_by_sentence(agent, table):
    """predict_batch returns P(A) looked up by each state (the sentence)."""

    def predict_batch(states, questions, **kwargs):
        return [_result(table.get(s, 0.0)) for s in states]

    agent.predict_batch.side_effect = predict_batch


def _chunk(text, chunk_id="c"):
    return SimpleNamespace(id=chunk_id, text=text)


class TestInit:
    def test_load_arguments(self, fake_laya):
        module, _ = fake_laya
        LayaSentenceExtractor(
            model="org/laya", revision="abc", device="cpu", calibration="cal.json", nlp=FakeNLP()
        )
        module.load.assert_called_once_with(
            "org/laya", device="cpu", revision="abc", calibration="cal.json"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 1.5},
            {"threshold": -0.1},
            {"criteria": {"true": "yes", "false": "no"}},
            {"criteria": {"A": "yes"}},
            {"instructions": "Is the sentence relevant?"},
        ],
    )
    def test_invalid_arguments(self, fake_laya, kwargs):
        with pytest.raises(ValueError):
            LayaSentenceExtractor(nlp=FakeNLP(), **kwargs)

    def test_pipeline_without_sentence_boundaries(self, fake_laya):
        nlp = FakeNLP()
        nlp.pipe_names = ["tok2vec", "tagger"]
        with pytest.raises(ValueError, match="sentence boundaries"):
            LayaSentenceExtractor(nlp=nlp)


class TestExtractSpans:
    def test_keeps_sentences_at_or_above_threshold_in_order(self, fake_laya):
        _, agent = fake_laya
        text = "Első mondat. Második mondat! Harmadik mondat?"
        _scores_by_sentence(
            agent, {"Első mondat.": 0.9, "Második mondat!": 0.2, "Harmadik mondat?": 0.5}
        )
        ex = LayaSentenceExtractor(nlp=FakeNLP(), threshold=0.5)

        spans = ex.extract_spans("Kérdés?", [_chunk(text)])

        assert spans == {text: ["Első mondat.", "Harmadik mondat?"]}
        assert all(s in text for s in spans[text])

    def test_question_is_a_b_choice_with_user_question_in_instructions(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {})
        LayaSentenceExtractor(nlp=FakeNLP()).extract_spans(
            "Mennyi a {díj}?", [_chunk("Egy mondat.")]
        )

        questions = agent.predict_batch.call_args.args[1]
        assert questions["relevant"]["type"] == "choice"
        assert set(questions["relevant"]["criteria"]) == {"A", "B"}
        assert questions["relevant"]["instructions"].startswith("Kérdés: Mennyi a {díj}?\n")

    def test_custom_instruction_template(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {})
        ex = LayaSentenceExtractor(nlp=FakeNLP(), instructions="{question} -- relevant?")
        ex.extract_spans("Q", [_chunk("Egy.")])

        assert agent.predict_batch.call_args.args[1]["relevant"]["instructions"] == "Q -- relevant?"

    def test_state_is_the_trimmed_sentence(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {})
        LayaSentenceExtractor(nlp=FakeNLP()).extract_spans("Q", [_chunk("Egy.  Kettő.\nHárom.")])

        assert agent.predict_batch.call_args.args[0] == ["Egy.", "Kettő.", "Három."]

    def test_one_batch_call_across_chunks(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {"B1.": 0.8})
        ex = LayaSentenceExtractor(nlp=FakeNLP(), batch_size=7, max_len=2048)

        spans = ex.extract_spans("Q", [_chunk("A1. A2.", "a"), _chunk("B1. B2.", "b")])

        assert agent.predict_batch.call_count == 1
        assert len(agent.predict_batch.call_args.args[0]) == 4
        assert agent.predict_batch.call_args.kwargs == {
            "batch_size": 7,
            "lang": "hu",
            "max_len": 2048,
        }
        assert spans == {"A1. A2.": [], "B1. B2.": ["B1."]}

    def test_blank_chunks_and_no_results_skip_the_model(self, fake_laya):
        _, agent = fake_laya
        ex = LayaSentenceExtractor(nlp=FakeNLP())

        assert ex.extract_spans("Q", []) == {}
        assert ex.extract_spans("Q", [_chunk("  \n")]) == {"  \n": []}
        agent.predict_batch.assert_not_called()

    def test_result_count_mismatch_raises(self, fake_laya):
        _, agent = fake_laya
        agent.predict_batch.return_value = [_result(0.9)]
        ex = LayaSentenceExtractor(nlp=FakeNLP())

        with pytest.raises(ValueError, match="1 results for 2 sentences"):
            ex.extract_spans("Q", [_chunk("Egy. Kettő.")])

    def test_truncation_is_logged(self, fake_laya, caplog):
        _, agent = fake_laya
        agent.predict_batch.return_value = [_result(0.9, truncated=True)]
        ex = LayaSentenceExtractor(nlp=FakeNLP())

        ex.extract_spans("Q", [_chunk("Egy.")])

        assert "truncated 1 of 1" in caplog.text


class TestScores:
    def test_duplicate_chunk_texts_are_scored_once(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {"Egy.": 0.7})
        ex = LayaSentenceExtractor(nlp=FakeNLP())

        scores = ex.score_sentences("Q", [_chunk("Egy.", "a"), _chunk("Egy.", "b")])

        assert len(agent.predict_batch.call_args.args[0]) == 1
        assert scores == {"Egy.": [("Egy.", 0.7)]}

    def test_repeated_sentence_across_overlapping_chunks_is_judged_once(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {"Közös.": 0.6})
        ex = LayaSentenceExtractor(nlp=FakeNLP())

        scores = ex.score_sentences("Q", [_chunk("Egy. Közös.", "a"), _chunk("Közös. Kettő.", "b")])

        assert agent.predict_batch.call_args.args[0] == ["Egy.", "Közös.", "Kettő."]
        assert scores["Egy. Közös."][1] == ("Közös.", 0.6)
        assert scores["Közös. Kettő."][0] == ("Közös.", 0.6)

    def test_score_sentences_returns_raw_probabilities(self, fake_laya):
        _, agent = fake_laya
        _scores_by_sentence(agent, {"Egy.": 0.25, "Kettő.": 0.75})
        ex = LayaSentenceExtractor(nlp=FakeNLP())

        scores = ex.score_sentences("Q", [_chunk("Egy. Kettő.")])

        assert scores == {"Egy. Kettő.": [("Egy.", 0.25), ("Kettő.", 0.75)]}

    def test_spans_from_scores_threshold_sweep(self):
        scores = {"t": [("a", 0.3), ("b", 0.6), ("c", 0.9)]}
        assert LayaSentenceExtractor.spans_from_scores(scores, 0.5) == {"t": ["b", "c"]}
        assert LayaSentenceExtractor.spans_from_scores(scores, 0.95) == {"t": []}
