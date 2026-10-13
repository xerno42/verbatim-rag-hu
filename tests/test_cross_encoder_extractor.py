"""Tests for CrossEncoderSentenceExtractor with a mocked CrossEncoder and spaCy pipeline
(network-free)."""

import re
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
ce_extractor = pytest.importorskip("verbatim_rag.cross_encoder_extractor")
CrossEncoderSentenceExtractor = ce_extractor.CrossEncoderSentenceExtractor
torch = pytest.importorskip("torch")


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


@pytest.fixture
def fake_st():
    model = MagicMock()
    model.num_labels = 1
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = MagicMock(return_value=model)
    with patch.dict(sys.modules, {"sentence_transformers": module}):
        yield module, model


def _scores_by_sentence(model, table):
    """predict returns the score looked up by each pair's sentence."""

    def predict(pairs, **kwargs):
        return [table.get(sentence, 0.0) for _, sentence in pairs]

    model.predict.side_effect = predict


def _chunk(text, chunk_id="c"):
    return SimpleNamespace(id=chunk_id, text=text)


class TestInit:
    def test_load_arguments(self, fake_st):
        module, _ = fake_st
        CrossEncoderSentenceExtractor(
            model="org/ce", revision="abc", device="cpu", max_length=256, nlp=FakeNLP()
        )
        args, kwargs = module.CrossEncoder.call_args
        assert args == ("org/ce",)
        assert kwargs["revision"] == "abc"
        assert kwargs["device"] == "cpu"
        assert kwargs["max_length"] == 256
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)

    def test_default_model(self, fake_st):
        module, _ = fake_st
        CrossEncoderSentenceExtractor(nlp=FakeNLP())
        assert module.CrossEncoder.call_args.args == ("BAAI/bge-reranker-v2-m3",)

    @pytest.mark.parametrize("threshold", [1.5, -0.1])
    def test_invalid_threshold(self, fake_st, threshold):
        with pytest.raises(ValueError, match="threshold"):
            CrossEncoderSentenceExtractor(nlp=FakeNLP(), threshold=threshold)

    def test_pipeline_without_sentence_boundaries(self, fake_st):
        nlp = FakeNLP()
        nlp.pipe_names = ["tok2vec", "tagger"]
        with pytest.raises(ValueError, match="sentence boundaries"):
            CrossEncoderSentenceExtractor(nlp=nlp)

    def test_multi_label_model_is_rejected(self, fake_st):
        _, model = fake_st
        model.num_labels = 3
        with pytest.raises(ValueError, match="3 output labels"):
            CrossEncoderSentenceExtractor(nlp=FakeNLP())


class TestExtractSpans:
    def test_keeps_sentences_at_or_above_threshold(self, fake_st):
        _, model = fake_st
        text = "Első mondat. Második mondat! Harmadik mondat?"
        _scores_by_sentence(
            model, {"Első mondat.": 0.9, "Második mondat!": 0.2, "Harmadik mondat?": 0.5}
        )
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP(), threshold=0.5)

        spans = ex.extract_spans("Kérdés?", [_chunk(text)])

        assert spans == {text: ["Első mondat.", "Harmadik mondat?"]}
        assert all(s in text for s in spans[text])

    def test_pairs_are_question_and_trimmed_sentence(self, fake_st):
        _, model = fake_st
        _scores_by_sentence(model, {})
        CrossEncoderSentenceExtractor(nlp=FakeNLP()).extract_spans(
            "Mennyi a díj?", [_chunk("Egy.  Kettő.\nHárom.")]
        )

        assert model.predict.call_args.args[0] == [
            ("Mennyi a díj?", "Egy."),
            ("Mennyi a díj?", "Kettő."),
            ("Mennyi a díj?", "Három."),
        ]

    def test_one_predict_call_across_chunks(self, fake_st):
        _, model = fake_st
        _scores_by_sentence(model, {"B1.": 0.8})
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP(), batch_size=7)

        spans = ex.extract_spans("Q", [_chunk("A1. A2.", "a"), _chunk("B1. B2.", "b")])

        assert model.predict.call_count == 1
        assert len(model.predict.call_args.args[0]) == 4
        assert model.predict.call_args.kwargs == {
            "batch_size": 7,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        assert spans == {"A1. A2.": [], "B1. B2.": ["B1."]}

    def test_blank_chunks_and_no_results_skip_the_model(self, fake_st):
        _, model = fake_st
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        assert ex.extract_spans("Q", []) == {}
        assert ex.extract_spans("Q", [_chunk("  \n")]) == {"  \n": []}
        model.predict.assert_not_called()

    def test_score_count_mismatch_raises(self, fake_st):
        _, model = fake_st
        model.predict.return_value = [0.9]
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        with pytest.raises(ValueError, match="1 scores for 2 sentences"):
            ex.extract_spans("Q", [_chunk("Egy. Kettő.")])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_scores_raise(self, fake_st, bad):
        _, model = fake_st
        model.predict.return_value = [0.9, bad]
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        with pytest.raises(ValueError, match="1 non-finite"):
            ex.extract_spans("Q", [_chunk("Egy. Kettő.")])

    def test_model_errors_propagate(self, fake_st):
        _, model = fake_st
        model.predict.side_effect = RuntimeError("CUDA out of memory")
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        with pytest.raises(RuntimeError, match="out of memory"):
            ex.extract_spans("Q", [_chunk("Egy.")])


class TestScores:
    def test_duplicate_chunk_texts_are_scored_once(self, fake_st):
        _, model = fake_st
        _scores_by_sentence(model, {"Egy.": 0.7})
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        scores = ex.score_sentences("Q", [_chunk("Egy.", "a"), _chunk("Egy.", "b")])

        assert len(model.predict.call_args.args[0]) == 1
        assert scores == {"Egy.": [("Egy.", 0.7)]}

    def test_repeated_sentence_across_overlapping_chunks_is_scored_once(self, fake_st):
        _, model = fake_st
        _scores_by_sentence(model, {"Közös.": 0.6})
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        scores = ex.score_sentences("Q", [_chunk("Egy. Közös.", "a"), _chunk("Közös. Kettő.", "b")])

        assert [s for _, s in model.predict.call_args.args[0]] == ["Egy.", "Közös.", "Kettő."]
        assert scores["Egy. Közös."][1] == ("Közös.", 0.6)
        assert scores["Közös. Kettő."][0] == ("Közös.", 0.6)

    def test_scores_are_plain_floats(self, fake_st):
        np = pytest.importorskip("numpy")
        _, model = fake_st
        model.predict.return_value = np.array([0.25, 0.75], dtype=np.float32)
        ex = CrossEncoderSentenceExtractor(nlp=FakeNLP())

        scores = ex.score_sentences("Q", [_chunk("Egy. Kettő.")])

        assert scores == {"Egy. Kettő.": [("Egy.", 0.25), ("Kettő.", 0.75)]}
        assert all(type(p) is float for _, p in scores["Egy. Kettő."])

    def test_spans_are_ordered_by_relevance(self):
        scores = {
            "weak": [("w1", 0.55), ("w2", 0.6)],
            "none": [("n1", 0.1)],
            "strong": [("s1", 0.7), ("s2", 0.98), ("s3", 0.7)],
        }
        spans = CrossEncoderSentenceExtractor.spans_from_scores(scores, 0.5)

        assert list(spans) == ["strong", "weak", "none"]
        assert spans == {"strong": ["s2", "s1", "s3"], "weak": ["w2", "w1"], "none": []}

    def test_spans_from_scores_threshold_sweep(self):
        scores = {"t": [("a", 0.3), ("b", 0.6), ("c", 0.9)]}
        assert CrossEncoderSentenceExtractor.spans_from_scores(scores, 0.5) == {"t": ["c", "b"]}
        assert CrossEncoderSentenceExtractor.spans_from_scores(scores, 0.95) == {"t": []}
