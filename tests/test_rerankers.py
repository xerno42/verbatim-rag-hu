"""Tests for JinaV3Reranker with a mocked transformers AutoModel (network-free)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
rerankers = pytest.importorskip("verbatim_rag.rerankers")
JinaV3Reranker = rerankers.JinaV3Reranker
SearchResult = pytest.importorskip("verbatim_rag.vector_stores.base").SearchResult


def _results(n):
    return [
        SearchResult(id=f"c{i}", score=1.0, metadata={}, text=f"text {i}", enhanced_text=f"enh {i}")
        for i in range(n)
    ]


def _ranking(order):
    """What the Jina model returns: dicts sorted by relevance, with the input index."""
    return [
        {"document": "", "relevance_score": 1.0 - 0.1 * r, "index": i} for r, i in enumerate(order)
    ]


@pytest.fixture
def fake_auto_model():
    model = MagicMock()
    auto = MagicMock()
    auto.from_pretrained.return_value = model
    module = types.ModuleType("transformers")
    module.AutoModel = auto
    module.AutoTokenizer = MagicMock()
    with patch.dict(sys.modules, {"transformers": module}):
        yield auto, model


@pytest.fixture
def fake_tokenizer(fake_auto_model):
    tokenizer = MagicMock(pad_token="<pad>", model_max_length=131072)
    sys.modules["transformers"].AutoTokenizer.from_pretrained.return_value = tokenizer
    return sys.modules["transformers"].AutoTokenizer, tokenizer


class TestJinaV3Reranker:
    def test_defaults_to_v35_with_remote_code(self, fake_auto_model):
        auto, model = fake_auto_model
        JinaV3Reranker()
        args, kwargs = auto.from_pretrained.call_args
        assert args[0] == "jinaai/jina-reranker-v3.5"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["revision"] is None
        # The pinned transformers (4.53.3) forwards an unknown `dtype` kwarg to the model __init__
        assert kwargs["torch_dtype"] == "auto"
        assert "dtype" not in kwargs
        model.eval.assert_called_once()
        model.to.assert_not_called()

    def test_revision_and_device_passed_through(self, fake_auto_model):
        auto, model = fake_auto_model
        JinaV3Reranker(revision="abc123", device="cuda")
        assert auto.from_pretrained.call_args.kwargs["revision"] == "abc123"
        model.to.assert_called_once_with("cuda")

    def test_tokenizer_is_pinned_to_the_revision(self, fake_auto_model, fake_tokenizer):
        _, model = fake_auto_model
        auto_tokenizer, tokenizer = fake_tokenizer
        JinaV3Reranker(revision="abc123")
        args, kwargs = auto_tokenizer.from_pretrained.call_args
        assert args[0] == "jinaai/jina-reranker-v3.5"
        assert kwargs["revision"] == "abc123"
        assert model._tokenizer is tokenizer
        assert tokenizer.padding_side == "left"
        assert tokenizer.model_max_length == 131072

    def test_missing_pad_token_falls_back_to_unk(self, fake_auto_model, fake_tokenizer):
        _, tokenizer = fake_tokenizer
        tokenizer.pad_token = None
        tokenizer.unk_token = "<unk>"
        JinaV3Reranker()
        assert tokenizer.pad_token == "<unk>"
        tokenizer.convert_tokens_to_ids.assert_called_once_with("<unk>")

    def test_max_block_tokens_sets_the_packing_capacity(self, fake_auto_model, fake_tokenizer):
        _, tokenizer = fake_tokenizer
        JinaV3Reranker(revision="abc123", max_block_tokens=4096)
        assert tokenizer.model_max_length == 4096 + JinaV3Reranker._MAX_DOC_LENGTH

    def test_max_block_tokens_warns_when_unpinned_or_tiny(self, fake_auto_model, caplog):
        with caplog.at_level("WARNING", logger="verbatim_rag.rerankers"):
            JinaV3Reranker(max_block_tokens=512)
        assert "pinned revision" in caplog.text
        assert "own block" in caplog.text

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_max_block_tokens_raises(self, fake_auto_model, value):
        with pytest.raises(ValueError, match="max_block_tokens"):
            JinaV3Reranker(max_block_tokens=value)

    def test_max_block_tokens_needs_the_jina_remote_code(self, fake_auto_model):
        _, model = fake_auto_model
        del model._ensure_tokenizer
        with pytest.raises(RuntimeError, match="_ensure_tokenizer"):
            JinaV3Reranker(max_block_tokens=4096)

    def test_default_rerank_k_is_100(self, fake_auto_model):
        assert JinaV3Reranker().rerank_k == 100

    def test_reorders_by_model_ranking(self, fake_auto_model):
        _, model = fake_auto_model
        model.rerank.return_value = _ranking([2, 0, 1])
        out = JinaV3Reranker().rerank("kérdés", _results(3))
        assert [r.id for r in out] == ["c2", "c0", "c1"]
        assert model.rerank.call_args.args == ("kérdés", ["text 0", "text 1", "text 2"])

    def test_sorts_by_score_when_model_output_is_unsorted(self, fake_auto_model):
        _, model = fake_auto_model
        model.rerank.return_value = [
            {"document": "", "relevance_score": 0.1, "index": 0},
            {"document": "", "relevance_score": 0.9, "index": 1},
            {"document": "", "relevance_score": 0.5, "index": 2},
        ]
        out = JinaV3Reranker().rerank("q", _results(3))
        assert [r.id for r in out] == ["c1", "c2", "c0"]

    def test_tail_beyond_rerank_k_is_kept_in_order(self, fake_auto_model):
        _, model = fake_auto_model
        model.rerank.return_value = _ranking([1, 0])
        out = JinaV3Reranker(rerank_k=2).rerank("q", _results(5))
        assert [r.id for r in out] == ["c1", "c0", "c2", "c3", "c4"]
        assert len(model.rerank.call_args.args[1]) == 2

    def test_enhanced_text_field(self, fake_auto_model):
        _, model = fake_auto_model
        model.rerank.return_value = _ranking([0])
        JinaV3Reranker(text_field="enhanced_text").rerank("q", _results(1))
        assert model.rerank.call_args.args[1] == ["enh 0"]

    def test_empty_results_skip_the_model(self, fake_auto_model):
        _, model = fake_auto_model
        assert JinaV3Reranker().rerank("q", []) == []
        model.rerank.assert_not_called()

    @pytest.mark.parametrize("order", [[0], [0, 0, 1], [0, 1, 5]])
    def test_malformed_model_output_raises(self, fake_auto_model, order):
        # VerbatimRAG._apply_reranker catches this and falls back to the retrieval order.
        _, model = fake_auto_model
        model.rerank.return_value = _ranking(order)
        with pytest.raises(ValueError, match="indices"):
            JinaV3Reranker().rerank("q", _results(3))

    def test_missing_transformers_raises_import_error(self):
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                JinaV3Reranker()
