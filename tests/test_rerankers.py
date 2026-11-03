"""Tests for JinaV3Reranker and Qwen3Reranker with mocked transformers models (network-free)."""

import math
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
rerankers = pytest.importorskip("verbatim_rag.rerankers")
JinaV3Reranker = rerankers.JinaV3Reranker
Qwen3Reranker = rerankers.Qwen3Reranker
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

    def test_unaccented_question_is_restored_from_the_candidates(self, fake_auto_model):
        _, model = fake_auto_model
        model.rerank.return_value = _ranking([0, 1])
        results = _results(2)
        results[0].text = "A szerződés díja."
        JinaV3Reranker().rerank("szerzodes dija", results)
        assert model.rerank.call_args.args[0] == "szerződés díja"
        JinaV3Reranker(restore_accents=False).rerank("szerzodes dija", results)
        assert model.rerank.call_args.args[0] == "szerzodes dija"

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


class _FakeQwenTokenizer:
    """Whitespace tokenizer with a growing vocabulary; id 0 is padding."""

    unk_token_id = None

    def __init__(self, with_yes_no=True):
        self.vocab = {"yes": 1, "no": 2} if with_yes_no else {}
        self.words = {i: w for w, i in self.vocab.items()}
        self.calls = []

    def _ids(self, text):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 3
                self.words[self.vocab[word]] = word
            ids.append(self.vocab[word])
        return ids

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return self._ids(text)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __call__(self, pairs, **kwargs):
        self.calls.append((list(pairs), kwargs))
        return {"input_ids": [self._ids(p)[: kwargs["max_length"]] for p in pairs]}

    def pad(self, batch, padding, return_tensors):
        import torch

        ids = batch["input_ids"]
        width = max(len(x) for x in ids)
        rows = [[0] * (width - len(x)) + x for x in ids]
        mask = [[0] * (width - len(x)) + [1] * len(x) for x in ids]
        return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(mask)}

    def document(self, row):
        """The document words of one padded prompt (between <Document>: and the suffix)."""
        words = [self.words[i] for i in row if i]
        if "<Document>:" not in words:  # truncated away
            return ""
        start = words.index("<Document>:") + 1
        return " ".join(words[start : words.index("<|im_end|>", start)])


@pytest.fixture
def fake_qwen():
    """(AutoModelForCausalLM, AutoTokenizer, model, tokenizer); the model's yes-logit for a
    document is model.yes_logit[document text] (no-logit 0), so P(yes) = sigmoid of it."""
    torch = pytest.importorskip("torch")
    tokenizer = _FakeQwenTokenizer()
    model = MagicMock()
    model.device = torch.device("cpu")
    model.yes_logit = {}
    model.batches = []

    def forward(input_ids, attention_mask, logits_to_keep, use_cache):
        assert (logits_to_keep, use_cache) == (1, False)
        model.batches.append(input_ids)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 1000)
        for b, row in enumerate(input_ids.tolist()):
            logits[b, -1, 1] = model.yes_logit.get(tokenizer.document(row), 0.0)
        return types.SimpleNamespace(logits=logits)

    model.side_effect = forward
    auto_model = MagicMock()
    auto_model.from_pretrained.return_value = model
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained.return_value = tokenizer
    module = types.ModuleType("transformers")
    module.AutoModelForCausalLM = auto_model
    module.AutoTokenizer = auto_tokenizer
    with patch.dict(sys.modules, {"transformers": module}):
        yield auto_model, auto_tokenizer, model, tokenizer


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TestQwen3Reranker:
    def test_defaults_load_plain_causal_lm(self, fake_qwen):
        auto_model, auto_tokenizer, model, _ = fake_qwen
        reranker = Qwen3Reranker()
        args, kwargs = auto_model.from_pretrained.call_args
        assert args[0] == "Qwen/Qwen3-Reranker-0.6B"
        # No trust_remote_code; float32 on cpu (bf16 is slow there and batch-dependent).
        assert kwargs == {"torch_dtype": sys.modules["torch"].float32, "revision": None}
        args, kwargs = auto_tokenizer.from_pretrained.call_args
        assert args[0] == "Qwen/Qwen3-Reranker-0.6B"
        assert kwargs == {"revision": None, "padding_side": "left"}
        model.eval.assert_called_once()
        model.to.assert_not_called()
        assert reranker.rerank_k == 100
        assert reranker.instruction == Qwen3Reranker.DEFAULT_INSTRUCTION

    def test_revision_pins_model_and_tokenizer(self, fake_qwen):
        auto_model, auto_tokenizer, model, _ = fake_qwen
        Qwen3Reranker(revision="abc123", device="cuda")
        assert auto_model.from_pretrained.call_args.kwargs["revision"] == "abc123"
        assert auto_model.from_pretrained.call_args.kwargs["torch_dtype"] == "auto"
        assert auto_tokenizer.from_pretrained.call_args.kwargs["revision"] == "abc123"
        model.to.assert_called_once_with("cuda")

    def test_prompt_is_prefix_pair_suffix_with_truncation_budget(self, fake_qwen):
        _, _, model, tokenizer = fake_qwen
        reranker = Qwen3Reranker(instruction="Find it", max_length=1000)
        reranker.rerank("kérdés", _results(1))
        ((pairs, kwargs),) = tokenizer.calls
        assert pairs == ["<Instruct>: Find it\n<Query>: kérdés\n<Document>: text 0"]
        prefix = tokenizer.encode(Qwen3Reranker._PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(Qwen3Reranker._SUFFIX, add_special_tokens=False)
        assert kwargs["max_length"] == 1000 - len(prefix) - len(suffix)
        assert kwargs["truncation"] == "longest_first"
        row = model.batches[0][0].tolist()
        assert row == prefix + tokenizer.encode(pairs[0], add_special_tokens=False) + suffix

    def test_score_is_p_yes_against_no(self, fake_qwen):
        _, _, model, _ = fake_qwen
        model.yes_logit = {"a": 2.0, "b": -1.0, "c": 0.0}
        scores = Qwen3Reranker().score("q", ["a", "b", "c"])
        assert scores == pytest.approx([_sigmoid(2.0), _sigmoid(-1.0), 0.5])

    def test_reorders_by_score_and_ties_keep_retrieval_order(self, fake_qwen):
        _, _, model, _ = fake_qwen
        model.yes_logit = {"text 1": 1.0, "text 3": 3.0}  # text 0 and text 2 tie at 0
        out = Qwen3Reranker().rerank("q", _results(4))
        assert [r.id for r in out] == ["c3", "c1", "c0", "c2"]

    def test_tail_beyond_rerank_k_is_kept_in_order(self, fake_qwen):
        _, _, model, _ = fake_qwen
        model.yes_logit = {"text 1": 1.0}
        out = Qwen3Reranker(rerank_k=2).rerank("q", _results(5))
        assert [r.id for r in out] == ["c1", "c0", "c2", "c3", "c4"]
        assert model.batches[0].shape[0] == 2

    def test_batch_size_splits_forward_passes(self, fake_qwen):
        _, _, model, _ = fake_qwen
        model.yes_logit = {"text 4": 1.0, "text 0": -1.0}
        out = Qwen3Reranker(batch_size=2).rerank("q", _results(5))
        assert [b.shape[0] for b in model.batches] == [2, 2, 1]
        assert [r.id for r in out] == ["c4", "c1", "c2", "c3", "c0"]

    def test_enhanced_text_field(self, fake_qwen):
        _, _, _, tokenizer = fake_qwen
        Qwen3Reranker(text_field="enhanced_text").rerank("q", _results(1))
        assert tokenizer.calls[0][0][0].endswith("<Document>: enh 0")

    def test_empty_results_skip_the_model(self, fake_qwen):
        _, _, model, _ = fake_qwen
        assert Qwen3Reranker().rerank("q", []) == []
        assert model.batches == []

    def test_unaccented_question_is_restored_from_the_candidates(self, fake_qwen):
        _, _, _, tokenizer = fake_qwen
        results = _results(2)
        results[0].text = "A szerződés díja."
        Qwen3Reranker().rerank("szerzodes dija", results)
        assert "<Query>: szerződés díja\n" in tokenizer.calls[-1][0][0]
        Qwen3Reranker(restore_accents=False).rerank("szerzodes dija", results)
        assert "<Query>: szerzodes dija\n" in tokenizer.calls[-1][0][0]

    def test_non_finite_scores_raise(self, fake_qwen):
        _, _, model, _ = fake_qwen
        model.yes_logit = {"text 1": float("nan")}
        with pytest.raises(ValueError, match="non-finite"):
            Qwen3Reranker().rerank("q", _results(2))

    def test_checkpoint_without_yes_no_tokens_raises(self, fake_qwen):
        _, auto_tokenizer, _, _ = fake_qwen
        auto_tokenizer.from_pretrained.return_value = _FakeQwenTokenizer(with_yes_no=False)
        with pytest.raises(ValueError, match="yes"):
            Qwen3Reranker()

    def test_prompts_at_max_length_are_logged(self, fake_qwen, caplog):
        with caplog.at_level("WARNING", logger="verbatim_rag.rerankers"):
            Qwen3Reranker(max_length=1000).rerank("q", _results(1))
        assert "max_length" not in caplog.text
        reranker = Qwen3Reranker()
        reranker._pair_budget = 3  # "<Instruct>: Given a" and the rest cut away
        with caplog.at_level("WARNING", logger="verbatim_rag.rerankers"):
            reranker.rerank("q", _results(2))
        assert "2 prompt(s) reached max_length" in caplog.text

    @pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"max_length": 5}, {"instruction": " "}])
    def test_invalid_sizes_raise(self, fake_qwen, kwargs):
        with pytest.raises(ValueError):
            Qwen3Reranker(**kwargs)

    @pytest.mark.parametrize("missing", ["transformers", "torch"])
    def test_missing_dependency_raises_import_error(self, missing):
        with patch.dict(sys.modules, {missing: None}):
            with pytest.raises(ImportError, match="torch transformers"):
                Qwen3Reranker()
