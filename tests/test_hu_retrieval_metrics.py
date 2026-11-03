"""Metric helpers of the Hungarian retrieval eval runner (network-free)."""

import json

import pytest

from evaluation.hu_legal.run_retrieval import QueryCache, breakdown, first_hit_rank, summarize


class TestFirstHitRank:
    def test_first_gold_wins(self):
        assert first_hit_rank(["a", "g2", "g1"], ["g1", "g2"]) == 2

    def test_miss(self):
        assert first_hit_rank(["a", "b"], ["g"]) is None

    def test_empty_results(self):
        assert first_hit_rank([], ["g"]) is None


class TestSummarize:
    def test_recall_and_mrr(self):
        s = summarize([1, 4, 50, None])
        assert s["n"] == 4
        assert s["recall@10"] == pytest.approx(0.5)
        assert s["recall@100"] == pytest.approx(0.75)
        assert s["mrr@100"] == pytest.approx((1 + 1 / 4 + 1 / 50) / 4)

    def test_rank_beyond_top_k_is_a_miss(self):
        assert summarize([101])["recall@100"] == 0

    def test_empty(self):
        assert summarize([]) == {"n": 0}


def test_breakdown_groups_by_field():
    rows = [
        {"difficulty": "easy", "ranks": {"dense": 1}},
        {"difficulty": "hard", "ranks": {"dense": None}},
        {"difficulty": "easy", "ranks": {"dense": 20}},
    ]
    out = breakdown(rows, "dense", "difficulty")
    assert out["easy"]["n"] == 2 and out["easy"]["recall@10"] == 0.5
    assert out["hard"]["recall@100"] == 0


def test_query_cache_embeds_once_and_counts_empty_sparse():
    calls = []

    class Provider:
        def embed_text(self, text):
            calls.append(text)
            return {}

    cache = QueryCache(Provider())
    cache.embed_text("q")
    cache.embed_text("q")
    assert calls == ["q"]
    assert cache.empty == 1


class _FakeIndex:
    dense_provider = sparse_provider = object()

    def __init__(self, ids):
        self.ids = ids

    def query(self, text, k, **kwargs):
        from verbatim_rag.vector_stores.base import SearchResult

        return [SearchResult(id=i, score=1.0, metadata={}, text=i) for i in self.ids]


class _ReverseReranker:
    def rerank(self, question, results):
        return list(reversed(results))


def test_rerank_source_prefers_fused():
    from evaluation.hu_legal.run_retrieval import rerank_source

    assert rerank_source(["sparse", "dense", "fused"]) == "fused"
    assert rerank_source(["dense"]) == "dense"
    assert rerank_source(["sparse", "dense", "folded", "fused", "fused_folded"]) == "fused_folded"


def test_run_queries_ranks_reranked_list_from_the_fused_hits():
    pytest.importorskip("verbatim_rag.vector_stores.base")
    from types import SimpleNamespace

    from evaluation.hu_legal.run_retrieval import run_queries

    args = SimpleNamespace(dense_weight=0.5, sparse_weight=0.5, rrf_k=60)
    queries = [{"query": "q", "gold_chunk_ids": ["c"], "difficulty": "easy"}]
    index = _FakeIndex(["a", "b", "c"])

    rows = run_queries(index, queries, ["dense", "fused"], args, _ReverseReranker())
    assert rows[0]["ranks"] == {"dense": 3, "fused": 3, "reranked": 1}

    rows = run_queries(index, queries, ["dense", "fused"], args)
    assert "reranked" not in rows[0]["ranks"]


def test_run_queries_with_folded_reranks_the_three_way_fusion():
    pytest.importorskip("verbatim_rag.fused_index")
    from types import SimpleNamespace

    from evaluation.hu_legal.run_retrieval import run_queries

    args = SimpleNamespace(
        dense_weight=0.5, sparse_weight=0.5, folded=True, folded_weight=0.5, rrf_k=60
    )
    queries = [{"query": "q", "gold_chunk_ids": ["c"], "difficulty": "easy"}]
    methods = ["dense", "folded", "fused", "fused_folded"]
    rows = run_queries(
        _FakeIndex(["a", "b", "c"]), queries, methods, args, _ReverseReranker(), _FakeIndex(["c"])
    )
    ranks = rows[0]["ranks"]
    assert ranks["folded"] == 1
    assert ranks["fused"] == 3
    assert ranks["fused_folded"] == 1  # c 2/63 + 1/61, a 2/61, b 2/62
    assert ranks["reranked"] == 3  # the reversed fused_folded list: b, a, c


def test_folded_meta_check(tmp_path):
    pytest.importorskip("verbatim_rag.hungarian_bm25")
    from types import SimpleNamespace

    from evaluation.hu_legal.run_retrieval import (
        FOLDED_DB,
        FOLDED_META,
        check_folded_meta,
        folded_meta,
    )

    args = SimpleNamespace(bm25_model="hu_core_news_lg", k1=1.2, b=0.75)
    corpus = [{"chunk_id": "c1", "text": "szöveg"}]
    expected = folded_meta(corpus, args)
    assert check_folded_meta(tmp_path, expected) is False

    (tmp_path / FOLDED_META).write_text(json.dumps(expected), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        check_folded_meta(tmp_path, expected)
    (tmp_path / FOLDED_DB).write_text("db")
    assert check_folded_meta(tmp_path, expected) is True
    with pytest.raises(ValueError, match="does not match"):
        check_folded_meta(tmp_path, folded_meta(corpus, SimpleNamespace(**{**vars(args), "k1": 2})))


class TestRerankArgs:
    def parse(self, *argv, enabled=True):
        import argparse

        from evaluation.hu_legal.run_retrieval import add_rerank_args, check_rerank_args

        p = argparse.ArgumentParser()
        add_rerank_args(p)
        args = p.parse_args(list(argv))
        check_rerank_args(p, args, enabled=enabled)
        return args

    def test_model_defaults_follow_the_reranker(self):
        args = self.parse()
        assert (args.reranker, args.rerank_model) == ("jina", "jinaai/jina-reranker-v3.5")
        assert self.parse("--reranker", "qwen3").rerank_model == "Qwen/Qwen3-Reranker-0.6B"
        args = self.parse("--reranker", "qwen3", "--rerank-model", "my/model")
        assert args.rerank_model == "my/model"

    @pytest.mark.parametrize(
        "argv",
        [
            ["--reranker", "qwen3", "--rerank-block-tokens", "4096"],
            ["--rerank-batch-size", "8"],
            ["--rerank-instruction", "x"],
            ["--reranker", "qwen3", "--rerank-batch-size", "0"],
            ["--reranker", "qwen3", "--rerank-instruction", " "],
        ],
    )
    def test_flags_of_the_other_reranker_are_rejected(self, argv):
        with pytest.raises(SystemExit):
            self.parse(*argv)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--rerank-block-tokens", "4096"],
            ["--reranker", "qwen3", "--rerank-batch-size", "8"],
            ["--reranker", "qwen3", "--rerank-instruction", "x"],
            ["--reranker", "qwen3"],
            ["--reranker", "jina"],
            ["--rerank-model", "my/model"],
            ["--rerank-revision", "abc"],
        ],
    )
    def test_reranker_settings_need_reranking(self, argv):
        self.parse(*argv)
        with pytest.raises(SystemExit):
            self.parse(*argv, enabled=False)

    def test_make_reranker_dispatches_and_logs(self, monkeypatch):
        rerankers = pytest.importorskip("verbatim_rag.rerankers")
        from unittest.mock import MagicMock

        from evaluation.hu_legal.run_retrieval import make_reranker, rerank_config

        jina, qwen = MagicMock(), MagicMock()
        monkeypatch.setattr(rerankers, "JinaV3Reranker", jina)
        monkeypatch.setattr(rerankers, "Qwen3Reranker", qwen)

        args = self.parse("--rerank-block-tokens", "4096")
        args.device, args.no_accent_restore = "cuda", False
        make_reranker(args, 100)
        assert jina.call_args.kwargs == {
            "model": "jinaai/jina-reranker-v3.5",
            "rerank_k": 100,
            "device": "cuda",
            "revision": None,
            "max_block_tokens": 4096,
            "restore_accents": True,
        }
        assert rerank_config(args, jina.return_value, 100)["max_block_tokens"] == 4096
        qwen.assert_not_called()

        args = self.parse("--reranker", "qwen3", "--rerank-instruction", "Find it")
        args.device, args.no_accent_restore = "cpu", True
        reranker = make_reranker(args, 50)
        assert qwen.call_args.kwargs == {
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "rerank_k": 50,
            "device": "cpu",
            "revision": None,
            "restore_accents": False,
            "instruction": "Find it",  # batch_size unset: the class default applies
        }
        reranker.batch_size, reranker.max_length, reranker.instruction = 16, 8192, "Find it"
        reranker.model.dtype = "torch.float32"
        config = rerank_config(args, reranker, 50)
        assert config["reranker"] == "qwen3"
        assert (config["batch_size"], config["instruction"]) == (16, "Find it")
        assert config["dtype"] == "torch.float32"
        assert "max_block_tokens" not in config

        args = self.parse("--reranker", "qwen3", "--rerank-batch-size", "4")
        args.device, args.no_accent_restore = "cpu", False
        make_reranker(args, 50)
        assert qwen.call_args.kwargs["batch_size"] == 4
        assert "instruction" not in qwen.call_args.kwargs
