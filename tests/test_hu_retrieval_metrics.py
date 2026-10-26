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
