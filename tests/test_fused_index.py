"""Tests for FusedIndex with fake indexes (network-free)."""

import pytest

# verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
fused_index = pytest.importorskip("verbatim_rag.fused_index")
FusedIndex = fused_index.FusedIndex
SearchResult = pytest.importorskip("verbatim_rag.vector_stores.base").SearchResult
merge_hybrid_results = pytest.importorskip(
    "verbatim_rag.vector_stores.hybrid_search"
).merge_hybrid_results


def _hits(ids):
    return [SearchResult(id=i, score=0.0, metadata={}, text=f"text {i}") for i in ids]


class FakeIndex:
    """Returns a fixed list per search_type and records every call."""

    def __init__(self, by_method, dense=True, sparse=True):
        self.by_method = by_method
        self.calls = []
        self.vector_store = object()
        self.dense_provider = object() if dense else None
        self.sparse_provider = object() if sparse else None

    def query(self, text=None, k=5, search_type="auto", **kwargs):
        self.calls.append({"text": text, "k": k, "search_type": search_type, **kwargs})
        return _hits(self.by_method.get(search_type, []))[:k]


def make(main=None, folded=None):
    if main is None:
        main = {"dense": ["a", "b", "c"], "sparse": ["b", "c", "d"]}
    index = FakeIndex(main)
    channel = FakeIndex({"sparse": folded if folded is not None else ["d", "e"]}, dense=False)
    return FusedIndex(index, {"folded": channel}), index, channel


class TestFusedIndex:
    def test_without_channel_weight_passes_through_unchanged(self):
        fused, index, channel = make()
        weights = {"dense": 0.5, "sparse": 0.5, "folded": 0}
        fused.query("q", k=3, hybrid_weights=weights, rrf_k=10)
        assert index.calls == [
            {
                "text": "q",
                "k": 3,
                "search_type": "auto",
                "filter": None,
                "search_params": None,
                "hybrid_weights": {"dense": 0.5, "sparse": 0.5},
                "rrf_k": 10,
            }
        ]
        assert channel.calls == []

    def test_fuses_like_the_store(self):
        fused, index, channel = make()
        weights = {"dense": 0.5, "sparse": 0.25, "folded": 0.25}
        out = fused.query("q", k=4, hybrid_weights=weights, rrf_k=60)

        lists = {
            "dense": [{"id": i} for i in ["a", "b", "c"]],
            "sparse": [{"id": i} for i in ["b", "c", "d"]],
            "folded": [{"id": i} for i in ["d", "e"]],
        }
        expected = merge_hybrid_results(lists, 4, weights, 60)
        assert [r.id for r in out] == [h["id"] for h in expected]
        assert [r.score for r in out] == pytest.approx([h["distance"] for h in expected])
        assert out[0].text == f"text {out[0].id}"

    def test_each_leg_fetches_twice_k_with_its_search_type(self):
        fused, index, channel = make()
        fused.query("q", k=5, filter="x", hybrid_weights={"dense": 1, "sparse": 1, "folded": 1})
        assert [(c["search_type"], c["k"], c["filter"]) for c in index.calls] == [
            ("dense", 10, "x"),
            ("sparse", 10, "x"),
        ]
        assert [(c["search_type"], c["k"]) for c in channel.calls] == [("sparse", 10)]

    def test_empty_channel_is_left_out(self):
        fused, _, _ = make(folded=[])
        out = fused.query("q", k=3, hybrid_weights={"dense": 0.5, "sparse": 0.5, "folded": 0.5})
        assert [r.id for r in out] == ["b", "c", "a"]

    def test_no_results_anywhere(self):
        fused, _, _ = make(main={}, folded=[])
        assert fused.query("q", k=3, hybrid_weights={"dense": 1, "folded": 1}) == []

    def test_unusable_weights_are_dropped_like_the_store(self, caplog):
        fused, index, channel = make()
        weights = {"folded": 1, "bm25": 1, "full_text": 1, "dense": "x", "sparse": 0}
        out = fused.query("q", k=2, hybrid_weights=weights)
        assert [r.id for r in out] == ["d", "e"]
        assert index.calls == [] and len(channel.calls) == 1
        assert "unsupported hybrid method 'bm25'" in caplog.text
        assert "non-positive weight for method 'dense'" in caplog.text

    def test_base_method_without_provider_is_skipped(self):
        main = FakeIndex({"dense": ["a"], "sparse": ["b"]}, sparse=False)
        fused = FusedIndex(main, {"folded": FakeIndex({"sparse": ["c"]}, dense=False)})
        fused.query("q", k=2, hybrid_weights={"dense": 1, "sparse": 1, "folded": 1})
        assert [c["search_type"] for c in main.calls] == ["dense"]

    def test_fixed_method_order_whatever_the_dict_order(self):
        fused, index, channel = make()
        a = fused.query("q", k=4, hybrid_weights={"folded": 1, "sparse": 1, "dense": 1})
        b = fused.query("q", k=4, hybrid_weights={"dense": 1, "sparse": 1, "folded": 1})
        assert [r.id for r in a] == [r.id for r in b]
        assert [c["search_type"] for c in index.calls[:2]] == ["dense", "sparse"]

    def test_channel_name_clash_raises(self):
        with pytest.raises(ValueError, match="clash"):
            FusedIndex(FakeIndex({}), {"sparse": FakeIndex({})})

    def test_channel_without_sparse_provider_raises(self):
        with pytest.raises(ValueError, match="sparse provider"):
            FusedIndex(FakeIndex({}), {"folded": FakeIndex({}, sparse=False)})

    def test_vector_store_passthrough(self):
        fused, index, _ = make()
        assert fused.vector_store is index.vector_store

    def test_score_is_one_minus_fused_rrf(self):
        fused, _, _ = make(main={"dense": ["a"]}, folded=["a"])
        (out,) = fused.query("q", k=1, hybrid_weights={"dense": 1, "folded": 1})
        assert out.score == pytest.approx(1 - 1 / 61)
