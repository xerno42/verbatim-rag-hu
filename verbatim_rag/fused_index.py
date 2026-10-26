"""
RRF fusion of a VerbatimIndex with extra sparse-only indexes.

A Milvus collection's schema is fixed and LocalMilvusStore has one sparse field, so a second
sparse channel (accent-folded BM25) lives in its own Milvus Lite db with its own
VerbatimIndex. FusedIndex queries every weighted channel separately and fuses the lists with
the store's own weighted RRF (merge_hybrid_results), so VerbatimIndex and LocalMilvusStore
stay as shipped.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

from verbatim_rag.vector_stores.base import SearchResult
from verbatim_rag.vector_stores.hybrid_search import merge_hybrid_results

logger = logging.getLogger(__name__)

_BASE_METHODS = ("dense", "sparse")


class FusedIndex:
    """Stands in for VerbatimIndex: query() plus vector_store.

    channels maps a method name (e.g. "folded") to a sparse-only VerbatimIndex over the same
    chunk ids. A query whose hybrid_weights give no channel a positive weight goes to the main
    index unchanged, so its results are identical to the plain index's. Otherwise each
    weighted method fetches 2 * k results, as BaseMilvusStore's hybrid search does, and the
    lists are fused with the same weighted RRF, base methods first (dense, sparse) and then the
    channels. As in the store, non-positive or non-numeric weights and unknown methods
    (full_text included: Milvus Lite has none) are dropped with a warning, and so is a base
    method the main index has no provider for. A method with no results (an empty sparse
    query vector returns none) is left out; with normalized weights that does not change the
    order of the rest.
    """

    def __init__(self, index, channels: Dict[str, Any]):
        clash = set(channels) & set(_BASE_METHODS + ("full_text",))
        if clash:
            raise ValueError(f"channel names clash with index methods: {sorted(clash)}")
        missing = [
            name for name, ch in channels.items() if not getattr(ch, "sparse_provider", None)
        ]
        if missing:
            raise ValueError(f"channels without a sparse provider: {sorted(missing)}")
        self.index = index
        self.channels = dict(channels)

    @property
    def vector_store(self):
        return self.index.vector_store

    def query(
        self,
        text: Optional[str] = None,
        k: int = 5,
        search_type: str = "auto",
        filter: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        weights = self._usable_weights(hybrid_weights or {})
        if not text or not any(m in self.channels for m in weights):
            base = None
            if hybrid_weights is not None:
                base = {m: w for m, w in hybrid_weights.items() if m not in self.channels}
            return self.index.query(
                text,
                k=k,
                search_type=search_type,
                filter=filter,
                search_params=search_params,
                hybrid_weights=base,
                rrf_k=rrf_k,
            )
        lists: Dict[str, List[Dict[str, Any]]] = {}
        for method in weights:
            index = self.channels.get(method, self.index)
            hits = index.query(
                text,
                k=2 * k,
                search_type="sparse" if method in self.channels else method,
                filter=filter,
                search_params=search_params,
            )
            if hits:
                lists[method] = [{"id": hit.id, "result": hit} for hit in hits]
        if not lists:
            return []
        merged = merge_hybrid_results(lists, k, weights, rrf_k, log_label=type(self).__name__)
        out = []
        for hit in merged:
            result = copy.copy(hit["result"])
            result.score = hit["distance"]  # as convert_hits_to_results: 1 - fused RRF score
            out.append(result)
        return out

    def _usable_weights(self, hybrid_weights: Dict[str, float]) -> Dict[str, float]:
        """Positive numeric weights of the methods this index can run, in fixed order (the
        order breaks RRF ties)."""
        usable = {}
        for method in (*_BASE_METHODS, *self.channels):
            if method not in hybrid_weights:
                continue
            weight = hybrid_weights[method]
            if not isinstance(weight, (int, float)) or weight <= 0:
                logger.warning("Ignoring non-positive weight for method '%s': %s", method, weight)
            elif method in _BASE_METHODS and not getattr(self.index, f"{method}_provider", None):
                logger.warning("Ignoring '%s': the index has no %s provider", method, method)
            else:
                usable[method] = float(weight)
        for method in hybrid_weights:
            if method not in usable and method not in _BASE_METHODS and method not in self.channels:
                logger.warning("Ignoring unsupported hybrid method '%s'", method)
        return usable
