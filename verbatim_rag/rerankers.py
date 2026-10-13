"""
Reranker interfaces and provider adapters for VerbatimRAG.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from verbatim_rag.vector_stores.base import SearchResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        raise NotImplementedError

    async def rerank_async(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        return await asyncio.to_thread(self.rerank, question, results)


class BaseReranker(Reranker):
    """Base helper for rerankers with shared options."""

    def __init__(self, rerank_k: int = 50, text_field: str = "text"):
        self.rerank_k = rerank_k
        self.text_field = text_field

    def _split_results(self, results: List[SearchResult]):
        head = results[: self.rerank_k]
        tail = results[self.rerank_k :]
        return head, tail

    def _get_texts(self, results: List[SearchResult]) -> List[str]:
        if self.text_field == "enhanced_text":
            return [r.enhanced_text or r.text for r in results]
        return [r.text for r in results]


class CohereReranker(BaseReranker):
    """Cohere reranker adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            import cohere
        except ImportError as exc:
            raise ImportError("pip install cohere") from exc
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        response = self.client.rerank(model=self.model, query=question, documents=texts)
        ranked = [head[item.index] for item in response.results]
        return ranked + tail


class JinaReranker(BaseReranker):
    """Jina reranker adapter (HTTP API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "jina-reranker-v1-base-en",
        base_url: str = "https://api.jina.ai/v1/rerank",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        try:
            import requests
        except ImportError as exc:
            raise ImportError("pip install requests") from exc

        payload = {
            "model": self.model,
            "query": question,
            "documents": self._get_texts(head),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ranked = [head[item["index"]] for item in data.get("results", [])]
        return ranked + tail


class SentenceTransformersReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers") from exc
        self.model = CrossEncoder(model, device=device, trust_remote_code=True)

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        pairs = [(question, text) for text in texts]
        scores = self.model.predict(pairs)
        ranked = [r for _, r in sorted(zip(scores, head), reverse=True)]
        return ranked + tail


class JinaV3Reranker(BaseReranker):
    """Local Jina listwise reranker (``jina-reranker-v3.5`` by default).

    The model packs all candidates and the query into one forward pass, and its sliding-window
    layers force an explicit attention mask, so attention memory grows with the square of the
    total length (100 legal chunks is ~12k tokens, ~4.5 GiB per attention matrix in bf16).
    ``max_block_tokens`` caps the length of each pass. Every document is then scored against one
    block-weighted query embedding, so scores share a scale, but each document is embedded in its
    own block's context: the ranking can differ from a single pass.

    The block arithmetic mirrors the remote ``rerank()`` at revision e8a93f33f0b2.
    """

    # max_doc_length in the remote rerank(): a block is closed once its remaining capacity
    # (tokenizer.model_max_length - 2 * query tokens - doc tokens so far) drops to this.
    _MAX_DOC_LENGTH = 8192
    # Queries run up to 1024 tokens; below ~2 * query tokens every document is its own block.
    _MIN_SENSIBLE_BLOCK_TOKENS = 2048

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v3.5",
        rerank_k: int = 100,
        text_field: str = "text",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        max_block_tokens: Optional[int] = None,
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        if max_block_tokens is not None and max_block_tokens <= 0:
            raise ValueError(f"max_block_tokens must be positive, got {max_block_tokens}")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc
        self.model_name = model
        self.revision = revision
        self.max_block_tokens = max_block_tokens
        self.model = AutoModel.from_pretrained(
            model,
            # transformers is pinned to 4.53.3, which predates the `dtype` alias (4.56)
            torch_dtype="auto",
            trust_remote_code=True,
            revision=revision,
        )
        # The remote _ensure_tokenizer() loads the tokenizer from main, ignoring `revision`.
        # Pre-set it (with the same setup) so a pinned run is pinned end to end.
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "left"
        self.model._tokenizer = tokenizer
        if device is not None:
            self.model.to(device)
        self.model.eval()
        if max_block_tokens is not None:
            self._limit_block_tokens(max_block_tokens)

    def _limit_block_tokens(self, max_block_tokens: int) -> None:
        """Approximate cap: the remote packing counts only document and query tokens, so a block
        also carries ~100 prompt tokens, ~12 wrapper tokens per document and may overshoot by its
        last document (<= 8192 tokens). Such a block makes the tokenizer log a harmless
        "sequence length is longer than the specified maximum" warning; nothing is truncated."""
        if not hasattr(self.model, "_ensure_tokenizer"):
            raise RuntimeError(
                f"{self.model_name} has no _ensure_tokenizer(); max_block_tokens depends on "
                "the block packing in jina-reranker-v3.5's remote rerank()"
            )
        if self.revision is None:
            logger.warning(
                "max_block_tokens without a pinned revision: the block arithmetic was checked "
                "against e8a93f33f0b2 and may not match newer remote code"
            )
        if max_block_tokens < self._MIN_SENSIBLE_BLOCK_TOKENS:
            logger.warning(
                "max_block_tokens=%d: long queries will put each document in its own block "
                "(one forward pass per document, no listwise context)",
                max_block_tokens,
            )
        # Only rerank()'s capacity check reads this; truncation uses explicit max_lengths.
        self.model._tokenizer.model_max_length = max_block_tokens + self._MAX_DOC_LENGTH

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        ranked = self.model.rerank(question, self._get_texts(head))
        # Sort explicitly: the model card does not state that rerank() output is pre-sorted.
        ranked = sorted(ranked, key=lambda item: item["relevance_score"], reverse=True)
        order = [item["index"] for item in ranked]
        if sorted(order) != list(range(len(head))):
            raise ValueError(
                f"Jina reranker returned {len(order)} indices for {len(head)} documents"
            )
        return [head[i] for i in order] + tail
