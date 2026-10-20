"""
Local demo of the Hungarian pipeline on the eval set, served to the React UI in frontend/.

    python -m evaluation.hu_legal.demo --index-tag baseline
    DEEPSEEK_API_KEY=... python -m evaluation.hu_legal.demo --index-tag baseline \
        --extractors crossencoder,llm                # adds the LLM extractor to the toggle
    cd frontend && npm install && npm run dev     # http://localhost:3000, proxies /api to :8000

The pipeline is the one `run_extraction --mode retrieved --rerank` measures: fused Hungarian
BM25 + Qwen3 retrieval of --retrieve-k chunks, Jina v3.5 rerank, extraction on the top
--extract-k chunks, and a static Hungarian template (no LLM writes framing text). Defaults
fit a 16 GB GPU (--device cuda --batch-size 32 --rerank-block-tokens 4096).

Each request picks one of the loaded extractors (the UI has a toggle):

- crossencoder (default): CrossEncoderSentenceExtractor, sentences kept at score >=
  --ce-threshold.
- llm (opt-in): LLMSpanExtractor through an OpenAI-compatible --api-base, temperature 0, one
  batched prompt per query (as in the eval). DeepSeek V4 Flash by default; the key is read
  from the environment variable named by --llm-key-env, never from OPENAI_API_KEY. One
  preflight call at startup fails early on a bad key, model or endpoint. Each query sends
  its top --extract-k corpus chunks to that API. No extraction run with this model is
  logged yet, so the UI's verdicts for it are not backed by eval numbers.

Differences from the generic API (api/app.py), which this server replaces for the demo:

- Request-side k, hybrid weights and RRF k are ignored: every query runs the measured config.
- Reranker errors and lost LLM calls are shown as errors. VerbatimRAG would fall back to the
  fused order, and LLMSpanExtractor returns no spans, which would read as an abstention.
- Every span must be an exact substring of its chunk (the spec's verbatim assertion);
  others are dropped and logged, never shown. The answer text is not whitespace-normalized,
  so spans stay exact there too (the UI's markdown rendering may still reflow them). No
  spans gives a fixed Hungarian abstention.
- /api/demo/queries serves the split's queries with gold chunk ids, so the UI can pick an
  eval query and mark gold chunks. Queries run one at a time (one GPU). StreamingRAG calls
  retrieval synchronously, so the server is unresponsive for the seconds a query's
  retrieval and rerank take. LLM calls time out after 60 s (one retry), so an extraction
  left running by a closed tab holds up the next query at most that long.

The index in data/index/<tag>/ is reused when its index_meta.json matches the corpus and
provider settings (the run_extraction --index-tag check). Without a completed index, --build
builds one there (the Qwen3 pass takes hours on CPU). Corpus and queries are CC BY-NC-ND 4.0:
the server binds to 127.0.0.1 by default; don't expose it.
"""

import argparse
import asyncio
import copy
import ipaddress
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from verbatim_core.extractors import SpanExtractor
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates import StaticTemplate, TemplateManager

from evaluation.hu_legal.run_extraction import LogCounter, check_index_meta, open_index
from evaluation.hu_legal.run_retrieval import (
    DATA,
    INDEX_META,
    QueryCache,
    index_meta,
    load_split,
    read_jsonl,
)

logger = logging.getLogger("hu_legal.demo")

EXTRACTORS = ("crossencoder", "llm")
ABSTAIN_HU = "A rendelkezésre álló dokumentumok nem tartalmaznak választ erre a kérdésre."
TEMPLATE_HU = "## Válasz\n\nSzó szerinti részletek a forrásdokumentumokból:\n\n[DISPLAY_SPANS]"
# The revision the baseline retrieval run used; JinaV3Reranker's block arithmetic mirrors it.
RERANK_REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
MAX_QUESTION_CHARS = 1000
# What /api/demo/queries exposes per query: no passage text, no source chunk id.
QUERY_FIELDS = (
    "qid",
    "query",
    "answerable",
    "unanswerable_reason",
    "gold_chunk_ids",
    "query_type",
    "legal_domain",
    "difficulty",
    "bucket",
)
_TAG_RE = re.compile(r"[\w.-]+")

# Held while the GPU models run (retrieval + rerank, extraction). The asyncio lock in the
# app queues requests; this one also covers a worker thread that outlives its request: a
# client disconnect cancels the stream, but not the asyncio.to_thread extraction call.
MODEL_LOCK = threading.Lock()
# openai's default is 600 s with 2 retries; a hung call would hold MODEL_LOCK that long.
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 1


def wait_for_models() -> None:
    with MODEL_LOCK:
        pass


# ---- pipeline pieces (network-free, tested) --------------------------------


class HungarianStaticTemplate(StaticTemplate):
    """Static template whose answer without spans is the fixed Hungarian abstention message
    (the stock filler writes an English one)."""

    def fill(self, template, display_spans, citation_spans) -> str:
        if not any(span.get("text", "").strip() for span in display_spans):
            return ABSTAIN_HU
        return super().fill(template, display_spans, citation_spans)


def hungarian_template_manager() -> TemplateManager:
    manager = TemplateManager(default_mode="static")
    manager.strategies["static"] = HungarianStaticTemplate(TEMPLATE_HU)
    return manager


class UncleanedResponseBuilder(ResponseBuilder):
    """Leaves the filled template as is. ResponseBuilder.clean_answer targets LLM-written
    framing (quotes, escaped newlines, runs of spaces), but here the only non-template text
    is the spans, and collapsing double spaces (common in the corpus) would make them stop
    being exact substrings of their chunk."""

    def clean_answer(self, answer: str) -> str:
        return answer or ""


class RerankedRetriever:
    """Stands in for VerbatimIndex in VerbatimRAG: fused top retrieve_k, rerank, top k.

    The fusion weights and RRF k are fixed here, so request-side values are ignored and the
    demo runs the measured configuration. Reranker errors propagate (VerbatimRAG's own
    reranker hook would log them and keep the fused order). Results carry their chunk id
    and document id in metadata for the UI.
    """

    def __init__(self, index, reranker, retrieve_k: int, hybrid_weights: Dict[str, float], rrf_k):
        if retrieve_k < 1:
            raise ValueError(f"retrieve_k must be >= 1, got {retrieve_k}")
        self.index = index
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.hybrid_weights = dict(hybrid_weights)
        self.rrf_k = rrf_k

    @property
    def vector_store(self):
        return self.index.vector_store

    def query(
        self,
        text: str,
        k: int = 5,
        filter: Optional[str] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,  # ignored, see class docstring
        rrf_k: Optional[int] = None,  # ignored
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        with MODEL_LOCK:
            hits = self.index.query(
                text,
                k=max(self.retrieve_k, k),
                filter=filter,
                hybrid_weights=dict(self.hybrid_weights),
                rrf_k=self.rrf_k,
                search_params=search_params,
            )
            if self.reranker is not None:
                hits = self.reranker.rerank(text, hits)
        return [self._for_display(hit) for hit in hits[:k]]

    @staticmethod
    def _for_display(hit):
        shown = copy.copy(hit)
        metadata = dict(hit.metadata or {})
        metadata["chunk_id"] = hit.id
        metadata.setdefault("title", metadata.get("document_id") or hit.id)
        shown.metadata = metadata
        return shown


class CheckedExtractor(SpanExtractor):
    """Wraps an extractor so that what reaches the UI is verbatim and errors are not silent.

    - Spans that are blank, not exact substrings of their chunk, or keyed by a chunk that
      was not passed in are dropped and logged (the spec's verbatim assertion). With the
      static template and UncleanedResponseBuilder the answer text carries them unchanged.
    - LLMSpanExtractor logs a failed call and returns no spans for it; any such log line
      during the call raises instead, so the UI shows an error, not an abstention. Only
      records from the calling thread count (the sync LLMSpanExtractor logs there), so a
      worker thread left over from a cancelled request cannot fail this one. This covers
      the sync extract_spans only (what StreamingRAG calls); extract_spans_async would log
      from the event loop thread and is not wrapped.
    """

    def __init__(self, inner: SpanExtractor):
        self.inner = inner

    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        counter = LogCounter()
        thread = threading.get_ident()
        counter.addFilter(lambda record: record.thread == thread)
        core_logger = logging.getLogger("verbatim_core")
        with MODEL_LOCK:
            core_logger.addHandler(counter)
            try:
                spans = self.inner.extract_spans(question, search_results)
            finally:
                core_logger.removeHandler(counter)
        if counter.errors:
            raise RuntimeError(
                f"{counter.errors} extraction call(s) failed; see the server log for the cause"
            )
        texts = {getattr(r, "text", "") or "" for r in search_results}
        checked: Dict[str, List[str]] = {}
        for text, found in spans.items():
            if text not in texts:
                logger.error("extractor returned %d span(s) for a chunk not passed in", len(found))
                continue
            kept = [span for span in found if span.strip() and span in text]
            if len(kept) < len(found):
                logger.error("verbatim check dropped %d span(s)", len(found) - len(kept))
            checked[text] = kept
        return checked


def query_view(query: Dict[str, Any]) -> Dict[str, Any]:
    return {field: query.get(field) for field in QUERY_FIELDS}


# ---- server ----------------------------------------------------------------


class StreamRequest(BaseModel):
    """What the UI sends; its num_docs and other fields are ignored."""

    question: str
    extractor: Optional[str] = None


def create_app(
    pipelines: Dict[str, Any],
    default_extractor: str,
    queries: List[Dict[str, Any]],
    info: Dict[str, Any],
):
    """FastAPI app with the endpoints the frontend uses plus /api/demo/*.

    pipelines maps extractor name to a VerbatimRAG; info is shown in the UI as-is.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    from verbatim_rag.streaming import StreamingRAG

    if default_extractor not in pipelines:
        raise ValueError(f"default extractor {default_extractor!r} not in {sorted(pipelines)}")
    app = FastAPI(title="Verbatim RAG Hungarian demo")
    lock = asyncio.Lock()
    query_payload = [query_view(q) for q in queries]

    @app.get("/api/status")
    async def status():
        return {"resources_loaded": True, "message": "Hungarian demo ready"}

    @app.get("/api/demo/config")
    async def config():
        return {
            "extractors": list(pipelines),
            "default_extractor": default_extractor,
            **info,
        }

    @app.get("/api/demo/queries")
    async def demo_queries():
        return {"split": info.get("split"), "queries": query_payload}

    @app.post("/api/query/stream")
    async def stream(request: StreamRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(
                status_code=400, detail=f"Question too long (max {MAX_QUESTION_CHARS} characters)"
            )
        name = request.extractor or default_extractor
        if name not in pipelines:
            raise HTTPException(
                status_code=400, detail=f"Unknown extractor {name!r}; loaded: {list(pipelines)}"
            )
        streaming = StreamingRAG(pipelines[name])

        async def stages():
            async with lock:
                # Wait off the event loop for a leftover extraction thread to release the
                # models; retrieval then takes MODEL_LOCK on the loop without blocking it.
                await asyncio.to_thread(wait_for_models)
                logger.info("[%s] %s", name, question)
                async for stage in streaming.stream_query(question):
                    if stage.get("type") == "error":
                        logger.error("[%s] query failed: %s", name, stage.get("error"))
                    yield json.dumps(stage) + "\n"

        return StreamingResponse(
            stages(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    return app


# ---- building the pipeline ---------------------------------------------------


def make_extractors(args, nlp) -> Dict[str, SpanExtractor]:
    extractors: Dict[str, SpanExtractor] = {}
    for name in args.extractors:
        if name == "crossencoder":
            from verbatim_rag.cross_encoder_extractor import CrossEncoderSentenceExtractor

            extractors[name] = CrossEncoderSentenceExtractor(
                model=args.ce_model,
                revision=args.ce_revision,
                device=args.ce_device or args.device,
                threshold=args.ce_threshold,
                batch_size=args.ce_batch_size,
                max_length=args.ce_max_length,
                nlp=nlp,
            )
        else:
            from verbatim_core.extractors import LLMSpanExtractor
            from verbatim_core.llm_client import LLMClient

            key = os.environ.get(args.llm_key_env)
            if not key:
                raise SystemExit(
                    f"--extractors includes llm but ${args.llm_key_env} is not set "
                    "(or run with --extractors crossencoder)"
                )
            client = LLMClient(
                model=args.llm_model, temperature=0.0, api_base=args.api_base, api_key=key
            )
            client.client = client.client.with_options(
                timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES
            )
            client.complete("Reply with OK.")  # raises on a bad key, model or api-base
            extractors[name] = LLMSpanExtractor(
                llm_client=client, extraction_mode="batch", batch_size=1000
            )
    return extractors


def index_dir(data_dir: Path, tag: str) -> Path:
    """data/index/<tag>/, refusing tags that would resolve elsewhere (--build deletes it)."""
    if not _TAG_RE.fullmatch(tag) or tag in (".", ".."):
        raise ValueError(f"--index-tag must be a plain directory name, got {tag!r}")
    return data_dir / "index" / tag


def prepare_index_dir(args, corpus) -> tuple:
    """(run_dir, reuse). Runs before any model loads: a mismatched index fails in seconds,
    and a rebuild starts from an empty dir so no stale BM25 stats are loaded."""
    run_dir = index_dir(args.data_dir, args.index_tag)
    if (run_dir / INDEX_META).exists():
        try:
            check_index_meta(run_dir, index_meta(corpus, args, dense=True, sparse=True))
        except ValueError as e:
            raise SystemExit(str(e)) from None
        return run_dir, True
    if not args.build:
        raise SystemExit(
            f"no completed index in {run_dir}: pass --build to build it "
            f"(or point --index-tag at an index built by run_retrieval)"
        )
    shutil.rmtree(run_dir, ignore_errors=True)  # never reuse unchecked stats
    run_dir.mkdir(parents=True)
    return run_dir, False


def build_pipelines(args) -> Dict[str, Any]:
    from verbatim_rag.hungarian_bm25 import HungarianBM25Provider
    from verbatim_rag.qwen3_embedding import Qwen3EmbeddingProvider
    from verbatim_rag.rerankers import JinaV3Reranker

    corpus = read_jsonl(args.data_dir / "corpus.jsonl")
    run_dir, reuse = prepare_index_dir(args, corpus)
    sparse = QueryCache(
        HungarianBM25Provider(
            stats_path=str(run_dir / "bm25_hu_stats.json"),
            model=args.bm25_model,
            k1=args.k1,
            b=args.b,
            batch_size=args.batch_size,
        )
    )
    dense = QueryCache(
        Qwen3EmbeddingProvider(
            model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            revision=args.dense_revision,
        )
    )
    reranker = JinaV3Reranker(
        model=args.rerank_model,
        rerank_k=args.retrieve_k,
        device=args.device,
        revision=args.rerank_revision,
        max_block_tokens=args.rerank_block_tokens,
    )
    # Load the extractors before indexing so a bad model id or key fails in seconds.
    extractors = make_extractors(args, nlp=sparse._provider.nlp)
    logger.info("%s index in %s", "reusing" if reuse else "building", run_dir)
    index = open_index(run_dir, corpus, sparse, dense, args, reuse)
    retriever = RerankedRetriever(
        index,
        reranker,
        retrieve_k=args.retrieve_k,
        hybrid_weights={"dense": args.dense_weight, "sparse": args.sparse_weight},
        rrf_k=args.rrf_k,
    )
    return {
        name: make_pipeline(
            retriever, extractor, args.extract_k, getattr(extractor, "llm_client", None)
        )
        for name, extractor in extractors.items()
    }


def make_pipeline(retriever, extractor: SpanExtractor, k: int, llm_client=None):
    """VerbatimRAG over the retriever: checked extractor, Hungarian static template, answer
    left uncleaned. llm_client is never called in static mode (None makes a default one)."""
    from verbatim_rag.core import VerbatimRAG

    rag = VerbatimRAG(
        index=retriever,
        k=k,
        extractor=CheckedExtractor(extractor),
        template_manager=hungarian_template_manager(),
        llm_client=llm_client,
    )
    rag.response_builder = UncleanedResponseBuilder()
    return rag


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def parse_extractors(value: str) -> List[str]:
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [n for n in names if n not in EXTRACTORS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"choose from {','.join(EXTRACTORS)}, got {value!r}")
    return list(dict.fromkeys(names))


def demo_info(args, n_queries: int) -> Dict[str, Any]:
    """Settings shown in the UI next to the extractor toggle."""
    info: Dict[str, Any] = {
        "split": args.split,
        "n_queries": n_queries,
        "retrieve_k": args.retrieve_k,
        "extract_k": args.extract_k,
        "dense_model": args.dense_model,
        "rerank_model": args.rerank_model,
    }
    if "crossencoder" in args.extractors:
        info["crossencoder"] = {"model": args.ce_model, "threshold": args.ce_threshold}
    if "llm" in args.extractors:
        info["llm"] = {"model": args.llm_model}
    return info


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--index-tag", required=True, help="Index in data/index/<NAME>/")
    p.add_argument("--build", action="store_true", help="Build the index if none is complete")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--split", choices=["test", "dev", "train"], default="test")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda", help="Device for the dense model and reranker")
    p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    g = p.add_argument_group("extractors")
    g.add_argument(
        "--extractors",
        type=parse_extractors,
        default=["crossencoder"],
        help="Comma-separated extractors to load, selectable per query. llm is opt-in: "
        "it sends the top chunks to --api-base",
    )
    g.add_argument("--default-extractor", choices=EXTRACTORS, help="Default: first loaded")
    g.add_argument("--ce-model", default="BAAI/bge-reranker-v2-m3")
    g.add_argument("--ce-revision", help="HF commit sha to pin the cross-encoder checkpoint")
    g.add_argument("--ce-device", help="Default: --device")
    g.add_argument("--ce-batch-size", type=int, default=32)
    g.add_argument("--ce-max-length", type=int, default=512)
    g.add_argument("--ce-threshold", type=float, default=0.5)
    g.add_argument("--llm-model", default="deepseek-v4-flash")
    g.add_argument("--api-base", default="https://api.deepseek.com/v1")
    g.add_argument("--llm-key-env", default="DEEPSEEK_API_KEY", help="Env var holding the key")
    g = p.add_argument_group("retrieval (same defaults as run_extraction)")
    g.add_argument("--retrieve-k", type=int, default=100, help="Fused candidates reranked")
    g.add_argument("--extract-k", type=int, default=5, help="Top chunks passed to the extractor")
    g.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-0.6B")
    g.add_argument("--dense-revision", help="HF commit sha to pin the dense model")
    g.add_argument("--bm25-model", default="hu_core_news_lg")
    g.add_argument("--k1", type=float, default=1.2)
    g.add_argument("--b", type=float, default=0.75)
    g.add_argument("--dense-weight", type=float, default=0.5)
    g.add_argument("--sparse-weight", type=float, default=0.5)
    g.add_argument("--rrf-k", type=int, default=60)
    g.add_argument("--rerank-model", default="jinaai/jina-reranker-v3.5")
    g.add_argument("--rerank-revision", default=RERANK_REVISION)
    g.add_argument(
        "--rerank-block-tokens",
        type=int,
        default=4096,
        help="Approximate cap on tokens per reranker forward pass (4096 fits a 16 GB GPU)",
    )
    args = p.parse_args(argv)
    if args.extract_k < 1 or args.retrieve_k < args.extract_k:
        p.error("need 1 <= --extract-k <= --retrieve-k")
    if not 0.0 <= args.ce_threshold <= 1.0:
        p.error("--ce-threshold must be in [0, 1]")
    try:
        index_dir(args.data_dir, args.index_tag)
    except ValueError as e:
        p.error(str(e))
    if args.default_extractor is None:
        args.default_extractor = args.extractors[0]
    elif args.default_extractor not in args.extractors:
        p.error(f"--default-extractor {args.default_extractor} is not in --extractors")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not is_loopback(args.host):
        logger.warning(
            "--host %s is reachable from other machines; the corpus and queries "
            "are CC BY-NC-ND 4.0, keep the demo private",
            args.host,
        )
    if "crossencoder" in args.extractors and not args.ce_revision:
        logger.warning("crossencoder without --ce-revision: checkpoint is not pinned")
    queries, n_excluded = load_split(args.data_dir / "queries.jsonl", args.split)
    logger.info("%s split: %d queries (%d excluded)", args.split, len(queries), n_excluded)
    pipelines = build_pipelines(args)
    app = create_app(pipelines, args.default_extractor, queries, demo_info(args, len(queries)))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
