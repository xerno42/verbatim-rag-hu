"""
Retrieval eval on the Hungarian legal set: sparse (Hungarian BM25), dense (Qwen3) and fused.

    python -m evaluation.hu_legal.run_retrieval [--split test] [--limit N] [--skip-dense]

Indexes every corpus row as one chunk keyed by chunk_id (bypassing DocumentProcessor so
gold ids line up), fits BM25 on exactly those strings, then runs each answerable query of
the split three ways through VerbatimIndex.query: sparse only, dense only, and RRF-fused.
A query is a hit at rank r when any of its gold_chunk_ids (equivalent overlapping chunks)
is the r-th result. Reported per method: Recall@10, Recall@100 and MRR@100, overall and
split by query_type, legal_domain, difficulty and bucket.

With --rerank, the top-100 of the fused list (or of the single method when only one provider
runs) is reranked by Jina reranker v3.5 and reported as a fourth method, "reranked". It is a
permutation of the same 100 candidates, so its Recall@100 equals its input's; Recall@10 and
MRR are what change. Extraction and abstention are not implemented yet, so they are not measured.
Unanswerable queries are skipped here; only their count is reported.

The index and BM25 stats go under data/ (gitignored, licence). The run log written to
runs/ holds aggregate numbers and the full config only, no corpus text.
"""

import argparse
import json
import logging
import random
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).parent
DATA = HERE / "data"
RUNS = HERE / "runs"

BREAKDOWNS = ["query_type", "legal_domain", "difficulty", "bucket"]
CUTOFFS = (10, 100)
METHODS = ("sparse", "dense", "fused")
RERANKED = "reranked"

logger = logging.getLogger("hu_legal.run_retrieval")


# ---- metrics (pure, network-free) ------------------------------------------


def first_hit_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> Optional[int]:
    """1-based rank of the first result that is a gold chunk, or None."""
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold:
            return rank
    return None


def summarize(ranks: Sequence[Optional[int]], cutoffs: Sequence[int] = CUTOFFS) -> Dict[str, Any]:
    """Recall@k (any gold chunk in the top k) and MRR@max(cutoffs) from first-hit ranks."""
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    top = max(cutoffs)
    out: Dict[str, Any] = {"n": n}
    for k in cutoffs:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    out[f"mrr@{top}"] = sum(1 / r for r in ranks if r is not None and r <= top) / n
    return out


def breakdown(rows: List[Dict[str, Any]], method: str, field: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Optional[int]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row["ranks"][method])
    return {value: summarize(ranks) for value, ranks in sorted(groups.items())}


# ---- data ------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class QueryCache:
    """Wraps a provider so a query embedded for several methods is computed once.

    Also counts empty sparse query vectors (no query term in the BM25 vocabulary).
    """

    def __init__(self, provider):
        self._provider = provider
        self._cache: Dict[str, Any] = {}
        self.empty = 0

    def embed_text(self, text):
        if text not in self._cache:
            vec = self._provider.embed_text(text)
            if isinstance(vec, dict) and not vec:
                self.empty += 1
            self._cache[text] = vec
        return self._cache[text]

    def embed_batch(self, texts):
        return self._provider.embed_batch(texts)

    def get_dimension(self):
        return self._provider.get_dimension()


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


# ---- indexing --------------------------------------------------------------


def build_index(
    corpus: List[Dict[str, Any]],
    db_path: Path,
    sparse,
    dense,
    batch_size: int,
):
    """Create a fresh Milvus Lite db with one row per corpus chunk and return the index."""
    from verbatim_rag.index import VerbatimIndex
    from verbatim_rag.vector_stores import LocalMilvusStore

    store = LocalMilvusStore(
        db_path=str(db_path),
        dense_dim=dense.get_dimension() if dense else 384,
        enable_dense=dense is not None,
        enable_sparse=sparse is not None,
    )
    index = VerbatimIndex(vector_store=store, dense_provider=dense, sparse_provider=sparse)

    texts = [row["text"] for row in corpus]
    if sparse is not None:
        t0 = time.time()
        sparse._provider.fit(texts)
        logger.info("BM25 fit on %d chunks in %.0fs", len(texts), time.time() - t0)

    for start in range(0, len(corpus), batch_size):
        batch = corpus[start : start + batch_size]
        batch_texts = [row["text"] for row in batch]
        t0 = time.time()
        index._store_chunks(
            ids=[row["chunk_id"] for row in batch],
            texts=batch_texts,
            enhanced_texts=batch_texts,
            dense_embeddings=dense.embed_batch(batch_texts) if dense else None,
            sparse_embeddings=sparse.embed_batch(batch_texts) if sparse else None,
            metadatas=[
                {"document_id": row["doc_id"], "source": row["source"], "bucket": row["bucket"]}
                for row in batch
            ],
        )
        logger.info(
            "indexed %d/%d (%.1fs for last batch)",
            min(start + batch_size, len(corpus)),
            len(corpus),
            time.time() - t0,
        )
    return index


# ---- run -------------------------------------------------------------------


def rerank_source(methods: Sequence[str]) -> str:
    """The method whose top-k feeds the reranker: fused when available, else the only method."""
    return "fused" if "fused" in methods else methods[0]


def run_queries(
    index,
    queries: List[Dict[str, Any]],
    methods: Sequence[str],
    args,
    reranker=None,
) -> List[Dict]:
    k = max(CUTOFFS)
    source = rerank_source(methods)
    fused_weights = {"dense": args.dense_weight, "sparse": args.sparse_weight}
    rows = []
    for i, q in enumerate(queries, start=1):
        ranks = {}
        for method in methods:
            if method == "fused":
                hits = index.query(q["query"], k=k, hybrid_weights=fused_weights, rrf_k=args.rrf_k)
            else:
                hits = index.query(q["query"], k=k, search_type=method)
            ranks[method] = first_hit_rank([h.id for h in hits], q["gold_chunk_ids"])
            if reranker is not None and method == source:
                reranked = reranker.rerank(q["query"], hits)
                ranks[RERANKED] = first_hit_rank([h.id for h in reranked], q["gold_chunk_ids"])
        rows.append({**{f: q.get(f) for f in BREAKDOWNS}, "ranks": ranks})
        if i % 100 == 0:
            logger.info("queried %d/%d", i, len(queries))
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--split", choices=["test", "dev", "train"], default="test")
    p.add_argument(
        "--limit", type=int, help="Seeded random sample of N answerable queries (smoke test)"
    )
    p.add_argument("--skip-dense", action="store_true", help="BM25 only; no fused result")
    p.add_argument("--skip-sparse", action="store_true", help="Dense only; no fused result")
    p.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--dense-revision", help="HF commit sha to pin the dense model")
    p.add_argument("--query-instruction", help="English task text replacing Qwen's query prompt")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--rerank", action="store_true", help="Rerank the top-100 with Jina reranker v3.5"
    )
    p.add_argument("--rerank-model", default="jinaai/jina-reranker-v3.5")
    p.add_argument("--rerank-revision", help="HF commit sha to pin the reranker (remote code)")
    p.add_argument("--rerank-k", type=int, default=max(CUTOFFS), help="Candidates reranked")
    p.add_argument("--bm25-model", default="hu_core_news_lg")
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--sparse-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0, help="Seed for the --limit sample")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64, help="Indexing/embedding batch size")
    p.add_argument("--tag", help="Name for this run; also names the index dir under data/")
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    args = p.parse_args(argv)
    if args.skip_dense and args.skip_sparse:
        p.error("nothing to run: both providers skipped")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = datetime.now(timezone.utc)
    tag = args.tag or started.strftime("%Y%m%d-%H%M%S")

    corpus = read_jsonl(args.data_dir / "corpus.jsonl")
    all_queries = read_jsonl(args.data_dir / "queries.jsonl")
    split = [q for q in all_queries if q["split"] == args.split]
    queries = [q for q in split if q["answerable"]]
    n_unanswerable = len(split) - len(queries)
    if args.limit is not None:
        queries = random.Random(args.seed).sample(queries, min(args.limit, len(queries)))
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    logger.info(
        "corpus %d chunks; %s split: %d answerable queries (%d unanswerable skipped)",
        len(corpus),
        args.split,
        len(queries),
        n_unanswerable,
    )

    sparse = dense = None
    run_dir = args.data_dir / "index" / tag
    shutil.rmtree(run_dir, ignore_errors=True)  # never reuse an index or BM25 stats
    run_dir.mkdir(parents=True)
    if not args.skip_sparse:
        from verbatim_rag.hungarian_bm25 import HungarianBM25Provider

        sparse = QueryCache(
            HungarianBM25Provider(
                stats_path=str(run_dir / "bm25_hu_stats.json"),
                model=args.bm25_model,
                k1=args.k1,
                b=args.b,
                batch_size=args.batch_size,
            )
        )
    if not args.skip_dense:
        from verbatim_rag.qwen3_embedding import Qwen3EmbeddingProvider

        dense = QueryCache(
            Qwen3EmbeddingProvider(
                model_name=args.dense_model,
                device=args.device,
                batch_size=args.batch_size,
                query_instruction=args.query_instruction,
                revision=args.dense_revision,
            )
        )

    # Load before indexing so a bad model id or offline machine fails in seconds, not hours.
    reranker = None
    if args.rerank:
        if not args.rerank_revision:
            logger.warning("--rerank without --rerank-revision: remote code is not pinned")
        from verbatim_rag.rerankers import JinaV3Reranker

        reranker = JinaV3Reranker(
            model=args.rerank_model,
            rerank_k=args.rerank_k,
            device=args.device,
            revision=args.rerank_revision,
        )
    db_path = run_dir / "milvus.db"
    t0 = time.time()
    index = build_index(corpus, db_path, sparse, dense, args.batch_size)
    index_seconds = time.time() - t0

    methods = [m for m in METHODS if (m != "sparse" or sparse) and (m != "dense" or dense)]
    if not (sparse and dense):
        methods = [m for m in methods if m != "fused"]
    t0 = time.time()
    rows = run_queries(index, queries, methods, args, reranker)
    query_seconds = time.time() - t0

    reported = methods + [RERANKED] if reranker is not None else methods
    results = {
        method: {
            "overall": summarize([r["ranks"][method] for r in rows]),
            **{f: breakdown(rows, method, f) for f in BREAKDOWNS},
        }
        for method in reported
    }
    log = {
        "tag": tag,
        "started": started.isoformat(),
        "git_sha": git_sha(),
        "config": {
            "split": args.split,
            "limit": args.limit,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "chunk_text": "raw corpus text (no DocumentProcessor title/metadata enhancement)",
            "dense_model": None if dense is None else args.dense_model,
            "dense_revision": args.dense_revision,  # None = latest at run time; pin it
            "query_instruction": args.query_instruction,
            "device": args.device,
            "bm25": None
            if sparse is None
            else {
                "model": args.bm25_model,
                "model_version": sparse._provider.nlp.meta.get("version"),
                "k1": args.k1,
                "b": args.b,
            },
            "fusion": {
                "dense_weight": args.dense_weight,
                "sparse_weight": args.sparse_weight,
                "rrf_k": args.rrf_k,
            },
            "rerank": None
            if reranker is None
            else {
                "model": args.rerank_model,
                "revision": args.rerank_revision,  # None = latest at run time; pin it
                "rerank_k": args.rerank_k,
                "source": rerank_source(methods),
            },
            "cutoffs": list(CUTOFFS),
        },
        "dataset": {
            "revision": manifest.get("revision"),
            "corpus_chunks": len(corpus),
            "answerable_queries": len(queries),
            "unanswerable_skipped": n_unanswerable,
        },
        "empty_sparse_queries": None if sparse is None else sparse.empty,
        "seconds": {"index": round(index_seconds), "query": round(query_seconds)},
        "results": results,
    }
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = args.runs_dir / f"{tag}.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.split} split, {len(queries)} answerable queries -> {out}")
    print(f"{'method':8} " + " ".join(f"{m:>11}" for m in ("recall@10", "recall@100", "mrr@100")))
    for method in reported:
        o = results[method]["overall"]
        print(f"{method:8} {o['recall@10']:11.3f} {o['recall@100']:11.3f} {o['mrr@100']:11.3f}")
    if sparse is not None:
        print(f"empty sparse query vectors: {sparse.empty}")


if __name__ == "__main__":
    main()
