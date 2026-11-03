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
runs) is reranked by --reranker (jina: jina-reranker-v3.5, the default; qwen3:
Qwen3-Reranker-0.6B) and reported as a fourth method, "reranked". It is a
permutation of the same 100 candidates, so its Recall@100 equals its input's; Recall@10 and
MRR are what change. Extraction and abstention are measured by run_extraction.

--folded adds the accent-folded BM25 channel (its own Milvus db and stats next to the index,
with folded_meta.json) as two more methods: "folded" alone and "fused_folded", the RRF of dense,
sparse and folded, which then feeds the reranker. --strip-query-accents drops the accents of
every query, to measure questions typed without them; the reranker restores them from its
candidates unless --no-accent-restore.
Unanswerable queries are skipped here; only their count is reported.

The index and BM25 stats go under data/index/<tag>/ (gitignored, licence), with an
index_meta.json written once indexing completes so run_extraction --index-tag can reuse it.
The run log written to runs/ holds aggregate numbers and the full config only, no corpus text.
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).parent
DATA = HERE / "data"
RUNS = HERE / "runs"

BREAKDOWNS = ["query_type", "legal_domain", "difficulty", "bucket"]
CUTOFFS = (10, 100)
METHODS = ("sparse", "dense", "fused")
FOLDED = "folded"  # accent-folded BM25, its own Milvus db next to the main index
FUSED_FOLDED = "fused_folded"  # RRF of dense, sparse and folded
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


def load_split(path: Path, split: str) -> Tuple[List[Dict[str, Any]], int]:
    """Queries of one split, minus those annotated as not answered by their own passage
    (`excluded`: true gold unknown, so not scored). Returns (queries, n_excluded)."""
    rows = [q for q in read_jsonl(path) if q["split"] == split]
    kept = [q for q in rows if not q.get("excluded")]
    return kept, len(rows) - len(kept)


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


INDEX_META = "index_meta.json"


def corpus_fingerprint(corpus: List[Dict[str, Any]]) -> str:
    """Order-independent hash of the (chunk_id, text) pairs."""
    h = hashlib.sha256()
    for chunk_id, text in sorted((row["chunk_id"], row["text"]) for row in corpus):
        h.update(json.dumps([chunk_id, text], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def index_meta(corpus: List[Dict[str, Any]], args, dense: bool, sparse: bool) -> Dict[str, Any]:
    """What the stored index depends on; written once indexing completes, checked on reuse."""
    return {
        "corpus_chunks": len(corpus),
        "corpus_sha256": corpus_fingerprint(corpus),
        "chunk_text": "raw corpus text",
        "dense": {"model": args.dense_model, "revision": args.dense_revision} if dense else None,
        "bm25": {"model": args.bm25_model, "k1": args.k1, "b": args.b} if sparse else None,
    }


def write_index_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    (run_dir / INDEX_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


# The folded channel sits next to the main index with its own meta file, so index_meta.json
# and every index built before it stay valid, and it can be added to an existing index dir.
FOLDED_META = "folded_meta.json"
FOLDED_DB = "milvus_folded.db"
FOLDED_STATS = "bm25_hu_folded_stats.json"
_REBUILD_FOLDED = f"delete {FOLDED_META}, {FOLDED_DB} and {FOLDED_STATS} there to rebuild it"


def folded_meta(corpus: List[Dict[str, Any]], args) -> Dict[str, Any]:
    from verbatim_rag.hungarian_bm25 import ANALYZER_VERSION, FOLDED_ANALYZER_VERSION

    return {
        "corpus_chunks": len(corpus),
        "corpus_sha256": corpus_fingerprint(corpus),
        "chunk_text": "raw corpus text",
        "bm25_folded": {
            "model": args.bm25_model,
            "k1": args.k1,
            "b": args.b,
            # The folded rule delegates non-alphabetic tokens to the lemma provider's rule.
            "analyzer_version": ANALYZER_VERSION,
            "folded_analyzer_version": FOLDED_ANALYZER_VERSION,
        },
    }


def check_folded_meta(run_dir: Path, expected: Dict[str, Any]) -> bool:
    """True when run_dir holds a completed folded channel built with these settings, False
    when it holds none; raises when it holds one built with other settings."""
    path = run_dir / FOLDED_META
    if not path.exists():
        return False
    found = json.loads(path.read_text(encoding="utf-8"))
    if found != expected:
        diff = {
            k: {"index": found.get(k), "run": expected.get(k)}
            for k in sorted(set(found) | set(expected))
            if found.get(k) != expected.get(k)
        }
        raise ValueError(
            f"folded channel in {run_dir} does not match this run: {json.dumps(diff)}; "
            f"{_REBUILD_FOLDED}"
        )
    if not (run_dir / FOLDED_DB).exists():
        raise ValueError(f"{run_dir / FOLDED_DB} missing next to {FOLDED_META}; {_REBUILD_FOLDED}")
    return True


def open_folded_index(run_dir: Path, corpus: List[Dict[str, Any]], nlp, args, build: bool):
    """(QueryCache over the folded BM25 provider, its sparse-only VerbatimIndex).

    Reuses a completed folded channel in run_dir, or builds one when `build` is set (a spaCy
    pass over the corpus, no dense model). Shares `nlp` with the lemma provider.
    """
    from verbatim_rag.hungarian_bm25 import HungarianFoldedBM25Provider
    from verbatim_rag.index import VerbatimIndex
    from verbatim_rag.vector_stores import LocalMilvusStore

    expected = folded_meta(corpus, args)
    reuse = check_folded_meta(run_dir, expected)
    if not reuse:
        if not build:
            raise ValueError(f"no completed folded channel in {run_dir}")
        for name in (FOLDED_DB, FOLDED_STATS):  # leftovers of an interrupted build
            (run_dir / name).unlink(missing_ok=True)
    folded = QueryCache(
        HungarianFoldedBM25Provider(
            stats_path=str(run_dir / FOLDED_STATS),
            model=args.bm25_model,
            k1=args.k1,
            b=args.b,
            batch_size=args.batch_size,
            nlp=nlp,
        )
    )
    db_path = run_dir / FOLDED_DB
    if not reuse:
        logger.info("building the folded BM25 channel in %s", db_path)
        index = build_index(corpus, db_path, folded, None, args.batch_size)
        (run_dir / FOLDED_META).write_text(json.dumps(expected, indent=2), encoding="utf-8")
        return folded, index
    if folded._provider._stale_reason:  # fail now, not on the first query
        raise ValueError(f"{folded._provider._stale_reason}; {_REBUILD_FOLDED}")
    if not folded._provider.fitted_on(row["text"] for row in corpus):
        raise ValueError(
            f"folded BM25 stats in {run_dir} were not fitted on this corpus; {_REBUILD_FOLDED}"
        )
    store = LocalMilvusStore(db_path=str(db_path), enable_dense=False, enable_sparse=True)
    rows = store.client.query(
        collection_name=store.collection_name, filter="", output_fields=["count(*)"]
    )[0]["count(*)"]
    if rows != len(corpus):
        raise ValueError(f"folded index {db_path} holds {rows} chunks, corpus has {len(corpus)}")
    return folded, VerbatimIndex(vector_store=store, sparse_provider=folded)


# ---- accents ---------------------------------------------------------------


def add_accent_args(p) -> None:
    """Flags shared by run_retrieval and run_extraction (folded channel, unaccented queries)."""
    p.add_argument(
        "--folded",
        action="store_true",
        help="Add the accent-folded BM25 channel (own Milvus db next to the index) to the fusion",
    )
    p.add_argument("--folded-weight", type=float, default=0.5)
    p.add_argument(
        "--strip-query-accents",
        action="store_true",
        help="Drop the accents from every query, to measure questions typed without them",
    )
    p.add_argument(
        "--no-accent-restore",
        action="store_true",
        help="Do not restore the accents of an unaccented question before the reranker and "
        "the sentence judge",
    )


def apply_query_accents(queries: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    if not args.strip_query_accents:
        return queries
    from verbatim_rag.hungarian_accents import strip_accents

    return [{**q, "query": strip_accents(q["query"])} for q in queries]


def count_unaccented(queries: List[Dict[str, Any]]) -> int:
    """Queries without any diacritic: the ones accent restoration acts on."""
    from verbatim_rag.hungarian_accents import has_diacritics

    return sum(not has_diacritics(q["query"]) for q in queries)


# ---- reranker --------------------------------------------------------------

RERANK_MODELS = {"jina": "jinaai/jina-reranker-v3.5", "qwen3": "Qwen/Qwen3-Reranker-0.6B"}
DEFAULT_RERANKER = "jina"
# Every flag add_rerank_args defines; all default to None so check_rerank_args sees what was set.
_RERANK_FLAGS = ("reranker", "model", "revision", "block_tokens", "batch_size", "instruction")


def add_rerank_args(
    p,
    block_tokens_default: str = "all candidates in one pass",
    revision_default: str = "latest (unpinned)",
) -> None:
    """Reranker choice and settings, shared by run_retrieval, run_extraction and the demo
    (which fills in its own jina defaults after parsing, described by the *_default texts)."""
    p.add_argument(
        "--reranker",
        choices=sorted(RERANK_MODELS),
        help=f"jina: listwise jina-reranker-v3.5 (CC-BY-NC-4.0, remote code); qwen3: pointwise "
        f"Qwen3-Reranker-0.6B (Apache-2.0). Default: {DEFAULT_RERANKER}",
    )
    p.add_argument("--rerank-model", help="HF model id (default: the --reranker's model)")
    p.add_argument(
        "--rerank-revision", help=f"HF commit sha to pin the reranker; default: {revision_default}"
    )
    p.add_argument(
        "--rerank-block-tokens",
        type=int,
        help="jina only: approximate cap on tokens per reranker forward pass (4096 fits a 16 GB "
        f"GPU); default: {block_tokens_default}",
    )
    p.add_argument(
        "--rerank-batch-size",
        type=int,
        help="qwen3 only: (question, chunk) prompts per forward pass "
        "(default: Qwen3Reranker's, 16)",
    )
    p.add_argument(
        "--rerank-instruction",
        help="qwen3 only: English task instruction replacing Qwen3Reranker.DEFAULT_INSTRUCTION",
    )


def check_rerank_args(p, args, enabled: bool = True) -> None:
    """Reject flags of the other reranker (or any, when reranking is off) and fill in the
    reranker and model id, in place."""
    if not enabled:
        for flag in _RERANK_FLAGS:
            attr = flag if flag == "reranker" else f"rerank_{flag}"
            if getattr(args, attr) is not None:
                p.error(f"--{attr.replace('_', '-')} needs --rerank")
    if args.reranker is None:
        args.reranker = DEFAULT_RERANKER
    if args.reranker == "qwen3" and args.rerank_block_tokens is not None:
        p.error("--rerank-block-tokens only applies to --reranker jina")
    if args.reranker == "jina":
        if args.rerank_batch_size is not None:
            p.error("--rerank-batch-size only applies to --reranker qwen3")
        if args.rerank_instruction is not None:
            p.error("--rerank-instruction only applies to --reranker qwen3")
    if args.rerank_batch_size is not None and args.rerank_batch_size <= 0:
        p.error("--rerank-batch-size must be positive")
    if args.rerank_instruction is not None and not args.rerank_instruction.strip():
        p.error("--rerank-instruction must not be empty")
    if args.rerank_model is None:
        args.rerank_model = RERANK_MODELS[args.reranker]


def make_reranker(args, rerank_k: int):
    """The reranker check_rerank_args validated, on args.device."""
    restore = not args.no_accent_restore
    if args.reranker == "qwen3":
        if not args.rerank_revision:
            logger.warning("reranker without --rerank-revision: weights are not pinned")
        from verbatim_rag.rerankers import Qwen3Reranker

        # Unset options keep Qwen3Reranker's own defaults.
        optional = {"instruction": args.rerank_instruction, "batch_size": args.rerank_batch_size}
        return Qwen3Reranker(
            model=args.rerank_model,
            rerank_k=rerank_k,
            device=args.device,
            revision=args.rerank_revision,
            restore_accents=restore,
            **{k: v for k, v in optional.items() if v is not None},
        )
    if not args.rerank_revision:
        logger.warning("reranker without --rerank-revision: remote code is not pinned")
    from verbatim_rag.rerankers import JinaV3Reranker

    return JinaV3Reranker(
        model=args.rerank_model,
        rerank_k=rerank_k,
        device=args.device,
        revision=args.rerank_revision,
        max_block_tokens=args.rerank_block_tokens,
        restore_accents=restore,
    )


def rerank_config(args, reranker, rerank_k: int) -> Dict[str, Any]:
    """Run-log entry for the reranker make_reranker built."""
    config: Dict[str, Any] = {
        "reranker": args.reranker,
        "model": args.rerank_model,
        "revision": args.rerank_revision,  # None = latest at run time; pin it
        "rerank_k": rerank_k,
    }
    if args.reranker == "qwen3":
        config["batch_size"] = reranker.batch_size
        config["max_length"] = reranker.max_length
        config["instruction"] = reranker.instruction
        config["dtype"] = str(reranker.model.dtype)  # float32 on cpu, else the checkpoint's
    else:
        config["max_block_tokens"] = args.rerank_block_tokens
    return config


# ---- run -------------------------------------------------------------------


def rerank_source(methods: Sequence[str]) -> str:
    """The method whose top-k feeds the reranker: the widest fusion available, else the only
    method."""
    for method in (FUSED_FOLDED, "fused"):
        if method in methods:
            return method
    return methods[0]


def fusion_weights(args) -> Dict[str, float]:
    """RRF weights per method; folded only when the run has the folded channel."""
    weights = {"dense": args.dense_weight, "sparse": args.sparse_weight}
    if getattr(args, "folded", False):
        weights[FOLDED] = args.folded_weight
    return weights


def run_queries(
    index,
    queries: List[Dict[str, Any]],
    methods: Sequence[str],
    args,
    reranker=None,
    folded_index=None,
) -> List[Dict]:
    k = max(CUTOFFS)
    source = rerank_source(methods)
    fused_weights = {"dense": args.dense_weight, "sparse": args.sparse_weight}
    fused_folded = None
    if folded_index is not None:
        from verbatim_rag.fused_index import FusedIndex

        fused_folded = FusedIndex(index, {FOLDED: folded_index})
    rows = []
    for i, q in enumerate(queries, start=1):
        ranks = {}
        for method in methods:
            if method == "fused":
                hits = index.query(q["query"], k=k, hybrid_weights=fused_weights, rrf_k=args.rrf_k)
            elif method == FUSED_FOLDED:
                hits = fused_folded.query(
                    q["query"], k=k, hybrid_weights=fusion_weights(args), rrf_k=args.rrf_k
                )
            elif method == FOLDED:
                hits = folded_index.query(q["query"], k=k, search_type="sparse")
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
    p.add_argument("--rerank", action="store_true", help="Rerank the top-100 with --reranker")
    p.add_argument("--rerank-k", type=int, default=max(CUTOFFS), help="Candidates reranked")
    add_rerank_args(p)
    p.add_argument("--bm25-model", default="hu_core_news_lg")
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--sparse-weight", type=float, default=0.5)
    add_accent_args(p)
    p.add_argument("--seed", type=int, default=0, help="Seed for the --limit sample")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16, help="Indexing/embedding batch size")
    p.add_argument("--tag", help="Name for this run; also names the index dir under data/")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rebuilding data/index/<tag>/ when it already holds a completed index",
    )
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    args = p.parse_args(argv)
    check_rerank_args(p, args, enabled=args.rerank)
    if args.skip_dense and args.skip_sparse:
        p.error("nothing to run: both providers skipped")
    if args.folded and (args.skip_dense or args.skip_sparse):
        p.error("--folded needs both providers (it fuses dense, sparse and folded)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = datetime.now(timezone.utc)
    tag = args.tag or started.strftime("%Y%m%d-%H%M%S")

    corpus = read_jsonl(args.data_dir / "corpus.jsonl")
    split, n_excluded = load_split(args.data_dir / "queries.jsonl", args.split)
    queries = [q for q in split if q["answerable"]]
    n_unanswerable = len(split) - len(queries)
    if args.limit is not None:
        queries = random.Random(args.seed).sample(queries, min(args.limit, len(queries)))
    queries = apply_query_accents(queries, args)
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    logger.info(
        "corpus %d chunks; %s split: %d answerable queries (%d unanswerable, %d excluded)",
        len(corpus),
        args.split,
        len(queries),
        n_unanswerable,
        n_excluded,
    )

    sparse = dense = None
    run_dir = args.data_dir / "index" / tag
    if (run_dir / INDEX_META).exists() and not args.overwrite:
        p.error(f"{run_dir} holds a completed index; pass --overwrite to rebuild it")
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
        reranker = make_reranker(args, args.rerank_k)
    db_path = run_dir / "milvus.db"
    t0 = time.time()
    index = build_index(corpus, db_path, sparse, dense, args.batch_size)
    folded = folded_index = None
    if args.folded:
        folded, folded_index = open_folded_index(
            run_dir, corpus, sparse._provider.nlp, args, build=True
        )
    index_seconds = time.time() - t0
    write_index_meta(run_dir, index_meta(corpus, args, dense is not None, sparse is not None))

    methods = [m for m in METHODS if (m != "sparse" or sparse) and (m != "dense" or dense)]
    if not (sparse and dense):
        methods = [m for m in methods if m != "fused"]
    if folded_index is not None:
        methods[methods.index("fused") : methods.index("fused")] = [FOLDED]
        methods.append(FUSED_FOLDED)
    t0 = time.time()
    rows = run_queries(index, queries, methods, args, reranker, folded_index)
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
                "folded_weight": args.folded_weight if folded else None,
                "rrf_k": args.rrf_k,
            },
            "folded_index": folded_meta(corpus, args)["bm25_folded"] if folded else None,
            "query_accents": "stripped" if args.strip_query_accents else "as written",
            "accent_restore": not args.no_accent_restore,
            "rerank": None
            if reranker is None
            else {
                **rerank_config(args, reranker, args.rerank_k),
                "source": rerank_source(methods),
            },
            "cutoffs": list(CUTOFFS),
        },
        "dataset": {
            "revision": manifest.get("revision"),
            "corpus_chunks": len(corpus),
            "answerable_queries": len(queries),
            "unanswerable_skipped": n_unanswerable,
            "excluded_by_annotation": n_excluded,
        },
        "empty_sparse_queries": None if sparse is None else sparse.empty,
        "empty_folded_queries": None if folded is None else folded.empty,
        "unaccented_queries": count_unaccented(queries),
        "seconds": {"index": round(index_seconds), "query": round(query_seconds)},
        "results": results,
    }
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = args.runs_dir / f"{tag}.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.split} split, {len(queries)} answerable queries -> {out}")
    print(f"{'method':12} " + " ".join(f"{m:>11}" for m in ("recall@10", "recall@100", "mrr@100")))
    for method in reported:
        o = results[method]["overall"]
        print(f"{method:12} {o['recall@10']:11.3f} {o['recall@100']:11.3f} {o['mrr@100']:11.3f}")
    if sparse is not None:
        print(f"empty sparse query vectors: {sparse.empty}")
    if folded is not None:
        print(f"empty folded query vectors: {folded.empty}")


if __name__ == "__main__":
    main()
