"""
Extraction eval on the Hungarian legal set: cross-encoder sentence judge or the LLM ceiling.

    python -m evaluation.hu_legal.run_extraction --mode oracle [--extractor crossencoder|llm|overlap]
    python -m evaluation.hu_legal.run_extraction --mode retrieved [--index-tag NAME] [--rerank]

Two modes:

- oracle: each answerable query gets its own passage as the only search result, which
  isolates the extractor from retrieval. Unanswerable queries are skipped (their removed
  passage would answer them); only their count is reported. With --distractors N, N hard
  negatives are added: chunks of the same document sharing no 8-word shingle with the
  passage (so they cannot hold its text), seeded per qid. A judge that says yes to
  everything then shows up as a high distractor_answer_rate.
- retrieved: every query of the split, unanswerable ones included, goes through retrieval
  (fused when both providers run), optional Jina v3.5 rerank, and extraction on the top
  --extract-k chunks. The index is built fresh under data/index/<tag>/ (refused if that holds
  a completed index, unless --overwrite), or reused with --index-tag when its index_meta.json
  matches the corpus content and provider settings and the stored row count matches.

Queries annotated as not answered by their own passage (`excluded`) are never scored.

Extractors:

- crossencoder: CrossEncoderSentenceExtractor (BAAI/bge-reranker-v2-m3 by default, sigmoid
  scores). Sentences are scored once per query and every value in --thresholds is reported
  as its own setting ("crossencoder@0.5"), for threshold tuning on dev.
- llm: LLMSpanExtractor through an OpenAI-compatible --api-base at temperature 0, the
  phase 1 quality ceiling. The key comes from OPENAI_API_KEY. One preflight call fails the
  run early on a bad key or endpoint. LLMSpanExtractor swallows call errors and returns no
  spans, so the runner counts them from its log (extractor_errors, which would otherwise
  read as abstention) together with spans its verbatim check dropped (dropped_spans).
  One batched prompt per query, so numbers compare across --distractors and --extract-k.
- overlap: no model; a sentence scores the share of the question's 4+ character words it
  contains. The floor for judge variants, reported like crossencoder ("overlap@0.5").

Metrics per setting, overall and split by query_type, legal_domain, difficulty and bucket
(retrieved mode adds unanswerable_reason):

- verbatim_pass_rate: share of emitted spans that are exact substrings of their chunk, keyed
  by a chunk that was actually passed in (target 1.0). Failures are logged by qid only.
- answer_rate / mean_spans on answerable queries, over all chunks (any span = answered);
  gold_answer_rate / mean_gold_spans count gold chunks only, so they stay comparable between
  oracle runs with and without distractors.
- oracle: gold_answer_rate (any span in the passage), coverage (share of passage characters
  covered), and with distractors distractor_answer_rate (share of distractor chunks with a
  span) and span_gold_chunk_precision.
- chunk_auc (crossencoder and overlap): how often a gold chunk's best sentence outscores a non-gold
  chunk's best sentence, leaving out non-gold chunks that share an 8-word shingle with the
  passage (they can hold the answer itself). Threshold-free, so it compares judge variants
  directly; retrieved-mode AUC covers only queries with a gold chunk in context.
- retrieved: abstention_accuracy (answered iff answerable), false_abstention_rate,
  false_answer_rate, gold_in_context (a gold chunk among the extracted chunks),
  answer_rate_gold_in_context, and span_gold_chunk_precision (share of spans taken from
  a gold chunk), a proxy until gold spans are annotated.
- word-level precision/recall/F1 (SQuAD-style token overlap) against gold_spans, only for
  queries that have them (n_gold). None do until the annotation pass is merged.

The run log in runs/ holds the config and aggregates only, no corpus text.
--dump-predictions writes per-query spans and scores under data/predictions/ (gitignored,
licence).
"""

import argparse
import json
import logging
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evaluation.hu_legal.overlap import shingles
from evaluation.hu_legal.run_retrieval import (
    BREAKDOWNS,
    DATA,
    INDEX_META,
    RUNS,
    QueryCache,
    build_index,
    git_sha,
    index_meta,
    load_split,
    read_jsonl,
    write_index_meta,
)

MODES = ("oracle", "retrieved")
EXTRACTORS = ("crossencoder", "llm", "overlap")
SCORING_EXTRACTORS = ("crossencoder", "overlap")  # per-sentence scores: threshold sweep + chunk_auc
RETRIEVED_BREAKDOWNS = BREAKDOWNS + ["unanswerable_reason"]

_WORD_RE = re.compile(r"\w+")

logger = logging.getLogger("hu_legal.run_extraction")


# ---- metrics (pure, network-free) ------------------------------------------


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _words(texts: Iterable[str]) -> Counter:
    return Counter(w.lower() for text in texts for w in _WORD_RE.findall(text))


def word_f1(pred_texts: Iterable[str], gold_texts: Iterable[str]) -> Dict[str, float]:
    """SQuAD-style word overlap (multiset of lowercased \\w+ tokens) of predicted vs gold text."""
    pred, gold = _words(pred_texts), _words(gold_texts)
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def verbatim_failures(
    spans_by_chunk: Dict[str, List[str]], chunk_texts: Optional[Iterable[str]] = None
) -> int:
    """Spans that are not an exact substring of their chunk. With chunk_texts, spans keyed by
    a text that was not among the extractor's inputs fail too."""
    known = None if chunk_texts is None else set(chunk_texts)
    return sum(
        1
        for text, spans in spans_by_chunk.items()
        for s in spans
        if not s or s not in text or (known is not None and text not in known)
    )


def covered_chars(text: str, spans: Iterable[str]) -> int:
    """Characters of text covered by the union of the spans (first occurrence of each)."""
    intervals = sorted((i, i + len(span)) for span in spans if span and (i := text.find(span)) >= 0)
    total, end = 0, 0
    for start, stop in intervals:
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def score_prediction(
    spans_by_chunk: Dict[str, List[str]], results: List[Any], query: Dict[str, Any]
) -> Dict[str, Any]:
    """Per-query numbers for one setting. Identical span texts (overlapping chunks) count once
    for F1; gold_chars is the union of covered characters in gold chunks."""
    ids_by_text: Dict[str, set] = defaultdict(set)
    for r in results:
        ids_by_text[r.text].add(r.id)
    gold_ids = set(query.get("gold_chunk_ids") or [])
    n_spans = gold_chunk_spans = gold_chars = distractors_answered = 0
    gold_answered = False
    unique: List[str] = []
    for text, spans in spans_by_chunk.items():
        from_gold = bool(ids_by_text.get(text, set()) & gold_ids)
        if from_gold:
            gold_chars += covered_chars(text, spans)
            gold_answered = gold_answered or bool(spans)
        elif spans:
            distractors_answered += 1
        for span in spans:
            n_spans += 1
            gold_chunk_spans += from_gold
            if span not in unique:
                unique.append(span)
    out = {
        "n_spans": n_spans,
        "gold_chunk_spans": gold_chunk_spans,
        "verbatim_failures": verbatim_failures(spans_by_chunk, (r.text for r in results)),
        "gold_chars": gold_chars,
        "gold_answered": gold_answered,
        "distractors_answered": distractors_answered,
    }
    if query.get("answerable") and query.get("gold_spans"):
        out["f1"] = word_f1(unique, [g["text"] for g in query["gold_spans"]])
    return out


def summarize(rows: List[Dict[str, Any]], setting: str, mode: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    if not rows:
        return out
    preds = [r["settings"][setting] for r in rows]

    def answered(row):
        return row["settings"][setting]["n_spans"] > 0

    n_spans = sum(p["n_spans"] for p in preds)
    failures = sum(p["verbatim_failures"] for p in preds)
    out["verbatim_failures"] = failures
    out["verbatim_pass_rate"] = 1 - failures / n_spans if n_spans else None
    out["extractor_errors"] = sum(p.get("extractor_errors", 0) for p in preds)
    if any("dropped_spans" in p for p in preds):
        out["dropped_spans"] = sum(p.get("dropped_spans", 0) for p in preds)

    ans = [r for r in rows if r["answerable"]]
    out["n_answerable"] = len(ans)
    out["answer_rate"] = _mean(answered(r) for r in ans)
    out["mean_spans"] = _mean(r["settings"][setting]["n_spans"] for r in ans)
    out["mean_gold_spans"] = _mean(r["settings"][setting]["gold_chunk_spans"] for r in ans)

    if mode == "oracle":
        out["gold_answer_rate"] = _mean(r["settings"][setting]["gold_answered"] for r in ans)
        out["coverage"] = _mean(
            r["settings"][setting]["gold_chars"] / r["passage_chars"]
            for r in ans
            if r["passage_chars"]
        )
        n_distractors = sum(r.get("n_distractors", 0) for r in ans)
        if n_distractors:
            out["distractor_answer_rate"] = (
                sum(r["settings"][setting]["distractors_answered"] for r in ans) / n_distractors
            )
            out["span_gold_chunk_precision"] = (
                sum(r["settings"][setting]["gold_chunk_spans"] for r in ans) / n_spans
                if n_spans
                else None
            )
    else:
        unans = [r for r in rows if not r["answerable"]]
        out["n_unanswerable"] = len(unans)
        out["abstention_accuracy"] = _mean(answered(r) == r["answerable"] for r in rows)
        out["false_abstention_rate"] = _mean(not answered(r) for r in ans)
        out["false_answer_rate"] = _mean(answered(r) for r in unans)
        out["gold_in_context"] = _mean(r["gold_in_context"] for r in ans)
        out["answer_rate_gold_in_context"] = _mean(answered(r) for r in ans if r["gold_in_context"])
        ans_spans = sum(r["settings"][setting]["n_spans"] for r in ans)
        out["span_gold_chunk_precision"] = (
            sum(r["settings"][setting]["gold_chunk_spans"] for r in ans) / ans_spans
            if ans_spans
            else None
        )

    auc = [r["auc"] for r in rows if r.get("auc") and r["auc"][1]]
    if auc:
        out["chunk_auc"] = sum(w for w, _ in auc) / sum(n for _, n in auc)

    f1s = [p["f1"] for p in preds if "f1" in p]
    out["n_gold"] = len(f1s)
    for key in ("precision", "recall", "f1"):
        out[key] = _mean(f[key] for f in f1s)
    return out


def breakdown(rows: List[Dict[str, Any]], setting: str, mode: str, field: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {value: summarize(group, setting, mode) for value, group in sorted(groups.items())}


def setting_names(extractor: str, thresholds: List[float]) -> List[str]:
    if extractor in SCORING_EXTRACTORS:
        return [f"{extractor}@{t:g}" for t in thresholds]
    return [extractor]


def check_index_meta(run_dir: Path, expected: Dict[str, Any]) -> None:
    """Raise unless run_dir holds a completed index built with the expected settings."""
    path = run_dir / INDEX_META
    if not path.exists():
        raise ValueError(f"{path} missing: index incomplete or built before index_meta existed")
    if not (run_dir / "milvus.db").exists():
        raise ValueError(f"{run_dir / 'milvus.db'} missing next to {INDEX_META}")
    found = json.loads(path.read_text(encoding="utf-8"))
    if found != expected:
        diff = {
            k: {"index": found.get(k), "run": expected.get(k)}
            for k in sorted(set(found) | set(expected))
            if found.get(k) != expected.get(k)
        }
        raise ValueError(f"index {run_dir} does not match this run: {json.dumps(diff)}")


# ---- data ------------------------------------------------------------------


def oracle_results(
    query: Dict[str, Any], distractors: Iterable[Dict[str, Any]] = (), seed: int = 0
) -> List[Any]:
    """The query's own passage, plus distractor chunks in a seeded random order."""
    results = [SimpleNamespace(id=query["source_chunk_id"], text=query["passage"])]
    results += [SimpleNamespace(id=c["chunk_id"], text=c["text"]) for c in distractors]
    random.Random(f"{seed}:{query['qid']}:order").shuffle(results)
    return results


def distractor_pools(corpus: List[Dict[str, Any]]) -> Dict[str, List[Tuple[Dict, frozenset]]]:
    pools: Dict[str, List[Tuple[Dict, frozenset]]] = defaultdict(list)
    for chunk in corpus:
        pools[chunk["doc_id"]].append((chunk, frozenset(shingles(chunk["text"]))))
    return pools


def sample_distractors(
    query: Dict[str, Any], pools: Dict[str, List[Tuple[Dict, frozenset]]], k: int, seed: int
) -> List[Dict[str, Any]]:
    """k hard negatives: chunks of the same document that share no 8-word shingle with the
    passage, so they cannot contain its text. Other documents fill in when the same one runs
    short. Seeded per qid, so every variant sees the same distractors."""
    rng = random.Random(f"{seed}:{query['qid']}")
    passage = shingles(query["passage"])
    exclude = set(query.get("gold_chunk_ids") or []) | {query["source_chunk_id"]}

    def eligible(pairs):
        return [c for c, sh in pairs if c["chunk_id"] not in exclude and not (sh & passage)]

    picked = rng.sample(same := eligible(pools.get(query["doc_id"], [])), min(k, len(same)))
    if len(picked) < k:
        others = eligible(
            p for doc, pairs in pools.items() if doc != query["doc_id"] for p in pairs
        )
        picked += rng.sample(others, min(k - len(picked), len(others)))
    return picked


def chunk_auc_pairs(
    scores: Dict[str, List[Tuple[str, float]]],
    results: List[Any],
    gold_ids: set,
    passage: Optional[str] = None,
) -> Tuple[float, int]:
    """Threshold-free: how often a gold chunk's best sentence outscores a non-gold chunk's
    best sentence (ties count half). Returns (wins, pairs).

    With `passage`, non-gold chunks sharing an 8-word shingle with it are left out: they can
    hold the answer sentence itself (overlap below the gold threshold), which would make a
    perfect judge tie. Oracle distractors never share one, so both modes count the same pairs.
    """
    passage_sh = shingles(passage) if passage else set()
    best = {r.id: max((p for _, p in scores.get(r.text, [])), default=0.0) for r in results}
    gold = [v for r in results if r.id in gold_ids for v in [best[r.id]]]
    other = [
        best[r.id]
        for r in results
        if r.id not in gold_ids and not (passage_sh and shingles(r.text) & passage_sh)
    ]
    wins = sum((g > o) + 0.5 * (g == o) for g in gold for o in other)
    return wins, len(gold) * len(other)


def parse_thresholds(value: str) -> List[float]:
    thresholds = sorted({float(v) for v in value.split(",") if v.strip()})
    if not thresholds or not all(0.0 <= t <= 1.0 for t in thresholds):
        raise argparse.ArgumentTypeError(f"thresholds must be in [0, 1]: {value!r}")
    return thresholds


# ---- extractors ------------------------------------------------------------


def make_extractor(args, nlp=None):
    if args.extractor == "overlap":
        if nlp is None:
            import spacy

            nlp = spacy.load(args.bm25_model, exclude=["parser", "ner"])
        return OverlapScorer(nlp)
    if args.extractor == "crossencoder":
        from verbatim_rag.cross_encoder_extractor import CrossEncoderSentenceExtractor

        if not args.ce_revision:
            logger.warning(
                "--extractor crossencoder without --ce-revision: checkpoint is not pinned"
            )
        return CrossEncoderSentenceExtractor(
            model=args.ce_model,
            revision=args.ce_revision,
            device=args.ce_device,
            batch_size=args.ce_batch_size,
            max_length=args.ce_max_length,
            spacy_model=args.bm25_model,
            nlp=nlp,
        )
    from verbatim_core.extractors import LLMSpanExtractor
    from verbatim_core.llm_client import LLMClient

    client = LLMClient(model=args.llm_model, temperature=0.0, api_base=args.api_base)
    # Always one batched prompt per query, so results compare across --distractors and
    # --extract-k ("auto" switches to per-document prompts above its batch_size).
    return LLMSpanExtractor(llm_client=client, extraction_mode="batch", batch_size=1000)


class OverlapScorer:
    """Eval-only baseline, no model: a sentence scores the share of the question's words of 4+
    characters (lowercased) that it contains. Same HuSpaCy sentences as crossencoder, so it is the
    floor a judge variant has to beat. Flattered here: synthetic queries are written from
    their passage."""

    def __init__(self, nlp):
        self.nlp = nlp

    @staticmethod
    def _words(text: str) -> set:
        return {w.lower() for w in _WORD_RE.findall(text)}

    def score_sentences(self, question: str, search_results: List[Any]):
        from verbatim_rag.cross_encoder_extractor import CrossEncoderSentenceExtractor

        qwords = {w for w in self._words(question) if len(w) > 3}
        texts = [getattr(r, "text", "") or "" for r in search_results]
        scores: Dict[str, List[Tuple[str, float]]] = {text: [] for text in texts}
        todo = [text for text in dict.fromkeys(texts) if text.strip()]
        for text, doc in zip(todo, self.nlp.pipe(todo, batch_size=64)):
            for start, end in CrossEncoderSentenceExtractor._sentence_offsets(text, doc):
                sentence = text[start:end]
                overlap = len(self._words(sentence) & qwords) / len(qwords) if qwords else 0.0
                scores[text].append((sentence, overlap))
        return scores

    @staticmethod
    def spans_from_scores(scores, threshold: float) -> Dict[str, List[str]]:
        return {text: [s for s, p in scored if p >= threshold] for text, scored in scores.items()}


# LLMSpanExtractor log lines that mean spans were lost: a per-document call failed, or the
# LLM client gave up on a whole call. "Batch extraction failed" is not counted: the
# extractor then retries each document individually.
_LOST_PREFIXES = (
    "Individual fallback extraction failed",
    "Individual extraction failed",
    "Async individual fallback extraction failed",
    "Async individual extraction failed",
    "Span extraction failed",
    "Async span extraction failed",
)


class LogCounter(logging.Handler):
    """Counts what LLMSpanExtractor only logs: failed extraction calls (it then returns no
    spans, which would read as abstention) and spans dropped by its verbatim check."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.errors = 0
        self.dropped = 0

    def emit(self, record):
        msg = str(record.msg)
        if msg.startswith("Span not found"):
            self.dropped += 1
        elif msg.startswith(_LOST_PREFIXES):
            self.errors += 1


def predict(
    extractor, args, question: str, results: List[Any]
) -> Tuple[Dict[str, Dict[str, List[str]]], Optional[Dict[str, List[Tuple[str, float]]]]]:
    """Spans per setting, plus the sentence scores of a scoring extractor (None for the LLM)."""
    if args.extractor in SCORING_EXTRACTORS:
        scores = extractor.score_sentences(question, results)
        names = setting_names(args.extractor, args.thresholds)
        return {
            name: extractor.spans_from_scores(scores, t) for name, t in zip(names, args.thresholds)
        }, scores
    return {"llm": extractor.extract_spans(question, results)}, None


# ---- retrieval -------------------------------------------------------------


def open_index(run_dir: Path, corpus, sparse, dense, args, reuse: bool):
    from verbatim_rag.index import VerbatimIndex
    from verbatim_rag.vector_stores import LocalMilvusStore

    db_path = run_dir / "milvus.db"
    if not reuse:
        index = build_index(corpus, db_path, sparse, dense, args.batch_size)
        write_index_meta(run_dir, index_meta(corpus, args, dense is not None, sparse is not None))
        return index
    if sparse is not None and not sparse._provider.fitted_on(row["text"] for row in corpus):
        raise ValueError(f"BM25 stats in {run_dir} were not fitted on this corpus")
    store = LocalMilvusStore(
        db_path=str(db_path),
        dense_dim=dense.get_dimension() if dense else 384,
        enable_dense=dense is not None,
        enable_sparse=sparse is not None,
    )
    rows = store.client.query(
        collection_name=store.collection_name, filter="", output_fields=["count(*)"]
    )[0]["count(*)"]
    if rows != len(corpus):
        raise ValueError(f"index {db_path} holds {rows} chunks, corpus has {len(corpus)}")
    return VerbatimIndex(vector_store=store, dense_provider=dense, sparse_provider=sparse)


def retrieve(index, question: str, args, reranker=None) -> List[Any]:
    if not (args.skip_dense or args.skip_sparse):
        weights = {"dense": args.dense_weight, "sparse": args.sparse_weight}
        hits = index.query(question, k=args.retrieve_k, hybrid_weights=weights, rrf_k=args.rrf_k)
    else:
        method = "sparse" if args.skip_dense else "dense"
        hits = index.query(question, k=args.retrieve_k, search_type=method)
    if reranker is not None:
        hits = reranker.rerank(question, hits)
    return hits[: args.extract_k]


# ---- run -------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--extractor", choices=EXTRACTORS, default="crossencoder")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--split", choices=["test", "dev", "train"], default="test")
    p.add_argument("--limit", type=int, help="Seeded random sample of N queries (smoke test)")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seeds the --limit sample, distractor choice and oracle result order",
    )
    p.add_argument("--tag", help="Name for this run; a fresh index goes to data/index/<tag>/")
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    p.add_argument(
        "--dump-predictions",
        action="store_true",
        help="Write per-query spans and scores to data/predictions/<tag>.jsonl",
    )
    g = p.add_argument_group("crossencoder")
    g.add_argument("--ce-model", default="BAAI/bge-reranker-v2-m3")
    g.add_argument("--ce-revision", help="HF commit sha to pin the cross-encoder checkpoint")
    g.add_argument("--ce-device", help="cuda / mps / cpu (default: sentence-transformers picks)")
    g.add_argument("--ce-batch-size", type=int, default=16)
    g.add_argument(
        "--ce-max-length", type=int, default=512, help="Token budget per (question, sentence)"
    )
    g.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.3,0.5,0.7,0.9"),
        help="Comma-separated sentence score cutoffs (crossencoder, overlap), each a setting",
    )
    p.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="oracle: add N same-document chunks sharing no 8-word shingle with the passage",
    )
    g = p.add_argument_group("llm")
    g.add_argument("--llm-model", default="gpt-4o-mini")
    g.add_argument("--api-base", default="https://api.openai.com/v1")
    g = p.add_argument_group("retrieved mode (same defaults as run_retrieval)")
    g.add_argument("--index-tag", help="Reuse data/index/<NAME>/ (checked via index_meta.json)")
    g.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rebuilding data/index/<tag>/ when it already holds a completed index",
    )
    g.add_argument("--retrieve-k", type=int, default=100)
    g.add_argument("--extract-k", type=int, default=5, help="Top chunks passed to the extractor")
    g.add_argument("--skip-dense", action="store_true", help="BM25 only; no fusion")
    g.add_argument("--skip-sparse", action="store_true", help="Dense only; no fusion")
    g.add_argument("--dense-model", default="Qwen/Qwen3-Embedding-0.6B")
    g.add_argument("--dense-revision", help="HF commit sha to pin the dense model")
    g.add_argument("--query-instruction", help="English task text replacing Qwen's query prompt")
    g.add_argument("--device", default="cpu", help="Device for the dense model and reranker")
    g.add_argument("--rerank", action="store_true", help="Rerank with Jina reranker v3.5")
    g.add_argument("--rerank-model", default="jinaai/jina-reranker-v3.5")
    g.add_argument("--rerank-revision", help="HF commit sha to pin the reranker (remote code)")
    g.add_argument(
        "--rerank-block-tokens",
        type=int,
        help="Approximate cap on tokens per reranker forward pass (4096 fits a 16 GB GPU); "
        "default: all candidates in one pass",
    )
    g.add_argument(
        "--bm25-model",
        default="hu_core_news_lg",
        help="HuSpaCy model for BM25 lemmas and sentence splitting",
    )
    g.add_argument("--k1", type=float, default=1.2)
    g.add_argument("--b", type=float, default=0.75)
    g.add_argument("--dense-weight", type=float, default=0.5)
    g.add_argument("--sparse-weight", type=float, default=0.5)
    g.add_argument("--rrf-k", type=int, default=60)
    g.add_argument("--batch-size", type=int, default=16, help="Indexing/embedding batch size")
    args = p.parse_args(argv)
    retrieved = args.mode == "retrieved"
    if retrieved and args.skip_dense and args.skip_sparse:
        p.error("nothing to retrieve with: both providers skipped")
    if not retrieved and args.index_tag:
        p.error("--index-tag only applies to --mode retrieved")
    if args.distractors < 0 or (retrieved and args.distractors):
        p.error("--distractors takes N >= 0 and only applies to --mode oracle")
    if args.extract_k < 1 or args.retrieve_k < args.extract_k:
        p.error("need 1 <= --extract-k <= --retrieve-k")
    if args.rerank_block_tokens is not None and not args.rerank:
        p.error("--rerank-block-tokens needs --rerank")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = datetime.now(timezone.utc)
    tag = args.tag or started.strftime("%Y%m%d-%H%M%S")

    split, n_excluded = load_split(args.data_dir / "queries.jsonl", args.split)
    queries = split if retrieved else [q for q in split if q["answerable"]]
    n_skipped = len(split) - len(queries)
    if args.limit is not None:
        queries = random.Random(args.seed).sample(queries, min(args.limit, len(queries)))
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    logger.info(
        "%s mode, %s split: %d queries (%d unanswerable skipped, %d excluded)",
        args.mode,
        args.split,
        len(queries),
        n_skipped,
        n_excluded,
    )

    corpus: List[Dict[str, Any]] = []
    sparse = dense = reranker = None
    run_dir: Optional[Path] = None
    reuse = False
    pools = None
    if not retrieved and args.distractors:
        pools = distractor_pools(read_jsonl(args.data_dir / "corpus.jsonl"))
    if retrieved:
        corpus = read_jsonl(args.data_dir / "corpus.jsonl")
        if args.index_tag:
            run_dir = args.data_dir / "index" / args.index_tag
            expected = index_meta(corpus, args, not args.skip_dense, not args.skip_sparse)
            try:
                check_index_meta(run_dir, expected)
            except ValueError as e:
                p.error(str(e))
            reuse = True
        else:
            run_dir = args.data_dir / "index" / tag
            if (run_dir / INDEX_META).exists() and not args.overwrite:
                p.error(
                    f"{run_dir} holds a completed index: reuse it with --index-tag {tag}, "
                    "or pass --overwrite to rebuild"
                )
            shutil.rmtree(run_dir, ignore_errors=True)  # never reuse unchecked stats
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
        if args.rerank:
            if not args.rerank_revision:
                logger.warning("--rerank without --rerank-revision: remote code is not pinned")
            from verbatim_rag.rerankers import JinaV3Reranker

            reranker = JinaV3Reranker(
                model=args.rerank_model,
                rerank_k=args.retrieve_k,
                device=args.device,
                revision=args.rerank_revision,
                max_block_tokens=args.rerank_block_tokens,
            )

    # Load the extractor before indexing so a bad model id fails in seconds, not hours.
    extractor = make_extractor(args, nlp=sparse._provider.nlp if sparse else None)
    counter = None
    if args.extractor == "llm":
        extractor.llm_client.complete("Reply with OK.")  # raises on a bad key, model or api-base
        counter = LogCounter()
        logging.getLogger("verbatim_core").addHandler(counter)
    index = None
    index_seconds = 0.0
    if retrieved:
        t0 = time.time()
        index = open_index(run_dir, corpus, sparse, dense, args, reuse)
        index_seconds = time.time() - t0

    settings = setting_names(args.extractor, args.thresholds)
    dump = None
    if args.dump_predictions:
        dump_path = args.data_dir / "predictions" / f"{tag}.jsonl"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump = open(dump_path, "w", encoding="utf-8")

    rows = []
    extract_seconds = retrieve_seconds = 0.0
    try:
        for i, q in enumerate(queries, start=1):
            t0 = time.time()
            if retrieved:
                results = retrieve(index, q["query"], args, reranker)
            else:
                distractors = (
                    sample_distractors(q, pools, args.distractors, args.seed) if pools else []
                )
                results = oracle_results(q, distractors, args.seed)
            retrieve_seconds += time.time() - t0
            t0 = time.time()
            errors_before, dropped_before = (counter.errors, counter.dropped) if counter else (0, 0)
            preds, scores = predict(extractor, args, q["query"], results)
            extract_seconds += time.time() - t0

            gold_ids = set(q.get("gold_chunk_ids") or [])
            row = {
                **{f: q.get(f) for f in RETRIEVED_BREAKDOWNS},
                "answerable": bool(q["answerable"]),
                "gold_in_context": any(r.id in gold_ids for r in results)
                if q["answerable"]
                else None,
                "passage_chars": None if retrieved else len(q["passage"]),
                "n_distractors": sum(r.id not in gold_ids for r in results),
                "auc": chunk_auc_pairs(scores, results, gold_ids, q.get("passage"))
                if scores is not None and q["answerable"]
                else None,
                "settings": {s: score_prediction(preds[s], results, q) for s in settings},
            }
            if counter is not None:
                llm = row["settings"]["llm"]
                llm["extractor_errors"] = counter.errors - errors_before
                llm["dropped_spans"] = counter.dropped - dropped_before
                if llm["extractor_errors"]:
                    logger.error(
                        "qid=%s: %d failed extraction calls, their spans are lost",
                        q["qid"],
                        llm["extractor_errors"],
                    )
            for s in settings:
                if row["settings"][s]["verbatim_failures"]:
                    logger.error(
                        "verbatim assertion failed: qid=%s setting=%s spans=%d",
                        q["qid"],
                        s,
                        row["settings"][s]["verbatim_failures"],
                    )
            rows.append(row)
            if dump is not None:
                record = {
                    "qid": q["qid"],
                    "chunks": [
                        {
                            "id": r.id,
                            "spans": {s: preds[s].get(r.text, []) for s in settings},
                            **({"scores": scores.get(r.text, [])} if scores is not None else {}),
                        }
                        for r in results
                    ],
                }
                dump.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                logger.info("extracted %d/%d", i, len(queries))
    finally:
        if dump is not None:
            dump.close()
        if counter is not None:
            logging.getLogger("verbatim_core").removeHandler(counter)

    fields = RETRIEVED_BREAKDOWNS if retrieved else BREAKDOWNS
    results_by_setting = {
        s: {
            "overall": summarize(rows, s, args.mode),
            **{f: breakdown(rows, s, args.mode, f) for f in fields},
        }
        for s in settings
    }
    extractor_config: Dict[str, Any]
    if args.extractor == "crossencoder":
        extractor_config = {
            "model": args.ce_model,
            "revision": args.ce_revision,  # None = latest at run time; pin it
            "batch_size": args.ce_batch_size,
            "max_length": args.ce_max_length,
            "activation": "sigmoid",
            "device": str(extractor.model.device),
            "thresholds": args.thresholds,
            "spacy_model": args.bm25_model,
            "spacy_model_version": extractor.nlp.meta.get("version"),
        }
    elif args.extractor == "overlap":
        extractor_config = {
            "thresholds": args.thresholds,
            "spacy_model": args.bm25_model,
            "spacy_model_version": extractor.nlp.meta.get("version"),
        }
    else:
        extractor_config = {
            "model": args.llm_model,
            "api_base": args.api_base,
            "temperature": 0,
            "extraction_mode": "batch",
        }
    log = {
        "tag": tag,
        "started": started.isoformat(),
        "git_sha": git_sha(),
        "config": {
            "mode": args.mode,
            "extractor": args.extractor,
            "split": args.split,
            "limit": args.limit,
            "seed": args.seed,
            "distractors": args.distractors,
            args.extractor: extractor_config,
            "retrieval": None
            if not retrieved
            else {
                "index_dir": str(run_dir),
                "index_reused": reuse,
                "index": index_meta(corpus, args, dense is not None, sparse is not None),
                "query_instruction": args.query_instruction,
                "fusion": None
                if args.skip_dense or args.skip_sparse
                else {
                    "dense_weight": args.dense_weight,
                    "sparse_weight": args.sparse_weight,
                    "rrf_k": args.rrf_k,
                },
                "retrieve_k": args.retrieve_k,
                "extract_k": args.extract_k,
                "rerank": None
                if reranker is None
                else {
                    "model": args.rerank_model,
                    "revision": args.rerank_revision,
                    "max_block_tokens": args.rerank_block_tokens,
                },
                "device": args.device,
            },
        },
        "dataset": {
            "revision": manifest.get("revision"),
            "queries": len(queries),
            "unanswerable_skipped": n_skipped,
            "excluded_by_annotation": n_excluded,
        },
        "empty_sparse_queries": None if sparse is None else sparse.empty,
        "seconds": {
            "index": round(index_seconds),
            "retrieve": round(retrieve_seconds),
            "extract": round(extract_seconds),
            "extract_ms_per_query": round(1000 * extract_seconds / len(queries))
            if queries
            else None,
        },
        "results": results_by_setting,
    }
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = args.runs_dir / f"extraction-{tag}.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    if retrieved:
        columns = ["answer_rate", "abstention_accuracy", "false_answer_rate", "gold_in_context"]
        columns += ["span_gold_chunk_precision", "chunk_auc", "mean_spans"]
    else:
        columns = ["gold_answer_rate", "coverage", "mean_gold_spans"]
        if args.distractors:
            columns += ["distractor_answer_rate", "span_gold_chunk_precision", "chunk_auc"]
    columns += ["verbatim_pass_rate", "f1"]

    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "-"

    print(f"\n{args.mode} mode, {args.split} split, {len(queries)} queries -> {out}")
    print(f"{'setting':16} " + " ".join(f"{c[:12]:>12}" for c in columns) + f" {'n_gold':>7}")
    for s in settings:
        o = results_by_setting[s]["overall"]
        print(f"{s:16} " + " ".join(f"{fmt(o.get(c)):>12}" for c in columns) + f" {o['n_gold']:>7}")
        if o["extractor_errors"]:
            print(f"WARNING {s}: {o['extractor_errors']} extractor errors scored as no spans")
        if o.get("dropped_spans"):
            print(f"{s}: {o['dropped_spans']} LLM spans dropped as not verbatim")


if __name__ == "__main__":
    main()
