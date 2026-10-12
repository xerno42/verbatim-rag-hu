# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Hungarian-language fork of [VerbatimRAG](https://github.com/KRLabsOrg/verbatim-rag) ("verbatim-rag-hu"): extractive QA where every answer sentence is copied verbatim from a source chunk and cited by document + character offsets. Hallucination is prevented structurally (no LLM writes factual content), not detected afterwards. The failure mode is "picked the wrong sentence", which is measurable against gold spans.

The approach is to swap four components through the repo's existing interfaces instead of forking core code: dense embeddings, sparse embeddings, reranker, span extractor. `DocumentProcessor`, `VerbatimIndex`, `LocalMilvusStore` and the template manager are used as shipped. Single machine, Milvus Lite only. The FastAPI/React parts (`api/`, `frontend/`) are prototypes outside the compatibility gate and out of scope for v1.

### Implementation status vs. the spec

Only some of the spec exists in code. Check before assuming a class is there.

| Piece | Status |
|---|---|
| `HungarianBM25Provider` ([verbatim_rag/hungarian_bm25.py](verbatim_rag/hungarian_bm25.py)) | Written, committed |
| Retrieval eval runner ([evaluation/hu_legal/run_retrieval.py](evaluation/hu_legal/run_retrieval.py)) | Written; Recall/MRR for sparse, dense, fused, plus `reranked` with `--rerank` (Jina v3.5 on the fused top-100; not yet run on the eval set). No F1, abstention or assertion metrics yet |
| Hungarian eval set builder ([evaluation/hu_legal/](evaluation/hu_legal/)) | Committed, with tests |
| Qwen3-Embedding dense provider (subclass of `SentenceTransformersProvider`, `prompt_name="query"` in `embed_text` only, dim 1024) | Not yet written |
| Reranker: `JinaV3Reranker(BaseReranker)` in [rerankers.py](verbatim_rag/rerankers.py), default `jinaai/jina-reranker-v3.5` (0.6B listwise, CC-BY-NC-4.0, `trust_remote_code`; `rerank_k=100` in, tail kept in retrieval order) | Written, tests mock `transformers`. Replaces the planned `Qwen3Reranker-0.6B`; the sentence-level extractor still needs a per-sentence scorer (listwise scores are only comparable within one call) |
| `SentenceRerankExtractor(SpanExtractor)` (HuSpaCy sentence split → per-sentence relevance scorer, undecided now that the reranker is listwise Jina → threshold) | Not yet written |
| Verbatim assertion (every emitted sentence is an exact substring of its source chunk; failure is logged, never shown) and fixed Hungarian abstention message | Not yet written |

Phases: 0 eval set → 1 baseline (Qwen3 + Hungarian BM25 + Jina v3.5 reranker + sentence-rerank extractor, with `LLMSpanExtractor` via an OpenAI-compatible `api_base` as the quality ceiling) → 2 tune weights/cutoffs (4B models only on GPU) → 3 conditional (trained mmBERT/huBERT extractor, or a free-form path with a Hungarian LettuceDetect). Numeric targets are set only after the phase 1 baseline; every change to tokenization, models, weights or thresholds is run against the full eval set and logged with its config.

## Commands

```bash
# Setup (Python >=3.10). verbatim_rag imports verbatim_core, which lives in packages/core
pip install -e packages/core/
pip install -e ".[dev]"
python -c "import huspacy; huspacy.download('hu_core_news_lg')"   # model used by HungarianBM25Provider
# if that fails (old huspacy): pip install https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-3.8.1-py3-none-any.whl

# Tests / lint (same as CI)
pytest tests/ -v
pytest tests/test_hu_eval_set.py::TestResolveQuote -v      # single class/test
ruff check packages/core/verbatim_core/ verbatim_rag/ api/ tests/
ruff format --check packages/core/verbatim_core/ verbatim_rag/ api/ tests/

# Retrieval eval (BM25 / dense / fused; needs spacy, hu_core_news_lg and Qwen3-Embedding; CPU run takes hours)
python -m evaluation.hu_legal.run_retrieval [--split test] [--limit N] [--skip-dense] [--rerank] [--tag NAME]

# Build the Hungarian eval set (writes gitignored data/; CC BY-NC-ND 4.0 source licence)
python -m evaluation.hu_legal.build_eval_set [--revision <hf-sha>] [--seed N]
```

Ruff: line length 100, rules E/F/W/I, E501 ignored. pytest runs with `pythonpath = ["."]` (so `evaluation.*` imports work) and `asyncio_mode = "auto"`. CI only installs `packages/core` and runs network-free tests; anything needing HF models, spaCy models or Milvus must be tested separately and must not be added to the default CI path. The repo `.venv` currently lacks pytest and ruff (`uvx ruff` works); `pytest` and `ruff` otherwise come from `~/.local/bin`.

## Architecture

Two Python packages in one repo, with `verbatim_rag` layered on `verbatim_core`:

- `packages/core/verbatim_core/` is the lightweight reusable core (question + context → cited excerpts): span extractors (`SpanExtractor`, `LLMSpanExtractor`, `ModelSpanExtractor`, `SemanticHighlightExtractor`), `LLMClient`, templates, `ResponseBuilder`, `VerbatimTransform`.
- `verbatim_rag/` is the full RAG pipeline. Several modules (`extractors.py`, `llm_client.py`, `response_builder.py`) are thin re-exports of `verbatim_core`. Put new extractor logic in the core class hierarchy or subclass it; don't edit the re-export shims.

Query flow in `VerbatimRAG.query` ([verbatim_rag/core.py](verbatim_rag/core.py)): `VerbatimIndex.query` (hybrid search via the vector store) → optional `reranker.rerank(question, results)` (failures are swallowed with a warning and fall back to the original order) → `extractor.extract_spans(question, results)` returning `{chunk text: [spans]}` → spans split into display/citation-only (`max_display_spans`) → `TemplateManager` fills the template → `ResponseBuilder` builds the cited response. `template_mode="static"` is required for the Hungarian build: the default `"contextual"` has an LLM generate framing text.

Provider interfaces to implement, all in `verbatim_rag/`:
- `DenseEmbeddingProvider` / `SparseEmbeddingProvider` ([embedding_providers.py](verbatim_rag/embedding_providers.py)): `embed_text` (query side), `embed_batch` (document side), `get_dimension`. `VerbatimIndex` calls `embed_batch` at ingest and `embed_text` at query time, which is what makes the BM25 split below work.
- `Reranker` / `BaseReranker` ([rerankers.py](verbatim_rag/rerankers.py)): `rerank(question, results)`. `BaseReranker` provides `rerank_k` and `text_field` (`"text"` vs `"enhanced_text"`).
- `LocalMilvusStore` ([vector_stores/milvus_local.py](verbatim_rag/vector_stores/milvus_local.py)) with fusion in `BaseMilvusStore._hybrid_search_with_weights` (RRF, weights per method, spec starts at dense 0.5 / sparse 0.5, k=60). It forces `enable_full_text=False` because Milvus Lite has no native BM25, which is why BM25 is computed client-side.

### Hungarian BM25 invariants ([hungarian_bm25.py](verbatim_rag/hungarian_bm25.py))

BM25 is split so that the inner product (Milvus IP metric on `sparse_vector`) equals the BM25 score: document vectors carry the saturated, length-normalized TF (`embed_batch`), query vectors carry IDF (`embed_text`). Do not break these:

- `fit()` must run once before indexing, on exactly the chunk strings that will be passed to `embed_batch` (not whole documents), or length normalization is wrong. Stats persist to `bm25_hu_stats.json`; `embed_*` raise if unfitted.
- The identical analysis function (lowercased lemma, alphabetic or numeric tokens, stop words dropped) must run at index and query time. A mismatch silently halves recall.
- Term ids are `crc32(term) & 0x7FFFFFFF` (stable across processes; never use `hash()`).
- An empty query vector (all terms unseen) must drop `sparse` from the hybrid weights for that query rather than sending `{}` to Milvus. This is a spec requirement the caller must handle; the provider just returns `{}`.
- Re-running `fit()` updates IDF immediately; re-index only when average chunk length drifts, since `avgdl` is baked into document vectors.
- A Milvus collection's schema is fixed at creation. Qwen3-Embedding-0.6B is 1024-dim vs the repo default of 384 (`LocalMilvusStore(dense_dim=...)`), so use a new db file, never an existing index.

### Hungarian text processing

All Hungarian analysis goes through one HuSpaCy pipeline, `hu_core_news_lg`, with parser and NER disabled (lemmas for BM25, sentence boundaries for extraction). Use `nlp.pipe(texts, batch_size=64)` for indexing. Naive sentence splitters break on ordinals (`1990. évi`) and abbreviations (`kft.`, `stb.`), and every broken sentence becomes a broken answer. Known gap: compounds (`szerződés` vs `munkaszerződés`) are not matched by lemmatization; add a Hunspell decompounder only if the eval shows it.

### Evaluation set ([evaluation/hu_legal/README.md](evaluation/hu_legal/README.md))

Built from the Hungarian slice of the JudicialMind legal dataset. Index each corpus row as one chunk keyed by `chunk_id`, bypassing `DocumentProcessor` chunking so gold ids line up. Where the README differs from the original spec, the README wins: buckets A/B/C are passage lengths, not splits (splits are drawn separately); `gold_chunk_ids` include overlapping chunks; unanswerable questions remove whole overlap clusters from the corpus. Metrics: Recall@100 per method (BM25, dense, fused), Recall@10 and MRR after reranking, word-level F1 vs gold spans, abstention accuracy, verbatim assertion pass rate (target 100%). Split every metric by `query_type`, `legal_domain` and `difficulty`. `data/` and `annotations/` are gitignored because of the licence; check it before sharing derived data.

## Conventions

- Contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md): explain any change to span validation, citations, templates, schemas or defaults, and update [CHANGELOG.md](CHANGELOG.md) for user-facing behavior.
- Docker Compose (`docker compose up --build`) runs the API and frontend demo stack from `.env`; the container installs from the generated lock `docker/constraints.txt` (regeneration steps in `docker/overrides.txt`).

## Post implementation

Always run the code reviewer agent on the changes and fix its findings. Update this file after changing code in the repo; `AGENTS.md` is a symlink, don't change it. Do not commit the changes you made. Do NOT run the eval set.