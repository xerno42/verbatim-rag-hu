"""
Hungarian BM25 as a sparse-vector provider for VerbatimRAG on Milvus Lite.

Milvus Lite has no native BM25, so BM25 is computed client-side and stored in
the existing `sparse_vector` field (IP metric). The formula is split so that
the inner product of query and document vectors equals the BM25 score:

    doc vector   : tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    query vector : idf(t)

IDF lives on the query side, so adding documents never requires re-embedding
existing ones -- only the stats file changes.

VerbatimIndex calls embed_batch() at ingest and embed_text() at query time:
    embed_batch -> document side
    embed_text  -> query side

Requires milvus-lite<3: milvus-lite 3.x does not return the exact inner product
for SPARSE_FLOAT_VECTOR/IP search, which breaks the BM25 ranking.
"""

import hashlib
import json
import logging
import math
import os
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from verbatim_rag.embedding_providers import SparseEmbeddingProvider
from verbatim_rag.hungarian_accents import fold

logger = logging.getLogger(__name__)

# Bump whenever _term() changes: stats fitted with another analyzer are rejected.
ANALYZER_VERSION = 2

# Hungarian attaches suffixes to abbreviations, numbers and symbols with a
# hyphen ("EU-s", "GDPR-t", "2020-ban"); tails up to this length are dropped.
_MAX_HYPHEN_SUFFIX = 4


class HungarianBM25Provider(SparseEmbeddingProvider):
    def __init__(
        self,
        stats_path: str,
        model: str = "hu_core_news_lg",
        k1: float = 1.2,
        b: float = 0.75,
        batch_size: int = 64,
        nlp: Optional[Any] = None,
    ):
        """
        Args:
            stats_path: Corpus statistics file. Keep it next to the Milvus db it
                belongs to; it is derived from the corpus, so mind its licence.
            model: HuSpaCy model, loaded with parser and NER excluded.
            k1, b: BM25 parameters, baked into document vectors at index time.
            batch_size: nlp.pipe batch size.
            nlp: Preloaded spaCy pipeline (overrides `model`), e.g. to share one
                pipeline with the sentence splitter.
        """
        if nlp is None:
            import spacy

            nlp = spacy.load(model, exclude=["parser", "ner"])
            if not any("lemmatizer" in name for name in nlp.pipe_names):
                raise ValueError(f"{model} has no lemmatizer component: {nlp.pipe_names}")
        self.nlp = nlp
        self.stats_path = Path(stats_path)
        self.k1, self.b = k1, b
        self.batch_size = batch_size
        self.n_docs = 0
        self.avgdl = 0.0
        self.df: Counter = Counter()
        self.fingerprint: Optional[str] = None
        self._stale_reason: Optional[str] = None
        if self.stats_path.exists():
            self._load()

    # ---- analysis ---------------------------------------------------------

    @staticmethod
    def _term(t) -> Optional[str]:
        if t.is_stop:
            return None
        if t.is_alpha:
            return t.lemma_.lower()
        # Numbers, citations, abbreviations, hyphenated forms ("1990.", "6:519.",
        # "Ptk.", "EU-s"). Lemmas are unreliable here ("LXV." -> "00."), so use
        # the surface form without trailing periods or a hyphenated suffix.
        form = t.text.lower().rstrip(".")
        head, sep, tail = form.rpartition("-")
        if sep and head and tail.isalpha() and len(tail) <= _MAX_HYPHEN_SUFFIX:
            form = head.rstrip(".")
        return form if any(c.isalnum() for c in form) else None

    def _terms(self, doc) -> List[str]:
        return [term for term in map(self._term, doc) if term]

    def _analyze(self, texts: Iterable[str]) -> List[List[str]]:
        return [self._terms(d) for d in self.nlp.pipe(texts, batch_size=self.batch_size)]

    @staticmethod
    def _term_id(term: str) -> int:
        # Stable across processes (unlike hash()), fits Milvus sparse indices.
        return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF

    @staticmethod
    def _fingerprint(texts: Iterable[str]) -> str:
        # Order-independent, so re-ordered ingestion of the same chunks still matches.
        digests = sorted(hashlib.sha256(t.encode("utf-8")).digest() for t in texts)
        return hashlib.sha256(b"".join(digests)).hexdigest()

    def _config(self) -> Dict[str, Any]:
        meta = getattr(self.nlp, "meta", {})
        model = f"{meta.get('lang', '?')}_{meta.get('name', '?')}-{meta.get('version', '?')}"
        return {"analyzer_version": ANALYZER_VERSION, "model": model, "k1": self.k1, "b": self.b}

    # ---- corpus statistics ------------------------------------------------

    def fit(self, texts: List[str]) -> None:
        """Compute N, avgdl and document frequencies, and save them.

        `texts` must be exactly the strings VerbatimIndex passes to embed_batch:
        each chunk's `enhanced_content` (chunk text plus the "# title" header and
        the Document/Source/metadata footer), not the raw chunk text. Fitting on
        raw text makes avgdl too small and over-penalizes every document vector
        without any error. Title and metadata values are therefore BM25 terms too.
        """
        if not texts:
            raise ValueError("fit() needs at least one chunk")
        docs = self._analyze(texts)
        self.n_docs = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.n_docs
        self.df = Counter(term for d in docs for term in set(d))
        self.fingerprint = self._fingerprint(texts)
        self._stale_reason = None
        self._log_collisions()
        self._save()

    def fitted_on(self, texts: Iterable[str]) -> bool:
        """True if the loaded stats were fitted on exactly these chunk strings."""
        return self.fingerprint is not None and self.fingerprint == self._fingerprint(texts)

    def _log_collisions(self) -> None:
        seen: Dict[int, str] = {}
        collisions = []
        for term in self.df:
            other = seen.setdefault(self._term_id(term), term)
            if other != term:
                collisions.append((other, term))
        if collisions:
            logger.warning(
                f"{len(collisions)} BM25 term-id collisions (weights merge): {collisions[:5]}"
            )

    def _save(self) -> None:
        payload = {
            "config": self._config(),
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": self.fingerprint,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "df": self.df,
        }
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.stats_path.with_name(self.stats_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.stats_path)

    def _load(self) -> None:
        data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.n_docs = data["n_docs"]
        self.avgdl = data["avgdl"]
        self.df = Counter(data["df"])
        self.fingerprint = data.get("fingerprint")
        saved, current = data.get("config"), self._config()
        if saved != current:
            self._stale_reason = (
                f"{self.stats_path} was fitted with {saved}, provider is {current}; "
                "call fit() and re-index"
            )
            logger.warning(self._stale_reason)
        else:
            logger.info(
                f"Loaded BM25 stats from {self.stats_path}: {self.n_docs} chunks, "
                f"avgdl {self.avgdl:.1f}, fitted {data.get('fitted_at')}"
            )

    def _require_fit(self) -> None:
        if self._stale_reason:
            raise RuntimeError(f"HungarianBM25Provider: {self._stale_reason}")
        if self.n_docs == 0:
            raise RuntimeError("HungarianBM25Provider: call fit() on the chunk corpus first")
        if self.avgdl == 0:
            raise RuntimeError(
                f"HungarianBM25Provider: fit() saw {self.n_docs} chunks but none had "
                "indexable terms"
            )

    # ---- SparseEmbeddingProvider interface --------------------------------

    def embed_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """Document side: saturated, length-normalized term frequencies.

        A chunk with no indexable terms gets {} (Milvus Lite accepts it).
        """
        self._require_fit()
        out = []
        for terms in self._analyze(texts):
            tf = Counter(terms)
            norm = self.k1 * (1 - self.b + self.b * len(terms) / self.avgdl)
            out.append({self._term_id(t): c * (self.k1 + 1) / (c + norm) for t, c in tf.items()})
        return out

    def embed_text(self, text: str) -> Dict[int, float]:
        """Query side: IDF weight per unique query term.

        Returns {} when no query term occurs in the corpus; the caller must then
        drop "sparse" from the hybrid weights instead of searching with it.
        """
        self._require_fit()
        vec = {}
        for t in set(self._analyze([text])[0]):
            df = self.df.get(t, 0)
            if df:
                vec[self._term_id(t)] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
        return vec

    def get_dimension(self) -> int:
        return 2**31


# Bump whenever HungarianFoldedBM25Provider._term() changes.
FOLDED_ANALYZER_VERSION = 1


class HungarianFoldedBM25Provider(HungarianBM25Provider):
    """BM25 over accent-folded surface forms, for questions typed without accents.

    A second sparse channel next to the lemma one: "szerzodes" matches "szerződés", which the
    lemma channel misses because HuSpaCy cannot lemmatize the unaccented form. Surface forms,
    not lemmas, for the same reason; inflected forms therefore only match themselves.

    Stop words are dropped by their folded form on both sides, so the unaccented "es" goes
    the same way as "és". The price: content words whose folded form is a folded stop word
    ("hat" six vs "hát", "fele" half vs "felé", "tobb") are dropped from this channel too; the
    lemma channel still has them. Its stats are tagged with analyzer "folded", so a lemma stats file is
    never loaded by this provider or the other way round.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._folded_stops = {fold(w) for w in self.nlp.Defaults.stop_words}

    def _term(self, t) -> Optional[str]:
        if t.is_alpha:
            if t.is_stop:
                return None
            form = fold(t.text)
        else:
            term = HungarianBM25Provider._term(t)
            form = fold(term) if term else None
        return None if form in self._folded_stops else form

    def _config(self) -> Dict[str, Any]:
        return {
            **super()._config(),
            "analyzer": "folded",
            "folded_analyzer_version": FOLDED_ANALYZER_VERSION,
        }
