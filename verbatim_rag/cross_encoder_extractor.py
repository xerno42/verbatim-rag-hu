"""
Sentence-level span extractor with a pointwise cross-encoder as the relevance judge.

Each chunk is split into sentences with HuSpaCy, and every (question, sentence) pair is
scored by a sentence-transformers CrossEncoder (BAAI/bge-reranker-v2-m3 by default:
XLM-RoBERTa-large, multilingual, Apache-2.0). Sentences scoring >= threshold are kept.

Spans are returned in relevance order, because VerbatimRAG flattens the dict in order and
displays the first `max_display_spans`: chunks by their best kept score, and spans within a
chunk by score (ties keep document order). Chunks without spans come last.

The judge is pointwise, unlike the listwise JinaV3Reranker: each pair is scored on its own,
so one threshold means the same for every question and the extractor can abstain (no
sentence passes). The raw logit goes through an explicit sigmoid, whatever activation the
checkpoint stores, so scores are in [0, 1] for every model. This assumes the head outputs a
logit, as relevance rerankers like bge do; a head that already outputs a probability would be
squashed into [0.5, 0.73]. Non-finite scores raise instead of reading as abstention.

Every emitted span is a slice of the chunk text at HuSpaCy sentence offsets, so it is an
exact substring of its source chunk by construction.

The scores are not calibrated for this task; tune `threshold` on the dev split.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from verbatim_core.extractors import SpanExtractor

logger = logging.getLogger(__name__)


class CrossEncoderSentenceExtractor(SpanExtractor):
    """Keep the HuSpaCy sentences of each chunk that a cross-encoder scores as relevant."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        max_length: int = 512,
        spacy_model: str = "hu_core_news_lg",
        nlp: Optional[Any] = None,
        spacy_batch_size: int = 64,
    ):
        """
        Args:
            model: CrossEncoder checkpoint (Hub id or local path) with a single output logit.
            revision: Hub commit sha to pin the checkpoint.
            device: "cuda" / "mps" / "cpu", or None to let sentence-transformers pick.
            threshold: Minimum sigmoid score for a sentence to be kept.
            batch_size: (question, sentence) pairs per forward pass.
            max_length: Token budget per pair; longer pairs are truncated.
            spacy_model: HuSpaCy model, loaded with parser and NER excluded.
            nlp: Preloaded spaCy pipeline (overrides `spacy_model`), e.g. the one
                HungarianBM25Provider already holds.
            spacy_batch_size: nlp.pipe batch size.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers") from exc

        if nlp is None:
            import spacy

            nlp = spacy.load(spacy_model, exclude=["parser", "ner"])
        if not any(name in ("senter", "parser") for name in nlp.pipe_names):
            raise ValueError(f"spaCy pipeline has no sentence boundaries: {nlp.pipe_names}")
        self.nlp = nlp
        self.spacy_batch_size = spacy_batch_size

        self.model_name = model
        self.revision = revision
        self.model = CrossEncoder(
            model,
            revision=revision,
            device=device,
            max_length=max_length,
            activation_fn=torch.nn.Sigmoid(),
        )
        if self.model.num_labels != 1:
            raise ValueError(
                f"{model} has {self.model.num_labels} output labels; a relevance "
                "cross-encoder with a single logit is required"
            )

        self.threshold = threshold
        self.batch_size = batch_size
        self.max_length = max_length

    # ---- sentences ----------------------------------------------------------

    @staticmethod
    def _sentence_offsets(text: str, doc) -> List[Tuple[int, int]]:
        """(start, end) of each sentence with surrounding whitespace trimmed."""
        offsets = []
        for sent in doc.sents:
            start, end = sent.start_char, sent.end_char
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                offsets.append((start, end))
        return offsets

    # ---- scoring ------------------------------------------------------------

    def score_sentences(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Every sentence of each chunk with its score in [0, 1], in document order."""
        texts = [getattr(r, "text", "") or "" for r in search_results]
        scores: Dict[str, List[Tuple[str, float]]] = {text: [] for text in texts}
        todo = [text for text in dict.fromkeys(texts) if text.strip()]

        sentences: List[Tuple[str, str]] = []
        for text, doc in zip(todo, self.nlp.pipe(todo, batch_size=self.spacy_batch_size)):
            sentences += [(text, text[s:e]) for s, e in self._sentence_offsets(text, doc)]
        if not sentences:
            return scores

        # A pair depends only on the sentence, so a sentence repeated across overlapping
        # chunks gets the same score; score each distinct sentence once.
        distinct = list(dict.fromkeys(sentence for _, sentence in sentences))
        predicted = self.model.predict(
            [(question, sentence) for sentence in distinct],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if len(predicted) != len(distinct):
            raise ValueError(
                f"cross-encoder returned {len(predicted)} scores for {len(distinct)} sentences"
            )

        p_by_sentence = {s: float(p) for s, p in zip(distinct, predicted)}
        bad = sum(not math.isfinite(p) for p in p_by_sentence.values())
        if bad:
            raise ValueError(f"cross-encoder returned {bad} non-finite scores")
        for text, sentence in sentences:
            scores[text].append((sentence, p_by_sentence[sentence]))
        return scores

    @staticmethod
    def spans_from_scores(
        scores: Dict[str, List[Tuple[str, float]]], threshold: float
    ) -> Dict[str, List[str]]:
        """Sentences >= threshold, most relevant first (see the module docstring)."""
        kept = {
            text: sorted(
                ((sentence, p) for sentence, p in scored if p >= threshold),
                key=lambda item: item[1],
                reverse=True,
            )
            for text, scored in scores.items()
        }
        order = sorted(
            kept, key=lambda text: kept[text][0][1] if kept[text] else -1.0, reverse=True
        )
        return {text: [sentence for sentence, _ in kept[text]] for text in order}

    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        return self.spans_from_scores(
            self.score_sentences(question, search_results), self.threshold
        )
