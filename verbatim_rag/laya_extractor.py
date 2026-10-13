"""
Sentence-level span extractor with laya-multilingual as the relevance judge.

Each chunk is split into sentences with HuSpaCy, and every sentence is judged by
convaiinnovations/laya-multilingual (mmBERT-base, non-generative) with a
two-option choice question: A = helps answer the question, B = does not. Sentences
with P(A) >= threshold are kept in document order.

A/B choice is used instead of laya's `noul` type: the model card reports that
`noul` under-reports "true" and recommends a two-option choice with neutral keys.

The user's question goes into laya's `instructions` (laya asks questions *about*
a state) and the state is the sentence alone. On the dev split with hard
distractors (run_extraction --distractors 2, 200 queries) this beat showing ±1
neighbouring sentences (chunk AUC 0.660 vs 0.598 with English instructions), and
Hungarian instructions beat English ones (0.693 vs 0.660 without context).

Every emitted span is a slice of the chunk text at HuSpaCy sentence offsets, so it
is an exact substring of its source chunk by construction.

laya ships uncalibrated (temperatures 1.0, over-confident) and is weak zero-shot;
tune `threshold` on the dev split, or pass a fitted `calibration` file.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from verbatim_core.extractors import SpanExtractor

logger = logging.getLogger(__name__)

QUESTION_ID = "relevant"

# "{question}" is replaced by the user's question at query time.
DEFAULT_INSTRUCTIONS = (
    "Kérdés: {question}\n"
    "A mondat tartalmaz olyan információt, amely legalább részben megválaszolja a kérdést?"
)
DEFAULT_CRITERIA = {
    "A": "Igen, a mondat segít megválaszolni a kérdést.",
    "B": "Nem, a mondat nem segít megválaszolni a kérdést.",
}


class LayaSentenceExtractor(SpanExtractor):
    """Keep the HuSpaCy sentences of each chunk that laya judges relevant."""

    def __init__(
        self,
        model: str = "convaiinnovations/laya-multilingual",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        calibration: Optional[str] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        max_len: Optional[int] = None,
        lang: Optional[str] = "hu",
        instructions: str = DEFAULT_INSTRUCTIONS,
        criteria: Optional[Dict[str, str]] = None,
        spacy_model: str = "hu_core_news_lg",
        nlp: Optional[Any] = None,
        spacy_batch_size: int = 64,
    ):
        """
        Args:
            model: laya checkpoint (Hub id or local path).
            revision: Hub commit sha to pin the checkpoint.
            device: "cuda" / "mps" / "cpu", or None to let laya pick.
            calibration: Temperature JSON written by laya's calibration tools.
            threshold: Minimum P(A) for a sentence to be kept.
            batch_size: States per laya forward pass.
            max_len: Token budget per state (laya default 1024).
            lang: Language hint; only selects per-language temperatures.
            instructions: Instruction template containing "{question}".
            criteria: Option descriptions keyed exactly "A" (relevant) and "B".
            spacy_model: HuSpaCy model, loaded with parser and NER excluded.
            nlp: Preloaded spaCy pipeline (overrides `spacy_model`), e.g. the one
                HungarianBM25Provider already holds.
            spacy_batch_size: nlp.pipe batch size.
        """
        criteria = dict(DEFAULT_CRITERIA if criteria is None else criteria)
        if set(criteria) != {"A", "B"}:
            raise ValueError(
                f"criteria must have exactly the keys 'A' and 'B', got {sorted(criteria)}"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if "{question}" not in instructions:
            raise ValueError("instructions must contain the placeholder {question}")

        if nlp is None:
            import spacy

            nlp = spacy.load(spacy_model, exclude=["parser", "ner"])
        if not any(name in ("senter", "parser") for name in nlp.pipe_names):
            raise ValueError(f"spaCy pipeline has no sentence boundaries: {nlp.pipe_names}")
        self.nlp = nlp
        self.spacy_batch_size = spacy_batch_size

        try:
            import laya
        except ImportError as exc:
            raise ImportError("pip install laya") from exc
        self.model_name = model
        self.revision = revision
        self.agent = laya.load(model, device=device, revision=revision, calibration=calibration)

        self.threshold = threshold
        self.batch_size = batch_size
        self.max_len = max_len
        self.lang = lang
        self.instructions = instructions
        self.criteria = criteria

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

    def questions(self, question: str) -> Dict[str, Dict[str, Any]]:
        """The laya question set for one user question."""
        instructions = self.instructions.replace("{question}", question)
        return {
            QUESTION_ID: {"type": "choice", "instructions": instructions, "criteria": self.criteria}
        }

    # ---- scoring ------------------------------------------------------------

    def score_sentences(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Every sentence of each chunk with its P(A), in document order."""
        texts = [getattr(r, "text", "") or "" for r in search_results]
        scores: Dict[str, List[Tuple[str, float]]] = {text: [] for text in texts}
        todo = [text for text in dict.fromkeys(texts) if text.strip()]

        sentences: List[Tuple[str, str]] = []
        for text, doc in zip(todo, self.nlp.pipe(todo, batch_size=self.spacy_batch_size)):
            sentences += [(text, text[s:e]) for s, e in self._sentence_offsets(text, doc)]
        if not sentences:
            return scores

        # The state is the sentence alone, so a sentence repeated across overlapping chunks
        # gets the same score; judge each distinct sentence once.
        states = list(dict.fromkeys(sentence for _, sentence in sentences))
        results = self.agent.predict_batch(
            states,
            self.questions(question),
            batch_size=self.batch_size,
            lang=self.lang,
            max_len=self.max_len,
        )
        if len(results) != len(states):
            raise ValueError(f"laya returned {len(results)} results for {len(states)} sentences")

        p_by_state = {}
        truncated = 0
        for state, result in zip(states, results):
            p_by_state[state] = float(result["answers"][QUESTION_ID]["probabilities"]["A"])
            truncated += bool(result.get("usage", {}).get("truncated"))
        for text, sentence in sentences:
            scores[text].append((sentence, p_by_state[sentence]))
        if truncated:
            logger.warning(
                "laya truncated %d of %d sentences; raise max_len", truncated, len(states)
            )
        return scores

    @staticmethod
    def spans_from_scores(
        scores: Dict[str, List[Tuple[str, float]]], threshold: float
    ) -> Dict[str, List[str]]:
        return {
            text: [sentence for sentence, p in scored if p >= threshold]
            for text, scored in scores.items()
        }

    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        return self.spans_from_scores(
            self.score_sentences(question, search_results), self.threshold
        )
