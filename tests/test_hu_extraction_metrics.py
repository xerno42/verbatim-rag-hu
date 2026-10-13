"""Metric helpers of the Hungarian extraction eval runner (network-free)."""

import json
import logging
from types import SimpleNamespace

import pytest

from evaluation.hu_legal.run_extraction import (
    LogCounter,
    breakdown,
    check_index_meta,
    chunk_auc_pairs,
    covered_chars,
    distractor_pools,
    oracle_results,
    parse_thresholds,
    predict,
    sample_distractors,
    score_prediction,
    setting_names,
    summarize,
    verbatim_failures,
    word_f1,
)
from evaluation.hu_legal.run_retrieval import (
    INDEX_META,
    index_meta,
    load_split,
    write_index_meta,
)


def _r(chunk_id, text):
    return SimpleNamespace(id=chunk_id, text=text)


class TestWordF1:
    def test_exact_match(self):
        assert word_f1(["A szerződés semmis."], ["a szerződés  semmis"])["f1"] == 1.0

    def test_partial(self):
        f = word_f1(["a b c d"], ["c d e"])
        assert f["precision"] == pytest.approx(0.5)
        assert f["recall"] == pytest.approx(2 / 3)
        assert f["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))

    def test_empty_prediction(self):
        assert word_f1([], ["valami"]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_both_empty(self):
        assert word_f1([], [])["f1"] == 1.0

    def test_counts_repeated_words_as_multiset(self):
        assert word_f1(["a a"], ["a"])["precision"] == pytest.approx(0.5)


class TestVerbatim:
    def test_counts_non_substrings_and_empty_spans(self):
        spans = {"Egy. Kettő.": ["Egy.", "Három.", ""], "x": []}
        assert verbatim_failures(spans) == 2

    def test_all_verbatim(self):
        assert verbatim_failures({"Egy. Kettő.": ["Kettő."]}) == 0

    def test_span_keyed_by_unknown_chunk_fails(self):
        spans = {"Egy.": ["Egy."], "Kettő.": ["Kettő."]}
        assert verbatim_failures(spans, ["Egy."]) == 1


class TestCoveredChars:
    def test_overlapping_spans_count_once(self):
        assert covered_chars("Egy kettő három.", ["Egy kettő", "kettő három", "kettő"]) == 15

    def test_missing_and_empty_spans_ignored(self):
        assert covered_chars("Egy.", ["", "Kettő.", "Egy."]) == 4


class TestScorePrediction:
    def test_gold_chunk_spans_and_dedup(self):
        results = [_r("g", "Egy. Kettő."), _r("x", "Egy. Három.")]
        query = {"answerable": True, "gold_chunk_ids": ["g"]}
        pred = {"Egy. Kettő.": ["Egy.", "Kettő."], "Egy. Három.": ["Egy."]}

        out = score_prediction(pred, results, query)

        assert out["n_spans"] == 3
        assert out["gold_chunk_spans"] == 2
        assert out["gold_chars"] == len("Egy. Kettő.") - 1  # union of spans in the gold chunk
        assert out["gold_answered"] is True
        assert out["distractors_answered"] == 1
        assert "f1" not in out  # no gold spans annotated

    def test_f1_only_with_gold_spans(self):
        query = {"answerable": True, "gold_chunk_ids": ["g"], "gold_spans": [{"text": "Kettő."}]}
        out = score_prediction({"Egy. Kettő.": ["Kettő."]}, [_r("g", "Egy. Kettő.")], query)
        assert out["f1"]["f1"] == 1.0


def _row(answerable, n_spans, gold_in_context=None, gold_chunk_spans=0, **fields):
    return {
        "answerable": answerable,
        "gold_in_context": gold_in_context,
        "passage_chars": 100,
        "settings": {
            "s": {
                "n_spans": n_spans,
                "gold_chunk_spans": gold_chunk_spans,
                "verbatim_failures": 0,
                "gold_chars": 25 * n_spans,
                "gold_answered": n_spans > 0,
                "distractors_answered": 0,
            }
        },
        **fields,
    }


class TestSummarize:
    def test_oracle_with_distractors(self):
        row = _row(True, 3, gold_chunk_spans=2, n_distractors=2, auc=(1.5, 2))
        row["settings"]["s"]["distractors_answered"] = 1
        s = summarize([row], "s", "oracle")
        assert s["gold_answer_rate"] == 1.0
        assert s["distractor_answer_rate"] == 0.5
        assert s["span_gold_chunk_precision"] == pytest.approx(2 / 3)
        assert s["chunk_auc"] == 0.75
        assert s["mean_gold_spans"] == 2

    def test_oracle(self):
        rows = [_row(True, 2), _row(True, 0)]
        s = summarize(rows, "s", "oracle")
        assert s["answer_rate"] == 0.5
        assert s["mean_spans"] == 1.0
        assert s["coverage"] == pytest.approx(0.25)
        assert s["verbatim_pass_rate"] == 1.0
        assert s["n_gold"] == 0 and s["f1"] is None
        assert "abstention_accuracy" not in s

    def test_retrieved_abstention(self):
        rows = [
            _row(True, 2, gold_in_context=True, gold_chunk_spans=1),
            _row(True, 0, gold_in_context=False),
            _row(False, 0),
            _row(False, 1),
        ]
        s = summarize(rows, "s", "retrieved")
        assert s["abstention_accuracy"] == 0.5
        assert s["false_abstention_rate"] == 0.5
        assert s["false_answer_rate"] == 0.5
        assert s["gold_in_context"] == 0.5
        assert s["answer_rate_gold_in_context"] == 1.0
        assert s["span_gold_chunk_precision"] == 0.5

    def test_no_spans_has_no_pass_rate(self):
        assert summarize([_row(True, 0)], "s", "oracle")["verbatim_pass_rate"] is None

    def test_empty(self):
        assert summarize([], "s", "oracle") == {"n": 0}

    def test_breakdown(self):
        rows = [_row(True, 1, query_type="a"), _row(True, 0, query_type="b")]
        b = breakdown(rows, "s", "oracle", "query_type")
        assert b["a"]["answer_rate"] == 1.0
        assert b["b"]["answer_rate"] == 0.0


class TestSettings:
    def test_threshold_settings(self):
        assert setting_names("crossencoder", [0.3, 0.5]) == ["crossencoder@0.3", "crossencoder@0.5"]
        assert setting_names("llm", [0.5]) == ["llm"]
        assert setting_names("overlap", [0.5]) == ["overlap@0.5"]

    def test_parse_thresholds_sorted_and_validated(self):
        assert parse_thresholds("0.7, 0.3,0.7") == [0.3, 0.7]
        with pytest.raises(Exception):
            parse_thresholds("1.5")

    def test_predict_sweeps_thresholds_from_one_scoring(self):
        class FakeJudge:
            calls = 0

            def score_sentences(self, question, results):
                FakeJudge.calls += 1
                return {"t": [("a", 0.4), ("b", 0.8)]}

            @staticmethod
            def spans_from_scores(scores, threshold):
                return {k: [s for s, p in v if p >= threshold] for k, v in scores.items()}

        args = SimpleNamespace(extractor="crossencoder", thresholds=[0.3, 0.5, 0.9])
        preds, scores = predict(FakeJudge(), args, "q", [_r("c", "t")])

        assert FakeJudge.calls == 1
        assert preds == {
            "crossencoder@0.3": {"t": ["a", "b"]},
            "crossencoder@0.5": {"t": ["b"]},
            "crossencoder@0.9": {"t": []},
        }
        assert scores == {"t": [("a", 0.4), ("b", 0.8)]}

    def test_oracle_results_use_the_query_passage(self):
        [r] = oracle_results({"qid": "q", "source_chunk_id": "c1", "passage": "Szöveg."})
        assert (r.id, r.text) == ("c1", "Szöveg.")


CORPUS = [{"chunk_id": "a", "text": "Egy."}, {"chunk_id": "b", "text": "Kettő."}]


class TestIndexMeta:
    def _args(self, **kw):
        base = dict(dense_model="m", dense_revision=None, bm25_model="hu", k1=1.2, b=0.75)
        return SimpleNamespace(**{**base, **kw})

    def _index_dir(self, tmp_path, meta):
        write_index_meta(tmp_path, meta)
        (tmp_path / "milvus.db").touch()
        return tmp_path

    def test_matching_meta_passes(self, tmp_path):
        meta = index_meta(CORPUS, self._args(), dense=True, sparse=True)
        check_index_meta(self._index_dir(tmp_path, meta), meta)

    def test_missing_meta(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            check_index_meta(tmp_path, {})

    def test_missing_db(self, tmp_path):
        meta = index_meta(CORPUS, self._args(), dense=True, sparse=True)
        write_index_meta(tmp_path, meta)
        with pytest.raises(ValueError, match="milvus.db missing"):
            check_index_meta(tmp_path, meta)

    def test_mismatch_names_the_field(self, tmp_path):
        run_dir = self._index_dir(tmp_path, index_meta(CORPUS, self._args(), True, True))
        with pytest.raises(ValueError) as e:
            check_index_meta(run_dir, index_meta(CORPUS, self._args(k1=1.5), True, True))
        assert "bm25" in str(e.value) and "corpus_chunks" not in str(e.value)
        assert json.loads((tmp_path / INDEX_META).read_text())["bm25"]["k1"] == 1.2

    def test_same_size_corpus_with_other_content_mismatches(self, tmp_path):
        other = [{"chunk_id": "a", "text": "Egy."}, {"chunk_id": "b", "text": "Három."}]
        run_dir = self._index_dir(tmp_path, index_meta(CORPUS, self._args(), True, True))
        with pytest.raises(ValueError, match="corpus_sha256"):
            check_index_meta(run_dir, index_meta(other, self._args(), True, True))

    def test_fingerprint_ignores_corpus_order(self):
        args = self._args()
        assert index_meta(CORPUS, args, True, True) == index_meta(CORPUS[::-1], args, True, True)


class TestLoadSplit:
    def test_drops_excluded_queries(self, tmp_path):
        rows = [
            {"qid": "1", "split": "test"},
            {"qid": "2", "split": "test", "excluded": True},
            {"qid": "3", "split": "dev"},
        ]
        path = tmp_path / "queries.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        queries, n_excluded = load_split(path, "test")

        assert [q["qid"] for q in queries] == ["1"]
        assert n_excluded == 1


class TestLogCounter:
    def test_counts_errors_and_dropped_spans(self):
        counter = LogCounter()
        log = logging.getLogger("verbatim_core.extractors")
        log.addHandler(counter)
        try:
            log.error("Individual extraction failed for document: %s", "401")
            log.warning("Batch extraction failed for chunk starting at %d, ...", 0)
            log.warning("Span not found verbatim in document: '%s...'", "x")
            log.info("not counted")
        finally:
            log.removeHandler(counter)
        assert (counter.errors, counter.dropped) == (1, 1)  # batch failure is retried

    def test_summary_reports_errors_and_drops(self):
        row = _row(True, 0)
        row["settings"]["s"].update(extractor_errors=1, dropped_spans=2)
        s = summarize([row], "s", "oracle")
        assert (s["extractor_errors"], s["dropped_spans"]) == (1, 2)
        assert "dropped_spans" not in summarize([_row(True, 0)], "s", "oracle")


class TestDistractors:
    CORPUS = [
        {"chunk_id": "gold", "doc_id": "d", "text": "a b c d e f g h i j"},
        {"chunk_id": "overlap", "doc_id": "d", "text": "x a b c d e f g h y"},
        {"chunk_id": "same1", "doc_id": "d", "text": "k l m n o p q r s t"},
        {"chunk_id": "same2", "doc_id": "d", "text": "u v w x y z k l m n"},
        {"chunk_id": "other", "doc_id": "e", "text": "1 2 3 4 5 6 7 8 9 0"},
    ]
    QUERY = {
        "qid": "q1",
        "doc_id": "d",
        "source_chunk_id": "gold",
        "gold_chunk_ids": ["gold"],
        "passage": "a b c d e f g h i j",
    }

    def test_same_document_chunks_without_shared_shingles(self):
        pools = distractor_pools(self.CORPUS)
        picked = sample_distractors(self.QUERY, pools, 2, seed=0)
        assert sorted(c["chunk_id"] for c in picked) == ["same1", "same2"]

    def test_other_documents_fill_in_and_sampling_is_seeded(self):
        pools = distractor_pools(self.CORPUS)
        picked = sample_distractors(self.QUERY, pools, 5, seed=0)
        assert sorted(c["chunk_id"] for c in picked) == ["other", "same1", "same2"]
        assert picked == sample_distractors(self.QUERY, pools, 5, seed=0)

    def test_oracle_results_include_distractors(self):
        results = oracle_results(self.QUERY, self.CORPUS[2:4], seed=0)
        assert sorted(r.id for r in results) == ["gold", "same1", "same2"]


class TestChunkAuc:
    def test_best_sentence_per_chunk(self):
        results = [_r("g", "G"), _r("x", "X"), _r("y", "Y")]
        scores = {"G": [("a", 0.2), ("b", 0.8)], "X": [("c", 0.9)], "Y": [("d", 0.8)]}
        assert chunk_auc_pairs(scores, results, {"g"}) == (0.5, 2)

    def test_negatives_sharing_passage_text_are_left_out(self):
        passage = "a b c d e f g h i j"
        results = [_r("g", passage), _r("x", "z a b c d e f g h y"), _r("y", "k l m n o p q r s t")]
        scores = {
            passage: [("s", 0.9)],
            "z a b c d e f g h y": [("s", 0.9)],
            results[2].text: [("t", 0.1)],
        }
        assert chunk_auc_pairs(scores, results, {"g"}, passage) == (1, 1)
        assert chunk_auc_pairs(scores, results, {"g"}) == (1.5, 2)

    def test_no_negatives(self):
        assert chunk_auc_pairs({"G": [("a", 1.0)]}, [_r("g", "G")], {"g"}) == (0, 0)


class TestOverlapScorer:
    def test_scores_share_of_question_words(self):
        pytest.importorskip("verbatim_rag.cross_encoder_extractor")
        from evaluation.hu_legal.run_extraction import OverlapScorer

        class NLP:
            def pipe(self, texts, batch_size=64):
                for text in texts:
                    yield SimpleNamespace(
                        sents=[
                            SimpleNamespace(start_char=0, end_char=text.index(".") + 1),
                            SimpleNamespace(start_char=text.index(".") + 1, end_char=len(text)),
                        ]
                    )

        text = "A felmondási idő harminc nap. Más téma."
        scores = OverlapScorer(NLP()).score_sentences("Mennyi a felmondási idő?", [_r("c", text)])
        assert scores == {
            text: [("A felmondási idő harminc nap.", 0.5), ("Más téma.", 0.0)]
        }  # "mennyi" missing
        assert OverlapScorer.spans_from_scores(scores, 0.5) == {
            text: ["A felmondási idő harminc nap."]
        }
