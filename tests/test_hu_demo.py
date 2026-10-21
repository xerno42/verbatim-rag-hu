"""Hungarian demo server pieces (evaluation/hu_legal/demo.py), network-free."""

import argparse
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluation.hu_legal.demo import (
    ABSTAIN_HU,
    CPU_BATCH_SIZE,
    GPU_BATCH_SIZE,
    TEMPLATE_HU,
    CheckedExtractor,
    HungarianStaticTemplate,
    RerankedRetriever,
    UncleanedResponseBuilder,
    hungarian_template_manager,
    index_dir,
    is_loopback,
    parse_extractors,
    prepare_index_dir,
    query_view,
    resolve_device_defaults,
)
from evaluation.hu_legal.run_retrieval import INDEX_META, index_meta, write_index_meta

# Double space on purpose: 2503 of 8820 corpus chunks have one, and the answer must keep it.
CHUNK_A = "Első mondat. A munkáltató  köteles bért fizetni."
SPAN_A = "A munkáltató  köteles bért fizetni."
CHUNK_B = "Más szöveg a szabadságról."


def hit(chunk_id, text, document_id="doc1"):
    return SimpleNamespace(id=chunk_id, text=text, score=1.0, metadata={"document_id": document_id})


class FakeExtractor:
    def __init__(self, spans=None, log_error=None):
        self.spans = spans or {}
        self.log_error = log_error

    def extract_spans(self, question, search_results):
        if self.log_error:
            logging.getLogger("verbatim_core.extractors").error(self.log_error)
        return self.spans


class TestHungarianTemplate:
    def test_no_spans_gives_fixed_abstention(self):
        assert HungarianStaticTemplate(TEMPLATE_HU).fill(TEMPLATE_HU, [], []) == ABSTAIN_HU

    def test_blank_spans_give_fixed_abstention(self):
        spans = [{"text": "  ", "doc_text": CHUNK_A}]
        assert HungarianStaticTemplate(TEMPLATE_HU).fill(TEMPLATE_HU, spans, []) == ABSTAIN_HU

    def test_spans_fill_hungarian_template(self):
        spans = [{"text": SPAN_A, "doc_text": CHUNK_A}]
        out = HungarianStaticTemplate(TEMPLATE_HU).fill(TEMPLATE_HU, spans, [])
        assert out.startswith("## Válasz")
        assert f"[1] {SPAN_A}" in out
        assert ABSTAIN_HU not in out

    def test_manager_uses_it(self):
        manager = hungarian_template_manager()
        assert manager.current_mode == "static"
        assert manager.process("Kérdés?", [], []) == ABSTAIN_HU


def test_uncleaned_response_builder_keeps_spans_exact():
    answer = f'"[1] {SPAN_A}"'
    assert UncleanedResponseBuilder().clean_answer(answer) == answer
    assert UncleanedResponseBuilder().clean_answer(None) == ""


class TestRerankedRetriever:
    def make(self, hits, reranker=None):
        index = MagicMock()
        index.query.return_value = hits
        retriever = RerankedRetriever(
            index, reranker, retrieve_k=100, hybrid_weights={"dense": 0.5, "sparse": 0.5}, rrf_k=60
        )
        return retriever, index

    def test_retrieves_retrieve_k_with_fixed_fusion_and_cuts_after_rerank(self):
        hits = [hit(f"c{i}", f"text {i}") for i in range(4)]
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, h: list(reversed(h))
        retriever, index = self.make(hits, reranker)

        out = retriever.query("kérdés", k=2, hybrid_weights={"dense": 1.0}, rrf_k=1)

        index.query.assert_called_once_with(
            "kérdés",
            k=100,
            filter=None,
            hybrid_weights={"dense": 0.5, "sparse": 0.5},
            rrf_k=60,
            search_params=None,
        )
        reranker.rerank.assert_called_once_with("kérdés", hits)
        assert [h.id for h in out] == ["c3", "c2"]

    def test_metadata_for_display_without_mutating_hits(self):
        hits = [hit("c0", "text", document_id="22016A0316(01)_ref")]
        retriever, _ = self.make(hits)
        (out,) = retriever.query("q", k=5)
        assert out.metadata == {
            "document_id": "22016A0316(01)_ref",
            "chunk_id": "c0",
            "title": "22016A0316(01)_ref",
        }
        assert out.text == "text"
        assert hits[0].metadata == {"document_id": "22016A0316(01)_ref"}

    def test_reranker_errors_propagate(self):
        reranker = MagicMock()
        reranker.rerank.side_effect = RuntimeError("CUDA out of memory")
        retriever, _ = self.make([hit("c0", "t")], reranker)
        with pytest.raises(RuntimeError, match="out of memory"):
            retriever.query("q", k=1)

    def test_vector_store_passthrough(self):
        retriever, index = self.make([])
        assert retriever.vector_store is index.vector_store


class TestCheckedExtractor:
    results = [hit("a", CHUNK_A), hit("b", CHUNK_B)]

    def test_keeps_verbatim_spans(self):
        spans = {CHUNK_A: [SPAN_A], CHUNK_B: []}
        out = CheckedExtractor(FakeExtractor(spans)).extract_spans("q", self.results)
        assert out == spans

    def test_drops_non_verbatim_blank_and_unknown_chunks(self, caplog):
        spans = {
            CHUNK_A: [SPAN_A, "A munkáltató köteles bért fizetni.", "", " "],
            "nem átadott chunk": ["nem átadott chunk"],
        }
        with caplog.at_level(logging.ERROR, logger="hu_legal.demo"):
            out = CheckedExtractor(FakeExtractor(spans)).extract_spans("q", self.results)
        assert out == {CHUNK_A: [SPAN_A]}
        assert "verbatim check dropped 3 span(s)" in caplog.text
        assert "chunk not passed in" in caplog.text

    def test_lost_llm_call_raises_instead_of_abstaining(self):
        inner = FakeExtractor({CHUNK_A: []}, log_error="Individual fallback extraction failed: 401")
        core_logger = logging.getLogger("verbatim_core")
        handlers = list(core_logger.handlers)
        with pytest.raises(RuntimeError, match="1 extraction call"):
            CheckedExtractor(inner).extract_spans("q", self.results)
        assert core_logger.handlers == handlers

    def test_other_log_lines_do_not_raise(self):
        inner = FakeExtractor({CHUNK_A: []}, log_error="Span not found in document")
        assert CheckedExtractor(inner).extract_spans("q", self.results) == {CHUNK_A: []}

    def test_failures_logged_by_other_threads_do_not_count(self):
        """A worker left over from a cancelled request must not fail the current one."""

        class OtherThreadFails:
            def extract_spans(self, question, search_results):
                def lost_call():
                    logging.getLogger("verbatim_core.extractors").error(
                        "Individual fallback extraction failed: timeout"
                    )

                worker = threading.Thread(target=lost_call)
                worker.start()
                worker.join()
                return {CHUNK_A: [SPAN_A]}

        out = CheckedExtractor(OtherThreadFails()).extract_spans("q", self.results)
        assert out == {CHUNK_A: [SPAN_A]}


class TestIndexDir:
    def test_plain_tag(self, tmp_path):
        assert index_dir(tmp_path, "baseline") == tmp_path / "index" / "baseline"

    @pytest.mark.parametrize("tag", ["", ".", "..", "../data", "a/b", "/abs"])
    def test_rejects_tags_outside_index_dir(self, tmp_path, tag):
        with pytest.raises(ValueError, match="plain directory name"):
            index_dir(tmp_path, tag)


class TestPrepareIndexDir:
    corpus = [{"chunk_id": "c1", "text": "szöveg"}]

    def args(self, tmp_path, build=False):
        return SimpleNamespace(
            data_dir=tmp_path,
            index_tag="demo",
            build=build,
            dense_model="Qwen/Qwen3-Embedding-0.6B",
            dense_revision=None,
            bm25_model="hu_core_news_lg",
            k1=1.2,
            b=0.75,
        )

    def completed(self, tmp_path, args, corpus):
        run_dir = tmp_path / "index" / "demo"
        run_dir.mkdir(parents=True)
        (run_dir / "milvus.db").write_text("db")
        write_index_meta(run_dir, index_meta(corpus, args, dense=True, sparse=True))
        return run_dir

    def test_reuses_matching_completed_index(self, tmp_path):
        args = self.args(tmp_path, build=True)
        run_dir = self.completed(tmp_path, args, self.corpus)
        assert prepare_index_dir(args, self.corpus) == (run_dir, True)
        assert (run_dir / "milvus.db").exists()  # --build never touches a completed index

    def test_mismatched_index_exits(self, tmp_path):
        args = self.args(tmp_path)
        self.completed(tmp_path, args, [{"chunk_id": "c1", "text": "más"}])
        with pytest.raises(SystemExit, match="does not match"):
            prepare_index_dir(args, self.corpus)

    def test_incomplete_index_without_build_exits(self, tmp_path):
        (tmp_path / "index" / "demo").mkdir(parents=True)
        with pytest.raises(SystemExit, match="--build"):
            prepare_index_dir(self.args(tmp_path), self.corpus)

    def test_build_starts_from_an_empty_dir(self, tmp_path):
        run_dir = tmp_path / "index" / "demo"
        run_dir.mkdir(parents=True)
        (run_dir / "bm25_hu_stats.json").write_text("{}")  # stale stats of an aborted build
        sibling = tmp_path / "index" / "baseline"
        sibling.mkdir()
        assert prepare_index_dir(self.args(tmp_path, build=True), self.corpus) == (run_dir, False)
        assert list(run_dir.iterdir()) == []
        assert not (run_dir / INDEX_META).exists()
        assert sibling.exists()


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", True), ("::1", True), ("localhost", True), ("0.0.0.0", False)],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_query_view_has_no_passage_text():
    query = {
        "qid": "q1",
        "query": "Mi?",
        "answerable": True,
        "gold_chunk_ids": ["c1"],
        "passage": "licensed text",
        "source_chunk_id": "c1",
    }
    view = query_view(query)
    assert "passage" not in view and "source_chunk_id" not in view
    assert view["qid"] == "q1" and view["gold_chunk_ids"] == ["c1"]
    assert view["legal_domain"] is None


class TestParseExtractors:
    def test_dedupes_in_order(self):
        assert parse_extractors("llm, crossencoder,llm") == ["llm", "crossencoder"]

    @pytest.mark.parametrize("value", ["", "overlap", "crossencoder,laya"])
    def test_rejects_unknown_or_empty(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_extractors(value)


class TestApp:
    """Endpoints the frontend uses, with the server's own pipelines over fake retrieval."""

    @pytest.fixture
    def client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        # verbatim_rag pulls in pymilvus etc., which the core-only CI install lacks.
        pytest.importorskip("verbatim_rag.core")
        SearchResult = pytest.importorskip("verbatim_rag.vector_stores.base").SearchResult
        from fastapi.testclient import TestClient

        from evaluation.hu_legal.demo import create_app, make_pipeline

        hits = [
            SearchResult(id="a", score=1.0, metadata={"document_id": "d1"}, text=CHUNK_A),
            SearchResult(id="b", score=0.5, metadata={"document_id": "d2"}, text=CHUNK_B),
        ]
        index = MagicMock()
        index.query.return_value = hits
        retriever = RerankedRetriever(
            index, None, retrieve_k=100, hybrid_weights={"dense": 0.5, "sparse": 0.5}, rrf_k=60
        )
        pipelines = {
            "crossencoder": make_pipeline(
                retriever, FakeExtractor({CHUNK_A: [SPAN_A]}), 2, MagicMock()
            ),
            "llm": make_pipeline(retriever, FakeExtractor({}), 2, MagicMock()),
        }
        queries = [{"qid": "q1", "query": "Mi?", "answerable": True, "passage": "x"}]
        app = create_app(pipelines, "crossencoder", queries, {"split": "test"})
        return TestClient(app)

    @staticmethod
    def stages(response):
        assert response.status_code == 200
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def test_status_config_and_queries(self, client):
        assert client.get("/api/status").json()["resources_loaded"] is True
        config = client.get("/api/demo/config").json()
        assert config["extractors"] == ["crossencoder", "llm"]
        assert config["default_extractor"] == "crossencoder"
        payload = client.get("/api/demo/queries").json()
        assert payload["split"] == "test"
        assert payload["queries"][0]["qid"] == "q1"
        assert "passage" not in payload["queries"][0]

    def test_stream_answers_verbatim_with_default_extractor(self, client):
        stages = self.stages(
            client.post("/api/query/stream", json={"question": "Mit kell fizetni?", "num_docs": 9})
        )
        assert [s["type"] for s in stages] == ["documents", "progress", "highlights", "answer"]
        docs = stages[0]["data"]
        assert [d["metadata"]["chunk_id"] for d in docs] == ["a", "b"]
        assert stages[2]["data"][0]["highlights"][0]["text"] == SPAN_A
        assert f"[1] {SPAN_A}" in stages[-1]["data"]["answer"]  # double space kept

    def test_stream_abstains_in_hungarian(self, client):
        stages = self.stages(
            client.post("/api/query/stream", json={"question": "Mi?", "extractor": "llm"})
        )
        assert stages[-1]["data"]["answer"] == ABSTAIN_HU

    def test_rejects_unknown_extractor_and_empty_question(self, client):
        assert (
            client.post("/api/query/stream", json={"question": "Mi?", "extractor": "x"}).status_code
            == 400
        )
        assert client.post("/api/query/stream", json={"question": "  "}).status_code == 400


class TestResolveDeviceDefaults:
    def args(self, **kw):
        return SimpleNamespace(
            **{"device": "cpu", "ce_device": None, "batch_size": None, "ce_batch_size": None, **kw}
        )

    def test_cpu_gets_small_batches(self):
        a = self.args()
        resolve_device_defaults(a)
        assert (a.device, a.batch_size, a.ce_batch_size) == ("cpu", CPU_BATCH_SIZE, CPU_BATCH_SIZE)

    def test_cuda_gets_gpu_batches(self):
        a = self.args(device="cuda:1")
        resolve_device_defaults(a)
        assert (a.batch_size, a.ce_batch_size) == (GPU_BATCH_SIZE, GPU_BATCH_SIZE)

    def test_cross_encoder_batch_follows_its_own_device(self):
        a = self.args(ce_device="cuda")
        resolve_device_defaults(a)
        assert (a.batch_size, a.ce_batch_size) == (CPU_BATCH_SIZE, GPU_BATCH_SIZE)

    def test_explicit_values_are_kept(self):
        a = self.args(batch_size=3, ce_batch_size=5)
        resolve_device_defaults(a)
        assert (a.batch_size, a.ce_batch_size) == (3, 5)

    @pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
    def test_auto_follows_cuda_availability(self, monkeypatch, available, expected):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        a = self.args(device="auto")
        resolve_device_defaults(a)
        assert a.device == expected
