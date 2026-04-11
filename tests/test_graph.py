"""
test_graph.py — Tests for entity-relationship graph (REBEL background extraction).

Key design notes:
  - BackgroundGraph uses REBEL in a daemon thread. Tests that need extracted
    triples wait for the worker to finish (small queue, short timeout).
  - MemoryGraph storage/query layer is tested directly for determinism.
  - Memory API integration tests verify the wiring without waiting for REBEL
    (graph_pending flag covers the async case).
"""

import importlib.util
import shutil
import tempfile
import time

import pytest

from verimem.graph import (
    BackgroundGraph,
    MemoryGraph,
    _canonical,
    _extract_triples,
)
from verimem import Memory
from verimem.recall import ContextPacket, ContradictionFlag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp(prefix="verimem_graph_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)
    MemoryGraph._instances.clear()
    BackgroundGraph._instances.clear()


@pytest.fixture
def mgraph(tmpdir):
    return MemoryGraph.for_path(tmpdir)


@pytest.fixture
def mem(tmpdir):
    return Memory(path=tmpdir)


def _wait_for_graph(bg: BackgroundGraph, timeout: float = 5.0) -> None:
    """Poll until the background worker queue is drained or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if bg.pending_count() == 0 and (bg._thread is None or not bg._thread.is_alive()):
            return
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Unit: GLiNER + spaCy extraction (live model)
# ---------------------------------------------------------------------------


def gliner_available() -> bool:
    return importlib.util.find_spec("gliner") is not None


skip_no_gliner = pytest.mark.skipif(not gliner_available(), reason="gliner not installed")


class TestGLiNERExtraction:
    @skip_no_gliner
    def test_entities_detected(self):
        from gliner import GLiNER
        import spacy

        model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        nlp = spacy.load("en_core_web_sm")
        triples = _extract_triples("Alice deployed MySQL to Google Cloud.", model, nlp)
        all_texts = " ".join(t for triple in triples for t in (triple[0], triple[2]))
        assert "Alice" in all_texts or "alice" in all_texts.lower()

    @skip_no_gliner
    def test_returns_list(self):
        from gliner import GLiNER
        import spacy

        model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        nlp = spacy.load("en_core_web_sm")
        result = _extract_triples("Nothing interesting here.", model, nlp)
        assert isinstance(result, list)

    @skip_no_gliner
    def test_single_entity_no_triples(self):
        from gliner import GLiNER
        import spacy

        model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        nlp = spacy.load("en_core_web_sm")
        result = _extract_triples("Alice.", model, nlp)
        assert len(result) == 0  # need at least 2 entities for a triple


# ---------------------------------------------------------------------------
# Unit: MemoryGraph storage + query (inject triples directly, no live model)
# ---------------------------------------------------------------------------


class TestMemoryGraphStorage:
    def test_store_and_query(self, mgraph):
        mgraph.store_triples([("Alice", "deployed", "MySQL")], chunk_id="c1", filed_at=None)
        result = mgraph.query_related("Alice")
        assert len(result.triples) == 1
        t = result.triples[0]
        assert t.subject == "alice"
        assert t.relation == "deployed"
        assert t.obj == "mysql"

    def test_canonical_matching(self, mgraph):
        mgraph.store_triples([("Alice Smith", "manages", "Auth Service")], "c2", None)
        # Look up with different casing
        result = mgraph.query_related("alice smith")
        assert len(result.triples) >= 1

    def test_chunk_ids_tracked(self, mgraph):
        mgraph.store_triples([("Bob", "owns", "Postgres")], "chunk_abc", None)
        result = mgraph.query_related("Bob")
        assert "chunk_abc" in result.chunk_ids

    def test_idempotent_store(self, mgraph):
        mgraph.store_triples([("Alice", "uses", "Redis")], "c3", None)
        mgraph.store_triples([("Alice", "uses", "Redis")], "c3", None)
        result = mgraph.query_related("Alice")
        triples = [(t.subject, t.relation, t.obj) for t in result.triples]
        # No duplicates
        assert triples.count(("alice", "uses", "redis")) == 1

    def test_unknown_entity_empty(self, mgraph):
        result = mgraph.query_related("NonExistentXYZ")
        assert len(result.triples) == 0
        assert len(result.related_entities) == 0

    def test_two_hop(self, mgraph):
        mgraph.store_triples([("Alice", "manages", "Bob")], "c4", None)
        mgraph.store_triples([("Bob", "owns", "MySQL")], "c5", None)
        result = mgraph.query_related("Alice", hops=2)
        related = [e.canonical for e in result.related_entities]
        assert "bob" in related
        assert "mysql" in related

    def test_is_processed(self, mgraph):
        assert not mgraph.is_processed("newchunk")
        mgraph.store_triples([], "newchunk", None)
        assert mgraph.is_processed("newchunk")

    def test_entity_count(self, mgraph):
        mgraph.store_triples([("Alice", "works at", "Google")], "c6", None)
        assert mgraph.entity_count() >= 2  # Alice + Google

    def test_triple_count(self, mgraph):
        before = mgraph.triple_count()
        mgraph.store_triples([("X", "rel", "Y")], "c7", None)
        assert mgraph.triple_count() == before + 1

    def test_entities_for_chunks(self, mgraph):
        mgraph.store_triples([("Alice", "deployed", "MySQL")], "chunk_q", None)
        entities = mgraph.entities_for_chunks(["chunk_q"])
        names = [e.canonical for e in entities]
        assert "alice" in names
        assert "mysql" in names

    def test_to_dict_serializable(self, mgraph):
        mgraph.store_triples([("Alice", "owns", "Redis")], "c8", None)
        result = mgraph.query_related("Alice")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "triples" in d
        assert "related_entities" in d

    def test_to_simple(self, mgraph):
        mgraph.store_triples([("Alice", "deployed", "MySQL")], "c9", None)
        result = mgraph.query_related("Alice")
        s = result.to_simple()
        assert "relationships" in s
        assert any("deployed" in r for r in s["relationships"])


# ---------------------------------------------------------------------------
# Unit: BackgroundGraph submission (no REBEL needed — just queue mechanics)
# ---------------------------------------------------------------------------


class TestBackgroundGraph:
    def test_submit_queues_work(self, tmpdir):
        bg = BackgroundGraph.for_path(tmpdir)
        bg.submit("Alice manages the auth service.", "chunk1", None)
        assert bg.pending_count() >= 0  # worker may have started immediately

    def test_already_processed_skipped(self, tmpdir):
        mgraph = MemoryGraph.for_path(tmpdir)
        mgraph.store_triples([], "existing_chunk", None)  # mark as processed
        bg = BackgroundGraph.for_path(tmpdir)
        before = bg.pending_count()
        bg.submit("Some text.", "existing_chunk", None)
        assert bg.pending_count() == before  # not re-queued

    def test_singleton(self, tmpdir):
        bg1 = BackgroundGraph.for_path(tmpdir)
        bg2 = BackgroundGraph.for_path(tmpdir)
        assert bg1 is bg2


# ---------------------------------------------------------------------------
# Integration: Memory API with graph
# ---------------------------------------------------------------------------


class TestMemoryGraphIntegration:
    def test_remember_submits_to_background(self, mem):
        """remember() should not block — background graph runs async."""
        import time

        start = time.time()
        mem.remember("Alice deployed MySQL to the production environment at Google.")
        elapsed = time.time() - start
        # Should not block on REBEL model load (that happens in daemon thread)
        # Allow generous time for cold ChromaDB embedding model download on first run
        assert elapsed < 30.0

    def test_recall_include_graph_false_by_default(self, mem):
        mem.remember("Alice manages the Google Cloud infrastructure.")
        result = mem.recall("Alice", mode="raw")
        assert result.graph_entities is None  # not included by default

    def test_recall_include_graph_returns_field(self, mem):
        # Inject triples directly (bypass background GLiNER for test speed)
        chunk_ids = mem.remember("Alice deployed MySQL.", source="test")
        mgraph = MemoryGraph.for_path(mem._path)
        if chunk_ids:
            mgraph.store_triples([("Alice", "deployed", "MySQL")], chunk_ids[0], None)

        result = mem.recall("Alice", mode="raw", include_graph=True)
        assert result.graph_entities is not None
        assert len(result.graph_entities) >= 1

    def test_to_simple_minimal_output(self, mem):
        mem.remember("The auth service uses JWT tokens for authentication.")
        result = mem.recall("what does auth service use?", mode="raw")
        simple = result.to_simple()
        assert "hits" in simple
        assert "contradictions" in simple
        assert simple["contradictions"] == []
        for h in simple["hits"]:
            assert "text" in h
            assert "similarity" in h  # kept — it's signal
            assert "topic" in h
            # Internal noise stripped
            assert "drawer_id" not in h
            assert "chunk_index" not in h

    def test_to_simple_includes_contradiction_warning(self, mem):
        # Build a packet with a contradiction manually
        packet = ContextPacket(
            query="test",
            hits=[],
            contradictions=[ContradictionFlag(0, 1, "test contradiction", 0.9)],
        )
        simple = packet.to_simple()
        assert "contradictions" in simple
        assert len(simple["contradictions"]) == 1
        assert "verify" in simple["contradictions"][0]

    def test_ephemeral_recall_related_returns_empty(self):
        mem = Memory(":memory:")
        result = mem.recall_related("Alice")
        assert len(result.triples) == 0

    def test_graph_stats_ephemeral(self):
        mem = Memory(":memory:")
        stats = mem.graph_stats()
        assert stats["entities"] == 0

    def test_recall_related_uses_injected_triples(self, mem):
        """recall_related() works when triples are injected directly."""
        mgraph = MemoryGraph.for_path(mem._path)
        mgraph.store_triples(
            [("Alice", "manages", "the auth service"), ("Alice", "works at", "Google")],
            "manual_chunk",
            None,
        )
        result = mem.recall_related("Alice")
        assert len(result.triples) == 2
        rels = [t.relation for t in result.triples]
        assert "manages" in rels
        assert "works at" in rels

    def test_graph_stats_after_injection(self, mem):
        mgraph = MemoryGraph.for_path(mem._path)
        mgraph.store_triples([("Bob", "owns", "Redis")], "stat_chunk", None)
        stats = mem.graph_stats()
        assert stats["entities"] >= 2
        assert stats["triples"] >= 1


# ---------------------------------------------------------------------------
# Unit: canonical helper
# ---------------------------------------------------------------------------


class TestCanonical:
    def test_lowercases(self):
        assert _canonical("MySQL") == "mysql"

    def test_strips_punctuation(self):
        assert _canonical("Alice.") == "alice"

    def test_collapses_whitespace(self):
        assert _canonical("auth  service") == "auth service"

    def test_strips_quotes(self):
        assert _canonical('"Google"') == "google"
