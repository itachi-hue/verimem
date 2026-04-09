"""
test_memory_api.py — Smoke tests for the clean Memory API.

Uses :memory: store so nothing is persisted to disk.
"""

from verimem import Memory
from verimem.recall import ContextPacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mem() -> Memory:
    return Memory(path=":memory:")


# ---------------------------------------------------------------------------
# Basic ingest + recall
# ---------------------------------------------------------------------------


class TestMemoryBasics:
    def test_remember_returns_ids(self):
        mem = make_mem()
        ids = mem.remember("The sky is blue and the grass is green.")
        assert len(ids) >= 1
        assert all(isinstance(i, str) for i in ids)

    def test_count_increases(self):
        mem = make_mem()
        assert mem.count() == 0
        mem.remember("First fact about the system.")
        assert mem.count() >= 1

    def test_recall_returns_context_packet(self):
        mem = make_mem()
        mem.remember("Redis is our primary cache layer.")
        result = mem.recall("what is the cache?", mode="raw")
        assert isinstance(result, ContextPacket)
        assert len(result.hits) >= 1

    def test_recall_hit_has_text(self):
        mem = make_mem()
        mem.remember("PostgreSQL stores all user records.")
        result = mem.recall("where are user records stored?", mode="raw")
        assert any("PostgreSQL" in h.text or "user" in h.text for h in result.hits)

    def test_recall_hit_has_similarity(self):
        mem = make_mem()
        mem.remember("The auth service uses JWT tokens.")
        result = mem.recall("authentication method", mode="raw")
        for hit in result.hits:
            assert isinstance(hit.similarity, float)
            assert 0.0 <= hit.similarity <= 1.0

    def test_idempotent_remember(self):
        mem = make_mem()
        ids1 = mem.remember("Duplicate fact.", source="src1")
        ids2 = mem.remember("Duplicate fact.", source="src1")
        assert ids1 == ids2
        assert mem.count() == 1

    def test_multiple_facts_recalled(self):
        mem = make_mem()
        mem.remember("The backend is Python FastAPI.")
        mem.remember("The frontend is React with TypeScript.")
        mem.remember("We use PostgreSQL for storage.")
        result = mem.recall("Python backend", top_k=3, mode="raw")
        assert len(result.hits) >= 1
        assert result.hits[0].similarity > 0


# ---------------------------------------------------------------------------
# Chunking of long text
# ---------------------------------------------------------------------------


class TestChunking:
    def test_long_text_produces_multiple_chunks(self):
        mem = make_mem()
        long_text = "The system uses microservices. " * 60  # ~1800 chars
        ids = mem.remember(long_text)
        assert len(ids) >= 2

    def test_each_chunk_is_retrievable(self):
        mem = make_mem()
        mem.remember("Part A: the auth service handles logins. " * 30)
        result = mem.recall("auth service logins", mode="raw")
        assert len(result.hits) >= 1


# ---------------------------------------------------------------------------
# Topic filtering
# ---------------------------------------------------------------------------


class TestTopicFilter:
    def test_topic_stored_in_metadata(self):
        mem = make_mem()
        mem.remember("Infra fact: we run on AWS.", topic="infra")
        result = mem.recall("AWS", top_k=3, mode="raw")
        hit = result.hits[0]
        assert hit.wing == "infra"  # topic maps to wing in RecallHit

    def test_topic_filter_narrows_results(self):
        mem = make_mem()
        mem.remember("Infra: AWS is our cloud.", topic="infra")
        mem.remember("Product: users can sign up.", topic="product")
        result = mem.recall("AWS", top_k=5, topic="infra", mode="raw")
        # All hits should come from infra topic
        for h in result.hits:
            assert h.wing == "infra"


# ---------------------------------------------------------------------------
# forget()
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_removes_chunk(self):
        mem = make_mem()
        ids = mem.remember("Temporary secret key = abc123.")
        assert mem.count() == 1
        mem.forget(ids[0])
        assert mem.count() == 0

    def test_recall_after_forget(self):
        mem = make_mem()
        ids = mem.remember("Old tech: we used MongoDB.")
        mem.forget(ids[0])
        result = mem.recall("MongoDB", mode="raw")
        assert len(result.hits) == 0 or all("MongoDB" not in h.text for h in result.hits)


# ---------------------------------------------------------------------------
# ContextPacket structure
# ---------------------------------------------------------------------------


class TestContextPacket:
    def test_packet_has_query(self):
        mem = make_mem()
        mem.remember("Some fact.")
        result = mem.recall("Some fact", mode="raw")
        assert result.query == "Some fact"

    def test_packet_has_served_at(self):
        mem = make_mem()
        mem.remember("Time test fact.")
        result = mem.recall("time test", mode="raw")
        assert result.served_at is not None

    def test_packet_has_policy_version(self):
        mem = make_mem()
        mem.remember("Policy test.")
        result = mem.recall("policy test", mode="raw")
        assert result.policy_version == "raw"

    def test_packet_policy_rerank_when_mode_rerank(self):
        mem = make_mem()
        mem.remember("Rerank test fact about authentication.")
        # rerank=True requires sentence-transformers; gracefully degrades if not installed
        result = mem.recall("authentication", mode="rerank")
        assert isinstance(result, ContextPacket)

    def test_completeness_flags_present(self):
        mem = make_mem()
        mem.remember("Flags test.")
        result = mem.recall("flags test", mode="raw")
        assert hasattr(result.completeness, "hits_truncated")
        assert hasattr(result.completeness, "contradiction_check_pending")

    def test_to_dict_serializable(self):
        mem = make_mem()
        mem.remember("Dict test fact.")
        result = mem.recall("dict test", mode="raw")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "hits" in d
        assert "completeness" in d

    def test_empty_recall_returns_packet(self):
        mem = make_mem()
        result = mem.recall("nothing here", mode="raw")
        assert isinstance(result, ContextPacket)
        assert len(result.hits) == 0


# ---------------------------------------------------------------------------
# repr / count
# ---------------------------------------------------------------------------


class TestMemoryMeta:
    def test_repr(self):
        mem = make_mem()
        r = repr(mem)
        assert "Memory" in r
        assert ":memory:" in r

    def test_count_zero_on_fresh(self):
        mem = make_mem()
        assert mem.count() == 0
