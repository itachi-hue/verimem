"""
Tests for VeriMem structured recall: ContextPacket, policy presets, revision,
Memory.recall integration, utilities, BackgroundNLI.
"""

import os

from verimem import Memory
from verimem.recall import (
    CompletenessFlags,
    ContextPacket,
    ContradictionFlag,
    RecallHit,
    apply_freshness,
    compute_retrieval_uncertainty,
    detect_contradictions,
)
from verimem.policy import (
    RecallPolicy,
    get_default_policy,
    get_policy,
    register_policy,
)
from verimem.revision import bump_revision, get_revision
from verimem.background_nli import BackgroundNLI


class TestContextPacket:
    def test_empty_packet_serialises(self):
        p = ContextPacket(query="test")
        d = p.to_dict()
        assert d["query"] == "test"
        assert d["hits"] == []
        assert d["contradictions"] == []
        assert d["completeness"]["hits_truncated"] is False

    def test_recall_hit_to_dict(self):
        h = RecallHit(
            text="JWT tokens expire after 24 hours.",
            wing="project",
            room="backend",
            source_file="auth.py",
            similarity=0.92,
            chunk_index=0,
            filed_at="2026-01-01T00:00:00",
        )
        d = h.to_dict()
        assert d["similarity"] == 0.92
        assert d["wing"] == "project"
        assert d["filed_at"] == "2026-01-01T00:00:00"

    def test_completeness_any_truncated(self):
        c = CompletenessFlags(hits_truncated=True)
        assert c.any_truncated() is True
        c2 = CompletenessFlags()
        assert c2.any_truncated() is False

    def test_completeness_pending_flag_in_dict(self):
        c = CompletenessFlags(contradiction_check_pending=True)
        d = c.to_dict()
        assert d["contradiction_check_pending"] is True

    def test_completeness_pending_false_by_default(self):
        c = CompletenessFlags()
        assert c.contradiction_check_pending is False

    def test_packet_with_hits_serialises_fully(self):
        h = RecallHit(text="hello", wing="w", room="r", source_file="f.py", similarity=0.8)
        cf = ContradictionFlag(hit_a_idx=0, hit_b_idx=1, reason='"up" vs "down"', confidence=0.7)
        p = ContextPacket(
            query="status",
            hits=[h],
            contradictions=[cf],
            completeness=CompletenessFlags(hits_truncated=True),
            policy_version="tight",
            store_revision=42,
        )
        d = p.to_dict()
        assert len(d["hits"]) == 1
        assert len(d["contradictions"]) == 1
        assert d["store_revision"] == 42
        assert d["policy_version"] == "tight"
        assert d["completeness"]["hits_truncated"] is True


class TestRetrievalUncertainty:
    def test_empty_similarities_insufficient(self):
        u = compute_retrieval_uncertainty([])
        assert u.retrieval_insufficient is True
        assert u.confidence_q == 0.0
        assert u.best_match_score == 0.0

    def test_single_hit_no_ambiguity(self):
        u = compute_retrieval_uncertainty([0.9])
        assert u.ambiguity == 0.0
        assert abs(u.confidence_q - 0.9) < 1e-6

    def test_tied_hits_high_ambiguity(self):
        u = compute_retrieval_uncertainty([0.8, 0.8, 0.8], softmax_tau=0.12)
        assert u.ambiguity > 0.5
        assert u.confidence_q < 0.8


class TestPolicy:
    def test_builtin_presets_registered(self):
        for name in ("default", "tight", "wide", "fresh", "audit"):
            pol = get_policy(name)
            assert pol is not None, f"preset '{name}' not registered"
            assert pol.policy_version == name

    def test_default_policy(self):
        pol = get_default_policy()
        assert pol.policy_version == "default"
        assert pol.n_results >= 1

    def test_register_custom_policy(self):
        custom = RecallPolicy(policy_version="my-custom-v1", n_results=7, min_similarity=0.4)
        register_policy(custom)
        retrieved = get_policy("my-custom-v1")
        assert retrieved is not None
        assert retrieved.n_results == 7
        assert retrieved.min_similarity == 0.4

    def test_unknown_policy_returns_none(self):
        assert get_policy("totally-unknown-xyz") is None

    def test_tight_has_no_decay(self):
        pol = get_policy("tight")
        assert pol.tau_seconds == 0.0

    def test_audit_includes_all_metadata(self):
        pol = get_policy("audit")
        assert pol.include_all_metadata is True

    def test_wide_skips_contradictions(self):
        pol = get_policy("wide")
        assert pol.detect_contradictions is False


class TestRevision:
    def test_initial_revision_is_zero(self, tmp_dir):
        store = os.path.join(tmp_dir, "rev_store")
        os.makedirs(store)
        assert get_revision(store) == 0

    def test_bump_increments(self, tmp_dir):
        store = os.path.join(tmp_dir, "rev_store2")
        os.makedirs(store)
        r1 = bump_revision(store)
        r2 = bump_revision(store)
        r3 = bump_revision(store)
        assert r1 == 1
        assert r2 == 2
        assert r3 == 3

    def test_get_revision_does_not_increment(self, tmp_dir):
        store = os.path.join(tmp_dir, "rev_store3")
        os.makedirs(store)
        bump_revision(store)
        bump_revision(store)
        assert get_revision(store) == 2
        assert get_revision(store) == 2


class TestMemoryRecallIntegration:
    def test_recall_returns_context_packet(self, tmp_dir):
        path = os.path.join(tmp_dir, "mem1")
        mem = Memory(path=path)
        mem.remember("Authentication uses JWT with 24h expiry.", source="doc")
        packet = mem.recall("how does auth work?", mode="raw")
        assert isinstance(packet, ContextPacket)
        assert packet.query == "how does auth work?"
        assert len(packet.hits) > 0

    def test_recall_hits_have_provenance(self, tmp_dir):
        path = os.path.join(tmp_dir, "mem2")
        mem = Memory(path=path)
        mem.remember("PostgreSQL 15 with Alembic migrations.", source="db.py", topic="project")
        packet = mem.recall("database", mode="raw")
        for hit in packet.hits:
            assert hit.source_file != ""
            assert 0.0 <= hit.similarity <= 1.0

    def test_recall_stamps_policy_version(self, tmp_dir):
        path = os.path.join(tmp_dir, "mem3")
        mem = Memory(path=path)
        mem.remember("React frontend uses TanStack Query.")
        packet = mem.recall("frontend", mode="raw")
        assert packet.policy_version == "raw"
        packet2 = mem.recall("frontend", mode="rerank")
        assert packet2.policy_version == "rerank"

    def test_recall_topic_filter(self, tmp_dir):
        path = os.path.join(tmp_dir, "mem4")
        mem = Memory(path=path)
        mem.remember("Backend topic alpha", topic="backend")
        mem.remember("Frontend topic beta", topic="frontend")
        packet = mem.recall("topic", topic="backend", mode="raw")
        for h in packet.hits:
            assert h.wing == "backend"

    def test_store_revision_on_packet(self, tmp_dir):
        path = os.path.join(tmp_dir, "mem5")
        mem = Memory(path=path)
        mem.remember("Sprint planning note.")
        packet = mem.recall("sprint", mode="raw")
        assert isinstance(packet.store_revision, int)


class TestFreshness:
    def _make_hit(self, sim, filed_at):
        return RecallHit(
            text="test",
            wing="w",
            room="r",
            source_file="f.py",
            similarity=sim,
            filed_at=filed_at,
        )

    def test_no_decay_preserves_similarity_order(self):
        hits = [
            self._make_hit(0.9, "2026-01-01T00:00:00"),
            self._make_hit(0.7, "2026-03-01T00:00:00"),
        ]
        result = apply_freshness(hits, tau_seconds=0)
        assert result[0].similarity == 0.9

    def test_decay_sets_freshness_score(self):
        old = self._make_hit(0.95, "2024-01-01T00:00:00")
        result = apply_freshness([old], tau_seconds=86_400 * 30)
        assert result[0].freshness_score <= result[0].similarity

    def test_missing_filed_at_gets_neutral_score(self):
        h = self._make_hit(0.75, None)
        result = apply_freshness([h], tau_seconds=3600)
        assert result[0].freshness_score == 0.75

    def test_freshness_score_set_on_all_hits(self):
        hits = [
            self._make_hit(0.9, "2026-01-01T00:00:00"),
            self._make_hit(0.8, "2026-03-01T00:00:00"),
            self._make_hit(0.7, None),
        ]
        apply_freshness(hits, tau_seconds=86_400)
        for h in hits:
            assert h.freshness_score is not None


class TestContradictionDetection:
    def _hit(self, text, sim=0.85):
        return RecallHit(text=text, wing="w", room="r", source_file="f.py", similarity=sim)

    def test_detects_healthy_vs_down(self):
        hits = [
            self._hit("The payment service is healthy and responding normally."),
            self._hit("The payment service is down — all requests failing."),
        ]
        flags = detect_contradictions(hits)
        assert len(flags) > 0
        assert any("healthy" in f.reason or "down" in f.reason for f in flags)

    def test_no_false_positives_on_unrelated(self):
        hits = [
            self._hit("Authentication uses JWT tokens."),
            self._hit("The database is PostgreSQL 15."),
        ]
        flags = detect_contradictions(hits)
        assert len(flags) == 0

    def test_returns_correct_indices(self):
        hits = [
            self._hit("Deploy succeeded."),
            self._hit("Deploy failed — build error."),
        ]
        flags = detect_contradictions(hits)
        assert len(flags) > 0
        f = flags[0]
        assert f.hit_a_idx in (0, 1)
        assert f.hit_b_idx in (0, 1)
        assert f.hit_a_idx != f.hit_b_idx

    def test_low_similarity_hits_excluded(self):
        hits = [
            self._hit("Service is healthy.", sim=0.3),
            self._hit("Service is down.", sim=0.3),
        ]
        flags = detect_contradictions(hits, min_similarity=0.6)
        assert len(flags) == 0

    def test_contradiction_flag_has_confidence(self):
        hits = [
            self._hit("The system is online."),
            self._hit("The system is offline."),
        ]
        flags = detect_contradictions(hits)
        if flags:
            assert 0.0 <= flags[0].confidence <= 1.0


class TestBackgroundNLI:
    def test_lookup_empty_on_cold_cache(self, tmp_dir):
        store = os.path.join(tmp_dir, "nli_store1")
        os.makedirs(store)
        nli = BackgroundNLI(store)
        hits = [
            RecallHit(
                text="Service is healthy.",
                wing="w",
                room="r",
                source_file="a.py",
                similarity=0.9,
                drawer_id="id-a",
            ),
            RecallHit(
                text="Service is down.",
                wing="w",
                room="r",
                source_file="b.py",
                similarity=0.88,
                drawer_id="id-b",
            ),
        ]
        assert nli.lookup(hits) == []

    def test_submit_returns_true_for_uncached_pairs(self, tmp_dir):
        store = os.path.join(tmp_dir, "nli_store2")
        os.makedirs(store)
        nli = BackgroundNLI(store)
        hits = [
            RecallHit(
                text="Approved.",
                wing="w",
                room="r",
                source_file="a.py",
                similarity=0.9,
                drawer_id="x1",
            ),
            RecallHit(
                text="Rejected.",
                wing="w",
                room="r",
                source_file="b.py",
                similarity=0.85,
                drawer_id="x2",
            ),
        ]
        pending = nli.submit_hits(hits)
        assert pending is True

    def test_submit_false_when_no_eligible_hits(self, tmp_dir):
        store = os.path.join(tmp_dir, "nli_store3")
        os.makedirs(store)
        nli = BackgroundNLI(store)
        hits = [
            RecallHit(
                text="a", wing="w", room="r", source_file="f.py", similarity=0.1, drawer_id="y1"
            ),
            RecallHit(
                text="b", wing="w", room="r", source_file="f.py", similarity=0.1, drawer_id="y2"
            ),
        ]
        assert nli.submit_hits(hits) is False

    def test_pending_flag_bool_on_memory_recall(self, tmp_dir):
        path = os.path.join(tmp_dir, "nli_mem")
        mem = Memory(path=path)
        mem.remember("Service is healthy.")
        mem.remember("Service is down.")
        packet = mem.recall("service status", mode="raw")
        assert isinstance(packet.completeness.contradiction_check_pending, bool)

    def test_for_store_returns_singleton(self, tmp_dir):
        store = os.path.join(tmp_dir, "nli_singleton")
        os.makedirs(store)
        a = BackgroundNLI.for_store(store)
        b = BackgroundNLI.for_store(store)
        assert a is b
