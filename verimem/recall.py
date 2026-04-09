"""
recall.py — Structured output types for VeriMem search.

Defines ContextPacket, RecallHit, and CompletenessFlags so callers
(MCP, Python API, tests) all work with the same inspectable object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class RecallHit:
    """One verbatim drawer returned by a search."""

    text: str
    wing: str
    room: str
    source_file: str
    similarity: float
    chunk_index: int = 0
    filed_at: Optional[str] = None  # ISO timestamp stored at mine time
    ingest_age_seconds: Optional[float] = None  # seconds since filed_at, if known
    freshness_score: Optional[float] = None  # similarity × decay(age), if decay applied
    drawer_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "wing": self.wing,
            "room": self.room,
            "source_file": self.source_file,
            "similarity": self.similarity,
            "chunk_index": self.chunk_index,
            "filed_at": self.filed_at,
            "ingest_age_seconds": self.ingest_age_seconds,
            "freshness_score": self.freshness_score,
            "drawer_id": self.drawer_id,
        }


@dataclass
class ContradictionFlag:
    """Two hits that appear semantically contradictory."""

    hit_a_idx: int  # index into ContextPacket.hits
    hit_b_idx: int
    reason: str  # short human-readable reason
    confidence: float  # 0-1

    def to_dict(self) -> dict:
        return {
            "hit_a_idx": self.hit_a_idx,
            "hit_b_idx": self.hit_b_idx,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class CompletenessFlags:
    """Honest accounting of any caps that were hit during recall."""

    hits_truncated: bool = False  # got exactly n_results, may be more
    empty_after_filter: bool = False  # wing/room filter returned nothing
    filter_fallback: bool = False  # filter was dropped and search ran unfiltered
    contradiction_check_skipped: bool = False  # too few hits to bother (<2)
    contradiction_check_pending: bool = False  # NLI running in background; check next call

    def any_truncated(self) -> bool:
        return self.hits_truncated or self.filter_fallback

    def to_dict(self) -> dict:
        return {
            "hits_truncated": self.hits_truncated,
            "empty_after_filter": self.empty_after_filter,
            "filter_fallback": self.filter_fallback,
            "contradiction_check_skipped": self.contradiction_check_skipped,
            "contradiction_check_pending": self.contradiction_check_pending,
        }


@dataclass
class ContextPacket:
    """
    The full structured result of a recall search.

    Default output via to_simple(): clean, minimal — just text, age, topic.
    Full output via to_dict(): everything including provenance, revision, flags.

    Returned by Memory.recall().
    """

    query: str
    hits: List[RecallHit] = field(default_factory=list)
    contradictions: List[ContradictionFlag] = field(default_factory=list)
    completeness: CompletenessFlags = field(default_factory=CompletenessFlags)
    policy_version: str = "default"
    store_revision: Optional[int] = None
    served_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    filters_applied: dict = field(default_factory=dict)
    graph_entities: Optional[List[dict]] = None  # populated when include_graph=True

    def to_simple(self) -> dict:
        """
        Clean, high-signal output for agents.

        Keeps everything useful, removes internal noise:
          - text, topic, similarity, human-readable age
          - contradictions as readable pairs ("hit 0 vs hit 2: NLI score 0.94")
          - note if contradiction check still running in background
          - graph entities if include_graph=True was passed to recall()

        Strips internal IDs (drawer_id, chunk_index, room, revision,
        policy version, raw seconds) — none of that is useful to an LLM.
        """

        def _human_age(seconds: Optional[float]) -> Optional[str]:
            if seconds is None:
                return None
            m = int(seconds / 60)
            if m < 1:
                return "just now"
            if m < 60:
                return f"{m} minute{'s' if m != 1 else ''} ago"
            h = int(m / 60)
            if h < 24:
                return f"{h} hour{'s' if h != 1 else ''} ago"
            d = int(h / 24)
            if d < 30:
                return f"{d} day{'s' if d != 1 else ''} ago"
            w = int(d / 7)
            return f"{w} week{'s' if w != 1 else ''} ago"

        simple_hits = []
        for h in self.hits:
            entry: dict = {
                "text": h.text,
                "topic": h.wing or "general",
                "similarity": h.similarity,
            }
            age_str = _human_age(h.ingest_age_seconds)
            if age_str:
                entry["age"] = age_str
            simple_hits.append(entry)

        result: dict = {"hits": simple_hits}

        if self.contradictions:
            readable = []
            for c in self.contradictions:
                readable.append(
                    f"hit {c.hit_a_idx} vs hit {c.hit_b_idx}: {c.reason} "
                    f"(confidence {c.confidence:.2f}) — verify before acting"
                )
            result["contradictions"] = readable

        if self.completeness.contradiction_check_pending:
            result["note"] = (
                "contradiction check still running — re-query next turn for full signal"
            )

        if self.graph_entities:
            result["entities"] = self.graph_entities

        return result

    def to_dict(self) -> dict:
        """Full output — everything including provenance, IDs, flags."""
        return {
            "query": self.query,
            "served_at": self.served_at,
            "policy_version": self.policy_version,
            "store_revision": self.store_revision,
            "filters_applied": self.filters_applied,
            "completeness": self.completeness.to_dict(),
            "hits": [h.to_dict() for h in self.hits],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "graph_entities": self.graph_entities,
        }


# ---------------------------------------------------------------------------
# Freshness helpers
# ---------------------------------------------------------------------------


def _parse_ts(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def apply_freshness(hits: List[RecallHit], tau_seconds: float) -> List[RecallHit]:
    """
    Re-rank hits by freshness_score = similarity × e^(−age / tau).

    tau_seconds is the half-life in seconds (tau = half_life / ln(2)).
    If a hit has no filed_at, its freshness_score equals its similarity
    (no penalty / no bonus — neutral).

    Returns hits sorted by freshness_score descending.
    """
    if tau_seconds <= 0:
        # No decay — just return as-is
        for h in hits:
            h.freshness_score = h.similarity
        return hits

    now = datetime.now(timezone.utc)
    for h in hits:
        ts = _parse_ts(h.filed_at)
        if ts is not None:
            age_s = max(0.0, (now - ts).total_seconds())
            h.ingest_age_seconds = age_s
            decay = math.exp(-age_s / tau_seconds)
            h.freshness_score = round(h.similarity * decay, 4)
        else:
            h.freshness_score = h.similarity

    hits.sort(key=lambda h: h.freshness_score, reverse=True)
    return hits


# ---------------------------------------------------------------------------
# Contradiction detection (retrieve-set only — cheap)
# ---------------------------------------------------------------------------

# Pairs of antonym-ish tokens that are strong contradiction signals.
_ANTONYM_PAIRS = [
    ("healthy", "down"),
    ("healthy", "failing"),
    ("healthy", "outage"),
    ("up", "down"),
    ("success", "fail"),
    ("succeeded", "failed"),
    ("approved", "rejected"),
    ("approved", "denied"),
    ("complete", "incomplete"),
    ("complete", "pending"),
    ("active", "inactive"),
    ("active", "disabled"),
    ("resolved", "unresolved"),
    ("online", "offline"),
    ("running", "stopped"),
    ("pass", "fail"),
    ("open", "closed"),
    ("enabled", "disabled"),
    ("true", "false"),
    ("yes", "no"),
]


def _tokens(text: str) -> set:
    import re

    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def detect_contradictions(
    hits: List[RecallHit], min_similarity: float = 0.6
) -> List[ContradictionFlag]:
    """
    Lightweight contradiction detection on the retrieved set.

    Checks every pair of high-similarity hits for antonym tokens.
    No model required — pure heuristic, fast, zero extra latency.

    For richer NLI-based detection, wire in a cross-encoder model
    (see project21/ashnode/nli.py as a reference).
    """
    flags: List[ContradictionFlag] = []

    high_sim = [h for h in hits if h.similarity >= min_similarity]
    for i in range(len(high_sim)):
        for j in range(i + 1, len(high_sim)):
            ha, hb = high_sim[i], high_sim[j]
            ta, tb = _tokens(ha.text), _tokens(hb.text)
            for word_a, word_b in _ANTONYM_PAIRS:
                if word_a in ta and word_b in tb:
                    flags.append(
                        ContradictionFlag(
                            hit_a_idx=hits.index(ha),
                            hit_b_idx=hits.index(hb),
                            reason=f'"{word_a}" vs "{word_b}"',
                            confidence=0.7,
                        )
                    )
                    break
                if word_b in ta and word_a in tb:
                    flags.append(
                        ContradictionFlag(
                            hit_a_idx=hits.index(ha),
                            hit_b_idx=hits.index(hb),
                            reason=f'"{word_b}" vs "{word_a}"',
                            confidence=0.7,
                        )
                    )
                    break

    return flags
