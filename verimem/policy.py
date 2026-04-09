"""
policy.py — Named recall policy presets (metadata for tooling / future hooks).

Fields describe how aggressive retrieval should be:
  - n_results / k : how many hits to return
  - tau_seconds   : freshness decay half-life (0 = no decay)
  - min_similarity: drop hits below this score
  - detect_contradictions: whether to run retrieve-set contradiction check
  - include_all_metadata: include full raw metadata dict in each hit

`Memory.recall()` uses ``mode=`` (``raw`` / ``hybrid`` / ``rerank`` / ``hybrid_rerank``); presets
here are optional metadata for tooling. ``ContextPacket.policy_version`` echoes the recall ``mode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RecallPolicy:
    """
    Versioned configuration snapshot for recall behaviour.

    Attributes
    ----------
    policy_version : str
        Stable identifier.  Stamped on every ContextPacket so you know
        exactly which rules produced it.
    n_results : int
        Max hits to return from Chroma (k).
    tau_seconds : float
        Freshness decay half-life in seconds (0 = no decay).
        Common values:
          3 600        = 1 hour
          86 400       = 1 day
          604 800      = 1 week
          2_592_000    = 30 days  (default)
    min_similarity : float
        Drop hits with similarity < this value.  0.0 keeps everything.
    detect_contradictions : bool
        Run retrieve-set antonym/heuristic contradiction check.
    include_all_metadata : bool
        Attach the raw Chroma metadata dict to each RecallHit.
    """

    policy_version: str = "default"
    n_results: int = 5
    tau_seconds: float = 0.0  # 0 = no decay by default
    min_similarity: float = 0.0
    detect_contradictions: bool = True
    include_all_metadata: bool = False
    rerank: bool = False  # local cross-encoder rerank (sentence-transformers; core dep)
    rerank_pool: int = 20  # fetch this many from Chroma, rerank, return n_results


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, RecallPolicy] = {}


def register_policy(policy: RecallPolicy) -> None:
    """Register a policy by its version string."""
    _REGISTRY[policy.policy_version] = policy


def get_policy(version: str) -> Optional[RecallPolicy]:
    """Return a registered policy, or None if not found."""
    return _REGISTRY.get(version)


def get_default_policy() -> RecallPolicy:
    return _REGISTRY.get("default", RecallPolicy())


# ---------------------------------------------------------------------------
# Built-in presets — registered at import time
# ---------------------------------------------------------------------------

_PRESETS = [
    RecallPolicy(
        policy_version="default",
        n_results=5,
        tau_seconds=0.0,
        min_similarity=0.0,
        detect_contradictions=True,
    ),
    RecallPolicy(
        policy_version="tight",
        n_results=3,
        tau_seconds=0.0,
        min_similarity=0.5,
        detect_contradictions=True,
    ),
    RecallPolicy(
        policy_version="wide",
        n_results=15,
        tau_seconds=0.0,
        min_similarity=0.0,
        detect_contradictions=False,
    ),
    RecallPolicy(
        policy_version="fresh",
        n_results=5,
        tau_seconds=0.0,
        min_similarity=0.3,
        detect_contradictions=True,
    ),
    RecallPolicy(
        policy_version="audit",
        n_results=20,
        tau_seconds=0.0,
        min_similarity=0.0,
        detect_contradictions=False,
        include_all_metadata=True,
    ),
    RecallPolicy(
        policy_version="rerank",
        n_results=5,
        tau_seconds=0.0,
        min_similarity=0.0,
        detect_contradictions=True,
        rerank=True,
        rerank_pool=20,
    ),
]

for _p in _PRESETS:
    register_policy(_p)
