"""
Dense (embedding distance) + BM25 fusion, with optional thin phrase/entity multipliers.

Used by Memory.recall() (hybrid / hybrid_rerank modes) and by benchmark scripts.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

# BM25 parameters (Okapi)
_K1 = 1.5
_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _extract_quoted_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    for pat in (r"'([^']{3,80})'", r'"([^"]{3,80})"'):
        phrases.extend(re.findall(pat, text))
    return [p.strip() for p in phrases if len(p.strip()) >= 3]


_NOT_ENTITY = frozenset(
    """
    What When Where Who How Which Did Do Was Were Have Has Had Is Are The My Our Their
    Can Could Would Should Will Shall May Might Monday Tuesday Wednesday Thursday Friday
    Saturday Sunday January February March April June July August September October
    November December In On At For To Of With By From And But I It Its This That These
    Those Previously Recently Also Just Very More Said Speaker Person Time Date Year Day
    """.split()
)


def extract_capitalized_entities(text: str) -> List[str]:
    words = re.findall(r"\b[A-Z][a-z]{2,15}\b", text)
    return list({w for w in words if w not in _NOT_ENTITY})


class BM25:
    """Okapi BM25 over a fixed corpus (tokenized)."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        self._corpus = [list(doc) for doc in corpus_tokens]
        self._N = len(self._corpus)
        if self._N == 0:
            self._avgdl = 0.0
            self._doc_freqs: list[dict[str, int]] = []
            self._idf: dict[str, float] = {}
            return

        df: dict[str, int] = {}
        self._doc_freqs = []
        for doc in self._corpus:
            freqs: dict[str, int] = {}
            for t in doc:
                freqs[t] = freqs.get(t, 0) + 1
            self._doc_freqs.append(freqs)
            for t in freqs:
                df[t] = df.get(t, 0) + 1

        self._avgdl = sum(len(d) for d in self._corpus) / self._N
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log(1.0 + (self._N - freq + 0.5) / (freq + 0.5))

    @property
    def num_docs(self) -> int:
        return self._N

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        if self._N == 0:
            return []
        q_terms = [t for t in query_tokens if t in self._idf]
        if not q_terms:
            return [0.0] * self._N

        out: List[float] = []
        for i, freqs in enumerate(self._doc_freqs):
            dl = len(self._corpus[i])
            denom_norm = _K1 * (1.0 - _B + _B * dl / self._avgdl) if self._avgdl > 0 else _K1
            s = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf[term]
                s += idf * (tf * (_K1 + 1.0)) / (tf + denom_norm)
            out.append(s)
        return out


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [1.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def thin_boost_multiplier(
    query: str,
    doc: str,
    *,
    phrase_boost: bool = False,
    entity_boost: bool = False,
) -> float:
    """Small multiplicative bump when optional signals fire (default off)."""
    m = 1.0
    doc_l = doc.lower()
    if phrase_boost:
        phrases = _extract_quoted_phrases(query)
        if phrases and any(p.lower() in doc_l for p in phrases):
            m *= 1.08
    if entity_boost:
        ents = extract_capitalized_entities(query)
        if ents and any(e.lower() in doc_l for e in ents):
            m *= 1.06
    return m


def fused_rank_from_dense_results(
    candidate_indices: List[int],
    distances: List[float],
    corpus: List[str],
    query: str,
    *,
    lexical_weight: float = 0.35,
    phrase_boost: bool = False,
    entity_boost: bool = False,
    corpus_tokens: Optional[List[List[str]]] = None,
    bm25_index: Optional[BM25] = None,
) -> List[int]:
    """
    Re-rank dense top candidates by fusing cosine distance with BM25.

    candidate_indices: corpus row indices from embedding search (best-first).
    distances: parallel Chroma / store distances (lower = closer).
    corpus: full document strings (BM25 statistics over entire corpus).
    corpus_tokens: if provided (and ``bm25_index`` is None), pre-tokenized rows parallel to ``corpus``.
    bm25_index: reusable BM25 built over the same corpus (skips index rebuild each query).
    lexical_weight: weight on normalized BM25 in [0,1]; dense gets (1 - lexical_weight).
    """
    if not candidate_indices:
        return []
    if len(candidate_indices) != len(distances):
        raise ValueError("candidate_indices and distances length mismatch")

    w = min(1.0, max(0.0, lexical_weight))
    if bm25_index is not None:
        if bm25_index.num_docs != len(corpus):
            raise ValueError("bm25_index corpus size must match corpus")
        bm25 = bm25_index
    elif corpus_tokens is not None:
        if len(corpus_tokens) != len(corpus):
            raise ValueError("corpus_tokens length must match corpus")
        bm25 = BM25(corpus_tokens)
    else:
        bm25 = BM25([tokenize(d) for d in corpus])
    q_toks = tokenize(query)
    bm25_all = bm25.scores(q_toks)

    dense_raw = [1.0 / (1.0 + float(d)) for d in distances]
    lex_raw = [bm25_all[i] for i in candidate_indices]

    d_n = _minmax(dense_raw)
    l_n = _minmax(lex_raw)

    scored: List[tuple[float, int, int]] = []
    for j, idx in enumerate(candidate_indices):
        base = (1.0 - w) * d_n[j] + w * l_n[j]
        doc_text = corpus[idx] if 0 <= idx < len(corpus) else ""
        boost = thin_boost_multiplier(
            query,
            doc_text,
            phrase_boost=phrase_boost,
            entity_boost=entity_boost,
        )
        combined = base * boost
        scored.append((combined, -j, idx))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored]


def full_ranking_after_fusion(
    candidate_indices: List[int],
    distances: List[float],
    corpus: List[str],
    query: str,
    *,
    lexical_weight: float = 0.35,
    phrase_boost: bool = False,
    entity_boost: bool = False,
    corpus_tokens: Optional[List[List[str]]] = None,
    bm25_index: Optional[BM25] = None,
) -> List[int]:
    """Fused order for candidates, then append remaining corpus indices."""
    fused_order = fused_rank_from_dense_results(
        candidate_indices,
        distances,
        corpus,
        query,
        lexical_weight=lexical_weight,
        phrase_boost=phrase_boost,
        entity_boost=entity_boost,
        corpus_tokens=corpus_tokens,
        bm25_index=bm25_index,
    )
    seen = set(fused_order)
    rest = [i for i in range(len(corpus)) if i not in seen]
    return fused_order + rest
