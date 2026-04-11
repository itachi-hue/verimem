"""
memory.py — Clean agent-centric API for VeriMem.

Two functions. No wings. No rooms. No YAML config.

    from verimem import Memory

    mem = Memory()                          # persists to ~/.verimem by default
    mem.remember("Postgres is the DB")      # ingest anything
    result = mem.recall("what database?")   # → ContextPacket (default mode: rerank)

Design decisions:
  - Text is chunked at 800 chars with 100-char overlap (same as miner.py).
  - Chunks are stored verbatim with a UTC timestamp and an optional topic tag.
  - recall(..., mode=...) selects retrieval: rerank (default) = dense + local cross-encoder;
    raw = dense only; hybrid / hybrid_rerank add BM25 fusion (optional). Same four modes
    as benchmarks/longmemeval_bench.py.
  - recall() pipeline (depends on mode):
      rerank (default): dense search → cross-encoder → decay
      raw: dense search → decay (no BM25, no cross-encoder)
      hybrid: dense → BM25 fusion → decay (no CE unless hybrid_rerank)
      hybrid_rerank: hybrid fusion → cross-encoder → decay
    Step 3 only reshuffles near-equal hits. A much more relevant older chunk still
    beats a less relevant new one. Set decay_days=0 to skip decay entirely.
  - Background NLI surfaces contradictions between hits, cached in SQLite.
  - Returns a ContextPacket — same structured object used by MCP and benchmarks.

For agents, the full API is:

    mem.remember(text, source=None, topic=None)             → list[str]  (chunk IDs)
    mem.recall(query, top_k=5, decay_days=30)  → ContextPacket  # default: rerank
    mem.forget(chunk_id)                                    → None
    mem.count()                                             → int
    mem.revision()                                          → int  (monotonic write counter)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

from .recall import (
    CompletenessFlags,
    ContextPacket,
    ContradictionFlag,
    RecallHit,
    compute_retrieval_uncertainty,
)
from .revision import bump_revision, get_revision
from .background_nli import BackgroundNLI, score_contradictions_sync
from .reranker import CrossEncoderReranker
from .graph import MemoryGraph, BackgroundGraph, GraphRecallResult
from .fast_store import FastStore, is_available as _usearch_available
from .hybrid_retrieval import BM25, full_ranking_after_fusion, tokenize

logger = logging.getLogger("verimem_mcp")

_COLLECTION_NAME = "verimem_memories"
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100
_MIN_CHUNK_SIZE = 30
_DEFAULT_PATH = str(Path.home() / ".verimem")
_DEFAULT_TOPIC = "general"
_MIN_SIMILARITY_FOR_NLI = 0.45
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Recall: dense-only, hybrid (dense+BM25), or those plus local cross-encoder — same four modes as longmemeval_bench.
RETRIEVAL_MODES = ("raw", "hybrid", "rerank", "hybrid_rerank")
DEFAULT_RETRIEVAL_MODE = "rerank"


def _resolve_recall_mode(mode: str, rerank: Optional[bool]) -> str:
    if rerank is not None:
        warnings.warn(
            "recall(rerank=...) is deprecated; use mode='raw', 'hybrid', 'rerank', or 'hybrid_rerank'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "raw" if not rerank else "rerank"
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
    return mode


def _scan_corpus_sorted(col, where: Optional[dict]) -> Tuple[List[str], List[str], List[dict]]:
    """Stable (id-sorted) corpus for hybrid BM25; respects topic filter when ``where`` is set."""
    if hasattr(col, "scan_all"):
        data = col.scan_all(where)
    else:
        kw: dict = {"include": ["documents", "metadatas"]}
        if where:
            kw["where"] = where
        data = col.get(**kw)
    ids = list(data.get("ids") or [])
    docs = list(data.get("documents") or [])
    metas = list(data.get("metadatas") or [])
    triples = sorted(zip(ids, docs, metas), key=lambda x: x[0])
    if not triples:
        return [], [], []
    i, d, m = zip(*triples)
    return list(i), list(d), list(m)


# ---------------------------------------------------------------------------
# Embedding layer — SentenceTransformer + in-process cache (Ashnode-style)
# ---------------------------------------------------------------------------
# Mirrors ashnode/embeddings.py exactly:
#   - SentenceTransformer (PyTorch) instead of ChromaDB's ONNX runtime
#   - Module-level model singleton → loaded once, stays warm
#   - sha256(text) → vector cache → cache hit ≈ 0.01ms vs ~50ms ONNX call
#
# Result: recall latency drops from ~290ms to ~10-20ms on cache miss,
# ~1-3ms on cache hit (same query seen before).

_st_model_cache: Dict[Tuple[str, str], object] = {}
_embedding_cache: Dict[str, List[float]] = {}
_embedding_cache_lock = threading.Lock()


def _cache_key(model_name: str, text: str) -> str:
    return f"{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"


def _resolve_compute_device(device: str) -> str:
    """``auto`` → cuda if available, else cpu. ``cuda``/``gpu`` require CUDA or fall back with a warning."""
    raw = (device or "auto").strip()
    low = raw.lower()
    if low == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    if low in ("gpu", "cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        logger.warning("Memory device=cuda requested but CUDA is not available; using cpu")
        return "cpu"
    if low == "cpu":
        return "cpu"
    return raw


def _sentence_transformer_device_kw(device: str) -> Optional[str]:
    if not device or device.lower() == "cpu":
        return None
    return device


def _get_st_model(model_name: str, device: str = "cpu"):
    key = (model_name, device)
    if key not in _st_model_cache:
        from sentence_transformers import SentenceTransformer

        kw: dict = {}
        td = _sentence_transformer_device_kw(device)
        if td:
            kw["device"] = td
        _st_model_cache[key] = SentenceTransformer(model_name, **kw)
        logger.debug(
            "EmbeddingLayer: loaded SentenceTransformer %r on %s",
            model_name,
            td or "cpu",
        )
    return _st_model_cache[key]


def _embed(text: str, model_name: str = _DEFAULT_EMBEDDING_MODEL, device: str = "cpu") -> List[float]:
    key = _cache_key(model_name, text)
    with _embedding_cache_lock:
        if key in _embedding_cache:
            return _embedding_cache[key]
    model = _get_st_model(model_name, device)
    vec = model.encode(text, normalize_embeddings=True).tolist()
    with _embedding_cache_lock:
        _embedding_cache[key] = vec
    return vec


def _embed_batch(
    texts: List[str], model_name: str = _DEFAULT_EMBEDDING_MODEL, device: str = "cpu"
) -> List[List[float]]:
    """Batch embedding with cache — uncached texts are embedded in one model.encode() call."""
    results: List[Optional[List[float]]] = [None] * len(texts)
    uncached_idx: List[int] = []
    uncached_texts: List[str] = []

    with _embedding_cache_lock:
        for i, text in enumerate(texts):
            key = _cache_key(model_name, text)
            if key in _embedding_cache:
                results[i] = _embedding_cache[key]
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)

    if uncached_texts:
        model = _get_st_model(model_name, device)
        vecs = model.encode(
            uncached_texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        with _embedding_cache_lock:
            for i, text, vec in zip(uncached_idx, uncached_texts, vecs):
                _embedding_cache[_cache_key(model_name, text)] = vec
                results[i] = vec

    return results  # type: ignore[return-value]


class _VeriMemEmbeddingFunction:
    """
    ChromaDB-compatible embedding function backed by SentenceTransformer + cache.
    Passed to get_or_create_collection() so ChromaDB never touches ONNX.
    """

    def __init__(self, model_name: str = _DEFAULT_EMBEDDING_MODEL, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        # Pre-load model at construction time so first recall() is fast
        _get_st_model(model_name, device)

    def name(self) -> str:
        return f"verimem-st-{self._model_name}"

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: List[str]) -> List[List[float]]:
        return _embed_batch(input, self._model_name, self._device)


# Decay: default half-life of 30 days. After 30 days a chunk's freshness_score
# is 0.5× its similarity; after 60 days 0.25×. Set decay_days=0 to disable.
_DEFAULT_DECAY_DAYS = 30.0


# ---------------------------------------------------------------------------
# Chunker (same params as miner.py for consistency)
# ---------------------------------------------------------------------------


def _chunk(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= _MIN_CHUNK_SIZE:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _decay_factor(filed_at: Optional[str], half_life_days: float) -> float:
    """
    Exponential decay: returns 1.0 if no timestamp or decay disabled,
    otherwise e^(-age_days * ln(2) / half_life_days).

    At age=0              → 1.0  (full weight)
    At age=half_life_days → 0.5  (half weight)
    At age=2×half_life   → 0.25 (quarter weight)
    """
    if half_life_days <= 0 or not filed_at:
        return 1.0
    try:
        dt = datetime.fromisoformat(filed_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
        return math.exp(-age_days * math.log(2) / half_life_days)
    except Exception:
        return 1.0


def _age_seconds(filed_at: Optional[str]) -> Optional[float]:
    if not filed_at:
        return None
    try:
        dt = datetime.fromisoformat(filed_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None


def _chunk_id(text: str, source: str, chunk_index: int) -> str:
    """Stable ID: sha1 of content + source + index."""
    payload = f"{source}::{chunk_index}::{text[:200]}"
    return hashlib.sha1(payload.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------


class Memory:
    """
    Agent-centric memory store.

    Parameters
    ----------
    path : str
        Where to persist data. Defaults to ~/.verimem.
        Set to ":memory:" for an ephemeral in-process store (tests/notebooks).

    Storage backend (auto-selected):
        usearch installed  →  FastStore (Rust HNSW in-RAM + SQLite). Low-ms dense search.
        usearch missing    →  ChromaDB fallback (slower on disk).
        path == ":memory:" →  ChromaDB EphemeralClient (always, for test isolation).

    Default ``recall`` mode is ``rerank`` (dense + ms-marco-MiniLM cross-encoder). Call
    ``warm_retrieval_models()`` after startup to avoid first-hit model load; install
    ``optimum[onnxruntime]`` so the reranker can use ONNX (~lower CPU latency).
    """

    def __init__(
        self,
        path: str = _DEFAULT_PATH,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        device: str = "auto",
    ) -> None:
        self._path = path
        self._ephemeral = path == ":memory:"
        self._compute_device = _resolve_compute_device(device)
        self._ef = _VeriMemEmbeddingFunction(embedding_model, self._compute_device)
        self._client = None  # only set when using ChromaDB

        if self._ephemeral:
            # Ephemeral always uses ChromaDB EphemeralClient — FastStore needs a real path.
            # Each instance gets a unique collection to avoid cross-test contamination.
            self._client = chromadb.EphemeralClient()
            col_name = f"verimem_{uuid.uuid4().hex[:12]}"
            self._col = self._client.get_or_create_collection(
                name=col_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug("Memory: ephemeral ChromaDB store")
        elif _usearch_available():
            # usearch installed → use FastStore (Rust HNSW + SQLite).
            # ~10× faster recall than ChromaDB's disk-backed hnswlib.
            Path(path).mkdir(parents=True, exist_ok=True)
            self._col = FastStore(path)
            logger.debug("Memory: FastStore backend (usearch + SQLite) at %s", path)
        else:
            # ChromaDB fallback — works everywhere, no extra dependencies.
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=path)
            self._col = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug("Memory: ChromaDB backend at %s", path)

        # Hybrid path: skip full corpus scan + BM25 index rebuild when store revision/count/filter unchanged.
        self._hybrid_cache_key: Optional[tuple] = None
        self._hybrid_cache_bundle: Optional[Tuple[List[str], List[str], List[dict], BM25]] = None
        # Skip repeated COUNT(*) / col.count() on every recall while the store is unchanged.
        self._cached_chunk_count: Optional[int] = None

    def _hybrid_bundle_fingerprint(self, where_eff: Optional[dict]) -> tuple:
        wf = json.dumps(where_eff, sort_keys=True, default=str) if where_eff else ""
        cnt = self._count_chunks()
        if self._ephemeral:
            return ("eph", id(self._col), cnt, wf)
        rev = self.revision()
        return ("persist", rev if rev is not None else -1, cnt, wf)

    def _clear_hybrid_cache(self) -> None:
        self._hybrid_cache_key = None
        self._hybrid_cache_bundle = None
        self._cached_chunk_count = None

    def _count_chunks(self) -> int:
        if self._cached_chunk_count is not None:
            return self._cached_chunk_count
        n = self._col.count()
        self._cached_chunk_count = n
        return n

    def warm_retrieval_models(self) -> None:
        """
        Load embedding and cross-encoder weights into memory (no-op if already loaded).

        First ``recall(..., mode='rerank'|'hybrid_rerank')`` otherwise pays a one-time model
        load; call this after process start or ``Memory()`` creation to keep steady-state recall
        latency predictable (especially with ``optimum[onnxruntime]`` for the reranker).
        """
        _get_st_model(self._ef._model_name, self._compute_device)
        CrossEncoderReranker.get(device=self._compute_device)._load()

    def _hits_hybrid(
        self,
        query: str,
        query_vec: List[float],
        where: Optional[dict],
        topic: Optional[str],
        completeness: CompletenessFlags,
        rerank_pool: int,
        hybrid_lexical_weight: float,
        min_similarity: float,
        decay_days: float,
        use_ce: bool,
        top_k: int,
    ) -> List[RecallHit]:
        fp_req = self._hybrid_bundle_fingerprint(where)
        bm25_index: Optional[BM25] = None
        if self._hybrid_cache_key == fp_req and self._hybrid_cache_bundle is not None:
            corpus_ids, corpus_docs, corpus_metas, bm25_index = self._hybrid_cache_bundle
        else:
            corpus_ids, corpus_docs, corpus_metas = _scan_corpus_sorted(self._col, where)
            where_eff: Optional[dict] = where

            if not corpus_ids and where:
                completeness.filter_fallback = True
                fp_full = self._hybrid_bundle_fingerprint(None)
                if self._hybrid_cache_key == fp_full and self._hybrid_cache_bundle is not None:
                    corpus_ids, corpus_docs, corpus_metas, bm25_index = self._hybrid_cache_bundle
                else:
                    corpus_ids, corpus_docs, corpus_metas = _scan_corpus_sorted(self._col, None)
                where_eff = None

            if not corpus_ids:
                completeness.empty_after_filter = bool(topic)
                return []

            if bm25_index is None:
                bm25_index = BM25([tokenize(d) for d in corpus_docs])
                self._hybrid_cache_key = self._hybrid_bundle_fingerprint(where_eff)
                self._hybrid_cache_bundle = (corpus_ids, corpus_docs, corpus_metas, bm25_index)

        if not corpus_ids:
            completeness.empty_after_filter = bool(topic)
            return []

        id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
        n_fetch = min(rerank_pool, len(corpus_ids))

        def _do_query(w: Optional[dict]):
            kwargs: dict = {
                "query_embeddings": [query_vec],
                "n_results": max(1, min(n_fetch, len(corpus_ids))),
                "include": ["documents", "metadatas", "distances"],
            }
            if w:
                kwargs["where"] = w
            return self._col.query(**kwargs)

        try:
            raw = _do_query(where)
        except Exception as e:
            logger.error("Memory.recall hybrid query failed: %s", e)
            return []

        docs = raw["documents"][0]
        if not docs and where:
            completeness.filter_fallback = True
            try:
                raw = _do_query(None)
            except Exception as e2:
                logger.error("Memory.recall hybrid fallback query failed: %s", e2)
                return []
            docs = raw["documents"][0]

        if not docs:
            completeness.empty_after_filter = bool(topic)
            return []

        raw_ids = raw["ids"][0]
        dists = raw["distances"][0]
        cand_idx: List[int] = []
        cand_dists: List[float] = []
        dense_sim_by_idx: Dict[int, float] = {}
        for rid, dist in zip(raw_ids, dists):
            if rid in id_to_idx:
                i = id_to_idx[rid]
                cand_idx.append(i)
                cand_dists.append(float(dist))
                dense_sim_by_idx[i] = round(1.0 - float(dist), 4)

        if not cand_idx:
            return []

        fused_order = full_ranking_after_fusion(
            cand_idx,
            cand_dists,
            corpus_docs,
            query,
            lexical_weight=hybrid_lexical_weight,
            bm25_index=bm25_index,
        )

        hits: List[RecallHit] = []
        for idx in fused_order:
            if len(hits) >= rerank_pool:
                break
            sim = dense_sim_by_idx.get(idx, 0.0)
            if sim < min_similarity:
                continue
            meta = corpus_metas[idx]
            filed_at = meta.get("filed_at")
            decay = _decay_factor(filed_at, decay_days)
            hits.append(
                RecallHit(
                    text=corpus_docs[idx],
                    wing=meta.get("topic", _DEFAULT_TOPIC),
                    room="memory",
                    source_file=meta.get("source", "direct"),
                    similarity=sim,
                    chunk_index=meta.get("chunk_index", 0),
                    filed_at=filed_at,
                    drawer_id=corpus_ids[idx],
                    freshness_score=round(sim * decay, 4),
                    ingest_age_seconds=_age_seconds(filed_at),
                )
            )

        if use_ce and hits:
            reranker = CrossEncoderReranker.get(device=self._compute_device)
            hits = reranker.rerank(query, hits, top_n=top_k)
        else:
            hits = hits[:top_k]

        return hits

    def _postprocess_recall(
        self,
        query: str,
        hits: List[RecallHit],
        completeness: CompletenessFlags,
        policy_version: str,
        top_k: int,
        decay_days: float,
        include_graph: bool,
        sync_contradictions: bool = False,
        include_uncertainty: bool = True,
        uncertainty_softmax_tau: float = 0.12,
        uncertainty_min_confidence: float = 0.35,
        uncertainty_min_best_similarity: float = 0.52,
    ) -> ContextPacket:
        if decay_days > 0:
            hits = sorted(hits, key=lambda h: h.freshness_score, reverse=True)

        if len(hits) >= top_k:
            completeness.hits_truncated = True

        contradictions: List[ContradictionFlag] = []
        if len(hits) >= 2:
            if sync_contradictions:
                try:
                    persist = not self._ephemeral
                    contradictions = score_contradictions_sync(
                        hits,
                        store_path=self._path if persist else None,
                        persist_cache=persist,
                    )
                except Exception as exc:
                    logger.warning("Memory: sync contradiction scoring failed: %s", exc)
            elif not self._ephemeral:
                nli = BackgroundNLI.for_store(self._path)
                contradictions = nli.lookup(hits)
                pending = nli.submit_hits(hits)
                if pending and not contradictions:
                    completeness.contradiction_check_pending = True
        if len(hits) < 2:
            completeness.contradiction_check_skipped = True

        graph_entities = None
        if include_graph and not self._ephemeral and hits:
            chunk_ids = [h.drawer_id for h in hits if h.drawer_id]
            if chunk_ids:
                mgraph = MemoryGraph.for_path(self._path)
                entities = mgraph.entities_for_chunks(chunk_ids)
                if entities:
                    graph_entities = [e.to_dict() for e in entities]
            if not graph_entities:
                bg = BackgroundGraph.for_path(self._path)
                if bg.pending_count() > 0:
                    completeness.contradiction_check_pending = True

        retrieval_u = None
        if include_uncertainty:
            sims = [h.similarity for h in hits]
            retrieval_u = compute_retrieval_uncertainty(
                sims,
                softmax_tau=uncertainty_softmax_tau,
                min_confidence=uncertainty_min_confidence,
                min_best_similarity=uncertainty_min_best_similarity,
            )

        return ContextPacket(
            query=query,
            hits=hits,
            contradictions=contradictions,
            completeness=completeness,
            policy_version=policy_version,
            store_revision=self.revision(),
            graph_entities=graph_entities,
            retrieval_uncertainty=retrieval_u,
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        source: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[str]:
        """
        Store text in memory.

        Parameters
        ----------
        text   : the raw text to store (conversations, docs, facts — anything)
        source : optional label for where this came from (filename, URL, agent name…)
        topic  : optional category/tag for filtering at recall time

        Returns
        -------
        List of chunk IDs that were stored (useful for forget()).
        Already-existing chunks are skipped (idempotent).
        """
        if not text or not text.strip():
            return []

        source = source or "direct"
        topic = topic or _DEFAULT_TOPIC
        now = datetime.now(timezone.utc).isoformat()

        chunks = _chunk(text)
        ids_to_add: List[str] = []
        docs_to_add: List[str] = []
        metas_to_add: List[dict] = []
        ids_stored: List[str] = []

        for i, chunk in enumerate(chunks):
            cid = _chunk_id(chunk, source, i)
            # Skip already-stored chunks (content-addressed, idempotent)
            try:
                existing = self._col.get(ids=[cid], include=["metadatas"])
            except Exception:
                existing = {"ids": []}
            if existing["ids"]:
                ids_stored.append(cid)
                continue
            ids_to_add.append(cid)
            docs_to_add.append(chunk)
            metas_to_add.append(
                {
                    "source": source,
                    "topic": topic,
                    "chunk_index": i,
                    "filed_at": now,
                }
            )
            ids_stored.append(cid)

        if ids_to_add:
            # Embed all new chunks in one batched call (cached — same text never re-embedded)
            embeddings_to_add = _embed_batch(docs_to_add, self._ef._model_name, self._compute_device)
            self._col.add(
                ids=ids_to_add,
                documents=docs_to_add,
                embeddings=embeddings_to_add,
                metadatas=metas_to_add,
            )
            if not self._ephemeral:
                bump_revision(self._path)
                # Submit to background REBEL graph extraction — non-blocking
                bg = BackgroundGraph.for_path(self._path)
                for cid, doc, meta in zip(ids_to_add, docs_to_add, metas_to_add):
                    bg.submit(doc, cid, meta.get("filed_at"))
            self._clear_hybrid_cache()
            logger.debug(
                "Memory.remember: stored %d chunks from source=%r", len(ids_to_add), source
            )

        return ids_stored

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 5,
        topic: Optional[str] = None,
        mode: str = DEFAULT_RETRIEVAL_MODE,
        rerank: Optional[bool] = None,
        rerank_pool: int = 20,
        hybrid_lexical_weight: float = 0.35,
        min_similarity: float = 0.0,
        decay_days: float = _DEFAULT_DECAY_DAYS,
        include_graph: bool = False,
        sync_contradictions: bool = False,
        include_uncertainty: bool = True,
        uncertainty_softmax_tau: float = 0.12,
        uncertainty_min_confidence: float = 0.35,
        uncertainty_min_best_similarity: float = 0.52,
    ) -> ContextPacket:
        """
        Recall memories relevant to query.

        Parameters
        ----------
        query        : what you want to remember
        top_k        : number of hits to return
        topic        : filter to a specific topic (set at remember() time)
        mode         : ``rerank`` (dense + cross-encoder, default), ``raw`` (dense only),
                       ``hybrid`` (dense + BM25), ``hybrid_rerank`` (hybrid + CE).
        rerank       : deprecated; use ``mode`` instead. If set, maps to ``raw`` / ``rerank``.
        rerank_pool  : candidate pool size before cross-encoder; also dense fetch size for hybrid.
        hybrid_lexical_weight : BM25 weight in ``[0, 1]`` for hybrid modes (dense gets ``1 - w``).
        min_similarity: drop hits below this cosine similarity
        decay_days   : freshness half-life in days (default 30). Set to 0 to disable.
        include_graph : if True, attach entity nodes from recalled chunks to ``graph_entities``.
        sync_contradictions : if True, run batched NLI on all eligible hit pairs **before** returning
            (same model/threshold as background NLI). Adds latency (~tens of ms CPU for typical k);
            persists to ``contradiction_cache.db`` on disk stores. Ignores async lookup/submit for this call.
        include_uncertainty : if True (default), attach ``retrieval_uncertainty`` / ``retrieval`` in
            ``to_dict()`` / ``to_simple()`` from hit similarities (entropy + best-match; advisory flag).
        uncertainty_softmax_tau : temperature for softmax over hit similarities when computing ambiguity.
        uncertainty_min_confidence : flag ``retrieval_insufficient`` if ``confidence_q`` is below this.
        uncertainty_min_best_similarity : flag ``retrieval_insufficient`` if best hit similarity is below this.

        Returns
        -------
        ContextPacket — call .to_simple() for clean agent output,
                        .to_dict() for full provenance and debugging.
        """
        resolved = _resolve_recall_mode(mode, rerank)
        use_hybrid = resolved in ("hybrid", "hybrid_rerank")
        use_ce = resolved in ("rerank", "hybrid_rerank")

        completeness = CompletenessFlags()
        where = {"topic": topic} if topic else None
        query_vec = _embed(query, self._ef._model_name, self._compute_device)

        if use_hybrid:
            hits = self._hits_hybrid(
                query,
                query_vec,
                where,
                topic,
                completeness,
                rerank_pool,
                hybrid_lexical_weight,
                min_similarity,
                decay_days,
                use_ce,
                top_k,
            )
            return self._postprocess_recall(
                query,
                hits,
                completeness,
                resolved,
                top_k,
                decay_days,
                include_graph,
                sync_contradictions,
                include_uncertainty,
                uncertainty_softmax_tau,
                uncertainty_min_confidence,
                uncertainty_min_best_similarity,
            )

        chroma_n = rerank_pool if use_ce else top_k

        try:
            kwargs: dict = {
                "query_embeddings": [query_vec],
                "n_results": min(chroma_n, self._count_chunks() or 1),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            raw = self._col.query(**kwargs)
        except Exception as e:
            if where:
                completeness.filter_fallback = True
                try:
                    kwargs.pop("where", None)
                    kwargs["n_results"] = min(chroma_n, self._count_chunks() or 1)
                    raw = self._col.query(**kwargs)
                except Exception as e2:
                    logger.error("Memory.recall failed: %s", e2)
                    return ContextPacket(
                        query=query,
                        completeness=completeness,
                        retrieval_uncertainty=compute_retrieval_uncertainty(
                            [],
                            softmax_tau=uncertainty_softmax_tau,
                            min_confidence=uncertainty_min_confidence,
                            min_best_similarity=uncertainty_min_best_similarity,
                        )
                        if include_uncertainty
                        else None,
                    )
            else:
                logger.error("Memory.recall failed: %s", e)
                return ContextPacket(
                    query=query,
                    completeness=completeness,
                    retrieval_uncertainty=compute_retrieval_uncertainty(
                        [],
                        softmax_tau=uncertainty_softmax_tau,
                        min_confidence=uncertainty_min_confidence,
                        min_best_similarity=uncertainty_min_best_similarity,
                    )
                    if include_uncertainty
                    else None,
                )

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]
        ids = raw["ids"][0] if raw.get("ids") else [None] * len(docs)

        if not docs:
            completeness.empty_after_filter = bool(where)
            return ContextPacket(
                query=query,
                completeness=completeness,
                store_revision=self.revision(),
                retrieval_uncertainty=compute_retrieval_uncertainty(
                    [],
                    softmax_tau=uncertainty_softmax_tau,
                    min_confidence=uncertainty_min_confidence,
                    min_best_similarity=uncertainty_min_best_similarity,
                )
                if include_uncertainty
                else None,
            )

        hits = []
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            sim = round(1 - dist, 4)
            if sim < min_similarity:
                continue
            filed_at = meta.get("filed_at")
            decay = _decay_factor(filed_at, decay_days)
            hits.append(
                RecallHit(
                    text=doc,
                    wing=meta.get("topic", _DEFAULT_TOPIC),
                    room="memory",
                    source_file=meta.get("source", "direct"),
                    similarity=sim,
                    chunk_index=meta.get("chunk_index", 0),
                    filed_at=filed_at,
                    drawer_id=cid,
                    freshness_score=round(sim * decay, 4),
                    ingest_age_seconds=_age_seconds(filed_at),
                )
            )

        if use_ce and hits:
            reranker = CrossEncoderReranker.get(device=self._compute_device)
            hits = reranker.rerank(query, hits, top_n=top_k)
        else:
            hits = hits[:top_k]

        return self._postprocess_recall(
            query,
            hits,
            completeness,
            resolved,
            top_k,
            decay_days,
            include_graph,
            sync_contradictions,
            include_uncertainty,
            uncertainty_softmax_tau,
            uncertainty_min_confidence,
            uncertainty_min_best_similarity,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def forget(self, chunk_id: str) -> None:
        """Delete a specific chunk by ID."""
        try:
            self._col.delete(ids=[chunk_id])
            if not self._ephemeral:
                bump_revision(self._path)
            self._clear_hybrid_cache()
        except Exception as e:
            logger.warning("Memory.forget(%r) failed: %s", chunk_id, e)

    def count(self) -> int:
        """Total number of chunks stored."""
        try:
            return self._count_chunks()
        except Exception:
            return 0

    def revision(self) -> Optional[int]:
        """Monotonic write counter — increments on every remember()/forget()."""
        if self._ephemeral:
            return None
        return get_revision(self._path)

    def recall_related(
        self,
        entity: str,
        hops: int = 1,
    ) -> GraphRecallResult:
        """
        Entity-centric graph query — returns all relationships for an entity.

        Uses spaCy NER + dependency parsing extracted at remember() time.
        No LLM required. Covers named entities: people, orgs, products, places.

        Parameters
        ----------
        entity : str   — entity to look up ("Alice", "MySQL", "auth service")
        hops   : int   — traversal depth (1=direct, 2=friends-of-friends)

        Returns
        -------
        GraphRecallResult with:
          .entity          — canonical entity name
          .entity_type     — PERSON / ORG / PRODUCT / ...
          .triples         — List[RelationTriple]: (subject, relation, object)
          .related_entities — List[EntityNode]: everything connected within hops
          .chunk_ids        — chunk IDs from remember() that mention this entity

        Example
        -------
        result = mem.recall_related("Alice")
        for t in result.triples:
            print(f"{t.subject} --[{t.relation}]--> {t.obj}")
        # Alice --[deployed]--> MySQL
        # Alice --[manages]--> auth service
        """
        if self._ephemeral:
            # Graph not persisted for ephemeral stores — return empty result
            return GraphRecallResult(entity=entity, entity_type="")
        graph = MemoryGraph.for_path(self._path)
        return graph.query_related(entity, hops=hops)

    def graph_stats(self) -> dict:
        """Return entity and triple counts from the graph."""
        if self._ephemeral:
            return {"entities": 0, "triples": 0}
        graph = MemoryGraph.for_path(self._path)
        return {
            "entities": graph.entity_count(),
            "triples": graph.triple_count(),
        }

    def __repr__(self) -> str:
        return f"Memory(path={self._path!r}, chunks={self._count_chunks()})"
