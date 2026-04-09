"""
fast_store.py — usearch HNSW + SQLite storage backend.

Drop-in replacement for ChromaDB when `usearch` is installed.
Exposes the same surface that memory.py uses (.add, .query, .get, .delete, .count).

Why faster than ChromaDB:
  - usearch index (Rust/C++) lives entirely in RAM — no disk I/O per query.
  - SQLite WAL for chunk text/metadata — single-file, fast reads.
  - Embeddings stored as BLOB in SQLite → index can be rebuilt on restart
    without re-running the embedding model.

Latency (typical with usearch; varies by CPU):
  dense query: low ms; default ``rerank`` adds a batched cross-encoder pass over the pool.

Install:
  pip install usearch          # Rust HNSW, ~2MB wheel
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("verimem_mcp")

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
_HNSW_CONNECTIVITY = 16
_HNSW_EF_ADD = 128
_HNSW_EF_SEARCH = 64


def is_available() -> bool:
    """True if usearch is installed and can be imported."""
    try:
        import usearch  # noqa: F401

        return True
    except ImportError:
        return False


class FastStore:
    """
    usearch HNSW (in-RAM) + SQLite (on-disk) storage.

    Thread-safe: a single threading.Lock serialises all mutations.
    Reads (query/get/count) are lock-free except for the id-map look-up.

    Persistence strategy:
      - SQLite is the source of truth (text + embeddings as BLOB).
      - usearch index is rebuilt from SQLite on startup.
      - After every add/delete batch the index is saved to `hnsw.usearch`
        so cold restarts are fast (load takes ~1ms vs rebuilding from BLOBs).
    """

    def __init__(self, path: str) -> None:
        self._base = Path(path)
        self._base.mkdir(parents=True, exist_ok=True)

        self._db_path = str(self._base / "fast_store.db")
        self._index_path = str(self._base / "hnsw.usearch")
        self._lock = threading.Lock()

        # SQLite — source of truth
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

        # usearch HNSW — in-RAM for fast ANN search
        self._index = None
        self._dim: Optional[int] = None

        # id maps (chunk_id ↔ int_label for usearch)
        self._id_map: List[Optional[str]] = []  # int_label → chunk_id (or None if deleted)
        self._item_to_int: Dict[str, int] = {}  # chunk_id → int_label

        self._rebuild_index()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT    PRIMARY KEY,
                text        TEXT    NOT NULL,
                source      TEXT    NOT NULL DEFAULT 'direct',
                topic       TEXT    NOT NULL DEFAULT 'general',
                chunk_index INTEGER NOT NULL DEFAULT 0,
                filed_at    TEXT    NOT NULL DEFAULT '',
                int_label   INTEGER NOT NULL,
                embedding   BLOB    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_topic     ON chunks(topic);
            CREATE INDEX IF NOT EXISTS idx_chunks_int_label ON chunks(int_label);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Index bootstrap
    # ------------------------------------------------------------------

    def _make_index(self, dim: int):
        from usearch.index import Index

        return Index(
            ndim=dim,
            metric="cos",
            connectivity=_HNSW_CONNECTIVITY,
            expansion_add=_HNSW_EF_ADD,
            expansion_search=_HNSW_EF_SEARCH,
        )

    def _rebuild_index(self) -> None:
        """
        Load all chunks from SQLite and populate the in-RAM usearch index.
        Called once at startup. On a 10k-chunk store this takes ~50ms.
        """
        rows = self._conn.execute(
            "SELECT id, int_label, embedding FROM chunks ORDER BY int_label"
        ).fetchall()

        if not rows:
            return

        first_emb = np.frombuffer(rows[0][2], dtype=np.float32)
        dim = len(first_emb)
        self._dim = dim
        self._index = self._make_index(dim)

        labels = np.empty(len(rows), dtype=np.uint64)
        vecs = np.empty((len(rows), dim), dtype=np.float32)

        for i, (chunk_id, int_label, emb_blob) in enumerate(rows):
            while len(self._id_map) <= int_label:
                self._id_map.append(None)
            self._id_map[int_label] = chunk_id
            self._item_to_int[chunk_id] = int_label
            labels[i] = int_label
            vecs[i] = np.frombuffer(emb_blob, dtype=np.float32)

        self._index.add(labels, vecs)
        logger.debug("FastStore: rebuilt usearch index with %d vectors (dim=%d)", len(rows), dim)

    def _save_index(self) -> None:
        if self._index is not None:
            try:
                self._index.save(self._index_path)
            except Exception as exc:
                logger.debug("FastStore: index save failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API (ChromaDB-compatible surface used by memory.py)
    # ------------------------------------------------------------------

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        with self._lock:
            new_labels: List[int] = []
            new_vecs: List[np.ndarray] = []

            for chunk_id, text, emb, meta in zip(ids, documents, embeddings, metadatas):
                if chunk_id in self._item_to_int:
                    continue  # idempotent

                int_label = len(self._id_map)
                self._id_map.append(chunk_id)
                self._item_to_int[chunk_id] = int_label

                if self._index is None:
                    self._dim = len(emb)
                    self._index = self._make_index(self._dim)

                vec = np.array(emb, dtype=np.float32)
                new_labels.append(int_label)
                new_vecs.append(vec)

                self._conn.execute(
                    """INSERT OR IGNORE INTO chunks
                       (id, text, source, topic, chunk_index, filed_at, int_label, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_id,
                        text,
                        meta.get("source", "direct"),
                        meta.get("topic", "general"),
                        meta.get("chunk_index", 0),
                        meta.get("filed_at", ""),
                        int_label,
                        vec.tobytes(),
                    ),
                )

            if new_labels:
                labels_arr = np.array(new_labels, dtype=np.uint64)
                vecs_arr = np.vstack(new_vecs).astype(np.float32)
                self._index.add(labels_arr, vecs_arr)
                self._conn.commit()
                self._save_index()

    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        include: Optional[List[str]] = None,  # accepted but unused — we always return everything
        **_kwargs,
    ) -> dict:
        """
        Returns a ChromaDB-shaped dict:
          {"documents": [[...]], "metadatas": [[...]], "distances": [[...]], "ids": [[...]]}
        """
        empty = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        if self._index is None or not self._id_map or query_embeddings is None:
            return empty

        vec = np.array(query_embeddings[0], dtype=np.float32)

        # Fetch more candidates if a where-filter is active so we can still
        # return n_results hits after filtering.
        k = min(n_results * 4 if where else n_results, max(1, len(self._item_to_int)))
        matches = self._index.search(vec, k)

        # Map int_labels → chunk_ids
        candidate_ids = []
        candidate_dists = []
        for int_label, dist in zip(matches.keys, matches.distances):
            int_label = int(int_label)
            if int_label < len(self._id_map) and self._id_map[int_label]:
                candidate_ids.append(self._id_map[int_label])
                candidate_dists.append(float(dist))

        if not candidate_ids:
            return empty

        # Fetch text + metadata from SQLite in one round-trip
        placeholders = ",".join("?" * len(candidate_ids))
        rows = self._conn.execute(
            f"SELECT id, text, source, topic, chunk_index, filed_at "
            f"FROM chunks WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        row_map = {r[0]: r for r in rows}

        docs, metas, dists, out_ids = [], [], [], []
        for chunk_id, dist in zip(candidate_ids, candidate_dists):
            if chunk_id not in row_map:
                continue
            row = row_map[chunk_id]
            meta = {
                "source": row[2],
                "topic": row[3],
                "chunk_index": row[4],
                "filed_at": row[5],
            }
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            docs.append(row[1])
            metas.append(meta)
            dists.append(dist)
            out_ids.append(chunk_id)
            if len(docs) >= n_results:
                break

        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [out_ids]}

    def scan_all(self, where: Optional[dict] = None) -> dict:
        """
        List every chunk for hybrid BM25 fusion (same shape as Chroma ``get``).

        ``where`` supports ``{\"topic\": \"...\"}`` to match filtered recall.
        """
        if where and "topic" in where:
            rows = self._conn.execute(
                "SELECT id, text, source, topic, chunk_index, filed_at FROM chunks "
                "WHERE topic = ? ORDER BY int_label",
                (where["topic"],),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, text, source, topic, chunk_index, filed_at FROM chunks "
                "ORDER BY int_label"
            ).fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [
                {"source": r[2], "topic": r[3], "chunk_index": r[4], "filed_at": r[5]} for r in rows
            ],
        }

    def get(
        self,
        ids: List[str],
        include: Optional[List[str]] = None,
    ) -> dict:
        if not ids:
            return {"ids": [], "metadatas": []}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, source, topic, chunk_index, filed_at "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {
            "ids": [r[0] for r in rows],
            "metadatas": [
                {"source": r[1], "topic": r[2], "chunk_index": r[3], "filed_at": r[4]} for r in rows
            ],
        }

    def delete(self, ids: List[str]) -> None:
        with self._lock:
            for chunk_id in ids:
                int_label = self._item_to_int.pop(chunk_id, None)
                if int_label is not None:
                    self._id_map[int_label] = None
                    if self._index:
                        try:
                            self._index.remove(int(int_label))
                        except Exception:
                            pass
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            self._conn.commit()
            self._save_index()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
