"""
background_nli.py — Async NLI contradiction detection, Ashnode-style.

A daemon thread consumes (id_a, text_a, id_b, text_b) pairs from a queue,
scores them with a cross-encoder NLI model, and caches results in a local
SQLite DB. Callers get zero search-latency: lookup() is instant from cache,
and submit_hits() fires off pairs to the background for next time.

Architecture mirrors Ashnode's BackgroundBrain:
  - ingest → submit pairs for background scoring
  - query  → lookup cached scores → surface ContradictionFlags
  - daemon thread lazy-loads the NLI model on first use

If the NLI model fails to load (e.g. import error), submit_hits() no-ops and lookup() returns [].

sentence-transformers is a **core** dependency for VeriMem. Optional: ``pip install verimem[nli]`` for ONNX-backed cross-encoder speedups elsewhere in the stack.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .recall import ContradictionFlag, RecallHit

logger = logging.getLogger("verimem_mcp")

_DB_NAME = "contradiction_cache.db"
_DEFAULT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
_CONTRADICTION_THRESHOLD = 0.5  # minimum contradiction softmax score to surface as flag
_MIN_SIMILARITY_FOR_NLI = 0.45  # skip pairs where both hits are low-confidence matches
_QUEUE_TIMEOUT_S = 15  # worker drains until idle for this many seconds

# Shared CrossEncoder for synchronous batched scoring (one load per process / model name).
_sync_ce_lock = threading.Lock()
_sync_ce_model = None
_sync_ce_model_name: Optional[str] = None


def _db_path(store_path: str) -> Path:
    return Path(store_path) / _DB_NAME


def _init_db(db: Path) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute("""
        CREATE TABLE IF NOT EXISTS nli_cache (
            id_a       TEXT NOT NULL,
            id_b       TEXT NOT NULL,
            label      TEXT NOT NULL,
            score      REAL NOT NULL,
            checked_at TEXT NOT NULL,
            PRIMARY KEY (id_a, id_b)
        )
    """)
    con.commit()
    con.close()


def score_contradictions_sync(
    hits: List["RecallHit"],
    *,
    store_path: Optional[str] = None,
    persist_cache: bool = True,
    model_name: str = _DEFAULT_MODEL,
) -> List["ContradictionFlag"]:
    """
    Score all eligible hit pairs in **one batched** cross-encoder forward (sync path).

    Same eligibility and threshold as the background worker. Optionally persists rows to
    ``contradiction_cache.db`` when ``store_path`` is set and not ``\":memory:\"``.
    """
    from .recall import ContradictionFlag  # local import

    eligible = [h for h in hits if h.drawer_id and h.similarity >= _MIN_SIMILARITY_FOR_NLI]
    if len(eligible) < 2:
        return []

    id_to_pos = {h.drawer_id: i for i, h in enumerate(hits) if h.drawer_id}
    pair_rows: list[tuple[str, str, str, str]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            ha, hb = eligible[i], eligible[j]
            pair_rows.append((ha.drawer_id, ha.text, hb.drawer_id, hb.text))

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.debug("score_contradictions_sync: sentence-transformers not installed")
        return []

    global _sync_ce_model, _sync_ce_model_name
    with _sync_ce_lock:
        if _sync_ce_model is None or _sync_ce_model_name != model_name:
            _sync_ce_model = CrossEncoder(model_name)
            _sync_ce_model_name = model_name
        model = _sync_ce_model

    contra_idx = BackgroundNLI._get_contra_idx(model)
    text_pairs = [(r[1], r[3]) for r in pair_rows]

    try:
        scores = model.predict(
            text_pairs,
            apply_softmax=True,
            show_progress_bar=False,
            batch_size=max(1, len(text_pairs)),
        )
    except Exception as exc:
        logger.warning("score_contradictions_sync: predict failed: %s", exc)
        return []

    flags: List[ContradictionFlag] = []
    now = datetime.now(timezone.utc).isoformat()

    can_persist = bool(
        persist_cache
        and store_path
        and store_path != ":memory:"
    )
    con = None
    if can_persist:
        try:
            db = _db_path(store_path)
            _init_db(db)
            con = sqlite3.connect(str(db))
        except Exception as exc:
            logger.debug("score_contradictions_sync: cache open failed: %s", exc)
            can_persist = False

    for (id_a, _, id_b, _), score_row in zip(pair_rows, scores):
        score_row_list = score_row
        contra_score = float(score_row_list[contra_idx])
        label_idx = int(score_row_list.argmax())
        label_map = {contra_idx: "contradiction"}
        label = label_map.get(label_idx, "non-contradiction")
        if contra_score >= _CONTRADICTION_THRESHOLD:
            label = "contradiction"
        if can_persist and con is not None:
            try:
                con.execute(
                    "INSERT OR REPLACE INTO nli_cache "
                    "(id_a, id_b, label, score, checked_at) VALUES (?,?,?,?,?)",
                    (id_a, id_b, label, contra_score, now),
                )
            except Exception as exc:
                logger.debug("score_contradictions_sync: cache write failed: %s", exc)
        if contra_score >= _CONTRADICTION_THRESHOLD:
            ia = id_to_pos.get(id_a, 0)
            ib = id_to_pos.get(id_b, 0)
            flags.append(
                ContradictionFlag(
                    hit_a_idx=ia,
                    hit_b_idx=ib,
                    reason=f"NLI contradiction (score={contra_score:.2f})",
                    confidence=round(contra_score, 3),
                )
            )

    if con is not None:
        try:
            con.commit()
            con.close()
        except Exception:
            pass

    return flags


class BackgroundNLI:
    """
    One instance per on-disk store, shared across searches.

    Usage
    -----
    nli = BackgroundNLI.for_store(store_path)

    # At search time — zero latency:
    flags   = nli.lookup(hits)      # cached results from previous runs
    pending = nli.submit_hits(hits) # queue pairs for background scoring
    # pending=True  → set completeness.contradiction_check_pending
    # pending=False → all pairs already cached (or no pairs to check)
    """

    _instances: dict[str, "BackgroundNLI"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def for_store(cls, store_path: str, model: str = _DEFAULT_MODEL) -> "BackgroundNLI":
        """Return the singleton BackgroundNLI for this store directory."""
        with cls._registry_lock:
            if store_path not in cls._instances:
                cls._instances[store_path] = cls(store_path, model)
            return cls._instances[store_path]

    def __init__(self, store_path: str, model: str = _DEFAULT_MODEL) -> None:
        self._store_path = store_path
        self._model_name = model
        self._db = _db_path(store_path)
        _init_db(self._db)
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_hits(self, hits: List["RecallHit"]) -> bool:
        """
        Enqueue NLI jobs for all uncached pairs in the hit set.

        Returns True if any new pairs were enqueued (caller should set
        completeness.contradiction_check_pending = True on this packet).
        Returns False if everything was already cached or no pairs exist.
        """
        eligible = [h for h in hits if h.drawer_id and h.similarity >= _MIN_SIMILARITY_FOR_NLI]
        if len(eligible) < 2:
            return False

        pairs: list[tuple[str, str, str, str]] = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                ha, hb = eligible[i], eligible[j]
                pairs.append((ha.drawer_id, ha.text, hb.drawer_id, hb.text))

        queued = False
        try:
            con = sqlite3.connect(str(self._db))
            cur = con.cursor()
            for id_a, text_a, id_b, text_b in pairs:
                row = cur.execute(
                    "SELECT 1 FROM nli_cache WHERE (id_a=? AND id_b=?) OR (id_a=? AND id_b=?)",
                    (id_a, id_b, id_b, id_a),
                ).fetchone()
                if not row:
                    self._queue.put((id_a, text_a, id_b, text_b))
                    queued = True
            con.close()
        except Exception as exc:
            logger.debug("BackgroundNLI.submit_hits batch probe error: %s", exc)
            for id_a, text_a, id_b, text_b in pairs:
                if not self._is_cached(id_a, id_b):
                    self._queue.put((id_a, text_a, id_b, text_b))
                    queued = True

        if queued:
            self._ensure_worker()

        return queued

    def lookup(self, hits: List["RecallHit"]) -> List["ContradictionFlag"]:
        """
        Return ContradictionFlags from the cache for the given hit set.

        Only returns flags that have already been computed (zero latency).
        Pairs not yet in the cache are simply omitted — they'll be there
        next time after submit_hits() processes them in the background.
        """
        from .recall import ContradictionFlag  # local import to avoid circular

        ids = [h.drawer_id for h in hits if h.drawer_id]
        if len(ids) < 2:
            return []

        hit_id_to_idx = {h.drawer_id: i for i, h in enumerate(hits) if h.drawer_id}
        flags: List[ContradictionFlag] = []

        try:
            con = sqlite3.connect(str(self._db))
            cur = con.cursor()
            for i, id_a in enumerate(ids):
                for id_b in ids[i + 1 :]:
                    row = cur.execute(
                        """SELECT label, score FROM nli_cache
                           WHERE (id_a=? AND id_b=?) OR (id_a=? AND id_b=?)""",
                        (id_a, id_b, id_b, id_a),
                    ).fetchone()
                    if (
                        row
                        and row[0] == "contradiction"
                        and float(row[1]) >= _CONTRADICTION_THRESHOLD
                    ):
                        flags.append(
                            ContradictionFlag(
                                hit_a_idx=hit_id_to_idx.get(id_a, 0),
                                hit_b_idx=hit_id_to_idx.get(id_b, 0),
                                reason=f"NLI contradiction (score={float(row[1]):.2f})",
                                confidence=round(float(row[1]), 3),
                            )
                        )
            con.close()
        except Exception as exc:
            logger.debug("BackgroundNLI.lookup error: %s", exc)

        return flags

    def pending_count(self) -> int:
        """How many pairs are still waiting to be scored."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_cached(self, id_a: str, id_b: str) -> bool:
        try:
            con = sqlite3.connect(str(self._db))
            row = con.execute(
                "SELECT 1 FROM nli_cache WHERE (id_a=? AND id_b=?) OR (id_a=? AND id_b=?)",
                (id_a, id_b, id_b, id_a),
            ).fetchone()
            con.close()
            return row is not None
        except Exception:
            return False

    def _ensure_worker(self) -> None:
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._worker,
                    name="verimem-nli",
                    daemon=True,
                )
                self._thread.start()
                logger.debug("BackgroundNLI worker started for %s", self._store_path)

    def _worker(self) -> None:
        """Daemon worker: lazy-load NLI model, drain queue, write cache."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.debug(
                "sentence-transformers not installed — NLI worker exiting. "
                "Install with: pip install sentence-transformers"
            )
            # Drain to prevent unbounded queue growth
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return

        try:
            model = CrossEncoder(self._model_name)
            # Get contradiction label index from the model's config
            contra_idx = self._get_contra_idx(model)
        except Exception as exc:
            logger.warning("BackgroundNLI: failed to load model %r: %s", self._model_name, exc)
            return

        logger.debug("BackgroundNLI: model %r loaded, draining queue", self._model_name)

        con = sqlite3.connect(str(self._db))
        batch: list[tuple] = []

        def _flush(batch: list) -> None:
            if not batch:
                return
            pairs = [(id_a, text_a, id_b, text_b) for id_a, text_a, id_b, text_b in batch]
            try:
                scores = model.predict(
                    [(p[1], p[3]) for p in pairs],
                    apply_softmax=True,
                    show_progress_bar=False,
                )
                now = datetime.now(timezone.utc).isoformat()
                for (id_a, _, id_b, _), score_row in zip(pairs, scores):
                    contra_score = float(score_row[contra_idx])
                    # Pick the dominant label
                    label_idx = int(score_row.argmax())
                    label_map = {contra_idx: "contradiction"}
                    # entailment and neutral get generic labels
                    label = label_map.get(label_idx, "non-contradiction")
                    # Always record the contradiction score for threshold checking
                    if contra_score >= _CONTRADICTION_THRESHOLD:
                        label = "contradiction"
                    con.execute(
                        "INSERT OR REPLACE INTO nli_cache "
                        "(id_a, id_b, label, score, checked_at) VALUES (?,?,?,?,?)",
                        (id_a, id_b, label, contra_score, now),
                    )
                con.commit()
            except Exception as exc:
                logger.debug("BackgroundNLI flush error: %s", exc)

        try:
            while True:
                try:
                    item = self._queue.get(timeout=_QUEUE_TIMEOUT_S)
                    batch.append(item)
                    # Micro-batch: process up to 16 pairs at a time
                    if len(batch) >= 16:
                        _flush(batch)
                        batch = []
                except queue.Empty:
                    _flush(batch)
                    batch = []
                    break  # idle for QUEUE_TIMEOUT_S — exit, thread restarts if new work arrives
        finally:
            _flush(batch)
            con.close()
            logger.debug("BackgroundNLI worker finished for %s", self._store_path)

    @staticmethod
    def _get_contra_idx(model) -> int:
        """Return the index of the 'contradiction' label in the model's output."""
        try:
            label2id = model.model.config.label2id
            # Normalise to lowercase
            label2id = {k.lower(): v for k, v in label2id.items()}
            return int(label2id.get("contradiction", 0))
        except Exception:
            return 0  # convention for most NLI cross-encoders
