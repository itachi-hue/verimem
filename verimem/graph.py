"""
graph.py — Entity-relationship graph for VeriMem Memory.

Two layers:

1. MemoryGraph   — SQLite storage + query. Exposes recall_related().

2. BackgroundGraph — Daemon thread that lazy-loads GLiNER small (~67MB,
                     urchade/gliner_small-v2.1) for zero-shot NER, plus the
                     already-installed spaCy sm model for relation extraction
                     between the entities GLiNER found.

                     Why this split:
                       - GLiNER is accurate at entity detection — no more
                         "Alice" classified as ORG. Zero-shot so no training.
                       - spaCy sm does dependency parsing between *known*
                         entities, which is reliable once the tokens are anchored.
                       - Total new download: ~67MB (vs ~1.5GB for REBEL).
                       - Swap model any time by changing _GLINER_MODEL.

Install: pip install gliner  (transformers + torch already present)
Model: urchade/gliner_small-v2.1 — cached to HuggingFace cache on first use.

Usage
-----
    bg = BackgroundGraph.for_path(path)
    bg.submit(text, chunk_id, filed_at)   # non-blocking
    bg.pending_count()

    mgraph = MemoryGraph.for_path(path)
    result  = mgraph.query_related("Alice", hops=1)   # instant, SQLite cache
"""

from __future__ import annotations

import logging
import queue
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("verimem_mcp")

_DB_NAME = "entity_graph.db"
_GLINER_MODEL = "urchade/gliner_small-v2.1"
_SPACY_MODEL = "en_core_web_sm"  # already installed; used for dep parse only
_QUEUE_TIMEOUT_S = 15
_BATCH_SIZE = 8  # GLiNER is fast — process up to N texts at a time
_MAX_TEXT_CHARS = 1000  # truncate very long chunks before NER

# Zero-shot entity types for GLiNER
_ENTITY_TYPES = [
    "person",
    "organization",
    "product",
    "technology",
    "location",
    "event",
    "concept",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EntityNode:
    canonical: str  # lowercased, normalised
    surface: str  # original surface form
    entity_type: str  # REBEL doesn't give types — populated by NER if available
    node_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {"canonical": self.canonical, "surface": self.surface, "type": self.entity_type}

    def to_simple(self) -> str:
        label = f" ({self.entity_type})" if self.entity_type else ""
        return f"{self.surface}{label}"


@dataclass
class RelationTriple:
    subject: str
    relation: str
    obj: str
    subject_type: str = ""
    obj_type: str = ""
    chunk_id: Optional[str] = None
    filed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "chunk_id": self.chunk_id,
            "filed_at": self.filed_at,
        }

    def to_simple(self) -> str:
        return f"{self.subject} → {self.relation} → {self.obj}"


@dataclass
class GraphRecallResult:
    entity: str
    entity_type: str
    triples: List[RelationTriple] = field(default_factory=list)
    related_entities: List[EntityNode] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    graph_pending: bool = False  # True if BackgroundGraph still has queued work

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "triples": [t.to_dict() for t in self.triples],
            "related_entities": [e.to_dict() for e in self.related_entities],
            "chunk_ids": self.chunk_ids,
            "graph_pending": self.graph_pending,
        }

    def to_simple(self) -> dict:
        """Compact, agent-readable view — no internal IDs."""
        return {
            "entity": self.entity,
            "relationships": [t.to_simple() for t in self.triples],
            "connected_to": [e.to_simple() for e in self.related_entities],
        }


# ---------------------------------------------------------------------------
# Extraction: GLiNER (entities) + spaCy sm (relations between known entities)
# ---------------------------------------------------------------------------


def _extract_triples(
    text: str,
    gliner_model,
    spacy_nlp,
) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples from text.

    Steps:
      1. GLiNER finds entity spans with accurate type labels.
      2. spaCy parses dependency tree (already installed, fast).
      3. For each sentence, pair up entities and find the ROOT verb
         that connects them as subject/object.

    Returns list of (subject_text, relation, object_text).
    """
    text = text[:_MAX_TEXT_CHARS]

    # Step 1: GLiNER entities — accurate zero-shot NER
    try:
        gliner_ents = gliner_model.predict_entities(text, _ENTITY_TYPES, threshold=0.5)
    except Exception:
        return []

    if len(gliner_ents) < 2:
        return []  # need at least two entities for a triple

    # Build a char-offset → entity label map
    ent_spans: List[dict] = [
        {"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"]}
        for e in gliner_ents
    ]

    # Step 2: spaCy dependency parse
    try:
        doc = spacy_nlp(text)
    except Exception:
        return []

    triples: List[Tuple[str, str, str]] = []

    for sent in doc.sents:
        s_start = sent.start_char
        s_end = sent.end_char

        # Entities within this sentence
        sent_ents = [e for e in ent_spans if e["start"] >= s_start and e["end"] <= s_end]
        if len(sent_ents) < 2:
            continue

        # Find the ROOT verb in this sentence
        root_verb = next(
            (tok for tok in sent if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX")),
            None,
        )
        if root_verb is None:
            # Fallback: use any verb
            root_verb = next((tok for tok in sent if tok.pos_ == "VERB"), None)
        if root_verb is None:
            # No verb — use "related_to" as a generic connector
            rel = "related to"
        else:
            rel = root_verb.lemma_.lower()

        # Pair up entities — subject = first in sentence, object = others
        for i in range(len(sent_ents)):
            for j in range(i + 1, len(sent_ents)):
                subj_e = sent_ents[i]
                obj_e = sent_ents[j]
                triples.append((subj_e["text"], rel, obj_e["text"]))

    return triples


def _canonical(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?\"'")
    return text


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical TEXT NOT NULL UNIQUE,
            surface   TEXT NOT NULL,
            etype     TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_canonical TEXT NOT NULL,
            chunk_id         TEXT NOT NULL,
            PRIMARY KEY (entity_canonical, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS triples (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            subj      TEXT NOT NULL,
            relation  TEXT NOT NULL,
            obj       TEXT NOT NULL,
            subj_type TEXT DEFAULT '',
            obj_type  TEXT DEFAULT '',
            chunk_id  TEXT,
            filed_at  TEXT,
            UNIQUE (subj, relation, obj, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS processed_chunks (
            chunk_id TEXT PRIMARY KEY
        );
    """)
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# MemoryGraph — storage + query
# ---------------------------------------------------------------------------


class MemoryGraph:
    """
    SQLite-backed entity graph. Populated by BackgroundGraph, queried here.
    """

    _instances: dict[str, "MemoryGraph"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def for_path(cls, path: str) -> "MemoryGraph":
        with cls._registry_lock:
            if path not in cls._instances:
                cls._instances[path] = cls(path)
            return cls._instances[path]

    def __init__(self, path: str) -> None:
        self._path = path
        self._db = Path(path) / _DB_NAME
        _init_db(self._db)

    def store_triples(
        self,
        triples: List[Tuple[str, str, str]],
        chunk_id: str,
        filed_at: Optional[str],
    ) -> None:
        """Write extracted triples (and their entity nodes) to SQLite."""
        try:
            con = sqlite3.connect(str(self._db))
            now = filed_at or datetime.now(timezone.utc).isoformat()

            for subj_raw, rel, obj_raw in triples:
                subj = _canonical(subj_raw)
                obj = _canonical(obj_raw)
                rel = rel.strip().lower()
                if not subj or not obj or not rel:
                    continue

                # Upsert entities
                for canonical, surface in ((subj, subj_raw), (obj, obj_raw)):
                    con.execute(
                        "INSERT OR IGNORE INTO entities (canonical, surface, etype) VALUES (?,?,'')",
                        (canonical, surface.strip()),
                    )
                    con.execute(
                        "INSERT OR IGNORE INTO entity_chunks (entity_canonical, chunk_id) VALUES (?,?)",
                        (canonical, chunk_id),
                    )

                con.execute(
                    """INSERT OR IGNORE INTO triples
                       (subj, relation, obj, chunk_id, filed_at)
                       VALUES (?,?,?,?,?)""",
                    (subj, rel, obj, chunk_id, now),
                )

            con.execute("INSERT OR IGNORE INTO processed_chunks (chunk_id) VALUES (?)", (chunk_id,))
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("MemoryGraph.store_triples error: %s", e)

    def is_processed(self, chunk_id: str) -> bool:
        try:
            con = sqlite3.connect(str(self._db))
            row = con.execute(
                "SELECT 1 FROM processed_chunks WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            con.close()
            return row is not None
        except Exception:
            return False

    def query_related(self, entity: str, hops: int = 1) -> GraphRecallResult:
        canonical = _canonical(entity)
        result = GraphRecallResult(entity=canonical, entity_type="")

        try:
            con = sqlite3.connect(str(self._db))

            row = con.execute(
                "SELECT etype, surface FROM entities WHERE canonical=?", (canonical,)
            ).fetchone()
            if row:
                result.entity_type = row[0]
                result.entity = row[1]

            result.chunk_ids = [
                r[0]
                for r in con.execute(
                    "SELECT chunk_id FROM entity_chunks WHERE entity_canonical=?", (canonical,)
                ).fetchall()
            ]

            seen = {canonical}
            frontier = {canonical}
            all_triples: List[RelationTriple] = []

            for _ in range(hops):
                next_frontier = set()
                for node in frontier:
                    rows = con.execute(
                        """SELECT subj, relation, obj, subj_type, obj_type, chunk_id, filed_at
                           FROM triples WHERE subj=? OR obj=?""",
                        (node, node),
                    ).fetchall()
                    for subj, rel, obj, st, ot, cid, fat in rows:
                        all_triples.append(
                            RelationTriple(
                                subject=subj,
                                relation=rel,
                                obj=obj,
                                subject_type=st or "",
                                obj_type=ot or "",
                                chunk_id=cid,
                                filed_at=fat,
                            )
                        )
                        for neighbour in (subj, obj):
                            if neighbour not in seen:
                                seen.add(neighbour)
                                next_frontier.add(neighbour)
                frontier = next_frontier

            result.triples = _dedupe_triples(all_triples)

            related = seen - {canonical}
            if related:
                placeholders = ",".join("?" * len(related))
                known = {
                    r[0]: EntityNode(canonical=r[0], surface=r[1], entity_type=r[2])
                    for r in con.execute(
                        f"SELECT canonical, surface, etype FROM entities WHERE canonical IN ({placeholders})",
                        list(related),
                    ).fetchall()
                }
                result.related_entities = [
                    known.get(c, EntityNode(canonical=c, surface=c, entity_type=""))
                    for c in related
                ]

            con.close()
        except Exception as e:
            logger.warning("MemoryGraph.query_related error: %s", e)

        return result

    def entities_for_chunks(self, chunk_ids: List[str]) -> List[EntityNode]:
        """Return all entities mentioned in the given chunk IDs."""
        if not chunk_ids:
            return []
        try:
            con = sqlite3.connect(str(self._db))
            placeholders = ",".join("?" * len(chunk_ids))
            rows = con.execute(
                f"""SELECT DISTINCT e.canonical, e.surface, e.etype
                    FROM entities e
                    JOIN entity_chunks ec ON e.canonical = ec.entity_canonical
                    WHERE ec.chunk_id IN ({placeholders})""",
                chunk_ids,
            ).fetchall()
            con.close()
            return [EntityNode(canonical=r[0], surface=r[1], entity_type=r[2]) for r in rows]
        except Exception:
            return []

    def find_entity(self, text: str) -> Optional[EntityNode]:
        canonical = _canonical(text)
        try:
            con = sqlite3.connect(str(self._db))
            row = con.execute(
                "SELECT canonical, surface, etype, id FROM entities WHERE canonical=?", (canonical,)
            ).fetchone()
            con.close()
            if row:
                return EntityNode(
                    canonical=row[0], surface=row[1], entity_type=row[2], node_id=row[3]
                )
        except Exception:
            return None
        return None

    def entity_count(self) -> int:
        try:
            con = sqlite3.connect(str(self._db))
            n = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            con.close()
            return n
        except Exception:
            return 0

    def triple_count(self) -> int:
        try:
            con = sqlite3.connect(str(self._db))
            n = con.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            con.close()
            return n
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# BackgroundGraph — REBEL extraction daemon
# ---------------------------------------------------------------------------


class BackgroundGraph:
    """
    Daemon thread that extracts entity-relation triples using REBEL
    (babelscape/rebel-large) and writes them to MemoryGraph's SQLite.

    submit() is non-blocking — remember() returns immediately.
    The REBEL model is lazy-loaded on first use (~1.5GB download on cold start).

    Usage
    -----
    bg = BackgroundGraph.for_path(path)
    bg.submit(text, chunk_id, filed_at)   # queue for background extraction
    bg.pending_count()                     # how many still queued
    """

    _instances: dict[str, "BackgroundGraph"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def for_path(cls, path: str) -> "BackgroundGraph":
        with cls._registry_lock:
            if path not in cls._instances:
                cls._instances[path] = cls(path)
            return cls._instances[path]

    def __init__(self, path: str) -> None:
        self._path = path
        self._graph = MemoryGraph.for_path(path)
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()

    def submit(self, text: str, chunk_id: str, filed_at: Optional[str] = None) -> None:
        """
        Queue text for background triple extraction.
        No-op if this chunk_id was already processed.
        """
        if self._graph.is_processed(chunk_id):
            return
        self._queue.put((text, chunk_id, filed_at))
        self._ensure_worker()

    def pending_count(self) -> int:
        return self._queue.qsize()

    def _ensure_worker(self) -> None:
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._worker,
                    name="verimem-graph",
                    daemon=True,
                )
                self._thread.start()
                logger.debug("BackgroundGraph worker started for %s", self._path)

    def _worker(self) -> None:
        """Lazy-load GLiNER + spaCy sm, drain queue, write triples to SQLite."""

        # Load GLiNER (zero-shot NER, ~67MB)
        try:
            from gliner import GLiNER
        except ImportError:
            logger.warning(
                "BackgroundGraph: gliner not installed — graph extraction disabled. "
                "Install: pip install gliner"
            )
            self._drain_queue()
            return

        try:
            logger.debug("BackgroundGraph: loading GLiNER model %r (~67MB)", _GLINER_MODEL)
            gliner_model = GLiNER.from_pretrained(_GLINER_MODEL)
            logger.debug("BackgroundGraph: GLiNER loaded")
        except Exception as e:
            logger.warning("BackgroundGraph: failed to load GLiNER: %s", e)
            self._drain_queue()
            return

        # Load spaCy sm for dependency parsing (already installed)
        try:
            import spacy

            spacy_nlp = spacy.load(_SPACY_MODEL)
        except Exception as e:
            logger.warning("BackgroundGraph: spaCy %r not available: %s", _SPACY_MODEL, e)
            spacy_nlp = None

        def _process_one(text: str, cid: str, fat: Optional[str]) -> None:
            try:
                if spacy_nlp is not None:
                    triples = _extract_triples(text, gliner_model, spacy_nlp)
                else:
                    # GLiNER-only fallback: store entities, no relations
                    raw = gliner_model.predict_entities(
                        text[:_MAX_TEXT_CHARS], _ENTITY_TYPES, threshold=0.5
                    )
                    # Pair consecutive entities as loosely related
                    triples = []
                    ents = [e["text"] for e in raw]
                    for i in range(len(ents) - 1):
                        triples.append((ents[i], "related to", ents[i + 1]))

                self._graph.store_triples(triples, cid, fat)
                logger.debug("BackgroundGraph: chunk %s → %d triples", cid, len(triples))
            except Exception as e:
                logger.debug("BackgroundGraph chunk error %s: %s", cid, e)
                self._graph.store_triples([], cid, fat)  # mark processed

        try:
            while True:
                try:
                    item = self._queue.get(timeout=_QUEUE_TIMEOUT_S)
                    _process_one(*item)
                except queue.Empty:
                    break  # idle — exit, restart on next submit()
        finally:
            logger.debug("BackgroundGraph worker finished for %s", self._path)

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedupe_triples(triples: List[RelationTriple]) -> List[RelationTriple]:
    seen = set()
    out = []
    for t in triples:
        key = (t.subject, t.relation, t.obj)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out
