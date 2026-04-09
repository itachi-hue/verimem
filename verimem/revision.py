"""
revision.py — Monotonic store revision counter.

Increments whenever chunks are added or removed. Persisted in SQLite under the
store directory.

Stamping ``store_revision`` on each ``ContextPacket`` supports cache
invalidation and reasoning about what was visible at a given revision.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DB_NAME = "store_revision.db"
_LEGACY_DB_NAME = "palace_revision.db"


def _db_path(store_path: str) -> str:
    base = Path(store_path)
    new_p = base / _DB_NAME
    legacy_p = base / _LEGACY_DB_NAME
    if new_p.exists():
        return str(new_p)
    if legacy_p.exists():
        try:
            legacy_p.rename(new_p)
            return str(new_p)
        except OSError:
            return str(legacy_p)
    return str(new_p)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS revision (id INTEGER PRIMARY KEY, rev INTEGER NOT NULL DEFAULT 0)"
    )
    row = conn.execute("SELECT COUNT(*) FROM revision").fetchone()
    if row[0] == 0:
        conn.execute("INSERT INTO revision (id, rev) VALUES (1, 0)")
    conn.commit()


def get_revision(store_path: str) -> int:
    """Return the current revision without incrementing."""
    try:
        with sqlite3.connect(_db_path(store_path)) as conn:
            _ensure_table(conn)
            row = conn.execute("SELECT rev FROM revision WHERE id = 1").fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def bump_revision(store_path: str) -> int:
    """Increment revision and return the new value."""
    try:
        with sqlite3.connect(_db_path(store_path)) as conn:
            _ensure_table(conn)
            conn.execute("UPDATE revision SET rev = rev + 1 WHERE id = 1")
            conn.commit()
            row = conn.execute("SELECT rev FROM revision WHERE id = 1").fetchone()
            return row[0] if row else 0
    except Exception:
        return 0
