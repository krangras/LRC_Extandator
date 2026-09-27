"""Small robust SQLite cache for expensive alignment results."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 1


class AlignmentCache:
    def __init__(self, path: str | os.PathLike[str] | None = None, max_bytes: int = 2 * 1024**3):
        root = Path(path) if path else Path(__file__).resolve().parent / ".cache" / "alignment_v7.sqlite3"
        root.parent.mkdir(parents=True, exist_ok=True)
        self.path = root
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_db(self) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS alignments (
                    cache_key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_alignments_access ON alignments(last_access)")

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT payload, schema_version FROM alignments WHERE cache_key=?", (key,)
            ).fetchone()
            if not row:
                return None
            if int(row[1]) != CACHE_SCHEMA_VERSION:
                con.execute("DELETE FROM alignments WHERE cache_key=?", (key,))
                return None
            con.execute("UPDATE alignments SET last_access=? WHERE cache_key=?", (time.time(), key))
        try:
            return json.loads(zlib.decompress(row[0]).decode("utf-8"))
        except Exception:
            with self._lock, self._connect() as con:
                con.execute("DELETE FROM alignments WHERE cache_key=?", (key,))
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        packed = zlib.compress(raw, 6)
        now = time.time()
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO alignments(cache_key,payload,byte_size,created_at,last_access,schema_version)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    byte_size=excluded.byte_size,
                    last_access=excluded.last_access,
                    schema_version=excluded.schema_version
                """,
                (key, packed, len(packed), now, now, CACHE_SCHEMA_VERSION),
            )
        self.prune()

    def prune(self) -> None:
        with self._lock, self._connect() as con:
            total = int(con.execute("SELECT COALESCE(SUM(byte_size),0) FROM alignments").fetchone()[0])
            if total <= self.max_bytes:
                return
            for key, size in con.execute(
                "SELECT cache_key,byte_size FROM alignments ORDER BY last_access ASC"
            ).fetchall():
                con.execute("DELETE FROM alignments WHERE cache_key=?", (key,))
                total -= int(size)
                if total <= int(self.max_bytes * 0.85):
                    break

    def clear(self) -> int:
        with self._lock, self._connect() as con:
            count = int(con.execute("SELECT COUNT(*) FROM alignments").fetchone()[0])
            con.execute("DELETE FROM alignments")
            return count

    def info(self) -> dict[str, Any]:
        with self._lock, self._connect() as con:
            count, size = con.execute(
                "SELECT COUNT(*),COALESCE(SUM(byte_size),0) FROM alignments"
            ).fetchone()
        return {"entries": int(count), "bytes": int(size), "path": str(self.path)}
