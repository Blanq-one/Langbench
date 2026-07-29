"""Persistent raw-response cache (tier 1 of the two-tier storage design).

Contains corpus text and full API responses; lives under data/cache/ which is
gitignored. Every LLM call in the harness goes through this cache, so re-runs
after a crash or across days cost zero API calls for already-answered items.

Key = sha256 over the canonical JSON of
    (provider, model_id, prompt_template_version, params, input_text)
exactly as the spec requires. Changing a prompt template bumps its version
string, which changes every key, which is the point.

SQLite is used synchronously; call sites in async code wrap access in
asyncio.to_thread. # DECISION: sync sqlite + to_thread over aiosqlite — one
fewer dependency, identical semantics at this call volume.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path("data/cache/raw.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_cache (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_utc TEXT NOT NULL DEFAULT (datetime('now')),
    response_json TEXT NOT NULL
);
"""


def cache_key(
    provider: str,
    model_id: str,
    prompt_version: str,
    params: dict[str, Any],
    input_text: str,
) -> str:
    canonical = json.dumps(
        {
            "provider": provider,
            "model_id": model_id,
            "prompt_version": prompt_version,
            "params": params,
            "input_text": input_text,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RawCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # timeout=30: writer threads (one per model loop via to_thread)
            # can collide on the write lock, e.g. during a WAL checkpoint or
            # right after a sleep/wake; the 5s default crashed a live pass
            # (2026-07-27, "database is locked"). # DECISION 41
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT response_json FROM raw_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        loaded = json.loads(row[0])
        assert isinstance(loaded, dict)
        return loaded

    def put(
        self,
        key: str,
        provider: str,
        model_id: str,
        prompt_version: str,
        response: dict[str, Any],
    ) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO raw_cache "
            "(key, provider, model_id, prompt_version, response_json) VALUES (?,?,?,?,?)",
            (key, provider, model_id, prompt_version, json.dumps(response, ensure_ascii=False)),
        )
        conn.commit()

    def count(self) -> int:
        row = self._conn().execute("SELECT COUNT(*) FROM raw_cache").fetchone()
        return int(row[0])
