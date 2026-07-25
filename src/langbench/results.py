"""Results DB (tier 2 of the two-tier storage design).

COMMITTED to the repo. Holds only derived, non-textual data: per-item metric
values, predicted CEFR labels, token counts, latencies, format-failure flags,
judge rubric scores, plus (task, lang, model, prompt version, sample id).
Never source text, never reference corrections, never model-corrected text —
those live only in the gitignored raw cache. This is what makes the repo
reproducible without redistributing corpus data.

The payload_json column is validated against an allowlist of scalar fields
per record kind so corpus text cannot leak in by accident.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_PATH = Path("data/results/results.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    task TEXT NOT NULL,               -- 'gec' | 'cefr' | 'feedback'
    lang TEXT NOT NULL,
    model_key TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    format_ok INTEGER NOT NULL,       -- 0 = persistent parse failure (scored failure)
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms REAL,
    created_utc TEXT NOT NULL DEFAULT (datetime('now')),
    payload_json TEXT NOT NULL,       -- allowlisted scalar fields only
    PRIMARY KEY (task, lang, model_key, prompt_version, sample_id)
);
CREATE INDEX IF NOT EXISTS idx_records_model ON records (model_key, task);
"""

# Allowlist per task: field name -> allowed python types. Anything else is
# rejected loudly. Strings are allowed only where the value set is closed
# (labels, flags) — never free text.
_ALLOWED_FIELDS: dict[str, dict[str, tuple[type, ...]]] = {
    "gec": {
        "gleu": (float, int),
        "edit_rate": (float, int),        # fraction of source tokens changed
        "rewrote_everything": (bool,),    # sanity flag: near-total rewrite
    },
    "cefr": {
        "pred_label": (str,),             # one of A1..C2 or A/B/C band
        "gold_label": (str,),             # closed-set gold; needed for QWK in report
        "gold_granularity": (str,),       # 'six_level' | 'band'
        "correct": (bool,),
        "adjacent": (bool,),
    },
    "feedback": {
        "judge_correct_errors": (float, int),
        "judge_correction_accuracy": (float, int),
        "judge_explanation_clarity": (float, int),
        "judge_no_hallucinated": (float, int),
        "n_errors_reported": (int,),
    },
}

_CLOSED_STRING_VALUES = {
    "pred_label": {"A1", "A2", "B1", "B2", "C1", "C2", "A", "B", "C", "UNPARSEABLE"},
    "gold_label": {"A1", "A2", "B1", "B2", "C1", "C2", "A", "B", "C"},
    "gold_granularity": {"six_level", "band"},
}


class PayloadValidationError(ValueError):
    pass


def validate_payload(task: str, payload: dict[str, Any]) -> None:
    if task not in _ALLOWED_FIELDS:
        raise PayloadValidationError(f"unknown task {task!r}")
    allowed = _ALLOWED_FIELDS[task]
    for field, value in payload.items():
        if field not in allowed:
            raise PayloadValidationError(
                f"field {field!r} not allowlisted for task {task!r}; "
                "the results DB must never carry corpus or free text"
            )
        if not isinstance(value, allowed[field]) or isinstance(value, bool) != (
            bool in allowed[field]
        ) and isinstance(value, bool):
            raise PayloadValidationError(
                f"field {field!r} has type {type(value).__name__}, "
                f"expected one of {[t.__name__ for t in allowed[field]]}"
            )
        if field in _CLOSED_STRING_VALUES and value not in _CLOSED_STRING_VALUES[field]:
            raise PayloadValidationError(
                f"field {field!r} value {value!r} outside closed set "
                f"{sorted(_CLOSED_STRING_VALUES[field])}"
            )


class ResultsDB:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_RESULTS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def upsert(
        self,
        *,
        task: str,
        lang: str,
        model_key: str,
        prompt_version: str,
        sample_id: str,
        format_ok: bool,
        payload: dict[str, Any],
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        validate_payload(task, payload)
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO records "
            "(task, lang, model_key, prompt_version, sample_id, format_ok, "
            " prompt_tokens, completion_tokens, latency_ms, payload_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task,
                lang,
                model_key,
                prompt_version,
                sample_id,
                int(format_ok),
                prompt_tokens,
                completion_tokens,
                latency_ms,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()

    def has(
        self, task: str, lang: str, model_key: str, prompt_version: str, sample_id: str
    ) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM records WHERE task=? AND lang=? AND model_key=? "
            "AND prompt_version=? AND sample_id=?",
            (task, lang, model_key, prompt_version, sample_id),
        ).fetchone()
        return row is not None

    def fetch(
        self,
        task: str | None = None,
        lang: str | None = None,
        model_key: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, args = [], []
        for col, val in (("task", task), ("lang", lang), ("model_key", model_key)):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn().execute(f"SELECT * FROM records {where}", args).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    def count_today(self, model_key: str) -> int:
        """Requests recorded for this model in the current UTC day.

        A conservative floor for --resume daily-quota seeding: cached hits
        also produce records, so this can overcount actual API spend, which
        only makes the limiter more cautious, never less.
        """
        row = self._conn().execute(
            "SELECT COUNT(*) FROM records WHERE model_key=? "
            "AND date(created_utc) = date('now')",
            (model_key,),
        ).fetchone()
        return int(row[0])
