"""Results DB: the payload allowlist is the mechanism that keeps corpus text
out of the COMMITTED database, so its rejections matter as much as its
happy path."""

from __future__ import annotations

from pathlib import Path

import pytest

from langbench.results import PayloadValidationError, ResultsDB, validate_payload


@pytest.fixture
def db(tmp_path: Path) -> ResultsDB:
    return ResultsDB(tmp_path / "results.sqlite")


GEC_PAYLOAD = {"gleu": 0.5, "edit_rate": 0.1, "rewrote_everything": False}


class TestAllowlist:
    def test_valid_payloads_pass(self) -> None:
        validate_payload("gec", GEC_PAYLOAD)
        validate_payload("cefr", {
            "pred_label": "B1", "gold_label": "B2",
            "gold_granularity": "six_level", "correct": False, "adjacent": True,
        })
        validate_payload("feedback", {
            "judge_correct_errors": 4, "judge_correction_accuracy": 3,
            "judge_explanation_clarity": 5, "judge_no_hallucinated": 4,
            "n_errors_reported": 2,
        })

    def test_free_text_field_rejected(self) -> None:
        with pytest.raises(PayloadValidationError, match="not allowlisted"):
            validate_payload("gec", {"corrected_text": "the learner wrote this"})

    def test_open_string_value_rejected(self) -> None:
        with pytest.raises(PayloadValidationError, match="closed set"):
            validate_payload("cefr", {"pred_label": "the text seems intermediate"})

    def test_gold_label_cannot_be_unparseable(self) -> None:
        with pytest.raises(PayloadValidationError, match="closed set"):
            validate_payload("cefr", {"gold_label": "UNPARSEABLE"})

    def test_unknown_task_rejected(self) -> None:
        with pytest.raises(PayloadValidationError, match="unknown task"):
            validate_payload("vibes", {})


class TestStorage:
    def _put(self, db: ResultsDB, sample_id: str = "s1", model: str = "p/m") -> None:
        db.upsert(
            task="gec", lang="en", model_key=model, prompt_version="v1",
            sample_id=sample_id, format_ok=True, payload=GEC_PAYLOAD,
            prompt_tokens=10, completion_tokens=5, latency_ms=100.0,
        )

    def test_upsert_has_fetch(self, db: ResultsDB) -> None:
        assert not db.has("gec", "en", "p/m", "v1", "s1")
        self._put(db)
        assert db.has("gec", "en", "p/m", "v1", "s1")
        recs = db.fetch(task="gec", model_key="p/m")
        assert len(recs) == 1
        assert recs[0]["payload"]["gleu"] == 0.5
        assert recs[0]["format_ok"] == 1

    def test_upsert_replaces_not_duplicates(self, db: ResultsDB) -> None:
        self._put(db)
        self._put(db)
        assert len(db.fetch()) == 1

    def test_reject_at_write_time_too(self, db: ResultsDB) -> None:
        with pytest.raises(PayloadValidationError):
            db.upsert(
                task="gec", lang="en", model_key="p/m", prompt_version="v1",
                sample_id="s9", format_ok=True,
                payload={"source_text": "learner corpus text"},
            )

    def test_count_today(self, db: ResultsDB) -> None:
        assert db.count_today("p/m") == 0
        self._put(db, "s1")
        self._put(db, "s2")
        assert db.count_today("p/m") == 2
        assert db.count_today("other/m") == 0
