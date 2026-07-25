"""The CI smoke eval: 5 items per task, fully offline, fake adapters, real
runner/judge/report code paths, committed hand-made fixtures. This is the
one-shot build's proof of internal consistency."""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from langbench.cache import RawCache
from langbench.data.schema import read_jsonl, write_manifest
from langbench.ratelimit import RateLimiter
from langbench.report import build_all, emit_bot_config
from langbench.results import ResultsDB
from langbench.runner import Runner

from .conftest import BrokenAdapter, FakeAdapter, make_registry

N = 5


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path) -> Path:
    """A repo-shaped tmp dir: prepared data + manifests, cwd switched into it
    (schema.py paths are cwd-relative by design)."""
    monkeypatch.chdir(tmp_path)
    prepared = tmp_path / "data" / "prepared"
    prepared.mkdir(parents=True)
    shutil.copy(fixtures_dir / "en_samples.jsonl", prepared / "en.jsonl")
    samples = read_jsonl(prepared / "en.jsonl")
    assert len(samples) == N
    for task in ("gec", "cefr", "feedback"):
        write_manifest("en", task, samples, seed=42)
    return tmp_path


def build_runner(tmp_path: Path, client: httpx.AsyncClient) -> tuple[Runner, ResultsDB]:
    reg = make_registry(n_items=N)
    results = ResultsDB(tmp_path / "results.sqlite")
    runner = Runner(
        registry=reg,
        cache=RawCache(tmp_path / "raw.sqlite"),
        results=results,
        limiter=RateLimiter(),
        client=client,
    )
    runner._adapters["fakeprov"] = FakeAdapter()  # type: ignore[assignment]
    runner._adapters["fakejudge"] = FakeAdapter()  # type: ignore[assignment]
    return runner, results


async def test_smoke_eval_end_to_end(workspace: Path) -> None:
    async with httpx.AsyncClient() as client:
        runner, results = build_runner(workspace, client)
        reg = runner.reg
        # Swap the broken candidate in for real: good model + broken model.
        runner._adapters["fakeprov"] = _RoutingAdapter()  # type: ignore[assignment]
        models = reg.enabled_candidate_models()
        assert {m.key for m in models} == {"fakeprov/good", "fakeprov/broken"}

        plan = runner.load_work(["en"])
        assert plan.total_items() == 3 * N

        estimate = runner.estimate(plan, models)
        assert "fakeprov/good" in estimate and "calls pending" in estimate
        assert "Judge" in estimate

        await runner.run_candidates(plan, models)
        await runner.run_judge(plan, models)

        # Every (task, model, item) is accounted for; nothing silently dropped.
        for task in ("gec", "cefr", "feedback"):
            for m in models:
                recs = results.fetch(task=task, model_key=m.key)
                assert len(recs) == N, f"{task}/{m.key}: {len(recs)} != {N}"

        good_gec = results.fetch(task="gec", model_key="fakeprov/good")
        assert all(r["format_ok"] for r in good_gec)
        broken_gec = results.fetch(task="gec", model_key="fakeprov/broken")
        assert all(not r["format_ok"] for r in broken_gec)  # scored failures
        assert all(r["payload"]["gleu"] == 0.0 for r in broken_gec)

        broken_cefr = results.fetch(task="cefr", model_key="fakeprov/broken")
        assert all(r["payload"]["pred_label"] == "UNPARSEABLE" for r in broken_cefr)

        broken_fb = results.fetch(task="feedback", model_key="fakeprov/broken")
        assert all(r["payload"]["judge_correct_errors"] == 1 for r in broken_fb)

        # Resume: a second pass finds zero pending work and calls nothing new.
        adapter = _RoutingAdapter()
        runner._adapters["fakeprov"] = adapter  # type: ignore[assignment]
        await runner.run_candidates(plan, models)
        assert adapter.good.calls == 0

        # Report + leaderboard + bot config, all from the results DB alone.
        reports = build_all(results, reg, workspace / "REPORT.md",
                            workspace / "docs")
        assert (workspace / "REPORT.md").exists()
        assert (workspace / "docs" / "index.html").exists()
        assert (workspace / "docs" / "frontier.png").exists()

        by_key = {r.model_key: r for r in reports}
        assert by_key["fakeprov/good"].format_reliability == 1.0
        assert by_key["fakeprov/broken"].format_reliability == 0.0
        assert by_key["fakeprov/good"].composite > by_key["fakeprov/broken"].composite
        assert by_key["fakeprov/good"].qwk_band_en is not None
        assert by_key["fakeprov/good"].qwk_six is None  # fixture is band-only

        report_md = (workspace / "REPORT.md").read_text(encoding="utf-8")
        assert "UNCALIBRATED" in report_md  # honest clarity note, no hand labels

        winner = emit_bot_config(reports, reg, workspace / "bot-config.yaml")
        assert winner == "fakeprov/good"  # broken model fails the 95% gate
        cfg_text = (workspace / "bot-config.yaml").read_text(encoding="utf-8")
        assert "model_id: good" in cfg_text


async def test_daily_quota_parks_one_model_only(workspace: Path) -> None:
    async with httpx.AsyncClient() as client:
        runner, results = build_runner(workspace, client)
        reg = runner.reg
        models = reg.enabled_candidate_models()
        # Choke ONE model's daily quota; the other must still finish.
        runner.limiter.register("fakeprov/broken", reg.model("fakeprov/broken").rate_limit)
        runner.limiter.register("fakeprov/good", reg.model("fakeprov/good").rate_limit)
        runner.limiter.preload_daily_usage(
            "fakeprov/broken", reg.model("fakeprov/broken").rate_limit.rpd
        )
        plan = runner.load_work(["en"])
        await runner.run_candidates(plan, models)
        assert len(results.fetch(task="gec", model_key="fakeprov/good")) == N
        assert len(results.fetch(task="gec", model_key="fakeprov/broken")) == 0


class _RoutingAdapter:
    """Routes to FakeAdapter for model 'good' and BrokenAdapter for 'broken'
    so one provider entry serves both candidates, like real providers do."""

    def __init__(self) -> None:
        self.good = FakeAdapter()
        self.broken = BrokenAdapter()

    async def chat(self, model, req):  # type: ignore[no-untyped-def]
        if model.model_id == "broken":
            return await self.broken.chat(model, req)
        return await self.good.chat(model, req)
