"""Async orchestration: the cache -> limiter -> adapter call path, the
pre-run quota budget estimator, resume semantics, and scoring glue.

Execution model:
- Models run concurrently with each other; items run sequentially WITHIN a
  model. At 15-30 RPM per model, intra-model parallelism buys nothing and
  complicates limiter reasoning. # DECISION
- Resume is the default: items already in the results DB are skipped, the
  limiter's daily usage is seeded conservatively from the results DB, and
  every live response lands in the raw cache so nothing is ever paid for
  twice.
- DailyQuotaExhausted parks ONE model for the day; the others keep going.
- Provider errors after retries leave the item PENDING for the next resume
  (transient infra is not the model's fault). Persistent PARSE failures are
  scored failures with format_ok=0 (they are the model's fault). # DECISION

Phases:
- candidates: GEC + CEFR + feedback-generation calls on candidate models
- judge: Gemini judge calls over candidate feedback outputs (feedback outputs
  are re-obtained via the raw cache; if missing they are produced on the fly)
- calibration: ~30 English items -> judge.calibrate()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx

from langbench.cache import RawCache, cache_key
from langbench.config import ModelConfig, Registry
from langbench.data.schema import (
    BAND_OF,
    PREPARED_DIR,
    Sample,
    load_manifest_ids,
    read_jsonl,
)
from langbench.judge import (
    CalibrationReport,
    JudgeScores,
    build_judge_request,
    calibrate,
)
from langbench.metrics.gleu import (
    REWRITE_THRESHOLD,
    edit_rate,
    sentence_gleu,
    tokenize,
)
from langbench.parsing import Failed, Ok, build_repair_request, parse
from langbench.providers import (
    ChatRequest,
    ChatResponse,
    ProviderAdapter,
    ProviderError,
    build_adapter,
)
from langbench.ratelimit import DailyQuotaExhausted, RateLimiter
from langbench.results import ResultsDB
from langbench.tasks import cefr as cefr_task
from langbench.tasks import feedback as feedback_task
from langbench.tasks import gec as gec_task

log = logging.getLogger("langbench.runner")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("langbench")
    root.handlers = [handler]
    root.setLevel(level)


@dataclass
class TaskWork:
    task: str  # 'gec' | 'cefr' | 'feedback'
    lang: str
    samples: list[Sample]


@dataclass
class WorkPlan:
    items: list[TaskWork] = field(default_factory=list)

    def total_items(self) -> int:
        return sum(len(t.samples) for t in self.items)


class Runner:
    def __init__(
        self,
        registry: Registry,
        cache: RawCache,
        results: ResultsDB,
        limiter: RateLimiter,
        client: httpx.AsyncClient,
    ) -> None:
        self.reg = registry
        self.cache = cache
        self.results = results
        self.limiter = limiter
        self.client = client
        self._adapters: dict[str, ProviderAdapter] = {}

    # ------------------------------------------------------------ setup

    def _adapter_for(self, model: ModelConfig) -> ProviderAdapter:
        prov = self.reg.providers[model.provider]
        if prov.name not in self._adapters:
            self._adapters[prov.name] = build_adapter(prov, self.client)
        return self._adapters[prov.name]

    def _register_limits(self, models: list[ModelConfig]) -> None:
        for m in models:
            self.limiter.register(m.key, m.rate_limit)
            self.limiter.preload_daily_usage(m.key, self.results.count_today(m.key))

    # ------------------------------------------------------------ work plan

    def load_work(self, langs: list[str] | None = None) -> WorkPlan:
        plan = WorkPlan()
        for lang in langs or self.reg.eval.languages:
            try:
                samples = read_jsonl(PREPARED_DIR / f"{lang}.jsonl")
            except FileNotFoundError as e:
                log.warning("skipping %s: %s", lang, e)
                continue
            by_id = {s.id: s for s in samples}
            for task in ("gec", "cefr", "feedback"):
                ids = load_manifest_ids(lang, task)
                if ids is None:
                    log.info("no manifest for %s/%s (run prepare_data.py --manifests)",
                             lang, task)
                    continue
                missing = [i for i in ids if i not in by_id]
                if missing:
                    raise ValueError(
                        f"manifest {lang}/{task} references {len(missing)} ids not in "
                        f"{lang}.jsonl (first: {missing[0]!r}); re-run prepare_data.py "
                        "or rebuild manifests — mismatched manifests would silently "
                        "change the eval set"
                    )
                plan.items.append(TaskWork(task, lang, [by_id[i] for i in ids]))
        return plan

    # ------------------------------------------------------------ estimator

    def estimate(self, plan: WorkPlan, models: list[ModelConfig]) -> str:
        """Pre-run quota budget: candidate calls per (provider, model) vs daily
        limits, and judge calls vs the Gemini limit, SEPARATELY."""
        judge = self.reg.judge_model()
        lines = ["QUOTA BUDGET ESTIMATE (pending items; raw-cache hits cost 0 calls)",
                 "", "Candidate models:"]
        for m in models:
            pending = 0
            for tw in plan.items:
                v = self.reg.eval.prompt_versions[tw.task]
                pending += sum(
                    not self.results.has(tw.task, tw.lang, m.key, v, s.id)
                    for s in tw.samples
                )
            days = pending / m.rate_limit.rpd if m.rate_limit.rpd else float("inf")
            lines.append(
                f"  {m.key}: {pending} calls pending vs {m.rate_limit.rpd}/day "
                f"=> ~{max(days, 0):.1f} calendar day(s) (repair retries add up to 2x worst case)"
            )
        v = self.reg.eval.prompt_versions["feedback"]
        judge_pending = 0
        for tw in plan.items:
            if tw.task != "feedback":
                continue
            for m in models:
                judge_pending += sum(
                    not self.results.has("feedback", tw.lang, m.key, v, s.id)
                    for s in tw.samples
                )
        jdays = judge_pending / judge.rate_limit.rpd if judge.rate_limit.rpd else float("inf")
        lines += [
            "",
            "Judge (separate budget — this is usually the long pole):",
            f"  {judge.key}: {judge_pending} judge calls pending vs "
            f"{judge.rate_limit.rpd}/day => ~{jdays:.1f} calendar day(s)",
            "",
            "Run with --resume (default) across days; completed items are never re-called.",
            "Caveat: only RPM/RPD are modeled. Provider TPM (tokens/minute) caps are not;",
            "on Groq the 8B model's 6K TPM can bind before its RPM does, so expect some",
            "429 backoffs the day-count above does not predict.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------ call path

    async def _call(
        self, model: ModelConfig, req: ChatRequest, prompt_version: str
    ) -> ChatResponse:
        key = cache_key(
            provider=model.provider,
            model_id=model.model_id,
            prompt_version=prompt_version,
            # extra_body is part of the key on purpose: it changes response
            # semantics (e.g. reasoning_format), so a cached pre-extra_body
            # response must never satisfy a post-extra_body request.
            params={
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
                "extra_body": model.extra_body,
            },
            input_text=f"{req.system or ''}\n\x00\n{req.user}",
        )
        cached = await asyncio.to_thread(self.cache.get, key)
        if cached is not None:
            # Latency on cache hits is the original live measurement. # DECISION
            return ChatResponse.from_cacheable(cached)
        await self.limiter.acquire(model.key)
        resp = await self._adapter_for(model).chat(model, req)
        await asyncio.to_thread(
            self.cache.put, key, model.provider, model.model_id, prompt_version,
            resp.to_cacheable(),
        )
        return resp

    async def _call_parsed(
        self,
        model: ModelConfig,
        req: ChatRequest,
        prompt_version: str,
        schema: type[Any],
    ) -> tuple[Any | None, ChatResponse]:
        """One call + at most one repair turn. Returns (parsed_or_None, last_resp)."""
        resp = await self._call(model, req, prompt_version)
        result = parse(resp.text, schema)
        if isinstance(result, Ok):
            return result.value, resp
        assert isinstance(result, Failed)
        repair = build_repair_request(req, resp.text, result.error)
        resp2 = await self._call(model, repair, prompt_version)
        result2 = parse(resp2.text, schema)
        if isinstance(result2, Ok):
            return result2.value, resp2
        return None, resp2

    # ------------------------------------------------------------ candidates

    async def run_candidates(self, plan: WorkPlan, models: list[ModelConfig]) -> None:
        self._register_limits(models)
        await asyncio.gather(*(self._model_loop(m, plan) for m in models))

    async def _model_loop(self, model: ModelConfig, plan: WorkPlan) -> None:
        for tw in plan.items:
            version = self.reg.eval.prompt_versions[tw.task]
            for sample in tw.samples:
                if self.results.has(tw.task, tw.lang, model.key, version, sample.id):
                    continue
                try:
                    await self._run_item(model, tw, version, sample)
                except DailyQuotaExhausted as e:
                    log.warning("%s parked for the day: %s", model.key, e)
                    return
                except ProviderError as e:
                    log.error("%s item %s left pending (provider error): %s",
                              model.key, sample.id, e)
                    continue
        log.info("%s: candidate work complete", model.key)

    async def _run_item(
        self, model: ModelConfig, tw: TaskWork, version: str, sample: Sample
    ) -> None:
        if tw.task == "gec":
            req = gec_task.build_request(
                sample.source_text, version, tw.lang, model.max_output_tokens
            )
            parsed, resp = await self._call_parsed(model, req, version, gec_task.Output)
            if parsed is None:
                payload: dict[str, Any] = {
                    "gleu": 0.0, "edit_rate": 0.0, "rewrote_everything": False
                }
                ok = False
            else:
                er = edit_rate(sample.source_text, parsed.corrected)
                payload = {
                    "gleu": sentence_gleu(
                        sample.source_text, parsed.corrected, sample.reference_corrections
                    ),
                    "edit_rate": er,
                    "rewrote_everything": er > REWRITE_THRESHOLD,
                }
                ok = True
        elif tw.task == "cefr":
            req = cefr_task.build_request(
                sample.source_text, version, tw.lang, model.max_output_tokens
            )
            parsed, resp = await self._call_parsed(model, req, version, cefr_task.Output)
            gran = sample.cefr_granularity or "six_level"
            gold = sample.cefr_label or ""
            if parsed is None:
                payload = {
                    "pred_label": "UNPARSEABLE", "gold_label": gold,
                    "gold_granularity": gran, "correct": False, "adjacent": False,
                }
                ok = False
            else:
                pred, gold_cmp = parsed.level, gold
                if gran == "band":
                    pred, gold_cmp = BAND_OF[pred], BAND_OF[gold]
                scale = ["A", "B", "C"] if gran == "band" else \
                        ["A1", "A2", "B1", "B2", "C1", "C2"]
                dist = abs(scale.index(pred) - scale.index(gold_cmp))
                payload = {
                    "pred_label": parsed.level, "gold_label": gold,
                    "gold_granularity": gran, "correct": dist == 0, "adjacent": dist <= 1,
                }
                ok = True
        elif tw.task == "feedback":
            # Generation only; judging happens in run_judge. Record nothing in
            # the results DB here — the raw cache holds the output.
            req = feedback_task.build_request(
                sample.source_text, version, tw.lang, model.max_output_tokens
            )
            await self._call_parsed(model, req, version, feedback_task.Output)
            return
        else:
            raise ValueError(f"unknown task {tw.task!r}")

        self.results.upsert(
            task=tw.task, lang=tw.lang, model_key=model.key, prompt_version=version,
            sample_id=sample.id, format_ok=ok, payload=payload,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )

    # ------------------------------------------------------------ judge

    async def run_judge(self, plan: WorkPlan, models: list[ModelConfig]) -> None:
        judge_model = self.reg.judge_model()
        self._register_limits(models + [judge_model])
        version = self.reg.eval.prompt_versions["feedback"]
        for tw in plan.items:
            if tw.task != "feedback":
                continue
            for model in models:
                for sample in tw.samples:
                    if self.results.has("feedback", tw.lang, model.key, version, sample.id):
                        continue
                    try:
                        await self._judge_item(judge_model, model, tw, version, sample)
                    except DailyQuotaExhausted as e:
                        log.warning("judge quota exhausted, stopping for today: %s", e)
                        return
                    except ProviderError as e:
                        log.error("judge item %s/%s left pending: %s",
                                  model.key, sample.id, e)
                        continue
        log.info("judge phase complete")

    async def _judge_item(
        self,
        judge_model: ModelConfig,
        model: ModelConfig,
        tw: TaskWork,
        version: str,
        sample: Sample,
    ) -> None:
        # Re-obtain the candidate's feedback (cache hit if candidates phase ran).
        cand_req = feedback_task.build_request(
            sample.source_text, version, tw.lang, model.max_output_tokens
        )
        cand_parsed, cand_resp = await self._call_parsed(
            model, cand_req, version, feedback_task.Output
        )
        if cand_parsed is None:
            # Candidate couldn't produce valid feedback: scored failure with
            # floor rubric scores. # DECISION
            self.results.upsert(
                task="feedback", lang=tw.lang, model_key=model.key,
                prompt_version=version, sample_id=sample.id, format_ok=False,
                payload={
                    "judge_correct_errors": 1, "judge_correction_accuracy": 1,
                    "judge_explanation_clarity": 1, "judge_no_hallucinated": 1,
                    "n_errors_reported": 0,
                },
                prompt_tokens=cand_resp.prompt_tokens,
                completion_tokens=cand_resp.completion_tokens,
                latency_ms=cand_resp.latency_ms,
            )
            return
        jreq = build_judge_request(
            sample_id=sample.id,
            source=sample.source_text,
            gold_references=sample.reference_corrections,
            feedback_json=cand_parsed.model_dump_json(),
        )
        jparsed, _jresp = await self._call_parsed(
            judge_model, jreq, "judge-v1", JudgeScores
        )
        if jparsed is None:
            raise ProviderError(
                f"judge produced unparseable scores twice for {sample.id}; left pending"
            )
        self.results.upsert(
            task="feedback", lang=tw.lang, model_key=model.key, prompt_version=version,
            sample_id=sample.id, format_ok=True,
            payload={
                "judge_correct_errors": jparsed.correct_errors,
                "judge_correction_accuracy": jparsed.correction_accuracy,
                "judge_explanation_clarity": jparsed.explanation_clarity,
                "judge_no_hallucinated": jparsed.no_hallucinated,
                "n_errors_reported": len(cand_parsed.errors),
            },
            prompt_tokens=cand_resp.prompt_tokens,
            completion_tokens=cand_resp.completion_tokens,
            latency_ms=cand_resp.latency_ms,
        )

    # ------------------------------------------------------------ calibration

    async def run_calibration(self, models: list[ModelConfig]) -> CalibrationReport:
        """~30 English judge items across the FIRST enabled candidate model.
        Calibration grounds the judge itself, so one candidate's outputs
        suffice; using more would mix candidate quality into judge agreement.
        # DECISION"""
        judge_model = self.reg.judge_model()
        cfg = self.reg.eval.judge
        if not models:
            raise ValueError("no enabled candidate models to calibrate against")
        model = models[0]
        self._register_limits([model, judge_model])
        version = self.reg.eval.prompt_versions["feedback"]
        lang = cfg.calibration_language
        plan = self.load_work([lang])
        feedback_items = [tw for tw in plan.items if tw.task == "feedback"]
        if not feedback_items:
            raise ValueError(f"no feedback manifest for {lang}; run prepare_data.py")
        samples = feedback_items[0].samples[: cfg.calibration_items]

        items: list[dict[str, Any]] = []
        for sample in samples:
            cand_req = feedback_task.build_request(
                sample.source_text, version, lang, model.max_output_tokens
            )
            cand_parsed, _ = await self._call_parsed(
                model, cand_req, version, feedback_task.Output
            )
            if cand_parsed is None:
                continue
            jreq = build_judge_request(
                sample_id=sample.id,
                source=sample.source_text,
                gold_references=sample.reference_corrections,
                feedback_json=cand_parsed.model_dump_json(),
            )
            jparsed, _ = await self._call_parsed(judge_model, jreq, "judge-v1", JudgeScores)
            if jparsed is None:
                continue
            # Automatic spurious-edit proxy: corrected text contains tokens
            # absent from BOTH the source and every gold reference. # DECISION
            cand_tokens = set(tokenize(cand_parsed.corrected))
            src_tokens = set(tokenize(sample.source_text))
            gold_tokens: set[str] = set()
            for g in sample.reference_corrections:
                gold_tokens |= set(tokenize(g))
            spurious = bool(cand_tokens - src_tokens - gold_tokens)
            items.append(
                {
                    "sample_id": sample.id,
                    "source": sample.source_text,
                    "golds": sample.reference_corrections,
                    "corrected": cand_parsed.corrected,
                    "judge": jparsed,
                    "spurious_edits": spurious,
                }
            )
        return calibrate(items)
