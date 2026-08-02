#!/usr/bin/env python3
"""Unattended daily pass (Part 3): candidates, then coverage-gated judging.

Designed to run from Windows Task Scheduler via scripts/daily_pass.cmd.
Everything it does is what the manual daily resume did, mechanized:

1. JUDGE FIRST (DECISION 43), one batch per (model, lang), smallest pending
   first: a batch is eligible only when EVERY pending feedback item has its
   candidate primary call cached AND, if the primary is unparseable, its
   repair turn cached — the one-call rule: the judge phase must never
   trigger a live candidate generation. Smallest-first makes the first
   batch a cheap probe of the judge quota, fired at schedule time sharp.
   Judge runs BEFORE candidates because TPD-bound candidate passes grind
   all day without exiting, which starved the judge stage entirely
   (observed 2026-07-29: ~101 judgeable items + a full Gemini day unused);
   the judge's Gemini spend has zero contention with Groq candidates.
   Batches are computed from the cache as of NOW — gens produced by
   today's candidates run get judged tomorrow.
2. GEMINI STOP-RULE (narrowed, DECISION 45): trips ONLY on daily-quota
   exhaustion — a 429 body whose quotaId contains "PerDay" (e.g.
   GenerateRequestsPerDayPerProjectPerModel). Burst per-MINUTE 429s
   (quotaId ...PerMinute..., observed 2026-08-01) self-heal in seconds and
   are ordinary retryable backoff; stopping on them would serialize
   end-game judge batches to one per day. On a trip: stop all further
   judging, leave a GEMINI-429-STOP marker in the log for review.
   Candidates still run after a judge stop — Groq quota is independent.
3. GUARD, scoped to candidates only: if a run_eval.py process is already
   alive (e.g. yesterday's TPD-ground pass), skip the CANDIDATES stage —
   never run two candidate passes at once (they double-hammer the shared
   rate buckets). The judge stage is NOT guarded: it makes zero live
   candidate calls (one-call rule) and safely overlaps a live pass.
4. CANDIDATES: run scripts/run_eval.py (resume semantics built in; crashed
   or parked models leave PENDING items for tomorrow, DECISIONs 17/41).

Logs: logs/automation/YYYY-MM-DD_HHMMSS.log — one file PER INVOCATION,
never shared: concurrent instances are legal (a judge stage may overlap a
still-running guarded pass), and Windows append handles inherited by long
-lived child processes write at stale offsets, silently clobbering other
writers of the same file (observed 2026-07-29: a test fire's entire judge
stage vanished from the shared day log). Review globs the day's files.
Exit code 0 = ran to plan (including "nothing to do"); 1 = candidates
skipped by the guard (judge stage still ran); 2 = candidates subprocess
failed; 3 = judging stopped on the Gemini stop-rule (candidates still ran).
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import langbench.tasks.feedback as feedback_task  # noqa: E402
from langbench.cache import RawCache, cache_key  # noqa: E402
from langbench.config import ModelConfig, load_registry  # noqa: E402
from langbench.data.schema import PREPARED_DIR, load_manifest_ids, read_jsonl  # noqa: E402
from langbench.parsing import Failed, build_repair_request, parse  # noqa: E402
from langbench.providers.base import ChatRequest  # noqa: E402
from langbench.results import ResultsDB  # noqa: E402

LOG_DIR = REPO / "logs" / "automation"
# This script already runs inside the uv-managed venv (daily_pass.cmd enters
# it via `python -m uv run`), so children reuse the same interpreter.
RUN_EVAL = [sys.executable, "scripts/run_eval.py"]
# Daily-quota exhaustion ONLY (DECISION 45). The quotaId value follows
# "quotaId" after JSON punctuation that the log may render escaped
# (\"quotaId\": \"GenerateRequestsPerDay...\"); [^A-Za-z]+ spans it either
# way. PerMinute burst 429s must NOT match — they are retryable noise.
GEMINI_DAILY_QUOTA_RE = re.compile(r"quotaId[^A-Za-z]+\w*PerDay", re.IGNORECASE)


def log_line(log: Path, msg: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[daily_pass {stamp}] {msg}"
    print(line)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def another_pass_alive() -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine "
         "-match 'run_eval\\.py' }).Count"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return bool(out) and out != "0"


def run_logged(args: list[str], log: Path) -> int:
    """Run a subprocess appending combined output to the day log."""
    with log.open("a", encoding="utf-8") as f:
        proc = subprocess.run(
            args, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False
        )
    return proc.returncode


def feedback_batch_state(
    cache: RawCache, results: ResultsDB, model: ModelConfig, lang: str, version: str
) -> tuple[int, bool]:
    """(pending_count, fully_cached) for one (model, lang) feedback batch."""
    ids = load_manifest_ids(lang, "feedback")
    if ids is None:
        return 0, False
    by_id = {s.id: s for s in read_jsonl(PREPARED_DIR / f"{lang}.jsonl")}
    pending = 0
    fully_cached = True
    for i in ids:
        if results.has("feedback", lang, model.key, version, i):
            continue
        pending += 1
        req = feedback_task.build_request(
            by_id[i].source_text, version, lang, model.output_budget("feedback")
        )

        def key_for(r: ChatRequest) -> str:
            # Mirrors Runner._call's key construction exactly; drift here
            # would silently break the one-call gate.
            return cache_key(
                provider=model.provider, model_id=model.model_id,
                prompt_version=version,
                params={"temperature": r.temperature, "max_tokens": r.max_tokens,
                        "extra_body": model.extra_body},
                input_text=f"{r.system or ''}\n\x00\n{r.user}",
            )

        cached = cache.get(key_for(req))
        if cached is None:
            fully_cached = False
            continue
        parsed = parse(cached["text"], feedback_task.Output)
        if isinstance(parsed, Failed):
            repair = build_repair_request(req, cached["text"], parsed.error)
            if cache.get(key_for(repair)) is None:
                fully_cached = False
    return pending, fully_cached


def run_judge_stage(log: Path) -> bool:
    """Judge every fully-cached batch, smallest first. Returns False when the
    Gemini stop-rule fired (further judging halted for the day)."""
    reg = load_registry()
    cache = RawCache()
    results = ResultsDB()
    version = reg.eval.prompt_versions["feedback"]
    batches: list[tuple[int, str, str]] = []  # (pending, model_key, lang)
    for model in reg.enabled_candidate_models():
        for lang in reg.eval.languages:
            if not model.covers_lang(lang):
                continue
            pending, fully_cached = feedback_batch_state(
                cache, results, model, lang, version
            )
            if pending == 0:
                continue
            if not fully_cached:
                log_line(log, f"judge {model.key}/{lang}: {pending} pending but "
                              "not fully cached — locked (one-call rule)")
                continue
            batches.append((pending, model.key, lang))

    if not batches:
        log_line(log, "no judge batches unlocked")
        return True

    batches.sort()  # smallest first: the first batch doubles as the quota probe
    for pending, model_key, lang in batches:
        log_line(log, f"judge {model_key}/{lang} starting ({pending} pending)")
        mark = log.stat().st_size
        rc = run_logged(RUN_EVAL + ["--phase", "judge",
                                    "--models", model_key, "--langs", lang], log)
        with log.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(mark)
            batch_out = f.read()
        if GEMINI_DAILY_QUOTA_RE.search(batch_out):
            log_line(log, "GEMINI-429-STOP: DAILY quota 429 seen in judge "
                          "output; stopping all judging until reviewed "
                          "(stop-rule, PerDay quotaId only)")
            return False
        log_line(log, f"judge {model_key}/{lang} done rc={rc}")

    log_line(log, "all unlocked judge batches done")
    return True


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}.log"

    # Judge FIRST (DECISION 43): fires the Gemini probe at schedule time and
    # cannot be starved by a TPD-ground candidates run that never exits. Not
    # guarded — it overlaps a live candidates pass safely (one-call rule).
    judge_ok = run_judge_stage(log)

    if another_pass_alive():
        log_line(log, "GUARD: run_eval.py already alive; skipping candidates "
                      "stage this fire")
        return 1

    log_line(log, "candidates pass starting")
    rc = run_logged(RUN_EVAL, log)
    if rc != 0:
        log_line(log, f"candidates pass FAILED rc={rc} — items stay pending; "
                      "review the log")
        return 2
    log_line(log, "candidates pass done")
    return 0 if judge_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
