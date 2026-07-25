# Langbench

A reproducible harness that benchmarks free-tier LLMs on multilingual
grammatical error correction (GEC) and CEFR level classification, producing a
quality-per-dollar leaderboard with confidence intervals — plus PolyglotBot,
a Matrix bot that deploys the benchmark winner as a live language-feedback
bot.

Built as a follow-up to a Pangea Chat intern task (an LLM-powered FastAPI
grammar-feedback service). The question this repo answers: which free or
cheap model gives Pangea-style feedback quality closest to the expensive
default, at what cost per 1,000 learner messages — and can it run on Matrix
today?

Results live in [REPORT.md](REPORT.md) and the
[leaderboard](docs/index.html) once evaluation runs complete. Both are built
from the committed results database, so anyone can regenerate them without
API keys or corpus downloads.

## Quick start (offline, no keys)

```bash
uv sync
ruff check . && mypy . && pytest
```

The entire test suite runs offline, including a 5-item end-to-end smoke eval
against committed hand-made fixtures (invented sentences, canned responses —
never real corpus text).

## Running the evaluation

```bash
cp .env.example .env            # add at least GROQ_API_KEY and GEMINI_API_KEY
uv run python scripts/discover_models.py         # reconcile model ids
uv run python scripts/prepare_data.py --dataset wi-locness   # prints download steps
uv run python scripts/prepare_data.py --dataset merlin --lang de   # ... it, cs
uv run python scripts/prepare_data.py --dataset cowsl2h
uv run python scripts/prepare_data.py --manifests
uv run python scripts/run_eval.py --dry-run      # quota budget estimate
uv run python scripts/run_eval.py                # candidates phase (daily)
uv run python scripts/run_eval.py --phase judge  # judge phase (daily)
uv run python scripts/run_eval.py --calibrate-judge
uv run python scripts/build_report.py --emit-bot-config
```

The full eval does not fit in one day of free quotas and is not supposed to:
the runner prints a per-(provider, model) budget estimate before any call,
every response is cached on disk, and re-running is free for completed items.
Judge calls (50 items x 5 languages x N models) are budgeted separately —
they are usually the long pole.

## What is measured

**Task 1 — GEC.** Minimal correction of learner text. Primary metric: GLEU
(cross-lingual). ERRANT P/R/F0.5 additionally for English (optional extra:
`uv sync --extra errant`). An edit-rate sanity check flags models that
rewrite instead of correcting.

**Task 2 — CEFR classification.** A1-C2 prediction. Metrics: exact accuracy,
adjacent (+/-1) accuracy, macro-F1, and quadratic weighted kappa as the
headline. Label granularity is respected, not fudged: MERLIN (de/it/cs) has
six levels; English (W&I+LOCNESS) has A/B/C bands and is scored on bands;
Spanish (COWS-L2H) has course levels, which are not CEFR labels, so Spanish
is GEC-only. QWK is never pooled across granularities.

**Task 3 — Feedback quality (judged subsample).** On 50 stratified items per
language, models produce Pangea-style structured feedback (error spans,
category, correction, one-sentence explanation). A Gemini judge — a different
provider family from every candidate, enforced by config validation — scores
four rubric dimensions at temperature 0 with per-item randomized dimension
order. Calibration is scoped honestly: gold corrections ground the
correctness dimensions only; explanation clarity is reported as uncalibrated
unless hand labels are supplied (`data/calibration/clarity_labels.jsonl`).

**Statistics.** Bootstrap 95% CIs on every headline number; paired bootstrap
for model-vs-model deltas; overlapping CIs are stated plainly as "not
settled". No single-run point estimate is presented as a ranking.

**Cost.** Measured token usage x public list prices, reported as $ per 1,000
learner messages. Free-tier runs cost $0; list prices are used so the numbers
survive a move to paid tiers. GPT-4o-mini (Pangea's known production model)
is a disabled-by-default reference entry; its list price anchors the cost
axis either way.

## Design notes

**Rate limits are per (provider, model).** Groq gives its 70B models a far
smaller daily quota than its 8B models; a provider-level limiter would be
wrong. Token buckets enforce RPM and RPD per model; daily exhaustion parks
one model and the rest keep running; `--resume` (the default, and the only
mode) picks up where quotas stopped.

**Two-tier storage.** (1) A raw response cache (SQLite, gitignored) holds
full API responses — it contains corpus text and is never committed. (2) The
results DB (SQLite, committed) holds only derived scalars: scores, labels,
token counts, latencies, flags. A per-task field allowlist rejects free text
at write time, so corpus text physically cannot enter the committed artifact.
This is why the repo is reproducible without redistributing any corpus (see
[DATA_LICENSES.md](DATA_LICENSES.md)).

**Structured output everywhere.** Tasks demand strict JSON, parsed with
pydantic; one bounded repair attempt; a second failure is a scored failure
that counts against the model in the format-reliability column — never
silently dropped.

**Prompts are versioned artifacts.** The version string participates in the
cache key and the results DB primary key; editing a template requires bumping
it.

## PolyglotBot

`scripts/build_report.py --emit-bot-config` writes the benchmark winner
(gated on >= 95% format reliability) into `bot/config.yaml`, and the bot runs
exactly that config — the eval-to-deployment link is a generated file, not a
claim. v1 scope: unencrypted rooms, text only, one language per room, opt-in
feedback (default off), per-user cooldown, `!lang` / `!feedback` / `!level` /
`!help`. The bot stores nothing on disk — no cache, no logs of message
content; the `!level` window is in-memory and bounded. Deployment guide
(Oracle Cloud Always Free, Docker or systemd): [bot/DEPLOY.md](bot/DEPLOY.md).

## Repo layout

```
config/           providers, model registry (per-model rate limits, prices), eval settings
src/langbench/    providers, ratelimit, cache, results, tasks, parsing, metrics, judge, runner, report
scripts/          discover_models, prepare_data, run_eval, build_report
bot/              PolyglotBot (matrix-nio), its tests, Dockerfile, deploy docs
tests/            fully offline suite incl. the smoke eval; hand-made fixtures
docs/             generated leaderboard (GitHub Pages)
data/manifests/   committed sampling manifests (IDs + seeds; never texts)
```

## Caveats, stated once

Free-tier latency is not paid-tier latency. Model IDs, rate limits, prices,
and dataset URLs in config were verified against live endpoints after the
initial build (`# VERIFY` markers track this; see HANDOFF_TO_CLAUDE_CODE.md).
The judge is a single model with a single rubric prompt; its correctness
dimensions are calibration-checked against gold corrections, its clarity
dimension only if hand labels exist. Languages: en, de, it, cs, es; Spanish
GEC-only.

## License

MIT for the code. Datasets are obtained by each user under their own
licenses; nothing from any corpus is committed here.
