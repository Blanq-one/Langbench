# HANDOFF TO CLAUDE CODE

Work order for live verification and fix-up of the one-shot build. Written by
the build itself; nothing in this repo has touched a live API or a real
corpus file yet. Two mid-build review fixes are already applied (Gemini key
moved out of URLs into the `x-goog-api-key` header; bot disk cache removed
for privacy) — do not re-introduce either.

**First command, before anything live:**

```bash
uv sync && ruff check . && mypy . && pytest
```

The suite is designed to pass fully offline with zero keys. If ruff or
mypy-strict flag stragglers, fix them first — the build could not execute
these tools, only design for them. Likely mypy-strict nits: the
`# type: ignore` assignments injecting fake adapters in
tests/test_runner_smoke.py (tests are outside mypy's files anyway), Any
leakage around `matplotlib`/`nio` (overrides exist in pyproject), and the
`_qwk_stat` closure typing in report.py.

---

## A. Every `# VERIFY` item, grouped, with the verification step

### A1. Provider endpoints and wire formats
| Where | What | How to verify |
|---|---|---|
| config/providers.yaml (all 5 base_url) | base URLs | one live smoke call per provider (see C4) |
| src/langbench/providers/groq.py, mistral.py | OpenAI-compat shape incl. usage block | smoke call; inspect `usage.prompt_tokens/completion_tokens` |
| src/langbench/providers/gemini.py | `POST /models/{id}:generateContent`, `x-goog-api-key` header, `usageMetadata` field names | Gemini REST docs + smoke call. Key must NEVER move into the URL (leaks via httpx exception reprs) |
| src/langbench/providers/openrouter.py:20 | HTTP-Referer / X-Title attribution headers + real repo URL | OpenRouter docs |
| scripts/discover_models.py:47 | Gemini `GET /models` response shape | run the script with a key |

### A2. Model registry (config/models.yaml — every row)
- Run `uv run python scripts/discover_models.py`, diff printed ids against
  models.yaml, fix drifted ids (gemma2-9b-it is the most likely retirement;
  OpenRouter `:free` slugs churn constantly).
- Fill REAL per-(provider, model) RPM/RPD from each provider dashboard/docs.
  Do not copy limits across rows — the Groq 70B-vs-8B daily gap is the whole
  reason limits are per model. The Gemini judge RPD drives the judge-day
  math in the estimator; get it right.
- Fill list prices (USD per 1M tokens) from provider pricing pages,
  including gpt-4o-mini (it anchors the cost axis even while disabled).

### A3. Datasets (highest-risk area of the whole build)
- scripts/prepare_data.py: all three dataset URLs and the W&I M2 filenames.
- src/langbench/data/parsers.py — ALL parsers were written against
  documented formats, sight unseen. Expected fix-ups:
  - **MERLIN**: section headers ("Overall CEFR rating:", "Learner text:",
    "Target hypothesis 1:") and directory layout are guesses at the
    plain-text export; the real CLARIN distribution may be XML — if so,
    rewrite parse_merlin_* accordingly and mirror the change in
    tests/fixtures/merlin_de.txt + tests/test_parsers_data.py.
  - **COWS-L2H**: original/ vs corrected/ sibling-directory layout and
    whether corrected2/ exists; check the repo's actual tree.
  - **W&I M2**: lowest risk (M2 is a stable standard), but confirm the
    band-in-filename convention and the dev-file name.
  - Parsers fail loudly with expected-structure messages by design; when one
    fires, fix parser AND fixture together so the tests keep pinning reality.
- DATA_LICENSES.md: URLs, license terms/versions. Confirm W&I terms permit
  this use; MERLIN CC BY-SA version; COWS-L2H repo license file.

### A4. Metrics
- src/langbench/metrics/gleu.py: sanity-check ~10 scores against the
  reference implementation (github.com/cnap/gec-ranking) on English items.
  The source-ngram penalty and multi-reference mean are the places to watch.
- src/langbench/metrics/errant_wrapper.py: `uv sync --extra errant`,
  `python -m spacy download en_core_web_sm`, confirm `errant.load("en")` and
  the annotate/parse API against the installed version.

### A5. Bot
- bot/polyglotbot/main.py: matrix-nio AsyncClient callback signature,
  RoomMessageText fields (`server_timestamp`, `event_id`), and the
  `m.relates_to` thread payload shape against the installed nio version.
  Test in a scratch unencrypted room before publicizing.
- bot/DEPLOY.md: Oracle Always Free shape names.

### A6. Tooling
- .github/workflows/ci.yaml: astral-sh/setup-uv action major version.
- pyproject pins (httpx, pydantic, matrix-nio, errant ranges) against
  current releases at `uv sync` time.

## B. Every `# DECISION`, one-line rationale each

1. Parsers live in src/langbench/data/parsers.py, not inside
   prepare_data.py — importable means unit-testable.
2. Sync sqlite + asyncio.to_thread over aiosqlite — one fewer dep, same
   semantics at this call volume.
3. Rate limiter is stateless across restarts; --resume reseeds daily usage
   from the results DB via count_today, which can only overcount (cache hits
   also create records) — conservative by construction.
4. preload_daily_usage takes max(current, seeded) — reseeding must never
   lower known usage (a mid-build review fix; regression-tested in
   tests/test_ratelimit.py and the quota-parking smoke test).
5. Results DB enforces a per-task field allowlist + closed string sets —
   corpus text physically cannot enter the committed artifact.
6. gold_label is stored in the results DB — QWK needs (pred, gold) and gold
   CEFR labels are closed-set values, not corpus text.
7. W&I ABCN dev file carries no per-sentence band, so English CEFR uses the
   banded train files; dev feeds GEC. [AMENDED by DECISION 34, 2026-07-25:
   v2.1 ships per-band dev files, so dev sentences do carry bands; English
   CEFR pools banded train + dev.]
8. COWS-L2H course levels are not CEFR; Spanish is GEC-only (a mapping was
   rejected as indefensible without external calibration).
9. CLC-FCE deferred: registration-gated, spec said "if easily obtainable".
10. UNPARSEABLE CEFR predictions score maximally wrong on ordinal metrics —
    format failures can never help a model.
11. GLEU multi-reference: mean over references (equivalent in expectation to
    Napoles's per-iteration reference sampling at 1-2 refs).
12. Metrics hand-rolled (QWK, macro-F1, Spearman, GLEU) — no
    scikit-learn/scipy/nltk in the dependency tree.
13. Judge dimension order is seeded per sample_id: randomized across items
    (position-bias control), deterministic per item (cache-stable).
14. Judge calibration mode lives in run_eval.py (--calibrate-judge), not a
    separate script; it uses ONE candidate's outputs so judge agreement is
    not confounded with candidate quality.
15. Spurious-edit proxy for calibration: corrected text contains tokens
    absent from both source and every gold.
16. Runner concurrency: models parallel, items sequential per model — at
    15-30 RPM, intra-model parallelism buys nothing.
17. Provider errors leave items PENDING (infra's fault, retried next run);
    persistent parse failures are SCORED failures (model's fault).
18. Candidate format failure on the feedback task records floor rubric
    scores (all 1s) with format_ok=0.
19. Cache hits reuse the original live latency measurement.
20. There is deliberately no --no-resume: re-spending free quota by accident
    is the worst failure mode; redo an item by deleting its results-DB row.
21. Cost basis: feedback-task tokens (closest to bot traffic), GEC fallback.
22. Composite quality: mean of available rescaled components (GLEU; clipped
    QWK, six-level preferred; judge (x-1)/4) — formula printed in REPORT.md.
23. Bot-config winner: best composite among models >= 95% format
    reliability; ties to the cheaper model.
24. Bot behavior knobs come from env vars; the generated bot/config.yaml
    stays purely the eval output.
25. Bot "other bots" heuristic: localpart contains "bot", plus always skip
    self.
26. Bot stays silent on infra errors and on correct messages; only rate
    limiting gets an in-room reply; failed attempts don't consume the
    cooldown; feedback capped at 5 items; replies are m.notice.
27. **Bot has ZERO disk persistence** (deviates from spec §5 "reuse cache"):
    the raw cache stores full responses quoting learner messages, which
    would falsify the !help privacy note; live messages don't repeat so the
    cache saved nothing. systemd ProtectSystem=strict now enforces this at
    the OS level. (Mid-build review decision.)
28. English is band-granular and never pooled with six-level QWK; the report
    carries two separate QWK columns.
29. (Live integration, 2026-07-24) Fourth Groq candidate added
    (openai/gpt-oss-20b): models run in parallel loops so an extra candidate
    is ~free in calendar time, and family diversity (Llama/Qwen/OpenAI-OSS)
    strengthens the comparison. qwen/qwen3.6-27b replaces the retired
    gemma2-9b-it.
30. (Live integration, 2026-07-24) Judge is gemini-3.5-flash-lite, not a
    regular Flash model: this account's Free tier caps ALL regular Flash
    models at 20 RPD; Flash-Lite's 500 RPD makes the ~1,030-call judge phase
    ~2-3 days instead of 50+. gemma-4-26b (30 RPM / 14400 RPD, free) sits in
    models.yaml as a commented-out quota-emergency judge fallback.
31. (Live integration, 2026-07-24) No TPM buckets in the rate limiter:
    RPM/RPD dominate at this call volume and 429 backoff absorbs the rest.
    Groq's 6K TPM on the 8B model may bind before RPM; the estimator prints
    this caveat and the models.yaml row carries the note.
32. (Live integration, 2026-07-24) qwen3.6-27b runs with
    extra_body reasoning_format=hidden. "none" would change the model's
    capability and answer a question nobody asked (Qwen-without-reasoning
    isn't what a bot operator would deploy); leaving it raw would punish
    Qwen for our parser's first-balanced-brace heuristic — a harness
    artifact, not model quality. Hidden preserves capability while keeping
    content parseable. extra_body is hashed into the cache key (semantics
    change => cache invalidation) and emit_bot_config propagates it so the
    bot deploys with the identical wire settings the benchmark measured.
    Reasoning models' completion_tokens (and $/1K messages) deliberately
    include reasoning overhead — a bot operator pays it too.
33. (Live integration, 2026-07-24) Gemini adapter joins ALL non-thought
    text parts instead of reading parts[0]: responses already carry
    thoughtSignature fields, and a future thought/text part split must not
    break the judge phase mid-run.
34. (Live integration, 2026-07-25) AMENDS DECISION 7: W&I v2.1 ships
    per-band dev M2 files (A/B/C/N.dev.gold.bea19.m2), so dev sentences DO
    carry bands — the "dev is band-less" premise held only for the pooled
    ABCN.dev file. prepare_wi_locness now parses banded dev (pooled file
    kept as fallback); English CEFR pools banded train + dev, GEC unchanged.
35. (Live integration, 2026-07-25) Reasoning models (qwen3.6-27b,
    gpt-oss-20b) get max_output_tokens 4096: hidden reasoning consumes the
    budget before the answer (C6: 79/94 qwen responses hit
    finish_reason=length with all 1024 tokens spent reasoning, content
    empty). The resulting token cost stays in the honest cost column.
    max_tokens is per-model in the cache key, so other models' cached calls
    stay valid.
36. (Live integration, 2026-07-25) gleu.py adopts the reference
    implementation's exact behaviors (cnap/gec-ranking, sentence-level
    smooth=True): zero stats smooth to 1 (not a 1e-9 log floor) and the
    source-ngram penalty is a SET difference (reference-present n-gram types
    are never penalized). Verified 10/10 exact on real W&I items; the old
    smoothing diverged by up to 0.19 on zero-match sentences — ranking-
    changing. "GLEU (Napoles et al.)" in the report means the reference
    implementation, full stop.
37. (Live integration, 2026-07-27) qwen3.6-27b DROPPED as a candidate:
    evaluated for feasibility, infeasible at free tier. Hidden reasoning
    (DECISION 32) averaged 1,319 tokens/request on single-sentence English
    GEC (llama-8B: ~27 completion tokens), billing against Groq's 200K
    tokens/DAY budget => ~4 weeks for the full manifest. This FALSIFIES
    DECISION 29's "fourth candidate is ~free in calendar time" premise: the
    TPD limit class was modeled neither by the transcribed dashboard table
    (A2) nor by the estimator (DECISION 31 covered TPM only). Cached data
    and partial English rows retained; excluded from the headline table
    (paired comparisons need full coverage); finding reported as a named
    subsection. Remaining models' rpm/rpd derated to observed token budgets
    (0.8 x budget / avg request tokens): llama-8B 10/800, 70B 17/160,
    gpt-oss 30/200 (no TPM 429 ever observed for gpt-oss, so its rpm is not
    derated on invented numbers). Pacing is harness scheduling, not model
    behavior; rate limits are not in cache keys.
38. (Live integration, 2026-07-26) The 0.8 safety factor on rpd is dropped
    (70B 160->200, gpt-oss 200->250; llama's 800 kept, it finishes
    regardless): the failure mode it protected against is now cheap —
    end-of-day TPD 429s burn a few bounded retries and items go PENDING
    into tomorrow's pass (minutes of churn) — while the protection cost
    ~2 calendar days. PENDING-is-cheap changes the tradeoff.

## C. Live-integration checklist, in order

1. `uv sync && ruff check . && mypy . && pytest` — all green offline first.
2. `cp .env.example .env`, add GROQ_API_KEY + GEMINI_API_KEY (minimum).
3. `uv run python scripts/discover_models.py` — reconcile models.yaml ids;
   fill real rate limits and prices (A2).
4. One live smoke call per enabled provider (a 3-line script or
   `run_eval.py --models <one> --langs en` against a 1-item manifest);
   confirm response parsing and usage fields per adapter (A1).
5. Download corpora per `prepare_data.py` instructions; run it per dataset;
   fix parsers against real formats (A3 — budget the most time here);
   `--manifests` last.
6. Tiny live eval: `run_eval.py --langs en --models <one> <one-more>`.
   NOTE (corrected 2026-07-25 after it caused an over-scale run): run scale
   comes from the COMMITTED MANIFESTS (data/manifests/), not eval.yaml —
   n_auto only affects `prepare_data.py --manifests`. There is no small-run
   knob; restrict scope with --langs/--models and know it runs the full
   manifest for that slice. Also: on Windows, killing run_eval.py requires
   killing the PROCESS TREE (uv spawns a python child that survives the
   parent; an orphan will silently keep spending quota).
7. Restore sample sizes; `run_eval.py --dry-run`; sanity-check the quota
   estimate against the real limits you filled in.
8. Full candidates phase daily with defaults until pending hits zero.
9. `run_eval.py --phase judge` daily (this is the long pole: ~1,500 judge
   calls vs the Gemini daily cap).
10. `run_eval.py --calibrate-judge`; optionally hand-label ~30 clarity items
    into data/calibration/clarity_labels.jsonl and rerun.
11. `build_report.py` — read REPORT.md skeptically (CI overlaps, rewrite
    flags, cost sanity). Then `--emit-bot-config`.
12. Bot: scratch Matrix account + unencrypted test room; run locally first
    (`uv sync --extra bot`, PYTHONPATH=bot, `python -m polyglotbot.main`);
    verify threads render, cooldown works, !level answers; then deploy per
    bot/DEPLOY.md.
13. Enable GitHub Pages on /docs for the leaderboard.

## D. Honest low-confidence list (weakest code first)

1. **Dataset parsers** (data/parsers.py) — written blind; MERLIN most likely
   to be structurally wrong (possibly XML, not the assumed plain text).
2. **Provider response schemas** — the OpenAI-compat assumption is solid for
   Groq/OpenRouter/Mistral but usage-field names and error bodies vary;
   Gemini's shape was written from memory.
3. **GLEU implementation** — the formula is from the papers, not diffed
   against the reference implementation; the smoothing choice (1e-9 log
   floor for zero precisions) is defensible but unvalidated.
4. **matrix-nio integration** (bot/polyglotbot/main.py) — API written from
   memory; the thread-relation dict and the server_timestamp backlog guard
   need a live room to confirm.
5. **Rate-limit placeholder numbers** — every RPM/RPD is invented; the
   estimator's calendar-day math is only as good as those rows.
6. **ERRANT wrapper** — guarded and optional, but the annotate/parse call
   pattern may not match the current errant release.
7. **discover_models.py Gemini listing** — endpoint shape unverified.
8. **mypy-strict cleanliness** — designed for, never executed; expect a
   short fix pass.

End of handoff. The build's contract was: structurally complete, internally
consistent (automated import-graph + payload-allowlist + config-reference
sweep passed), offline-test-designed, honestly annotated. Everything live is
yours.
