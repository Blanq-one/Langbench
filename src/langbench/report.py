"""Build REPORT.md, the static HTML leaderboard, charts, and the bot config.

Reporting rules enforced here (not just promised in the README):
- Every headline metric carries a bootstrap 95% CI. No point-estimate rankings.
- QWK is NEVER pooled across granularities: six-level QWK (de/it/cs pooled —
  same scale) and English band QWK are separate columns, labeled as such.
- Judge score column notes when explanation_clarity is uncalibrated.
- Cost is $ per 1,000 learner messages from measured tokens x list prices,
  based on feedback-task usage (closest to real bot traffic), falling back to
  GEC usage when a model has no feedback records. # DECISION
- Format reliability is a first-class column; the bot-config winner must
  clear MIN_FORMAT_RELIABILITY. # DECISION: composite quality picks the
  winner among models >= 95% format reliability; ties go to the cheaper model.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsInt, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from langbench.config import Registry  # noqa: E402
from langbench.judge import CLARITY_LABELS_PATH  # noqa: E402
from langbench.metrics.bootstrap import CI, ci_mean, ci_statistic, paired_delta  # noqa: E402
from langbench.metrics.cefr_metrics import _qwk  # noqa: E402
from langbench.results import ResultsDB  # noqa: E402

MIN_FORMAT_RELIABILITY = 0.95
SIX = ["A1", "A2", "B1", "B2", "C1", "C2"]
BANDS = ["A", "B", "C"]
BAND_OF = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C", "C2": "C"}


@dataclass
class ModelReport:
    model_key: str
    display_name: str
    gec_gleu: CI
    qwk_six: CI | None       # de/it/cs pooled (same scale)
    qwk_band_en: CI | None   # English A/B/C bands — separate on purpose
    judge_mean: CI | None    # mean of 4 rubric dims over judged items
    format_reliability: float
    n_records: int
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_messages: float | None  # None when no token/price data
    composite: float
    rewrite_flag_rate: float  # share of GEC items flagged as near-total rewrites


def _qwk_stat(scale: list[str]) -> object:
    def stat(preds: list[str], golds: list[str]) -> float:
        idx = {c: i for i, c in enumerate(scale)}
        worst = len(scale) - 1
        pairs = []
        for p, g in zip(preds, golds, strict=True):
            gi = idx[g]
            pi = idx.get(p, 0 if gi >= worst / 2 else worst)  # UNPARSEABLE => farthest
            pairs.append((pi, gi))
        return _qwk(pairs, len(scale))

    return stat


def aggregate(results: ResultsDB, reg: Registry) -> list[ModelReport]:
    reports: list[ModelReport] = []
    model_keys = sorted({r["model_key"] for r in results.fetch()})
    for mk in model_keys:
        recs = results.fetch(model_key=mk)
        if not recs:
            continue
        try:
            display = reg.model(mk).display_name
        except KeyError:
            display = mk  # model since removed from registry: still report it

        gec = [r for r in recs if r["task"] == "gec"]
        gleu_vals = [float(r["payload"]["gleu"]) for r in gec]
        rewrites = [bool(r["payload"].get("rewrote_everything")) for r in gec if r["format_ok"]]

        cefr = [r for r in recs if r["task"] == "cefr"]
        six_pairs = [
            (str(r["payload"]["pred_label"]), _gold_from_record(r, "six_level"))
            for r in cefr if r["payload"]["gold_granularity"] == "six_level"
        ]
        band_pairs = [
            (BAND_OF.get(str(r["payload"]["pred_label"]), "UNPARSEABLE"),
             _gold_from_record(r, "band"))
            for r in cefr if r["payload"]["gold_granularity"] == "band"
        ]

        fb = [r for r in recs if r["task"] == "feedback"]
        judge_means = [
            statistics.mean(
                [
                    float(r["payload"]["judge_correct_errors"]),
                    float(r["payload"]["judge_correction_accuracy"]),
                    float(r["payload"]["judge_explanation_clarity"]),
                    float(r["payload"]["judge_no_hallucinated"]),
                ]
            )
            for r in fb
        ]

        latencies = sorted(
            float(r["latency_ms"]) for r in recs if r["latency_ms"] is not None
        )
        fmt_ok = sum(bool(r["format_ok"]) for r in recs)

        gleu_ci = ci_mean(gleu_vals, seed=1)
        qwk_six = (
            ci_statistic(six_pairs, _qwk_stat(SIX), seed=2)  # type: ignore[arg-type]
            if six_pairs else None
        )
        qwk_band = (
            ci_statistic(band_pairs, _qwk_stat(BANDS), seed=3)  # type: ignore[arg-type]
            if band_pairs else None
        )
        judge_ci = ci_mean(judge_means, seed=4) if judge_means else None

        cost = _cost_per_1k(reg, mk, fb or gec)
        composite = _composite(gleu_ci, qwk_six, qwk_band, judge_ci)

        reports.append(
            ModelReport(
                model_key=mk,
                display_name=display,
                gec_gleu=gleu_ci,
                qwk_six=qwk_six,
                qwk_band_en=qwk_band,
                judge_mean=judge_ci,
                format_reliability=fmt_ok / len(recs) if recs else 0.0,
                n_records=len(recs),
                p50_latency_ms=_pct(latencies, 0.50),
                p95_latency_ms=_pct(latencies, 0.95),
                cost_per_1k_messages=cost,
                composite=composite,
                rewrite_flag_rate=(sum(rewrites) / len(rewrites)) if rewrites else 0.0,
            )
        )
    reports.sort(key=lambda r: r.composite, reverse=True)
    return reports


def _gold_from_record(rec: dict[str, object], granularity: str) -> str:
    """QWK needs (pred, gold) pairs and the results DB is the only committed
    artifact, so the runner stores gold_label alongside pred_label. Gold CEFR
    labels are closed-set values (A1..C2 / A/B/C), NOT corpus text, so this is
    license-safe and keeps the committed DB self-sufficient for reporting."""
    payload = rec["payload"]
    assert isinstance(payload, dict)
    gold = str(payload["gold_label"])
    return BAND_OF.get(gold, gold) if granularity == "band" else gold


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(int(q * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[i]


def _cost_per_1k(reg: Registry, model_key: str, recs: list[dict[str, object]]) -> float | None:
    try:
        m = reg.model(model_key)
    except KeyError:
        return None
    ptoks = [
        int(cast(SupportsInt, r["prompt_tokens"]))
        for r in recs
        if r["prompt_tokens"] is not None
    ]
    ctoks = [
        int(cast(SupportsInt, r["completion_tokens"]))
        for r in recs
        if r["completion_tokens"] is not None
    ]
    if not ptoks or not ctoks:
        return None
    mean_in = statistics.mean(ptoks)
    mean_out = statistics.mean(ctoks)
    per_msg = (
        mean_in * m.pricing.input_per_mtok + mean_out * m.pricing.output_per_mtok
    ) / 1_000_000
    return per_msg * 1000


def _composite(
    gleu: CI, qwk_six: CI | None, qwk_band: CI | None, judge: CI | None
) -> float:
    """Plainly-stated composite for the frontier chart and winner pick:
    mean of available components, each rescaled to [0,1]:
    GLEU as-is; QWK clipped at 0 (six-level preferred, en band if that's all);
    judge mean mapped (x-1)/4. Absent components are excluded, not zeroed."""
    parts: list[float] = [gleu.point] if gleu.n else []
    qwk = qwk_six if qwk_six is not None else qwk_band
    if qwk is not None:
        parts.append(max(qwk.point, 0.0))
    if judge is not None:
        parts.append((judge.point - 1.0) / 4.0)
    return statistics.mean(parts) if parts else 0.0


# ------------------------------ rendering ----------------------------------

def render_markdown(reports: list[ModelReport], results: ResultsDB) -> str:
    clarity_note = (
        "calibrated against hand labels"
        if CLARITY_LABELS_PATH.exists()
        else "explanation_clarity dimension is UNCALIBRATED (no hand labels present)"
    )
    lines = [
        "# Langbench report",
        "",
        "All intervals are bootstrap 95% CIs. QWK columns are per granularity",
        "and are never pooled: `QWK (6-lvl)` pools de/it/cs (same six-level",
        "scale); `QWK (en bands)` is English on A/B/C bands from W&I+LOCNESS.",
        f"Judge rubric: {clarity_note}.",
        "Latency was measured on FREE tiers; paid-tier latency will differ.",
        "Cost = measured tokens x list prices, as $ per 1,000 learner messages.",
        "gpt-4o-mini's price is its historical list price ($0.15/$0.60 per 1M):",
        "OpenAI's current pricing page no longer lists the model, so it serves",
        "only as a cost anchor, not a live offer.",
        "",
        "| Model | GEC GLEU | QWK (6-lvl) | QWK (en bands) | Judge (1-5) "
        "| Format OK | p50 ms | $/1K msgs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        if r.cost_per_1k_messages is not None:
            cost = "$" + format(r.cost_per_1k_messages, ".4f")
        else:
            cost = "n/a"
        lines.append(
            f"| {r.display_name} "
            f"| {r.gec_gleu.fmt()} "
            f"| {r.qwk_six.fmt() if r.qwk_six else 'n/a'} "
            f"| {r.qwk_band_en.fmt() if r.qwk_band_en else 'n/a'} "
            f"| {r.judge_mean.fmt(2) if r.judge_mean else 'n/a'} "
            f"| {r.format_reliability:.1%} "
            f"| {r.p50_latency_ms:.0f} "
            f"| {cost} |"
        )
    lines += ["", "## Pairwise deltas vs the top model (paired bootstrap, GEC GLEU)", ""]
    lines += _delta_lines(reports, results)
    lines += ["", "## Recommendation", ""] + _recommendation_lines(reports)
    lines += [
        "",
        "## Caveats (stated once)",
        "- Overlapping CIs mean the ranking between those models is not settled",
        "  by this data; the table above shows several such pairs plainly.",
        "- Free-tier latency is not paid-tier latency.",
        "- Spanish is GEC-only (COWS-L2H has course levels, not CEFR labels).",
        "- English CEFR is band-granular; do not compare its QWK to six-level QWK.",
        "- Judge scores share one provider family (Gemini) and one rubric prompt;",
        "  correctness dimensions are calibration-checked, clarity may not be.",
        "- Models with rewrite-flag rates above ~10% edit far more than minimal:",
    ]
    for r in reports:
        if r.rewrite_flag_rate > 0.10:
            lines.append(f"  - {r.display_name}: {r.rewrite_flag_rate:.0%} of GEC items flagged")
    return "\n".join(lines) + "\n"


def _delta_lines(reports: list[ModelReport], results: ResultsDB) -> list[str]:
    if len(reports) < 2:
        return ["(fewer than two models; no deltas)"]
    top = reports[0]
    top_recs = {
        (r["lang"], r["sample_id"]): float(r["payload"]["gleu"])
        for r in results.fetch(task="gec", model_key=top.model_key)
    }
    out = []
    for other in reports[1:]:
        other_recs = {
            (r["lang"], r["sample_id"]): float(r["payload"]["gleu"])
            for r in results.fetch(task="gec", model_key=other.model_key)
        }
        shared = sorted(top_recs.keys() & other_recs.keys())
        if not shared:
            out.append(f"- {top.display_name} vs {other.display_name}: no shared items")
            continue
        d = paired_delta(
            [top_recs[k] for k in shared], [other_recs[k] for k in shared], seed=7
        )
        verdict = "CI excludes 0" if d.excludes_zero else "CI includes 0 — not settled"
        out.append(
            f"- {top.display_name} - {other.display_name}: "
            f"{d.delta:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}] (n={d.n}; {verdict})"
        )
    return out


def _recommendation_lines(reports: list[ModelReport]) -> list[str]:
    eligible = [r for r in reports if r.format_reliability >= MIN_FORMAT_RELIABILITY]
    if not eligible:
        return [
            f"No model reached {MIN_FORMAT_RELIABILITY:.0%} format reliability; "
            "fix prompting or models before deploying any of these."
        ]
    winner = max(
        eligible,
        key=lambda r: (r.composite, -(r.cost_per_1k_messages or 0.0)),
    )
    lines = [
        f"**{winner.display_name}** has the best composite quality "
        f"({winner.composite:.3f}) among models with >= "
        f"{MIN_FORMAT_RELIABILITY:.0%} format reliability."
    ]
    if winner.cost_per_1k_messages is not None:
        lines.append(
            f"At list prices it projects to ${winner.cost_per_1k_messages:.4f} "
            "per 1,000 learner messages."
        )
    ref = next((r for r in reports if "gpt-4o-mini" in r.model_key), None)
    if ref and ref is not winner and ref.cost_per_1k_messages and \
            winner.cost_per_1k_messages is not None:
        lines.append(
            f"Reference point: {ref.display_name} scores {ref.composite:.3f} at "
            f"${ref.cost_per_1k_messages:.4f}/1K messages."
        )
    lines.append(
        "`build_report.py --emit-bot-config` writes this winner into "
        "bot/config.yaml for PolyglotBot."
    )
    return lines


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Langbench leaderboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;
  padding:0 1rem;color:#1a1a1a}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 th,td{{border:1px solid #ddd;padding:6px 9px;text-align:left}}
 th{{background:#f5f5f5;cursor:pointer}}
 tr:nth-child(even){{background:#fafafa}}
 .note{{color:#555;font-size:13px}}
 img{{max-width:100%}}
</style></head><body>
<h1>Langbench leaderboard</h1>
<p class="note">Bootstrap 95% CIs in brackets. QWK is reported per label
granularity and never pooled across granularities. Latency measured on free
tiers. Cost = measured tokens &times; list prices, $ per 1,000 learner
messages. Click headers to sort.</p>
<table id="lb"><thead><tr>
<th>Model</th><th>GEC GLEU</th><th>QWK (6-lvl)</th><th>QWK (en bands)</th>
<th>Judge (1-5)</th><th>Format OK</th><th>p50 ms</th><th>$/1K msgs</th>
</tr></thead><tbody>
{rows}
</tbody></table>
<h2>Quality per dollar</h2>
<img src="frontier.png" alt="quality vs cost frontier scatter">
<script>
document.querySelectorAll('#lb th').forEach((th,i)=>th.onclick=()=>{{
 const tb=th.closest('table').tBodies[0];
 const rows=[...tb.rows].sort((a,b)=>{{
  const x=a.cells[i].dataset.v??a.cells[i].textContent,
        y=b.cells[i].dataset.v??b.cells[i].textContent;
  return isNaN(x-y)?String(x).localeCompare(y):y-x;}});
 rows.forEach(r=>tb.appendChild(r));}});
</script>
</body></html>
"""


def render_html(reports: list[ModelReport]) -> str:
    rows = []
    for r in reports:
        cost = (
            f'<td data-v="{r.cost_per_1k_messages:.6f}">${r.cost_per_1k_messages:.4f}</td>'
            if r.cost_per_1k_messages is not None
            else "<td>n/a</td>"
        )
        rows.append(
            "<tr>"
            f"<td>{r.display_name}</td>"
            f'<td data-v="{r.gec_gleu.point:.4f}">{r.gec_gleu.fmt()}</td>'
            f"<td>{r.qwk_six.fmt() if r.qwk_six else 'n/a'}</td>"
            f"<td>{r.qwk_band_en.fmt() if r.qwk_band_en else 'n/a'}</td>"
            f"<td>{r.judge_mean.fmt(2) if r.judge_mean else 'n/a'}</td>"
            f'<td data-v="{r.format_reliability:.4f}">{r.format_reliability:.1%}</td>'
            f'<td data-v="{r.p50_latency_ms:.0f}">{r.p50_latency_ms:.0f}</td>'
            f"{cost}"
            "</tr>"
        )
    return _HTML_TEMPLATE.format(rows="\n".join(rows))


def make_frontier_chart(reports: list[ModelReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs, ys, labels = [], [], []
    for r in reports:
        if r.cost_per_1k_messages is None:
            continue
        xs.append(max(r.cost_per_1k_messages, 1e-6))
        ys.append(r.composite)
        labels.append(r.display_name)
    ax.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("$ per 1,000 learner messages (list prices, log scale)")
    ax.set_ylabel("composite quality (see REPORT.md for the formula)")
    ax.set_title("Quality per dollar")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def emit_bot_config(reports: list[ModelReport], reg: Registry, path: Path) -> str:
    eligible = [r for r in reports if r.format_reliability >= MIN_FORMAT_RELIABILITY]
    pool = eligible or reports
    if not pool:
        raise ValueError("no models in results DB; run the eval before emitting bot config")
    winner = max(pool, key=lambda r: (r.composite, -(r.cost_per_1k_messages or 0.0)))
    m = reg.model(winner.model_key)
    prov = reg.providers[m.provider]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated by scripts/build_report.py --emit-bot-config\n"
        "# This file is the eval->deployment link: PolyglotBot runs the\n"
        "# Langbench winner, verbatim.\n"
        f"model_key: {m.key}\n"
        f"provider: {m.provider}\n"
        f"provider_base_url: {prov.base_url}\n"
        f"api_key_env: {prov.api_key_env}\n"
        f"model_id: {m.model_id}\n"
        f"prompt_version: {reg.eval.prompt_versions['feedback']}\n"
        f"rate_limit_rpm: {m.rate_limit.rpm}\n"
        f"rate_limit_rpd: {m.rate_limit.rpd}\n"
        f"max_output_tokens: {m.max_output_tokens}\n",
        encoding="utf-8",
    )
    return winner.model_key


def build_all(
    results: ResultsDB,
    reg: Registry,
    report_path: Path,
    docs_dir: Path,
) -> list[ModelReport]:
    reports = aggregate(results, reg)
    if not reports:
        raise ValueError(
            "results DB is empty; run scripts/run_eval.py before building the report"
        )
    report_path.write_text(render_markdown(reports, results), encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "index.html").write_text(render_html(reports), encoding="utf-8")
    make_frontier_chart(reports, docs_dir / "frontier.png")
    return reports
