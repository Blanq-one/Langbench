"""CEFR classification metrics, granularity-aware.

Rules (from the spec, enforced here):
- Predictions are always six-level; where gold is band-granular (English
  W&I), predictions are collapsed to A/B/C before scoring.
- QWK is computed per granularity and NEVER pooled across granularities.
  Callers pass items of a single granularity; mixing raises.

Implemented by hand (exact formulas below) rather than pulling in
scikit-learn for four small functions. # DECISION
"""

from __future__ import annotations

from dataclasses import dataclass

from langbench.data.schema import BAND_OF

SIX = ["A1", "A2", "B1", "B2", "C1", "C2"]
BANDS = ["A", "B", "C"]


@dataclass
class CefrScores:
    granularity: str
    n: int
    accuracy: float
    adjacent_accuracy: float
    macro_f1: float
    qwk: float
    per_item_correct: list[bool]  # for bootstrap on accuracy


def collapse_to_band(label: str) -> str:
    if label not in BAND_OF:
        raise ValueError(f"unknown CEFR label {label!r}")
    return BAND_OF[label]


def score_cefr(
    preds: list[str], golds: list[str], granularity: str
) -> CefrScores:
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} preds vs {len(golds)} golds")
    if granularity == "six_level":
        scale = SIX
        norm_pred, norm_gold = list(preds), list(golds)
    elif granularity == "band":
        scale = BANDS
        norm_pred = [collapse_to_band(p) if p != "UNPARSEABLE" else p for p in preds]
        norm_gold = list(golds)
    else:
        raise ValueError(f"granularity must be six_level|band, got {granularity!r}")

    for g in norm_gold:
        if g not in scale:
            raise ValueError(f"gold label {g!r} not on the {granularity} scale — "
                             "never mix granularities in one scoring call")

    idx = {label: i for i, label in enumerate(scale)}
    # UNPARSEABLE predictions (persistent format failures) score as maximally
    # wrong on ordinal metrics: distance = full scale width. They already
    # count against format reliability separately.
    worst = len(scale) - 1

    def pred_index(p: str, gold_i: int) -> int:
        if p in idx:
            return idx[p]
        return 0 if gold_i >= worst / 2 else worst  # farthest end from gold

    pairs = [(pred_index(p, idx[g]), idx[g]) for p, g in zip(norm_pred, norm_gold, strict=True)]
    correct = [p == g for p, g in pairs]
    adjacent = [abs(p - g) <= 1 for p, g in pairs]

    return CefrScores(
        granularity=granularity,
        n=len(pairs),
        accuracy=sum(correct) / len(pairs) if pairs else 0.0,
        adjacent_accuracy=sum(adjacent) / len(pairs) if pairs else 0.0,
        macro_f1=_macro_f1(pairs, len(scale)),
        qwk=_qwk(pairs, len(scale)),
        per_item_correct=correct,
    )


def _macro_f1(pairs: list[tuple[int, int]], k: int) -> float:
    f1s = []
    for c in range(k):
        tp = sum(1 for p, g in pairs if p == c and g == c)
        fp = sum(1 for p, g in pairs if p == c and g != c)
        fn = sum(1 for p, g in pairs if p != c and g == c)
        if tp + fp + fn == 0:
            continue  # class absent from both preds and golds: skip, don't zero-pad
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _qwk(pairs: list[tuple[int, int]], k: int) -> float:
    """Quadratic weighted kappa. w[i][j] = (i-j)^2 / (k-1)^2;
    kappa = 1 - sum(w*O) / sum(w*E), E from marginal outer product."""
    n = len(pairs)
    if n == 0:
        return 0.0
    obs = [[0.0] * k for _ in range(k)]
    for p, g in pairs:
        obs[g][p] += 1
    gold_marg = [sum(obs[i][j] for j in range(k)) for i in range(k)]
    pred_marg = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
            num += w * obs[i][j]
            den += w * (gold_marg[i] * pred_marg[j] / n)
    if den == 0:
        return 0.0
    return 1.0 - num / den
