"""LLM-as-judge (Gemini) for feedback quality, plus its calibration routine.

Design constraints honored:
- Judge is a different provider family from every candidate (config enforces).
- Temperature 0, fixed rubric prompt version, dimension ORDER randomized per
  item (seeded by item id) to blunt position bias.
- Calibration is scoped honestly: gold references ground only the two
  correctness dimensions (correct_errors, correction_accuracy, and the
  no_hallucinated check). 'explanation_clarity' has NO gold signal — the
  calibration report either uses the optional hand-label file
  (data/calibration/clarity_labels.jsonl: {"sample_id": ..., "clarity": 1-5})
  or explicitly marks clarity as UNCALIBRATED. Never implies whole-rubric
  validation from partial evidence.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from langbench.metrics.gleu import sentence_gleu
from langbench.providers.base import ChatRequest

JUDGE_PROMPT_VERSION = "judge-v1"

CLARITY_LABELS_PATH = Path("data/calibration/clarity_labels.jsonl")

DIMENSIONS = {
    "correct_errors": "Did the feedback identify the genuine errors in the learner text "
    "(compare against the gold correction)? 1 = missed most, 5 = found all genuine errors.",
    "correction_accuracy": "Are the proposed corrections accurate, judged against the gold "
    "correction? 1 = mostly wrong, 5 = all corrections right.",
    "explanation_clarity": "Are the explanations clear, short, and understandable for a "
    "language learner at this level? 1 = confusing, 5 = very clear.",
    "no_hallucinated": "Did the feedback avoid flagging things that are NOT errors "
    "(per the gold correction)? 1 = many invented errors, 5 = none invented.",
}


class JudgeScores(BaseModel):
    correct_errors: int = Field(ge=1, le=5)
    correction_accuracy: int = Field(ge=1, le=5)
    explanation_clarity: int = Field(ge=1, le=5)
    no_hallucinated: int = Field(ge=1, le=5)


_SYSTEM = (
    "You are a strict, consistent evaluator of language-learning feedback. "
    "You compare a tutoring system's feedback against a gold-standard "
    "correction and score it on fixed dimensions. Respond with ONLY a JSON "
    "object mapping each dimension name to an integer 1-5. No fences, no "
    "commentary."
)

_USER_TEMPLATE = """Learner text:
{source}

Gold-standard correction(s):
{golds}

Feedback produced by the system under evaluation (JSON):
{feedback_json}

Score the feedback on these dimensions (1-5 each):
{dimensions}

Respond with ONLY: {{{keys_hint}}}"""


def build_judge_request(
    sample_id: str,
    source: str,
    gold_references: list[str],
    feedback_json: str,
    max_tokens: int = 512,
) -> ChatRequest:
    # Deterministic per-item dimension order: seeded by sample id, so the
    # order varies across items (position-bias control) but is reproducible
    # and cache-stable for the same item.
    order = list(DIMENSIONS)
    random.Random(f"{JUDGE_PROMPT_VERSION}:{sample_id}").shuffle(order)
    dims_block = "\n".join(f"- {name}: {DIMENSIONS[name]}" for name in order)
    keys_hint = ", ".join(f'"{name}": <1-5>' for name in order)
    golds = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(gold_references)) or "(none)"
    return ChatRequest(
        system=_SYSTEM,
        user=_USER_TEMPLATE.format(
            source=source,
            golds=golds,
            feedback_json=feedback_json,
            dimensions=dims_block,
            keys_hint=keys_hint,
        ),
        temperature=0.0,
        max_tokens=max_tokens,
    )


# --------------------------- calibration ----------------------------------

@dataclass
class CalibrationReport:
    n_items: int
    # Spearman rank correlation between judge correction_accuracy and the
    # automatic gold-grounded signal (GLEU of the feedback's corrected text).
    correctness_spearman: float
    # Agreement rate between judge no_hallucinated>=4 and the automatic
    # zero-spurious-edit check.
    hallucination_agreement: float
    clarity_calibrated: bool
    clarity_spearman: float | None  # only if hand labels exist

    def summary(self) -> str:
        lines = [
            f"Judge calibration (n={self.n_items}, correctness dimensions only):",
            f"  correction_accuracy vs gold-GLEU (Spearman): {self.correctness_spearman:.3f}",
            f"  no_hallucinated vs automatic check (agreement): "
            f"{self.hallucination_agreement:.3f}",
        ]
        if self.clarity_calibrated and self.clarity_spearman is not None:
            lines.append(
                f"  explanation_clarity vs hand labels (Spearman): {self.clarity_spearman:.3f}"
            )
        else:
            lines.append(
                "  explanation_clarity: UNCALIBRATED (no gold signal exists; add "
                f"hand labels at {CLARITY_LABELS_PATH} to calibrate — ~30 items)"
            )
        return "\n".join(lines)


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return float("nan")

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def calibrate(
    items: list[dict[str, object]],
) -> CalibrationReport:
    """items: [{sample_id, source, golds: [..], corrected: str,
                judge: JudgeScores, spurious_edits: bool}]
    built by the runner from ~30 English judge items."""
    judge_acc: list[float] = []
    auto_gleu: list[float] = []
    hall_agree = 0
    for it in items:
        judge: JudgeScores = it["judge"]  # type: ignore[assignment]
        golds: list[str] = it["golds"]  # type: ignore[assignment]
        judge_acc.append(float(judge.correction_accuracy))
        auto_gleu.append(
            sentence_gleu(str(it["source"]), str(it["corrected"]), golds)
        )
        judge_clean = judge.no_hallucinated >= 4
        auto_clean = not bool(it["spurious_edits"])
        hall_agree += int(judge_clean == auto_clean)

    clarity_spearman: float | None = None
    calibrated = False
    if CLARITY_LABELS_PATH.exists():
        hand: dict[str, float] = {}
        with CLARITY_LABELS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    hand[str(rec["sample_id"])] = float(rec["clarity"])
        pairs = [
            (float(it["judge"].explanation_clarity), hand[str(it["sample_id"])])  # type: ignore[attr-defined]
            for it in items
            if str(it["sample_id"]) in hand
        ]
        if len(pairs) >= 10:
            clarity_spearman = _spearman([p for p, _ in pairs], [h for _, h in pairs])
            calibrated = True

    return CalibrationReport(
        n_items=len(items),
        correctness_spearman=_spearman(judge_acc, auto_gleu),
        hallucination_agreement=hall_agree / len(items) if items else float("nan"),
        clarity_calibrated=calibrated,
        clarity_spearman=clarity_spearman,
    )
