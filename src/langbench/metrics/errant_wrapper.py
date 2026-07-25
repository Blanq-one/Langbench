"""ERRANT (M2-based P/R/F0.5) for ENGLISH only. Optional dependency.

Install with: uv sync --extra errant && python -m spacy download en_core_web_sm
# VERIFY: current errant API (errant.load('en'), annotator.parse/annotate)
# and required spaCy model name.

Everything here degrades loudly-but-gracefully: if errant is missing the
caller gets ErrantUnavailable and the report prints 'not installed' instead
of a number. Nothing in the offline test suite imports errant for real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ErrantUnavailable(RuntimeError):
    pass


@dataclass
class ErrantScores:
    precision: float
    recall: float
    f05: float
    tp: int
    fp: int
    fn: int


def _load_annotator() -> Any:
    try:
        import errant  # noqa: PLC0415
    except ImportError as e:
        raise ErrantUnavailable(
            "errant is not installed. Install: uv sync --extra errant && "
            "python -m spacy download en_core_web_sm"
        ) from e
    return errant.load("en")  # VERIFY


def errant_scores(
    sources: list[str], candidates: list[str], references: list[list[str]]
) -> ErrantScores:
    """Corpus-level P/R/F0.5. Candidate edits vs the best-matching reference's
    edits per sentence (standard multi-reference ERRANT practice: score
    against each reference, keep the best F0.5 alignment per sentence).
    """
    annotator = _load_annotator()
    tp = fp = fn = 0
    for src, cand, refs in zip(sources, candidates, references, strict=True):
        orig = annotator.parse(src)
        cand_edits = {
            (e.o_start, e.o_end, e.c_str)
            for e in annotator.annotate(orig, annotator.parse(cand))
        }
        best: tuple[int, int, int] | None = None
        for ref in refs:
            ref_edits = {
                (e.o_start, e.o_end, e.c_str)
                for e in annotator.annotate(orig, annotator.parse(ref))
            }
            s_tp = len(cand_edits & ref_edits)
            s_fp = len(cand_edits - ref_edits)
            s_fn = len(ref_edits - cand_edits)
            if best is None or _f05(s_tp, s_fp, s_fn) > _f05(*best):
                best = (s_tp, s_fp, s_fn)
        assert best is not None
        tp += best[0]
        fp += best[1]
        fn += best[2]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return ErrantScores(precision=p, recall=r, f05=_f05(tp, fp, fn), tp=tp, fp=fp, fn=fn)


def _f05(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    if p + r == 0:
        return 0.0
    beta2 = 0.25
    return (1 + beta2) * p * r / (beta2 * p + r)
