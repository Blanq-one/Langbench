"""Metric behavior tests: directionality and known values, not just smoke."""

from __future__ import annotations

import pytest

from langbench.metrics.bootstrap import ci_mean, ci_statistic, paired_delta
from langbench.metrics.cefr_metrics import collapse_to_band, score_cefr
from langbench.metrics.gleu import edit_rate, sentence_gleu

SRC = "I is very happy today ."
REF = "I am very happy today ."


class TestGleu:
    def test_perfect_correction_scores_high(self) -> None:
        assert sentence_gleu(SRC, REF, [REF]) > 0.95

    def test_unchanged_error_penalized_vs_fix(self) -> None:
        fixed = sentence_gleu(SRC, REF, [REF])
        lazy = sentence_gleu(SRC, SRC, [REF])  # kept the error
        assert fixed > lazy

    def test_source_ngram_penalty_bites(self) -> None:
        # Keeping the erroneous 'is' should cost more than a neutral miss.
        kept_error = sentence_gleu(SRC, "I is very happy today .", [REF])
        neutral_miss = sentence_gleu(SRC, "I be very happy today .", [REF])
        assert kept_error <= neutral_miss + 1e-9

    def test_multi_reference_mean(self) -> None:
        ref2 = "I am so happy today ."
        both = sentence_gleu(SRC, REF, [REF, ref2])
        only1 = sentence_gleu(SRC, REF, [REF])
        assert both <= only1  # second ref matches worse, mean pulls down

    def test_empty_candidate_is_zero(self) -> None:
        assert sentence_gleu(SRC, "", [REF]) == 0.0

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            sentence_gleu(SRC, REF, [])

    # Two reference-implementation behaviors, pinned after the 2026-07-25
    # side-by-side against cnap/gec-ranking (DECISION 36).

    def test_zero_stat_smoothing_matches_reference(self) -> None:
        # A perfect 2-token correction has NO 3-/4-grams; the reference
        # smooths those zero stats to 1 (=> log 1/1 = 0), so the score is a
        # clean 1.0. The old 1e-9 log floor crushed this to ~3e-5.
        assert sentence_gleu("a b", "a b", ["a b"]) == pytest.approx(1.0)

    def test_penalty_is_set_difference_matches_reference(self) -> None:
        # src has 'the' twice, ref once: count-aware subtraction would
        # penalize the surplus 'the'; the reference's set difference never
        # penalizes an n-gram type the reference contains.
        # Hand-computed reference stats for cand == src:
        #   n=1: 4/5, n=2: (0->1)/4, n=3: (0->1)/3, n=4: (0->1)/2, bp=1
        expected = (4 / 5 * 1 / 4 * 1 / 3 * 1 / 2) ** 0.25
        got = sentence_gleu(
            "the cat saw the dog", "the cat saw the dog", ["the cat saw a dog"]
        )
        assert got == pytest.approx(expected)


class TestEditRate:
    def test_identity_is_zero(self) -> None:
        assert edit_rate(SRC, SRC) == 0.0

    def test_one_token_change(self) -> None:
        assert edit_rate(SRC, REF) == pytest.approx(1 / 6)

    def test_total_rewrite_is_high(self) -> None:
        assert edit_rate(SRC, "completely different sentence here entirely now") > 0.8


class TestCefr:
    def test_perfect_six_level(self) -> None:
        labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        s = score_cefr(labels, labels, "six_level")
        assert s.accuracy == 1.0
        assert s.adjacent_accuracy == 1.0
        assert s.qwk == pytest.approx(1.0)
        assert s.macro_f1 == pytest.approx(1.0)

    def test_adjacent_counts_off_by_one(self) -> None:
        s = score_cefr(["A2", "B2"], ["A1", "B1"], "six_level")
        assert s.accuracy == 0.0
        assert s.adjacent_accuracy == 1.0

    def test_band_collapsing(self) -> None:
        assert collapse_to_band("B1") == "B"
        s = score_cefr(["B1", "B2", "C1"], ["B", "B", "C"], "band")
        assert s.accuracy == 1.0  # B1 and B2 both collapse to gold band B

    def test_unparseable_is_maximally_wrong(self) -> None:
        s = score_cefr(["UNPARSEABLE"], ["C2"], "six_level")
        assert s.accuracy == 0.0 and s.adjacent_accuracy == 0.0

    def test_granularities_never_mix(self) -> None:
        with pytest.raises(ValueError, match="never mix"):
            score_cefr(["B1"], ["B1"], "band")  # six-level gold on band call

    def test_constant_predictor_gets_zero_qwk(self) -> None:
        golds = ["A1", "A2", "B1", "B2", "C1", "C2"] * 5
        preds = ["B1"] * len(golds)
        s = score_cefr(preds, golds, "six_level")
        assert abs(s.qwk) < 0.15  # ~0 for an uninformative predictor


class TestBootstrap:
    def test_ci_contains_point(self) -> None:
        ci = ci_mean([0.5, 0.6, 0.7, 0.55, 0.65] * 10, seed=1)
        assert ci.lo <= ci.point <= ci.hi
        assert ci.n == 50

    def test_ci_narrows_with_n(self) -> None:
        wide = ci_mean([0.0, 1.0] * 5, seed=1)
        narrow = ci_mean([0.0, 1.0] * 500, seed=1)
        assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)

    def test_paired_delta_detects_clear_gap(self) -> None:
        a = [0.8 + 0.01 * (i % 3) for i in range(100)]
        b = [0.5 + 0.01 * (i % 3) for i in range(100)]
        d = paired_delta(a, b, seed=1)
        assert d.excludes_zero and d.delta == pytest.approx(0.3, abs=0.01)

    def test_paired_delta_requires_alignment(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            paired_delta([1.0], [1.0, 2.0])

    def test_ci_statistic_runs_on_qwk_like_stat(self) -> None:
        pairs = [("B1", "B1"), ("A1", "A2"), ("C1", "C1")] * 10

        def acc(preds: list[str], golds: list[str]) -> float:
            return sum(p == g for p, g in zip(preds, golds, strict=True)) / len(preds)

        ci = ci_statistic(pairs, acc, seed=1, iters=200)
        assert ci.lo <= ci.point <= ci.hi
        assert ci.point == pytest.approx(2 / 3, abs=0.01)
