"""Dataset parsers against the committed hand-made fixtures. These pin the
DOCUMENTED formats; when real files differ, Claude Code updates parser and
fixture together (HANDOFF item)."""

from __future__ import annotations

from pathlib import Path

import pytest

from langbench.data.parsers import (
    ParserFormatError,
    parse_cowsl2h_dir,
    parse_m2_file,
    parse_merlin_file,
)


class TestM2:
    def test_parses_blocks(self, fixtures_dir: Path) -> None:
        samples = parse_m2_file(fixtures_dir / "sample.m2", "B", "dev")
        assert len(samples) == 3
        assert all(s.lang == "en" and s.cefr_label == "B" for s in samples)
        assert all(s.cefr_granularity == "band" for s in samples)

    def test_edit_application(self, fixtures_dir: Path) -> None:
        samples = parse_m2_file(fixtures_dir / "sample.m2", "B", "dev")
        s0 = samples[0]
        assert s0.source_text == "I is going to school yesterday ."
        # Annotator 0 and annotator 1 give two distinct references.
        assert "I am going to school yesterday ." in s0.reference_corrections
        assert "I was went to school yesterday ." in s0.reference_corrections
        assert len(s0.reference_corrections) == 2

    def test_noop_means_source_is_reference(self, fixtures_dir: Path) -> None:
        samples = parse_m2_file(fixtures_dir / "sample.m2", "B", "dev")
        s2 = samples[2]
        assert s2.reference_corrections == ["This sentence is fine ."]

    def test_native_band_has_no_label(self, fixtures_dir: Path) -> None:
        samples = parse_m2_file(fixtures_dir / "sample.m2", "N", "dev")
        assert all(s.cefr_label is None for s in samples)

    def test_garbage_fails_loudly(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.m2"
        bad.write_text("this is not an m2 file\n", encoding="utf-8")
        with pytest.raises(ParserFormatError, match="expected first line"):
            parse_m2_file(bad, "A", "dev")


def _merlin_text(
    rating: str = "B1",
    learner: str = "Hallo Welt.",
    th1_block: str = "Target hypothesis 1:\n\nHallo , Welt .",
) -> str:
    """Minimal file in the real v1.2 block structure."""
    return (
        "METADATA\n\nGeneral:\nAuthor ID: t-001\n\nRating:\n"
        f"Overall CEFR rating: {rating}\n\n----------------\n\n"
        f"Learner text:\n\n{learner}\n\n----------------\n\n"
        f"{th1_block}\n\n----------------\n\nNo target hypothesis 2 available.\n"
    )


class TestMerlin:
    def test_parses_fixture(self, fixtures_dir: Path) -> None:
        s = parse_merlin_file(fixtures_dir / "merlin_de.txt", "de")
        assert s is not None
        assert s.cefr_label == "B1"
        assert s.cefr_granularity == "six_level"
        assert "gestern in die Schule" in s.source_text
        assert "Target hypothesis" not in s.source_text  # block splitting worked
        assert "----" not in s.source_text  # separators never leak into text
        # TH1 is the only reference; TH2 (appropriateness) is excluded. DECISION
        assert len(s.reference_corrections) == 1
        assert "bin gestern" in s.reference_corrections[0]
        assert "wunderschöner" not in s.reference_corrections[0]

    def test_unrated_and_empty_ratings_give_no_label(self, tmp_path: Path) -> None:
        for raw in ("unrated", "EMPTY"):
            f = tmp_path / f"{raw}.txt"
            f.write_text(_merlin_text(rating=raw), encoding="utf-8")
            s = parse_merlin_file(f, "it")
            assert s is not None
            assert s.cefr_label is None and s.cefr_granularity is None
            assert s.reference_corrections  # still GEC-usable via TH1

    def test_th1_sentinel_means_no_references(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text(
            _merlin_text(th1_block="No target hypothesis 1 available."), encoding="utf-8"
        )
        s = parse_merlin_file(f, "cs")
        assert s is not None
        assert s.reference_corrections == []

    def test_empty_learner_text_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text(_merlin_text(learner=""), encoding="utf-8")
        assert parse_merlin_file(f, "de") is None

    def test_missing_rating_fails_loudly(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text(
            "METADATA\nno rating here\n\n----------------\n\nLearner text:\n\nHallo Welt.\n",
            encoding="utf-8",
        )
        with pytest.raises(ParserFormatError, match="Overall CEFR rating"):
            parse_merlin_file(f, "de")

    def test_bad_rating_value_fails_loudly(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text(_merlin_text(rating="Z9"), encoding="utf-8")
        with pytest.raises(ParserFormatError, match="Z9"):
            parse_merlin_file(f, "de")

    def test_flat_file_fails_loudly(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("Overall CEFR rating: B1\nLearner text:\nHallo.\n", encoding="utf-8")
        with pytest.raises(ParserFormatError, match="dash-separated"):
            parse_merlin_file(f, "de")


class TestCows:
    def test_parses_tree(self, fixtures_dir: Path) -> None:
        samples = parse_cowsl2h_dir(fixtures_dir / "cowsl2h")
        # essay 999 has no correction; term SU17 has no corrected/ dir => both skipped
        assert len(samples) == 1
        s = samples[0]
        assert s.lang == "es"
        assert s.id == "cows-vacation_F17_essays_120.F17_Vacation.txt"
        assert s.cefr_label is None  # course levels are not CEFR; GEC-only
        # Pairing is by participant-id prefix despite the case difference
        # (essays/...F17_Vacation.txt vs corrected/...F17_vacation.corrected.txt),
        # and the ' (1)' second-instructor correction becomes a second reference.
        assert len(s.reference_corrections) == 2
        assert any("A mí me gusta el clima" in r for r in s.reference_corrections)
        # The misfiled annotation file in corrected/ is excluded: error markup
        # must never become a GEC reference. And the byte-identical typo'd
        # duplicate ('corrrected') collapses by content-dedupe — still 2 refs.
        assert not any("<pr:" in r for r in s.reference_corrections)

    def test_wrong_layout_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "whatever").mkdir()
        with pytest.raises(ParserFormatError, match="essays"):
            parse_cowsl2h_dir(tmp_path)
