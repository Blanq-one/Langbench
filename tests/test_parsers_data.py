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


class TestMerlin:
    def test_parses_fixture(self, fixtures_dir: Path) -> None:
        s = parse_merlin_file(fixtures_dir / "merlin_de.txt", "de")
        assert s.cefr_label == "B1"
        assert s.cefr_granularity == "six_level"
        assert "gestern in die Schule" in s.source_text
        assert len(s.reference_corrections) == 1
        assert "bin gestern" in s.reference_corrections[0]
        assert "Target hypothesis" not in s.source_text  # section splitting worked

    def test_missing_rating_fails_loudly(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("Learner text:\nHallo Welt.\n", encoding="utf-8")
        with pytest.raises(ParserFormatError, match="Overall CEFR rating"):
            parse_merlin_file(f, "de")

    def test_bad_rating_value_fails_loudly(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text(
            "Overall CEFR rating: Z9\nLearner text:\nHallo.\n", encoding="utf-8"
        )
        with pytest.raises(ParserFormatError, match="Z9"):
            parse_merlin_file(f, "de")


class TestCows:
    def test_parses_tree(self, fixtures_dir: Path) -> None:
        samples = parse_cowsl2h_dir(fixtures_dir / "cowsl2h")
        # essay2 has no corrected counterpart => skipped
        assert len(samples) == 1
        s = samples[0]
        assert s.lang == "es"
        assert s.cefr_label is None  # course levels are not CEFR; GEC-only
        assert len(s.reference_corrections) == 2  # both annotators picked up

    def test_wrong_layout_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "whatever").mkdir()
        with pytest.raises(ParserFormatError, match="original"):
            parse_cowsl2h_dir(tmp_path)
