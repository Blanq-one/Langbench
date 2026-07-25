"""Dataset parsers: W&I+LOCNESS (en), MERLIN (de/it/cs), COWS-L2H (es).

These are the single most likely Claude Code fix-up points in the repo: they
were written against DOCUMENTED formats without live files. Every parser
therefore (a) names the exact structure it expects in its docstring, and
(b) raises ParserFormatError with the offending path + line and what was
expected, never failing silently or producing partial garbage.

# MERLIN: VERIFIED 2026-07-24 against the real v1.2 plain-text download.
# COWS-L2H: VERIFIED 2026-07-25 against the real GitHub repo.
# W&I+LOCNESS: still # VERIFY against the real download.
"""

from __future__ import annotations

import re
from pathlib import Path

from langbench.data.schema import Sample

M2_LEVELS = {"A", "B", "C", "N"}  # N = native (LOCNESS), no CEFR label


class ParserFormatError(ValueError):
    def __init__(self, path: Path, detail: str, line_no: int | None = None) -> None:
        loc = f"{path}:{line_no}" if line_no else str(path)
        super().__init__(
            f"{loc}: {detail}\n"
            "This parser was written against the documented format without a live "
            "file; if the real file differs, fix the parser (see HANDOFF_TO_CLAUDE_CODE.md)."
        )


# ---------------------------------------------------------------------------
# W&I+LOCNESS (BEA-2019). M2 format:
#   S <space-tokenized source sentence>
#   A <start> <end>|||<type>|||<correction>|||REQUIRED|||-NONE-|||<annotator>
#   (blank line between sentences)
# CEFR band comes from the filename: A.train.gold.bea19.m2, B.dev..., etc.
# 'noop' edit type means the sentence is already correct.
# ---------------------------------------------------------------------------

def parse_m2_file(path: Path, band: str, split: str) -> list[Sample]:
    if band not in M2_LEVELS:
        raise ParserFormatError(path, f"band {band!r} not in {sorted(M2_LEVELS)}")
    text = path.read_text(encoding="utf-8")
    samples: list[Sample] = []
    blocks = [b for b in text.split("\n\n") if b.strip()]
    for bi, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if not lines[0].startswith("S "):
            raise ParserFormatError(
                path, f"block {bi}: expected first line to start with 'S ', "
                f"got: {lines[0][:80]!r}"
            )
        src_tokens = lines[0][2:].split(" ")
        # Collect edits per annotator id; apply annotator 0's edits as the
        # primary reference, others as additional references.
        edits_by_annotator: dict[int, list[tuple[int, int, str]]] = {}
        for ln in lines[1:]:
            if not ln.startswith("A "):
                raise ParserFormatError(
                    path, f"block {bi}: expected 'A ' edit line, got: {ln[:80]!r}"
                )
            fields = ln[2:].split("|||")
            if len(fields) < 6:
                raise ParserFormatError(
                    path, f"block {bi}: edit line has {len(fields)} '|||' fields, "
                    f"expected >= 6: {ln[:120]!r}"
                )
            span, etype, corr = fields[0], fields[1], fields[2]
            annot = int(fields[5])
            if etype == "noop":
                edits_by_annotator.setdefault(annot, [])
                continue
            try:
                start_s, end_s = span.split(" ")
                start, end = int(start_s), int(end_s)
            except ValueError as e:
                raise ParserFormatError(
                    path, f"block {bi}: bad edit span {span!r}"
                ) from e
            edits_by_annotator.setdefault(annot, []).append((start, end, corr))

        refs: list[str] = []
        for annot in sorted(edits_by_annotator):
            refs.append(_apply_edits(src_tokens, edits_by_annotator[annot]))
        if not refs:
            refs = [" ".join(src_tokens)]  # unannotated => source is the reference

        samples.append(
            Sample(
                id=f"wi-{split}-{band}-{bi:05d}",
                lang="en",
                source_text=" ".join(src_tokens),
                reference_corrections=refs,
                cefr_label=band if band != "N" else None,
                cefr_granularity="band" if band != "N" else None,
                split=split,
            )
        )
    if not samples:
        raise ParserFormatError(path, "no sentence blocks found in M2 file")
    return samples


def _apply_edits(tokens: list[str], edits: list[tuple[int, int, str]]) -> str:
    out = list(tokens)
    # Apply right-to-left so indices stay valid.
    for start, end, corr in sorted(edits, key=lambda e: e[0], reverse=True):
        if not (0 <= start <= end <= len(out)):
            raise ValueError(f"edit span ({start},{end}) outside sentence of {len(out)} tokens")
        replacement = corr.split(" ") if corr and corr != "-NONE-" else []
        out[start:end] = replacement
    return " ".join(out)


# ---------------------------------------------------------------------------
# MERLIN v1.2 plain-text distribution (merlin-text-v1.2.zip). Format VERIFIED
# 2026-07-24 against the real CLARIN download.
# One file per learner text: meta_ltext_THs/{czech,german,italian}/<id>.txt
# Blocks separated by lines of dashes ('----------------'):
#   METADATA block containing 'Overall CEFR rating: <A1..C2 | EMPTY | unrated>'
#   'Learner text:' block (original formatting, may span many lines)
#   'Target hypothesis 1:' block, or the literal sentinel line
#     'No target hypothesis 1 available.'
#   'Target hypothesis 2:' block, or its sentinel likewise
# EMPTY/unrated ratings => cefr_label None (still GEC-usable when TH1 exists).
# DECISION: TH1 is the only GEC reference. TH1 is FALKO's minimal-correctness
# hypothesis; TH2 targets appropriateness and would reward stylistic rewrites.
# ---------------------------------------------------------------------------

_MERLIN_RATING_PREFIX = "Overall CEFR rating:"
_MERLIN_LEARNER_HEADER = "Learner text:"
_MERLIN_TH1_HEADER = "Target hypothesis 1:"
_MERLIN_SIX = {"A1", "A2", "B1", "B2", "C1", "C2"}
_MERLIN_UNRATED = {"EMPTY", "UNRATED"}  # both appear in v1.2 ('unrated' lowercased)
_MERLIN_SEPARATOR = re.compile(r"^-{4,}\s*$", re.MULTILINE)


def parse_merlin_file(path: Path, lang: str) -> Sample | None:
    """One MERLIN text file -> Sample, or None for structurally valid but
    unusable files (empty learner text). Structural surprises raise."""
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in _MERLIN_SEPARATOR.split(text) if b.strip()]
    if len(blocks) < 2:
        raise ParserFormatError(
            path,
            f"expected dash-separated METADATA + section blocks, got {len(blocks)} block(s)",
        )

    rating: str | None = None
    found_rating_line = False
    for i, ln in enumerate(blocks[0].split("\n"), 1):
        if ln.strip().startswith(_MERLIN_RATING_PREFIX):
            found_rating_line = True
            candidate = ln.split(":", 1)[1].strip().upper()
            if candidate in _MERLIN_SIX:
                rating = candidate
            elif candidate in _MERLIN_UNRATED:
                rating = None  # no CEFR label; file may still serve GEC
            else:
                raise ParserFormatError(
                    path,
                    f"CEFR rating {candidate!r} not one of {sorted(_MERLIN_SIX)} "
                    f"or {sorted(_MERLIN_UNRATED)}",
                    i,
                )
            break
    if not found_rating_line:
        raise ParserFormatError(
            path, f"no line starting with {_MERLIN_RATING_PREFIX!r} in the METADATA block"
        )

    learner = _merlin_block_body(blocks[1:], _MERLIN_LEARNER_HEADER)
    if learner is None:
        raise ParserFormatError(
            path, f"no block starting with {_MERLIN_LEARNER_HEADER!r} after the separator"
        )
    if not learner:
        return None  # empty learner text: valid file, unusable sample
    th1 = _merlin_block_body(blocks[1:], _MERLIN_TH1_HEADER)  # sentinel block => None

    return Sample(
        id=f"merlin-{lang}-{path.stem}",
        lang=lang,
        source_text=learner,
        reference_corrections=[th1] if th1 else [],
        cefr_label=rating,
        cefr_granularity="six_level" if rating else None,
        split="dev",
    )


def _merlin_block_body(blocks: list[str], header: str) -> str | None:
    """Body of the block starting with `header`, or None when no such block
    exists (e.g. the 'No target hypothesis N available.' sentinel instead)."""
    for b in blocks:
        if b.startswith(header):
            return b[len(header):].strip()
    return None


def parse_merlin_dir(root: Path, lang: str) -> list[Sample]:
    files = sorted(root.glob("*.txt"))
    if not files:
        raise ParserFormatError(
            root, "no *.txt files found; expected meta_ltext_THs/<language>/*.txt"
        )
    samples = [s for p in files if (s := parse_merlin_file(p, lang)) is not None]
    if not samples:
        raise ParserFormatError(root, "every file parsed to an unusable (empty) sample")
    return samples


# ---------------------------------------------------------------------------
# COWS-L2H (Spanish, GEC-only). Layout VERIFIED 2026-07-25 against the real
# repo (github.com/ucdaviscl/cowsl2h):
#   <topic>/<term>/essays/<pid>.<TERM>_<Topic>.txt           (learner original)
#   <topic>/<term>/corrected/<pid>.<term>_<topic>.corrected.txt
#   second-instructor corrections: same name with ' (1)' appended
# Pairing is by the participant-id prefix (before the first '.') within one
# term directory: essay and corrected filenames differ in case and suffix, so
# exact-name matching does not work. corrected/ holds plain corrected text
# (holistic instructor corrections); annotated/ holds per-error-type
# annotations and is NOT a GEC reference. Course levels are NOT CEFR labels,
# so cefr_label stays None (Spanish is GEC-only; see eval.yaml comment).
# ---------------------------------------------------------------------------

def parse_cowsl2h_dir(root: Path) -> list[Sample]:
    essays = sorted(root.rglob("essays/*.txt"))
    if not essays:
        raise ParserFormatError(
            root,
            "no files matching **/essays/*.txt; expected the COWS-L2H repo layout "
            "<topic>/<term>/essays/ with corrected/ sibling directories",
        )
    samples: list[Sample] = []
    for orig in essays:
        corr_dir = orig.parent.parent / "corrected"
        if not corr_dir.is_dir():
            continue  # this whole term has no corrections
        pid = orig.name.split(".", 1)[0]
        refs: list[str] = []
        for cand in sorted(corr_dir.iterdir()):
            # corrected/ dirs in the real repo also contain misfiled
            # annotation files ('anotated', 'annnotated', ...) full of error
            # markup. A real correction's filename always contains 'corr',
            # including the repo's typo variants ('correcteed', 'correced',
            # 'correted', 'corrected(1)').
            if (
                cand.suffix == ".txt"
                and cand.name.startswith(f"{pid}.")
                and "corr" in cand.name.lower()
            ):
                body = cand.read_text(encoding="utf-8", errors="replace").strip()
                # Dedupe by content: the repo contains byte-identical
                # duplicates under typo'd names ('corrrected', 'corrected.').
                if body and body not in refs:
                    refs.append(body)
        if not refs:
            continue  # uncorrected essays are unusable for GEC; skip
        src = orig.read_text(encoding="utf-8", errors="replace").strip()
        if not src:
            continue
        rel = orig.relative_to(root).as_posix().replace("/", "_")
        samples.append(
            Sample(
                id=f"cows-{rel}",
                lang="es",
                source_text=src,
                reference_corrections=refs,
                cefr_label=None,
                cefr_granularity=None,
                split="dev",
            )
        )
    if not samples:
        raise ParserFormatError(
            root, "found essays but zero with a corrected/ counterpart"
        )
    return samples
