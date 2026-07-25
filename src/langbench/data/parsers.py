"""Dataset parsers: W&I+LOCNESS (en), MERLIN (de/it/cs), COWS-L2H (es).

These are the single most likely Claude Code fix-up points in the repo: they
were written against DOCUMENTED formats without live files. Every parser
therefore (a) names the exact structure it expects in its docstring, and
(b) raises ParserFormatError with the offending path + line and what was
expected, never failing silently or producing partial garbage.

# VERIFY (all of this module): run scripts/prepare_data.py against real
# downloads and fix these parsers first.
"""

from __future__ import annotations

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
# MERLIN. Expected layout (plain-text distribution):
#   <root>/<lang_dir>/*.txt where each file has a metadata header containing
#   a line like 'Overall CEFR rating: B1', a 'Learner text:' section, and —
#   in the target-hypothesis variant — a 'Target hypothesis 1:' section.
# lang_dir naming and section headers are the top VERIFY item in this module.
# ---------------------------------------------------------------------------

_MERLIN_RATING_PREFIX = "Overall CEFR rating:"
_MERLIN_LEARNER_HEADER = "Learner text:"
_MERLIN_TH1_HEADER = "Target hypothesis 1:"
_MERLIN_SIX = {"A1", "A2", "B1", "B2", "C1", "C2"}


def parse_merlin_file(path: Path, lang: str) -> Sample:
    text = path.read_text(encoding="utf-8", errors="replace")
    rating: str | None = None
    for i, ln in enumerate(text.split("\n"), 1):
        if ln.strip().startswith(_MERLIN_RATING_PREFIX):
            candidate = ln.split(":", 1)[1].strip().upper()
            if candidate not in _MERLIN_SIX:
                raise ParserFormatError(
                    path, f"CEFR rating {candidate!r} not one of {sorted(_MERLIN_SIX)}", i
                )
            rating = candidate
            break
    if rating is None:
        raise ParserFormatError(
            path, f"no line starting with {_MERLIN_RATING_PREFIX!r} found in metadata header"
        )
    learner = _merlin_section(path, text, _MERLIN_LEARNER_HEADER)
    th1: str | None
    try:
        th1 = _merlin_section(path, text, _MERLIN_TH1_HEADER)
    except ParserFormatError:
        th1 = None  # target hypotheses ship separately in some distributions
    return Sample(
        id=f"merlin-{lang}-{path.stem}",
        lang=lang,
        source_text=learner,
        reference_corrections=[th1] if th1 else [],
        cefr_label=rating,
        cefr_granularity="six_level",
        split="dev",
    )


def _merlin_section(path: Path, text: str, header: str) -> str:
    if header not in text:
        raise ParserFormatError(path, f"section header {header!r} not found")
    after = text.split(header, 1)[1]
    # Section runs until the next 'Something:' header line or EOF.
    lines: list[str] = []
    for ln in after.split("\n"):
        stripped = ln.strip()
        if stripped.endswith(":") and len(stripped.split()) <= 5 and lines:
            break
        lines.append(ln)
    body = "\n".join(lines).strip()
    if not body:
        raise ParserFormatError(path, f"section {header!r} is empty")
    return body


def parse_merlin_dir(root: Path, lang: str) -> list[Sample]:
    files = sorted(root.glob("*.txt"))
    if not files:
        raise ParserFormatError(
            root, "no *.txt files found; expected MERLIN plain-text exports here"
        )
    return [parse_merlin_file(p, lang) for p in files]


# ---------------------------------------------------------------------------
# COWS-L2H (Spanish, GEC-only). Expected layout from the GitHub repo:
#   <root>/<course_dir>/original/<essay>.txt
#   <root>/<course_dir>/corrected/<essay>.txt   (annotator 1)
#   optionally .../corrected2/<essay>.txt       (annotator 2)
# Essays pair by identical filename. Course levels are NOT CEFR labels, so
# cefr_label stays None (Spanish is GEC-only; see eval.yaml comment).
# ---------------------------------------------------------------------------

def parse_cowsl2h_dir(root: Path) -> list[Sample]:
    originals = sorted(root.rglob("original/*.txt"))
    if not originals:
        raise ParserFormatError(
            root,
            "no files matching **/original/*.txt; expected the COWS-L2H repo layout "
            "with original/ and corrected/ sibling directories per course",
        )
    samples: list[Sample] = []
    for orig in originals:
        course_dir = orig.parent.parent
        refs: list[str] = []
        for corr_name in ("corrected", "corrected2"):
            cand = course_dir / corr_name / orig.name
            if cand.exists():
                body = cand.read_text(encoding="utf-8", errors="replace").strip()
                if body:
                    refs.append(body)
        if not refs:
            continue  # uncorrected essays are unusable for GEC; skip, count later
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
            root, "found original essays but zero with a corrected/ counterpart"
        )
    return samples
