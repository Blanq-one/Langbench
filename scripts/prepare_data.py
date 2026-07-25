#!/usr/bin/env python3
"""Prepare datasets: verify local downloads, normalize to JSONL, write manifests.

No corpus is downloaded automatically where registration or license
acknowledgment is involved; instead this prints exact instructions and expects
files under data/corpora/<dataset>/. Nothing under data/corpora or
data/prepared is ever committed (see .gitignore and DATA_LICENSES.md).

Usage:
    uv run python scripts/prepare_data.py --dataset wi-locness
    uv run python scripts/prepare_data.py --dataset merlin --lang de
    uv run python scripts/prepare_data.py --dataset cowsl2h
    uv run python scripts/prepare_data.py --manifests   # after all prepared
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langbench.config import load_registry  # noqa: E402
from langbench.data.parsers import (  # noqa: E402
    parse_cowsl2h_dir,
    parse_m2_file,
    parse_merlin_dir,
)
from langbench.data.schema import (  # noqa: E402
    PREPARED_DIR,
    Sample,
    deterministic_sample,
    read_jsonl,
    stratified_subsample,
    write_jsonl,
    write_manifest,
)

CORPORA = Path("data/corpora")

INSTRUCTIONS = {
    "wi-locness": """\
W&I+LOCNESS (BEA-2019 shared task data, Write & Improve + LOCNESS)
  1. Go to the BEA-2019 shared task data page and download the W&I+LOCNESS
     v2.1 package.                                   # VERIFY current URL
  2. Extract so that the M2 files land at:
       data/corpora/wi-locness/m2/{A,B,C}.train.gold.bea19.m2
       data/corpora/wi-locness/m2/ABCN.dev.gold.bea19.m2   # VERIFY filenames
  License: see DATA_LICENSES.md (non-commercial research use).""",
    "merlin": """\
MERLIN corpus (German / Italian / Czech learner texts, CEFR-rated)
  1. Download merlin-text-v1.2.zip (plain-text distribution) from the Eurac
     CLARIN repository — browser required (anti-bot protection):
       https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/59
  2. Extract under data/corpora/merlin/ so this path exists:
       data/corpora/merlin/merlin-text-v1.2/meta_ltext_THs/{czech,german,italian}/*.txt
     (any nesting works; the meta_ltext_THs directory is located by search)
  License: CC BY-SA 4.0, no registration (see DATA_LICENSES.md).""",
    "cowsl2h": """\
COWS-L2H (Corpus of Written Spanish, L2/Heritage speakers)
  1. git clone https://github.com/ucdaviscl/cowsl2h data/corpora/cowsl2h
  Layout: <topic>/<term>/essays/*.txt with corrected/ sibling directories;
  second-instructor corrections carry ' (1)' in the filename.
  License: Apache 2.0 (see DATA_LICENSES.md).""",
}


def prepare_wi_locness() -> list[Sample]:
    m2dir = CORPORA / "wi-locness" / "m2"
    if not m2dir.exists():
        _die("wi-locness", m2dir)
    samples: list[Sample] = []
    found = False
    for band in ("A", "B", "C"):
        for split, pattern in (("train", f"{band}.train.gold.bea19.m2"),):
            p = m2dir / pattern
            if p.exists():
                found = True
                samples.extend(parse_m2_file(p, band, split))
    dev = m2dir / "ABCN.dev.gold.bea19.m2"
    if dev.exists():
        found = True
        # Dev file mixes bands; band info is not recoverable per sentence from
        # this file alone, so dev sentences carry no CEFR label. # DECISION:
        # English CEFR eval uses the banded train files; GEC prefers dev.
        samples.extend(parse_m2_file(dev, "N", "dev"))
    if not found:
        _die("wi-locness", m2dir)
    return samples


# The zip nests language dirs under meta_ltext_THs/ (the variant that carries
# metadata + learner text + target hypotheses) with full language names.
MERLIN_LANG_DIRS = {"de": "german", "it": "italian", "cs": "czech"}


def prepare_merlin(lang: str) -> list[Sample]:
    base = CORPORA / "merlin"
    ths_dirs = sorted(base.rglob("meta_ltext_THs")) if base.exists() else []
    if not ths_dirs:
        _die("merlin", base / "merlin-text-v1.2" / "meta_ltext_THs")
    root = ths_dirs[0] / MERLIN_LANG_DIRS[lang]
    if not root.exists():
        _die("merlin", root)
    return parse_merlin_dir(root, lang)


def prepare_cowsl2h() -> list[Sample]:
    root = CORPORA / "cowsl2h"
    if not root.exists():
        _die("cowsl2h", root)
    return parse_cowsl2h_dir(root)


def _die(dataset: str, missing: Path) -> None:
    print(f"\nMISSING: {missing}\n", file=sys.stderr)
    print(INSTRUCTIONS[dataset], file=sys.stderr)
    raise SystemExit(2)


def build_manifests() -> None:
    """Deterministic per-language sampling + committed manifests."""
    reg = load_registry()
    for lang in reg.eval.languages:
        path = PREPARED_DIR / f"{lang}.jsonl"
        samples = read_jsonl(path)
        gec_pool = [s for s in samples if s.usable_for_gec()]
        cefr_pool = [s for s in samples if s.usable_for_cefr()]
        if gec_pool:
            picked = deterministic_sample(gec_pool, reg.eval.gec.n_auto_per_language,
                                          reg.eval.gec.seed)
            print(write_manifest(lang, "gec", picked, reg.eval.gec.seed))
            judge_pool = stratified_subsample(
                picked, reg.eval.judge.n_items_per_language, reg.eval.gec.seed
            )
            print(write_manifest(lang, "feedback", judge_pool, reg.eval.gec.seed))
        else:
            print(f"note: {lang} has no GEC references; skipping gec/feedback manifests")
        if cefr_pool:
            picked = deterministic_sample(cefr_pool, reg.eval.cefr.n_auto_per_language,
                                          reg.eval.cefr.seed)
            print(write_manifest(lang, "cefr", picked, reg.eval.cefr.seed))
        else:
            print(f"note: {lang} has no CEFR labels; skipping cefr manifest "
                  "(expected for es/COWS-L2H)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["wi-locness", "merlin", "cowsl2h"])
    parser.add_argument("--lang", help="merlin only: de|it|cs")
    parser.add_argument("--manifests", action="store_true",
                        help="build sampling manifests from prepared JSONL")
    args = parser.parse_args()

    if args.manifests:
        build_manifests()
        return 0
    if not args.dataset:
        parser.error("--dataset or --manifests required")

    if args.dataset == "wi-locness":
        samples = prepare_wi_locness()
        out = PREPARED_DIR / "en.jsonl"
    elif args.dataset == "merlin":
        if args.lang not in ("de", "it", "cs"):
            parser.error("--lang de|it|cs required for merlin")
        samples = prepare_merlin(args.lang)
        out = PREPARED_DIR / f"{args.lang}.jsonl"
    else:
        samples = prepare_cowsl2h()
        out = PREPARED_DIR / "es.jsonl"

    write_jsonl(samples, out)
    n_gec = sum(s.usable_for_gec() for s in samples)
    n_cefr = sum(s.usable_for_cefr() for s in samples)
    print(f"wrote {len(samples)} samples -> {out} (gec-usable: {n_gec}, cefr-usable: {n_cefr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
