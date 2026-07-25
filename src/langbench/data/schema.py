"""Common normalized schema + deterministic sampling.

Every dataset parser emits Sample objects; prepare_data.py writes them to
data/prepared/{lang}.jsonl (gitignored — contains corpus text). Sampling is
deterministic from a fixed seed over sorted IDs, and only the manifest
(IDs + seed + counts) is committed under data/manifests/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PREPARED_DIR = Path("data/prepared")
MANIFEST_DIR = Path("data/manifests")

SixLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
Band = Literal["A", "B", "C"]

BAND_OF: dict[str, str] = {
    "A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C", "C2": "C",
    "A": "A", "B": "B", "C": "C",
}


class Sample(BaseModel):
    id: str
    lang: str
    source_text: str
    reference_corrections: list[str] = Field(default_factory=list)
    # None for GEC-only datasets (COWS-L2H: course levels are not CEFR).
    cefr_label: str | None = None
    # 'six_level' (MERLIN) or 'band' (W&I+LOCNESS A/B/C); None when no label.
    cefr_granularity: Literal["six_level", "band"] | None = None
    split: str = "dev"

    def usable_for_gec(self) -> bool:
        return bool(self.reference_corrections)

    def usable_for_cefr(self) -> bool:
        return self.cefr_label is not None


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[Sample]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_data.py first — corpora are "
            "never committed to this repo (see DATA_LICENSES.md)."
        )
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Sample.model_validate_json(line))
            except Exception as e:  # noqa: BLE001 - re-raise with location
                raise ValueError(f"{path}:{i}: bad sample line: {e}") from e
    return out


def deterministic_sample(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """Seeded sample over ID-sorted input. Stable across runs and machines."""
    ordered = sorted(samples, key=lambda s: s.id)
    if n >= len(ordered):
        return ordered
    rng = random.Random(seed)
    return rng.sample(ordered, n)


def stratified_subsample(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """Judge subsample, stratified by CEFR label when labels exist, else plain
    deterministic sample. Keeps the 50-item judge set from skewing easy."""
    labeled = [s for s in samples if s.cefr_label]
    if not labeled:
        return deterministic_sample(samples, n, seed)
    by_label: dict[str, list[Sample]] = {}
    for s in sorted(samples, key=lambda x: x.id):
        by_label.setdefault(s.cefr_label or "_none", []).append(s)
    rng = random.Random(seed)
    picked: list[Sample] = []
    labels = sorted(by_label)
    i = 0
    while len(picked) < min(n, len(samples)):
        label = labels[i % len(labels)]
        bucket = by_label[label]
        if bucket:
            idx = rng.randrange(len(bucket))
            picked.append(bucket.pop(idx))
        i += 1
        if all(not b for b in by_label.values()):
            break
    return picked


def write_manifest(lang: str, task: str, samples: list[Sample], seed: int) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{lang}_{task}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"lang": lang, "task": task, "seed": seed, "n": len(samples),
             "ids": [s.id for s in samples]},
            f, indent=2,
        )
    return path


def load_manifest_ids(lang: str, task: str) -> list[str] | None:
    path = MANIFEST_DIR / f"{lang}_{task}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get("ids")
    assert isinstance(ids, list)
    return [str(i) for i in ids]
