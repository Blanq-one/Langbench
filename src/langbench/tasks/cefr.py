"""Task 2: CEFR level classification.

Models ALWAYS predict a full six-level label (A1..C2). Scoring collapses the
prediction to A/B/C bands where the gold labels are band-granular (English
W&I). Collapsing is a metrics-time concern, not a prompting concern —
prompting for bands on some languages would make outputs incomparable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from langbench.providers.base import ChatRequest
from langbench.tasks.gec import LANG_NAMES


class Output(BaseModel):
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]


_SYSTEM_V1 = (
    "You are an expert CEFR rater for {lang_name} learner writing. Rate the "
    "overall proficiency level of the text on the CEFR scale. Respond with "
    "ONLY a JSON object, no markdown fences, no commentary: "
    '{{"level": "<one of A1, A2, B1, B2, C1, C2>"}}'
)

_USER_V1 = "Learner text to rate:\n{text}"

PROMPTS: dict[str, tuple[str, str]] = {"v1": (_SYSTEM_V1, _USER_V1)}


def build_request(text: str, version: str, lang: str, max_tokens: int) -> ChatRequest:
    system, user = PROMPTS[version]
    return ChatRequest(
        system=system.format(lang_name=LANG_NAMES.get(lang, lang)),
        user=user.format(text=text),
        temperature=0.0,
        max_tokens=max_tokens,
    )
