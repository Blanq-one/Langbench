"""Task 3 (judge-only subsample): Pangea-style structured feedback.

Model output: identified error spans, closed-set category, correction, and a
one-sentence learner-friendly explanation per error, plus a fully corrected
version. The Gemini judge scores this against gold corrections (judge.py).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from langbench.providers.base import ChatRequest
from langbench.tasks.gec import LANG_NAMES

Category = Literal["grammar", "spelling", "punctuation", "word_choice", "word_order", "other"]


class FeedbackItem(BaseModel):
    span: str = Field(description="exact erroneous text from the input")
    category: Category
    correction: str
    explanation: str = Field(description="one learner-friendly sentence")


class Output(BaseModel):
    errors: list[FeedbackItem]
    corrected: str


_SYSTEM_V1 = (
    "You are a supportive {lang_name} language tutor. Analyze the learner's "
    "text and produce structured feedback. For each genuine error: the exact "
    "erroneous span copied from the input, a category (one of grammar, "
    "spelling, punctuation, word_choice, word_order, other), the correction, "
    "and ONE short, encouraging, learner-friendly sentence explaining it. Do "
    "not invent errors; if the text is fully correct, return an empty errors "
    "list. Respond with ONLY a JSON object, no markdown fences, matching: "
    '{{"errors": [{{"span": "...", "category": "...", "correction": "...", '
    '"explanation": "..."}}], "corrected": "<full corrected text>"}}'
)

_USER_V1 = "Learner text:\n{text}"

PROMPTS: dict[str, tuple[str, str]] = {"v1": (_SYSTEM_V1, _USER_V1)}


def build_request(text: str, version: str, lang: str, max_tokens: int) -> ChatRequest:
    system, user = PROMPTS[version]
    return ChatRequest(
        system=system.format(lang_name=LANG_NAMES.get(lang, lang)),
        user=user.format(text=text),
        temperature=0.0,
        max_tokens=max_tokens,
    )
