"""Task 1: Grammatical Error Correction (minimal edits)."""

from __future__ import annotations

from pydantic import BaseModel

from langbench.providers.base import ChatRequest

LANG_NAMES = {"en": "English", "de": "German", "it": "Italian", "cs": "Czech", "es": "Spanish"}


class Output(BaseModel):
    corrected: str


_SYSTEM_V1 = (
    "You are a precise grammatical error correction system for {lang_name} "
    "learner text. Apply the MINIMAL edits needed to make the text "
    "grammatically correct and natural. Do not rephrase, do not change "
    "meaning, do not add or remove content. If the text is already correct, "
    "return it unchanged. Respond with ONLY a JSON object, no markdown fences, "
    'no commentary: {{"corrected": "<corrected text>"}}'
)

_USER_V1 = "Text to correct:\n{text}"

PROMPTS: dict[str, tuple[str, str]] = {"v1": (_SYSTEM_V1, _USER_V1)}


def build_request(text: str, version: str, lang: str, max_tokens: int) -> ChatRequest:
    system, user = PROMPTS[version]
    lang_name = LANG_NAMES.get(lang, lang)
    return ChatRequest(
        system=system.format(lang_name=lang_name),
        user=user.format(text=text),
        temperature=0.0,
        max_tokens=max_tokens,
    )
