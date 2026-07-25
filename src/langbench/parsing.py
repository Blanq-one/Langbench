"""Strict structured-output parsing with one bounded repair attempt.

Flow (driven by the runner):
1. parse(text, Schema) -> Ok(model) | Failed(error_message)
2. On Failed, the runner sends ONE repair turn built by build_repair_request()
   (original prompt + raw output + the parse error) and parses again.
3. A second failure is recorded in the results DB as format_ok=False — a
   scored failure that counts against the model in the format-reliability
   column. Never silently dropped, never retried further.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from langbench.providers.base import ChatRequest

T = TypeVar("T", bound=BaseModel)


@dataclass
class Ok(Generic[T]):
    value: T


@dataclass
class Failed:
    error: str


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} region, tolerating pre/post chatter."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse(text: str, schema: type[T]) -> Ok[T] | Failed:
    candidate = _extract_json_object(_strip_fences(text))
    if candidate is None:
        return Failed("no JSON object found in output")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        return Failed(f"invalid JSON: {e}")
    try:
        return Ok(schema.model_validate(data))
    except ValidationError as e:
        # Compact, model-readable error for the repair turn.
        issues = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return Failed(f"JSON does not match required schema: {issues}")


def build_repair_request(original: ChatRequest, raw_output: str, error: str) -> ChatRequest:
    repair_user = (
        f"{original.user}\n\n"
        f"Your previous response was:\n{raw_output}\n\n"
        f"It failed validation: {error}\n"
        "Respond again with ONLY the corrected JSON object. No fences, no commentary."
    )
    return ChatRequest(
        system=original.system,
        user=repair_user,
        temperature=original.temperature,
        max_tokens=original.max_tokens,
    )
