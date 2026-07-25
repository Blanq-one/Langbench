"""Strict parsing + the one-repair protocol."""

from __future__ import annotations

from pydantic import BaseModel

from langbench.parsing import Failed, Ok, build_repair_request, parse
from langbench.providers.base import ChatRequest


class Toy(BaseModel):
    corrected: str


class TestParse:
    def test_clean_json(self) -> None:
        r = parse('{"corrected": "hi"}', Toy)
        assert isinstance(r, Ok) and r.value.corrected == "hi"

    def test_fenced_json(self) -> None:
        r = parse('```json\n{"corrected": "hi"}\n```', Toy)
        assert isinstance(r, Ok)

    def test_chatter_around_json(self) -> None:
        r = parse('Sure! Here you go: {"corrected": "hi"} Hope that helps!', Toy)
        assert isinstance(r, Ok) and r.value.corrected == "hi"

    def test_braces_inside_strings_do_not_confuse(self) -> None:
        r = parse('{"corrected": "use { and } carefully"}', Toy)
        assert isinstance(r, Ok) and "{" in r.value.corrected

    def test_no_json_fails(self) -> None:
        r = parse("I could not do that.", Toy)
        assert isinstance(r, Failed) and "no JSON" in r.error

    def test_invalid_json_fails(self) -> None:
        r = parse('{"corrected": }', Toy)
        assert isinstance(r, Failed) and "invalid JSON" in r.error

    def test_schema_mismatch_fails_with_field_name(self) -> None:
        r = parse('{"wrong_field": "hi"}', Toy)
        assert isinstance(r, Failed) and "corrected" in r.error


class TestRepair:
    def test_repair_request_carries_error_and_original(self) -> None:
        original = ChatRequest(system="sys", user="Fix: I is here", max_tokens=64)
        repair = build_repair_request(original, "garbage output", "no JSON object found")
        assert repair.system == "sys"
        assert "Fix: I is here" in repair.user
        assert "garbage output" in repair.user
        assert "no JSON object found" in repair.user
        assert repair.max_tokens == 64
