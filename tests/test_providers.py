"""Offline adapter wire-format tests: no network, canned bodies only.

Pins the two live-integration hardenings from 2026-07-24:
- OpenAI-compat adapters merge ModelConfig.extra_body into the request body
  (DECISION 32: Qwen reasoning_format=hidden).
- The Gemini adapter joins ALL non-thought text parts instead of reading
  parts[0] (DECISION 33: thoughtSignature / split-part tail risk).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from langbench.config import ModelConfig, Pricing, ProviderConfig, RateLimit
from langbench.providers.base import ChatRequest, OpenAICompatAdapter
from langbench.providers.gemini import GeminiAdapter


def _model(**overrides: Any) -> ModelConfig:
    base: dict[str, Any] = dict(
        key="fake/model",
        provider="fake",
        model_id="fake-model",
        display_name="Fake",
        enabled=True,
        rate_limit=RateLimit(rpm=1, rpd=1),
        pricing=Pricing(input_per_mtok=0.0, output_per_mtok=0.0),
        max_output_tokens=64,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="fake",
        base_url="https://fake.invalid/v1",
        api_key_env="FAKE_ADAPTER_TEST_KEY",
        enabled=True,
    )


class _CompatAdapter(OpenAICompatAdapter):
    pass


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("FAKE_ADAPTER_TEST_KEY", "not-a-real-key")
    async with httpx.AsyncClient() as c:
        yield c


class TestExtraBody:
    async def test_extra_body_merged_into_wire_body(self, client: httpx.AsyncClient) -> None:
        adapter = _CompatAdapter(_provider(), client)
        m = _model(extra_body={"reasoning_format": "hidden"})
        _, body, _ = adapter._build_http(m, ChatRequest(user="hi"))
        assert body["reasoning_format"] == "hidden"
        assert body["model"] == "fake-model"  # standard fields intact

    async def test_empty_extra_body_changes_nothing(self, client: httpx.AsyncClient) -> None:
        adapter = _CompatAdapter(_provider(), client)
        _, body, _ = adapter._build_http(_model(), ChatRequest(user="hi"))
        assert set(body) == {"model", "messages", "temperature", "max_tokens"}


class TestGeminiMultiPart:
    def _body(self, parts: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "candidates": [{"content": {"parts": parts, "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 17, "candidatesTokenCount": 5},
        }

    async def test_joins_all_text_parts(self, client: httpx.AsyncClient) -> None:
        adapter = GeminiAdapter(_provider(), client)
        body = self._body([{"text": "I am "}, {"text": "happy today."}])
        text, ptok, ctok = adapter._parse(body)
        assert text == "I am happy today."
        assert (ptok, ctok) == (17, 5)

    async def test_skips_thought_and_non_text_parts(self, client: httpx.AsyncClient) -> None:
        adapter = GeminiAdapter(_provider(), client)
        body = self._body([
            {"text": "internal chain of thought", "thought": True},
            {"thoughtSignature": "abc123"},
            {"text": "I am happy today.", "thoughtSignature": "def456"},
        ])
        text, _, _ = adapter._parse(body)
        assert text == "I am happy today."

    async def test_no_text_parts_raises_for_shape_error_path(
        self, client: httpx.AsyncClient
    ) -> None:
        adapter = GeminiAdapter(_provider(), client)
        body = self._body([{"thoughtSignature": "only-a-signature"}])
        with pytest.raises(KeyError):
            adapter._parse(body)
