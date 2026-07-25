"""Provider adapter base.

Adding a provider is a config change plus (at most) one small adapter class;
four of the five providers here are OpenAI-compatible and share one subclass.

Responsibilities of an adapter:
- translate ChatRequest -> provider wire format and back
- retry 429/5xx/network errors with exponential backoff + jitter, honoring
  Retry-After when present, bounded at MAX_ATTEMPTS
- surface token usage and wall-clock latency

NOT responsibilities of an adapter (they live in the runner):
- rate limiting (RateLimiter, per model)
- caching (RawCache)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from langbench.config import ModelConfig, ProviderConfig

log = logging.getLogger("langbench.providers")

MAX_ATTEMPTS = 5
BASE_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ProviderError(Exception):
    """Non-retryable or retries-exhausted provider failure."""


class MissingAPIKeyError(ProviderError):
    pass


class ChatRequest(BaseModel):
    system: str | None = None
    user: str
    temperature: float = 0.0
    max_tokens: int = 1024


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    raw: dict[str, Any]

    def to_cacheable(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_cacheable(cls, d: dict[str, Any]) -> ChatResponse:
        return cls.model_validate(d)


class ProviderAdapter(ABC):
    def __init__(
        self,
        provider: ProviderConfig,
        client: httpx.AsyncClient,
        sleeper: Any = None,
    ) -> None:
        self.provider = provider
        self.client = client
        self._sleep = sleeper or asyncio.sleep
        key = provider.api_key()
        if key is None:
            raise MissingAPIKeyError(
                f"provider {provider.name!r}: env var {provider.api_key_env} is empty. "
                "This adapter should never have been constructed; check enabled resolution."
            )
        self.api_key: str = key

    @abstractmethod
    def _build_http(
        self, model: ModelConfig, req: ChatRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Return (url, json_body, headers) for one chat call."""

    @abstractmethod
    def _parse(self, body: dict[str, Any]) -> tuple[str, int | None, int | None]:
        """Return (text, prompt_tokens, completion_tokens) from a 2xx body."""

    async def chat(self, model: ModelConfig, req: ChatRequest) -> ChatResponse:
        url, json_body, headers = self._build_http(model, req)
        last_err: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.monotonic()
            try:
                resp = await self.client.post(url, json=json_body, headers=headers, timeout=120.0)
            except httpx.HTTPError as e:
                last_err = e
                await self._backoff(attempt, retry_after=None, reason=f"network: {e!r}")
                continue
            latency_ms = (time.monotonic() - start) * 1000.0
            if resp.status_code in RETRYABLE_STATUS:
                last_err = ProviderError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = _parse_retry_after(resp)
                await self._backoff(attempt, retry_after, reason=f"HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    f"{self.provider.name}/{model.model_id}: HTTP {resp.status_code} "
                    f"(non-retryable): {resp.text[:500]}"
                )
            body = resp.json()
            try:
                text, ptok, ctok = self._parse(body)
            except (KeyError, IndexError, TypeError) as e:
                raise ProviderError(
                    f"{self.provider.name}/{model.model_id}: unexpected response shape "
                    f"({e!r}). First 500 chars: {str(body)[:500]}"
                ) from e
            return ChatResponse(
                text=text,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                latency_ms=latency_ms,
                raw=body,
            )
        raise ProviderError(
            f"{self.provider.name}/{model.model_id}: retries exhausted after "
            f"{MAX_ATTEMPTS} attempts; last error: {last_err!r}"
        )

    async def _backoff(self, attempt: int, retry_after: float | None, reason: str) -> None:
        if attempt >= MAX_ATTEMPTS:
            return
        if retry_after is not None:
            delay = min(retry_after, MAX_BACKOFF_S)
        else:
            delay = min(BASE_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
            delay *= 0.5 + random.random()  # jitter in [0.5x, 1.5x)
        log.warning(
            "retryable failure (%s), attempt %d/%d, sleeping %.1fs",
            reason,
            attempt,
            MAX_ATTEMPTS,
            delay,
        )
        await self._sleep(delay)


def _parse_retry_after(resp: httpx.Response) -> float | None:
    val = resp.headers.get("retry-after")
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


class OpenAICompatAdapter(ProviderAdapter):
    """Chat-completions wire format shared by Groq, OpenRouter, Mistral, OpenAI."""

    def _build_http(
        self, model: ModelConfig, req: ChatRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        messages: list[dict[str, str]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.user})
        body: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return f"{self.provider.base_url}/chat/completions", body, headers

    def _parse(self, body: dict[str, Any]) -> tuple[str, int | None, int | None]:
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return (
            str(text),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
