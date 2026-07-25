"""Provider adapters. build_adapter() is the single construction point the
runner uses; adding a provider means one adapter class + one line here +
config entries."""

from __future__ import annotations

import httpx

from langbench.config import ProviderConfig
from langbench.providers.base import (
    ChatRequest,
    ChatResponse,
    MissingAPIKeyError,
    ProviderAdapter,
    ProviderError,
)
from langbench.providers.gemini import GeminiAdapter
from langbench.providers.groq import GroqAdapter
from langbench.providers.mistral import MistralAdapter
from langbench.providers.openai import OpenAIAdapter
from langbench.providers.openrouter import OpenRouterAdapter

_ADAPTERS: dict[str, type[ProviderAdapter]] = {
    "groq": GroqAdapter,
    "openrouter": OpenRouterAdapter,
    "mistral": MistralAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
}


def build_adapter(provider: ProviderConfig, client: httpx.AsyncClient) -> ProviderAdapter:
    try:
        cls = _ADAPTERS[provider.name]
    except KeyError:
        raise ValueError(
            f"no adapter registered for provider {provider.name!r}; "
            f"known: {sorted(_ADAPTERS)}"
        ) from None
    return cls(provider, client)


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "MissingAPIKeyError",
    "ProviderAdapter",
    "ProviderError",
    "build_adapter",
]
