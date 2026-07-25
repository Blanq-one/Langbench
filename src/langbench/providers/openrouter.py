"""OpenRouter adapter. OpenAI-compatible; adds the attribution headers
OpenRouter asks free-tier users to send.

# VERIFY: header names (HTTP-Referer / X-Title) against current OpenRouter docs.
"""

from __future__ import annotations

from typing import Any

from langbench.config import ModelConfig
from langbench.providers.base import ChatRequest, OpenAICompatAdapter


class OpenRouterAdapter(OpenAICompatAdapter):
    def _build_http(
        self, model: ModelConfig, req: ChatRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        url, body, headers = super()._build_http(model, req)
        headers["HTTP-Referer"] = "https://github.com/Blanq-one/langbench"  # VERIFY
        headers["X-Title"] = "Langbench"
        return url, body, headers
