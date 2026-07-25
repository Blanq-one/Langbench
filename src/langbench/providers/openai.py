"""OpenAI adapter (gpt-4o-mini reference point; disabled by default in
config/models.yaml because it spends real money)."""

from __future__ import annotations

from langbench.providers.base import OpenAICompatAdapter


class OpenAIAdapter(OpenAICompatAdapter):
    pass
