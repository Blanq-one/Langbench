"""Groq adapter. Groq exposes an OpenAI-compatible chat completions API.

# VERIFIED 2026-07-24 (live smoke call): /chat/completions matches the OpenAI
# shape; usage.prompt_tokens / usage.completion_tokens present as coded.
"""

from __future__ import annotations

from langbench.providers.base import OpenAICompatAdapter


class GroqAdapter(OpenAICompatAdapter):
    pass
