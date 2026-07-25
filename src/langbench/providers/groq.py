"""Groq adapter. Groq exposes an OpenAI-compatible chat completions API.

# VERIFY: confirm with one live smoke call that /chat/completions and the
# usage block match the OpenAI shape (they did at time of writing).
"""

from __future__ import annotations

from langbench.providers.base import OpenAICompatAdapter


class GroqAdapter(OpenAICompatAdapter):
    pass
