"""Mistral La Plateforme adapter. Chat endpoint is OpenAI-compatible.

# VERIFY: Mistral historically matched the OpenAI chat/completions shape;
# confirm usage-block field names with one live smoke call.
"""

from __future__ import annotations

from langbench.providers.base import OpenAICompatAdapter


class MistralAdapter(OpenAICompatAdapter):
    pass
