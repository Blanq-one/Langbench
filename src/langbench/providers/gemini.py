"""Google AI Studio (Gemini) adapter — the judge. Distinct wire format.

# VERIFIED 2026-07-24 (live smoke call against gemini-3.5-flash-lite):
#   POST {base}/models/{model}:generateContent  (auth: x-goog-api-key header)
#   body: contents[].parts[].text, systemInstruction, generationConfig
#   resp: candidates[0].content.parts[0].text, usageMetadata.promptTokenCount /
#   candidatesTokenCount — all exactly as coded. Response parts also carry a
#   thoughtSignature field (ignored by the parser).
"""

from __future__ import annotations

from typing import Any

from langbench.config import ModelConfig
from langbench.providers.base import ChatRequest, ProviderAdapter


class GeminiAdapter(ProviderAdapter):
    def _build_http(
        self, model: ModelConfig, req: ChatRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": req.user}]}],
            "generationConfig": {
                "temperature": req.temperature,
                "maxOutputTokens": req.max_tokens,
            },
        }
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        url = f"{self.provider.base_url}/models/{model.model_id}:generateContent"
        # Auth via header, NOT ?key= query param: httpx exception reprs include
        # the full URL, so a key in the URL would leak into retry-path logs.
        # Gemini's REST API accepts x-goog-api-key.  # VERIFIED 2026-07-24
        return url, body, {"x-goog-api-key": self.api_key}

    def _parse(self, body: dict[str, Any]) -> tuple[str, int | None, int | None]:
        # Join ALL text parts, not parts[0]: Gemini may split a response into
        # multiple parts (and thinking models flag thought parts with
        # "thought": true, which must never leak into content). Non-text parts
        # (e.g. bare thoughtSignature) are skipped.
        parts = body["candidates"][0]["content"]["parts"]
        texts = [
            str(p["text"])
            for p in parts
            if isinstance(p, dict) and "text" in p and not p.get("thought")
        ]
        if not texts:
            raise KeyError("no non-thought text parts in candidates[0].content.parts")
        usage = body.get("usageMetadata") or {}
        return (
            "".join(texts),
            usage.get("promptTokenCount"),
            usage.get("candidatesTokenCount"),
        )
