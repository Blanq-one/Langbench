"""Google AI Studio (Gemini) adapter — the judge. Distinct wire format.

# VERIFY: endpoint path and response shape against current Gemini REST docs:
#   POST {base}/models/{model}:generateContent  (auth: x-goog-api-key header)
#   body: contents[].parts[].text, systemInstruction, generationConfig
#   resp: candidates[0].content.parts[0].text, usageMetadata token counts
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
        # Gemini's REST API accepts x-goog-api-key.  # VERIFY header name
        return url, body, {"x-goog-api-key": self.api_key}

    def _parse(self, body: dict[str, Any]) -> tuple[str, int | None, int | None]:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        usage = body.get("usageMetadata") or {}
        return (
            str(text),
            usage.get("promptTokenCount"),
            usage.get("candidatesTokenCount"),
        )
