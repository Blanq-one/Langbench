"""The bot's brain: LLM calls via the SAME modules the eval used.

Reused from langbench: ratelimit.py (so the bot cannot blow the free tier),
the provider adapters (retry/backoff), the versioned feedback/cefr prompts,
and parsing.py (one bounded repair). No duplicated LLM plumbing lives here.

Deliberately NOT reused: cache.py. The raw cache persists full responses —
which quote the learner's message — to disk, and !help promises "Nothing is
stored on disk." Live room messages effectively never repeat, so the cache
would save ~nothing while making that promise false. # DECISION: deviates
from spec §5 ("reuse cache + rate limiter"); privacy note wins.

Every public method returns a BrainResult and never raises: the bot must
answer *something* in-room even when the provider is down or the daily quota
is gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import httpx

from langbench.config import ModelConfig, Pricing, ProviderConfig, RateLimit
from langbench.parsing import Failed, Ok, build_repair_request, parse
from langbench.providers import ChatRequest, ChatResponse, build_adapter
from langbench.providers.base import ProviderError
from langbench.ratelimit import DailyQuotaExhausted, RateLimiter
from langbench.tasks import cefr as cefr_task
from langbench.tasks import feedback as feedback_task
from polyglotbot.config import WinnerConfig

log = logging.getLogger("polyglotbot.brain")

Status = Literal["ok", "rate_limited", "error", "unparseable"]


@dataclass
class FeedbackResult:
    status: Status
    items: list[feedback_task.FeedbackItem]
    corrected: str
    latency_ms: float = 0.0


@dataclass
class LevelResult:
    status: Status
    level: str = ""


class Brain:
    def __init__(self, winner: WinnerConfig, client: httpx.AsyncClient) -> None:
        self.winner = winner
        provider = ProviderConfig(
            name=winner.provider,
            base_url=winner.provider_base_url,
            api_key_env=winner.api_key_env,
            enabled=True,
        )
        self.model = ModelConfig(
            key=winner.model_key,
            provider=winner.provider,
            model_id=winner.model_id,
            display_name=winner.model_key,
            enabled=True,
            rate_limit=RateLimit(rpm=winner.rate_limit_rpm, rpd=winner.rate_limit_rpd),
            pricing=Pricing(input_per_mtok=0.0, output_per_mtok=0.0),
            max_output_tokens=winner.max_output_tokens,
            extra_body=winner.extra_body,
        )
        self.adapter = build_adapter(provider, client)
        self.limiter = RateLimiter()
        self.limiter.register(self.model.key, self.model.rate_limit)

    async def _call(self, req: ChatRequest) -> ChatResponse:
        # No disk cache on purpose: responses quote the learner's message and
        # !help promises nothing is stored on disk. See module docstring.
        await self.limiter.acquire(self.model.key)
        return await self.adapter.chat(self.model, req)

    async def feedback(self, text: str, lang: str) -> FeedbackResult:
        version = self.winner.prompt_version
        req = feedback_task.build_request(text, version, lang, self.model.max_output_tokens)
        try:
            resp = await self._call(req)
            result = parse(resp.text, feedback_task.Output)
            if isinstance(result, Failed):
                repair = build_repair_request(req, resp.text, result.error)
                resp = await self._call(repair)
                result = parse(resp.text, feedback_task.Output)
            if isinstance(result, Ok):
                return FeedbackResult(
                    status="ok",
                    items=result.value.errors,
                    corrected=result.value.corrected,
                    latency_ms=resp.latency_ms,
                )
            return FeedbackResult(status="unparseable", items=[], corrected="")
        except DailyQuotaExhausted:
            return FeedbackResult(status="rate_limited", items=[], corrected="")
        except ProviderError as e:
            log.error("feedback call failed: %s", e)
            return FeedbackResult(status="error", items=[], corrected="")

    async def level(self, recent_messages: list[str], lang: str) -> LevelResult:
        joined = "\n".join(recent_messages)
        # v1 pins the cefr prompt version to the feedback version's family
        # ("v1"); a future bot config could carry per-task versions. # DECISION
        version = "v1"
        req = cefr_task.build_request(joined, version, lang, self.model.max_output_tokens)
        try:
            resp = await self._call(req)
            result = parse(resp.text, cefr_task.Output)
            if isinstance(result, Failed):
                repair = build_repair_request(req, resp.text, result.error)
                resp = await self._call(repair)
                result = parse(resp.text, cefr_task.Output)
            if isinstance(result, Ok):
                return LevelResult(status="ok", level=result.value.level)
            return LevelResult(status="unparseable")
        except DailyQuotaExhausted:
            return LevelResult(status="rate_limited")
        except ProviderError as e:
            log.error("level call failed: %s", e)
            return LevelResult(status="error")
