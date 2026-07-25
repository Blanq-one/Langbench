"""Per-(provider, model) rate limiting.

Two enforced dimensions per model key:
- RPM via a token bucket (capacity = rpm, refill = rpm/60 tokens per second)
- RPD via a hard daily counter that resets at UTC midnight

The RPM path *waits*; the RPD path *raises* DailyQuotaExhausted so the runner
can park that model for the day and keep other models moving. This is what
makes multi-day --resume runs work instead of one model starving the queue.

Clock and sleep are injectable so tests run instantly and offline.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from langbench.config import RateLimit

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class DailyQuotaExhausted(Exception):
    """Raised when a model's RPD budget for the current UTC day is spent."""

    def __init__(self, model_key: str, rpd: int) -> None:
        super().__init__(f"{model_key}: daily quota of {rpd} requests exhausted (UTC day)")
        self.model_key = model_key
        self.rpd = rpd


def _utc_day(now_epoch: float) -> str:
    return datetime.fromtimestamp(now_epoch, tz=UTC).strftime("%Y-%m-%d")


@dataclass
class _BucketState:
    tokens: float
    last_refill: float
    day: str
    used_today: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateLimiter:
    """One limiter instance serves all models; state is keyed by model key.

    Persisted daily usage is deliberately NOT implemented here: the runner
    reconstructs a conservative picture from the results DB on --resume, and
    providers enforce the real limit server-side anyway (429s are retried by
    the adapter with backoff).  # DECISION: keep the limiter stateless across
    process restarts; persistence lives where the data already is.
    """

    def __init__(self, clock: Clock | None = None, sleeper: Sleeper | None = None) -> None:
        self._clock: Clock = clock or time.monotonic
        self._wall: Clock = time.time
        self._sleep: Sleeper = sleeper or asyncio.sleep
        self._buckets: dict[str, _BucketState] = {}
        self._limits: dict[str, RateLimit] = {}

    def register(self, model_key: str, limit: RateLimit) -> None:
        if model_key in self._buckets:
            return
        now = self._clock()
        self._limits[model_key] = limit
        self._buckets[model_key] = _BucketState(
            tokens=float(limit.rpm), last_refill=now, day=_utc_day(self._wall())
        )

    def preload_daily_usage(self, model_key: str, used_today: int) -> None:
        """Seed today's usage from the results DB on --resume. Takes the max
        of current and given: seeding may only make the limiter MORE cautious,
        never less (re-registration must not erase live usage)."""
        self._require(model_key)
        st = self._buckets[model_key]
        st.used_today = max(st.used_today, used_today)

    def _require(self, model_key: str) -> _BucketState:
        if model_key not in self._buckets:
            raise KeyError(f"model {model_key!r} not registered with rate limiter")
        return self._buckets[model_key]

    def remaining_today(self, model_key: str) -> int:
        st = self._require(model_key)
        self._roll_day(model_key, st)
        return max(0, self._limits[model_key].rpd - st.used_today)

    def _roll_day(self, model_key: str, st: _BucketState) -> None:
        today = _utc_day(self._wall())
        if st.day != today:
            st.day = today
            st.used_today = 0

    async def acquire(self, model_key: str) -> None:
        """Block until an RPM token is available; raise if RPD is spent."""
        st = self._require(model_key)
        limit = self._limits[model_key]
        async with st.lock:
            self._roll_day(model_key, st)
            if st.used_today >= limit.rpd:
                raise DailyQuotaExhausted(model_key, limit.rpd)
            while True:
                now = self._clock()
                elapsed = now - st.last_refill
                st.tokens = min(float(limit.rpm), st.tokens + elapsed * (limit.rpm / 60.0))
                st.last_refill = now
                if st.tokens >= 1.0:
                    st.tokens -= 1.0
                    st.used_today += 1
                    return
                deficit = 1.0 - st.tokens
                await self._sleep(deficit / (limit.rpm / 60.0))
