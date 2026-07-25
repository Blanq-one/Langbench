"""Rate limiter: RPM waits, RPD raises, day rollover, resume seeding.
Fake clock + fake sleeper make these instant and deterministic."""

from __future__ import annotations

import pytest

from langbench.config import RateLimit
from langbench.ratelimit import DailyQuotaExhausted, RateLimiter


class FakeTime:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def ft() -> FakeTime:
    return FakeTime()


def make(ft: FakeTime, rpm: int, rpd: int) -> RateLimiter:
    rl = RateLimiter(clock=ft.clock, sleeper=ft.sleep)
    rl.register("prov/model", RateLimit(rpm=rpm, rpd=rpd))
    return rl


class TestRpm:
    async def test_burst_up_to_rpm_is_instant(self, ft: FakeTime) -> None:
        rl = make(ft, rpm=5, rpd=1000)
        for _ in range(5):
            await rl.acquire("prov/model")
        assert ft.sleeps == []

    async def test_sixth_call_waits(self, ft: FakeTime) -> None:
        rl = make(ft, rpm=5, rpd=1000)
        for _ in range(5):
            await rl.acquire("prov/model")
        await rl.acquire("prov/model")
        assert len(ft.sleeps) >= 1
        assert sum(ft.sleeps) == pytest.approx(60 / 5, rel=0.01)

    async def test_tokens_refill_over_time(self, ft: FakeTime) -> None:
        rl = make(ft, rpm=60, rpd=1000)  # 1 token/second
        for _ in range(60):
            await rl.acquire("prov/model")
        ft.t += 10.0  # 10 tokens refill
        for _ in range(10):
            await rl.acquire("prov/model")
        assert ft.sleeps == []


class TestRpd:
    async def test_rpd_exhaustion_raises(self, ft: FakeTime) -> None:
        rl = make(ft, rpm=1000, rpd=3)
        for _ in range(3):
            await rl.acquire("prov/model")
        with pytest.raises(DailyQuotaExhausted) as exc:
            await rl.acquire("prov/model")
        assert exc.value.model_key == "prov/model"

    async def test_resume_seeding_counts_against_quota(self, ft: FakeTime) -> None:
        rl = make(ft, rpm=1000, rpd=5)
        rl.preload_daily_usage("prov/model", 4)
        await rl.acquire("prov/model")  # 5th of the day: fine
        with pytest.raises(DailyQuotaExhausted):
            await rl.acquire("prov/model")

    async def test_remaining_today(self, ft: FakeTime) -> None:
        rl = make(ft, rpm=1000, rpd=10)
        assert rl.remaining_today("prov/model") == 10
        await rl.acquire("prov/model")
        assert rl.remaining_today("prov/model") == 9


class TestIsolation:
    async def test_limits_are_per_model_not_per_provider(self, ft: FakeTime) -> None:
        rl = RateLimiter(clock=ft.clock, sleeper=ft.sleep)
        rl.register("groq/small", RateLimit(rpm=1000, rpd=100))
        rl.register("groq/big", RateLimit(rpm=1000, rpd=2))
        for _ in range(2):
            await rl.acquire("groq/big")
        with pytest.raises(DailyQuotaExhausted):
            await rl.acquire("groq/big")
        # Same provider, different model: unaffected. This is the whole point.
        for _ in range(50):
            await rl.acquire("groq/small")

    async def test_unregistered_model_is_a_loud_error(self, ft: FakeTime) -> None:
        rl = RateLimiter(clock=ft.clock, sleeper=ft.sleep)
        with pytest.raises(KeyError, match="not registered"):
            await rl.acquire("nope/nope")
