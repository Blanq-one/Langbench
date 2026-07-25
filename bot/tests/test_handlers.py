"""Offline tests for PolyglotBot's room logic. No matrix-nio, no network:
a fake brain and a fake clock exercise every v1 behavior path."""

from __future__ import annotations

import pytest
from polyglotbot.brain import FeedbackResult, LevelResult
from polyglotbot.config import BotSettings
from polyglotbot.handlers import HELP_TEXT, RATE_LIMITED_TEXT, Handlers

from langbench.tasks.feedback import FeedbackItem

BOT_ID = "@polyglotbot:example.org"
ROOM = "!room:example.org"
ALICE = "@alice:example.org"


def settings(**overrides: object) -> BotSettings:
    base = dict(
        homeserver="https://example.org",
        user_id=BOT_ID,
        access_token="t",
        default_lang="en",
        feedback_cooldown_s=30.0,
        max_message_chars=100,
        level_window_size=5,
        metrics_port=9100,
    )
    base.update(overrides)
    return BotSettings(**base)  # type: ignore[arg-type]


class FakeBrain:
    def __init__(self) -> None:
        self.feedback_result = FeedbackResult(
            status="ok",
            items=[FeedbackItem(span="I is", category="grammar",
                                correction="I am", explanation="Use 'am' with 'I'.")],
            corrected="I am here.",
            latency_ms=42.0,
        )
        self.level_result = LevelResult(status="ok", level="B1")
        self.feedback_calls: list[tuple[str, str]] = []
        self.level_calls: list[tuple[list[str], str]] = []

    async def feedback(self, text: str, lang: str) -> FeedbackResult:
        self.feedback_calls.append((text, lang))
        return self.feedback_result

    async def level(self, msgs: list[str], lang: str) -> LevelResult:
        self.level_calls.append((msgs, lang))
        return self.level_result


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


Env = tuple[Handlers, FakeBrain, FakeClock]


@pytest.fixture
def env() -> Env:
    brain = FakeBrain()
    clock = FakeClock()
    return Handlers(settings(), brain, clock=clock), brain, clock


async def _opt_in(h: Handlers) -> None:
    await h.handle_message(ROOM, ALICE, "!feedback on", "$e0")


class TestCommands:
    async def test_help(self, env: Env) -> None:
        h, _, _ = env
        replies = await h.handle_message(ROOM, ALICE, "!help", "$e1")
        assert replies == [replies[0]] and replies[0].body == HELP_TEXT

    async def test_lang_set_and_reject(self, env: Env) -> None:
        h, brain, _ = env
        ok = await h.handle_message(ROOM, ALICE, "!lang de", "$e1")
        assert "de" in ok[0].body
        bad = await h.handle_message(ROOM, ALICE, "!lang klingon", "$e2")
        assert bad[0].body.startswith("Usage:")
        await _opt_in(h)
        await h.handle_message(ROOM, ALICE, "hallo welt", "$e3")
        assert brain.feedback_calls[0][1] == "de"  # room language reached the brain

    async def test_feedback_default_off(self, env: Env) -> None:
        h, brain, _ = env
        replies = await h.handle_message(ROOM, ALICE, "my english are good", "$e1")
        assert replies == []
        assert brain.feedback_calls == []

    async def test_feedback_toggle(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        replies = await h.handle_message(ROOM, ALICE, "I is here", "$e1")
        assert len(replies) == 1
        assert replies[0].thread_root == "$e1"  # threaded reply
        assert "I am" in replies[0].body
        await h.handle_message(ROOM, ALICE, "!feedback off", "$e2")
        assert await h.handle_message(ROOM, ALICE, "more mistake", "$e3") == []

    async def test_unknown_command_silent(self, env: Env) -> None:
        h, _, _ = env
        assert await h.handle_message(ROOM, ALICE, "!weather", "$e1") == []


class TestGuards:
    async def test_ignores_self(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        assert await h.handle_message(ROOM, BOT_ID, "I is bot", "$e1") == []
        assert brain.feedback_calls == []

    async def test_ignores_other_bots(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        assert await h.handle_message(ROOM, "@weatherbot:example.org", "I is", "$e1") == []
        assert brain.feedback_calls == []

    async def test_max_length_guard(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        assert await h.handle_message(ROOM, ALICE, "x " * 200, "$e1") == []
        assert brain.feedback_calls == []

    async def test_cooldown(self, env: Env) -> None:
        h, brain, clock = env
        await _opt_in(h)
        r1 = await h.handle_message(ROOM, ALICE, "I is one", "$e1")
        assert len(r1) == 1
        r2 = await h.handle_message(ROOM, ALICE, "I is two", "$e2")
        assert r2 == []  # inside cooldown
        clock.t += 31.0
        r3 = await h.handle_message(ROOM, ALICE, "I is three", "$e3")
        assert len(r3) == 1
        assert len(brain.feedback_calls) == 2  # cooldown-suppressed msg never hit brain

    async def test_correct_message_gets_silence(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        brain.feedback_result = FeedbackResult(status="ok", items=[], corrected="Fine.")
        assert await h.handle_message(ROOM, ALICE, "This is fine.", "$e1") == []


class TestDegradation:
    async def test_rate_limited_reply(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        brain.feedback_result = FeedbackResult(status="rate_limited", items=[], corrected="")
        replies = await h.handle_message(ROOM, ALICE, "I is limited", "$e1")
        assert replies[0].body == RATE_LIMITED_TEXT

    async def test_provider_error_stays_quiet(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        brain.feedback_result = FeedbackResult(status="error", items=[], corrected="")
        assert await h.handle_message(ROOM, ALICE, "I is error", "$e1") == []

    async def test_error_does_not_consume_cooldown(self, env: Env) -> None:
        h, brain, _ = env
        await _opt_in(h)
        brain.feedback_result = FeedbackResult(status="error", items=[], corrected="")
        await h.handle_message(ROOM, ALICE, "I is error", "$e1")
        brain.feedback_result = FeedbackResult(
            status="ok",
            items=[FeedbackItem(span="I is", category="grammar",
                                correction="I am", explanation="x")],
            corrected="I am ok",
        )
        replies = await h.handle_message(ROOM, ALICE, "I is retry", "$e2")
        assert len(replies) == 1  # failed attempt didn't start the cooldown


class TestLevel:
    async def test_needs_three_messages(self, env: Env) -> None:
        h, brain, _ = env
        await h.handle_message(ROOM, ALICE, "one", "$e1")
        replies = await h.handle_message(ROOM, ALICE, "!level", "$e2")
        assert "at least 3" in replies[0].body
        assert brain.level_calls == []

    async def test_level_estimate(self, env: Env) -> None:
        h, brain, _ = env
        for i, msg in enumerate(["one message", "two message", "three message"]):
            await h.handle_message(ROOM, ALICE, msg, f"$m{i}")
        replies = await h.handle_message(ROOM, ALICE, "!level", "$e9")
        assert "B1" in replies[0].body
        assert brain.level_calls[0][0] == ["one message", "two message", "three message"]

    async def test_window_is_bounded(self, env: Env) -> None:
        h, brain, _ = env
        for i in range(10):  # window size is 5
            await h.handle_message(ROOM, ALICE, f"msg {i}", f"$m{i}")
        await h.handle_message(ROOM, ALICE, "!level", "$e9")
        assert len(brain.level_calls[0][0]) == 5
        assert brain.level_calls[0][0][0] == "msg 5"
