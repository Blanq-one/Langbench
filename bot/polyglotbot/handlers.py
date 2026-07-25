"""Room logic: commands, opt-in feedback, cooldowns, the !level window.

Deliberately free of matrix-nio: this module takes plain strings in and
returns Reply values out, with the Brain injected behind a Protocol. main.py
owns the Matrix wiring; bot/tests exercise everything here offline with a
fake brain and a fake clock.

v1 hard limits enforced here:
- text only (main.py filters event types; handlers also guard empty bodies)
- one target language per room (!lang <code>)
- feedback opt-in per room, DEFAULT OFF (!feedback on|off)
- max 1 feedback reply per user per cooldown window
- max message length guard (long messages get no feedback)
- no feedback on other bots' messages, never on our own
- message content held ONLY in a bounded in-memory window (for !level);
  nothing is persisted (privacy note included in !help)
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from polyglotbot import metrics
from polyglotbot.brain import FeedbackResult, LevelResult
from polyglotbot.config import BotSettings

SUPPORTED_LANGS = {"en", "de", "it", "cs", "es"}

HELP_TEXT = (
    "PolyglotBot commands:\n"
    "  !lang <code>      set this room's target language "
    f"({', '.join(sorted(SUPPORTED_LANGS))})\n"
    "  !feedback on|off  opt this room in/out of writing feedback (default: off)\n"
    "  !level            estimate the CEFR level of your recent messages\n"
    "  !help             this message\n"
    "Privacy: your recent messages are kept only in memory, in a small rolling "
    "window per room, solely for !level. Nothing is stored on disk and the "
    "window is lost on restart."
)

RATE_LIMITED_TEXT = "I'm rate-limited right now — try again in a minute."
ERROR_TEXT = "I couldn't reach my language model just now. Try again shortly."


class BrainLike(Protocol):
    async def feedback(self, text: str, lang: str) -> FeedbackResult: ...

    async def level(self, recent_messages: list[str], lang: str) -> LevelResult: ...


@dataclass
class Reply:
    body: str
    thread_root: str | None = None  # event id to thread under, None = plain message


@dataclass
class RoomState:
    lang: str
    feedback_on: bool = False
    last_feedback_at: dict[str, float] = field(default_factory=dict)  # sender -> ts
    windows: dict[str, deque[str]] = field(default_factory=dict)      # sender -> msgs


def _looks_like_bot(sender: str) -> bool:
    localpart = sender.split(":", 1)[0].lstrip("@").lower()
    return "bot" in localpart  # heuristic; also always skip our own user id  # DECISION


class Handlers:
    def __init__(
        self,
        settings: BotSettings,
        brain: BrainLike,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.brain = brain
        self.clock = clock or time.monotonic
        self.rooms: dict[str, RoomState] = {}

    def _room(self, room_id: str) -> RoomState:
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomState(lang=self.settings.default_lang)
        return self.rooms[room_id]

    async def handle_message(
        self, room_id: str, sender: str, body: str, event_id: str
    ) -> list[Reply]:
        if sender == self.settings.user_id:
            return []
        body = body.strip()
        if not body:
            return []
        metrics.MESSAGES_SEEN.labels(room=room_id).inc()

        if body.startswith("!"):
            return await self._handle_command(room_id, sender, body)

        if _looks_like_bot(sender):
            return []

        state = self._room(room_id)
        window = state.windows.setdefault(
            sender, deque(maxlen=self.settings.level_window_size)
        )
        window.append(body)

        if not state.feedback_on:
            return []
        if len(body) > self.settings.max_message_chars:
            return []
        now = self.clock()
        last = state.last_feedback_at.get(sender)
        if last is not None and now - last < self.settings.feedback_cooldown_s:
            return []

        result = await self.brain.feedback(body, state.lang)
        if result.status == "rate_limited":
            metrics.PROVIDER_ERRORS.labels(kind="rate_limited").inc()
            return [Reply(RATE_LIMITED_TEXT, thread_root=event_id)]
        if result.status in ("error", "unparseable"):
            metrics.PROVIDER_ERRORS.labels(kind=result.status).inc()
            return []  # stay quiet on infra errors; don't spam the room  # DECISION
        state.last_feedback_at[sender] = now
        metrics.FEEDBACK_SERVED.labels(room=room_id).inc()
        metrics.PROVIDER_LATENCY.observe(result.latency_ms)
        if not result.items:
            return []  # correct message: silence beats "well done!" noise  # DECISION
        return [Reply(_format_feedback(result), thread_root=event_id)]

    async def _handle_command(self, room_id: str, sender: str, body: str) -> list[Reply]:
        parts = body.split()
        cmd = parts[0].lower()
        state = self._room(room_id)
        metrics.COMMANDS.labels(command=cmd).inc()

        if cmd == "!help":
            return [Reply(HELP_TEXT)]

        if cmd == "!lang":
            if len(parts) != 2 or parts[1].lower() not in SUPPORTED_LANGS:
                return [Reply(
                    f"Usage: !lang <code> — one of {', '.join(sorted(SUPPORTED_LANGS))}"
                )]
            state.lang = parts[1].lower()
            return [Reply(f"Target language for this room set to {state.lang}.")]

        if cmd == "!feedback":
            if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
                return [Reply("Usage: !feedback on|off")]
            state.feedback_on = parts[1].lower() == "on"
            if state.feedback_on:
                return [Reply(
                    "Writing feedback is ON for this room. I'll reply in threads "
                    "to messages with corrections. !feedback off to stop."
                )]
            return [Reply("Writing feedback is OFF for this room.")]

        if cmd == "!level":
            window = state.windows.get(sender)
            if not window or len(window) < 3:
                return [Reply(
                    "I need at least 3 recent messages from you in this room "
                    "to estimate a level. Keep chatting!"
                )]
            result = await self.brain.level(list(window), state.lang)
            if result.status == "ok":
                return [Reply(
                    f"Based on your last {len(window)} messages, my CEFR estimate "
                    f"is {result.level}. (Rough single-model estimate, not an exam.)"
                )]
            if result.status == "rate_limited":
                return [Reply(RATE_LIMITED_TEXT)]
            return [Reply(ERROR_TEXT)]

        return []  # unknown ! command: not ours, stay silent


def _format_feedback(result: FeedbackResult) -> str:
    lines = ["Some writing feedback:"]
    for item in result.items[:5]:  # cap per-message noise  # DECISION
        lines.append(
            f'- "{item.span}" -> "{item.correction}" ({item.category}): '
            f"{item.explanation}"
        )
    if len(result.items) > 5:
        lines.append(f"(+{len(result.items) - 5} more)")
    lines.append(f"Corrected: {result.corrected}")
    return "\n".join(lines)
