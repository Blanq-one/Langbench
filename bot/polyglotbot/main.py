"""Matrix wiring: the ONLY module that imports matrix-nio.

Run: uv sync --extra bot && uv run python -m polyglotbot.main
(from the repo root, with bot/ on PYTHONPATH — the Dockerfile and systemd
unit both set this up; see bot/DEPLOY.md).

# VERIFY: matrix-nio API details below (AsyncClient callbacks, RoomMessageText
# fields, thread relation payload shape) against the installed nio version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from nio import AsyncClient, MatrixRoom, RoomMessageText

from polyglotbot import metrics
from polyglotbot.brain import Brain
from polyglotbot.config import load_settings, load_winner_config
from polyglotbot.handlers import Handlers, Reply

log = logging.getLogger("polyglotbot")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            },
            ensure_ascii=False,
        )


async def _send(client: AsyncClient, room_id: str, reply: Reply) -> None:
    content: dict[str, Any] = {"msgtype": "m.notice", "body": reply.body}
    if reply.thread_root:
        content["m.relates_to"] = {  # VERIFY thread relation shape
            "rel_type": "m.thread",
            "event_id": reply.thread_root,
        }
    await client.room_send(room_id, message_type="m.room.message", content=content)


async def amain() -> int:
    load_dotenv()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    settings = load_settings()
    winner = load_winner_config()
    metrics.serve(settings.metrics_port)

    start_ts_ms = int(time.time() * 1000)

    async with httpx.AsyncClient() as http:
        brain = Brain(winner, http)
        handlers = Handlers(settings, brain)

        client = AsyncClient(settings.homeserver, settings.user_id)
        client.access_token = settings.access_token
        client.user_id = settings.user_id

        async def on_message(room: MatrixRoom, event: RoomMessageText) -> None:
            # Ignore history replayed on startup; only react to live traffic.
            if getattr(event, "server_timestamp", 0) < start_ts_ms:
                return
            try:
                replies = await handlers.handle_message(
                    room.room_id, event.sender, event.body, event.event_id
                )
                for reply in replies:
                    await _send(client, room.room_id, reply)
            except Exception:  # noqa: BLE001 — the sync loop must never die
                log.exception("handler crashed on event %s", event.event_id)

        client.add_event_callback(on_message, RoomMessageText)
        log.info("PolyglotBot up as %s on %s (model: %s)",
                 settings.user_id, settings.homeserver, winner.model_key)
        await client.sync_forever(timeout=30000, full_state=True)
    return 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
