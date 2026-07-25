"""Prometheus metrics with a no-op fallback.

prometheus-client ships in the 'bot' extra; the offline test suite (plain
`uv sync`) must import handlers.py without it, so everything here degrades to
no-ops when the library is absent and handlers.py depends only on this module.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("polyglotbot.metrics")

try:
    from prometheus_client import Counter, Histogram, start_http_server

    _AVAILABLE = True
except ImportError:  # plain uv sync, tests
    _AVAILABLE = False

    class _Noop:
        def labels(self, *a: Any, **kw: Any) -> _Noop:
            return self

        def inc(self, *a: Any, **kw: Any) -> None:
            pass

        def observe(self, *a: Any, **kw: Any) -> None:
            pass

    def Counter(*a: Any, **kw: Any) -> Any:  # noqa: N802
        return _Noop()

    def Histogram(*a: Any, **kw: Any) -> Any:  # noqa: N802
        return _Noop()

    def start_http_server(*a: Any, **kw: Any) -> None:
        log.warning("prometheus-client not installed; /metrics disabled")


MESSAGES_SEEN = Counter("polyglot_messages_seen_total", "Messages processed", ["room"])
FEEDBACK_SERVED = Counter("polyglot_feedback_served_total", "Feedback replies sent", ["room"])
COMMANDS = Counter("polyglot_commands_total", "Commands handled", ["command"])
PROVIDER_ERRORS = Counter("polyglot_provider_errors_total", "Provider errors", ["kind"])
PROVIDER_LATENCY = Histogram(
    "polyglot_provider_latency_ms", "LLM latency (ms)",
    buckets=(100, 250, 500, 1000, 2000, 5000, 10000),
)


def serve(port: int) -> None:
    if _AVAILABLE:
        start_http_server(port)
        log.info("metrics on :%d/metrics", port)
    else:
        start_http_server(port)
