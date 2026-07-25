"""Shared test infrastructure. Everything here is offline: fake adapters
return canned text, the registry is built in Python (no env keys), and DBs
live in tmp dirs. No test in this suite may open a network connection."""

from __future__ import annotations

from pathlib import Path

import pytest

from langbench.config import (
    EvalConfig,
    JudgeSettings,
    ModelConfig,
    Pricing,
    ProviderConfig,
    RateLimit,
    Registry,
    TaskSampling,
)
from langbench.providers.base import ChatRequest, ChatResponse

FIXTURES = Path(__file__).parent / "fixtures"


def _extract_learner_text(user: str) -> str:
    for marker in ("Text to correct:\n", "Learner text to rate:\n", "Learner text:\n"):
        if marker in user:
            return user.split(marker, 1)[1].split("\n\nYour previous response was:", 1)[0]
    return user


class FakeAdapter:
    """Well-behaved candidate: valid JSON for every task, echoes the input
    back as its 'correction' (a lazy but format-perfect model)."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, model: ModelConfig, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        system = req.system or ""
        text = _extract_learner_text(req.user).strip()
        if "CEFR rater" in system:
            body = '{"level": "B1"}'
        elif "grammatical error correction" in system:
            body = '{"corrected": ' + _json_str(text) + "}"
        elif "language tutor" in system:
            body = (
                '{"errors": [{"span": "is", "category": "grammar", '
                '"correction": "are", "explanation": "Plural subjects take are."}], '
                '"corrected": ' + _json_str(text) + "}"
            )
        elif "evaluator" in system:
            body = (
                '{"correct_errors": 4, "correction_accuracy": 4, '
                '"explanation_clarity": 5, "no_hallucinated": 4}'
            )
        else:
            body = "{}"
        return ChatResponse(
            text=body, prompt_tokens=50, completion_tokens=20, latency_ms=10.0, raw={}
        )


class BrokenAdapter:
    """Format-hostile candidate: never returns JSON, ever — including on the
    repair turn. Exists to exercise the scored-failure path end to end."""

    async def chat(self, model: ModelConfig, req: ChatRequest) -> ChatResponse:
        return ChatResponse(
            text="I think the answer is probably fine, thanks for asking!",
            prompt_tokens=50, completion_tokens=15, latency_ms=12.0, raw={},
        )


def _json_str(s: str) -> str:
    import json

    return json.dumps(s, ensure_ascii=False)


def make_registry(n_items: int = 5) -> Registry:
    providers = {
        "fakeprov": ProviderConfig(
            name="fakeprov", base_url="http://invalid.test",
            api_key_env="FAKE_KEY_UNSET", enabled=True, role="candidate",
        ),
        "fakejudge": ProviderConfig(
            name="fakejudge", base_url="http://invalid.test",
            api_key_env="FAKE_JUDGE_KEY_UNSET", enabled=True, role="judge",
        ),
    }
    generous = RateLimit(rpm=100000, rpd=1000000)
    models = [
        ModelConfig(
            key="fakeprov/good", provider="fakeprov", model_id="good",
            display_name="Good Fake", enabled=True, rate_limit=generous,
            pricing=Pricing(input_per_mtok=0.10, output_per_mtok=0.20),
        ),
        ModelConfig(
            key="fakeprov/broken", provider="fakeprov", model_id="broken",
            display_name="Broken Fake", enabled=True, rate_limit=generous,
            pricing=Pricing(input_per_mtok=0.50, output_per_mtok=0.80),
        ),
        ModelConfig(
            key="fakejudge/judge", provider="fakejudge", model_id="judge",
            display_name="Fake Judge", enabled=True, rate_limit=generous,
            pricing=Pricing(input_per_mtok=0.0, output_per_mtok=0.0),
        ),
    ]
    eval_cfg = EvalConfig(
        languages=["en"],
        gec=TaskSampling(n_auto_per_language=n_items, seed=42),
        cefr=TaskSampling(n_auto_per_language=n_items, seed=42),
        judge=JudgeSettings(
            model_key="fakejudge/judge", temperature=0.0,
            n_items_per_language=n_items, calibration_items=3,
            calibration_language="en",
        ),
        prompt_versions={"gec": "v1", "cefr": "v1", "feedback": "v1"},
    )
    return Registry(providers=providers, models=models, eval=eval_cfg)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
