"""Config loading: the shipped config/ must parse and cross-validate, and
dangling references must fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from langbench.config import (
    ModelConfig,
    Pricing,
    RateLimit,
    load_registry,
    validate_request_caps,
)


class TestShippedConfigs:
    def test_shipped_configs_load(self) -> None:
        reg = load_registry()
        assert "groq" in reg.providers
        assert reg.judge_model().provider == "gemini"
        assert any(m.key == "openai/gpt-4o-mini" for m in reg.models)

    def test_gpt4o_mini_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        reg = load_registry()
        enabled = {m.key for m in reg.enabled_candidate_models()}
        assert "openai/gpt-4o-mini" not in enabled  # costs money; explicit opt-in

    def test_auto_disabled_without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
                    "OPENAI_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        reg = load_registry()
        assert reg.enabled_candidate_models() == []

    def test_auto_enabled_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("OPENROUTER_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
        reg = load_registry()
        enabled = {m.key for m in reg.enabled_candidate_models()}
        assert "groq/llama-3.1-8b-instant" in enabled
        assert not any(k.startswith("openrouter/") for k in enabled)


def _write(tmp: Path, name: str, body: str) -> None:
    (tmp / name).write_text(body, encoding="utf-8")


BASE_PROVIDERS = """
providers:
  p1: {base_url: http://x.test, api_key_env: K1, enabled: true, role: candidate}
  j1: {base_url: http://y.test, api_key_env: K2, enabled: true, role: judge}
"""
BASE_EVAL = """
languages: [en]
gec: {n_auto_per_language: 5, seed: 1}
cefr: {n_auto_per_language: 5, seed: 1}
judge: {model_key: JUDGEKEY, n_items_per_language: 5, calibration_items: 3}
prompt_versions: {gec: v1, cefr: v1, feedback: v1}
"""


def _models_yaml(provider: str, judge_provider: str) -> str:
    return f"""
models:
  - key: {provider}/m
    provider: {provider}
    model_id: m
    display_name: M
    enabled: true
    rate_limit: {{rpm: 1, rpd: 1}}
    pricing: {{input_per_mtok: 0, output_per_mtok: 0}}
  - key: {judge_provider}/j
    provider: {judge_provider}
    model_id: j
    display_name: J
    enabled: true
    rate_limit: {{rpm: 1, rpd: 1}}
    pricing: {{input_per_mtok: 0, output_per_mtok: 0}}
"""


class TestValidation:
    def test_dangling_provider_fails_loudly(self, tmp_path: Path) -> None:
        _write(tmp_path, "providers.yaml", BASE_PROVIDERS)
        _write(tmp_path, "models.yaml", _models_yaml("NOPE", "j1"))
        _write(tmp_path, "eval.yaml", BASE_EVAL.replace("JUDGEKEY", "j1/j"))
        with pytest.raises(ValueError, match="unknown provider"):
            load_registry(tmp_path)

    def test_judge_on_candidate_provider_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, "providers.yaml", BASE_PROVIDERS)
        _write(tmp_path, "models.yaml", _models_yaml("p1", "p1"))
        _write(tmp_path, "eval.yaml", BASE_EVAL.replace("JUDGEKEY", "p1/j"))
        with pytest.raises(ValueError, match="separate provider family"):
            load_registry(tmp_path)

    def test_missing_prompt_version_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, "providers.yaml", BASE_PROVIDERS)
        _write(tmp_path, "models.yaml", _models_yaml("p1", "j1"))
        bad_eval = BASE_EVAL.replace("JUDGEKEY", "j1/j").replace(
            "prompt_versions: {gec: v1, cefr: v1, feedback: v1}",
            "prompt_versions: {gec: v1, cefr: v1}",
        )
        _write(tmp_path, "eval.yaml", bad_eval)
        with pytest.raises(ValueError, match="feedback"):
            load_registry(tmp_path)


class TestRequestCapValidation:
    """DECISION 42: budgets that a provider admission-rejects (HTTP 413,
    prompt estimate + max_tokens over the per-request cap) must fail at
    config load, not churn as permanent pendings at run time."""

    def _model(self, **overrides: object) -> ModelConfig:
        base: dict[str, object] = dict(
            key="p1/m", provider="p1", model_id="m", display_name="M",
            enabled=True, rate_limit=RateLimit(rpm=1, rpd=1),
            pricing=Pricing(input_per_mtok=0, output_per_mtok=0),
        )
        base.update(overrides)
        return ModelConfig(**base)  # type: ignore[arg-type]

    def test_budget_over_cap_fails_loudly(self) -> None:
        m = self._model(
            max_output_tokens=4096,
            max_output_tokens_per_task={"gec": 8192},  # the live 2026-07-29 bug
            request_token_cap=8000,
            max_prompt_estimate=800,
        )
        with pytest.raises(ValueError, match="HTTP 413"):
            validate_request_caps([m])

    def test_admissible_budgets_pass(self) -> None:
        m = self._model(
            max_output_tokens=4096,
            max_output_tokens_per_task={"gec": 7000},
            request_token_cap=8000,
            max_prompt_estimate=800,
        )
        validate_request_caps([m])  # must not raise

    def test_cap_without_prompt_estimate_rejected(self) -> None:
        m = self._model(request_token_cap=8000)
        with pytest.raises(ValueError, match="max_prompt_estimate"):
            validate_request_caps([m])

    def test_no_cap_means_no_check(self) -> None:
        validate_request_caps([self._model(max_output_tokens=999999)])
