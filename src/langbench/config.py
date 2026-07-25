"""Typed configuration schemas and loaders.

Three config files drive everything:

- config/providers.yaml : provider base URLs, key env var names, enabled flags
- config/models.yaml    : model registry with per-(provider, model) rate limits
                          and list prices
- config/eval.yaml      : sample sizes, seeds, languages, judge settings

Design rules honored here:
- "enabled: auto" on a provider/model resolves to True iff the provider's API
  key env var is set and non-empty. This implements "enabled only if keys are
  present" without code changes per provider.
- No key is read at import time. Resolution happens when load_*() is called.
- Rate limits are per (provider, model), never per provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, Field, field_validator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

EnabledFlag = bool | Literal["auto"]


class ProviderConfig(BaseModel):
    """One entry in providers.yaml."""

    name: str
    base_url: str
    api_key_env: str
    enabled: EnabledFlag = "auto"
    # "candidate" providers supply models under evaluation; the "judge"
    # provider must be a different family (bias control, see README).
    role: Literal["candidate", "judge"] = "candidate"

    def resolved_enabled(self) -> bool:
        if self.enabled == "auto":
            return bool(os.environ.get(self.api_key_env, "").strip())
        return self.enabled

    def api_key(self) -> str | None:
        key = os.environ.get(self.api_key_env, "").strip()
        return key or None


class RateLimit(BaseModel):
    """Per-(provider, model) limits. Both dimensions enforced."""

    rpm: int = Field(gt=0, description="requests per minute")
    rpd: int = Field(gt=0, description="requests per day")


class Pricing(BaseModel):
    """List prices in USD per 1M tokens. Free tiers use the paid list price
    of the same model where one exists, else 0.0 (labelled 'free' in report)."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class ModelConfig(BaseModel):
    """One entry in models.yaml."""

    key: str  # unique registry key, e.g. "groq/llama-3.1-8b-instant"
    provider: str
    model_id: str
    display_name: str
    enabled: EnabledFlag = "auto"
    rate_limit: RateLimit
    pricing: Pricing
    max_output_tokens: int = 1024
    notes: str = ""

    @field_validator("key")
    @classmethod
    def key_has_provider_prefix(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError("model key must be '<provider>/<model_id-ish>'")
        return v


class JudgeSettings(BaseModel):
    model_key: str  # must reference a models.yaml entry on the judge provider
    temperature: float = 0.0
    n_items_per_language: int = 50
    calibration_items: int = 30
    calibration_language: str = "en"


class TaskSampling(BaseModel):
    n_auto_per_language: int = 200
    seed: int = 42


class EvalConfig(BaseModel):
    languages: list[str]
    gec: TaskSampling
    cefr: TaskSampling
    judge: JudgeSettings
    prompt_versions: dict[str, str]  # task name -> version string, e.g. {"gec": "v1"}


class Registry(BaseModel):
    """Everything loaded and cross-validated."""

    providers: dict[str, ProviderConfig]
    models: list[ModelConfig]
    eval: EvalConfig

    def enabled_candidate_models(self) -> list[ModelConfig]:
        out = []
        for m in self.models:
            prov = self.providers[m.provider]
            if prov.role != "candidate":
                continue
            if not prov.resolved_enabled():
                continue
            if m.enabled == "auto":
                if prov.resolved_enabled():
                    out.append(m)
            elif m.enabled:
                out.append(m)
        return out

    def judge_model(self) -> ModelConfig:
        key = self.eval.judge.model_key
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"judge model_key {key!r} not found in models.yaml")

    def model(self, key: str) -> ModelConfig:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"model key {key!r} not in registry")


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at top level")
    return data


def load_registry(config_dir: Path | None = None) -> Registry:
    """Load and cross-validate all three config files.

    Raises with a named, specific error if references dangle (model ->
    provider, judge -> model, judge model on a candidate provider, etc.).
    """
    cdir = config_dir or CONFIG_DIR

    praw = _load_yaml(cdir / "providers.yaml")
    providers: dict[str, ProviderConfig] = {}
    for name, body in cast(dict[str, object], praw.get("providers") or {}).items():
        if not isinstance(body, dict):
            raise ValueError(f"providers.yaml entry {name!r} must be a mapping")
        providers[name] = ProviderConfig(name=name, **body)

    mraw = _load_yaml(cdir / "models.yaml")
    models_raw = mraw.get("models")
    if not isinstance(models_raw, list):
        raise ValueError("models.yaml must have a top-level 'models' list")
    models = [ModelConfig(**m) for m in models_raw]

    eraw = _load_yaml(cdir / "eval.yaml")
    eval_cfg = EvalConfig(**eraw)  # type: ignore[arg-type]

    # Cross-validation with loud, specific errors.
    keys_seen: set[str] = set()
    for m in models:
        if m.provider not in providers:
            raise ValueError(f"models.yaml: {m.key} references unknown provider {m.provider!r}")
        if m.key in keys_seen:
            raise ValueError(f"models.yaml: duplicate model key {m.key!r}")
        keys_seen.add(m.key)

    reg = Registry(providers=providers, models=models, eval=eval_cfg)
    judge = reg.judge_model()
    if providers[judge.provider].role != "judge":
        raise ValueError(
            f"eval.yaml judge model {judge.key!r} is on provider {judge.provider!r} "
            "which is not role: judge. The judge must be a separate provider family."
        )
    for task in ("gec", "cefr", "feedback"):
        if task not in eval_cfg.prompt_versions:
            raise ValueError(f"eval.yaml prompt_versions missing entry for task {task!r}")
    return reg
