#!/usr/bin/env python3
"""Query provider model-listing endpoints and print fresh models.yaml entries.

Because this build could not verify live model IDs, config/models.yaml ships
with best-known entries marked # VERIFY. This script is the reconciliation
tool: run it with keys in .env, diff its output against models.yaml, and fix
IDs that drifted. It prints YAML to stdout (never overwrites config in place).

Rate limits and prices CANNOT be discovered from these endpoints; they still
come from provider dashboards/docs by hand. Emitted entries carry
'rate_limit: FILL_ME' sentinels so an unreviewed paste fails config loading
loudly instead of running with fake limits.

Usage:
    uv run python scripts/discover_models.py [--provider groq]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from langbench.config import load_registry  # noqa: E402


async def list_openai_compat(base_url: str, api_key: str) -> list[str]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return sorted(str(m["id"]) for m in body.get("data", []))


async def list_gemini(base_url: str, api_key: str) -> list[str]:
    # VERIFY: GET {base}/models returns {"models": [{"name": "models/..."}]}
    # Key goes in the x-goog-api-key header, never the URL (leaks via exception reprs).
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{base_url}/models", headers={"x-goog-api-key": api_key})
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return sorted(str(m["name"]).removeprefix("models/") for m in body.get("models", []))


def emit_yaml(provider: str, model_ids: list[str]) -> None:
    print(f"\n# ---- discovered on provider: {provider} ----")
    for mid in model_ids:
        safe = mid.replace("/", "-")
        print(
            f"""  - key: {provider}/{safe}
    provider: {provider}
    model_id: {mid}
    display_name: {mid}
    enabled: false            # enable deliberately after review
    rate_limit: FILL_ME       # from provider dashboard, per model
    pricing: {{ input_per_mtok: 0.0, output_per_mtok: 0.0 }}  # FILL_ME
    max_output_tokens: 1024"""
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="limit to one provider name")
    args = parser.parse_args()

    load_dotenv()
    reg = load_registry()
    exit_code = 0
    for name, prov in reg.providers.items():
        if args.provider and name != args.provider:
            continue
        key = os.environ.get(prov.api_key_env, "").strip()
        if not key:
            print(f"# {name}: skipped, {prov.api_key_env} not set", file=sys.stderr)
            continue
        try:
            if name == "gemini":
                ids = await list_gemini(prov.base_url, key)
            else:
                ids = await list_openai_compat(prov.base_url, key)
        except httpx.HTTPError as e:
            print(f"# {name}: FAILED to list models: {e!r}", file=sys.stderr)
            exit_code = 1
            continue
        emit_yaml(name, ids)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
