#!/usr/bin/env python3
"""Run the evaluation.

The quota budget estimate ALWAYS prints before any API call is made. Free
daily quotas will not fit the full eval in one sitting; run this daily with
defaults (--resume is the default and there is deliberately no --no-resume
that re-spends quota — to redo an item, delete its results-DB row).

Usage:
    uv run python scripts/run_eval.py --dry-run              # estimator only
    uv run python scripts/run_eval.py                        # candidates phase
    uv run python scripts/run_eval.py --phase judge
    uv run python scripts/run_eval.py --phase all
    uv run python scripts/run_eval.py --calibrate-judge
    uv run python scripts/run_eval.py --langs en de --models groq/llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from langbench.cache import RawCache  # noqa: E402
from langbench.config import load_registry  # noqa: E402
from langbench.ratelimit import RateLimiter  # noqa: E402
from langbench.results import ResultsDB  # noqa: E402
from langbench.runner import Runner, configure_logging  # noqa: E402


async def amain(args: argparse.Namespace) -> int:
    reg = load_registry()
    models = reg.enabled_candidate_models()
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m.key in wanted]
        missing = wanted - {m.key for m in models}
        if missing:
            print(f"unknown/disabled model keys: {sorted(missing)}", file=sys.stderr)
            return 2
    if not models:
        print(
            "No enabled candidate models. Set provider API keys in .env "
            "(enabled: auto) or flip 'enabled' in config/models.yaml.",
            file=sys.stderr,
        )
        return 2

    async with httpx.AsyncClient() as client:
        runner = Runner(
            registry=reg,
            cache=RawCache(),
            results=ResultsDB(),
            limiter=RateLimiter(),
            client=client,
        )
        plan = runner.load_work(args.langs)
        if plan.total_items() == 0:
            print(
                "Work plan is empty: no prepared data / manifests found. "
                "Run scripts/prepare_data.py first.",
                file=sys.stderr,
            )
            return 2

        print(runner.estimate(plan, models))
        if args.dry_run:
            return 0

        if args.calibrate_judge:
            report = await runner.run_calibration(models)
            print()
            print(report.summary())
            return 0

        if args.phase in ("candidates", "all"):
            await runner.run_candidates(plan, models)
        if args.phase in ("judge", "all"):
            await runner.run_judge(plan, models)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["candidates", "judge", "all"],
                        default="candidates")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the quota budget estimate and exit")
    parser.add_argument("--calibrate-judge", action="store_true",
                        help="run the ~30-item English judge calibration")
    parser.add_argument("--langs", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict to these model keys")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    configure_logging(10 if args.verbose else 20)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
