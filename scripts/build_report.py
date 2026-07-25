#!/usr/bin/env python3
"""Build REPORT.md, docs/index.html, docs/frontier.png from the committed
results DB. Needs no API keys and no corpus files — that is the point of the
two-tier storage design.

Usage:
    uv run python scripts/build_report.py
    uv run python scripts/build_report.py --emit-bot-config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langbench.config import load_registry  # noqa: E402
from langbench.report import build_all, emit_bot_config  # noqa: E402
from langbench.results import ResultsDB  # noqa: E402

BOT_CONFIG_PATH = Path("bot/config.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-bot-config", action="store_true",
                        help=f"write the winning config to {BOT_CONFIG_PATH}")
    parser.add_argument("--report-path", type=Path, default=Path("REPORT.md"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    reg = load_registry()
    results = ResultsDB()
    try:
        reports = build_all(results, reg, args.report_path, args.docs_dir)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"wrote {args.report_path} and {args.docs_dir}/index.html "
          f"({len(reports)} models)")

    if args.emit_bot_config:
        winner = emit_bot_config(reports, reg, BOT_CONFIG_PATH)
        print(f"bot config -> {BOT_CONFIG_PATH} (winner: {winner})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
