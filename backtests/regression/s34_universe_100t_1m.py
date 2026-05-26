"""S34 reproducer — 2022-2024, $1M, 100 tickers, three friction levels.

Reproduces the S34 backtest section of
``docs/SOUNDNESS_REVIEW_2026-05-26.md`` (also tracked in
``docs/ENGINE_BACKTEST_S34_UNIVERSE.md`` per PR #226). Largest universe;
concentration risk is explicit (BKNG carry +$31,576 on +$28,571 net
realized — single-ticker dominance the snapshot's ``concentration``
section will lock).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from backtests.regression._common import run_backtest, save_snapshot
from backtests.regression.universes import UNIVERSE_100

SNAPSHOT_ID = "s34_universe_100t_1m"
DOC = "docs/SOUNDNESS_REVIEW_2026-05-26.md"
FRICTION_LEVELS = ("none", "bid_ask", "full")

CANONICAL: dict = {
    "capital": 1_000_000.0,
    "tickers": list(UNIVERSE_100),
    "start": "2022-01-03",
    "end": "2024-12-31",
    "seed": 42,
    "top_n": 15,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
}


def run(**overrides) -> dict:
    """Run all three friction levels over the 100-ticker universe."""
    args = {**CANONICAL, **overrides}
    per_level: dict[str, dict] = {}
    headline = None
    for level in FRICTION_LEVELS:
        result = run_backtest(friction_level=level, **args)
        per_level[level] = result.metrics
        if level == "full":
            headline = result
    assert headline is not None
    return {
        "aggregate": headline.metrics["aggregate"],
        "per_year": headline.metrics["per_year"],
        "per_quartile": headline.metrics["per_quartile"],
        "per_friction_level": per_level,
        "fingerprint": headline.fingerprint,
    }


def build_payload(result: dict) -> dict:
    """Snapshot payload for S34 — wraps the multi-friction run dict."""
    return {"snapshot_id": SNAPSHOT_ID, "doc": DOC, **result}


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    capital: float = CANONICAL["capital"],
    start: str = CANONICAL["start"],
    end: str = CANONICAL["end"],
    seed: int = CANONICAL["seed"],
    top_n: int = CANONICAL["top_n"],
    output_dir: Path | None = None,
    update_snapshot: bool = False,
) -> None:
    """Execute S34 — 100 tickers × 3 friction levels."""
    args: dict = {"capital": capital, "start": start, "end": end, "seed": seed, "top_n": top_n}
    if output_dir is not None:
        args["output_dir"] = output_dir
    payload = build_payload(run(**args))
    print(json.dumps(payload, indent=2, default=str))
    if update_snapshot:
        path = save_snapshot(SNAPSHOT_ID, payload)
        print(f"\nSnapshot written to {path}", file=sys.stderr)


if __name__ == "__main__":
    app()
