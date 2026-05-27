"""S35 reproducer — 2018-2020 out-of-window, $100k, 24 tickers, three friction levels.

Reproduces ``docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md`` (PR #224).
S35's headline was produced against the **pre-PIT-fix** engine; PR4
will lock a fresh post-fix baseline. Use this reproducer as the source
of truth for that re-baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from backtests.regression._common import run_backtest
from backtests.regression.universes import UNIVERSE_24

SNAPSHOT_ID = "s35_oos_24t_100k"
DOC = "docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md"
FRICTION_LEVELS = ("none", "bid_ask", "full")

CANONICAL: dict = {
    "capital": 100_000.0,
    "tickers": list(UNIVERSE_24),
    "start": "2018-01-02",
    "end": "2020-12-31",
    "seed": 42,
    "top_n": 10,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
}


def run(**overrides) -> dict:
    """Run all three friction levels over the out-of-window 2018-2020 regime."""
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


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    capital: float = CANONICAL["capital"],
    start: str = CANONICAL["start"],
    end: str = CANONICAL["end"],
    seed: int = CANONICAL["seed"],
    top_n: int = CANONICAL["top_n"],
    output_dir: Path | None = None,
) -> None:
    """Execute S35 — 2018-2020 out-of-window."""
    args: dict = {"capital": capital, "start": start, "end": end, "seed": seed, "top_n": top_n}
    if output_dir is not None:
        args["output_dir"] = output_dir
    payload = run(**args)
    print(json.dumps({"snapshot_id": SNAPSHOT_ID, **payload}, indent=2, default=str))


if __name__ == "__main__":
    app()
