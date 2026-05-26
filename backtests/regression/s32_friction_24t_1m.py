"""S32 reproducer — 2022-2024, $1M, 24 tickers, three friction levels.

Reproduces ``docs/ENGINE_BACKTEST_S32_FRICTION.md``. Runs the driver
three times (``none``, ``bid_ask``, ``full``) and bundles the
metrics. The full-friction level is the headline; the other two are
held in ``per_friction_level`` for the friction-decomposition
assertions in ``tests/test_backtest_regression.py``.

Regenerate the snapshot with ``--update-snapshot`` after a deliberate
engine change. See TESTING.md "Backtest regression — re-baseline
workflow".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from backtests.regression._common import run_backtest_multi_friction, save_snapshot
from backtests.regression.universes import UNIVERSE_24

SNAPSHOT_ID = "s32_friction_24t_1m"
DOC = "docs/ENGINE_BACKTEST_S32_FRICTION.md"
FRICTION_LEVELS = ("none", "bid_ask", "full")

CANONICAL: dict = {
    "capital": 1_000_000.0,
    "tickers": list(UNIVERSE_24),
    "start": "2022-01-03",
    "end": "2024-12-31",
    "seed": 42,
    "top_n": 10,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
}


def run(**overrides) -> dict:
    """Run all three friction levels via the shared-rank multi-driver.
    Returns the metrics dict bundling full-friction headline plus
    ``per_friction_level``."""
    args = {**CANONICAL, **overrides}
    results = run_backtest_multi_friction(friction_levels=FRICTION_LEVELS, **args)
    headline = results["full"]
    per_level = {level: r.metrics for level, r in results.items()}
    return {
        "aggregate": headline.metrics["aggregate"],
        "per_year": headline.metrics["per_year"],
        "per_quartile": headline.metrics["per_quartile"],
        "per_friction_level": per_level,
        "fingerprint": headline.fingerprint,
    }


def build_payload(result: dict) -> dict:
    """Snapshot payload for S32 — wraps the multi-friction run dict."""
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
    """Execute S32 — all three friction levels. Prints aggregated metrics."""
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
