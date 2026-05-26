"""S27 reproducer — 2022-2024, $100k, 24 tickers, frictionless.

Reproduces ``docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md``. The
documented headline metrics (Spearman ρ = 0.2183, final NAV = $151,444,
hit-rate = 76.39 %, mean realized = $63.34) become assertions via the
regression test in ``tests/test_backtest_regression.py`` against the
snapshot at ``backtests/regression/snapshots/s27_ivpit_24t_100k.json``.

Regenerate the snapshot with ``--update-snapshot`` after a deliberate
engine change. See TESTING.md "Backtest regression — re-baseline
workflow".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from backtests.regression._common import BacktestResult, run_backtest, save_snapshot
from backtests.regression.universes import UNIVERSE_24

SNAPSHOT_ID = "s27_ivpit_24t_100k"
DOC = "docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md"

CANONICAL: dict = {
    "capital": 100_000.0,
    "tickers": list(UNIVERSE_24),
    "start": "2022-01-03",
    "end": "2024-12-31",
    "seed": 42,
    "friction_level": "none",
    "top_n": 10,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
}


def run(**overrides) -> BacktestResult:
    """Run the S27 reproducer. Pass ``output_dir=`` to persist
    ``rank_log.csv`` + ``metrics.json``."""
    args = {**CANONICAL, **overrides}
    return run_backtest(**args)


def build_payload(result: BacktestResult) -> dict:
    """Snapshot payload for S27 — single friction level."""
    return {
        "snapshot_id": SNAPSHOT_ID,
        "doc": DOC,
        "fingerprint": result.fingerprint,
        **result.metrics,
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
    update_snapshot: bool = False,
) -> None:
    """Execute the S27 reproducer. Prints the metrics dict as JSON.

    Use ``--update-snapshot`` to overwrite
    ``backtests/regression/snapshots/s27_ivpit_24t_100k.json`` — only
    after a deliberate engine change that warrants a new baseline.
    """
    result = run(
        capital=capital, start=start, end=end, seed=seed, top_n=top_n, output_dir=output_dir
    )
    payload = build_payload(result)
    print(json.dumps(payload, indent=2, default=str))
    if update_snapshot:
        path = save_snapshot(SNAPSHOT_ID, payload)
        print(f"\nSnapshot written to {path}", file=sys.stderr)


if __name__ == "__main__":
    app()
