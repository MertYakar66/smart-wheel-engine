"""SIM-200K — eight one-year wheel campaigns at $200,000 retail scale.

Reliability study: pretend eight different past dates were "day one" of
live trading, run the engine for one year from each, and observe the
distribution of outcomes. The question is not "does one window win" but
"is the engine's edge stable across entry regimes" — crash entry,
recovery, calm bull, bear, bottom entry, chop, late cycle, recent.

Setup is the CANONICAL harness (`run_backtest_multi_friction`,
`docs/ENGINE_BACKTEST_S43_ROLLING_MULTIWINDOW.md` lineage) with exactly
one knob changed: ``capital=200_000`` (retail account scale) instead of
$1M. Everything else matches S38/S43: 100-ticker universe
(`UNIVERSE_100`), 35-DTE / 25-delta short puts wheeled into covered
calls, three friction levels, `require_ev_authority=False` (caps-off
canonical config; R9/R10 would-fire analysis is post-hoc, as in S43).

Point-in-time discipline: every rank call inside the driver passes
``as_of=today.isoformat()`` — the engine sees only data dated on or
before the simulated day. Settlement uses the expiry day's close only
once the day-stepping loop reaches it. No window may extend past the
OHLCV frontier; `assert_data_window_available` enforces this at start.

Known honesty caveats (disclosed, not hidden — see the campaign doc):

- **Universe survivorship**: `UNIVERSE_100` is a fixed snapshot of
  current S&P members (commit 8a17b0b). A 2020 window therefore trades
  names known to have survived to 2026. Same property as S38/S43.
- **Window overlap**: the eight windows overlap in places; they are
  regime samples, not independent draws.
- **$200k buying-power saturation**: with cash-secured collateral
  (strike x 100) the book holds far fewer simultaneous positions than
  at $1M; high-priced names are naturally unaffordable.

Output layout (per window, under ``%TEMP%\\sim200k_backtest\\<window_id>``)::

    <WORK_DIR>/w1_2020_crash_entry/
        none/rank_log.csv          bid_ask/...        full/...
        none/metrics.json          (one set per friction level)
        none/tracker_state.json
        summary.json

Run all eight sequentially::

    python -m backtests.regression.sim200k_reliability all

Run one window (checkpoint-friendly; used by the parallel launcher)::

    python -m backtests.regression.sim200k_reliability one --window w4_2022_bear
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from backtests.regression._common import run_backtest_multi_friction
from backtests.regression.universes import UNIVERSE_100

SNAPSHOT_ID_PREFIX = "sim200k"
DOC = "docs/SIM_200K_RELIABILITY_2026-06.md"
FRICTION_LEVELS = ("none", "bid_ask", "full")


# Window definitions. Each entry is (window_id, start, end, note).
# All starts are >= 2020-01-02 (504-trading-day history gate fully open)
# and all ends are <= the 2026-06-04 OHLCV frontier.
WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "w1_2020_crash_entry",
        "2020-02-03",
        "2021-01-29",
        "worst-case timing: first trading month runs straight into the COVID crash",
    ),
    (
        "w2_2020_recovery",
        "2020-06-01",
        "2021-05-28",
        "post-crash recovery with elevated IV richness",
    ),
    (
        "w3_2021_calm_bull",
        "2021-01-04",
        "2021-12-31",
        "calm grind-up bull; low IV, premium-thin regime",
    ),
    (
        "w4_2022_bear",
        "2022-01-03",
        "2022-12-30",
        "rate-shock bear year; adverse regime for short puts",
    ),
    (
        "w5_2022_bottom_entry",
        "2022-10-03",
        "2023-09-29",
        "entry near the bear-market low; high IV at entry",
    ),
    (
        "w6_2023_chop",
        "2023-07-03",
        "2024-06-28",
        "chop into renewed bull; mixed regime",
    ),
    (
        "w7_2024_late_cycle",
        "2024-06-03",
        "2025-05-30",
        "late-cycle bull into 2025",
    ),
    (
        "w8_2025_recent",
        "2025-03-03",
        "2026-02-27",
        "most recent full year inside the data frontier",
    ),
)

# Canonical knobs — identical to S38/S43 except capital ($200k retail scale).
CANONICAL: dict = {
    "capital": 200_000.0,
    "tickers": list(UNIVERSE_100),
    "seed": 42,
    "top_n": 15,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
}


def _work_dir() -> Path:
    """Per-window output directory. Uses ``$TEMP`` so the large CSVs
    don't accidentally land in git."""
    temp = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
    return Path(temp) / "sim200k_backtest"


def _window_dir(window_id: str) -> Path:
    return _work_dir() / window_id


def _run_window(window_id: str, start: str, end: str, note: str) -> dict:
    """Run a single window and dump rank_log + metrics to disk."""
    out_dir = _window_dir(window_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 70}\nWindow {window_id}\n  start={start}  end={end}\n  note: {note}\n  output: {out_dir}\n{'=' * 70}",
        flush=True,
    )
    t0 = time.time()

    results = run_backtest_multi_friction(
        friction_levels=FRICTION_LEVELS,
        start=start,
        end=end,
        output_dir=out_dir,
        **CANONICAL,
    )

    elapsed = time.time() - t0
    headline = results["full"]
    summary = {
        "window_id": window_id,
        "start": start,
        "end": end,
        "note": note,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "aggregate_full": headline.metrics["aggregate"],
        "per_year_full": headline.metrics["per_year"],
        "per_quartile_full": headline.metrics["per_quartile"],
        "per_friction_aggregate": {level: r.metrics["aggregate"] for level, r in results.items()},
        "fingerprint_full": headline.fingerprint,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")

    print(
        f"\n[done] {window_id}: elapsed {elapsed / 3600:.2f}h | full-friction NAV "
        f"${headline.metrics['aggregate']['final_nav']:,.0f} ({100 * (headline.metrics['aggregate']['final_nav'] / CANONICAL['capital'] - 1):+.2f}%)",
        flush=True,
    )
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def all() -> None:
    """Run all eight windows sequentially."""
    print("SIM-200K — eight one-year windows. Engine SHA = branch HEAD.", flush=True)
    print(f"Universe: 100 tickers ({UNIVERSE_100[0]}, ..., {UNIVERSE_100[-1]}).", flush=True)
    print(f"Output root: {_work_dir()}", flush=True)
    summaries = []
    for window_id, start, end, note in WINDOWS:
        summary = _run_window(window_id, start, end, note)
        summaries.append(summary)
    with open(_work_dir() / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "windows": summaries,
            },
            f,
            indent=2,
            default=str,
        )
        f.write("\n")
    print(f"\n[ALL DONE] {len(summaries)} windows.\n", flush=True)


@app.command()
def one(window: str) -> None:
    """Run a single window by ``window_id`` (checkpoint-friendly)."""
    spec = next((w for w in WINDOWS if w[0] == window), None)
    if spec is None:
        ids = ", ".join(w[0] for w in WINDOWS)
        print(f"Unknown window {window!r}. Valid: {ids}", file=sys.stderr)
        raise typer.Exit(2)
    window_id, start, end, note = spec
    _run_window(window_id, start, end, note)


@app.command()
def info() -> None:
    """Print the window plan + output paths without running anything."""
    print("SIM-200K — eight one-year windows plan")
    print(f"Output root: {_work_dir()}")
    print(f"Friction levels: {FRICTION_LEVELS}")
    print(f"Canonical knobs: {CANONICAL}")
    print()
    print("Windows:")
    for window_id, start, end, note in WINDOWS:
        print(f"  {window_id}: {start} -> {end}  [{note}]")


if __name__ == "__main__":
    app()
