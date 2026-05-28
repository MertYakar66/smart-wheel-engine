"""S43 — Rolling 5-window backtest with post-#260 engine.

Tests whether S38's "engine underperforms SPY by 52pp over 2020-2024"
result generalises across rolling 5-year windows OR is window-specific
to 2020-2024.

Same setup as S38 (`docs/ENGINE_BACKTEST_S38_MULTIWINDOW.md`) except
the time window: 100-ticker universe (`UNIVERSE_100`), $1M capital,
35-DTE / 25-delta short puts wheeled into covered calls, three
friction levels, `require_ev_authority=False`. Engine is post-#260
(F4 realised-vol-ratio widening) + #255 R9 sector cap + #256 R10
single-name cap.

**Data-coverage constraint:** the Bloomberg OHLCV CSV starts
2018-01-02. With `enforce_history_gate=True` + `min_history_days=504`
(the engine defaults), the survivorship gate rejects all candidates
until ~2020-01-02 (504 trading days from the OHLCV start). This
truncates the early portion of windows that start before 2020-01-02.
Effective windows:

==========  ============================  ==========================
Window      Calendar period               Effective period
==========  ============================  ==========================
W1          2018-01-02 → 2022-12-31       ~2020-01-02 → 2022-12-31
W2          2019-01-02 → 2023-12-31       ~2020-01-02 → 2023-12-31
W3          2020-01-02 → 2024-12-31       2020-01-02 → 2024-12-31
W4          2021-01-02 → 2025-12-31       2021-01-02 → 2025-12-31
==========  ============================  ==========================

W3 (2020-2024) is the direct S38 re-run on the post-#260 engine —
delivers the "Δ vs pre-#260 baseline" comparison.

Output layout (per window, under ``%TEMP%\\s43_backtest\\<window_id>``)::

    <WORK_DIR>/w1_2018_2022/
        none/rank_log.csv      # one CSV per friction level
        none/metrics.json
        bid_ask/rank_log.csv
        bid_ask/metrics.json
        full/rank_log.csv
        full/metrics.json
    <WORK_DIR>/w2_2019_2023/...
    <WORK_DIR>/w3_2020_2024/...
    <WORK_DIR>/w4_2021_2025/...

Run all four sequentially::

    python -m backtests.regression.s43_rolling_multiwindow all

Run one window only (checkpoint-friendly)::

    python -m backtests.regression.s43_rolling_multiwindow one --window w3_2020_2024
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

SNAPSHOT_ID_PREFIX = "s43_rolling"
DOC = "docs/ENGINE_BACKTEST_S43_ROLLING_MULTIWINDOW.md"
FRICTION_LEVELS = ("none", "bid_ask", "full")


# Window definitions. Each entry is (window_id, start, end, note).
WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "w1_2018_2022",
        "2018-01-03",
        "2022-12-30",
        "2018-2022; gate truncates effective start to ~2020-01-02",
    ),
    (
        "w2_2019_2023",
        "2019-01-02",
        "2023-12-29",
        "2019-2023; gate truncates effective start to ~2020-01-02",
    ),
    (
        "w3_2020_2024",
        "2020-01-02",
        "2024-12-31",
        "2020-2024; direct S38 re-run on post-#260 engine",
    ),
    (
        "w4_2021_2025",
        "2021-01-04",
        "2025-12-31",
        "2021-2025; NEW clean 5y forward-anchored window",
    ),
)

# Canonical knobs — identical to S38/S34 except start/end.
CANONICAL: dict = {
    "capital": 1_000_000.0,
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
    return Path(temp) / "s43_backtest"


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
    """Run all four windows sequentially. ~12h wall-clock on the dev box."""
    print("S43 — Rolling multi-window. Engine SHA = origin/main HEAD (post-#260).", flush=True)
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
    """Run a single window by ``window_id`` (checkpoint-friendly).

    Valid window ids: w1_2018_2022, w2_2019_2023, w3_2020_2024, w4_2021_2025.
    """
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
    print("S43 — Rolling multi-window plan")
    print(f"Output root: {_work_dir()}")
    print(f"Friction levels: {FRICTION_LEVELS}")
    print(f"Canonical knobs: {CANONICAL}")
    print()
    print("Windows:")
    for window_id, start, end, note in WINDOWS:
        print(f"  {window_id}: {start} -> {end}  [{note}]")


if __name__ == "__main__":
    app()
