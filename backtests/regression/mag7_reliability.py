"""MAG7-ONLY reliability study — one-year wheel campaigns at $200,000.

Companion to ``sim200k_reliability.py`` (UNIVERSE_100) but restricted to the
**Magnificent Seven** mega-caps: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA.
The question is the same — *is the engine's edge stable across entry
regimes?* — but the universe is deliberately concentrated to the seven names
a real "Mag7 wheel" account would trade.

Honest framing (disclosed, not hidden):

- **Extreme concentration is structural, not a bug.** With seven names, one
  GICS cluster (Info-Tech / Comm-Services / Consumer-Disc), and $200k of
  cash-secured collateral (strike x 100), the book holds only a handful of
  simultaneous positions and single-name exposure runs very high. The armed
  R9 sector cap (25% NAV) and R10 single-name cap (10% NAV) would block
  almost every Mag7 open — so this study runs **caps-off canonical**
  (``require_ev_authority=False``) and reports the would-fire concentration
  post-hoc. A Mag7-only book is, by construction, an idiosyncratic-risk bet.
- **2022 is the stress test.** NVDA/META/TSLA drew down 50-70% in 2022; a
  Mag7 wheel entering then faces deep assignments. The 2022 windows are the
  honest reliability probe, not the bull windows.
- **Universe is fixed + survivorship-free by luck.** All seven survived;
  there is no membership drift to model here (unlike the S&P-500 study).

Point-in-time discipline is identical to ``sim200k_reliability.py``: every
rank passes ``as_of=today.isoformat()`` so the engine sees only data dated
on/before the simulated day; settlement uses the expiry-day close only once
the day-stepping loop reaches it; no window extends past the OHLCV frontier.

Data integrity pre-verified 2026-06-14: all seven OHLCV series are clean
split-adjusted (the AAPL/TSLA-2020, AMZN/GOOGL/TSLA-2022, NVDA-2021/2024
splits all show ratio ~1.0 across the split date — no BKNG/CVNA-style
unadjusted seam).

Run all eight windows::

    python -m backtests.regression.mag7_reliability all

Run one window::

    python -m backtests.regression.mag7_reliability one --window w4_2022_bear
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

DOC = "docs/MAG7_RELIABILITY_2026-06.md"
FRICTION_LEVELS = ("none", "bid_ask", "full")

# The Magnificent Seven (Alphabet via GOOGL; one share class to avoid
# double-counting). All present in the connector universe, OHLCV 2018+.
MAG7: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA")

# Same eight regime-diverse one-year windows as the UNIVERSE_100 study, so
# the two campaigns are directly comparable. All starts >= 2020-01-02 (the
# 504-day history gate is fully open) and all ends <= the 2026-06-04 frontier.
WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "w1_2020_crash_entry",
        "2020-02-03",
        "2021-01-29",
        "worst-case timing: first month runs into the COVID crash",
    ),
    (
        "w2_2020_recovery",
        "2020-06-01",
        "2021-05-28",
        "post-crash recovery with elevated IV richness",
    ),
    ("w3_2021_calm_bull", "2021-01-04", "2021-12-31", "calm grind-up bull; low IV, premium-thin"),
    (
        "w4_2022_bear",
        "2022-01-03",
        "2022-12-30",
        "rate-shock bear; Mag7 drew down 50-70% — the stress test",
    ),
    (
        "w5_2022_bottom_entry",
        "2022-10-03",
        "2023-09-29",
        "entry near the bear low; high IV at entry",
    ),
    ("w6_2023_chop", "2023-07-03", "2024-06-28", "chop into renewed bull; mixed regime"),
    ("w7_2024_late_cycle", "2024-06-03", "2025-05-30", "late-cycle bull into 2025"),
    (
        "w8_2025_recent",
        "2025-03-03",
        "2026-02-27",
        "most recent full year inside the data frontier",
    ),
)

# Canonical knobs — identical to S38/S43/SIM-200K except capital ($200k) and
# the Mag7 universe. top_n=15 is non-binding here (only 7 names exist).
CANONICAL: dict = {
    "capital": 200_000.0,
    "tickers": list(MAG7),
    "seed": 42,
    "top_n": 15,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
}


def _work_dir() -> Path:
    temp = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
    return Path(temp) / "mag7_backtest"


def _window_dir(window_id: str) -> Path:
    return _work_dir() / window_id


def _run_window(window_id: str, start: str, end: str, note: str) -> dict:
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
        "aggregate_full": headline.metrics["aggregate"],
        "per_year_full": headline.metrics["per_year"],
        "per_quartile_full": headline.metrics["per_quartile"],
        "per_friction_aggregate": {level: r.metrics["aggregate"] for level, r in results.items()},
        "fingerprint_full": headline.fingerprint,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    nav = headline.metrics["aggregate"]["final_nav"]
    print(
        f"\n[done] {window_id}: {elapsed / 60:.1f} min | full-friction NAV "
        f"${nav:,.0f} ({100 * (nav / CANONICAL['capital'] - 1):+.2f}%)",
        flush=True,
    )
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def all() -> None:
    """Run all eight windows sequentially."""
    print(
        "MAG7-RELIABILITY — eight one-year windows, $200k, AAPL/MSFT/GOOGL/AMZN/META/NVDA/TSLA.",
        flush=True,
    )
    print(f"Output root: {_work_dir()}", flush=True)
    summaries = [_run_window(*w) for w in WINDOWS]
    with open(_work_dir() / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now(UTC).isoformat(), "windows": summaries},
            f,
            indent=2,
            default=str,
        )
        f.write("\n")
    print(f"\n[ALL DONE] {len(summaries)} windows.\n", flush=True)


@app.command()
def one(window: str) -> None:
    """Run a single window by ``window_id``."""
    spec = next((w for w in WINDOWS if w[0] == window), None)
    if spec is None:
        ids = ", ".join(w[0] for w in WINDOWS)
        print(f"Unknown window {window!r}. Valid: {ids}", file=sys.stderr)
        raise typer.Exit(2)
    _run_window(*spec)


@app.command()
def smoke() -> None:
    """3-week single-window smoke to validate the driver end-to-end."""
    out_dir = _work_dir() / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_backtest_multi_friction(
        friction_levels=("full",),
        start="2024-06-03",
        end="2024-06-24",
        output_dir=out_dir,
        **CANONICAL,
    )
    agg = res["full"].metrics["aggregate"]
    print(
        f"[smoke] final_nav=${agg['final_nav']:,.2f} trades={agg.get('n_trades', agg.get('trades', '?'))}",
        flush=True,
    )


@app.command()
def info() -> None:
    print("MAG7-RELIABILITY — plan")
    print(f"Universe: {MAG7}")
    print(f"Output root: {_work_dir()}")
    print(f"Canonical knobs: {CANONICAL}")
    for w in WINDOWS:
        print(f"  {w[0]}: {w[1]} -> {w[2]}  [{w[3]}]")


if __name__ == "__main__":
    app()
