#!/usr/bin/env python
"""Realism diagnostic: prob_profit calibration + ev_dollars predictive power.

For each PIT ``as_of`` in a regime-spanning grid, rank the full universe and,
for every short-put candidate, compute the REALIZED hold-to-expiry outcome from
forward OHLCV (available because the data ends 2026-03-20 and we only score
as_of dates whose expiry falls on/before that). Two questions:

1. **prob_profit calibration** — bin candidates by predicted ``prob_profit``
   and compare each bin's mean prediction to its realized hit-rate.
2. **ev_dollars predictive power** — does a higher ``ev_dollars`` rank actually
   deliver higher realized P&L? (Pearson corr + quintile lift.) The cockpit
   trust layer asserts ev_dollars is a RANKING SCORE with ~0 correlation to
   realized dollars; this measures it.

Realized rules (engine-consistent, NOT "expired OTM"):
  * profit boolean: ``S_expiry > strike - premium`` (cash-secured put breakeven;
    using breakeven not ``> strike`` avoids the ~12pp methodology artifact HT-B
    flagged).
  * realized P&L per contract ($): ``(premium - max(0, strike - S_expiry)) * 100``.
Costs are omitted (makes realized marginally optimistic -> any over-confidence
shown is a LOWER bound). Hold-to-expiry, European-style (ignores early
assignment).

CAVEATS (read before quoting a bin): candidates are NOT independent — the same
name recurs across dates and windows overlap — so a bin's ``n`` is a count of
candidates, not independent trials; the Wilson interval shown is therefore
optimistically tight (the same caveat that gates the engine's own prob_profit
CI). sp500_ohlcv carries current-ish members, so delisted names are
under-sampled (mild survivorship bias). Treat this as evidence about
calibration SHAPE / EV-rank LIFT, not a significance test on any single bin.

§2: strictly read-only measurement. Every candidate is produced by the
authoritative ranker; nothing is re-scored or overridden.

Usage:
    python scripts/vnv_prob_profit_calibration.py --json out.json
    python scripts/vnv_prob_profit_calibration.py --limit 80   # quick smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from engine.wheel_runner import WheelRunner  # noqa: E402

DATA_END = pd.Timestamp("2026-03-20")

# Regime-spanning as_of grid (bi-monthly), all feasible: expiry (as_of+~35d)
# lands on/before the 2026-03-20 data end, and >=504 trading days of history
# exist (OHLCV starts 2018-01). Covers COVID recovery (2020), calm bull (2021),
# rate-hike bear (2022), recovery (2023), bull (2024-25), early 2026.
AS_OF_GRID = [
    "2020-06-15",
    "2020-09-15",
    "2020-11-16",
    "2021-01-15",
    "2021-03-15",
    "2021-05-14",
    "2021-07-15",
    "2021-09-15",
    "2021-11-15",
    "2022-01-14",
    "2022-03-15",
    "2022-05-16",
    "2022-07-15",
    "2022-09-15",
    "2022-11-15",
    "2023-01-13",
    "2023-03-15",
    "2023-05-15",
    "2023-07-14",
    "2023-09-15",
    "2023-11-15",
    "2024-01-16",
    "2024-03-15",
    "2024-05-15",
    "2024-07-15",
    "2024-09-16",
    "2024-11-15",
    "2025-01-15",
    "2025-03-14",
    "2025-05-15",
    "2025-07-15",
    "2025-09-15",
    "2025-11-14",
    "2026-01-15",
    "2026-02-13",
]

BINS = [
    (0.0, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 0.95),
    (0.95, 1.0001),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def regime_of(year: int) -> str:
    return {
        2020: "2020 covid-recovery",
        2021: "2021 calm-bull",
        2022: "2022 rate-bear",
        2023: "2023 recovery",
        2024: "2024 bull",
        2025: "2025 bull",
        2026: "2026 early",
    }.get(year, str(year))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="universe_limit (None=full)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runner = WheelRunner()
    conn = runner.connector
    print(f"provider={type(conn).__name__}  dates={len(AS_OF_GRID)}  limit={args.limit}")

    ohlcv_cache: dict[str, pd.DataFrame | None] = {}

    def realized_close(ticker: str, expiry: pd.Timestamp):
        if ticker not in ohlcv_cache:
            try:
                df = conn.get_ohlcv(ticker)
            except Exception:
                df = None
            if df is not None and len(df) and "close" in df.columns:
                df = df.sort_index()
            ohlcv_cache[ticker] = df
        df = ohlcv_cache[ticker]
        if df is None or not len(df):
            return None
        sub = df.loc[:expiry]
        if not len(sub):
            return None
        return float(sub["close"].iloc[-1]), sub.index[-1]

    # records: dicts {pp, realized, evd, pnl, year, tier}
    recs: list[dict] = []
    skipped_no_px = 0
    for as_of in AS_OF_GRID:
        asof_ts = pd.Timestamp(as_of)
        try:
            df = runner.rank_candidates_by_ev(
                tickers=None,
                universe_limit=args.limit,
                top_n=10_000,
                min_ev_dollars=-1e9,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
        except Exception as e:
            print(f"  {as_of}: rank FAILED {type(e).__name__}: {e}")
            continue
        n_used = 0
        for _, row in df.iterrows():
            try:
                dte = int(row["dte"])
                strike = float(row["strike"])
                prem = float(row["premium"])
                pp = float(row["prob_profit"])
                evd = float(row["ev_dollars"])
                tk = str(row["ticker"])
                tier = str(row.get("distribution_source", "?"))
            except (KeyError, TypeError, ValueError):
                continue
            expiry = asof_ts + pd.Timedelta(days=dte)
            if expiry > DATA_END:
                continue
            r = realized_close(tk, expiry)
            if r is None:
                skipped_no_px += 1
                continue
            s_exp, used = r
            if used <= asof_ts:  # no genuinely-forward price
                skipped_no_px += 1
                continue
            realized_pnl = (prem - max(0.0, strike - s_exp)) * 100.0
            recs.append(
                {
                    "pp": pp,
                    "realized": bool(s_exp > (strike - prem)),
                    "evd": evd,
                    "pnl": realized_pnl,
                    "year": asof_ts.year,
                    "tier": tier,
                }
            )
            n_used += 1
        print(f"  {as_of}: {n_used} scored")

    print(f"\ntotal candidate-outcomes={len(recs)}  skipped_no_price={skipped_no_px}")
    report: dict = {"total": len(recs), "skipped_no_price": skipped_no_px, "dates": AS_OF_GRID}

    # ---- 1. prob_profit calibration ----
    def calib(rows: list[dict], label: str) -> list[dict]:
        print(f"\n=== CALIBRATION — {label}  (n={len(rows)}) ===")
        print(f"{'bin':>12} {'n':>6} {'pred':>7} {'realized':>9} {'gap':>7}  wilson95")
        out = []
        for lo, hi in BINS:
            b = [r for r in rows if lo <= r["pp"] < hi]
            n = len(b)
            if n == 0:
                continue
            k = sum(1 for r in b if r["realized"])
            pred = sum(r["pp"] for r in b) / n
            real = k / n
            wl, wh = wilson(k, n)
            print(
                f"{f'[{lo:.2f},{hi:.2f})':>12} {n:6d} {pred:7.3f} "
                f"{real:9.3f} {real - pred:+7.3f}  [{wl:.3f},{wh:.3f}]"
            )
            out.append(
                {
                    "bin": [lo, hi],
                    "n": n,
                    "pred": round(pred, 4),
                    "realized": round(real, 4),
                    "gap": round(real - pred, 4),
                    "wilson": [round(wl, 4), round(wh, 4)],
                }
            )
        return out

    report["calibration_all"] = calib(recs, "ALL candidates")
    report["calibration_ev_pos"] = calib([r for r in recs if r["evd"] > 0], "ev_dollars > 0")

    print("\n=== TOP-BIN (prob_profit > 0.90) BY REGIME ===")
    print(f"{'regime':>22} {'n':>6} {'pred':>7} {'realized':>9} {'gap':>7}")
    by_year = defaultdict(list)
    for r in recs:
        by_year[r["year"]].append(r)
    regime_rows = []
    for year in sorted(by_year):
        top = [r for r in by_year[year] if r["pp"] > 0.90]
        if not top:
            continue
        n = len(top)
        k = sum(1 for r in top if r["realized"])
        pred = sum(r["pp"] for r in top) / n
        print(f"{regime_of(year):>22} {n:6d} {pred:7.3f} {k / n:9.3f} {k / n - pred:+7.3f}")
        regime_rows.append(
            {
                "regime": regime_of(year),
                "n": n,
                "pred": round(pred, 4),
                "realized": round(k / n, 4),
                "gap": round(k / n - pred, 4),
            }
        )
    report["top_bin_by_regime"] = regime_rows

    # ---- 2. ev_dollars predictive power ----
    print("\n=== EV-DOLLARS REALISM (does the EV rank predict realized $?) ===")
    evd = np.array([r["evd"] for r in recs], dtype=float)
    pnl = np.array([r["pnl"] for r in recs], dtype=float)
    corr = float(np.corrcoef(evd, pnl)[0, 1]) if len(recs) > 2 else float("nan")
    print(f"Pearson corr(ev_dollars, realized_pnl) = {corr:+.4f}  (n={len(recs)})")
    # quintile lift: sort by ev_dollars, mean realized pnl per quintile
    order = np.argsort(evd)
    q_means, q_pred = [], []
    if len(recs) >= 5:
        for qi, idx in enumerate(np.array_split(order, 5)):
            mp = float(pnl[idx].mean())
            me = float(evd[idx].mean())
            q_means.append(round(mp, 2))
            q_pred.append(round(me, 2))
            print(
                f"  EV-quintile {qi + 1} (mean ev_dollars {me:8.2f}): "
                f"mean realized $ = {mp:8.2f}  (n={len(idx)})"
            )
    pos = pnl[evd > 0]
    neg = pnl[evd <= 0]
    pos_m = float(pos.mean()) if len(pos) else float("nan")
    neg_m = float(neg.mean()) if len(neg) else float("nan")
    print(
        f"  mean realized $: ev>0 = {pos_m:+.2f} (n={len(pos)})  | "
        f"ev<=0 = {neg_m:+.2f} (n={len(neg)})"
    )
    report["ev_realism"] = {
        "pearson_corr": round(corr, 4),
        "quintile_mean_evd": q_pred,
        "quintile_mean_realized_pnl": q_means,
        "mean_pnl_ev_pos": round(pos_m, 2),
        "mean_pnl_ev_neg": round(neg_m, 2),
        "n_ev_pos": int(len(pos)),
        "n_ev_neg": int(len(neg)),
    }

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
