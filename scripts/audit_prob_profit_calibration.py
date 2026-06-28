#!/usr/bin/env python3
"""W3 — prob_profit calibration stratified by VIX regime (heavy-verify 2026-06-27, #436).

Validation-only. Extends the canonical realism diagnostic
(``scripts/vnv_prob_profit_calibration.py``) with the **VIX-regime** stratification
#436 W3 asks for: realized-vs-forecast ``prob_profit`` across the feasible rolling
as_of grid, stratified by (VIX regime × probability bin), each cell with a Wilson
95% interval. Quantifies the documented top-bin crisis over-confidence (prior
studies: realized ~0.57 vs ~0.96 forecast) and reports whether the over-confidence
is regime-dependent and how large.

Methodology parity: the as_of grid, prob bins, realized hold-to-expiry rule
(``S_expiry > strike - premium``), DATA_END (2026-03-20, which pre-dates the W1
2026-03-23 OHLCV splice so realized prices are never contaminated by D-W1-1), and
the Wilson helper are imported verbatim from the canonical driver.

HONESTY: candidates are NOT independent (names recur, windows overlap), so a
bin's n counts candidates not trials and the Wilson interval is optimistically
tight. No (regime × bin) cell with n < 30 is treated as conclusive. §2: strictly
read-only — every candidate is produced by the authoritative ranker; nothing is
re-scored.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_prob_profit_calibration.py [--limit N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from engine.wheel_runner import WheelRunner  # noqa: E402

OUT = REPO / "docs" / "verification_artifacts" / "data_wiring_2026-06-27"
OUT.mkdir(parents=True, exist_ok=True)

# Import the canonical driver's constants/helpers by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "_vnv_calib", REPO / "scripts" / "vnv_prob_profit_calibration.py"
)
_vnv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vnv)
AS_OF_GRID = _vnv.AS_OF_GRID
BINS = _vnv.BINS
DATA_END = _vnv.DATA_END
wilson = _vnv.wilson


# VIX-regime thresholds. R11 uses VIX > 25 as its elevated cut (DECISIONS.md D23);
# we report three buckets so the calm/elevated boundary straddles it.
def vix_regime(vix: float) -> str:
    if vix != vix:  # NaN
        return "unknown"
    if vix < 20.0:
        return "calm (<20)"
    if vix < 30.0:
        return "elevated (20-30)"
    return "crisis (>=30)"


def calib_rows(rows: list[dict]) -> list[dict]:
    out = []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["pp"] < hi]
        n = len(b)
        if n == 0:
            continue
        k = sum(1 for r in b if r["realized"])
        pred = sum(r["pp"] for r in b) / n
        wl, wh = wilson(k, n)
        out.append(
            {
                "bin": [lo, hi],
                "n": n,
                "pred": round(pred, 4),
                "realized": round(k / n, 4),
                "gap": round(k / n - pred, 4),
                "wilson": [round(wl, 4), round(wh, 4)],
                "conclusive_n_ge_30": n >= 30,
            }
        )
    return out


def top_bin(rows: list[dict], thresh: float) -> dict:
    top = [r for r in rows if r["pp"] > thresh]
    n = len(top)
    if n == 0:
        return {"n": 0}
    k = sum(1 for r in top if r["realized"])
    pred = sum(r["pp"] for r in top) / n
    wl, wh = wilson(k, n)
    return {
        "n": n,
        "pred": round(pred, 4),
        "realized": round(k / n, 4),
        "gap": round(k / n - pred, 4),
        "wilson": [round(wl, 4), round(wh, 4)],
        "conclusive_n_ge_30": n >= 30,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="universe_limit (None=full)")
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

    recs: list[dict] = []
    vix_by_date: dict[str, float] = {}
    for as_of in AS_OF_GRID:
        asof_ts = pd.Timestamp(as_of)
        vix = float(conn.get_vix_regime(as_of).get("vix", float("nan")))
        vix_by_date[as_of] = round(vix, 2)
        reg = vix_regime(vix)
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
                tk = str(row["ticker"])
            except (KeyError, TypeError, ValueError):
                continue
            expiry = asof_ts + pd.Timedelta(days=dte)
            if expiry > DATA_END:
                continue
            r = realized_close(tk, expiry)
            if r is None:
                continue
            s_exp, used = r
            if used <= asof_ts:
                continue
            recs.append(
                {
                    "pp": pp,
                    "realized": bool(s_exp > (strike - prem)),
                    "vix": vix,
                    "regime": reg,
                    "year": asof_ts.year,
                }
            )
            n_used += 1
        print(f"  {as_of}: VIX={vix:5.1f} {reg:<17} {n_used} scored")

    print(f"\ntotal candidate-outcomes={len(recs)}")

    report: dict = {
        "total": len(recs),
        "dates": len(AS_OF_GRID),
        "vix_by_date": vix_by_date,
        "data_end": str(DATA_END.date()),
        "calibration_all_bins": calib_rows(recs),
        "by_vix_regime": {},
        "top_bin_overall": {
            "gt_0.90": top_bin(recs, 0.90),
            "gt_0.95": top_bin(recs, 0.95),
        },
    }

    by_reg = defaultdict(list)
    for r in recs:
        by_reg[r["regime"]].append(r)
    for reg in ["calm (<20)", "elevated (20-30)", "crisis (>=30)"]:
        rows = by_reg.get(reg, [])
        report["by_vix_regime"][reg] = {
            "n": len(rows),
            "bins": calib_rows(rows),
            "top_bin_gt_0.90": top_bin(rows, 0.90),
            "top_bin_gt_0.95": top_bin(rows, 0.95),
        }

    (OUT / "w3_prob_profit_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ---- console summary ----
    print("\n=== TOP-BIN (prob_profit > 0.90) OVER-CONFIDENCE BY VIX REGIME ===")
    print(
        f"{'regime':<18}{'n':>6}{'pred':>8}{'realized':>10}{'gap':>8}{'wilson95':>20}{'concl':>7}"
    )
    for reg in ["calm (<20)", "elevated (20-30)", "crisis (>=30)"]:
        t = report["by_vix_regime"][reg]["top_bin_gt_0.90"]
        if t.get("n", 0) == 0:
            print(f"{reg:<18}{'0':>6}")
            continue
        w = f"[{t['wilson'][0]:.3f},{t['wilson'][1]:.3f}]"
        print(
            f"{reg:<18}{t['n']:>6}{t['pred']:>8.3f}{t['realized']:>10.3f}"
            f"{t['gap']:>+8.3f}{w:>20}{'Y' if t['conclusive_n_ge_30'] else 'n':>7}"
        )
    to = report["top_bin_overall"]["gt_0.90"]
    print(
        f"\noverall pp>0.90: n={to['n']} pred={to['pred']} realized={to['realized']} gap={to['gap']:+.3f}"
    )
    print(f"JSON → {OUT / 'w3_prob_profit_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
