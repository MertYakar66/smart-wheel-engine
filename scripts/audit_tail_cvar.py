#!/usr/bin/env python3
"""W5 — tail-fit / CVaR reliability (heavy-verify 2026-06-27, #436, stretch).

Validation-only. Over the feasible as_of grid, ranks the full universe and, for
every short-put candidate, records the engine's tail outputs (``cvar_5``,
``cvar_99_evt``, ``tail_xi``, ``heavy_tail``, ``distribution_source``,
``n_scenarios``) alongside the realized hold-to-expiry P&L from forward OHLCV.
Reports:

  1. **POT-GPD tail-fit status** — how often the GPD tail fit fires (non-NaN
     ``tail_xi``), the heavy-tail rate, and the ``tail_xi`` distribution.
  2. **Empirical CVaR breach frequency vs nominal** — fraction of candidates
     whose realized P&L is worse than the forecast ``cvar_5``, overall and by VIX
     regime; the crisis/calm breach **multiple** (prior finding: ~3.5× in crisis).
  3. **Forward-distribution cascade graceful degradation** — the
     ``distribution_source`` tier mix (``empirical_non_overlapping`` → block
     bootstrap → HAR-RV) by VIX regime and by sample thinness (``n_scenarios``).

Methodology parity with the calibration driver: same grid, same realized rule,
same DATA_END (pre-dates the W1 2026-03-23 splice). §2: strictly read-only.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_tail_cvar.py [--limit N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from engine.wheel_runner import WheelRunner  # noqa: E402

OUT = REPO / "docs" / "verification_artifacts" / "data_wiring_2026-06-27"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "_vnv_calib", REPO / "scripts" / "vnv_prob_profit_calibration.py"
)
_vnv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vnv)
AS_OF_GRID = _vnv.AS_OF_GRID
DATA_END = _vnv.DATA_END


def vix_regime(vix: float) -> str:
    if vix != vix:
        return "unknown"
    if vix < 20.0:
        return "calm (<20)"
    if vix < 30.0:
        return "elevated (20-30)"
    return "crisis (>=30)"


def _f(x) -> float:
    """None/NaN-safe float coercion → nan."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _breach_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    breaches = sum(1 for r in rows if r["realized_pnl"] < r["cvar_5"])
    return {"n": n, "breaches": breaches, "breach_rate": round(breaches / n, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
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
    for as_of in AS_OF_GRID:
        asof_ts = pd.Timestamp(as_of)
        vix = float(conn.get_vix_regime(as_of).get("vix", float("nan")))
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
                cvar_5 = float(row["cvar_5"])
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
                    "cvar_5": cvar_5,
                    "cvar_99_evt": _f(row.get("cvar_99_evt")),
                    "tail_xi": _f(row.get("tail_xi")),
                    "heavy_tail": bool(row.get("heavy_tail", False)),
                    "dist_source": str(row.get("distribution_source", "?")),
                    "n_scenarios": int(_ns) if pd.notna(_ns := row.get("n_scenarios")) else 0,
                    "realized_pnl": (prem - max(0.0, strike - s_exp)) * 100.0,
                    "regime": reg,
                }
            )
            n_used += 1
        print(f"  {as_of}: VIX={vix:5.1f} {reg:<17} {n_used} scored")

    n = len(recs)
    print(f"\ntotal candidate-outcomes={n}")

    xi = np.array([r["tail_xi"] for r in recs], dtype=float)
    xi_fit = xi[~np.isnan(xi)]
    report: dict = {
        "total": n,
        "data_end": str(DATA_END.date()),
        "tail_fit": {
            "gpd_fit_fired": int((~np.isnan(xi)).sum()),
            "gpd_fit_rate": round(float((~np.isnan(xi)).mean()), 4) if n else None,
            "heavy_tail_rate": round(float(np.mean([r["heavy_tail"] for r in recs])), 4)
            if n
            else None,
            "tail_xi_median": round(float(np.median(xi_fit)), 4) if len(xi_fit) else None,
            "tail_xi_p90": round(float(np.percentile(xi_fit, 90)), 4) if len(xi_fit) else None,
        },
        "cvar_breach": {"overall": _breach_stats(recs), "by_regime": {}},
        "distribution_source": {
            "overall": dict(Counter(r["dist_source"] for r in recs)),
            "by_regime": {},
            "by_thinness": {
                "n_lt_30": dict(Counter(r["dist_source"] for r in recs if r["n_scenarios"] < 30)),
                "n_ge_30": dict(Counter(r["dist_source"] for r in recs if r["n_scenarios"] >= 30)),
            },
        },
    }
    by_reg = defaultdict(list)
    for r in recs:
        by_reg[r["regime"]].append(r)
    for reg in ["calm (<20)", "elevated (20-30)", "crisis (>=30)"]:
        rows = by_reg.get(reg, [])
        report["cvar_breach"]["by_regime"][reg] = _breach_stats(rows)
        report["distribution_source"]["by_regime"][reg] = dict(
            Counter(r["dist_source"] for r in rows)
        )

    # crisis/calm breach multiple
    calm = report["cvar_breach"]["by_regime"].get("calm (<20)", {}).get("breach_rate")
    crisis = report["cvar_breach"]["by_regime"].get("crisis (>=30)", {}).get("breach_rate")
    report["cvar_breach"]["crisis_over_calm_multiple"] = (
        round(crisis / calm, 2) if calm and crisis else None
    )

    (OUT / "w5_tail_cvar.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== W5 TAIL / CVaR ===")
    tf = report["tail_fit"]
    print(
        f"GPD tail-fit rate={tf['gpd_fit_rate']} heavy_tail_rate={tf['heavy_tail_rate']} "
        f"xi_median={tf['tail_xi_median']}"
    )
    print("CVaR breach (realized < cvar_5) by regime:")
    for reg, s in report["cvar_breach"]["by_regime"].items():
        if s.get("n"):
            print(f"   {reg:<17} n={s['n']:>5} breach_rate={s['breach_rate']}")
    print(f"crisis/calm breach multiple = {report['cvar_breach']['crisis_over_calm_multiple']}")
    print(f"dist_source overall = {report['distribution_source']['overall']}")
    print(f"dist_source thin(n<30) = {report['distribution_source']['by_thinness']['n_lt_30']}")
    print(f"JSON → {OUT / 'w5_tail_cvar.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
