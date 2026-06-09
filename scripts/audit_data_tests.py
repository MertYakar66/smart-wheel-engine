#!/usr/bin/env python3
"""Data-layer *test-coverage* audit probes — Phase 1 discovery (additive, read-only).

Companion to ``scripts/audit_data_engine.py``. That script inventories the
bundled data and probes the ranker for the 2026-06-07 weakness register
(W1-W13). This one computes the *deeper, test-focused* byte-evidence the
2026-06-09 round needs — the figures the first script does not produce:

  * IV: SERVED-vs-RAW band reconciliation after #363's ``_clean_vol_iv_inplace``
    gate, (date,ticker) uniqueness, the absence of any tenor/moneyness column
    (the file is single-point 30d ATM IV, not a surface).
  * OHLCV: per-name depth distribution, the exact NaN-price rows, monotone-date
    integrity.
  * Fundamentals: GICS-11 canonical-set validity, eqy_dvd_yld_12m band.
  * Credit: S&P rating-ladder validity, Altman-Z / interest-coverage plausibility.
  * Dividends: epsilon-negative magnitude, future-declared rows.
  * Treasury: per-tenor negative / coverage floor.
  * End-to-end: data -> ``WheelRunner.rank_candidates_by_ev`` -> EVResult is
    FINITE and sign-bearing on a pinned real ticker set (no §2 bypass — the
    production ranker calls ``EVEngine.evaluate`` internally).

Read-only. Asserts nothing (Phase 2 turns these into tests). Never touches the
decision trio. Routes every engine probe through ``WheelRunner`` — no §2 bypass.

Usage::

    py -3.12 scripts/audit_data_tests.py --json out.json      # default as_of=frontier
    py -3.12 scripts/audit_data_tests.py --as-of 2026-06-04
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data_connector import MarketDataConnector, normalize_ticker  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

# Canonical GICS 11 sectors (the R9 sector-cap keys must come from this set).
GICS_11 = {
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
}

# S&P long-term issuer credit ladder (+ common non-rated sentinels).
SP_LADDER = {
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "RD",
    "SD",
    "D",
}
SP_NONRATED = {"NR", "N.A.", "NA", "", "nan", "None"}

DATA = "data/bloomberg/"


def _read(fn: str, **kw: Any) -> pd.DataFrame:
    return pd.read_csv(Path(DATA) / fn, low_memory=False, **kw)


def _frontier() -> str:
    oh = pd.to_datetime(_read("sp500_ohlcv.csv", usecols=["date"])["date"], errors="coerce")
    iv = pd.to_datetime(_read("sp500_vol_iv_full.csv", usecols=["date"])["date"], errors="coerce")
    return str(min(oh.max(), iv.max()))[:10]


def probe_iv(conn: MarketDataConnector) -> dict[str, Any]:
    raw = _read("sp500_vol_iv_full.csv")
    p = pd.to_numeric(raw["hist_put_imp_vol"], errors="coerce")
    c = pd.to_numeric(raw["hist_call_imp_vol"], errors="coerce")
    both = p.notna() & c.notna()
    out: dict[str, Any] = {
        "columns": list(raw.columns),
        "has_tenor_or_moneyness_col": any(
            k in c2.lower()
            for c2 in raw.columns
            for k in ("dte", "tenor", "expir", "moneyness", "strike", "delta")
        ),
        "raw_rows": int(len(raw)),
        "raw_put_min": float(np.nanmin(p)),
        "raw_put_max": float(np.nanmax(p)),
        "raw_in_0_to_3": int(((p > 0) & (p <= 3.0)).sum()),
        "raw_above_300": int((p > 300).sum()),
        "raw_above_10000_sentinel": int((p > 10000).sum()),
        "put_eq_call_exact_pct": round(float((p[both] == c[both]).mean()) * 100, 4),
        "dup_date_ticker": int(raw.duplicated(subset=["date", "ticker"]).sum()),
    }
    # SERVED band: the connector's cleaned read (#363 _clean_vol_iv_inplace).
    served = conn._load("vol_iv")
    sp = pd.to_numeric(served["hist_put_imp_vol"], errors="coerce")
    sc = pd.to_numeric(served["hist_call_imp_vol"], errors="coerce")
    out["served"] = {
        "rows": int(len(served)),
        "put_nonnull_min": float(np.nanmin(sp)) if sp.notna().any() else None,
        "put_nonnull_max": float(np.nanmax(sp)) if sp.notna().any() else None,
        "served_put_le_3_nonnull": int(((sp > 0) & (sp <= 3.0)).sum()),
        "served_put_gt_10000": int((sp > 10000).sum()),
        "served_call_le_3_nonnull": int(((sc > 0) & (sc <= 3.0)).sum()),
        "served_call_gt_10000": int((sc > 10000).sum()),
        "nulled_vs_raw_put": int(p.notna().sum() - sp.notna().sum()),
    }
    return out


def probe_ohlcv() -> dict[str, Any]:
    oh = _read("sp500_ohlcv.csv")
    oh["nt"] = oh["ticker"].map(normalize_ticker)
    oh["d"] = pd.to_datetime(oh["date"], errors="coerce")
    price_cols = [x for x in ("open", "high", "low", "close") if x in oh.columns]
    pc = oh[price_cols].apply(pd.to_numeric, errors="coerce")
    nan_mask = pc.isna().any(axis=1)
    nan_rows = oh.loc[nan_mask, ["nt", "date"]].astype(str).values.tolist()
    g = oh.groupby("nt")
    ndays = g["d"].count()
    # monotone: any name whose dates are not strictly increasing
    nonmono = int(sum(1 for _, s in g["d"] if not s.is_monotonic_increasing))
    return {
        "rows": int(len(oh)),
        "price_cols": price_cols,
        "names": int(oh["nt"].nunique()),
        "depth_min": int(ndays.min()),
        "depth_median": int(ndays.median()),
        "depth_max": int(ndays.max()),
        "names_below_504": int((ndays < 504).sum()),
        "nan_price_row_count": int(nan_mask.sum()),
        "nan_price_rows": nan_rows[:20],
        "dup_date_ticker": int(oh.duplicated(subset=["nt", "date"]).sum()),
        "names_nonmonotone_dates": nonmono,
    }


def probe_fundamentals() -> dict[str, Any]:
    fu = _read("sp500_fundamentals.csv")
    sec = fu["gics_sector_name"].astype(str)
    bad = sorted(set(sec[~sec.isin(GICS_11) & (sec != "nan")].unique()))
    y = pd.to_numeric(fu["eqy_dvd_yld_12m"], errors="coerce")
    return {
        "has_date_col": "date" in fu.columns,
        "rows": int(len(fu)),
        "sector_values": sorted(set(sec[sec != "nan"].unique())),
        "sectors_outside_gics11": bad,
        "rows_outside_gics11": int((~sec.isin(GICS_11) & (sec != "nan")).sum()),
        "sector_nan": int((sec == "nan").sum()),
        "yld_min": float(np.nanmin(y)),
        "yld_median": float(np.nanmedian(y)),
        "yld_max": float(np.nanmax(y)),
        "yld_negative": int((y < 0).sum()),
        "yld_nan": int(y.isna().sum()),
        "yld_above_25pct": int((y > 25).sum()),
    }


def probe_credit() -> dict[str, Any]:
    cr = _read("sp500_credit_risk.csv")
    rt = cr["rtg_sp_lt_lc_issuer_credit"].astype(str).str.strip()
    valid = rt.isin(SP_LADDER)
    nonrated = rt.isin(SP_NONRATED) | (rt == "nan")
    unknown = sorted(set(rt[~valid & ~nonrated].unique()))
    z = pd.to_numeric(cr["altman_z_score"], errors="coerce")
    ic = pd.to_numeric(cr["interest_coverage_ratio"], errors="coerce")
    return {
        "has_date_col": "date" in cr.columns,
        "rows": int(len(cr)),
        "rating_values": sorted(set(rt[rt != "nan"].unique())),
        "rating_valid_ladder": int(valid.sum()),
        "rating_nonrated_or_nan": int(nonrated.sum()),
        "rating_unknown_values": unknown,
        "altman_min": float(np.nanmin(z)),
        "altman_median": float(np.nanmedian(z)),
        "altman_max": float(np.nanmax(z)),
        "altman_nan": int(z.isna().sum()),
        "altman_negative": int((z < 0).sum()),
        "altman_abs_gt_100": int((z.abs() > 100).sum()),
        "intcov_min": float(np.nanmin(ic)),
        "intcov_max": float(np.nanmax(ic)),
        "intcov_nan": int(ic.isna().sum()),
    }


def probe_dividends(frontier: str) -> dict[str, Any]:
    dv = _read("sp500_dividends.csv")
    da = pd.to_numeric(dv["dividend_amount"], errors="coerce")
    ed = pd.to_datetime(dv["ex_date"], errors="coerce")
    front = pd.Timestamp(frontier)
    return {
        "rows": int(len(dv)),
        "negative_count": int((da < 0).sum()),
        "negative_min": float(da.min()),
        "materially_negative_lt_-1e-3": int((da < -1e-3).sum()),
        "future_declared_rows": int((ed > front).sum()),
        "ex_date_max": str(ed.max())[:10],
        "dividend_type_values": sorted(set(dv["dividend_type"].astype(str).unique()))[:15],
    }


def probe_treasury() -> dict[str, Any]:
    tr = _read("treasury_yields.csv")
    tr["d"] = pd.to_datetime(tr["date"], errors="coerce")
    out: dict[str, Any] = {"rows": int(len(tr)), "tenors": {}}
    for col in [x for x in tr.columns if x.startswith("rate") or x == "sofr"]:
        s = pd.to_numeric(tr[col], errors="coerce")
        first = tr.loc[s.notna(), "d"].min()
        out["tenors"][col] = {
            "first_nonnull": str(first)[:10] if pd.notna(first) else None,
            "nan_pct": round(float(s.isna().mean()) * 100, 1),
            "min": float(np.nanmin(s)) if s.notna().any() else None,
            "negative_count": int((s < 0).sum()),
        }
    return out


def probe_e2e(runner: WheelRunner, as_of: str) -> dict[str, Any]:
    """data -> WheelRunner.rank_candidates_by_ev -> EVResult: finite + sign-bearing."""
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "UNH"]
    frame = runner.rank_candidates_by_ev(
        tickers=tickers,
        top_n=20,
        min_ev_dollars=-1e9,
        as_of=as_of,
        include_diagnostic_fields=True,
    )
    out: dict[str, Any] = {"as_of": as_of, "tickers": tickers, "n_produced": int(len(frame))}
    if not len(frame):
        out["note"] = "no rows produced"
        return out
    nonfinite: dict[str, int] = {}
    for col in ("ev_dollars", "ev_raw", "prob_profit"):
        if col in frame.columns:
            s = pd.to_numeric(frame[col], errors="coerce")
            nonfinite[col] = int((~np.isfinite(s)).sum())
    ev = pd.to_numeric(frame["ev_dollars"], errors="coerce")
    top = frame.loc[ev.idxmax()]
    bot = frame.loc[ev.idxmin()]
    out.update(
        nonfinite_counts=nonfinite,
        ev_positive=int((ev > 0).sum()),
        ev_negative=int((ev < 0).sum()),
        ev_min=float(ev.min()),
        ev_max=float(ev.max()),
        example_pos=f"{top['ticker']} ev_dollars={float(top['ev_dollars']):.2f}",
        example_neg=f"{bot['ticker']} ev_dollars={float(bot['ev_dollars']):.2f}",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase-1 data-test-coverage audit probes (discovery).")
    ap.add_argument("--as-of", default="auto", help="YYYY-MM-DD or 'auto' (frontier)")
    ap.add_argument("--json", default=None, help="JSON sidecar path")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass

    runner = WheelRunner()
    conn = runner.connector
    provider = type(conn).__name__
    print(f"[audit-tests] provider={provider}")
    frontier = _frontier()
    as_of = frontier if args.as_of == "auto" else args.as_of
    print(f"[audit-tests] frontier={frontier} as_of={as_of}")

    payload = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider,
            "frontier": frontier,
            "as_of": as_of,
        },
        "iv": probe_iv(conn),
        "ohlcv": probe_ohlcv(),
        "fundamentals": probe_fundamentals(),
        "credit": probe_credit(),
        "dividends": probe_dividends(frontier),
        "treasury": probe_treasury(),
        "e2e": probe_e2e(runner, as_of),
    }

    text = json.dumps(payload, indent=2, default=str)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
        print(f"[audit-tests] wrote json -> {args.json}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
