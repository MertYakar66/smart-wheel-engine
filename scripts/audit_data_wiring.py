#!/usr/bin/env python3
"""W1 — Data-wiring accuracy audit (heavy-verify 2026-06-27, Mac terminal / #436).

Validation-only. Verifies that the ``MarketDataConnector`` serves every
committed Bloomberg source faithfully: row counts, date coverage, monotonic
dates, NaN / sentinel leakage (the deep-IV ``134217.7`` NULL-by-magnitude
sentinel and the sub-3% IV floor), unit sanity, and the OHLCV split-scale
discontinuity sweep (the NFLX / BKNG / CVNA suspected ~10x mis-scale).

Does NOT touch engine behaviour. Persists every result to JSON *before*
pretty-printing (a console-encoding crash must never lose compute).

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_data_wiring.py
Out:  docs/verification_artifacts/data_wiring_2026-06-27/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine.data_connector import MarketDataConnector  # noqa: E402

OUT = REPO / "docs" / "verification_artifacts" / "data_wiring_2026-06-27"
OUT.mkdir(parents=True, exist_ok=True)

# Tickers sampled for the per-ticker structural checks (liquid, full-history).
SAMPLE = ["AAPL", "MSFT", "JPM", "XOM", "UNH", "NFLX", "BKNG", "CVNA", "KO", "PG"]


def _dump(name: str, obj: object) -> None:
    """Persist a result to JSON immediately (before any console print)."""
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _ts(x) -> str | None:
    return None if x is None or pd.isna(x) else str(pd.Timestamp(x).date())


# ---------------------------------------------------------------------------
# OHLCV — structure + split-scale discontinuity sweep
# ---------------------------------------------------------------------------
def audit_ohlcv(c: MarketDataConnector) -> dict:
    univ = c.get_universe()
    # Per-ticker structural checks on the sample.
    per_ticker = {}
    for t in SAMPLE:
        df = c.get_ohlcv(t)
        if df.empty:
            per_ticker[t] = {"rows": 0, "status": "EMPTY"}
            continue
        idx = df.index
        mono = bool(idx.is_monotonic_increasing)
        dup = int(idx.duplicated().sum())
        # post-rename OHLC invariant: high >= max(o,c,l) and low <= min(o,c,h)
        s = df.dropna(subset=["open", "high", "low", "close"])
        bad_hi = int((s["high"] < s[["open", "close", "low"]].max(axis=1)).sum())
        bad_lo = int((s["low"] > s[["open", "close", "high"]].min(axis=1)).sum())
        nan_close = int(df["close"].isna().sum())
        nonpos = int((df["close"] <= 0).sum())
        per_ticker[t] = {
            "rows": int(len(df)),
            "date_min": _ts(idx.min()),
            "date_max": _ts(idx.max()),
            "monotonic": mono,
            "dup_dates": dup,
            "ohlc_invariant_violations": bad_hi + bad_lo,
            "nan_close": nan_close,
            "nonpositive_close": nonpos,
        }

    # Universe-wide split-scale discontinuity sweep: any close ratio outside
    # [0.5, 2.0] is a >2x single-day move (a candidate split-scale break).
    discontinuities = {}
    for t in univ:
        df = c.get_ohlcv(t)
        if df.empty or "close" not in df.columns:
            continue
        ser = df["close"].dropna()
        if len(ser) < 2:
            continue
        r = ser / ser.shift(1)
        big = r[(r < 0.5) | (r > 2.0)]
        if big.empty:
            continue
        ca = c.get_corporate_actions(t)
        sp = (
            ca[ca["action_type"].astype(str).str.strip() == "Stock Split"]
            if (not ca.empty and "action_type" in ca.columns)
            else pd.DataFrame()
        )
        splits = []
        if not sp.empty:
            for d, rt in zip(sp.get("effective_date", []), sp.get("ratio", []), strict=False):
                splits.append({"eff_date": _ts(d), "ratio": float(rt) if pd.notna(rt) else None})
        recs = []
        for d, ratio in big.items():
            recs.append(
                {
                    "date": _ts(d),
                    "prev_close": round(float(ser.shift(1).loc[d]), 4),
                    "close": round(float(ser.loc[d]), 4),
                    "ratio": round(float(ratio), 4),
                    "implied_factor": round(1.0 / float(ratio), 3) if ratio else None,
                }
            )
        discontinuities[t] = {"splits": splits, "jumps": recs}

    return {
        "universe_size": len(univ),
        "per_ticker": per_ticker,
        "discontinuity_count": len(discontinuities),
        "discontinuities": discontinuities,
    }


# ---------------------------------------------------------------------------
# vol_iv — IV sentinel + low-floor leakage; PERCENT-band verification
# ---------------------------------------------------------------------------
def audit_vol_iv(c: MarketDataConnector) -> dict:
    raw = pd.read_csv(REPO / "data" / "bloomberg" / "sp500_vol_iv_full.csv", low_memory=False)
    iv_cols = ["hist_put_imp_vol", "hist_call_imp_vol"]
    raw_stats = {}
    for col in iv_cols:
        v = pd.to_numeric(raw[col], errors="coerce")
        raw_stats[col] = {
            "rows": int(len(v)),
            "nan": int(v.isna().sum()),
            "le_3 (sub-floor)": int((v <= 3.0).sum()),
            "gt_10000 (sentinel)": int((v > 10_000).sum()),
            "near_134217_sentinel": int((v.between(134_000, 134_500)).sum()),
            "min": None if v.notna().sum() == 0 else round(float(v.min()), 4),
            "max": None if v.notna().sum() == 0 else round(float(v.max()), 4),
        }

    # Served band check: every served IV cell must be in (3.0, 10000].
    served_violations = []
    served_total = 0
    for t in c.get_universe():
        iv = c.get_iv_history(t)
        if iv.empty:
            continue
        for col in iv_cols:
            if col not in iv.columns:
                continue
            v = pd.to_numeric(iv[col], errors="coerce").dropna()
            served_total += int(len(v))
            bad = v[(v <= 3.0) | (v > 10_000)]
            if not bad.empty:
                served_violations.append(
                    {
                        "ticker": t,
                        "col": col,
                        "n": int(len(bad)),
                        "examples": [round(float(x), 4) for x in bad.head(3)],
                    }
                )
    return {
        "raw_csv": raw_stats,
        "served_iv_cells_checked": served_total,
        "served_band_violations": served_violations,
        "served_band_clean": len(served_violations) == 0,
        "note": "connector NULLs IV outside (3.0, 10000] on every read "
        "(_clean_vol_iv_inplace); served band should be clean even though "
        "the raw CSV carries sub-3 + sentinel cells.",
    }


# ---------------------------------------------------------------------------
# treasury / risk-free — coverage + unit sanity
# ---------------------------------------------------------------------------
def audit_treasury(c: MarketDataConnector) -> dict:
    raw = pd.read_csv(REPO / "data" / "bloomberg" / "treasury_yields.csv")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    tenors = [col for col in raw.columns if col.startswith("rate_") or col == "sofr"]
    cov = {}
    for ten in tenors:
        v = raw.dropna(subset=[ten])
        cov[ten] = {
            "coverage_start": _ts(v["date"].min()),
            "coverage_end": _ts(v["date"].max()),
            "rows": int(len(v)),
            "min_pct": round(float(v[ten].min()), 4) if len(v) else None,
            "max_pct": round(float(v[ten].max()), 4) if len(v) else None,
        }
    # served decimal sanity across regimes
    served = {}
    for asof in ["2018-06-01", "2020-03-15", "2021-05-01", "2024-01-02", "2026-06-01"]:
        served[asof] = round(c.get_risk_free_rate(asof, "rate_3m"), 6)
    return {
        "coverage": cov,
        "served_rate_3m_decimal": served,
        "monotonic_dates": bool(raw["date"].is_monotonic_increasing),
    }


# ---------------------------------------------------------------------------
# Generic structural pass over the remaining sources
# ---------------------------------------------------------------------------
def audit_generic_sources(c: MarketDataConnector) -> dict:
    out = {}
    # dividends
    div = c.get_dividends("AAPL")
    out["dividends"] = {
        "sample_ticker": "AAPL",
        "rows": int(len(div)),
        "ex_date_monotonic": bool(div["ex_date"].is_monotonic_increasing)
        if not div.empty
        else None,
        "neg_amount": int((pd.to_numeric(div.get("dividend_amount"), errors="coerce") < 0).sum())
        if not div.empty
        else None,
    }
    # corporate actions
    ca = c.get_corporate_actions("AAPL")
    out["corporate_actions"] = {
        "sample_ticker": "AAPL",
        "rows_disruptive": int(len(ca)),
        "eff_date_monotonic": bool(ca["effective_date"].is_monotonic_increasing)
        if not ca.empty
        else None,
        "action_types": sorted(ca["action_type"].astype(str).unique().tolist())
        if not ca.empty
        else [],
    }
    # fundamentals
    fund = c.get_fundamentals("AAPL")
    out["fundamentals"] = {
        "sample_ticker": "AAPL",
        "present": fund is not None,
        "keys": sorted(fund.keys()) if isinstance(fund, dict) else None,
        "implied_vol_atm": fund.get("implied_vol_atm") if isinstance(fund, dict) else None,
        "beta": fund.get("beta") if isinstance(fund, dict) else None,
    }
    # credit risk
    cr = c.get_credit_risk("AAPL")
    out["credit_risk"] = {
        "sample_ticker": "AAPL",
        "present": cr is not None,
        "keys": sorted(cr.keys()) if isinstance(cr, dict) else None,
    }
    # vix
    vix = c.get_vix()
    out["vix"] = {
        "rows": int(len(vix)),
        "date_min": _ts(vix.index.min()) if not vix.empty else None,
        "date_max": _ts(vix.index.max()) if not vix.empty else None,
        "vix_min": round(float(vix["vix"].min()), 2) if not vix.empty else None,
        "vix_max": round(float(vix["vix"].max()), 2) if not vix.empty else None,
        "monotonic": bool(vix.index.is_monotonic_increasing) if not vix.empty else None,
    }
    # earnings
    earn = (
        c.get_recent_earnings("AAPL", as_of="2024-01-02")
        if hasattr(c, "get_recent_earnings")
        else None
    )
    out["earnings"] = {
        "sample_ticker": "AAPL",
        "recent_rows": int(len(earn)) if isinstance(earn, pd.DataFrame) else None,
    }
    return out


def main() -> None:
    c = MarketDataConnector()
    results = {}

    results["ohlcv"] = audit_ohlcv(c)
    _dump("w1_ohlcv", results["ohlcv"])

    results["vol_iv"] = audit_vol_iv(c)
    _dump("w1_vol_iv", results["vol_iv"])

    results["treasury"] = audit_treasury(c)
    _dump("w1_treasury", results["treasury"])

    results["generic"] = audit_generic_sources(c)
    _dump("w1_generic_sources", results["generic"])

    _dump("w1_summary", results)

    # ---- console summary (after JSON is safely on disk) ----
    print("=== W1 DATA-WIRING AUDIT ===")
    o = results["ohlcv"]
    print(f"OHLCV: universe={o['universe_size']}, discontinuity tickers={o['discontinuity_count']}")
    for t, rec in o["discontinuities"].items():
        for j in rec["jumps"]:
            print(
                f"   {t} {j['date']}: {j['prev_close']} -> {j['close']} "
                f"(x{j['implied_factor']}) splits={rec['splits']}"
            )
    v = results["vol_iv"]
    print(
        f"vol_iv served band clean: {v['served_band_clean']} "
        f"({v['served_iv_cells_checked']} cells checked)"
    )
    tr = results["treasury"]
    print(
        f"treasury rate_3m coverage: {tr['coverage']['rate_3m']['coverage_start']} "
        f"-> {tr['coverage']['rate_3m']['coverage_end']}"
    )
    print(f"served rate_3m decimal: {tr['served_rate_3m_decimal']}")
    print(f"\nJSON written to {OUT}")


if __name__ == "__main__":
    main()
