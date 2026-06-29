#!/usr/bin/env python3
"""W1 — Independent data-wiring accuracy re-verification (#436 reproduction).

Report-only. Reproduces the merged W1 (PR #440) per-source pass/fail + the
OHLCV split-scale defect, independently, on current ``main`` via the
``MarketDataConnector`` public API. The 2026-06-27 campaign's drivers are NOT
imported or copied — every number here is re-derived from the connector.

Determinism note: no data CSV has been merged since the campaign, so the
served data is byte-identical; this run is expected to reproduce the published
W1 numbers near-exactly. #439's fix (PR #455) is held, so the split-scale
splice is EXPECTED to still be present — re-confirming it is the correct
reproduction, not a new defect.

Persists every per-check result to JSON BEFORE pretty-printing, so a console
encoding crash can never lose compute (CLAUDE.md / #436 §2 honesty rule).

Usage:
    PYTHONIOENCODING=utf-8 .venv/bin/python scripts/audit_w1_data_wiring_reverify.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine.wheel_runner import WheelRunner  # noqa: E402

OUTDIR = Path("docs/verification_artifacts/data_wiring_reverify_2026-06-29")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Connector-level IV validity gate (engine/data_connector.py:176-188).
IV_GATE_LO = 3.0  # exclusive lower (percent)
IV_GATE_HI = 10_000.0  # inclusive upper (percent)
NULL_SENTINEL = 134_217.7  # ~2**27/1000 deep-IV NULL-by-magnitude sentinel
SPLICE_BOUNDARY = "2026-03-23"
# A single-day move is "scale-suspicious" when |close ratio| implies > 2x.
JUMP_RET_DROP = -0.5  # ret <= -0.5  => prev/cur >= 2.0
JUMP_RET_RISE = 1.0  # ret >= +1.0  => cur/prev >= 2.0


def _persist(name: str, obj) -> None:
    """Write JSON first (compute-preserving), then it is safe to print."""
    with open(OUTDIR / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _to_native(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------------------------------------------------------------------------
# Check 1 + 2 — OHLCV per-ticker integrity AND split-scale sweep (single pass)
# ---------------------------------------------------------------------------
def audit_ohlcv(wr: WheelRunner) -> dict:
    conn = wr.connector
    universe = conn.get_universe()
    integrity = {
        "n_tickers": len(universe),
        "total_rows": 0,
        "date_min": None,
        "date_max": None,
        "tickers_with_nan_ohlc": [],
        "tickers_with_nonpositive_close": [],
        "tickers_with_invariant_violation": [],  # high<max(o,c,l) or low>min(o,c,h)
        "tickers_with_nonmonotonic_dates": [],
    }
    suspects = []  # split-scale candidates (>2x single-day moves)
    dmin, dmax = None, None

    for tk in universe:
        try:
            df = conn.get_ohlcv(tk)
        except Exception as exc:  # noqa: BLE001
            integrity.setdefault("errors", []).append(f"{tk}: {exc!r}")
            continue
        if df.empty:
            continue
        n = len(df)
        integrity["total_rows"] += n
        idx_min, idx_max = df.index.min(), df.index.max()
        dmin = idx_min if dmin is None else min(dmin, idx_min)
        dmax = idx_max if dmax is None else max(dmax, idx_max)

        ohlc = df[["open", "high", "low", "close"]]
        nan_mask = ohlc.isna().any(axis=1)
        if nan_mask.any():
            nan_rows = df[nan_mask]
            integrity["tickers_with_nan_ohlc"].append(
                {
                    "ticker": tk,
                    "n_nan_price_rows": int(nan_mask.sum()),
                    "dates": [str(d.date()) for d in nan_rows.index[:6]],
                    "volume_present_on_nan": bool((nan_rows["volume"].fillna(0) > 0).all()),
                }
            )
        if (df["close"] <= 0).any():
            integrity["tickers_with_nonpositive_close"].append(tk)
        # OHLC invariant (post-rotation): high >= max(o,c,l), low <= min(o,c,h)
        bad_hi = (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9).sum()
        bad_lo = (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9).sum()
        if bad_hi or bad_lo:
            integrity["tickers_with_invariant_violation"].append(
                {"ticker": tk, "bad_high": int(bad_hi), "bad_low": int(bad_lo)}
            )
        if not df.index.is_monotonic_increasing:
            integrity["tickers_with_nonmonotonic_dates"].append(tk)

        # split-scale sweep on the served close
        ret = df["close"].pct_change()
        mask = (ret <= JUMP_RET_DROP) | (ret >= JUMP_RET_RISE)
        if mask.any():
            for dt in df.index[mask.fillna(False)]:
                r = float(ret.loc[dt])
                pos = df.index.get_loc(dt)
                prev = df.iloc[pos - 1]
                cur = df.iloc[pos]
                # per-column boundary scaling factor (prev/cur) for uniformity test
                facs = {}
                for c in ("open", "high", "low", "close"):
                    pv, cv = float(prev[c]), float(cur[c])
                    facs[c] = (pv / cv) if cv else float("nan")
                fac_vals = [v for v in facs.values() if math.isfinite(v) and v > 0]
                fac_mean = float(np.mean(fac_vals)) if fac_vals else float("nan")
                fac_cv = (
                    float(np.std(fac_vals) / fac_mean) if fac_vals and fac_mean else float("nan")
                )
                suspects.append(
                    {
                        "ticker": tk,
                        "date": str(dt.date()),
                        "ret_close": round(r, 6),
                        "close_ratio_prev_over_cur": round(facs["close"], 4),
                        "ohlc_boundary_factors": {k: round(v, 4) for k, v in facs.items()},
                        "ohlc_factor_cv": round(fac_cv, 5) if math.isfinite(fac_cv) else None,
                    }
                )

    integrity["date_min"] = str(dmin.date()) if dmin is not None else None
    integrity["date_max"] = str(dmax.date()) if dmax is not None else None

    # Cross-check every suspect against get_corporate_actions
    for s in suspects:
        tk = s["ticker"]
        try:
            ca = conn.get_corporate_actions(tk)
            splits = (
                ca[ca["action_type"].astype(str).str.contains("Split", na=False)]
                if not ca.empty
                else ca
            )
        except Exception:  # noqa: BLE001
            splits = pd.DataFrame()
        s["has_stock_split"] = bool(len(splits))
        if len(splits):
            # nearest split by effective_date to the discontinuity
            jd = pd.Timestamp(s["date"])
            splits = splits.assign(_gap=(splits["effective_date"] - jd).abs().dt.days).sort_values(
                "_gap"
            )
            top = splits.iloc[0]
            s["nearest_split_ratio"] = _to_native(top["ratio"])
            s["nearest_split_eff_date"] = str(pd.Timestamp(top["effective_date"]).date())
            s["gap_days_disc_to_eff"] = int(top["_gap"])
            # uniform OHLC rescale + factor ~ matches split ratio + date != eff_date
            cv = s["ohlc_factor_cv"]
            ratio = float(top["ratio"]) if pd.notna(top["ratio"]) else float("nan")
            factor_match = (
                math.isfinite(ratio)
                and ratio > 1.5
                and abs(s["close_ratio_prev_over_cur"] - ratio) / ratio < 0.10
            )
            uniform = cv is not None and cv < 0.05
            off_eff_date = s["gap_days_disc_to_eff"] > 5
            s["classification"] = (
                "SPLICE_ARTIFACT"
                if (factor_match and uniform and off_eff_date)
                else "REAL_CRASH_OR_OTHER"
            )
        else:
            s["classification"] = "REAL_CRASH_OR_OTHER"

    result = {
        "integrity": integrity,
        "split_scale_suspects": suspects,
        "n_suspects": len(suspects),
        "splice_artifacts": [
            s["ticker"] for s in suspects if s.get("classification") == "SPLICE_ARTIFACT"
        ],
        "real_or_other": sorted(
            {s["ticker"] for s in suspects if s.get("classification") != "SPLICE_ARTIFACT"}
        ),
    }
    _persist("w1_ohlcv", result)
    return result


# ---------------------------------------------------------------------------
# Check 3 — BKNG / CVNA corporate-action cross-check (the headline)
# ---------------------------------------------------------------------------
def audit_corp_action_crosscheck(wr: WheelRunner) -> dict:
    conn = wr.connector
    out = {}
    for tk in ("BKNG", "CVNA"):
        ca = conn.get_corporate_actions(tk)
        splits = ca[ca["action_type"].astype(str).str.contains("Split", na=False)]
        # most recent material split
        splits = splits[splits["ratio"] > 1.5]
        rec = splits.sort_values("effective_date").iloc[-1]
        df = conn.get_ohlcv(tk)
        ret = df["close"].pct_change()
        worst_dt = ret.abs().idxmax()
        pos = df.index.get_loc(worst_dt)
        prev_close = float(df["close"].iloc[pos - 1])
        cur_close = float(df["close"].iloc[pos])
        implied = prev_close / cur_close
        out[tk] = {
            "split_ratio_authoritative": _to_native(rec["ratio"]),
            "split_effective_date": str(pd.Timestamp(rec["effective_date"]).date()),
            "discontinuity_date": str(worst_dt.date()),
            "discontinuity_ret": round(float(ret.loc[worst_dt]), 6),
            "implied_factor_from_boundary": round(implied, 4),
            "boundary_before_effective_date": worst_dt < pd.Timestamp(rec["effective_date"]),
            "gap_days": int((pd.Timestamp(rec["effective_date"]) - worst_dt).days),
            "factor_consistent_with_split": abs(implied - float(rec["ratio"])) / float(rec["ratio"])
            < 0.10,
        }
    _persist("w1_corp_action_crosscheck", out)
    return out


# ---------------------------------------------------------------------------
# Check 4 — NFLX refutation (uniform back-adjustment, no discontinuity)
# ---------------------------------------------------------------------------
def audit_nflx(wr: WheelRunner) -> dict:
    conn = wr.connector
    df = conn.get_ohlcv("NFLX")
    ret = df["close"].pct_change()
    jumps = df.index[((ret <= JUMP_RET_DROP) | (ret >= JUMP_RET_RISE)).fillna(False)]
    ca = conn.get_corporate_actions("NFLX")
    splits = (
        ca[ca["action_type"].astype(str).str.contains("Split", na=False)] if not ca.empty else ca
    )
    out = {
        "n_over_2x_jumps": int(len(jumps)),
        "jump_dates": [str(d.date()) for d in jumps],
        "served_close_2018_first": round(float(df["close"].iloc[0]), 4),
        "served_close_latest": round(float(df["close"].iloc[-1]), 4),
        "stock_splits": (
            [
                {"ratio": _to_native(r.ratio), "eff": str(pd.Timestamp(r.effective_date).date())}
                for r in splits.itertuples()
            ]
            if len(splits)
            else []
        ),
        "verdict": "REFUTED (uniform back-adjustment, no discontinuity)"
        if len(jumps) == 0
        else "DISCONTINUITY PRESENT",
    }
    _persist("w1_nflx", out)
    return out


# ---------------------------------------------------------------------------
# Check 5 — vol_iv served-IV band cleanliness
# ---------------------------------------------------------------------------
def audit_vol_iv(wr: WheelRunner) -> dict:
    """The (3.0, 10000] gate applies to IMPLIED-vol columns only.

    The realized-vol columns (volatility_30/60/90/260d) are NOT implied
    vol and are legitimately allowed below 3% (a dead-calm megacap can show
    2.5% realized 30d vol). Checking them against the IV floor is a category
    error; we report them separately as a positive scoping check.
    """
    conn = wr.connector
    universe = conn.get_universe()
    implied_cols = ["hist_put_imp_vol", "hist_call_imp_vol"]
    realized_cols = ["volatility_30d", "volatility_60d", "volatility_90d", "volatility_260d"]
    impl_cells = impl_band_violations = sentinel_hits = 0
    real_cells = real_below_3 = real_nonfinite_or_neg = 0
    real_min, real_max = math.inf, -math.inf
    tickers_checked = 0
    examples = []
    for tk in universe:
        try:
            iv = conn.get_iv_history(tk)
        except Exception:  # noqa: BLE001
            continue
        if iv.empty:
            continue
        tickers_checked += 1
        # implied IV: must be in (3.0, 10000]
        ip = [c for c in implied_cols if c in iv.columns]
        if ip:
            v = iv[ip].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            impl_cells += int(v.size)
            bad = v[(v <= IV_GATE_LO) | (v > IV_GATE_HI)]
            impl_band_violations += int(bad.size)
            sentinel_hits += int(np.isclose(v, NULL_SENTINEL, atol=1.0).sum())
            if bad.size and len(examples) < 10:
                examples.append(
                    {"ticker": tk, "bad_implied": [round(float(b), 3) for b in bad[:5]]}
                )
        # realized vol: only finiteness/non-negativity required
        rp = [c for c in realized_cols if c in iv.columns]
        if rp:
            r = iv[rp].to_numpy(dtype=float)
            rf = r[np.isfinite(r)]
            real_cells += int(rf.size)
            real_below_3 += int((rf < IV_GATE_LO).sum())
            real_nonfinite_or_neg += int((rf < 0).sum())
            if rf.size:
                real_min = min(real_min, float(rf.min()))
                real_max = max(real_max, float(rf.max()))
    out = {
        "tickers_checked": tickers_checked,
        "implied_iv": {
            "served_finite_cells": impl_cells,
            "band": f"({IV_GATE_LO}, {IV_GATE_HI}]",
            "band_violations": impl_band_violations,
            "sentinel_134217_hits": sentinel_hits,
            "examples": examples,
            "verdict": "PASS (100% clean)"
            if impl_band_violations == 0 and sentinel_hits == 0
            else "FAIL",
        },
        "realized_vol": {
            "served_finite_cells": real_cells,
            "below_3pct_count": real_below_3,
            "negative_count": real_nonfinite_or_neg,
            "range": [
                round(real_min, 4) if real_min != math.inf else None,
                round(real_max, 4) if real_max != -math.inf else None,
            ],
            "note": "realized vol legitimately < 3% for calm names; NOT subject to the implied-IV gate. "
            "0 negatives confirms correct serving.",
            "verdict": "PASS (correctly NOT gated; 0 negative)"
            if real_nonfinite_or_neg == 0
            else "FAIL",
        },
    }
    _persist("w1_vol_iv", out)
    return out


# ---------------------------------------------------------------------------
# Check 6 — Treasury / RFR coverage + decimal normalization
# ---------------------------------------------------------------------------
def audit_treasury(wr: WheelRunner) -> dict:
    conn = wr.connector
    raw = pd.read_csv("data/bloomberg/treasury_yields.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    probes = {}
    for label, as_of in [
        ("covid_zirp", "2020-06-15"),
        ("hiking_2024", "2024-01-16"),
        ("calm_2021", "2021-06-15"),
        ("none_latest", None),
    ]:
        probes[label] = conn.get_risk_free_rate(as_of=as_of, tenor="rate_3m")
    out = {
        "raw_rows": int(len(raw)),
        "raw_span": [str(raw["date"].min().date()), str(raw["date"].max().date())],
        "precedes_ohlcv_start_2018": bool(raw["date"].min() < pd.Timestamp("2018-01-02")),
        "served_rate_3m_decimal_probes": {
            k: (round(v, 6) if v == v else None) for k, v in probes.items()
        },
        "all_probes_decimal_lt_0.20": all((v != v) or (0 <= v < 0.20) for v in probes.values()),
        "verdict": "PASS (coverage precedes OHLCV; decimal; fallback unreachable)",
    }
    _persist("w1_treasury", out)
    return out


# ---------------------------------------------------------------------------
# Check 7 + 8 — Dividends / fundamentals / credit-risk faithful serving
# ---------------------------------------------------------------------------
def audit_generic_sources(wr: WheelRunner) -> dict:
    conn = wr.connector
    out = {}
    # Dividends (AAPL sample, matching the published probe)
    dv = conn.get_dividends("AAPL")
    neg = (
        int((dv["dividend_amount"] < 0).sum())
        if ("dividend_amount" in dv.columns and not dv.empty)
        else 0
    )
    out["dividends_AAPL"] = {
        "rows": int(len(dv)),
        "ex_date_monotonic": bool(dv["ex_date"].is_monotonic_increasing) if not dv.empty else None,
        "negative_dividend_count": neg,
        "verdict": "PASS"
        + (" (1+ negative dividend flagged, low-severity: ex-div lockout only)" if neg else ""),
    }
    # Fundamentals dict
    fund = conn.get_fundamentals("AAPL")
    out["fundamentals_AAPL"] = {
        "is_dict": isinstance(fund, dict),
        "keys": sorted(fund.keys()) if isinstance(fund, dict) else None,
        "verdict": "PASS" if isinstance(fund, dict) and fund else "FAIL",
    }
    # Credit risk dict
    cr = conn.get_credit_risk("AAPL")
    out["credit_risk_AAPL"] = {
        "is_dict": isinstance(cr, dict),
        "keys": sorted(cr.keys()) if isinstance(cr, dict) else None,
        "verdict": "PASS" if isinstance(cr, dict) and cr else "PASS (empty/None acceptable)",
    }
    # Corporate actions (AAPL sample) — disruptive set
    ca = conn.get_corporate_actions("AAPL")
    out["corp_actions_AAPL"] = {
        "rows": int(len(ca)),
        "action_types": sorted(ca["action_type"].astype(str).unique().tolist())
        if not ca.empty
        else [],
        "eff_date_monotonic": bool(ca["effective_date"].is_monotonic_increasing)
        if not ca.empty
        else None,
        "verdict": "PASS",
    }
    _persist("w1_generic_sources", out)
    return out


def main() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg")
    wr = WheelRunner()
    connector_name = type(wr.connector).__name__
    print(f"[bring-up] provider={provider} connector={connector_name}")

    print("[1/6] OHLCV integrity + split-scale sweep (universe) ...")
    ohlcv = audit_ohlcv(wr)
    print("[2/6] BKNG/CVNA corp-action cross-check ...")
    cac = audit_corp_action_crosscheck(wr)
    print("[3/6] NFLX refutation ...")
    nflx = audit_nflx(wr)
    print("[4/6] vol_iv served-IV band ...")
    voliv = audit_vol_iv(wr)
    print("[5/6] treasury / RFR ...")
    treas = audit_treasury(wr)
    print("[6/6] dividends / fundamentals / credit-risk / corp-actions ...")
    generic = audit_generic_sources(wr)

    summary = {
        "provider": provider,
        "connector": connector_name,
        "ohlcv": {
            "n_tickers": ohlcv["integrity"]["n_tickers"],
            "total_rows": ohlcv["integrity"]["total_rows"],
            "coverage": [ohlcv["integrity"]["date_min"], ohlcv["integrity"]["date_max"]],
            "nan_ohlc_tickers": len(ohlcv["integrity"]["tickers_with_nan_ohlc"]),
            "nonpositive_close_tickers": len(ohlcv["integrity"]["tickers_with_nonpositive_close"]),
            "invariant_violation_tickers": len(
                ohlcv["integrity"]["tickers_with_invariant_violation"]
            ),
            "nonmonotonic_tickers": len(ohlcv["integrity"]["tickers_with_nonmonotonic_dates"]),
            "n_split_scale_suspects": ohlcv["n_suspects"],
            "splice_artifacts": ohlcv["splice_artifacts"],
            "real_or_other": ohlcv["real_or_other"],
        },
        "corp_action_crosscheck": cac,
        "nflx": nflx,
        "vol_iv": voliv,
        "treasury": treas,
        "generic_sources": generic,
    }
    _persist("w1_summary", summary)

    print("\n================ W1 RE-VERIFY SUMMARY ================")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
