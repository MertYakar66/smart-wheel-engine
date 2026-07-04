#!/usr/bin/env python3
"""W2 — Independent engine output-realism re-verification (#436 reproduction).

Report-only. Reproduces the merged W2 (PR #441) realism table independently on
current ``main``: runs ``rank_candidates_by_ev`` over the full universe at the
same 5 regime as_of dates and re-checks finiteness, prob bounds, served-IV
band, premium/spot sanity, Greek-contract units, and regime magnitude scaling.

The 2026-06-27 driver is NOT imported. Greeks are recomputed from the canonical
``black_scholes_all_greeks`` (the ranker emits no Greek columns), exactly as the
W2 contract requires. The engine is deterministic (forward_distribution seed=42;
per-trade blake2b seed), and no data CSV has changed since the campaign, so this
run is expected to reproduce the published counts near-exactly.

Persists per-regime JSON BEFORE pretty-printing.

Usage:
    PYTHONIOENCODING=utf-8 .venv/bin/python scripts/audit_w2_output_realism_reverify.py
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

from engine.option_pricer import black_scholes_all_greeks  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

OUTDIR = Path("docs/verification_artifacts/data_wiring_reverify_2026-06-29")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Same 5 regime as_of dates the merged W2 driver used.
AS_OF = [
    ("calm_2021", "2021-06-15"),
    ("calm_2024", "2024-01-16"),
    ("crisis_2020", "2020-03-23"),
    ("bear_2022_jun", "2022-06-16"),
    ("bear_2022_oct", "2022-10-14"),
]
TOP_N = 600
IV_FLOOR_PCT = 3.0  # served IV (percent) must exceed this (connector gate lower bound)
IV_CEILING_PCT = 10_000.0  # served IV (percent) must not exceed this (connector gate upper bound)


def _persist(name: str, obj) -> None:
    with open(OUTDIR / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _greek_violations(row: pd.Series, rfr: float) -> str | None:
    """Recompute the canonical Greeks for a 25Δ short put; return a violation
    string or None. Contract (docs/GREEKS_UNIT_CONTRACT.md): put delta in
    [-1, 0]; gamma >= 0; vega >= 0."""
    try:
        S = float(row["spot"])
        K = float(row["strike"])
        T = float(row["dte"]) / 365.0
        sigma = float(row["iv"])
        if not all(_finite(v) for v in (S, K, T, sigma, rfr)) or T <= 0 or sigma <= 0 or S <= 0:
            return "non-evaluable greek inputs"
        g = black_scholes_all_greeks(S, K, T, rfr, sigma, "put", include_second_order=False)
        bad = []
        if not (-1.0 <= g["delta"] <= 0.0):
            bad.append(f"delta {g['delta']:.3f} outside [-1,0]")
        if g["gamma"] < 0:
            bad.append(f"gamma {g['gamma']:.5f} < 0")
        if g["vega"] < 0:
            bad.append(f"vega {g['vega']:.5f} < 0")
        return "; ".join(bad) if bad else None
    except Exception as exc:  # noqa: BLE001
        return f"greek raise: {exc!r}"


def audit_regime(wr: WheelRunner, label: str, as_of: str) -> dict:
    rfr = wr.connector.get_risk_free_rate(as_of=as_of, tenor="rate_3m")
    df = wr.rank_candidates_by_ev(
        as_of=as_of,
        top_n=TOP_N,
        min_ev_dollars=-1e9,
        include_diagnostic_fields=True,
    )
    n = len(df)
    res = {
        "label": label,
        "as_of": as_of,
        "rfr_decimal": round(float(rfr), 6),
        "n_candidates": n,
    }
    if n == 0:
        _persist(f"w2_{label}", res)
        return res

    # --- finiteness across the key served fields ---
    key_fields = ["ev_dollars", "premium", "iv", "spot", "prob_profit", "prob_assignment"]
    nonfinite = {}
    for c in key_fields:
        if c in df.columns:
            nf = int((~df[c].apply(_finite)).sum())
            if nf:
                nonfinite[c] = nf
    res["nonfinite_counts"] = nonfinite
    res["nonfinite_total"] = int(sum(nonfinite.values()))

    # --- probability bounds ---
    res["prob_profit_outside_01"] = int(((df["prob_profit"] < 0) | (df["prob_profit"] > 1)).sum())
    res["prob_assignment_outside_01"] = (
        int(((df["prob_assignment"] < 0) | (df["prob_assignment"] > 1)).sum())
        if "prob_assignment" in df.columns
        else None
    )

    # --- served IV band (decimal); percent must lie in the connector gate
    #     (IV_FLOOR_PCT, IV_CEILING_PCT]. Gating BOTH ends closes the
    #     finite-but-absurd-IV gap an adversarial review flagged. ---
    iv_pct = df["iv"] * 100.0
    res["iv_below_floor_count"] = int((iv_pct <= IV_FLOOR_PCT).sum())
    res["iv_above_ceiling_count"] = int((iv_pct > IV_CEILING_PCT).sum())
    res["iv_ceiling_decimal"] = round(float(df["iv"].max()), 4)
    res["iv_floor_decimal"] = round(float(df["iv"].min()), 4)

    # --- premium / spot sanity ---
    ratio = (df["premium"] / df["spot"]).replace([np.inf, -np.inf], np.nan).dropna()
    res["premium_spot_pct_median"] = round(float(ratio.median()) * 100.0, 3)
    res["premium_spot_pct_max"] = round(float(ratio.max()) * 100.0, 3)
    res["premium_nonpositive_count"] = int((df["premium"] <= 0).sum())
    res["spot_nonpositive_count"] = int((df["spot"] <= 0).sum())

    # --- Greek-contract violations (recomputed) ---
    gv = df.apply(lambda r: _greek_violations(r, float(rfr)), axis=1)
    viol = gv[gv.notna()]
    res["greek_violation_count"] = int(len(viol))
    res["greek_violation_examples"] = [
        {"ticker": df.loc[i, "ticker"], "issue": viol.loc[i]} for i in viol.index[:8]
    ]

    # --- outliers ---
    res["prob_profit_eq_1_count"] = int((df["prob_profit"] >= 1.0).sum())
    res["prob_profit_eq_1_tickers"] = df.loc[df["prob_profit"] >= 1.0, "ticker"].tolist()[:10]
    res["thin_n_scenarios_lt_30_count"] = (
        int((df["n_scenarios"] < 30).sum()) if "n_scenarios" in df.columns else None
    )

    _persist(f"w2_{label}", res)
    return res


def main() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg")
    wr = WheelRunner()
    connector_name = type(wr.connector).__name__
    print(f"[bring-up] provider={provider} connector={connector_name}")

    per_regime = []
    for label, as_of in AS_OF:
        print(f"[run] {label} @ {as_of} ...", flush=True)
        r = audit_regime(wr, label, as_of)
        print(
            f"      n={r['n_candidates']} nonfinite={r.get('nonfinite_total')} "
            f"greek_viol={r.get('greek_violation_count')} "
            f"prem/spot_med={r.get('premium_spot_pct_median')}% "
            f"iv_ceil={r.get('iv_ceiling_decimal')}",
            flush=True,
        )
        per_regime.append(r)

    totals = {
        "n_candidates": sum(r["n_candidates"] for r in per_regime),
        "nonfinite_total": sum(r.get("nonfinite_total", 0) for r in per_regime),
        "greek_violation_total": sum(r.get("greek_violation_count", 0) for r in per_regime),
        "prob_outside_01_total": sum(
            (r.get("prob_profit_outside_01", 0) or 0)
            + (r.get("prob_assignment_outside_01", 0) or 0)
            for r in per_regime
        ),
        "iv_below_floor_total": sum(r.get("iv_below_floor_count", 0) for r in per_regime),
        "iv_above_ceiling_total": sum(r.get("iv_above_ceiling_count", 0) for r in per_regime),
        "premium_nonpositive_total": sum(r.get("premium_nonpositive_count", 0) for r in per_regime),
    }
    # magnitude monotonicity: crisis premium/spot & iv-ceiling must dominate calm
    calm = [r for r in per_regime if r["label"].startswith("calm")]
    crisis = [r for r in per_regime if r["label"].startswith("crisis")]
    bear = [r for r in per_regime if r["label"].startswith("bear")]

    def _avg(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    magnitude = {
        "premium_spot_pct_median": {
            "calm": _avg(calm, "premium_spot_pct_median"),
            "bear": _avg(bear, "premium_spot_pct_median"),
            "crisis": _avg(crisis, "premium_spot_pct_median"),
        },
        "iv_ceiling_decimal": {
            "calm": _avg(calm, "iv_ceiling_decimal"),
            "bear": _avg(bear, "iv_ceiling_decimal"),
            "crisis": _avg(crisis, "iv_ceiling_decimal"),
        },
    }
    magnitude["monotone_calm_lt_bear_lt_crisis"] = bool(
        magnitude["premium_spot_pct_median"]["calm"]
        < magnitude["premium_spot_pct_median"]["bear"]
        < magnitude["premium_spot_pct_median"]["crisis"]
    )

    summary = {
        "provider": provider,
        "connector": connector_name,
        "per_regime": [
            {
                k: r[k]
                for k in (
                    "label",
                    "as_of",
                    "n_candidates",
                    "nonfinite_total",
                    "greek_violation_count",
                    "prob_profit_outside_01",
                    "prob_assignment_outside_01",
                    "iv_below_floor_count",
                    "iv_above_ceiling_count",
                    "iv_ceiling_decimal",
                    "premium_spot_pct_median",
                    "prob_profit_eq_1_count",
                    "thin_n_scenarios_lt_30_count",
                )
                if k in r
            }
            for r in per_regime
        ],
        "totals": totals,
        "magnitude_scaling": magnitude,
        "verdict": (
            "REALISTIC"
            if totals["nonfinite_total"] == 0
            and totals["greek_violation_total"] == 0
            and totals["prob_outside_01_total"] == 0
            and totals["iv_below_floor_total"] == 0
            and totals["iv_above_ceiling_total"] == 0
            and totals["premium_nonpositive_total"] == 0
            else "ANOMALIES PRESENT"
        ),
    }
    _persist("w2_summary", summary)

    print("\n================ W2 RE-VERIFY SUMMARY ================")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
