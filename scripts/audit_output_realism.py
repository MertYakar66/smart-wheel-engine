#!/usr/bin/env python3
"""W2 — Engine output-realism audit (heavy-verify 2026-06-27, #436, Mac terminal).

Validation-only. Runs ``WheelRunner.rank_candidates_by_ev`` over the full
universe at several historical ``as_of`` dates spanning regimes (calm + the
2020 / 2022 stress windows) and checks the served outputs are realistic:

  * ``premium`` vs ``spot`` ratio sane,
  * served ``iv`` (decimal in the ranker frame) > 0.03 — the connector gate
    nulls IV outside (3.0, 10000] PERCENT, so every served decimal IV > 0.03,
  * ``ev_dollars`` finite, signs/magnitudes sane,
  * Greeks in the documented contract units (docs/GREEKS_UNIT_CONTRACT.md):
    a 25-delta short put solves to delta ~ -0.25, gamma>=0, vega>=0,
  * ``prob_profit`` and ``prob_assignment`` in [0, 1].

Flags every candidate with a non-finite or absurd value and buckets by cause.
Persists JSON to ``docs/verification_artifacts/data_wiring_2026-06-27/`` before
pretty-printing. Does NOT touch engine behaviour.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_output_realism.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from engine.option_pricer import black_scholes_all_greeks  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

OUT = REPO / "docs" / "verification_artifacts" / "data_wiring_2026-06-27"
OUT.mkdir(parents=True, exist_ok=True)

# Historical as_of dates spanning regimes. OHLCV starts 2018-01-02 and the
# ranker's 504-trading-day history gate makes ~2020-01 the earliest feasible
# as_of, so the COVID window is sampled at its trough rather than its onset.
REGIMES = [
    ("calm_2021", "2021-06-15"),
    ("calm_2024", "2024-01-16"),
    ("covid_2020", "2020-03-23"),
    ("bear_2022_jun", "2022-06-16"),
    ("bear_2022_oct", "2022-10-14"),
]

# Realism bands (a candidate outside these is flagged + bucketed).
PREM_SPOT_MAX = 0.50  # premium should be a fraction of spot for a 35DTE put
SPOT_MIN = 1.0
IV_DECIMAL_MIN = 0.03  # connector floors PERCENT at 3.0 → decimal 0.03
IV_DECIMAL_MAX = 5.0  # 500% IV is the practical absurd ceiling


def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _greek_check(row: pd.Series, rfr: float) -> dict | None:
    """Compute put Greeks at the candidate and return a violation dict or None."""
    try:
        S = float(row["spot"])
        K = float(row["strike"])
        T = float(row["dte"]) / 365.0
        sigma = float(row["iv"])
        if not all(_finite(v) for v in (S, K, T, sigma)) or S <= 0 or sigma <= 0 or T <= 0:
            return {"ticker": row["ticker"], "issue": "non-evaluable greek inputs"}
        g = black_scholes_all_greeks(S, K, T, rfr, sigma, "put", include_second_order=False)
        viol = []
        if not (-1.0 <= g["delta"] <= 0.0):
            viol.append(f"put delta {g['delta']:.3f} outside [-1,0]")
        if g["gamma"] < -1e-9:
            viol.append(f"gamma {g['gamma']:.4g} < 0")
        if g["vega"] < -1e-9:
            viol.append(f"vega {g['vega']:.4g} < 0")
        if viol:
            return {
                "ticker": row["ticker"],
                "delta": round(g["delta"], 4),
                "issue": "; ".join(viol),
            }
        return None
    except Exception as exc:  # never crash the audit on one candidate
        return {"ticker": row.get("ticker"), "issue": f"greek raise: {exc!r}"}


def audit_regime(wr: WheelRunner, label: str, as_of: str) -> dict:
    df = wr.rank_candidates_by_ev(
        as_of=as_of, top_n=600, min_ev_dollars=-1e9, include_diagnostic_fields=True
    )
    vix = wr.connector.get_vix_regime(as_of)
    rfr = wr.connector.get_risk_free_rate(as_of, "rate_3m")
    drops = df.attrs.get("drops_summary", {})

    n = len(df)
    prem_spot = pd.to_numeric(df["premium"], errors="coerce") / pd.to_numeric(
        df["spot"], errors="coerce"
    )
    iv = pd.to_numeric(df["iv"], errors="coerce")
    pp = pd.to_numeric(df["prob_profit"], errors="coerce")
    pa = pd.to_numeric(df["prob_assignment"], errors="coerce")
    nsc = pd.to_numeric(df.get("n_scenarios"), errors="coerce")

    # Per-column non-finite tally.
    numeric_cols = [
        "spot",
        "strike",
        "premium",
        "iv",
        "ev_dollars",
        "ev_raw",
        "prob_profit",
        "prob_assignment",
        "roc",
        "cvar_5",
        "regime_multiplier",
    ]
    non_finite = {}
    for col in numeric_cols:
        if col in df.columns:
            bad = int((~pd.to_numeric(df[col], errors="coerce").map(_finite)).sum())
            if bad:
                non_finite[col] = bad

    # Outlier buckets.
    outliers: dict[str, list] = {
        "prem_spot_gt_max": df.loc[prem_spot > PREM_SPOT_MAX, "ticker"].tolist(),
        "spot_lt_min": df.loc[
            pd.to_numeric(df["spot"], errors="coerce") < SPOT_MIN, "ticker"
        ].tolist(),
        "iv_outside_band": df.loc[
            (iv <= IV_DECIMAL_MIN) | (iv > IV_DECIMAL_MAX), "ticker"
        ].tolist(),
        "prob_profit_outside_01": df.loc[(pp < 0) | (pp > 1), "ticker"].tolist(),
        "prob_assignment_outside_01": df.loc[(pa < 0) | (pa > 1), "ticker"].tolist(),
        "prob_profit_eq_1": df.loc[pp >= 1.0, "ticker"].tolist(),
        "prob_profit_thin_n_lt_30": df.loc[nsc < 30, "ticker"].tolist() if nsc is not None else [],
    }

    # Greeks pass.
    greek_violations = [g for _, row in df.iterrows() if (g := _greek_check(row, rfr))]

    return {
        "as_of": as_of,
        "vix_level": round(float(vix.get("vix", float("nan"))), 2),
        "vix_percentile": round(float(vix.get("vix_percentile", float("nan"))), 3),
        "rfr_3m_decimal": round(float(rfr), 6),
        "ranked_candidates": n,
        "drops_summary": drops,
        "premium_spot_ratio": {
            "min": round(float(prem_spot.min()), 5),
            "median": round(float(prem_spot.median()), 5),
            "max": round(float(prem_spot.max()), 5),
        },
        "iv_decimal": {"min": round(float(iv.min()), 4), "max": round(float(iv.max()), 4)},
        "ev_dollars": {
            "min": round(float(pd.to_numeric(df["ev_dollars"]).min()), 2),
            "max": round(float(pd.to_numeric(df["ev_dollars"]).max()), 2),
            "n_negative": int((pd.to_numeric(df["ev_dollars"]) < 0).sum()),
        },
        "prob_profit": {"min": round(float(pp.min()), 4), "max": round(float(pp.max()), 4)},
        "non_finite_by_column": non_finite,
        "outlier_counts": {k: len(v) for k, v in outliers.items()},
        "outlier_tickers": {k: v[:15] for k, v in outliers.items() if v},
        "greek_violation_count": len(greek_violations),
        "greek_violations": greek_violations[:15],
    }


def main() -> None:
    wr = WheelRunner()
    results = {}
    for label, as_of in REGIMES:
        try:
            results[label] = audit_regime(wr, label, as_of)
        except Exception as exc:  # persist what we have; never lose compute
            results[label] = {"as_of": as_of, "error": repr(exc)}
        # persist incrementally so a later crash never loses an earlier regime
        (OUT / "w2_output_realism.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    # ---- console summary (JSON already on disk) ----
    print("=== W2 OUTPUT-REALISM AUDIT ===")
    hdr = f"{'regime':<16}{'as_of':<12}{'VIX':>6}{'n':>5}{'ppmax':>7}{'prem/spot med':>14}{'nonfin':>7}{'greekV':>7}"
    print(hdr)
    for label, r in results.items():
        if "error" in r:
            print(f"{label:<16}{r['as_of']:<12}  ERROR {r['error'][:40]}")
            continue
        nf = sum(r["non_finite_by_column"].values())
        print(
            f"{label:<16}{r['as_of']:<12}{r['vix_level']:>6}{r['ranked_candidates']:>5}"
            f"{r['prob_profit']['max']:>7}{r['premium_spot_ratio']['median']:>14}"
            f"{nf:>7}{r['greek_violation_count']:>7}"
        )
    print(f"\nJSON → {OUT / 'w2_output_realism.json'}")


if __name__ == "__main__":
    main()
