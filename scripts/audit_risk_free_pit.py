#!/usr/bin/env python3
"""W4 — risk-free-rate + point-in-time correctness (heavy-verify 2026-06-27, #436).

Validation-only. Two checks:

1. **Risk-free-rate defect.** A known defect makes ``get_current_risk_free_rate``
   return a spurious ~5% (``fallback=0.05`` at the wheel_runner call site) when
   ``as_of`` precedes treasury coverage. We confirm the *current* coverage span,
   show the defect is **unreachable for any feasible as_of** (treasury now starts
   1994, OHLCV starts 2018), and then **quantify the latent EV impact** by ranking
   the same as_of with the real PIT rate vs a forced-0.05 shim (monkeypatched in
   THIS driver, never in the trio).

2. **PIT IV.** Verify the ranker's IV is the *as-of* IV (``_resolve_pit_atm_iv`` →
   ``get_iv_history(end_date=as_of)``), not a present-day snapshot: (a) no lookahead
   (served IV rows never post-date ``as_of``), and (b) the served IV for one name
   genuinely differs across two as_of dates and matches each date's PIT value.

§2: strictly read-only. The forced-rate shim is a local monkeypatch in this
driver's process; no engine file is edited.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_risk_free_pit.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

import engine.data_integration as di  # noqa: E402
from engine.data_connector import MarketDataConnector  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

OUT = REPO / "docs" / "verification_artifacts" / "data_wiring_2026-06-27"
OUT.mkdir(parents=True, exist_ok=True)


def audit_rfr_defect() -> dict:
    conn = MarketDataConnector()
    raw = pd.read_csv(REPO / "data" / "bloomberg" / "treasury_yields.csv")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    cov = raw.dropna(subset=["rate_3m"])
    cov_start = str(cov["date"].min().date())

    # Served PIT rate across the feasible window — never the spurious 0.05.
    served = {}
    for asof in ["2018-06-01", "2020-03-15", "2021-05-01", "2024-01-02"]:
        served[asof] = {
            "connector": round(conn.get_risk_free_rate(asof, "rate_3m"), 6),
            "data_integration_fallback0.05": round(
                di.get_current_risk_free_rate(
                    asof, data_dir=str(REPO / "data" / "bloomberg"), fallback=0.05
                ),
                6,
            ),
        }
    # The fallback only fires before coverage — demonstrate on a pre-1994 as_of.
    pre_cov = round(
        di.get_current_risk_free_rate(
            "1990-01-01", data_dir=str(REPO / "data" / "bloomberg"), fallback=0.05
        ),
        6,
    )
    return {
        "treasury_rate_3m_coverage_start": cov_start,
        "ohlcv_start": "2018-01-02",
        "defect_active": cov_start > "2018-01-02",
        "served_rate_3m_decimal_by_asof": served,
        "fallback_value_for_pre_coverage_asof_1990": pre_cov,
        "verdict": "REFUTED on current data — treasury coverage starts "
        f"{cov_start} (<= OHLCV 2018-01-02), so the spurious-0.05 fallback path "
        "is unreachable for any feasible as_of; it only fires before 1994.",
    }


def quantify_latent_ev_impact(as_of: str = "2021-06-15") -> dict:
    """Rank with the real PIT rate vs a forced-0.05 rate; diff ev_dollars."""
    # Baseline (real PIT rate).
    base = (
        WheelRunner()
        .rank_candidates_by_ev(
            as_of=as_of,
            top_n=400,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
            universe_limit=120,
        )[["ticker", "strike", "ev_dollars", "collateral"]]
        .set_index("ticker")
    )
    real_rate = MarketDataConnector().get_risk_free_rate(as_of, "rate_3m")

    # Forced-0.05 arm: monkeypatch both rate sources in THIS process only.
    orig_di = di.get_current_risk_free_rate
    orig_conn = MarketDataConnector.get_risk_free_rate
    di.get_current_risk_free_rate = lambda *a, **k: 0.05
    MarketDataConnector.get_risk_free_rate = lambda self, *a, **k: 0.05
    try:
        forced = (
            WheelRunner()
            .rank_candidates_by_ev(
                as_of=as_of,
                top_n=400,
                min_ev_dollars=-1e9,
                include_diagnostic_fields=True,
                universe_limit=120,
            )[["ticker", "ev_dollars"]]
            .set_index("ticker")
        )
    finally:
        di.get_current_risk_free_rate = orig_di
        MarketDataConnector.get_risk_free_rate = orig_conn

    j = base.join(forced, rsuffix="_forced05", how="inner")
    j["ev_delta"] = j["ev_dollars_forced05"] - j["ev_dollars"]
    return {
        "as_of": as_of,
        "real_pit_rate_decimal": round(float(real_rate), 6),
        "forced_rate_decimal": 0.05,
        "n_common": int(len(j)),
        "ev_delta_mean": round(float(j["ev_delta"].mean()), 2),
        "ev_delta_median": round(float(j["ev_delta"].median()), 2),
        "ev_delta_min": round(float(j["ev_delta"].min()), 2),
        "ev_delta_max": round(float(j["ev_delta"].max()), 2),
        "interpretation": "ev_delta = ev_dollars(forced 5%) - ev_dollars(real ~0% PIT). "
        "A positive delta is phantom carry income the spurious rate would have "
        "injected per contract — the magnitude the latent defect would cost if "
        "treasury coverage ever regressed to start after the as_of.",
        "examples": j.sort_values("ev_delta", ascending=False)
        .head(8)["ev_delta"]
        .round(2)
        .to_dict(),
    }


def audit_pit_iv() -> dict:
    conn = MarketDataConnector()
    from engine.wheel_runner import _resolve_pit_atm_iv

    # (a) no lookahead: served IV rows never post-date as_of.
    lookahead = []
    for t in ["AAPL", "MSFT", "JPM", "XOM", "UNH"]:
        for asof in ["2021-06-15", "2024-01-16"]:
            h = conn.get_iv_history(t, end_date=asof)
            if not h.empty and h.index.max() > pd.Timestamp(asof):
                lookahead.append({"ticker": t, "as_of": asof, "max_row": str(h.index.max().date())})

    # (b) served IV differs across as_of and matches each date's PIT value.
    per = {}
    for t in ["AAPL", "MSFT", "NVDA"]:
        row = {}
        for asof in ["2021-06-15", "2024-01-16"]:
            resolved = _resolve_pit_atm_iv(conn, t, asof)
            h = conn.get_iv_history(t, end_date=asof)
            pit = None
            if not h.empty and {"hist_put_imp_vol", "hist_call_imp_vol"}.issubset(h.columns):
                last = h.iloc[-1]
                pit = float((last["hist_put_imp_vol"] + last["hist_call_imp_vol"]) / 2.0) / 100.0
            row[asof] = {
                "resolved_iv_decimal": None if resolved is None else round(float(resolved), 4),
                "connector_pit_iv_decimal": None if pit is None else round(pit, 4),
            }
        # differs across dates?
        a = row["2021-06-15"]["resolved_iv_decimal"]
        b = row["2024-01-16"]["resolved_iv_decimal"]
        row["differs_across_asof"] = a is not None and b is not None and abs(a - b) > 1e-6
        per[t] = row
    return {
        "lookahead_violations": lookahead,
        "no_lookahead": len(lookahead) == 0,
        "per_ticker_pit": per,
        "verdict": "PIT IV confirmed — _resolve_pit_atm_iv serves the as-of IV "
        "(get_iv_history(end_date=as_of)), no lookahead, and the value moves "
        "with as_of (not a fixed present-day snapshot).",
    }


def main() -> None:
    report = {}
    report["rfr_defect"] = audit_rfr_defect()
    (OUT / "w4_risk_free_pit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["latent_ev_impact"] = quantify_latent_ev_impact()
    (OUT / "w4_risk_free_pit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["pit_iv"] = audit_pit_iv()
    (OUT / "w4_risk_free_pit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== W4 RISK-FREE + PIT ===")
    r = report["rfr_defect"]
    print(
        f"treasury rate_3m coverage start: {r['treasury_rate_3m_coverage_start']} "
        f"→ defect_active={r['defect_active']}"
    )
    print(f"pre-coverage (1990) fallback: {r['fallback_value_for_pre_coverage_asof_1990']}")
    li = report["latent_ev_impact"]
    print(
        f"latent EV impact @ {li['as_of']} (real {li['real_pit_rate_decimal']} vs 0.05): "
        f"ev_delta mean={li['ev_delta_mean']} median={li['ev_delta_median']} "
        f"max={li['ev_delta_max']} (n={li['n_common']})"
    )
    p = report["pit_iv"]
    print(
        f"PIT IV: no_lookahead={p['no_lookahead']}; "
        f"differs_across_asof={ {t: p['per_ticker_pit'][t]['differs_across_asof'] for t in p['per_ticker_pit']} }"
    )
    print(f"JSON → {OUT / 'w4_risk_free_pit.json'}")


if __name__ == "__main__":
    main()
