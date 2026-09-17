"""Tests for the V4 capacity-curve harness (backtests/capacity_curve.py).

Fast lane, engine-light: the impact math (against the hand formula), the
fill-decision ladder, the PIT ADV lookup (stub connector, no data files),
the corrected proportionality A/A check, and the knee table.  The full
ladder driver is exercised by the pilot run, not the fast lane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests import capacity_curve as cc

# ---------------------------------------------------------------------------
# Impact math — the engine's sqrt term, isolated
# ---------------------------------------------------------------------------


def test_impact_per_share_matches_hand_formula():
    # k * mid * sqrt(N / adv) = 0.10 * 2.50 * sqrt(10 / 1000) = 0.025
    assert cc.impact_per_share(2.50, 10, 1000.0) == pytest.approx(0.025)


def test_impact_per_share_zero_spread_isolates_size_term():
    # With adv huge, participation ~ 0 -> impact ~ 0 (no spread bleed-through).
    assert cc.impact_per_share(2.50, 1, 1e12) == pytest.approx(0.0, abs=1e-6)


def test_impact_grows_sqrt_in_contracts():
    a = cc.impact_per_share(2.50, 4, 1000.0)
    b = cc.impact_per_share(2.50, 16, 1000.0)
    assert b == pytest.approx(2.0 * a)  # sqrt(16/4) = 2


# ---------------------------------------------------------------------------
# decide_fill ladder
# ---------------------------------------------------------------------------


def test_decide_fill_control_passes_through():
    verdict, net, imp = cc.decide_fill(2.30, 2.50, 25, float("inf"))
    assert (verdict, net, imp) == ("control", 2.30, 0.0)


def test_decide_fill_no_adv_counts_loudly_but_fills():
    verdict, net, imp = cc.decide_fill(2.30, 2.50, 25, None)
    assert (verdict, net, imp) == ("no_adv", 2.30, 0.0)


def test_decide_fill_participation_refusal():
    # N=25 > 0.10 * 200 = 20 -> refused before any impact math.
    verdict, net, _ = cc.decide_fill(2.30, 2.50, 25, 200.0)
    assert verdict == "part_refused" and net == 0.0


def test_decide_fill_priced_out_when_impact_eats_premium():
    # impact = 0.10 * 2.50 * sqrt(100/1000) ~ 0.079 > premium 0.05 -> priced out.
    verdict, net, imp = cc.decide_fill(0.05, 2.50, 100, 1000.0)
    assert verdict == "priced_out" and net == 0.0 and imp > 0.05


def test_decide_fill_filled_nets_the_impact():
    verdict, net, imp = cc.decide_fill(2.30, 2.50, 10, 1000.0)
    assert verdict == "filled"
    assert imp == pytest.approx(0.025)
    assert net == pytest.approx(2.30 - 0.025)


# ---------------------------------------------------------------------------
# AdvLookup PIT behavior (stub connector, no data files)
# ---------------------------------------------------------------------------


class _StubConn:
    def __init__(self, frame_by_ticker):
        self._frames = frame_by_ticker
        self.calls = 0

    def get_liquidity(self, ticker, start_date=None, end_date=None):
        self.calls += 1
        return self._frames.get(ticker)


def _liq_frame(dates, vols):
    return pd.DataFrame(
        {"avg_vol_30d": vols, "turnover": 0.0, "shares_out": 0.0},
        index=pd.to_datetime(dates),
    )


def test_adv_lookup_pit_uses_last_obs_at_or_before_date():
    conn = _StubConn({"AAPL": _liq_frame(["2022-01-03", "2022-06-01"], [1e6, 2e6])})
    adv = cc.AdvLookup(conn)
    from datetime import date

    assert adv.adv_shares("AAPL", date(2022, 3, 1)) == 1e6  # later obs invisible
    assert adv.adv_shares("AAPL", date(2022, 6, 1)) == 2e6  # inclusive
    assert adv.adv_shares("AAPL", date(2021, 12, 31)) is None  # predates series
    assert conn.calls == 1  # cached: one fetch per ticker


def test_adv_lookup_missing_ticker_and_ratio_scaling():
    conn = _StubConn({})
    adv = cc.AdvLookup(conn)
    from datetime import date

    assert adv.adv_shares("ZZZ", date(2022, 3, 1)) is None
    conn2 = _StubConn({"A": _liq_frame(["2022-01-03"], [1e6])})
    adv2 = cc.AdvLookup(conn2)
    assert adv2.adv_contracts("A", date(2022, 3, 1), 1e-4) == pytest.approx(100.0)


def test_adv_lookup_nonpositive_and_nan_degrade_to_none():
    conn = _StubConn({"A": _liq_frame(["2022-01-03", "2022-02-01"], [np.nan, 0.0])})
    adv = cc.AdvLookup(conn)
    from datetime import date

    assert adv.adv_shares("A", date(2022, 3, 1)) is None  # 0.0 -> None; NaN dropped


# ---------------------------------------------------------------------------
# Linearity A/A (the corrected proportionality form) + knee table
# ---------------------------------------------------------------------------


def _pt(n, ratio, ret, bp=0, part=0, impact_share=0.0, opens=10):
    return {
        "n_contracts": n,
        "ratio": ratio,
        "return_pct": ret,
        "bp_refused": bp,
        "part_refused": part,
        "impact_share_of_premium": impact_share,
        "opens": opens,
    }


def test_linearity_check_proportional_controls_pass():
    points = {
        "N1": _pt(1, None, 2.0),
        "N5": _pt(5, None, 10.0),
        "N10": _pt(10, None, 20.0),
    }
    out = cc.linearity_check(points)
    assert out["verdict"] == "PASS"
    assert out["n_unthrottled"] == 3


def test_linearity_check_deviation_without_bp_refusal_fails():
    points = {"N1": _pt(1, None, 2.0), "N5": _pt(5, None, 9.0)}  # 9 != 5*2, no BP refusals
    assert cc.linearity_check(points)["verdict"] == "FAIL"


def test_linearity_check_throttled_deviation_is_reported_not_failed():
    points = {
        "N1": _pt(1, None, 2.0),
        "N5": _pt(5, None, 10.0),
        "N25": _pt(25, None, 30.0, bp=40),  # deviates, but BP-throttled
    }
    out = cc.linearity_check(points)
    assert out["verdict"] == "PASS"
    assert out["throttled_points"] == {"N25": 40}


def test_knee_table_argmax_and_shape():
    points = {
        "a": _pt(1, 1e-4, 2.0),
        "b": _pt(5, 1e-4, 8.0),
        "c": _pt(10, 1e-4, 7.0, part=12),
        "d": _pt(1, 1e-3, 2.0),
        "e": _pt(5, 1e-3, 9.5),
        "f": _pt(10, 1e-3, 11.0),
        "ctrl": _pt(25, None, 50.0),
    }
    table = cc.knee_table(points)
    assert [row["ratio"] for row in table] == [1e-4, 1e-3]
    assert table[0]["knee_n"] == 5  # 8.0 beats 2.0 and 7.0
    assert table[1]["knee_n"] == 10
    assert set(table[0]["per_n"]) == {"1", "5", "10"}


def test_ladder_point_key_naming():
    assert cc.LadderPoint(5, None).key == "N5_rcontrol"
    assert cc.LadderPoint(25, 1e-4).key == "N25_r0.0001"
