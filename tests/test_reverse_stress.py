"""Tests for the V5 reverse-stress harness (backtests/reverse_stress.py).

Fast lane, engine-light: the adversary's admissibility math (R10 contract
cap, R9 sector budget, collateral budget, losers-only), the search
aggregation, the saturated-book builder, and the assignment-wave replay on
a stub connector with hand-built price paths.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtests import reverse_stress as rs

# ---------------------------------------------------------------------------
# V5-a — worst admissible book
# ---------------------------------------------------------------------------


def _rows(specs):
    """specs: list of (ticker, strike, realized_pnl, ev, prob, cvar)."""
    return pd.DataFrame(
        [
            {
                "date": "2020-02-19",
                "ticker": t,
                "strike": k,
                "realized_pnl": pnl,
                "ev_dollars": ev,
                "prob_profit": prob,
                "cvar_5": cvar,
                "premium": 2.5,
                "vix_entry": 14.0,
            }
            for (t, k, pnl, ev, prob, cvar) in specs
        ]
    )


def test_adversary_respects_r10_contract_cap():
    rows = _rows([("BA", 300.0, -5000.0, 10.0, 0.95, -900.0)])
    book = rs.worst_admissible_book(rows, {"BA": "Industrials"})
    (pick,) = book["names"]
    # R10: floor(0.10 x 1M / 30k) = 3 contracts, not more.
    assert pick["n_contracts"] == 3
    assert book["loss_dollars"] == pytest.approx(15000.0)


def test_adversary_ignores_winners_and_negative_ev():
    rows = _rows(
        [
            ("WIN", 100.0, +500.0, 10.0, 0.95, -400.0),  # winner: never helps
            ("NEG", 100.0, -500.0, -1.0, 0.95, -400.0),  # not engine-tradeable
            ("LOSE", 100.0, -500.0, 10.0, 0.95, -400.0),
        ]
    )
    book = rs.worst_admissible_book(rows, {})
    assert [p["ticker"] for p in book["names"]] == ["LOSE"]


def test_adversary_r9_sector_budget_binds():
    # Two same-sector names, each R10-capped at 10 contracts x $10k = $100k;
    # sector cap $250k -> second name limited to 15 contracts total in sector.
    rows = _rows(
        [
            ("A1", 100.0, -1000.0, 10.0, 0.95, -400.0),
            ("A2", 100.0, -900.0, 10.0, 0.95, -400.0),
            ("A3", 100.0, -800.0, 10.0, 0.95, -400.0),
        ]
    )
    sectors = {"A1": "Tech", "A2": "Tech", "A3": "Tech"}
    book = rs.worst_admissible_book(rows, sectors)
    assert book["collateral_used"] == pytest.approx(rs.R9_CAP_PCT * rs.NAV)  # 250k, not 300k
    total_contracts = sum(p["n_contracts"] for p in book["names"])
    assert total_contracts == 25  # 250k / 10k


def test_adversary_unknown_sector_exempt_but_counted():
    rows = _rows([("X1", 100.0, -1000.0, 10.0, 0.95, -400.0)])
    book = rs.worst_admissible_book(rows, {})
    assert book["n_unknown_sector"] == 1


def test_adversary_top_bin_filter():
    rows = _rows(
        [
            ("HI", 100.0, -1000.0, 10.0, 0.95, -400.0),
            ("LO", 100.0, -2000.0, 10.0, 0.70, -400.0),  # bigger loser, below top bin
        ]
    )
    book = rs.worst_admissible_book(rows, {}, top_bin_only=True)
    assert [p["ticker"] for p in book["names"]] == ["HI"]


def test_adversary_none_when_no_admissible_loser():
    rows = _rows([("BIG", 15000.0, -1000.0, 10.0, 0.95, -400.0)])  # 1 contract > R10 cap
    assert rs.worst_admissible_book(rows, {}) is None
    assert rs.worst_admissible_book(_rows([]), {}) is None


def test_adversary_modeled_cvar_gap():
    rows = _rows([("BA", 100.0, -3000.0, 10.0, 0.95, -1000.0)])
    book = rs.worst_admissible_book(rows, {})
    # 10 contracts: loss 30k vs modeled book cvar -10k -> gap 3.0x.
    assert book["realized_over_modeled_cvar"] == pytest.approx(3.0)


def test_search_aggregates_ruin_and_strata():
    # One calm-entry date with a ruin-class book, one clean date.
    d1 = _rows([("BA", 100.0, -30000.0, 10.0, 0.95, -1000.0)])  # 10c x 30k = 300k = 30% NAV
    d2 = _rows([("OK", 100.0, -100.0, 10.0, 0.95, -1000.0)])
    d2["date"] = "2023-06-01"
    d2["vix_entry"] = 30.0
    table = pd.concat([d1, d2], ignore_index=True)
    out = rs.reverse_stress_search(table, {})
    assert out["n_dates"] == 2
    assert out["n_ruin_dates"] == 1
    assert out["ruin_dates"] == ["2020-02-19"]
    assert out["by_vix_band"]["calm"]["n_ruin"] == 1
    assert out["by_vix_band"]["crisis"]["n_ruin"] == 0


# ---------------------------------------------------------------------------
# V5-b — saturated book + assignment-wave replay
# ---------------------------------------------------------------------------


def test_build_saturated_book_fills_by_ev_until_budget():
    rows = _rows(
        [
            ("A", 4000.0, -1.0, 100.0, 0.9, -1.0),  # 400k collateral, best EV
            ("B", 4000.0, -1.0, 90.0, 0.9, -1.0),  # 400k
            ("C", 4000.0, -1.0, 80.0, 0.9, -1.0),  # 400k -> would exceed 1M, skipped
            ("D", 1000.0, -1.0, 70.0, 0.9, -1.0),  # 100k fits in the remainder
        ]
    )
    book = rs.build_saturated_book(rows)
    assert [p["ticker"] for p in book] == ["A", "B", "D"]
    assert sum(p["collateral"] for p in book) == pytest.approx(900_000.0)


class _PathConn:
    """Stub connector: fixed per-ticker close paths."""

    def __init__(self, paths):
        self._paths = paths

    def get_ohlcv(self, ticker, start_date=None, end_date=None):
        s = self._paths.get(ticker)
        if s is None:
            return pd.DataFrame()
        return pd.DataFrame({"close": s.values}, index=s.index)


def _path(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-02-19", periods=len(vals)))


def test_replay_assignment_wave_trough_terminal_and_assignment():
    # Strike 100, premium 2.5; path dips to 80 (trough) then recovers to 95:
    # assigned at expiry (95 < 100); trough pnl = (2.5 - 20) x 100 = -1750.
    book = [{"ticker": "BA", "strike": 100.0, "premium": 2.5, "collateral": 10_000.0}]
    conn = _PathConn({"BA": _path([100, 90, 80, 90, 95])})
    out = rs.replay_assignment_wave(book, conn, eve="2020-02-19", horizon_bdays=4)
    assert out["assignment_fraction"] == 1.0
    assert out["trough_pnl_dollars"] == pytest.approx(-1750.0)
    assert out["terminal_pnl_dollars"] == pytest.approx((2.5 - 5.0) * 100.0)
    assert out["trough_liquidation_pct_nav"] == pytest.approx(1750.0 / rs.NAV)


def test_replay_levered_counterfactual_calls_under_stress():
    # A crash path: deeper stress multiplier must call no later than milder.
    book = [{"ticker": "BA", "strike": 100.0, "premium": 2.5, "collateral": 10_000.0}]
    conn = _PathConn({"BA": _path([100, 70, 50, 40, 40])})
    out = rs.replay_assignment_wave(book, conn, eve="2020-02-19", horizon_bdays=4)
    lev = out["levered_counterfactual"]
    days = {m: v["first_call_bday"] for m, v in lev.items()}
    assert days["1.5"] is not None
    if days["1"] is not None:
        assert days["1.5"] <= days["1"]


def test_replay_handles_missing_paths():
    book = [{"ticker": "ZZZ", "strike": 100.0, "premium": 2.5, "collateral": 10_000.0}]
    out = rs.replay_assignment_wave(book, _PathConn({}), eve="2020-02-19")
    assert out["error"] == "no price paths"


# ---------------------------------------------------------------------------
# §10.2 — time-value marking (V5-b-full)
# ---------------------------------------------------------------------------


def _tv_book(iv=0.5):
    return [{"ticker": "BA", "strike": 100.0, "premium": 2.5, "collateral": 10_000.0, "iv": iv}]


def test_tv_marking_never_shallower_than_intrinsic_and_monotone_in_iv():
    conn = _PathConn({"BA": _path([100, 90, 80, 90, 95])})
    base = rs.replay_assignment_wave(_tv_book(), conn, eve="2020-02-19", horizon_bdays=4)
    tv1 = rs.replay_assignment_wave(
        _tv_book(), conn, eve="2020-02-19", horizon_bdays=4, marking="time_value", iv_mult=1.0
    )
    tv2 = rs.replay_assignment_wave(
        _tv_book(), conn, eve="2020-02-19", horizon_bdays=4, marking="time_value", iv_mult=2.0
    )
    # Expectation 1 (§10.2): TV trough damage >= intrinsic, and deeper IV -> deeper.
    assert tv1["trough_liquidation_pct_nav"] >= base["trough_liquidation_pct_nav"]
    assert tv2["trough_liquidation_pct_nav"] >= tv1["trough_liquidation_pct_nav"]
    assert tv1["marking"] == "time_value" and tv1["n_no_iv"] == 0


def test_tv_marking_without_iv_falls_back_to_intrinsic():
    conn = _PathConn({"BA": _path([100, 90, 80, 90, 95])})
    base = rs.replay_assignment_wave(_tv_book(iv=0.0), conn, eve="2020-02-19", horizon_bdays=4)
    tv = rs.replay_assignment_wave(
        _tv_book(iv=0.0), conn, eve="2020-02-19", horizon_bdays=4, marking="time_value"
    )
    assert tv["n_no_iv"] == 1
    assert tv["trough_pnl_dollars"] == pytest.approx(base["trough_pnl_dollars"])


def test_replay_rejects_unknown_marking():
    with pytest.raises(ValueError, match="marking"):
        rs.replay_assignment_wave(_tv_book(), _PathConn({}), eve="2020-02-19", marking="delta")


def test_build_saturated_book_carries_iv():
    rows = _rows([("A", 100.0, -1.0, 10.0, 0.9, -1.0)])
    assert rs.build_saturated_book(rows)[0]["iv"] == 0.0  # column absent -> 0.0
    rows["iv"] = 0.42
    assert rs.build_saturated_book(rows)[0]["iv"] == pytest.approx(0.42)
