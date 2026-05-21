"""Tests for :meth:`engine.wheel_tracker.WheelTracker.suggest_call_rolls`.

The covered-call-leg parallel of ``test_wheel_tracker_suggest_rolls.py``.
Covers the 5 spec'd properties -- challenged returns candidates, healthy
produces valid output, no candidates is empty-not-crash, recommend
implies roll_ev > hold_ev, EVEngine.evaluate is called per candidate --
plus the EV-metric regression (roll_ev nets the full buyback principal,
not just transaction costs) and schema / edge-case unit tests.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.ev_engine import EVEngine
from engine.wheel_tracker import (
    _ROLL_COLUMNS,
    PositionState,
    WheelTracker,
    _solve_call_strike,
)


# ----------------------------------------------------------------------
# Synthetic data + fake connector -- deterministic, no data-on-disk dep.
# ----------------------------------------------------------------------
def _synth_ohlcv(
    n: int = 800, seed: int = 1, mu: float = 0.0003, sigma: float = 0.012
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )


class _FakeConn:
    """Minimal connector for suggest_call_rolls tests."""

    def __init__(self, ohlcv: pd.DataFrame | None = None, rf: float = 0.04) -> None:
        self._oh = ohlcv if ohlcv is not None else _synth_ohlcv()
        self._rf = rf

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._oh

    def get_risk_free_rate(self, as_of=None, tenor: str = "rate_3m") -> float:
        return self._rf


def _make_tracker_with_covered_call(
    *,
    call_strike: float,
    call_premium: float,
    call_entry: date,
    call_expiry: date,
    iv: float = 0.25,
    ohlcv: pd.DataFrame | None = None,
    rf: float = 0.04,
    initial_capital: float = 100_000.0,
    connector: bool = True,
) -> WheelTracker:
    """Drive a position into COVERED_CALL state: short put -> assigned ->
    covered call. suggest_call_rolls reads only the call leg."""
    conn = _FakeConn(ohlcv=ohlcv, rf=rf) if connector else None
    t = WheelTracker(initial_capital=initial_capital, connector=conn)
    put_entry = call_entry - timedelta(days=40)
    ok = t.open_short_put(
        ticker="TEST",
        strike=95.0,
        premium=2.0,
        entry_date=put_entry,
        expiration_date=call_entry,
        iv=iv,
    )
    assert ok, "open_short_put failed -- check capital / margin in the test setup"
    ok = t.handle_put_assignment("TEST", call_entry, 95.0)
    assert ok, "handle_put_assignment failed"
    assert t.positions["TEST"].state == PositionState.STOCK_OWNED
    ok = t.open_covered_call(
        ticker="TEST",
        strike=call_strike,
        premium=call_premium,
        entry_date=call_entry,
        expiration_date=call_expiry,
        iv=iv,
    )
    assert ok, "open_covered_call failed"
    assert t.positions["TEST"].state == PositionState.COVERED_CALL
    return t


# ======================================================================
# Schema + edge cases
# ======================================================================
class TestSuggestCallRollsEdgeCases:
    def test_no_open_position_raises(self):
        t = WheelTracker(initial_capital=100_000.0, connector=_FakeConn())
        with pytest.raises(ValueError, match="No open position"):
            t.suggest_call_rolls(
                ticker="MISSING",
                as_of=date(2026, 1, 1),
                current_spot=100.0,
                current_iv=0.25,
                risk_free_rate=0.04,
            )

    def test_non_covered_call_state_raises(self):
        t = WheelTracker(initial_capital=100_000.0, connector=_FakeConn())
        t.open_short_put(
            ticker="TEST",
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
            iv=0.25,
        )
        assert t.positions["TEST"].state == PositionState.SHORT_PUT
        with pytest.raises(ValueError, match="COVERED_CALL"):
            t.suggest_call_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 15),
                current_spot=100.0,
                current_iv=0.25,
                risk_free_rate=0.04,
            )

    def test_negative_spot_raises(self):
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
        )
        with pytest.raises(ValueError, match="current_spot"):
            t.suggest_call_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 19),
                current_spot=-10.0,
                current_iv=0.25,
                risk_free_rate=0.04,
            )

    def test_out_of_range_iv_raises(self):
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
        )
        with pytest.raises(ValueError, match="current_iv"):
            t.suggest_call_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 19),
                current_spot=100.0,
                current_iv=5.0,
                risk_free_rate=0.04,
            )

    def test_past_expiry_returns_empty_schema(self):
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
        )
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 14),  # past the call expiry
            current_spot=108.0,
            current_iv=0.25,
            risk_free_rate=0.04,
        )
        assert df.empty
        assert list(df.columns) == _ROLL_COLUMNS

    def test_rf_resolved_from_connector_when_none(self):
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            rf=0.045,
        )
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 19),
            current_spot=106.0,
            current_iv=0.25,
            risk_free_rate=None,  # auto-resolve from the connector
        )
        assert list(df.columns) == _ROLL_COLUMNS

    def test_no_rf_no_connector_raises(self):
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            connector=False,
        )
        with pytest.raises(ValueError, match="no connector"):
            t.suggest_call_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 19),
                current_spot=106.0,
                current_iv=0.25,
                risk_free_rate=None,
            )


# ======================================================================
# The 5 spec'd tests
# ======================================================================
class TestSuggestCallRollsSpec:
    def test_challenged_covered_call_returns_candidates(self):
        """Spec 1: a covered call gone ITM (stock rallied through the
        strike) returns >= 1 structurally valid roll candidate."""
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            iv=0.25,
        )
        as_of = date(2026, 1, 26)  # 21 days in, 14 DTE remaining
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=as_of,
            current_spot=112.0,  # well above the 105 strike -- ITM / challenged
            current_iv=0.25,
            risk_free_rate=0.04,
            min_net_credit=-1_500.0,  # allow rescue debit rolls
        )
        assert not df.empty, "challenged covered call should produce >= 1 candidate"
        assert list(df.columns) == _ROLL_COLUMNS
        roll_evs = df["roll_ev"].tolist()
        assert roll_evs == sorted(roll_evs, reverse=True)
        for _, r in df.iterrows():
            assert r["new_strike"] > 112.0  # OTM at the current spot
            assert r["new_dte"] > 0
            assert r["new_expiry"] > as_of
            assert np.isfinite(r["new_premium"]) and r["new_premium"] > 0
            assert np.isfinite(r["buyback_cost"]) and r["buyback_cost"] > 0
            assert np.isfinite(r["roll_ev"])
            assert np.isfinite(r["hold_ev"])
            assert 0.0 <= r["prob_otm"] <= 1.0
            assert isinstance(r["recommend"], (bool, np.bool_))

    def test_healthy_position_produces_valid_output(self):
        """Spec 2: a healthy (deep-OTM) covered call returns a
        structurally valid DataFrame -- empty or all rows valid."""
        t = _make_tracker_with_covered_call(
            call_strike=115.0,
            call_premium=1.5,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            iv=0.25,
        )
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 19),
            current_spot=100.0,  # well below the 115 strike -- healthy
            current_iv=0.25,
            risk_free_rate=0.04,
        )
        assert list(df.columns) == _ROLL_COLUMNS
        for _, r in df.iterrows():
            assert np.isfinite(r["roll_ev"])
            assert np.isfinite(r["hold_ev"])
            assert isinstance(r["recommend"], (bool, np.bool_))

    def test_no_profitable_rolls_returns_empty(self):
        """Spec 3: an impossible credit floor prunes every candidate; the
        method returns an empty DataFrame with the correct schema."""
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            iv=0.25,
        )
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 19),
            current_spot=104.0,
            current_iv=0.25,
            risk_free_rate=0.04,
            min_net_credit=1_000_000.0,  # nothing can clear it
        )
        assert df.empty
        assert list(df.columns) == _ROLL_COLUMNS

    def test_recommend_implies_roll_ev_beats_hold_ev(self):
        """Spec 4: structural invariant -- every recommend=True row
        satisfies roll_ev > hold_ev."""
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            iv=0.25,
        )
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 26),
            current_spot=112.0,
            current_iv=0.25,
            risk_free_rate=0.04,
            min_net_credit=-1_500.0,
        )
        for _, r in df[df["recommend"]].iterrows():
            assert r["roll_ev"] > r["hold_ev"], (
                f"recommend=True but roll_ev {r['roll_ev']} <= hold_ev {r['hold_ev']}"
            )

    def test_section2_ev_engine_evaluate_called_per_candidate(self):
        """Spec 5 (section 2 invariant): EVEngine.evaluate is called once
        for the hold trade and once per surviving candidate -- no
        side-channel EV path. Mirrors suggest_rolls's Spec-5 test."""
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            iv=0.25,
        )
        original_evaluate = EVEngine.evaluate

        def _pass_through(self, *args, **kwargs):
            return original_evaluate(self, *args, **kwargs)

        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_eval:
            df = t.suggest_call_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 26),
                current_spot=112.0,
                current_iv=0.25,
                risk_free_rate=0.04,
                min_net_credit=-1_500.0,
            )
        assert mock_eval.call_count == 1 + len(df), (
            f"EVEngine.evaluate called {mock_eval.call_count} times; expected "
            f"{1 + len(df)} (1 hold + 1 per of {len(df)} candidates). A mismatch "
            f"suggests a side-channel EV path."
        )
        assert mock_eval.call_count >= 1


# ======================================================================
# EV-metric regression -- roll_ev nets the FULL buyback
# ======================================================================
class TestRollEvNetsBuybackPrincipal:
    def test_roll_ev_subtracts_full_buyback_not_just_txn_costs(self):
        """roll_ev = ev_dollars(new) - buyback_total_dollars, where
        buyback_total_dollars is the FULL cost to close the current call
        (BSM principal + exit-side txn costs). Under the txn-costs-only
        bug the gap (new_ev_dollars - roll_ev) collapses to a few dollars;
        it must instead be at least the buyback principal."""
        t = _make_tracker_with_covered_call(
            call_strike=105.0,
            call_premium=2.0,
            call_entry=date(2026, 1, 5),
            call_expiry=date(2026, 2, 9),
            iv=0.25,
        )
        df = t.suggest_call_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 26),
            current_spot=112.0,
            current_iv=0.25,
            risk_free_rate=0.04,
            min_net_credit=-1_500.0,
        )
        assert not df.empty
        for _, r in df.iterrows():
            gap = r["new_ev_dollars"] - r["roll_ev"]
            principal = r["buyback_cost"] * 100.0
            assert gap >= principal - 0.5, (
                f"roll_ev gap {gap:.2f} < buyback principal {principal:.2f} -- "
                f"roll_ev is not netting the full buyback cost"
            )


# ======================================================================
# Helper unit tests
# ======================================================================
class TestSolveCallStrike:
    def test_solves_a_real_25_delta(self):
        K = _solve_call_strike(spot=100.0, T=35 / 365.0, r=0.04, q=0.0, iv=0.25, target_delta=0.25)
        assert K is not None
        assert K > 100  # an OTM call strike sits above spot
        otm_pct = (K - 100) / 100
        assert 0.02 < otm_pct < 0.15

    def test_invalid_inputs_return_none(self):
        assert _solve_call_strike(spot=0, T=0.1, r=0.04, q=0, iv=0.25, target_delta=0.25) is None
        assert _solve_call_strike(spot=100, T=0, r=0.04, q=0, iv=0.25, target_delta=0.25) is None
        assert _solve_call_strike(spot=100, T=0.1, r=0.04, q=0, iv=0, target_delta=0.25) is None
        assert _solve_call_strike(spot=100, T=0.1, r=0.04, q=0, iv=0.25, target_delta=1.5) is None
        assert _solve_call_strike(spot=100, T=0.1, r=0.04, q=0, iv=0.25, target_delta=0.0) is None
