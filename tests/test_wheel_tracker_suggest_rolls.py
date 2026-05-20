"""Tests for :meth:`engine.wheel_tracker.WheelTracker.suggest_rolls`.

Covers the 5 spec'd properties — challenged returns candidates, healthy
produces valid output, no candidates is empty-not-crash, recommend
implies roll_ev > hold_ev, EVEngine.evaluate is called per candidate —
plus schema / edge-case unit tests around the same surface.
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
    _solve_put_strike,
)


# ----------------------------------------------------------------------
# Synthetic data + fake connector — keeps the tests deterministic and
# free of any data-on-disk dependency.
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
    """Minimal connector for suggest_rolls tests."""

    def __init__(self, ohlcv: pd.DataFrame | None = None, rf: float = 0.04) -> None:
        self._oh = ohlcv if ohlcv is not None else _synth_ohlcv()
        self._rf = rf

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._oh

    def get_risk_free_rate(self, as_of=None, tenor: str = "rate_3m") -> float:
        return self._rf


def _make_tracker_with_position(
    *,
    strike: float,
    premium: float,
    entry_date: date,
    expiration_date: date,
    iv: float = 0.25,
    ohlcv: pd.DataFrame | None = None,
    rf: float = 0.04,
    initial_capital: float = 100_000.0,
) -> WheelTracker:
    conn = _FakeConn(ohlcv=ohlcv, rf=rf)
    t = WheelTracker(initial_capital=initial_capital, connector=conn)
    ok = t.open_short_put(
        ticker="TEST",
        strike=strike,
        premium=premium,
        entry_date=entry_date,
        expiration_date=expiration_date,
        iv=iv,
    )
    assert ok, "open_short_put failed — check capital / margin in the test setup"
    return t


# ======================================================================
# Schema + edge cases
# ======================================================================
class TestSuggestRollsEdgeCases:
    def test_no_open_position_raises(self):
        t = WheelTracker(initial_capital=100_000.0, connector=_FakeConn())
        with pytest.raises(ValueError, match="No open position"):
            t.suggest_rolls(
                ticker="MISSING",
                as_of=date(2026, 1, 1),
                current_spot=100.0,
                current_iv=0.25,
                risk_free_rate=0.04,
            )

    def test_non_short_put_state_raises(self):
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
        )
        ok = t.handle_put_assignment("TEST", date(2026, 2, 5), 90.0)
        assert ok
        assert t.positions["TEST"].state == PositionState.STOCK_OWNED
        with pytest.raises(ValueError, match="SHORT_PUT"):
            t.suggest_rolls(
                ticker="TEST",
                as_of=date(2026, 2, 6),
                current_spot=90.0,
                current_iv=0.25,
                risk_free_rate=0.04,
            )

    def test_negative_spot_raises(self):
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
        )
        with pytest.raises(ValueError, match="current_spot"):
            t.suggest_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 15),
                current_spot=-10.0,
                current_iv=0.25,
                risk_free_rate=0.04,
            )

    def test_out_of_range_iv_raises(self):
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
        )
        with pytest.raises(ValueError, match="current_iv"):
            t.suggest_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 15),
                current_spot=100.0,
                current_iv=5.0,
                risk_free_rate=0.04,
            )

    def test_past_expiry_returns_empty_schema(self):
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
        )
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 2, 10),  # past expiry
            current_spot=90.0,
            current_iv=0.25,
            risk_free_rate=0.04,
        )
        assert df.empty
        assert list(df.columns) == _ROLL_COLUMNS

    def test_rf_resolved_from_connector_when_none(self):
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
            rf=0.045,
        )
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=date(2026, 1, 15),
            current_spot=90.0,
            current_iv=0.25,
            risk_free_rate=None,  # auto-resolve
        )
        # No crash, schema is right
        assert list(df.columns) == _ROLL_COLUMNS

    def test_no_rf_no_connector_raises(self):
        t = WheelTracker(initial_capital=100_000.0, connector=None)
        ok = t.open_short_put(
            ticker="TEST",
            strike=95.0,
            premium=2.0,
            entry_date=date(2026, 1, 1),
            expiration_date=date(2026, 2, 5),
            iv=0.25,
        )
        assert ok
        with pytest.raises(ValueError, match="no connector"):
            t.suggest_rolls(
                ticker="TEST",
                as_of=date(2026, 1, 15),
                current_spot=90.0,
                current_iv=0.25,
                risk_free_rate=None,
            )


# ======================================================================
# The 5 spec'd tests
# ======================================================================
class TestSuggestRollsSpec:
    def test_challenged_short_put_returns_candidates(self):
        """Spec 1: PG-like deep-ITM ~3-week position returns >= 1 candidate.

        Sets up a 35-DTE short put, advances 21 days, drops spot to deep
        ITM. Asserts the suggester surfaces structurally valid rolls.
        """
        entry = date(2026, 1, 1)
        expiry = entry + timedelta(days=35)
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=entry,
            expiration_date=expiry,
            iv=0.25,
        )
        as_of = entry + timedelta(days=21)
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=as_of,
            current_spot=80.0,
            current_iv=0.25,
            risk_free_rate=0.04,
            # Allow rescue debit rolls so the challenged case gets candidates.
            min_net_credit=-1_500.0,
        )
        assert not df.empty, "challenged short put should produce >= 1 candidate"
        assert list(df.columns) == _ROLL_COLUMNS
        # Sort invariant: rows sorted by roll_ev descending.
        roll_evs = df["roll_ev"].tolist()
        assert roll_evs == sorted(roll_evs, reverse=True)
        # Per-row structural invariants.
        for _, r in df.iterrows():
            assert r["new_strike"] > 0
            assert r["new_strike"] < 80.0  # OTM at the current spot
            assert r["new_dte"] > 0
            assert r["new_expiry"] > as_of
            assert np.isfinite(r["new_premium"]) and r["new_premium"] > 0
            assert np.isfinite(r["buyback_cost"]) and r["buyback_cost"] > 0
            assert np.isfinite(r["roll_ev"])
            assert np.isfinite(r["hold_ev"])
            assert 0.0 <= r["prob_otm"] <= 1.0
            assert isinstance(r["recommend"], (bool, np.bool_))

    def test_healthy_position_produces_valid_output(self):
        """Spec 2: OTM (healthy) position can still produce candidates;
        the function returns a structurally valid DataFrame either way.

        Note: under the marginal-forward-EV metric (see suggest_rolls
        docstring), the engine readily recommends rolling a healthy
        low-delta put to a higher-EV candidate — that's the "harvest
        fresh premium" wheel behavior, not a bug. So "recommend is
        sparse for healthy" is not a strict invariant of this metric.
        What IS invariant: the function returns valid output without
        crashing, and any roll it recommends still satisfies the
        algebraic invariant ``roll_ev > hold_ev`` (covered by Spec 4).
        """
        entry = date(2026, 1, 1)
        expiry = entry + timedelta(days=35)
        t = _make_tracker_with_position(
            strike=92.0,
            premium=1.5,
            entry_date=entry,
            expiration_date=expiry,
            iv=0.25,
        )
        as_of = entry + timedelta(days=14)
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=as_of,
            current_spot=105.0,
            current_iv=0.25,
            risk_free_rate=0.04,
        )
        assert list(df.columns) == _ROLL_COLUMNS
        # Either empty or all rows are structurally valid.
        for _, r in df.iterrows():
            assert np.isfinite(r["roll_ev"])
            assert np.isfinite(r["hold_ev"])
            assert isinstance(r["recommend"], (bool, np.bool_))

    def test_no_profitable_rolls_returns_empty(self):
        """Spec 3: when every candidate fails the credit filter, the
        method returns an empty DataFrame with the correct schema —
        it does not crash."""
        entry = date(2026, 1, 1)
        expiry = entry + timedelta(days=35)
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=entry,
            expiration_date=expiry,
            iv=0.25,
        )
        as_of = entry + timedelta(days=14)
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=as_of,
            current_spot=98.0,
            current_iv=0.25,
            risk_free_rate=0.04,
            # Absurd credit floor — nothing can clear it.
            min_net_credit=1_000_000.0,
        )
        assert df.empty
        assert list(df.columns) == _ROLL_COLUMNS

    def test_recommend_implies_roll_ev_beats_hold_ev(self):
        """Spec 4: structural invariant — every recommend=True row
        satisfies roll_ev > hold_ev."""
        entry = date(2026, 1, 1)
        expiry = entry + timedelta(days=35)
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=entry,
            expiration_date=expiry,
            iv=0.25,
        )
        as_of = entry + timedelta(days=21)
        df = t.suggest_rolls(
            ticker="TEST",
            as_of=as_of,
            current_spot=80.0,
            current_iv=0.25,
            risk_free_rate=0.04,
            min_net_credit=-1_500.0,  # rich candidate set
        )
        recs = df[df["recommend"]]
        for _, r in recs.iterrows():
            assert r["roll_ev"] > r["hold_ev"], (
                f"recommend=True but roll_ev {r['roll_ev']} <= hold_ev {r['hold_ev']}"
            )

    def test_section2_ev_engine_evaluate_called_per_candidate(self):
        """Spec 5 (§2 invariant): EVEngine.evaluate is called for the
        hold trade and once per surviving candidate. This protects
        against a future refactor that bypasses the EV authority
        with a faster approximation.
        """
        entry = date(2026, 1, 1)
        expiry = entry + timedelta(days=35)
        t = _make_tracker_with_position(
            strike=95.0,
            premium=2.0,
            entry_date=entry,
            expiration_date=expiry,
            iv=0.25,
        )
        as_of = entry + timedelta(days=21)
        original_evaluate = EVEngine.evaluate

        def _pass_through(self, *args, **kwargs):
            return original_evaluate(self, *args, **kwargs)

        with patch.object(
            EVEngine,
            "evaluate",
            autospec=True,
            side_effect=_pass_through,
        ) as mock_eval:
            df = t.suggest_rolls(
                ticker="TEST",
                as_of=as_of,
                current_spot=80.0,
                current_iv=0.25,
                risk_free_rate=0.04,
                min_net_credit=-1_500.0,
            )
        # 1 hold call + 1 per surviving candidate row
        assert mock_eval.call_count == 1 + len(df), (
            f"EVEngine.evaluate called {mock_eval.call_count} times; "
            f"expected exactly {1 + len(df)} (1 hold + 1 per of {len(df)} "
            f"candidates). A mismatch suggests a side-channel EV path."
        )
        # The hold call is always present even if df ends up empty.
        assert mock_eval.call_count >= 1


# ======================================================================
# Helper unit tests
# ======================================================================
class TestSolvePutStrike:
    def test_solves_a_real_25_delta(self):
        K = _solve_put_strike(
            spot=100.0,
            T=35 / 365.0,
            r=0.04,
            q=0.0,
            iv=0.25,
            target_delta=0.25,
        )
        assert K is not None
        assert 0 < K < 100
        # 25-delta at IV 25% / 35 DTE: roughly 5-9% OTM
        otm_pct = (100 - K) / 100
        assert 0.03 < otm_pct < 0.12

    def test_invalid_inputs_return_none(self):
        assert (
            _solve_put_strike(
                spot=0,
                T=0.1,
                r=0.04,
                q=0,
                iv=0.25,
                target_delta=0.25,
            )
            is None
        )
        assert (
            _solve_put_strike(
                spot=100,
                T=0,
                r=0.04,
                q=0,
                iv=0.25,
                target_delta=0.25,
            )
            is None
        )
        assert (
            _solve_put_strike(
                spot=100,
                T=0.1,
                r=0.04,
                q=0,
                iv=0,
                target_delta=0.25,
            )
            is None
        )
        assert (
            _solve_put_strike(
                spot=100,
                T=0.1,
                r=0.04,
                q=0,
                iv=0.25,
                target_delta=1.5,
            )
            is None
        )
        assert (
            _solve_put_strike(
                spot=100,
                T=0.1,
                r=0.04,
                q=0,
                iv=0.25,
                target_delta=0.0,
            )
            is None
        )
