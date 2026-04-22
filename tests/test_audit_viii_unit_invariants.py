"""
AUDIT-VIII unit / P&L regression tests.

Pins down the critical-issue fixes from the audit-VIII review:

* P0.1 — IV percent-vs-decimal normalisation in
  ``WheelRunner.rank_candidates_by_ev``. Bloomberg fundamentals return
  IV as a percent (e.g. ``26.17`` means 26.17%); the pre-fix code
  rejected that as ``iv > 5`` and silently produced zero tradeable
  candidates.

* P0.2 — Risk-free rate percent-vs-decimal in
  ``WheelRunner.rank_candidates_by_ev``. ``MarketDataConnector.get_risk_free_rate``
  returns the raw treasury CSV value which is also in percent
  (e.g. ``4.333`` means 4.333%). The pre-fix code used it directly in
  BSM, which blew up ``d1`` and made every synthetic put premium
  collapse below the ``0.05`` trade filter — again producing zero rows.

* P0.3 — ``datetime`` import missing for ``POST /api/news/ingest``
  would raise ``NameError`` the first time the endpoint was called.

* P1.1 — ``WheelTracker`` roll/close accounting. Closing a rolled put
  or a rolled covered call used to overwrite / double-count the
  running ``realized_pnl`` accumulator, producing wrong P&L on the
  final close of any rolled position. Covered by the small arithmetic
  traces below.

These tests are deliberately *invariant* tests: they don't depend on
any specific candidate emerging from real data, just on the unit
arithmetic behind the fixes.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner
from engine.wheel_tracker import PositionState, WheelTracker


# ======================================================================
# P0.1/P0.2 — unit normalisation inside rank_candidates_by_ev
# ======================================================================
def _make_ohlcv(n: int = 2400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )


class _PercentConnector:
    """Connector that returns IV and risk-free rate in PERCENT form —
    the Bloomberg format used by the real connector."""

    def __init__(self):
        self._ohlcv = _make_ohlcv()

    def get_universe(self):
        return ["AAA"]

    def get_ohlcv(self, ticker):
        return self._ohlcv

    def get_fundamentals(self, ticker):
        # IV & volatility in PERCENT (e.g. 26.17 means 26.17%).
        return {
            "implied_vol_atm": 26.17,
            "volatility_30d": 27.0,
            "dividend_yield": 0.6,  # percent
        }

    def get_risk_free_rate(self, as_of=None, tenor="rate_3m"):
        # Treasury CSV returns percent.
        return 4.333

    def get_next_earnings(self, ticker, as_of=None):
        return None

    def get_options(self, ticker):
        return pd.DataFrame()


class TestPercentDecimalNormalisation:
    def test_ev_ranker_accepts_percent_iv_and_rf_rate(self):
        runner = WheelRunner()
        with patch.object(
            WheelRunner,
            "connector",
            new_callable=lambda: property(lambda self: _PercentConnector()),
        ):
            df = runner.rank_candidates_by_ev(
                tickers=["AAA"],
                dte_target=35,
                delta_target=0.25,
                top_n=5,
                min_ev_dollars=-1e9,
                enforce_history_gate=False,
                enforce_chain_quality_gate=False,
            )
        # Before the audit-VIII fix this was an empty DataFrame because
        # IV 26.17 was treated as a decimal and rejected as degenerate.
        assert df is not None
        assert len(df) == 1
        row = df.iloc[0]
        assert row["ticker"] == "AAA"
        # IV was correctly normalised from 26.17% to ~0.2617.
        assert 0.20 <= float(row["iv"]) <= 0.30
        # Premium is sane (pre-fix was collapsed to < 0.05 by a
        # catastrophic risk-free rate of 4.333 treated as 433.3%).
        assert 0.3 <= float(row["premium"]) <= 20.0
        # Prob of profit is a valid probability.
        assert 0.0 <= float(row["prob_profit"]) <= 1.0

    def test_iv_boundary_decimal_untouched(self):
        """IVs already in decimal form (e.g. 0.28) must not get divided
        by 100 again. The guard ``iv > 3`` protects this path."""
        runner = WheelRunner()

        class DecimalConn(_PercentConnector):
            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.28,
                    "volatility_30d": 0.29,
                    "dividend_yield": 0.005,
                }

            def get_risk_free_rate(self, as_of=None, tenor="rate_3m"):
                return 0.0425  # already decimal

        with patch.object(
            WheelRunner,
            "connector",
            new_callable=lambda: property(lambda self: DecimalConn()),
        ):
            df = runner.rank_candidates_by_ev(
                tickers=["AAA"],
                dte_target=35,
                delta_target=0.25,
                top_n=5,
                min_ev_dollars=-1e9,
                enforce_history_gate=False,
                enforce_chain_quality_gate=False,
            )
        assert len(df) == 1
        assert 0.27 <= float(df.iloc[0]["iv"]) <= 0.30


# ======================================================================
# P0.3 — datetime module available for news ingest
# ======================================================================
class TestNewsIngestDatetimeImport:
    def test_engine_api_exports_datetime(self):
        """The POST /api/news/ingest handler uses ``datetime.utcnow``
        at import-time-of-call. If ``datetime`` is not imported at
        the module level, the first call raises NameError and the
        endpoint is effectively broken."""
        import engine_api

        assert hasattr(engine_api, "datetime"), (
            "engine_api module must expose datetime (needed by "
            "_handle_news_ingest)"
        )
        # Smoke-check: timezone.utc is also needed for the fix.
        assert hasattr(engine_api, "timezone")


# ======================================================================
# P1.1 — WheelTracker P&L accumulator invariants
# ======================================================================
class TestWheelTrackerPnLAccumulator:
    def _open_put(self, t, ticker, strike, prem, d0, days=30, iv=0.25):
        return t.open_short_put(
            ticker=ticker,
            strike=strike,
            premium=prem,
            entry_date=d0,
            expiration_date=d0 + timedelta(days=days),
            iv=iv,
        )

    def test_rolled_put_preserves_prior_leg_pnl(self):
        """A rolled put closed after one roll must reflect BOTH legs'
        P&L. The pre-audit code overwrote the accumulator in
        ``close_short_put`` and silently dropped the first leg."""
        t = WheelTracker(initial_capital=500_000)
        d0 = date(2026, 1, 1)
        ok = self._open_put(t, "AAPL", 100, 2.00, d0)
        assert ok
        # Roll: buy back at 0.50 (win $1.50 on leg 1), open new put at 2.50
        roll = t.roll_put(
            ticker="AAPL",
            roll_date=d0 + timedelta(days=15),
            new_strike=98,
            new_premium=2.50,
            new_expiration=d0 + timedelta(days=60),
            new_iv=0.27,
            buyback_price=0.50,
        )
        assert roll is not None
        # Close leg 2 at 1.00 (win $1.50 on leg 2).
        closed = t.close_short_put(
            ticker="AAPL",
            buyback_price=1.00,
            exit_date=d0 + timedelta(days=45),
            reason="profit_target",
        )
        assert closed is not None
        # Expected gross P&L = (2.00 - 0.50 + 2.50 - 1.00) * 100 = $300
        # Net = 300 - total transaction costs
        assert closed["realized_pnl"] == pytest.approx(300.0, rel=1e-6)
        assert closed["net_pnl"] == pytest.approx(
            300.0 - closed["transaction_costs"], rel=1e-6
        )

    def test_double_rolled_put_preserves_all_three_legs(self):
        """Two rolls → three legs. The pre-audit ``roll_put`` also
        overwrote the accumulator on the *second* roll, silently
        dropping the first leg's credit."""
        t = WheelTracker(initial_capital=500_000)
        d0 = date(2026, 1, 1)
        assert self._open_put(t, "MSFT", 200, 3.00, d0)
        t.roll_put(
            ticker="MSFT",
            roll_date=d0 + timedelta(days=10),
            new_strike=198,
            new_premium=3.50,
            new_expiration=d0 + timedelta(days=60),
            new_iv=0.27,
            buyback_price=1.00,
        )
        t.roll_put(
            ticker="MSFT",
            roll_date=d0 + timedelta(days=25),
            new_strike=196,
            new_premium=4.00,
            new_expiration=d0 + timedelta(days=90),
            new_iv=0.30,
            buyback_price=1.50,
        )
        closed = t.close_short_put(
            ticker="MSFT",
            buyback_price=2.00,
            exit_date=d0 + timedelta(days=75),
            reason="profit_target",
        )
        # Expected gross P&L = (3.00 - 1.00 + 3.50 - 1.50 + 4.00 - 2.00) * 100 = $600
        assert closed["realized_pnl"] == pytest.approx(600.0, rel=1e-6)

    def test_rolled_covered_call_preserves_prior_leg_pnl(self):
        """Analogous invariant for covered calls — previously the
        ``roll_call`` path double-counted premiums by using ``+=``
        on the same running accumulator that ``open_covered_call``
        had already credited."""
        t = WheelTracker(initial_capital=500_000)
        d0 = date(2026, 1, 1)
        # Drive state to STOCK_OWNED
        assert self._open_put(t, "KO", 50, 1.00, d0)
        assert t.handle_put_assignment("KO", d0 + timedelta(days=30), 48.0)
        assert t.positions["KO"].state == PositionState.STOCK_OWNED
        # Sell first CC and roll it once.
        assert t.open_covered_call(
            ticker="KO",
            strike=52,
            premium=0.80,
            entry_date=d0 + timedelta(days=31),
            expiration_date=d0 + timedelta(days=61),
            iv=0.22,
        )
        roll = t.roll_call(
            ticker="KO",
            roll_date=d0 + timedelta(days=45),
            new_strike=53,
            new_premium=1.00,
            new_expiration=d0 + timedelta(days=90),
            new_iv=0.24,
            buyback_price=0.20,
        )
        assert roll is not None
        # Close second CC at 0.30 (win 0.70 on leg 2).
        result = t.close_covered_call(
            ticker="KO",
            buyback_price=0.30,
            exit_date=d0 + timedelta(days=75),
            reason="profit_target",
        )
        assert result is not None
        # Expected gross P&L across all legs:
        #   put credit:        1.00 * 100 = +100
        #   CC1 net:  (0.80 - 0.20) * 100 = +60
        #   CC2 net:  (1.00 - 0.30) * 100 = +70
        # Position is still open (stock), so check the running accumulator.
        assert t.positions["KO"].realized_pnl == pytest.approx(230.0, rel=1e-6)
