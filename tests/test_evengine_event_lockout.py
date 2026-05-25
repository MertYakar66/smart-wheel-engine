"""Tests for EVEngine.evaluate's event-lockout short-circuit.

`engine/ev_engine.py:262-285` short-circuits evaluate() before the
forward-distribution, BSM, cost, regime, and dealer math whenever the
attached EventGate reports a candidate as blocked. Pre-this-file,
``tests/test_ev_engine_upgrades.py`` had zero coverage of this path
(no ``event`` / ``earnings`` / ``lockout`` mentions); the EventGate
class itself is covered by ``tests/test_event_gate.py`` but its
wire-in to EVEngine.evaluate is not.

These tests use the real production fixtures (real EventGate, real
ScheduledEvent, real ShortOptionTrade, real EVEngine) and assert on
the documented short-circuit return shape:

  * ``ev_dollars = 0.0`` (no rescue path even with dealer positioning
    attached)
  * ``event_lockout_reason`` populated with the gate's reason string
  * ``metadata = {"blocked": True}``
  * ``expected_days_held`` and ``regime_multiplier`` passed through from
    the trade
  * Tail diagnostics (``cvar_99_evt``, ``tail_xi``, ``heavy_tail``) and
    dealer fields default to NaN / NaN / False / identity since the
    short-circuit returns before they are populated

The boundary cases (event at ``trade_start``, ``trade_end``, within
back-buffer, within forward-buffer) exercise
``EventGate._event_touches_window``'s symmetric arithmetic through the
integration with EVEngine.evaluate — complementary to PR #180's
coverage of the same arithmetic from the data-layer side via
``get_recent_earnings``.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np

from engine.dealer_positioning import DealerAssumption, MarketStructure
from engine.ev_engine import EVEngine, ShortOptionTrade
from engine.event_gate import EventGate, ScheduledEvent


# ----------------------------------------------------------------------
# Shared fixtures (constructed in-test using real production classes)
# ----------------------------------------------------------------------
def _trade(
    *,
    underlying: str = "TEST",
    spot: float = 100.0,
    strike: float = 95.0,
    premium: float = 1.50,
    dte: int = 35,
    iv: float = 0.28,
    regime_multiplier: float = 1.0,
) -> ShortOptionTrade:
    """Real-shape ShortOptionTrade with sensible defaults for tests."""
    return ShortOptionTrade(
        option_type="put",
        underlying=underlying,
        spot=spot,
        strike=strike,
        premium=premium,
        dte=dte,
        iv=iv,
        regime_multiplier=regime_multiplier,
    )


def _gate_with_earnings(
    event_date: date,
    *,
    ticker: str = "TEST",
    earnings_buffer_days: int = 5,
) -> EventGate:
    """Real EventGate carrying a single earnings ScheduledEvent."""
    gate = EventGate(earnings_buffer_days=earnings_buffer_days)
    gate.add_event(ScheduledEvent(ticker=ticker, kind="earnings", event_date=event_date))
    return gate


def _synthetic_forward_returns(n: int = 250, seed: int = 1) -> np.ndarray:
    """Synthetic log-return path for the non-blocked tests.

    Synthetic is appropriate because the test surface (the lockout
    short-circuit at ``engine/ev_engine.py:262-285``) returns BEFORE the
    forward-distribution math; for the non-blocked tests we just need
    ``evaluate()`` to run to completion, not realistic distribution
    statistics. Same convention as ``tests/test_ev_engine_upgrades.py``.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.012, n)


# ======================================================================
# 1-5. Blocking cases — gate fires, short-circuit returns
# ======================================================================
class TestLockoutBlocking:
    """The short-circuit fires when EventGate.is_blocked returns True."""

    def test_event_on_trade_start_blocks(self):
        """Boundary case: earnings exactly ON trade_start."""
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)  # 35 DTE
        engine = EVEngine(event_gate=_gate_with_earnings(trade_start))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        assert result.ev_dollars == 0.0
        assert result.event_lockout_reason.startswith("event_lockout:earnings@")
        assert "2026-04-14" in result.event_lockout_reason
        assert result.metadata == {"blocked": True}

    def test_event_on_trade_end_blocks(self):
        """Other boundary: earnings exactly ON trade_end."""
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        engine = EVEngine(event_gate=_gate_with_earnings(trade_end))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        assert result.ev_dollars == 0.0
        assert "earnings@2026-05-19" in result.event_lockout_reason
        assert result.metadata == {"blocked": True}

    def test_event_strictly_inside_window_blocks(self):
        """Headline case: earnings midway through the DTE window."""
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 4, 28)))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        assert result.ev_dollars == 0.0
        assert "earnings@2026-04-28" in result.event_lockout_reason
        # The gate's symmetric ±5d buffer signature.
        assert "(±5d buffer)" in result.event_lockout_reason

    def test_event_within_back_buffer_before_window_blocks(self):
        """Earnings 2d BEFORE trade_start, buffer 5d → symmetric arithmetic blocks.

        Pins ``_event_touches_window``'s ``window_start = trade_start -
        timedelta(days=buf)`` extension through the EVEngine.evaluate
        integration.
        """
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 4, 12), earnings_buffer_days=5))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        assert result.ev_dollars == 0.0
        assert "earnings@2026-04-12" in result.event_lockout_reason
        assert result.metadata == {"blocked": True}

    def test_event_within_forward_buffer_after_window_blocks(self):
        """Earnings 2d AFTER trade_end, buffer 5d → forward buffer blocks.

        Pins ``window_end = trade_end + timedelta(days=buf)`` — the
        forward half of the same symmetric arithmetic.
        """
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 5, 21), earnings_buffer_days=5))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        assert result.ev_dollars == 0.0
        assert "earnings@2026-05-21" in result.event_lockout_reason
        assert result.metadata == {"blocked": True}


# ======================================================================
# 6-8. Non-blocking cases — short-circuit does NOT fire
# ======================================================================
class TestLockoutNonBlocking:
    """The short-circuit does NOT fire — evaluate runs to completion."""

    def test_event_outside_buffer_does_not_block(self):
        """Earnings 10d after trade_end, buffer 5d → outside window → no block."""
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 5, 29), earnings_buffer_days=5))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )
        # evaluate() ran to completion → event_lockout_reason stays empty.
        assert result.event_lockout_reason == ""
        # ``metadata`` may carry other keys populated by the full path
        # (e.g. distribution diagnostics); only the blocked-branch key
        # must be absent. Avoid asserting ev_dollars sign — depends on
        # BSM + synthetic returns and is not the test surface.
        assert "blocked" not in result.metadata

    def test_no_event_gate_means_no_short_circuit(self):
        """EVEngine(event_gate=None) runs evaluate regardless of trade dates."""
        engine = EVEngine(event_gate=None)
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=date(2026, 4, 14),
            trade_end=date(2026, 5, 19),
        )
        assert result.event_lockout_reason == ""
        assert "blocked" not in result.metadata

    def test_missing_trade_dates_skips_short_circuit(self):
        """event_gate attached but trade_start/trade_end = None → guard skips.

        Pins the second half of the guard ``if self.event_gate is not None
        and trade_start is not None and trade_end is not None``.
        """
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 4, 28)))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=None,
            trade_end=None,
        )
        assert result.event_lockout_reason == ""
        assert "blocked" not in result.metadata


# ======================================================================
# 9. Schema regression — blocked EVResult shape
# ======================================================================
class TestBlockedEVResultSchema:
    """Pin the full short-circuit return shape so the contract doesn't drift."""

    def test_blocked_evresult_schema_is_fully_zeroed_except_documented_passthroughs(
        self,
    ):
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        trade = _trade(dte=35, regime_multiplier=1.05)
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 4, 28)))
        result = engine.evaluate(
            trade=trade,
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
        )

        # --- Zeroed fields (per engine/ev_engine.py:266-285) ---
        assert result.ev_dollars == 0.0
        assert result.ev_per_day == 0.0
        assert result.prob_profit == 0.0
        assert result.prob_assignment == 0.0
        assert result.prob_touch == 0.0
        assert result.cvar_5 == 0.0
        assert result.omega_ratio == 0.0
        assert result.edge_vs_fair == 0.0
        assert result.fair_value == 0.0
        assert result.total_transaction_cost == 0.0
        assert result.breakeven_move_pct == 0.0
        assert result.mean_pnl == 0.0
        assert result.std_pnl == 0.0
        assert result.skew_pnl == 0.0

        # --- Passthrough fields (preserve trade's values) ---
        assert result.expected_days_held == float(max(trade.dte, 1))
        assert result.regime_multiplier == float(trade.regime_multiplier)

        # --- Tail / dealer fields are NOT set explicitly in the
        #     short-circuit; they fall through to dataclass defaults. ---
        assert math.isnan(result.cvar_99_evt)
        assert math.isnan(result.tail_xi)
        assert result.heavy_tail is False
        assert result.dealer_regime == ""
        assert result.dealer_multiplier == 1.0
        assert math.isnan(result.gex_total)
        assert math.isnan(result.gamma_flip_distance_pct)
        assert math.isnan(result.nearest_put_wall_strike)
        assert math.isnan(result.nearest_call_wall_strike)
        assert result.pinning_zones == []

        # --- Event-lockout-specific fields ---
        assert result.event_lockout_reason.startswith("event_lockout:earnings@")
        assert "(±5d buffer)" in result.event_lockout_reason
        assert result.metadata == {"blocked": True}


# ======================================================================
# 10. §2-adjacent — dealer multiplier NOT applied on the blocked path
# ======================================================================
class TestBlockedPathBypassesDealerMultiplier:
    """The short-circuit returns BEFORE the regime/dealer math, so no
    dealer rescue can lift a blocked candidate. §2-adjacent: reviewers
    can downgrade but no input can upgrade a blocked verdict to
    tradeable. Pins the call-order documented at
    ``engine/ev_engine.py:262-285``.
    """

    def test_blocked_path_does_not_apply_dealer_multiplier(self):
        trade_start = date(2026, 4, 14)
        trade_end = date(2026, 5, 19)
        # A real MarketStructure that, if applied, would scale ev_dollars
        # via dealer_regime_multiplier (clamped to [0.70, 1.05]).
        market_structure = MarketStructure(
            ticker="TEST",
            as_of=datetime(2026, 4, 14, 12, 0, 0),
            spot=100.0,
            expiry=date(2026, 5, 19),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="short_gamma_amplifying",
            confidence=0.9,
        )
        engine = EVEngine(event_gate=_gate_with_earnings(date(2026, 4, 28)))
        result = engine.evaluate(
            trade=_trade(),
            forward_log_returns=_synthetic_forward_returns(),
            trade_start=trade_start,
            trade_end=trade_end,
            market_structure=market_structure,
        )
        # Blocked: ev_dollars == 0.0 — no dealer rescue.
        assert result.ev_dollars == 0.0
        assert result.metadata == {"blocked": True}
        # The dealer fields are untouched by the short-circuit (defaults).
        # If dealer math ran, dealer_regime would be
        # "short_gamma_amplifying" and dealer_multiplier would be in
        # (0.70, 1.0) per the helper's known mapping for that regime.
        assert result.dealer_regime == ""
        assert result.dealer_multiplier == 1.0
