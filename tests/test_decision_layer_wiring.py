"""Production-caller wiring: rank_candidates_by_ev → token → tracker,
and PortfolioContext through build_dossiers so R7 / R8 fire live.

The D16 verdict-bound token (PR #145) and the D17 portfolio-risk
soft-warns (PR #165) shipped with no production caller; the audits in
PR #170 / #173 flagged that as the natural next gap. This file pins
the canonical chain:

1. ``WheelTracker.consume_ranker_row(row, entry_date)`` —
   ``issue_ev_authority_token`` + ``open_short_put(current_ev_dollars=...)``
   in one call.
2. ``WheelTracker.portfolio_context_snapshot(spot_prices, ...)`` —
   build a ``PortfolioContext`` from current tracker state.
3. ``build_dossiers(portfolio_context=ctx)`` — attach the context to
   every dossier in a ranking pass so R7 / R8 fire live.

Each test pins one contract; together they prevent the chain from
silently regressing back to dormant.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.candidate_dossier import (
    build_dossiers,
)
from engine.chart_context import ChartContext, Timeframe
from engine.portfolio_risk_gates import PortfolioContext
from engine.wheel_tracker import EVAuthorityRefused, WheelTracker


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------
def _ev_row(ticker: str = "TEST", **overrides) -> dict:
    """A ranker-row shape that satisfies issue_ev_authority_token's
    canonicalisation and open_short_put's positional args.

    Mirrors `engine/wheel_runner.py` row.update at the
    rank_candidates_by_ev tail — every field hashed by
    ``issue_ev_authority_token`` plus the open_short_put-required
    `iv` field.
    """
    row = {
        "ticker": ticker,
        "strike": 95.0,
        "premium": 1.20,
        "dte": 35,
        "ev_dollars": 25.0,
        "prob_profit": 0.72,
        "distribution_source": "empirical_non_overlapping",
        "iv": 0.28,
    }
    row.update(overrides)
    return row


class _StubChartProvider:
    """A `ChartContextProvider` that always returns a clean chart.

    Avoids the filesystem / network dependency of the real providers
    so build_dossiers wiring can be tested in isolation.
    """

    def __init__(self, spot: float = 95.0):
        self.spot = spot

    def fetch(
        self,
        ticker: str,
        timeframe: Timeframe,
        as_of: datetime | None = None,
    ) -> ChartContext:
        return ChartContext(
            ticker=ticker,
            timeframe=timeframe,
            captured_at=datetime(2026, 4, 25, 12, 0, 0),
            screenshot_path=Path("/tmp/stub.png"),
            visible_price=self.spot,
            visible_indicators={},  # no phase → R4 dormant
            source="stub",
        )


# ======================================================================
# 1. consume_ranker_row — ranker → token → open_short_put canonical chain
# ======================================================================
class TestConsumeRankerRow:
    def test_positive_ev_row_opens_position_non_strict(self):
        """Non-strict mode (no token requirement) still routes through
        issue_ev_authority_token and open_short_put successfully."""
        t = WheelTracker(initial_capital=100_000)
        row = _ev_row(ticker="AAPL", ev_dollars=120.0)
        ok = t.consume_ranker_row(row, entry_date=date(2026, 4, 14))
        assert ok is True
        assert "AAPL" in t.positions

    def test_negative_ev_row_refused_at_issuance(self):
        """D16 contract: issue_ev_authority_token raises
        EVAuthorityRefused when ev_dollars <= 0. consume_ranker_row
        propagates so the caller knows R1 refused at the launch gate."""
        t = WheelTracker(initial_capital=100_000)
        row = _ev_row(ticker="AAPL", ev_dollars=-30.0)
        with pytest.raises(EVAuthorityRefused):
            t.consume_ranker_row(row, entry_date=date(2026, 4, 14))
        # Position must NOT have been opened.
        assert "AAPL" not in t.positions
        # The refusal must be in the audit log.
        assert any(
            entry.get("action") == "refuse_issue" and entry.get("reason") == "non_positive_ev"
            for entry in t._ev_authority_log
        )

    def test_zero_ev_row_refused_at_issuance(self):
        """ev_dollars == 0 is also non-positive → refused."""
        t = WheelTracker(initial_capital=100_000)
        row = _ev_row(ticker="AAPL", ev_dollars=0.0)
        with pytest.raises(EVAuthorityRefused):
            t.consume_ranker_row(row, entry_date=date(2026, 4, 14))
        assert "AAPL" not in t.positions

    def test_strict_mode_positive_ev_full_chain(self):
        """Strict mode with a valid +EV row: token issued, token
        consumed with matching current_ev_dollars, position opened.

        Uses $10M NAV (matching the existing TestD17HardBlocks
        fixtures) so the D17 portfolio-delta cap — which scales as
        $300 per $100k NAV — has room for a normal AAPL short put.
        At $100k NAV the cap is just $300, which an ATM AAPL short
        put's dollar delta blows past structurally; the strict-mode
        chain test is therefore the natural place to mirror the
        existing $10M fixture."""
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        row = _ev_row(ticker="AAPL", ev_dollars=120.0, strike=180.0, premium=2.50)
        ok = t.consume_ranker_row(row, entry_date=date(2026, 4, 14))
        assert ok is True
        assert "AAPL" in t.positions
        # Token must have been consumed (set empty after one-use).
        assert len(t._ev_authority_tokens) == 0
        # Log carries an issue + consume pair.
        actions = [e.get("action") for e in t._ev_authority_log]
        assert "issue" in actions
        assert "consume" in actions

    def test_explicit_expiration_date_overrides_dte(self):
        """When expiration_date is passed explicitly, the row's dte
        is ignored. Pins the calendar-control escape hatch."""
        t = WheelTracker(initial_capital=100_000)
        row = _ev_row(ticker="AAPL", ev_dollars=50.0, dte=35)
        ok = t.consume_ranker_row(
            row,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 6, 20),  # 67 days, not 35
        )
        assert ok is True
        pos = t.positions["AAPL"]
        assert pos.put_expiration_date == date(2026, 6, 20)

    def test_duplicate_position_returns_false(self):
        """Open one position; consume a second row for the same ticker
        → token issues, open_short_put refuses (duplicate ticker), the
        helper returns False (no exception)."""
        t = WheelTracker(initial_capital=100_000)
        row = _ev_row(ticker="AAPL", ev_dollars=50.0)
        assert t.consume_ranker_row(row, entry_date=date(2026, 4, 14)) is True
        # Second consume — token issuance succeeds but open_short_put
        # refuses on duplicate.
        second = t.consume_ranker_row(row, entry_date=date(2026, 4, 15))
        assert second is False


# ======================================================================
# 2. portfolio_context_snapshot — tracker state → PortfolioContext
# ======================================================================
class TestPortfolioContextSnapshot:
    def test_empty_tracker_returns_empty_context(self):
        """Fresh tracker → no positions → empty held lists, NAV = cash."""
        t = WheelTracker(initial_capital=100_000)
        ctx = t.portfolio_context_snapshot(today=date(2026, 4, 14))
        assert ctx.held_option_positions == []
        assert ctx.stock_holdings == []
        assert ctx.nav == pytest.approx(100_000.0, abs=1.0)
        assert ctx.spot_prices == {}

    def test_short_put_position_lands_in_held_option_positions(self):
        """Open a short put; snapshot must surface the leg in
        held_option_positions with the expected option-dict fields."""
        t = WheelTracker(initial_capital=100_000)
        t.consume_ranker_row(
            _ev_row(ticker="AAPL", ev_dollars=50.0, strike=180.0, premium=2.50),
            entry_date=date(2026, 4, 14),
        )
        ctx = t.portfolio_context_snapshot(today=date(2026, 4, 14))
        assert len(ctx.held_option_positions) == 1
        leg = ctx.held_option_positions[0]
        assert leg["symbol"] == "AAPL"
        assert leg["option_type"] == "put"
        assert leg["strike"] == 180.0
        assert leg["is_short"] is True

    def test_kwargs_pass_through(self):
        """dealer_regime / returns_data / correlation_matrix /
        volatilities are pass-through; the snapshot doesn't synthesise
        them."""
        t = WheelTracker(initial_capital=100_000)
        regime_map = {"AAPL": "short_gamma_amplifying"}
        returns = pd.DataFrame({"x": [0.01, -0.02, 0.03]})
        ctx = t.portfolio_context_snapshot(
            spot_prices={"AAPL": 180.0},
            dealer_regime_by_ticker=regime_map,
            returns_data=returns,
            today=date(2026, 4, 14),
        )
        assert ctx.dealer_regime_by_ticker == regime_map
        assert ctx.returns_data is returns
        assert ctx.spot_prices == {"AAPL": 180.0}

    def test_nav_reflects_mark_to_market(self):
        """NAV should reflect the mark-to-market call, not a static
        cash read. Hard to assert exact value without a full mock; pin
        non-negative and finite."""
        t = WheelTracker(initial_capital=100_000)
        t.consume_ranker_row(
            _ev_row(ticker="AAPL", ev_dollars=50.0, strike=180.0, premium=2.50),
            entry_date=date(2026, 4, 14),
        )
        ctx = t.portfolio_context_snapshot(spot_prices={"AAPL": 180.0}, today=date(2026, 4, 14))
        assert ctx.nav > 0
        # NAV stays close to initial capital after one CSP — premium credited,
        # mark-to-market liability for the short put. ±$5k tolerance.
        assert abs(ctx.nav - 100_000.0) < 5_000.0


# ======================================================================
# 3. build_dossiers(portfolio_context=...) — context attaches to every dossier
# ======================================================================
class TestBuildDossiersPortfolioContext:
    def _ev_frame(self, n: int = 3) -> pd.DataFrame:
        """Minimal ev_frame for build_dossiers — enough columns to
        survive the reviewer's R1 / R5 path."""
        rows = []
        for i in range(n):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "strike": 100.0 + i,
                    "premium": 2.0,
                    "dte": 30,
                    "ev_dollars": 50.0 + i * 10,
                    "iv": 0.25,
                    "prob_profit": 0.72,
                    "distribution_source": "empirical_non_overlapping",
                    "spot": 100.0 + i,
                }
            )
        return pd.DataFrame(rows)

    def test_default_no_context_attached(self):
        """Without the kwarg, portfolio_context is None on every
        dossier — matches today's behaviour."""
        df = self._ev_frame(n=3)
        dossiers = build_dossiers(df, _StubChartProvider(), top_n=3)
        assert len(dossiers) == 3
        assert all(d.portfolio_context is None for d in dossiers)

    def test_passed_context_attached_to_every_dossier(self):
        """portfolio_context=ctx → every dossier carries the same
        instance (shared by reference; one snapshot per pass)."""
        df = self._ev_frame(n=3)
        ctx = PortfolioContext(nav=100_000.0, spot_prices={"T0": 100.0})
        dossiers = build_dossiers(df, _StubChartProvider(), top_n=3, portfolio_context=ctx)
        assert len(dossiers) == 3
        for d in dossiers:
            assert d.portfolio_context is ctx  # same instance, not a copy


# ======================================================================
# 4. End-to-end: PortfolioContext through build_dossiers fires R7 / R8
# ======================================================================
class TestR7R8FireLiveThroughBuildDossiers:
    """Pin the production wire: a constructed PortfolioContext, passed
    through build_dossiers, actually fires R7 / R8 — not just when
    attached post-hoc on a CandidateDossier as the existing
    TestD17DossierSoftWarns class tests."""

    def _ev_frame_one_row(self, ticker: str, ev_dollars: float = 50.0):
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "strike": 100.0,
                    "premium": 2.0,
                    "dte": 30,
                    "ev_dollars": ev_dollars,
                    "iv": 0.25,
                    "prob_profit": 0.72,
                    "distribution_source": "empirical_non_overlapping",
                    "spot": 100.0,
                }
            ]
        )

    def test_r7_var_breach_fires_through_build_dossiers(self):
        """A heavy-vol returns_data + tiny NAV makes VaR breach the
        5% cap; build_dossiers passes the context, R7 fires, verdict
        downgrades from proceed → review."""
        idx = pd.date_range("2026-01-01", periods=120, freq="B")
        returns = pd.DataFrame(
            {"portfolio": np.random.default_rng(7).normal(0, 0.08, 120)},
            index=idx,
        )
        ctx = PortfolioContext(
            nav=10_000.0,  # tiny NAV → R7 fires easily
            spot_prices={"TEST": 100.0},
            returns_data=returns,
        )
        df = self._ev_frame_one_row("TEST", ev_dollars=50.0)
        dossiers = build_dossiers(
            df, _StubChartProvider(spot=100.0), top_n=1, portfolio_context=ctx
        )
        assert len(dossiers) == 1
        d = dossiers[0]
        assert d.verdict == "review"
        assert d.verdict_reason == "portfolio_var_breach"

    def test_r8_short_gamma_regime_fires_through_build_dossiers(self):
        """Candidate ticker in short_gamma_amplifying regime → R8
        fires through the build_dossiers path."""
        ctx = PortfolioContext(
            nav=10_000_000.0,  # huge NAV so R7 / stress pass
            spot_prices={"TEST": 100.0},
            dealer_regime_by_ticker={"TEST": "short_gamma_amplifying"},
        )
        df = self._ev_frame_one_row("TEST", ev_dollars=50.0)
        dossiers = build_dossiers(
            df, _StubChartProvider(spot=100.0), top_n=1, portfolio_context=ctx
        )
        assert len(dossiers) == 1
        d = dossiers[0]
        assert d.verdict == "review"
        assert d.verdict_reason == "short_gamma_regime"

    def test_r7_r8_cannot_upgrade_negative_ev_through_wire(self):
        """§2-adjacent: even with an attached PortfolioContext,
        R1 (negative EV → blocked) wins. R7 / R8 only fire when
        verdict is currently proceed, and the reviewer's verdict
        starts at blocked for ev<=0 candidates."""
        ctx = PortfolioContext(
            nav=10_000_000.0,
            spot_prices={"TEST": 100.0},
            dealer_regime_by_ticker={"TEST": "short_gamma_amplifying"},
        )
        df = self._ev_frame_one_row("TEST", ev_dollars=-50.0)
        dossiers = build_dossiers(
            df, _StubChartProvider(spot=100.0), top_n=1, portfolio_context=ctx
        )
        d = dossiers[0]
        # R1 blocks first; R8 never gets a chance to fire because
        # R8 is gated on verdict == "proceed".
        assert d.verdict == "blocked"
        assert d.verdict_reason == "negative_ev"


# ======================================================================
# 5. End-to-end smoke: snapshot → build_dossiers → consume top row
# ======================================================================
class TestEndToEndChain:
    """Smoke test the full operator-facing flow: take a ranker row,
    snapshot the tracker into a PortfolioContext, attach to dossiers,
    consume the top dossier's row via consume_ranker_row."""

    def test_snapshot_then_consume_top_row_round_trip(self):
        """Empty tracker → snapshot (NAV = cash, empty positions) →
        build dossier with that context → consume the top row →
        position appears in tracker."""
        t = WheelTracker(initial_capital=100_000)
        ctx = t.portfolio_context_snapshot(today=date(2026, 4, 14))
        # Build a one-row dossier set.
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "strike": 180.0,
                    "premium": 2.50,
                    "dte": 30,
                    "ev_dollars": 120.0,
                    "iv": 0.25,
                    "prob_profit": 0.72,
                    "distribution_source": "empirical_non_overlapping",
                    "spot": 180.0,
                }
            ]
        )
        dossiers = build_dossiers(
            df, _StubChartProvider(spot=180.0), top_n=1, portfolio_context=ctx
        )
        d = dossiers[0]
        # Empty tracker + positive EV → proceed.
        assert d.verdict == "proceed"
        # Consume the row through the canonical chain.
        ok = t.consume_ranker_row(d.ev_row, entry_date=date(2026, 4, 14))
        assert ok is True
        assert "AAPL" in t.positions
        # Re-snapshot now reflects the new position.
        ctx2 = t.portfolio_context_snapshot(spot_prices={"AAPL": 180.0}, today=date(2026, 4, 14))
        assert len(ctx2.held_option_positions) == 1
        assert ctx2.held_option_positions[0]["symbol"] == "AAPL"
