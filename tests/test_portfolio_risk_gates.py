"""Unit tests for ``engine/portfolio_risk_gates.py`` (D17 / #154 C4
Phase 1).

Pins:
- The adapter ``take_snapshot`` produces the right shape for each
  ``WheelPosition`` state (SHORT_PUT, STOCK_OWNED, COVERED_CALL).
- Each of the five gate functions returns ``GateResult`` with the
  correct ``passed``/``reason``/``details`` for both pass and fail
  paths, against the locked D17 defaults.
- The "missing data → skip" semantics from Q3 (VaR, dealer regime).
- The audit-log details bag carries the field names the schema-
  closure regression (``tests/test_ev_authority_log_schema.py``)
  will pin in Phase 2.

No tracker / dossier integration here — Phase 1 is the gate library
alone. Phase 2 wires the hard-blocks; Phase 3 wires the soft-warns.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from engine.portfolio_risk_gates import (
    _C4_VOL_SPIKE_SCENARIO,
    GateResult,
    PortfolioSnapshot,
    check_dealer_regime,
    check_kelly_size,
    check_portfolio_delta,
    check_sector_cap,
    check_stress_scenario,
    check_var,
    take_snapshot,
)
from engine.wheel_tracker import PositionState, WheelPosition

# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------
_TODAY = date(2026, 5, 23)
_EXPIRY = date(2026, 6, 19)


def _put_position(ticker: str, strike: float, iv: float = 0.25) -> WheelPosition:
    """Build a SHORT_PUT WheelPosition for adapter / gate tests."""
    return WheelPosition(
        ticker=ticker,
        state=PositionState.SHORT_PUT,
        entry_date=_TODAY,
        put_strike=strike,
        put_premium=strike * 0.02,
        put_entry_date=_TODAY,
        put_dte_at_entry=27,
        put_entry_iv=iv,
        put_expiration_date=_EXPIRY,
    )


def _stock_position(ticker: str, shares: int, basis: float) -> WheelPosition:
    """Build a STOCK_OWNED position (post-assignment)."""
    return WheelPosition(
        ticker=ticker,
        state=PositionState.STOCK_OWNED,
        entry_date=_TODAY,
        stock_shares=shares,
        stock_basis=basis,
        stock_acquisition_date=_TODAY,
    )


def _covered_call_position(
    ticker: str,
    shares: int,
    basis: float,
    call_strike: float,
    iv: float = 0.22,
) -> WheelPosition:
    """Build a COVERED_CALL position (assigned + sold call)."""
    return WheelPosition(
        ticker=ticker,
        state=PositionState.COVERED_CALL,
        entry_date=_TODAY,
        stock_shares=shares,
        stock_basis=basis,
        stock_acquisition_date=_TODAY,
        call_strike=call_strike,
        call_premium=call_strike * 0.015,
        call_entry_date=_TODAY,
        call_dte_at_entry=27,
        call_entry_iv=iv,
        call_expiration_date=_EXPIRY,
    )


def _candidate_put(ticker: str = "TEST", strike: float = 100.0) -> dict:
    """A canonical short-put candidate dict (matches the adapter shape)."""
    return {
        "symbol": ticker,
        "option_type": "put",
        "strike": strike,
        "dte": 27,
        "iv": 0.25,
        "contracts": 1,
        "is_short": True,
    }


# ======================================================================
# 1. take_snapshot — the WheelPosition → upstream-API adapter
# ======================================================================
class TestTakeSnapshot:
    def test_empty_returns_empty_snapshot(self):
        snap = take_snapshot({}, today=_TODAY)
        assert isinstance(snap, PortfolioSnapshot)
        assert snap.option_positions == []
        assert snap.stock_holdings == []

    def test_short_put_emits_option_only(self):
        snap = take_snapshot({"AAPL": _put_position("AAPL", 180.0)}, today=_TODAY)
        assert len(snap.option_positions) == 1
        assert snap.stock_holdings == []
        opt = snap.option_positions[0]
        assert opt["symbol"] == "AAPL"
        assert opt["option_type"] == "put"
        assert opt["strike"] == 180.0
        assert opt["iv"] == 0.25
        assert opt["contracts"] == 1
        assert opt["is_short"] is True
        # DTE = (expiry - today).days
        assert opt["dte"] == (_EXPIRY - _TODAY).days

    def test_stock_owned_emits_holding_only(self):
        snap = take_snapshot({"MSFT": _stock_position("MSFT", 100, 350.0)}, today=_TODAY)
        assert snap.option_positions == []
        assert snap.stock_holdings == [("MSFT", 100)]

    def test_covered_call_emits_both(self):
        snap = take_snapshot(
            {"AAPL": _covered_call_position("AAPL", 100, 180.0, 190.0)},
            today=_TODAY,
        )
        assert len(snap.option_positions) == 1
        assert snap.option_positions[0]["option_type"] == "call"
        assert snap.option_positions[0]["strike"] == 190.0
        assert snap.stock_holdings == [("AAPL", 100)]

    def test_mixed_portfolio(self):
        snap = take_snapshot(
            {
                "AAPL": _put_position("AAPL", 180.0),
                "MSFT": _stock_position("MSFT", 100, 350.0),
                "GOOG": _covered_call_position("GOOG", 50, 150.0, 160.0),
            },
            today=_TODAY,
        )
        assert len(snap.option_positions) == 2  # put + call
        assert len(snap.stock_holdings) == 2  # msft + goog
        symbols = {p["symbol"] for p in snap.option_positions}
        assert symbols == {"AAPL", "GOOG"}
        holdings = dict(snap.stock_holdings)
        assert holdings == {"MSFT": 100, "GOOG": 50}

    def test_dte_floors_at_zero_for_past_expiry(self):
        pos = _put_position("AAPL", 180.0)
        pos.put_expiration_date = _TODAY - timedelta(days=5)
        snap = take_snapshot({"AAPL": pos}, today=_TODAY)
        assert snap.option_positions[0]["dte"] == 0

    def test_short_put_with_missing_iv_is_skipped(self):
        pos = _put_position("AAPL", 180.0)
        pos.put_entry_iv = None
        snap = take_snapshot({"AAPL": pos}, today=_TODAY)
        # Skipped silently — no option dict for malformed state.
        assert snap.option_positions == []


# ======================================================================
# 2. check_sector_cap — tracker hard-block
# ======================================================================
class TestCheckSectorCap:
    def test_empty_portfolio_passes(self):
        result = check_sector_cap(
            symbol="AAPL",
            proposed_notional=18_000.0,  # 100 * 180
            held_option_positions=[],
            nav=100_000.0,
        )
        assert result.passed is True
        assert result.reason is None
        # DEFAULT_SECTOR_MAP labels AAPL as "Information Technology"
        # (the full GICS sector name).
        assert result.details["sector"] == "Information Technology"

    def test_breach_at_default_25_pct(self):
        # Add ~$24k AAPL already, propose another $5k → 29% in Tech sector.
        held = [
            {"symbol": "AAPL", "strike": 240.0, "contracts": 1},  # $24k notional
        ]
        result = check_sector_cap(
            symbol="AAPL",
            proposed_notional=5_000.0,
            held_option_positions=held,
            nav=100_000.0,
        )
        assert result.passed is False
        assert result.reason == "sector_cap_breach"
        assert result.details["post_open_sector_pct"] > 0.25
        assert result.details["sector_limit"] == 0.25

    def test_custom_max_sector_pct_overrides_default(self):
        held = [{"symbol": "AAPL", "strike": 180.0, "contracts": 1}]  # $18k
        result = check_sector_cap(
            symbol="AAPL",
            proposed_notional=5_000.0,  # → $23k = 23%
            held_option_positions=held,
            nav=100_000.0,
            max_sector_pct=0.20,  # < 23% triggers breach
        )
        assert result.passed is False
        assert result.reason == "sector_cap_breach"

    def test_unknown_sector_uses_unknown_bucket(self):
        result = check_sector_cap(
            symbol="ZZZZ_NOT_REAL",
            proposed_notional=5_000.0,
            held_option_positions=[],
            nav=100_000.0,
        )
        # Should pass at 5% but record the sector as "Unknown" (the
        # DEFAULT_SECTOR_MAP fallback).
        assert result.passed is True
        assert result.details["sector"] == "Unknown"

    def test_details_post_open_sector_pct_key_is_consistent_across_pass_and_fail(self):
        # S31 F7 regression: previously the pass path emitted
        # `post_open_sector_pct` while the fail path emitted `sector_pct`
        # (same value, different key). Caller-side `.get("post_open_sector_pct", 0)`
        # against a failing result silently defaulted to 0, masking the
        # breach magnitude. Both paths must now emit the same key.
        passing = check_sector_cap(
            symbol="AAPL",
            proposed_notional=5_000.0,
            held_option_positions=[],
            nav=100_000.0,
        )
        failing = check_sector_cap(
            symbol="AAPL",
            proposed_notional=5_000.0,
            held_option_positions=[{"symbol": "AAPL", "strike": 240.0, "contracts": 1}],
            nav=100_000.0,
        )
        assert passing.passed is True
        assert failing.passed is False
        assert "post_open_sector_pct" in passing.details
        assert "post_open_sector_pct" in failing.details
        # Old key must NOT appear in either — the rename is total.
        assert "sector_pct" not in passing.details
        assert "sector_pct" not in failing.details

    def test_custom_sector_map(self):
        custom = {"FAKE": "Crypto"}
        result = check_sector_cap(
            symbol="FAKE",
            proposed_notional=1_000.0,
            held_option_positions=[],
            nav=100_000.0,
            sector_map=custom,
        )
        assert result.passed is True
        assert result.details["sector"] == "Crypto"


# ======================================================================
# 3. check_portfolio_delta — tracker hard-block
# ======================================================================
class TestCheckPortfolioDelta:
    def test_empty_portfolio_passes(self):
        result = check_portfolio_delta(
            held_option_positions=[],
            spot_prices={},
            candidate_option={},
            stock_holdings=[],
            nav=100_000.0,
        )
        assert result.passed is True

    def test_short_put_alone_within_cap(self):
        # Short put on $100 strike. Approximate delta ~ -0.3 for an ATM
        # put. Delta-dollars = -0.3 * 100 * 100 = -3000. Cap at $100k
        # NAV = ±$300. -3000 is way over → breach.
        result = check_portfolio_delta(
            held_option_positions=[],
            spot_prices={"TEST": 100.0},
            candidate_option=_candidate_put("TEST", 100.0),
            stock_holdings=[],
            nav=100_000.0,
        )
        # Short put has positive delta exposure (delta of put is
        # negative, short flips sign → positive). |delta| > cap.
        assert result.passed is False
        assert result.reason == "portfolio_delta_breach"
        assert "portfolio_delta_dollars" not in result.details  # different key on fail
        assert "post_open_delta_dollars" in result.details
        assert "delta_cap_dollars" in result.details

    def test_cap_scales_with_nav(self):
        # Same candidate, NAV 10x larger → cap is $3000, candidate
        # delta-dollars likely still within ±$3000 for a single ATM put.
        result = check_portfolio_delta(
            held_option_positions=[],
            spot_prices={"TEST": 100.0},
            candidate_option=_candidate_put("TEST", 100.0),
            stock_holdings=[],
            nav=1_000_000.0,  # 10x → cap ±$3000
        )
        # At $1M NAV the cap is ±$3000; ATM short put delta is ~$3000;
        # should be borderline. We just assert the cap scaled.
        assert result.details["delta_cap_dollars"] == pytest.approx(3000.0)

    def test_no_candidate_just_checks_held(self):
        # Pass empty candidate dict.
        result = check_portfolio_delta(
            held_option_positions=[],
            spot_prices={"AAPL": 180.0},
            candidate_option={},
            stock_holdings=[("AAPL", 100)],  # 100 shares = $18,000 delta
            nav=100_000.0,
        )
        # Stock delta = $18k; cap = $300. Should breach.
        assert result.passed is False
        assert result.details["current_portfolio_delta_dollars"] == pytest.approx(18000.0)


# ======================================================================
# 4. check_kelly_size — tracker hard-block
# ======================================================================
class TestCheckKellySize:
    def test_classic_winning_edge_passes(self):
        # Per-trade NAV cap formula (post-#163):
        #   kelly_recommended_max = kelly_fraction × NAV
        #                         = 0.5 × $100k = $50,000.
        # Margin $5k well below cap → pass. The win_rate / avg_win /
        # avg_loss inputs are forward-compat placeholders the current
        # formula ignores (their previous role in this test described
        # the binary-Kelly form that was rewritten in PR #163).
        result = check_kelly_size(
            margin_required=5_000.0,
            win_rate=0.7,
            avg_win=100.0,
            avg_loss=50.0,
            nav=100_000.0,
        )
        assert result.passed is True
        assert result.details["margin_required"] == 5_000.0
        assert result.details["kelly_fraction"] == 0.5
        # Pin the cap value, not just a positivity check — the previous
        # `> 0` assertion would have passed under the binary-Kelly form
        # too, masking the formula change.
        assert result.details["kelly_recommended_max"] == 50_000.0

    def test_margin_above_cap_refuses(self):
        # Half-Kelly cap = 0.5 × $100k = $50k. Margin $60k > cap.
        result = check_kelly_size(
            margin_required=60_000.0,
            win_rate=0.7,
            avg_win=100.0,
            avg_loss=50.0,
            nav=100_000.0,
        )
        assert result.passed is False
        assert result.reason == "kelly_size_exceeded"
        assert result.details["margin_required"] == 60_000.0
        assert result.details["kelly_recommended_max"] == 50_000.0

    def test_margin_at_cap_passes(self):
        # Boundary: margin exactly at the cap should pass.
        result = check_kelly_size(
            margin_required=50_000.0,
            win_rate=0.7,
            avg_win=100.0,
            avg_loss=50.0,
            nav=100_000.0,
        )
        assert result.passed is True
        assert result.details["kelly_recommended_max"] == 50_000.0

    def test_zero_nav_refuses_any_positive_margin(self):
        # Per-trade cap = 0.5 × 0 = 0; any positive margin refuses.
        result = check_kelly_size(
            margin_required=1.0,
            win_rate=0.7,
            avg_win=100.0,
            avg_loss=50.0,
            nav=0.0,
        )
        assert result.passed is False
        assert result.details["kelly_recommended_max"] == 0.0

    def test_win_rate_unused_today(self):
        """Forward-compat: win_rate is in the signature but the
        cap-based formula doesn't consume it. Out-of-range values
        therefore do NOT cause a refuse; the cap-vs-margin check is
        what matters. A future continuous-Kelly refinement may use
        the value."""
        result = check_kelly_size(
            margin_required=1.0,
            win_rate=1.5,  # nonsensical but not consumed
            avg_win=100.0,
            avg_loss=50.0,
            nav=100_000.0,
        )
        # $1 << $50k cap → passes regardless of win_rate.
        assert result.passed is True

    def test_custom_kelly_fraction(self):
        # Full-Kelly (kelly_fraction=1.0) gives a larger cap than
        # half-Kelly (0.5), for the same NAV.
        half = check_kelly_size(
            margin_required=60_000.0,
            win_rate=0.7,
            avg_win=100.0,
            avg_loss=50.0,
            nav=100_000.0,
            kelly_fraction=0.5,
        )
        full = check_kelly_size(
            margin_required=60_000.0,
            win_rate=0.7,
            avg_win=100.0,
            avg_loss=50.0,
            nav=100_000.0,
            kelly_fraction=1.0,
        )
        # Half-Kelly cap = 50k → refuses 60k. Full-Kelly cap = 100k → passes.
        assert half.passed is False
        assert full.passed is True
        assert full.details["kelly_recommended_max"] == 100_000.0


# ======================================================================
# 5. check_var — dossier soft-warn (R7)
# ======================================================================
class TestCheckVar:
    def test_missing_data_skips_not_refuses(self):
        """Q3 D11 anti-pattern: don't silently fall through to a
        parametric VaR when correlation or returns data missing."""
        result = check_var(
            held_option_positions=[],
            spot_prices={},
            candidate_option=_candidate_put(),
            nav=100_000.0,
            returns_data=None,
            correlation_matrix=None,
        )
        assert result.passed is True
        assert result.reason == "missing_data"
        assert result.details["var_check"] == "skipped"
        assert "skip_reason" in result.details

    def test_with_returns_data_computes_var(self):
        """If returns_data is provided, the gate computes VaR via the
        historical path (or covariance if matrix given)."""
        import numpy as np
        import pandas as pd

        # Synthetic returns: low vol → low VaR → should pass.
        idx = pd.date_range("2026-01-01", periods=60, freq="B")
        returns = pd.DataFrame(
            {"portfolio": np.random.default_rng(42).normal(0, 0.005, 60)},
            index=idx,
        )
        result = check_var(
            held_option_positions=[],
            spot_prices={"TEST": 100.0},
            candidate_option=_candidate_put("TEST", 100.0),
            nav=100_000.0,
            returns_data=returns,
        )
        assert result.reason != "missing_data"
        # Computed (passed or failed depending on VaR; just verify the
        # details bag was populated with VaR fields).
        assert "var_dollars" in result.details
        assert "var_pct" in result.details
        assert result.details["confidence"] == 0.95
        assert result.details["horizon_days"] == 30


# ======================================================================
# 6. check_stress_scenario — dossier soft-warn (R8 trigger 1)
# ======================================================================
class TestCheckStressScenario:
    def test_empty_portfolio_passes(self):
        result = check_stress_scenario(
            held_option_positions=[],
            spot_prices={},
            candidate_option={},
            nav=100_000.0,
        )
        assert result.passed is True
        assert result.details["note"] == "no_positions_to_stress"

    def test_default_scenario_is_c4_vol_spike(self):
        result = check_stress_scenario(
            held_option_positions=[],
            spot_prices={"TEST": 100.0},
            candidate_option=_candidate_put("TEST", 100.0),
            nav=100_000.0,
        )
        assert result.details["scenario_name"] == "C4 Vol Spike"

    def test_custom_scenario_override(self):
        from engine.stress_testing import Scenario, ScenarioType

        custom = Scenario(
            name="Mild Drop",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="-2% spot only",
            spot_change_pct=-0.02,
        )
        result = check_stress_scenario(
            held_option_positions=[],
            spot_prices={"TEST": 100.0},
            candidate_option=_candidate_put("TEST", 100.0),
            nav=100_000.0,
            scenario=custom,
        )
        assert result.details["scenario_name"] == "Mild Drop"

    def test_drawdown_threshold_breach_fails(self):
        # Large short put on $100 strike at small NAV → C4 vol spike
        # produces a big loss relative to NAV.
        result = check_stress_scenario(
            held_option_positions=[
                {
                    "symbol": "TEST",
                    "option_type": "put",
                    "strike": 100.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
            ],
            spot_prices={"TEST": 100.0},
            candidate_option={},
            nav=5_000.0,  # tiny NAV → drawdown % large
        )
        # 10% spot drop + 30% IV spike on a short put will mark to
        # market badly; at $5k NAV it crosses the 8% drawdown cap.
        assert result.passed is False
        assert result.reason == "stress_breach"
        assert result.details["drawdown_pct"] > 0.08


# ======================================================================
# 7. check_dealer_regime — dossier soft-warn (R8 trigger 2)
# ======================================================================
class TestCheckDealerRegime:
    def test_no_regime_data_skips(self):
        result = check_dealer_regime("AAPL", None)
        assert result.passed is True
        assert result.reason == "missing_data"

    def test_ticker_not_in_map_skips(self):
        result = check_dealer_regime("AAPL", {"MSFT": "neutral"})
        assert result.passed is True
        assert result.reason == "missing_data"

    def test_short_gamma_amplifying_fires(self):
        result = check_dealer_regime("AAPL", {"AAPL": "short_gamma_amplifying"})
        assert result.passed is False
        assert result.reason == "short_gamma_regime"
        assert result.details["dealer_regime"] == "short_gamma_amplifying"

    def test_long_gamma_passes(self):
        result = check_dealer_regime("AAPL", {"AAPL": "long_gamma_dampening"})
        assert result.passed is True
        assert result.reason is None

    def test_neutral_passes(self):
        result = check_dealer_regime("AAPL", {"AAPL": "neutral"})
        assert result.passed is True

    def test_near_flip_passes_but_logs(self):
        """near_flip is not the short-gamma trigger — only
        short_gamma_amplifying fires R8."""
        result = check_dealer_regime("AAPL", {"AAPL": "near_flip"})
        assert result.passed is True
        assert result.details["dealer_regime"] == "near_flip"


# ======================================================================
# 8. GateResult shape — pinned for Phase 2's audit-log schema integration
# ======================================================================
class TestGateResultShape:
    def test_pass_shape(self):
        r = GateResult(passed=True)
        assert r.passed is True
        assert r.reason is None
        assert r.details == {}

    def test_fail_shape(self):
        r = GateResult(passed=False, reason="sector_cap_breach", details={"sector": "Tech"})
        assert r.passed is False
        assert r.reason == "sector_cap_breach"
        assert r.details == {"sector": "Tech"}

    def test_skip_shape(self):
        r = GateResult(passed=True, reason="missing_data", details={"skip_reason": "test"})
        assert r.passed is True
        assert r.reason == "missing_data"


# ======================================================================
# 9. C4 vol-spike scenario sanity
# ======================================================================
class TestC4VolSpikeScenario:
    def test_scenario_matches_d17_spec(self):
        assert _C4_VOL_SPIKE_SCENARIO.spot_change_pct == -0.10
        assert _C4_VOL_SPIKE_SCENARIO.iv_change_pct == 0.30
        assert _C4_VOL_SPIKE_SCENARIO.iv_change_abs == 0.0
        assert _C4_VOL_SPIKE_SCENARIO.name == "C4 Vol Spike"
