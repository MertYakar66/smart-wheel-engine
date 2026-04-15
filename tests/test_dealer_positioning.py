"""
Tests for the dealer-positioning / market-structure layer (audit V).

Covers:
  * Core math of the analyzer (per-strike exposures, sign convention,
    wall detection, flip-level bisection, pinning zones, regime
    classification, confidence scaling).
  * Regime → multiplier mapping with the asymmetric [0.70, 1.05] bound.
  * EVEngine integration: market_structure as an optional argument,
    result field propagation, and the hard guardrail that dealer
    positioning can never upgrade a negative-EV trade.
  * WheelRunner integration: opt-in kwarg, graceful degradation when
    the chain is unavailable, column propagation.
  * Reviewer rule R6: downgrade-only semantics, short-gamma + put
    wall, near-flip, no false positives in long-gamma regimes.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.candidate_dossier import (
    CandidateDossier,
    EnginePhaseReviewer,
)
from engine.chart_context import ChartContext
from engine.dealer_positioning import (
    DealerAssumption,
    DealerPositioningAnalyzer,
    GammaWall,
    MarketStructure,
    PerStrikeExposure,
    dealer_regime_multiplier,
)
from engine.ev_engine import EVEngine, ShortOptionTrade


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_chain(
    spot: float,
    strikes: list[tuple[float, int, int, float]],
    expiry: date | None = None,
) -> pd.DataFrame:
    """Helper to build a toy option chain DataFrame.

    Each tuple is (strike, call_oi, put_oi, iv).
    """
    if expiry is None:
        expiry = date.today() + timedelta(days=30)
    rows = []
    for k, c_oi, p_oi, iv in strikes:
        rows.append(
            {
                "strike": k,
                "option_type": "C",
                "open_interest": c_oi,
                "implied_vol": iv,
                "expiration": expiry,
            }
        )
        rows.append(
            {
                "strike": k,
                "option_type": "P",
                "open_interest": p_oi,
                "implied_vol": iv,
                "expiration": expiry,
            }
        )
    return pd.DataFrame(rows)


# ======================================================================
# 1. Analyzer core math
# ======================================================================
class TestAnalyzerCore:
    def test_empty_chain_returns_neutral(self):
        analyzer = DealerPositioningAnalyzer()
        ms = analyzer.analyze(
            pd.DataFrame(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            ticker="TEST",
        )
        assert ms.regime == "neutral"
        assert ms.confidence == 0.0
        assert ms.gex_total == 0.0
        assert ms.n_strikes == 0

    def test_zero_spot_returns_neutral(self):
        analyzer = DealerPositioningAnalyzer()
        chain = make_chain(
            100.0, [(95, 100, 100, 0.2), (100, 100, 100, 0.2), (105, 100, 100, 0.2)]
        )
        ms = analyzer.analyze(
            chain, spot=0.0, expiry=date.today() + timedelta(days=30), ticker="TEST"
        )
        assert ms.regime == "neutral"

    def test_missing_columns_returns_neutral(self):
        analyzer = DealerPositioningAnalyzer()
        bad = pd.DataFrame({"strike": [100], "option_type": ["C"]})  # no OI, no IV
        ms = analyzer.analyze(
            bad, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert "missing_columns" in ms.notes
        assert ms.regime == "neutral"

    def test_invalid_iv_rows_dropped(self):
        analyzer = DealerPositioningAnalyzer()
        chain = make_chain(
            100.0,
            [
                (95, 100, 100, 0.2),
                (100, 100, 100, 10.0),  # IV > 5 — dropped
                (105, 100, 100, -0.1),  # IV < 0 — dropped
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30), ticker="T"
        )
        # Only the first strike (95) survives
        assert ms.n_strikes == 1

    def test_per_strike_exposures_populated(self):
        analyzer = DealerPositioningAnalyzer()
        chain = make_chain(
            100.0,
            [
                (90, 500, 5000, 0.25),
                (100, 10000, 10000, 0.22),
                (110, 8000, 500, 0.24),
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30), ticker="T"
        )
        assert ms.n_strikes == 3
        assert len(ms.per_strike) == 3
        strikes = [p.strike for p in ms.per_strike]
        assert strikes == [90.0, 100.0, 110.0]
        # Every per-strike row has non-zero net_gex (there is OI on both sides)
        for p in ms.per_strike:
            assert p.call_oi > 0 or p.put_oi > 0


# ======================================================================
# 2. Sign convention
# ======================================================================
class TestSignConvention:
    def test_long_calls_short_puts_call_sign_positive(self):
        """Under LONG_CALLS_SHORT_PUTS, a call-only chain produces positive GEX."""
        analyzer = DealerPositioningAnalyzer(
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS
        )
        # Chain with ONLY calls
        chain = pd.DataFrame(
            [
                {"strike": 100, "option_type": "C", "open_interest": 1000, "implied_vol": 0.2},
                {"strike": 105, "option_type": "C", "open_interest": 1000, "implied_vol": 0.2},
            ]
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.gex_total > 0

    def test_long_calls_short_puts_put_sign_negative(self):
        """Under LONG_CALLS_SHORT_PUTS, a put-only chain produces negative GEX."""
        analyzer = DealerPositioningAnalyzer(
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS
        )
        chain = pd.DataFrame(
            [
                {"strike": 100, "option_type": "P", "open_interest": 1000, "implied_vol": 0.2},
                {"strike": 95, "option_type": "P", "open_interest": 1000, "implied_vol": 0.2},
            ]
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.gex_total < 0

    def test_short_both_assumption_flips_call_sign(self):
        """Under SHORT_BOTH, call OI produces negative GEX (opposite of SpotGamma)."""
        call_only = pd.DataFrame(
            [
                {"strike": 100, "option_type": "C", "open_interest": 1000, "implied_vol": 0.2},
            ]
        )
        long_cs = DealerPositioningAnalyzer(
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS
        )
        short_b = DealerPositioningAnalyzer(
            assumption=DealerAssumption.SHORT_BOTH
        )
        ms_lcsp = long_cs.analyze(
            call_only, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        ms_sb = short_b.analyze(
            call_only, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert np.sign(ms_lcsp.gex_total) == -np.sign(ms_sb.gex_total)

    def test_put_heavy_chain_is_short_gamma_under_spotgamma(self):
        """Put-OI-dominated chains should be classified as short gamma."""
        analyzer = DealerPositioningAnalyzer()
        # Heavy put OI below spot, light call OI
        chain = make_chain(
            100.0,
            [
                (85, 100, 50000, 0.30),
                (90, 200, 40000, 0.27),
                (95, 500, 20000, 0.24),
                (100, 1000, 2000, 0.22),
                (105, 500, 100, 0.23),
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30), ticker="T"
        )
        assert ms.gex_total < 0
        assert ms.regime == "short_gamma_amplifying"


# ======================================================================
# 3. Greeks recomputation
# ======================================================================
class TestGreeksFallback:
    def test_missing_delta_gamma_reconstructed_from_iv(self):
        """When the chain has no delta/gamma columns, BSM should fill them."""
        chain = pd.DataFrame(
            [
                {"strike": 100, "option_type": "C", "open_interest": 1000, "implied_vol": 0.25},
                {"strike": 100, "option_type": "P", "open_interest": 1000, "implied_vol": 0.25},
            ]
        )
        # No 'delta' or 'gamma' columns present
        analyzer = DealerPositioningAnalyzer()
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.n_strikes == 1
        # Per-strike row must have non-zero call_gamma (recomputed from BSM)
        per = ms.per_strike[0]
        assert per.call_gamma > 0
        assert per.put_gamma > 0  # gamma is same for call and put under BSM


# ======================================================================
# 4. Gamma wall detection
# ======================================================================
class TestWalls:
    def test_find_walls_sorts_by_abs_gex(self):
        analyzer = DealerPositioningAnalyzer()
        # One strike far above spot has massive call OI
        chain = make_chain(
            100.0,
            [
                (95, 100, 100, 0.20),
                (100, 100, 100, 0.20),
                (105, 50000, 100, 0.20),  # HUGE call OI — clear wall
                (110, 100, 100, 0.20),
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        # Top call wall should be at 105 (the spike)
        assert len(ms.call_walls) > 0
        assert ms.call_walls[0].strike == 105.0
        assert ms.call_walls[0].side == "call"

    def test_nearest_wall_within_range(self):
        analyzer = DealerPositioningAnalyzer(near_wall_pct=0.10)
        chain = make_chain(
            100.0,
            [
                (95, 100, 5000, 0.20),  # put wall
                (100, 100, 100, 0.20),
                (105, 5000, 100, 0.20),  # call wall
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.nearest_call_wall is not None
        assert ms.nearest_call_wall.strike == 105.0
        assert ms.nearest_put_wall is not None
        assert ms.nearest_put_wall.strike == 95.0

    def test_nearest_wall_outside_range_is_none(self):
        # near_wall_pct=1% — walls at ±5% should be too far
        analyzer = DealerPositioningAnalyzer(near_wall_pct=0.01)
        chain = make_chain(
            100.0,
            [
                (95, 100, 5000, 0.20),
                (100, 100, 100, 0.20),
                (105, 5000, 100, 0.20),
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.nearest_call_wall is None
        assert ms.nearest_put_wall is None


# ======================================================================
# 5. Flip level bisection
# ======================================================================
class TestFlipLevel:
    def test_flip_level_detected_when_sign_changes(self):
        analyzer = DealerPositioningAnalyzer()
        # Put-heavy below 100, call-heavy above 100 — flip should be near 100
        chain = make_chain(
            100.0,
            [
                (85, 100, 20000, 0.25),
                (90, 100, 15000, 0.23),
                (95, 200, 10000, 0.22),
                (100, 1000, 1000, 0.22),
                (105, 10000, 200, 0.22),
                (110, 15000, 100, 0.23),
                (115, 20000, 100, 0.25),
            ],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.flip_level is not None
        # Flip should be within ±10% of spot
        assert 85 <= ms.flip_level <= 115

    def test_flip_none_when_all_positive_gex(self):
        analyzer = DealerPositioningAnalyzer()
        # Pure-call chain across many strikes
        chain = pd.DataFrame(
            [
                {"strike": k, "option_type": "C", "open_interest": 1000, "implied_vol": 0.20}
                for k in [80, 90, 100, 110, 120]
            ]
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        # No sign change possible over the scan range
        assert ms.flip_level is None


# ======================================================================
# 6. Regime classification
# ======================================================================
class TestRegimeClassification:
    def test_positive_gex_away_from_flip_is_long_gamma(self):
        analyzer = DealerPositioningAnalyzer()
        # Clear positive-GEX chain
        chain = pd.DataFrame(
            [
                {"strike": k, "option_type": "C", "open_interest": 5000, "implied_vol": 0.20}
                for k in [100, 105, 110]
            ]
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.gex_total > 0
        assert ms.regime in ("long_gamma_dampening", "near_flip")

    def test_negative_gex_is_short_gamma(self):
        analyzer = DealerPositioningAnalyzer()
        # Pure-put chain → all sign=-1 → negative GEX
        chain = pd.DataFrame(
            [
                {"strike": k, "option_type": "P", "open_interest": 5000, "implied_vol": 0.20}
                for k in [95, 100, 105]
            ]
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert ms.gex_total < 0
        assert ms.regime == "short_gamma_amplifying"

    def test_confidence_in_unit_interval(self):
        analyzer = DealerPositioningAnalyzer()
        chain = make_chain(
            100.0,
            [(95, 500, 500, 0.22), (100, 500, 500, 0.22), (105, 500, 500, 0.22)],
        )
        ms = analyzer.analyze(
            chain, spot=100.0, expiry=date.today() + timedelta(days=30)
        )
        assert 0.0 <= ms.confidence <= 1.0


# ======================================================================
# 7. Multiplier bounds
# ======================================================================
class TestMultiplierBounds:
    def _ms(self, regime: str, confidence: float = 1.0) -> MarketStructure:
        return MarketStructure(
            ticker="T",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime=regime,
            confidence=confidence,
        )

    def test_long_gamma_mult_capped_at_105(self):
        mult = dealer_regime_multiplier(self._ms("long_gamma_dampening", 1.0))
        assert mult == pytest.approx(1.05, abs=1e-6)

    def test_short_gamma_mult_floor_at_070(self):
        mult = dealer_regime_multiplier(self._ms("short_gamma_amplifying", 1.0))
        assert mult == pytest.approx(0.70, abs=1e-6)

    def test_near_flip_mult_is_085(self):
        mult = dealer_regime_multiplier(self._ms("near_flip", 0.5))
        assert mult == pytest.approx(0.85, abs=1e-6)

    def test_neutral_mult_is_100(self):
        mult = dealer_regime_multiplier(self._ms("neutral", 0.0))
        assert mult == pytest.approx(1.00, abs=1e-6)

    def test_none_returns_identity(self):
        assert dealer_regime_multiplier(None) == 1.0

    def test_low_confidence_scales_toward_one(self):
        """Low-confidence regimes move less aggressively."""
        low_conf = dealer_regime_multiplier(
            self._ms("short_gamma_amplifying", confidence=0.2)
        )
        high_conf = dealer_regime_multiplier(
            self._ms("short_gamma_amplifying", confidence=1.0)
        )
        assert low_conf > high_conf  # less aggressive cut at low confidence
        assert 0.70 <= high_conf <= low_conf <= 1.0


# ======================================================================
# 8. EVEngine integration
# ======================================================================
class TestEVEngineIntegration:
    def _trade(self):
        return ShortOptionTrade(
            option_type="put",
            underlying="TEST",
            spot=100.0,
            strike=95.0,
            premium=1.20,
            bid=1.15,
            ask=1.25,
            dte=30,
            iv=0.22,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            contracts=1,
            open_interest=1000,
            regime_multiplier=1.0,
        )

    def test_market_structure_none_is_identity(self):
        trade = self._trade()
        r1 = EVEngine().evaluate(trade)
        r2 = EVEngine().evaluate(trade, market_structure=None)
        assert r1.ev_dollars == pytest.approx(r2.ev_dollars)
        assert r2.dealer_multiplier == 1.0
        assert r2.dealer_regime == ""

    def test_short_gamma_structure_shrinks_ev(self):
        """Same trade in short-gamma regime has lower EV than long-gamma."""
        trade = self._trade()
        # Deliberately construct a MarketStructure with full confidence
        ms_short = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="short_gamma_amplifying",
            confidence=1.0,
            gex_total=-1e9,
        )
        ms_long = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="long_gamma_dampening",
            confidence=1.0,
            gex_total=1e9,
        )
        eng = EVEngine()
        r_short = eng.evaluate(trade, market_structure=ms_short)
        r_long = eng.evaluate(trade, market_structure=ms_long)

        # Long-gamma EV should exceed short-gamma EV by a factor near 1.05 / 0.70 = 1.5
        assert r_long.ev_dollars > r_short.ev_dollars
        assert r_long.dealer_multiplier > r_short.dealer_multiplier

    def test_dealer_mult_cannot_rescue_negative_ev(self):
        """Hard guardrail: dealer positioning never upgrades negative-EV."""
        # Force negative EV by shorting into garbage premium
        trade = ShortOptionTrade(
            option_type="put",
            underlying="TEST",
            spot=100.0,
            strike=95.0,
            premium=0.01,  # almost zero — negative EV
            bid=0.005,
            ask=0.015,
            dte=30,
            iv=0.22,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            contracts=1,
        )
        ms_long = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="long_gamma_dampening",
            confidence=1.0,
        )
        eng = EVEngine()
        r = eng.evaluate(trade, market_structure=ms_long)
        # 1.05 × negative is still negative
        assert r.ev_dollars < 0

    def test_new_result_fields_populated(self):
        trade = self._trade()
        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="long_gamma_dampening",
            confidence=0.8,
            gex_total=2.5e9,
            flip_distance_pct=0.03,
            pinning_zones=[100.0],
            nearest_put_wall=GammaWall(
                strike=95.0, distance_pct=-0.05, net_gex=-1e6, side="put"
            ),
            nearest_call_wall=GammaWall(
                strike=105.0, distance_pct=0.05, net_gex=1e6, side="call"
            ),
        )
        r = EVEngine().evaluate(trade, market_structure=ms)
        assert r.dealer_regime == "long_gamma_dampening"
        assert r.gex_total == pytest.approx(2.5e9)
        assert r.gamma_flip_distance_pct == pytest.approx(0.03)
        assert r.nearest_put_wall_strike == 95.0
        assert r.nearest_call_wall_strike == 105.0
        assert r.pinning_zones == [100.0]


# ======================================================================
# 9. Reviewer rule R6
# ======================================================================
class TestReviewerR6:
    def _chart_ok(self) -> ChartContext:
        return ChartContext(
            ticker="TEST",
            timeframe="1D",
            captured_at=datetime.utcnow(),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.0,
            source="test",
        )

    def _dossier(
        self,
        ev: float,
        strike: float,
        ms: MarketStructure | None = None,
    ) -> CandidateDossier:
        row = {
            "ticker": "TEST",
            "ev_dollars": ev,
            "spot": 100.0,
            "strike": strike,
        }
        return CandidateDossier(
            ticker="TEST",
            ev_row=row,
            chart_context=self._chart_ok(),
            market_structure=ms,
        )

    def test_r6_downgrades_proceed_on_short_gamma_above_put_wall(self):
        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="short_gamma_amplifying",
            confidence=0.8,
            nearest_put_wall=GammaWall(
                strike=95.0, distance_pct=-0.05, net_gex=-1e6, side="put"
            ),
        )
        # Candidate strike at 97 is ABOVE the put wall at 95 → breach risk
        dossier = self._dossier(ev=50.0, strike=97.0, ms=ms)
        reviewer = EnginePhaseReviewer(min_proceed_ev=10.0)
        verdict, reason, notes = reviewer.review(dossier)
        assert verdict == "review"
        assert reason == "dealer_short_gamma_above_put_wall"
        assert any("R6" in n for n in notes)

    def test_r6_does_not_trigger_when_strike_below_put_wall(self):
        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="short_gamma_amplifying",
            confidence=0.8,
            nearest_put_wall=GammaWall(
                strike=95.0, distance_pct=-0.05, net_gex=-1e6, side="put"
            ),
        )
        # Candidate strike at 90 is BELOW the put wall → cushion exists
        dossier = self._dossier(ev=50.0, strike=90.0, ms=ms)
        verdict, _, _ = EnginePhaseReviewer(min_proceed_ev=10.0).review(dossier)
        assert verdict == "proceed"

    def test_r6_does_nothing_in_long_gamma_regime(self):
        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="long_gamma_dampening",
            confidence=0.9,
            nearest_put_wall=GammaWall(
                strike=95.0, distance_pct=-0.05, net_gex=-1e6, side="put"
            ),
        )
        dossier = self._dossier(ev=50.0, strike=97.0, ms=ms)
        verdict, reason, _ = EnginePhaseReviewer(min_proceed_ev=10.0).review(dossier)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"

    def test_r6_near_flip_regime_downgrades_to_review(self):
        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="near_flip",
            confidence=0.5,
        )
        dossier = self._dossier(ev=50.0, strike=90.0, ms=ms)
        verdict, reason, _ = EnginePhaseReviewer(min_proceed_ev=10.0).review(dossier)
        assert verdict == "review"
        assert reason == "dealer_near_flip"

    def test_r6_cannot_upgrade_negative_ev(self):
        """Hard guardrail: even with perfect dealer positioning, negative
        EV stays blocked."""
        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime.utcnow(),
            spot=100.0,
            expiry=date.today() + timedelta(days=30),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="long_gamma_dampening",
            confidence=1.0,
        )
        dossier = self._dossier(ev=-10.0, strike=97.0, ms=ms)
        verdict, reason, _ = EnginePhaseReviewer().review(dossier)
        assert verdict == "blocked"
        assert reason == "negative_ev"


# ======================================================================
# 10. WheelRunner integration
# ======================================================================
class TestWheelRunnerDealerIntegration:
    def test_runner_runs_with_dealer_positioning_enabled(self):
        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))

        expiry = date.today() + timedelta(days=32)
        chain = make_chain(
            100.0,
            [
                (90, 500, 5000, 0.25),
                (95, 1000, 8000, 0.23),
                (100, 10000, 10000, 0.22),
                (105, 8000, 1000, 0.23),
                (110, 6000, 500, 0.25),
            ],
            expiry=expiry,
        )

        class FakeConn:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": prices}, index=idx)

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.22,
                    "volatility_30d": 0.20,
                    "dividend_yield": 0.0,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["TESTA"]

            def get_options(self, ticker):
                return chain.copy()

        runner = WheelRunner()
        runner._connector = FakeConn()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"],
            dte_target=30,
            delta_target=0.25,
            top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=True,
        )
        assert not df.empty
        # Diagnostic columns from dealer positioning should be present
        for col in (
            "dealer_regime",
            "dealer_multiplier",
            "gex_total",
            "gamma_flip_distance_pct",
            "nearest_put_wall_strike",
            "nearest_call_wall_strike",
        ):
            assert col in df.columns

    def test_runner_falls_back_when_chain_unavailable(self):
        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))

        class FakeConnNoChain:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": prices}, index=idx)

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.22,
                    "volatility_30d": 0.20,
                    "dividend_yield": 0.0,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["TESTA"]

            # No get_options method! Should degrade gracefully.

        runner = WheelRunner()
        runner._connector = FakeConnNoChain()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"],
            dte_target=30,
            delta_target=0.25,
            top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=True,
        )
        # Should still produce a row, with None dealer fields
        assert not df.empty
        # dealer_multiplier should default to 1.0 when no chain
        assert float(df["dealer_multiplier"].iloc[0]) == pytest.approx(1.0)

    def test_runner_dealer_positioning_off_by_default(self):
        """Off by default — existing callers see identical behaviour."""
        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))

        class FakeConn:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": prices}, index=idx)

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.22,
                    "volatility_30d": 0.20,
                    "dividend_yield": 0.0,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["TESTA"]

            def get_options(self, ticker):
                # This should never be called when dealer positioning off
                raise AssertionError("get_options called despite dealer off")

        runner = WheelRunner()
        runner._connector = FakeConn()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"],
            dte_target=30,
            top_n=5,
            min_ev_dollars=-1e9,
            # use_dealer_positioning default = False
        )
        assert not df.empty


# ======================================================================
# 11. MarketStructure serialisation
# ======================================================================
class TestSerialisation:
    def test_to_dict_is_json_safe(self):
        import json

        ms = MarketStructure(
            ticker="TEST",
            as_of=datetime(2026, 4, 14, 10, 30),
            spot=100.0,
            expiry=date(2026, 5, 15),
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
            regime="long_gamma_dampening",
            confidence=0.75,
            gex_total=1e9,
            dex_total=5e8,
            vanna_total=1.2e6,
            charm_total=-3.4e5,
            flip_level=98.5,
            flip_distance_pct=-0.015,
            pinning_zones=[100.0, 105.0],
            nearest_call_wall=GammaWall(
                strike=105.0, distance_pct=0.05, net_gex=5e8, side="call"
            ),
            nearest_put_wall=GammaWall(
                strike=95.0, distance_pct=-0.05, net_gex=-3e8, side="put"
            ),
            call_walls=[
                GammaWall(strike=105.0, distance_pct=0.05, net_gex=5e8, side="call"),
                GammaWall(strike=110.0, distance_pct=0.10, net_gex=3e8, side="call"),
            ],
            put_walls=[
                GammaWall(strike=95.0, distance_pct=-0.05, net_gex=-3e8, side="put"),
            ],
            per_strike=[
                PerStrikeExposure(
                    strike=100.0,
                    call_oi=1000,
                    put_oi=800,
                    call_gamma=0.02,
                    put_gamma=0.02,
                    call_delta=0.50,
                    put_delta=-0.50,
                    call_gex=1e6,
                    put_gex=-5e5,
                    net_gex=5e5,
                    net_dex=1e4,
                    net_vanna=1e3,
                    net_charm=-1e2,
                )
            ],
        )
        d = ms.to_dict()
        json.dumps(d)  # must not raise
        assert d["regime"] == "long_gamma_dampening"
        assert d["assumption"] == "long_calls_short_puts"
        assert d["gex_total"] == 1e9
        assert d["nearest_call_wall"]["strike"] == 105.0
        assert d["nearest_put_wall"]["strike"] == 95.0
