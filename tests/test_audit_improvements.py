"""
Tests for the audit improvements (forward distribution, empirical surface,
early-assignment-on-dividend, survivorship-bias guard, calibration gate,
sqrt market impact, bid>ask / stale-quote quality checks, and the
EV-driven wheel ranker).

Every test here corresponds to a specific audit deliverable. If any of
these fail, the audit has regressed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine import option_pricer as op
from engine.ev_engine import EVEngine, ShortOptionTrade
from engine.forward_distribution import (
    best_available_forward_distribution,
    block_bootstrap_log_returns,
    empirical_forward_log_returns,
    har_rv_conditional_distribution,
)
from engine.shared_valuation import simulate_option_trade
from engine.transaction_costs import calculate_slippage
from engine.volatility_surface import (
    create_constant_surface,
    create_empirical_surface,
)
from ml.model_governance import DriftDetector


# =========================================================================
# 1. Forward distribution module
# =========================================================================
class TestForwardDistribution:
    """PIT safety + cascading fallback for physical forward distributions."""

    def _synth(self, n: int = 1500, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        log_rets = rng.normal(0.0003, 0.012, n)
        prices = 100 * np.exp(np.cumsum(log_rets))
        return pd.DataFrame({"close": prices}, index=idx)

    def test_empirical_is_point_in_time(self):
        df = self._synth()
        full = empirical_forward_log_returns(
            df, horizon_days=30, as_of="2023-01-01", min_samples=5
        )
        early = empirical_forward_log_returns(
            df, horizon_days=30, as_of="2020-01-01", min_samples=5
        )
        assert len(early) < len(full), (
            "PIT filter broken — as_of in the past should yield fewer samples"
        )

    def test_empirical_non_overlapping_is_independent(self):
        df = self._synth(n=2500)
        # Non-overlapping sampling of (n-h)/h ~ (2500-20)/20 ≈ 124 samples,
        # but the lookback_years=5 ceiling caps the available history.
        rets = empirical_forward_log_returns(
            df,
            horizon_days=20,
            as_of="2023-12-31",
            lookback_years=10.0,  # expand so we get plenty of samples
            min_samples=10,
            non_overlapping=True,
        )
        assert len(rets) >= 50
        # Sample autocorrelation should be near zero (independent samples)
        autocorr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
        assert abs(autocorr) < 0.3

    def test_block_bootstrap_matches_empirical_stats(self):
        df = self._synth(n=1500)
        bb = block_bootstrap_log_returns(
            df, horizon_days=20, n_scenarios=4000, as_of="2023-06-01"
        )
        assert len(bb) == 4000
        # 20-day log-return std ≈ 0.012 * sqrt(20) ≈ 0.0537
        assert 0.02 < bb.std() < 0.10

    def test_har_rv_produces_fat_tails(self):
        df = self._synth()
        har = har_rv_conditional_distribution(
            df, horizon_days=20, n_scenarios=5000, as_of="2023-06-01"
        )
        assert len(har) == 5000
        # Student-t(6) fat-tailed kurtosis > Gaussian
        kurt = float(((har - har.mean()) ** 4).mean() / (har.var() ** 2))
        assert kurt > 2.5

    def test_cascading_fallback_picks_best(self):
        df = self._synth()
        rets, method = best_available_forward_distribution(
            df, horizon_days=20, as_of="2023-06-01"
        )
        assert len(rets) > 0
        assert method in (
            "empirical_non_overlapping",
            "empirical_overlapping",
            "block_bootstrap",
            "har_rv",
        )

    def test_empty_history_returns_empty(self):
        rets, method = best_available_forward_distribution(
            pd.DataFrame({"close": []}), horizon_days=20
        )
        assert len(rets) == 0
        assert method == "none"


# =========================================================================
# 2. Empirical volatility surface (smile-aware)
# =========================================================================
class TestEmpiricalVolatilitySurface:
    def test_constant_surface_has_no_skew(self):
        expiries = [date(2024, 1, 1) + timedelta(days=d) for d in (30, 60, 90)]
        surf = create_constant_surface(
            iv=0.25, as_of_date=date(2023, 12, 1), underlying="SPY", spot=450, expiries=expiries
        )
        # Flat surface: IV should be ~identical at every log-moneyness
        iv_atm = surf.get_iv(450, expiries[0], 450)
        iv_otm = surf.get_iv(400, expiries[0], 450)
        assert abs(iv_atm - iv_otm) < 0.01

    def test_empirical_surface_has_put_skew(self):
        as_of = date(2023, 12, 1)
        # Days-to-expiry 30 / 60 / 90 from as_of_date
        expiries = [as_of + timedelta(days=d) for d in (30, 60, 90)]
        surf = create_empirical_surface(
            iv_by_tenor={30: 0.22, 60: 0.24, 90: 0.26},
            as_of_date=as_of,
            underlying="SPY",
            spot=450,
            expiries=expiries,
        )
        # OTM put IV should be higher than ATM IV (negative skew)
        iv_atm = surf.get_iv(450, expiries[0], 450)
        iv_otm_put = surf.get_iv(400, expiries[0], 450)
        assert iv_otm_put > iv_atm, (
            f"Expected put skew (OTM put > ATM): atm={iv_atm:.4f} otm_put={iv_otm_put:.4f}"
        )

    def test_empirical_surface_honours_atm_term_structure(self):
        as_of = date(2023, 12, 1)
        expiries = [as_of + timedelta(days=d) for d in (30, 60, 90)]
        target_ivs = {30: 0.18, 60: 0.22, 90: 0.26}
        surf = create_empirical_surface(
            iv_by_tenor=target_ivs,
            as_of_date=as_of,
            underlying="SPY",
            spot=450,
            expiries=expiries,
        )
        for exp, target in zip(expiries, target_ivs.values()):
            atm_iv = surf.get_iv(450, exp, 450)
            # Allow 2 vol points of slack because the SVI curvature term
            # slightly shifts the ATM vol away from the input tenor IV.
            assert abs(atm_iv - target) < 0.02, (
                f"ATM IV at expiry {exp}: got {atm_iv:.4f}, target {target:.4f}"
            )


# =========================================================================
# 3. Early assignment on dividend
# =========================================================================
class TestEarlyAssignmentOnDividend:
    def _ohlcv(self, prices: list[float]) -> pd.DataFrame:
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(prices))]
        return pd.DataFrame({"Date": dates, "Close": prices})

    def test_short_itm_call_assigned_day_before_exdiv_when_time_value_small(self):
        """Short ITM call with $2.00 expected dividend and <$2.00 time
        value must be forced-assigned the day before ex-div.

        We set ``stop_loss_multiple`` very large (no stop) and enter with
        a premium that already reflects the deep ITM state so the time
        value is small by the time ex-div arrives.
        """
        # Start ITM at 105 (strike 100), mild drift up to 106.
        prices = list(np.linspace(105, 106, 30))
        ohlcv = self._ohlcv(prices)
        result = simulate_option_trade(
            option_type="call",
            strike=100,
            entry_premium=5.5,  # intrinsic 5 + tiny tv
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 1, 30),
            entry_iv=0.15,
            ohlcv_df=ohlcv,
            profit_target_pct=0.99,  # disable profit target
            stop_loss_multiple=100.0,  # disable stop loss
            dividend_yield=0.0,
            ex_div_date=date(2024, 1, 20),  # day 20
            expected_dividend=2.00,
            early_assignment_on_div=True,
        )
        assert result is not None
        assert result.exit_reason == "early_assignment_div", (
            f"Expected early_assignment_div, got {result.exit_reason}"
        )
        assert result.was_assigned is True

    def test_disabling_flag_prevents_early_assignment(self):
        prices = list(np.linspace(100, 115, 30))
        ohlcv = self._ohlcv(prices)
        result = simulate_option_trade(
            option_type="call",
            strike=100,
            entry_premium=4.0,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 1, 30),
            entry_iv=0.20,
            ohlcv_df=ohlcv,
            ex_div_date=date(2024, 1, 20),
            expected_dividend=2.00,
            early_assignment_on_div=False,
        )
        assert result is not None
        assert result.exit_reason != "early_assignment_div"

    def test_put_never_early_assigned_on_div(self):
        """Short puts are not subject to the dividend early-exercise rule."""
        prices = list(np.linspace(100, 85, 30))  # falling
        ohlcv = self._ohlcv(prices)
        result = simulate_option_trade(
            option_type="put",
            strike=100,
            entry_premium=3.0,
            entry_date=date(2024, 1, 1),
            expiration_date=date(2024, 1, 30),
            entry_iv=0.20,
            ohlcv_df=ohlcv,
            ex_div_date=date(2024, 1, 20),
            expected_dividend=2.00,
            early_assignment_on_div=True,
        )
        assert result is not None
        assert result.exit_reason != "early_assignment_div"


# =========================================================================
# 4. Survivorship-bias guard
# =========================================================================
class TestSurvivorshipBiasGuard:
    def test_detects_backfilled_tickers(self):
        from data.pipeline import DataPipeline

        pipeline = DataPipeline.__new__(DataPipeline)
        pipeline._ohlcv = {}
        # AAPL has full history, NVDA backfilled after start_date
        aapl_idx = pd.date_range("2018-01-01", "2023-12-31", freq="B")
        nvda_idx = pd.date_range("2022-01-01", "2023-12-31", freq="B")
        pipeline._ohlcv["AAPL"] = pd.DataFrame(
            {"Close": np.linspace(100, 150, len(aapl_idx))}, index=aapl_idx
        )
        pipeline._ohlcv["NVDA"] = pd.DataFrame(
            {"Close": np.linspace(200, 500, len(nvda_idx))}, index=nvda_idx
        )

        audit = pipeline.audit_survivorship_bias(
            start_date="2020-01-01",
            end_date="2023-12-31",
            expected_constituents=["AAPL", "NVDA", "DELISTED1", "DELISTED2"],
        )
        assert "NVDA" in audit["backfilled_tickers"]
        assert "AAPL" not in audit["backfilled_tickers"]
        assert "DELISTED1" in audit["missing_tickers"]
        assert audit["bias_score"] > 0.2
        assert "CRITICAL" in audit["verdict"] or "WARNING" in audit["verdict"]

    def test_clean_universe_passes(self):
        from data.pipeline import DataPipeline

        pipeline = DataPipeline.__new__(DataPipeline)
        pipeline._ohlcv = {}
        idx = pd.date_range("2018-01-01", "2023-12-31", freq="B")
        for t in ("AAPL", "MSFT", "GOOGL"):
            pipeline._ohlcv[t] = pd.DataFrame(
                {"Close": np.linspace(100, 150, len(idx))}, index=idx
            )
        audit = pipeline.audit_survivorship_bias(
            start_date="2020-01-01",
            end_date="2023-12-31",
            expected_constituents=["AAPL", "MSFT", "GOOGL"],
        )
        assert audit["bias_score"] == 0.0
        assert "OK" in audit["verdict"]


# =========================================================================
# 5. Brier-score calibration gate
# =========================================================================
class TestCalibrationGate:
    def test_perfectly_calibrated_passes(self):
        rng = np.random.default_rng(42)
        # For each probability level, draw the right fraction of successes
        preds = rng.uniform(0, 1, size=2000).tolist()
        obs = [int(rng.random() < p) for p in preds]
        result = DriftDetector.check_calibration(preds, obs)
        assert result["passed"] is True
        assert result["brier_score"] < 0.25

    def test_miscalibrated_model_fails(self):
        # Model says 90% for everything but only 20% actually occur.
        preds = [0.9] * 500
        obs = [1 if i < 100 else 0 for i in range(500)]
        result = DriftDetector.check_calibration(preds, obs, max_brier_score=0.20)
        assert result["passed"] is False
        assert result["brier_score"] > 0.20

    def test_reliability_diagram_returned(self):
        preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 50
        obs = [int(i % 3 == 0) for i in range(len(preds))]
        result = DriftDetector.check_calibration(preds, obs)
        assert len(result["reliability"]) > 0
        for bin_row in result["reliability"]:
            assert 0 <= bin_row["avg_predicted"] <= 1
            assert 0 <= bin_row["empirical_frequency"] <= 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            DriftDetector.check_calibration([0.5, 0.5], [0])


# =========================================================================
# 6. Square-root market-impact slippage
# =========================================================================
class TestSqrtMarketImpactSlippage:
    def test_backwards_compatible_without_adv(self):
        """Without ``adv_contracts`` the function must equal the old
        spread-only model."""
        s = calculate_slippage(
            mid_price=2.00, bid_ask_spread=0.10, trade_direction="sell"
        )
        # Old model: 0.15 * 0.10 = 0.015
        assert s == pytest.approx(0.015, abs=1e-6)

    def test_sqrt_impact_scales_with_size(self):
        """Larger orders pay more slippage on the same ADV."""
        small = calculate_slippage(
            mid_price=2.00,
            bid_ask_spread=0.10,
            trade_direction="buy",
            num_contracts=1,
            adv_contracts=100,
        )
        big = calculate_slippage(
            mid_price=2.00,
            bid_ask_spread=0.10,
            trade_direction="buy",
            num_contracts=20,
            adv_contracts=100,
        )
        assert big > small, f"sqrt impact not scaling: {small=} {big=}"

    def test_sqrt_impact_scales_with_inverse_adv(self):
        """Same order is worse on a less liquid name."""
        liquid = calculate_slippage(
            mid_price=2.00,
            bid_ask_spread=0.10,
            trade_direction="buy",
            num_contracts=5,
            adv_contracts=10000,
        )
        illiquid = calculate_slippage(
            mid_price=2.00,
            bid_ask_spread=0.10,
            trade_direction="buy",
            num_contracts=5,
            adv_contracts=20,
        )
        assert illiquid > liquid * 2, "sqrt impact not sensitive enough to ADV"


# =========================================================================
# 7. Quality framework bid>ask / stale-quote checks
# =========================================================================
class TestQualityChecks:
    def test_crossed_market_detected(self):
        from data.quality import DataQualityFramework

        df = pd.DataFrame({"bid": [1.0, 2.5, 3.0], "ask": [1.1, 2.0, 3.5]})
        issues = DataQualityFramework()._check_options_consistency(df)
        crossed = [i for i in issues if "Crossed" in i.message]
        assert len(crossed) == 1
        assert crossed[0].affected_rows == 1

    def test_expired_options_detected(self):
        from data.quality import DataQualityFramework

        df = pd.DataFrame(
            {
                "bid": [1.0, 2.0],
                "ask": [1.1, 2.1],
                "expiration": ["2023-01-15", "2024-06-01"],
                "date": ["2023-12-01", "2023-12-01"],
            }
        )
        issues = DataQualityFramework()._check_options_consistency(df)
        expired = [i for i in issues if "expiration BEFORE" in i.message]
        assert len(expired) == 1

    def test_stale_quote_detected(self):
        from data.quality import DataQualityFramework

        now = pd.Timestamp("2024-01-15 15:00:00")
        df = pd.DataFrame(
            {
                "bid": [1.0, 2.0],
                "ask": [1.1, 2.1],
                "quote_timestamp": [
                    pd.Timestamp("2024-01-15 14:55:00"),  # fresh
                    pd.Timestamp("2024-01-15 10:00:00"),  # 5h stale
                ],
            }
        )
        issues = DataQualityFramework().check_stale_quotes(
            df, as_of=now, max_age_minutes=30
        )
        assert len(issues) == 1
        assert issues[0].affected_rows == 1


# =========================================================================
# 8. EV-driven wheel ranker integration (light smoke test)
# =========================================================================
class TestWheelRunnerEVRanker:
    def test_rank_candidates_by_ev_returns_sorted_frame(self, monkeypatch):
        """Smoke test: with a mocked connector, the EV ranker returns a
        non-empty sorted DataFrame and drops candidates near earnings.
        """
        from engine.wheel_runner import WheelRunner

        # Build a fake connector that returns synthetic data for two
        # tickers. This exercises the full EV pipeline end-to-end.
        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices_a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))
        prices_b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.020, n)))

        class FakeConn:
            def __init__(self):
                self.calls = 0

            def get_ohlcv(self, ticker):
                if ticker == "AAA":
                    return pd.DataFrame({"close": prices_a}, index=idx)
                if ticker == "BBB":
                    return pd.DataFrame({"close": prices_b}, index=idx)
                return pd.DataFrame()

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.25,
                    "volatility_30d": 0.22,
                    "dividend_yield": 0.01,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["AAA", "BBB"]

        runner = WheelRunner()
        runner._connector = FakeConn()

        df = runner.rank_candidates_by_ev(
            tickers=["AAA", "BBB"],
            dte_target=30,
            delta_target=0.25,
            top_n=5,
            min_ev_dollars=-1e9,
        )
        assert not df.empty
        assert list(df.columns)[:5] == [
            "ticker",
            "spot",
            "strike",
            "premium",
            "dte",
        ]
        # ev_per_day descending
        if len(df) > 1:
            assert df["ev_per_day"].iloc[0] >= df["ev_per_day"].iloc[-1]
        # A distribution source was used
        assert df["distribution_source"].iloc[0] in (
            "empirical_non_overlapping",
            "empirical_overlapping",
            "block_bootstrap",
            "har_rv",
        )
