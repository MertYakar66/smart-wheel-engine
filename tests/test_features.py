"""
Tests for src/features modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dynamics import OptionsDynamics
from src.features.options import OptionsFeatures
from src.features.technical import TechnicalFeatures
from src.features.volatility import VolatilityFeatures

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_prices():
    """Create sample price series."""
    np.random.seed(42)
    n = 100
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def uptrend_prices():
    """Create uptrending price series."""
    n = 50
    prices = 100 + np.arange(n) * 0.5 + np.random.normal(0, 0.5, n)
    return pd.Series(prices)


@pytest.fixture
def downtrend_prices():
    """Create downtrending price series."""
    n = 50
    prices = 100 - np.arange(n) * 0.5 + np.random.normal(0, 0.5, n)
    return pd.Series(prices)


# =============================================================================
# TECHNICAL FEATURES TESTS
# =============================================================================


class TestSMA:
    """Test Simple Moving Average."""

    def test_sma_basic(self, sample_prices):
        """Test basic SMA calculation."""
        sma = TechnicalFeatures.sma(sample_prices, 10)

        assert len(sma) == len(sample_prices)
        assert sma.isna().sum() == 9  # First 9 values should be NaN

    def test_sma_values(self):
        """Test SMA values are correct."""
        prices = pd.Series([1, 2, 3, 4, 5])
        sma = TechnicalFeatures.sma(prices, 3)

        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])
        assert sma.iloc[2] == 2.0  # (1+2+3)/3
        assert sma.iloc[3] == 3.0  # (2+3+4)/3
        assert sma.iloc[4] == 4.0  # (3+4+5)/3


class TestEMA:
    """Test Exponential Moving Average."""

    def test_ema_basic(self, sample_prices):
        """Test basic EMA calculation."""
        ema = TechnicalFeatures.ema(sample_prices, 10)

        assert len(ema) == len(sample_prices)
        # EMA should have values from the start
        assert not ema.isna().all()

    def test_ema_follows_trend(self, uptrend_prices):
        """Test EMA follows price trend."""
        ema = TechnicalFeatures.ema(uptrend_prices, 10)

        # EMA should be generally increasing in uptrend
        # Compare first half average to second half average
        first_half = ema.iloc[10:25].mean()
        second_half = ema.iloc[35:50].mean()
        assert second_half > first_half


class TestMARatio:
    """Test Moving Average Ratio."""

    def test_ma_ratio_basic(self, sample_prices):
        """Test MA ratio calculation."""
        ratio = TechnicalFeatures.ma_ratio(sample_prices, 20)

        assert len(ratio) == len(sample_prices)
        # Ratio should be around 1.0 on average
        valid_ratio = ratio.dropna()
        assert 0.8 < valid_ratio.mean() < 1.2


class TestRSI:
    """Test Relative Strength Index."""

    def test_rsi_basic(self, sample_prices):
        """Test basic RSI calculation."""
        rsi = TechnicalFeatures.rsi(sample_prices, 14)

        assert len(rsi) == len(sample_prices)
        # First 14 values should be NaN
        assert rsi.iloc[:14].isna().all()

    def test_rsi_range(self, sample_prices):
        """Test RSI is in valid range."""
        rsi = TechnicalFeatures.rsi(sample_prices, 14)
        valid_rsi = rsi.dropna()

        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_uptrend_high(self, uptrend_prices):
        """Test RSI is high in uptrend."""
        rsi = TechnicalFeatures.rsi(uptrend_prices, 14)
        valid_rsi = rsi.dropna()

        # RSI should generally be above 50 in uptrend
        assert valid_rsi.mean() > 50

    def test_rsi_downtrend_low(self, downtrend_prices):
        """Test RSI is low in downtrend."""
        rsi = TechnicalFeatures.rsi(downtrend_prices, 14)
        valid_rsi = rsi.dropna()

        # RSI should generally be below 50 in downtrend
        assert valid_rsi.mean() < 50

    def test_rsi_all_gains(self):
        """Test RSI when all days are gains."""
        prices = pd.Series([100 + i for i in range(20)])  # Steady increase
        rsi = TechnicalFeatures.rsi(prices, 14)
        valid_rsi = rsi.dropna()

        # Should be 100 when all gains
        assert (valid_rsi == 100).any() or valid_rsi.mean() > 90

    def test_rsi_short_series(self):
        """Test RSI with series shorter than window."""
        prices = pd.Series([100, 101, 102])
        rsi = TechnicalFeatures.rsi(prices, 14)

        # All should be NaN
        assert rsi.isna().all()


class TestMACD:
    """Test MACD indicator."""

    def test_macd_basic(self, sample_prices):
        """Test basic MACD calculation."""
        macd_line, signal_line, histogram = TechnicalFeatures.macd(sample_prices)

        assert len(macd_line) == len(sample_prices)
        assert len(signal_line) == len(sample_prices)
        assert len(histogram) == len(sample_prices)

    def test_macd_histogram_equals_difference(self, sample_prices):
        """Test histogram equals MACD minus signal."""
        macd_line, signal_line, histogram = TechnicalFeatures.macd(sample_prices)

        # After warmup, histogram should equal macd - signal
        for i in range(30, len(sample_prices)):
            expected = macd_line.iloc[i] - signal_line.iloc[i]
            assert abs(histogram.iloc[i] - expected) < 1e-10


# =============================================================================
# VOLATILITY FEATURES TESTS
# =============================================================================


class TestVolatilityFeatures:
    """Test volatility feature calculations."""

    def test_realized_vol_close(self, sample_prices):
        """Test realized volatility calculation from log returns."""
        # Calculate log returns first
        log_returns = np.log(sample_prices / sample_prices.shift(1))
        vol = VolatilityFeatures.realized_volatility_close(log_returns, 21)

        assert len(vol) == len(sample_prices)
        # Should have NaN at the start (window - 1 + 1 from diff)
        assert vol.iloc[:21].isna().all()

    def test_realized_vol_close_annualized(self, sample_prices):
        """Test realized vol is annualized."""
        log_returns = np.log(sample_prices / sample_prices.shift(1))
        vol = VolatilityFeatures.realized_volatility_close(log_returns, 21, annualize=True)
        valid_vol = vol.dropna()

        # Annualized vol should typically be between 5% and 100%
        assert (valid_vol > 0.01).all()
        assert (valid_vol < 2.0).all()

    def test_parkinson_vol(self, sample_ohlcv):
        """Test Parkinson volatility estimator."""
        vol = VolatilityFeatures.realized_volatility_parkinson(
            sample_ohlcv["high"], sample_ohlcv["low"], 21
        )

        assert len(vol) == len(sample_ohlcv)
        valid_vol = vol.dropna()
        assert (valid_vol > 0).all()

    def test_garman_klass_vol(self, sample_ohlcv):
        """Test Garman-Klass volatility estimator."""
        vol = VolatilityFeatures.realized_volatility_garman_klass(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            21,
        )

        assert len(vol) == len(sample_ohlcv)
        valid_vol = vol.dropna()
        assert (valid_vol > 0).all()

    def test_iv_rank(self):
        """Test IV rank calculation."""
        # Create a IV series with known characteristics
        np.random.seed(42)
        iv = pd.Series(np.random.uniform(0.15, 0.45, 300))

        iv_rank = VolatilityFeatures.iv_rank(iv, lookback=252)

        valid_rank = iv_rank.dropna()
        if len(valid_rank) > 0:
            # IV rank returns 0-100 (percentage)
            assert (valid_rank >= 0).all()
            assert (valid_rank <= 100).all()

    def test_iv_rv_spread(self):
        """Test IV-RV spread calculation."""
        iv = pd.Series([0.25, 0.30, 0.28, 0.35])
        rv = pd.Series([0.20, 0.22, 0.24, 0.26])

        spread = VolatilityFeatures.iv_rv_spread(iv, rv)

        assert spread.iloc[0] == pytest.approx(0.05, abs=0.001)
        assert spread.iloc[3] == pytest.approx(0.09, abs=0.001)


# =============================================================================
# OPTIONS DYNAMICS TESTS
# =============================================================================


class TestOptionsDynamics:
    """Test options dynamics features."""

    def test_oi_change(self):
        """Test open interest change calculation."""
        oi = pd.Series([1000, 1100, 1050, 1200, 1150])
        change = OptionsDynamics.oi_change(oi)

        assert len(change) == len(oi)
        assert np.isnan(change.iloc[0])
        assert change.iloc[1] == 100
        assert change.iloc[2] == -50

    def test_oi_change_pct(self):
        """Test percentage change in OI."""
        oi = pd.Series([1000, 1100, 1050, 1200])
        change_pct = OptionsDynamics.oi_change_pct(oi)

        assert len(change_pct) == len(oi)
        assert change_pct.iloc[1] == pytest.approx(0.10, rel=0.01)

    def test_oi_acceleration(self):
        """Test OI acceleration (second derivative)."""
        oi = pd.Series([1000, 1100, 1200, 1400, 1500])
        accel = OptionsDynamics.oi_acceleration(oi)

        assert len(accel) == len(oi)
        # First two values should be NaN
        assert np.isnan(accel.iloc[0])
        assert np.isnan(accel.iloc[1])

    def test_iv_change(self):
        """Test IV change calculation."""
        iv = pd.Series([0.25, 0.28, 0.26, 0.30, 0.32])
        change = OptionsDynamics.iv_change(iv)

        assert change.iloc[1] == pytest.approx(0.03, abs=0.001)
        assert change.iloc[2] == pytest.approx(-0.02, abs=0.001)

    def test_iv_velocity(self):
        """Test IV velocity calculation."""
        iv = pd.Series([0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32])
        velocity = OptionsDynamics.iv_velocity(iv, window=5)

        assert len(velocity) == len(iv)
        # Steady increase: velocity should be positive
        valid_vel = velocity.dropna()
        if len(valid_vel) > 0:
            assert (valid_vel > 0).all()

    def test_iv_acceleration(self):
        """Test IV acceleration."""
        iv = pd.Series([0.20, 0.22, 0.25, 0.29, 0.34])
        accel = OptionsDynamics.iv_acceleration(iv)

        assert len(accel) == len(iv)

    def test_skew_change(self):
        """Test put-call skew change."""
        put_iv = pd.Series([0.30, 0.32, 0.35, 0.33])
        call_iv = pd.Series([0.25, 0.26, 0.28, 0.27])

        skew_delta = OptionsDynamics.skew_change(put_iv, call_iv)

        assert len(skew_delta) == len(put_iv)
        # First value should be NaN
        assert np.isnan(skew_delta.iloc[0])


# =============================================================================
# OPTIONS FEATURES TESTS
# =============================================================================


class TestOptionsFeatures:
    """Test options feature calculations."""

    def test_put_call_volume_ratio(self):
        """Test put/call volume ratio."""
        put_vol = pd.Series([1000, 1500, 2000, 1200])
        call_vol = pd.Series([500, 1000, 1000, 800])

        ratio = OptionsFeatures.put_call_volume_ratio(put_vol, call_vol)

        assert ratio.iloc[0] == 2.0  # 1000/500
        assert ratio.iloc[1] == 1.5  # 1500/1000

    def test_put_call_volume_ratio_zero_call(self):
        """Test put/call ratio when call volume is zero."""
        put_vol = pd.Series([1000, 1500])
        call_vol = pd.Series([500, 0])

        ratio = OptionsFeatures.put_call_volume_ratio(put_vol, call_vol)

        assert ratio.iloc[0] == 2.0
        assert np.isnan(ratio.iloc[1])  # Division by zero -> NaN

    def test_put_call_oi_ratio(self):
        """Test put/call OI ratio."""
        put_oi = pd.Series([5000, 6000, 7000])
        call_oi = pd.Series([2500, 3000, 3500])

        ratio = OptionsFeatures.put_call_oi_ratio(put_oi, call_oi)

        assert ratio.iloc[0] == 2.0
        assert ratio.iloc[1] == 2.0

    def test_oi_change(self):
        """Test OI change calculation."""
        oi = pd.Series([1000, 1100, 1200, 1150])

        change = OptionsFeatures.oi_change(oi)

        assert np.isnan(change.iloc[0])
        assert change.iloc[1] == 100
        assert change.iloc[2] == 100
        assert change.iloc[3] == -50

    def test_oi_change_pct(self):
        """Test percentage OI change."""
        oi = pd.Series([1000, 1100, 1210])

        change_pct = OptionsFeatures.oi_change_pct(oi)

        assert change_pct.iloc[1] == pytest.approx(0.10, rel=0.01)
        assert change_pct.iloc[2] == pytest.approx(0.10, rel=0.01)

    def test_unusual_volume_score(self):
        """Test unusual volume detection."""
        # Create volume with one spike
        volume = pd.Series([100] * 25 + [500])  # 25 normal days, then spike

        score = OptionsFeatures.unusual_volume_score(volume, window=20)

        # The spike should have a high z-score
        valid_scores = score.dropna()
        if len(valid_scores) > 0:
            assert valid_scores.iloc[-1] > 2.0  # Unusual

    def test_iv_skew(self):
        """Test IV skew calculation."""
        put_iv = pd.Series([0.30, 0.32, 0.35])
        call_iv = pd.Series([0.25, 0.28, 0.30])

        skew = OptionsFeatures.iv_skew(put_iv, call_iv)

        assert skew.iloc[0] == pytest.approx(0.05, abs=0.001)
        assert skew.iloc[2] == pytest.approx(0.05, abs=0.001)

    def test_iv_crush_expected(self):
        """Test expected IV crush calculation."""
        iv_current = pd.Series([0.40, 0.45, 0.50])
        iv_post_event = pd.Series([0.25, 0.28, 0.30])

        crush = OptionsFeatures.iv_crush_expected(iv_current, iv_post_event)

        assert crush.iloc[0] == pytest.approx(0.15, abs=0.001)
        assert crush.iloc[2] == pytest.approx(0.20, abs=0.001)

    def test_black_scholes_delta(self):
        """Test BS delta calculation."""
        # ATM call with 30 days to expiry
        delta = OptionsFeatures.black_scholes_delta(
            spot=100,
            strike=100,
            time_to_expiry=30 / 365,
            volatility=0.25,
            risk_free_rate=0.05,
            is_call=True,
        )

        # ATM call delta should be around 0.5
        assert 0.4 < delta < 0.6

    def test_black_scholes_delta_put(self):
        """Test BS delta for put."""
        delta = OptionsFeatures.black_scholes_delta(
            spot=100, strike=100, time_to_expiry=30 / 365, volatility=0.25, is_call=False
        )

        # ATM put delta should be around -0.5
        assert -0.6 < delta < -0.4


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_series(self):
        """Test with empty series."""
        empty = pd.Series([], dtype=float)

        sma = TechnicalFeatures.sma(empty, 10)
        assert len(sma) == 0

    def test_single_value(self):
        """Test with single value."""
        single = pd.Series([100.0])

        sma = TechnicalFeatures.sma(single, 10)
        assert len(sma) == 1
        assert sma.isna().all()

    def test_nan_handling(self, sample_prices):
        """Test handling of NaN values in input."""
        prices_with_nan = sample_prices.copy()
        prices_with_nan.iloc[50] = np.nan

        sma = TechnicalFeatures.sma(prices_with_nan, 10)
        # Should still compute (may propagate NaN)
        assert len(sma) == len(prices_with_nan)

    def test_constant_prices(self):
        """Test with constant prices."""
        constant = pd.Series([100.0] * 50)

        rsi = TechnicalFeatures.rsi(constant, 14)
        valid_rsi = rsi.dropna()

        # With no movement, RSI should be 50 (neutral)
        if len(valid_rsi) > 0:
            assert (valid_rsi == 50).all()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for features."""

    def test_all_technical_features(self, sample_ohlcv):
        """Test computing all technical features together."""
        df = sample_ohlcv.copy()

        # Add all technical features
        df["sma_20"] = TechnicalFeatures.sma(df["close"], 20)
        df["ema_20"] = TechnicalFeatures.ema(df["close"], 20)
        df["rsi_14"] = TechnicalFeatures.rsi(df["close"], 14)
        df["ma_ratio_20"] = TechnicalFeatures.ma_ratio(df["close"], 20)

        macd, signal, hist = TechnicalFeatures.macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

        # Check all columns exist
        expected = ["sma_20", "ema_20", "rsi_14", "ma_ratio_20", "macd", "macd_signal", "macd_hist"]
        for col in expected:
            assert col in df.columns

        # Check no all-NaN columns after warmup
        for col in expected:
            assert not df[col].iloc[30:].isna().all()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
