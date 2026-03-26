"""
Point-in-Time and Anti-Lookahead Tests

These tests ensure that features are computed using only information
available at the time of prediction. Any lookahead bias can lead to
unrealistic backtesting results and production failures.

Critical checks:
1. Rolling features only use past data
2. Forward labels use correct alignment
3. No future data leaks through index alignment
4. Train/test splits respect time boundaries
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.technical import TechnicalFeatures
from src.features.volatility import VolatilityFeatures
from src.features.labels import LabelGenerator


class TestRollingFeaturesPIT:
    """
    Test that rolling features don't use future data.

    A feature computed at time T should only depend on data at times <= T.
    """

    def test_sma_uses_only_past_data(self):
        """SMA at time T should only use data up to T."""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        # Compute SMA with window=3
        sma = prices.rolling(3).mean()

        # SMA[2] (third value) should be mean of [0,1,2] = (100+101+102)/3 = 101
        assert sma.iloc[2] == pytest.approx(101.0, rel=1e-10)

        # SMA[5] should be mean of [3,4,5] = (103+104+105)/3 = 104
        assert sma.iloc[5] == pytest.approx(104.0, rel=1e-10)

        # Verify first two values are NaN (not enough history)
        assert np.isnan(sma.iloc[0])
        assert np.isnan(sma.iloc[1])

    def test_rsi_no_lookahead(self):
        """RSI computed at time T uses only data up to T."""
        # Create price series where we know the pattern changes
        prices = pd.Series([100] * 10 + [110] * 5)  # Flat then up

        tf = TechnicalFeatures()
        rsi = tf.rsi(prices, window=5)

        # RSI at index 9 (last flat day) should still be 50
        # because it only sees flat prices
        assert rsi.iloc[9] == 50.0, f"RSI at flat should be 50, got {rsi.iloc[9]}"

        # RSI at index 10-14 should show uptrend
        # (seeing the jump from 100 to 110)
        assert rsi.iloc[-1] > 50.0, f"RSI after uptrend should be >50"

    def test_rv_no_future_returns(self):
        """Realized volatility uses only past returns."""
        np.random.seed(42)

        # Create returns: low vol then high vol
        returns_low = pd.Series(np.random.normal(0, 0.01, 50))  # 1% daily vol
        returns_high = pd.Series(np.random.normal(0, 0.05, 20))  # 5% daily vol
        returns = pd.concat([returns_low, returns_high], ignore_index=True)

        vf = VolatilityFeatures()
        rv = vf.realized_volatility_close(returns, window=10, annualize=True)

        # RV at index 49 (last low vol day) should be lower than RV at index 69
        rv_low_period = rv.iloc[49]
        rv_high_period = rv.iloc[69]

        assert rv_low_period < rv_high_period, \
            f"RV in low vol period ({rv_low_period}) should be < high vol period ({rv_high_period})"

        # Most importantly: RV at day 49 should NOT know about the upcoming high vol
        # If there was lookahead, it would be elevated
        assert rv_low_period < 0.30, \
            f"RV in low vol period should be low (<30%), got {rv_low_period:.2%}"

    def test_iv_rank_no_future_extremes(self):
        """IV Rank uses only historical min/max, not future."""
        # IV that is average, then spikes at end
        iv = pd.Series([0.20] * 50 + [0.25] * 10 + [0.40] * 5)

        vf = VolatilityFeatures()
        rank = vf.iv_rank(iv, lookback=30)

        # Rank at index 55 (IV=0.25) should be relative to [0.20, 0.25]
        # Not relative to the future spike to 0.40
        # Rank = (0.25 - 0.20) / (0.25 - 0.20) * 100 = 100
        assert rank.iloc[55] == pytest.approx(100.0, abs=1.0), \
            f"IV Rank at day 55 should be ~100 (highest so far), got {rank.iloc[55]}"


class TestForwardLabelsAlignment:
    """
    Test that forward labels are correctly aligned in time.

    A label at time T should represent information from time > T.
    """

    def test_forward_return_alignment(self):
        """Forward return at T should use price at T+N, not T-N."""
        # Prices: known pattern
        prices = pd.Series([100, 101, 102, 103, 104, 105])

        lg = LabelGenerator()
        fwd_ret = lg.forward_return(prices, periods=2)

        # Forward return at index 0 should be (102 - 100) / 100 = 2%
        expected = (102 - 100) / 100
        assert fwd_ret.iloc[0] == pytest.approx(expected, rel=1e-10), \
            f"Forward return[0] should be {expected}, got {fwd_ret.iloc[0]}"

        # Forward return at index 3 should be (105 - 103) / 103
        expected = (105 - 103) / 103
        assert fwd_ret.iloc[3] == pytest.approx(expected, rel=1e-10)

        # Last 2 values should be NaN (no future data)
        assert np.isnan(fwd_ret.iloc[-1])
        assert np.isnan(fwd_ret.iloc[-2])

    def test_forward_labels_nan_tail(self):
        """Forward labels must have NaN at the end (insufficient future data)."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        lg = LabelGenerator()

        for horizon in [1, 5, 10, 21]:
            fwd_ret = lg.forward_return(prices, periods=horizon)

            # Last `horizon` values must be NaN
            tail = fwd_ret.iloc[-horizon:]
            assert tail.isna().all(), \
                f"Last {horizon} values must be NaN for horizon={horizon}"

            # Value at index len-horizon-1 should NOT be NaN
            assert not np.isnan(fwd_ret.iloc[-horizon-1]), \
                f"Value at index -{horizon}-1 should not be NaN"

    def test_forward_binary_label_alignment(self):
        """Binary label (up/down) should predict future direction."""
        # Prices: down then up
        # Index:  0    1    2    3    4    5    6    7    8    9   10
        prices = pd.Series([100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100])

        lg = LabelGenerator()
        label = lg.forward_return_binary(prices, periods=2, threshold=0.0)

        # At index 4 (price=96), price[6]=96 -> return = 0, label = 0
        assert label.iloc[4] == 0, f"Expected 0 at index 4, got {label.iloc[4]}"

        # At index 5 (price=95), price[7]=97 -> return = 2/95 > 0, label = 1
        assert label.iloc[5] == 1, f"Expected 1 at index 5, got {label.iloc[5]}"

        # At index 0 (price=100), price[2]=98 -> return = -2/100 < 0, label = 0
        assert label.iloc[0] == 0, f"Expected 0 at index 0, got {label.iloc[0]}"

        # Last 2 values should be NaN (no future data)
        assert label.iloc[-2:].isna().all(), "Last 2 values should be NaN"

    def test_no_lookahead_in_label_shift(self):
        """Ensure shift direction is correct for forward labels."""
        # Simple test: if we shift(-1), we should get NEXT value
        s = pd.Series([1, 2, 3, 4, 5])

        forward = s.shift(-1)  # Next value
        backward = s.shift(1)  # Previous value

        # forward[0] should be 2 (next value)
        assert forward.iloc[0] == 2
        # backward[0] should be NaN (no previous)
        assert np.isnan(backward.iloc[0])
        # backward[1] should be 1 (previous value)
        assert backward.iloc[1] == 1


class TestFeatureIndexAlignment:
    """
    Test that feature joins don't cause lookahead through index misalignment.
    """

    def test_date_index_merge_no_future(self):
        """Merging DataFrames by date should not leak future data."""
        # Price data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'date': dates,
            'close': range(100, 110)
        })

        # Feature data (with same dates)
        features = pd.DataFrame({
            'date': dates,
            'feature': range(10)
        })

        # Merge on date
        merged = prices.merge(features, on='date')

        # Feature at row 5 should match the feature computed with data up to date 5
        assert merged.iloc[5]['feature'] == 5
        assert merged.iloc[5]['close'] == 105

    def test_asof_merge_uses_past_only(self):
        """as_of merge should use most recent past value, not future."""
        # Daily data
        daily_dates = pd.date_range('2024-01-01', periods=15, freq='D')
        daily = pd.DataFrame({
            'date': daily_dates,
            'daily_val': range(15)
        })

        # Weekly data (specific dates, not using 'W' frequency which starts on Sundays)
        weekly = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15']),
            'weekly_val': [100, 200, 300]
        })

        # as_of merge: for each daily date, get most recent weekly value
        daily = daily.sort_values('date')
        weekly = weekly.sort_values('date')

        merged = pd.merge_asof(
            daily,
            weekly,
            on='date',
            direction='backward'  # Only look at past
        )

        # Check that we're using past weekly values, not future
        # On 2024-01-01, weekly_val should be 100 (same day)
        # On 2024-01-05, weekly_val should still be 100 (most recent past)
        # On 2024-01-09, weekly_val should be 200 (2024-01-08 is most recent)
        assert merged[merged['date'] == '2024-01-01']['weekly_val'].iloc[0] == 100
        assert merged[merged['date'] == '2024-01-05']['weekly_val'].iloc[0] == 100
        assert merged[merged['date'] == '2024-01-09']['weekly_val'].iloc[0] == 200


class TestTrainTestSplitPIT:
    """
    Test that train/test splits respect time boundaries.
    """

    def test_no_future_in_training_set(self):
        """Training set should not contain dates after test set start."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': range(100)
        })

        # Split at day 80
        split_date = dates[79]
        train = df[df['date'] <= split_date]
        test = df[df['date'] > split_date]

        # Verify no overlap
        train_max = train['date'].max()
        test_min = test['date'].min()

        assert train_max < test_min, \
            f"Train max date {train_max} should be < test min date {test_min}"

    def test_embargo_period(self):
        """
        Embargo period should create a gap between train and test.

        This prevents leakage from labels that span across the boundary.
        """
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': range(100)
        })

        # Split at day 80 with 5-day embargo
        split_idx = 79
        embargo_days = 5

        train_end_idx = split_idx - embargo_days
        test_start_idx = split_idx + 1

        train = df.iloc[:train_end_idx + 1]
        test = df.iloc[test_start_idx:]

        # Gap should be at least embargo_days
        train_max_date = train['date'].max()
        test_min_date = test['date'].min()
        gap = (test_min_date - train_max_date).days

        assert gap >= embargo_days, \
            f"Gap ({gap} days) should be >= embargo ({embargo_days} days)"


class TestFeaturePipelinePIT:
    """
    Integration tests for full feature pipeline point-in-time correctness.
    """

    def test_feature_pipeline_no_lookahead(self):
        """
        End-to-end test: features at time T should not change
        when new data is added at time T+1.

        This is the gold standard PIT test.
        """
        np.random.seed(42)

        # Create base dataset
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        prices = pd.Series(np.random.randn(n).cumsum() + 100, index=dates)

        tf = TechnicalFeatures()
        vf = VolatilityFeatures()

        # Compute features with full data
        returns = prices.pct_change()
        rsi_full = tf.rsi(prices, window=14)
        rv_full = vf.realized_volatility_close(returns, window=21, annualize=True)

        # Compute features with truncated data (up to day 80)
        prices_partial = prices.iloc[:80]
        returns_partial = prices_partial.pct_change()
        rsi_partial = tf.rsi(prices_partial, window=14)
        rv_partial = vf.realized_volatility_close(returns_partial, window=21, annualize=True)

        # Features at day 79 should be IDENTICAL
        # (no lookahead means future data doesn't affect past features)
        day_79_full_rsi = rsi_full.iloc[79]
        day_79_partial_rsi = rsi_partial.iloc[79]

        assert day_79_full_rsi == pytest.approx(day_79_partial_rsi, rel=1e-10), \
            f"RSI at day 79 changed when future data added: {day_79_partial_rsi} -> {day_79_full_rsi}"

        day_79_full_rv = rv_full.iloc[79]
        day_79_partial_rv = rv_partial.iloc[79]

        assert day_79_full_rv == pytest.approx(day_79_partial_rv, rel=1e-10), \
            f"RV at day 79 changed when future data added: {day_79_partial_rv} -> {day_79_full_rv}"


class TestLoopParityForLabels:
    """
    Test that vectorized label implementations match loop-based ones.

    This ensures the vectorized code doesn't have subtle alignment bugs.
    """

    def test_forward_return_loop_parity(self):
        """Vectorized forward return should match explicit loop."""
        prices = pd.Series([100, 102, 101, 105, 103, 108, 110, 107, 112, 115])
        horizon = 3

        # Vectorized version
        lg = LabelGenerator()
        vec_result = lg.forward_return(prices, periods=horizon)

        # Loop version
        loop_result = pd.Series(np.nan, index=prices.index)
        for i in range(len(prices) - horizon):
            loop_result.iloc[i] = (prices.iloc[i + horizon] - prices.iloc[i]) / prices.iloc[i]

        # Compare non-NaN values
        for i in range(len(prices) - horizon):
            assert vec_result.iloc[i] == pytest.approx(loop_result.iloc[i], rel=1e-10), \
                f"Mismatch at index {i}: vec={vec_result.iloc[i]}, loop={loop_result.iloc[i]}"

        # Both should have NaN in same places
        assert vec_result.isna().equals(loop_result.isna()), \
            "NaN patterns don't match between vectorized and loop versions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
