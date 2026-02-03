"""
Tests for data validation module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_validation import (
    validate_and_normalize_iv,
    validate_option_data,
    validate_ohlcv_data,
    apply_liquidity_filter,
    ValidationSeverity
)


class TestIVValidation:
    """Tests for IV validation and normalization."""

    def test_normal_iv(self):
        """Normal IV values should pass through unchanged."""
        iv, warning = validate_and_normalize_iv(0.25)
        assert iv == 0.25
        assert warning is None

    def test_zero_iv(self):
        """Zero IV should return None."""
        iv, warning = validate_and_normalize_iv(0)
        assert iv is None
        assert "non-positive" in warning

    def test_negative_iv(self):
        """Negative IV should return None."""
        iv, warning = validate_and_normalize_iv(-0.25)
        assert iv is None
        assert "non-positive" in warning

    def test_nan_iv(self):
        """NaN IV should return None."""
        iv, warning = validate_and_normalize_iv(float('nan'))
        assert iv is None
        assert "NaN" in warning

    def test_percentage_format_iv(self):
        """IV in percentage format (>10) should be normalized."""
        iv, warning = validate_and_normalize_iv(25)  # 25% = 0.25
        assert iv == 0.25
        assert "percentage format" in warning

    def test_high_but_valid_iv(self):
        """High but valid IV (1-5) should pass with warning."""
        iv, warning = validate_and_normalize_iv(2.5)  # 250%
        assert iv == 2.5
        assert "high" in warning.lower()

    def test_very_low_iv(self):
        """Very low IV should pass with warning."""
        iv, warning = validate_and_normalize_iv(0.02)  # 2%
        assert iv == 0.02
        assert warning is not None and "low" in warning.lower()


class TestOptionDataValidation:
    """Tests for option data validation."""

    def test_valid_data(self):
        """Valid data should pass validation."""
        df = pd.DataFrame({
            'strike': [100, 105],
            'bid': [2.00, 1.50],
            'ask': [2.10, 1.60],
            'implied_vol': [0.25, 0.30],
            'underlying_price': [100, 100],
            'option_type': ['P', 'C'],
            'mid_price': [2.05, 1.55]
        })

        result = validate_option_data(df)
        assert len(result.valid_df) == 2
        assert len(result.invalid_df) == 0

    def test_bid_greater_than_ask(self):
        """Bid > Ask should be flagged as error."""
        df = pd.DataFrame({
            'strike': [100],
            'bid': [2.50],  # Bid > Ask is invalid
            'ask': [2.00],
            'implied_vol': [0.25],
            'underlying_price': [100],
        })

        result = validate_option_data(df)
        assert len(result.invalid_df) == 1
        assert any(i.field == 'bid_ask' for i in result.issues)

    def test_negative_prices(self):
        """Negative prices should be flagged."""
        df = pd.DataFrame({
            'strike': [100],
            'bid': [-1.00],
            'ask': [2.00],
            'implied_vol': [0.25],
            'underlying_price': [100],
        })

        result = validate_option_data(df)
        assert len(result.invalid_df) == 1

    def test_invalid_iv_flagged(self):
        """Invalid IV should be flagged and removed."""
        df = pd.DataFrame({
            'strike': [100, 105],
            'bid': [2.00, 1.50],
            'ask': [2.10, 1.60],
            'implied_vol': [0.25, -0.10],  # Second is invalid
            'underlying_price': [100, 100],
        })

        result = validate_option_data(df)
        # Invalid IV row should be removed
        assert len(result.invalid_df) == 1
        assert any(i.field == 'implied_vol' for i in result.issues)


class TestOHLCVValidation:
    """Tests for OHLCV data validation."""

    def test_valid_ohlcv(self):
        """Valid OHLCV should pass."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
        })

        result = validate_ohlcv_data(df)
        assert len(result.valid_df) == 2
        assert len(result.invalid_df) == 0

    def test_high_less_than_low(self):
        """High < Low should be flagged."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01']),
            'Open': [100],
            'High': [95],  # Less than Low
            'Low': [99],
            'Close': [97],
        })

        result = validate_ohlcv_data(df)
        assert len(result.invalid_df) == 1

    def test_negative_prices(self):
        """Negative OHLCV prices should be flagged."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01']),
            'Open': [-100],  # Negative
            'High': [102],
            'Low': [99],
            'Close': [101],
        })

        result = validate_ohlcv_data(df)
        assert len(result.invalid_df) == 1


class TestLiquidityFilter:
    """Tests for liquidity filtering."""

    def test_filters_low_oi(self):
        """Options with low OI should be filtered."""
        df = pd.DataFrame({
            'bid': [2.00, 2.00],
            'ask': [2.10, 2.10],
            'open_interest': [50, 200],  # First below threshold
        })

        filtered = apply_liquidity_filter(df, min_open_interest=100)
        assert len(filtered) == 1
        assert filtered.iloc[0]['open_interest'] == 200

    def test_filters_zero_bid(self):
        """Options with zero bid should be filtered."""
        df = pd.DataFrame({
            'bid': [0.00, 2.00],
            'ask': [2.10, 2.10],
            'open_interest': [200, 200],
        })

        filtered = apply_liquidity_filter(df, min_bid=0.01)
        assert len(filtered) == 1
        assert filtered.iloc[0]['bid'] == 2.00

    def test_filters_wide_spread(self):
        """Options with wide spread should be filtered."""
        df = pd.DataFrame({
            'bid': [1.00, 2.00],
            'ask': [2.00, 2.20],  # First has 67% spread, second has 9%
            'open_interest': [200, 200],
        })

        filtered = apply_liquidity_filter(df, max_spread_pct=0.50)
        assert len(filtered) == 1
        assert filtered.iloc[0]['bid'] == 2.00


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
