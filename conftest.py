"""
Pytest configuration and fixtures for Smart Wheel Engine.

This file configures:
1. Hypothesis profiles for property-based testing
2. Common test fixtures
3. Test markers
"""

import pytest
import numpy as np
import pandas as pd

# Configure hypothesis profiles
try:
    from hypothesis import settings, Verbosity

    # CI profile: more examples, faster deadline
    settings.register_profile(
        "ci",
        max_examples=200,
        deadline=None,
        verbosity=Verbosity.normal,
    )

    # Dev profile: fewer examples for faster iteration
    settings.register_profile(
        "dev",
        max_examples=50,
        deadline=None,
        verbosity=Verbosity.verbose,
    )

    # Debug profile: minimal examples with verbose output
    settings.register_profile(
        "debug",
        max_examples=10,
        deadline=None,
        verbosity=Verbosity.verbose,
    )

except ImportError:
    pass  # hypothesis not installed


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quant: marks tests as quantitative validation tests"
    )


@pytest.fixture
def sample_prices():
    """Generate sample price series for testing."""
    np.random.seed(42)
    n = 252  # One year of trading days
    returns = np.random.normal(0.0005, 0.02, n)  # ~12% annual return, 32% vol
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    return pd.Series(prices, index=dates, name='close')


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range('2024-01-01', periods=n, freq='B')

    # Generate realistic OHLCV data
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))

    # High/Low relative to close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))

    # Open is previous close with overnight gap
    open_prices = np.roll(close, 1) * (1 + np.random.normal(0, 0.002, n))
    open_prices[0] = 100

    # Volume
    volume = np.random.lognormal(15, 0.5, n)

    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


@pytest.fixture
def sample_iv_series():
    """Generate sample IV series for testing."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range('2024-01-01', periods=n, freq='B')

    # IV mean-reverts around 0.20 (20%)
    iv = 0.20 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.02, n)
    iv = np.clip(iv, 0.05, 0.80)  # Keep in realistic range

    return pd.Series(iv, index=dates, name='iv')


@pytest.fixture
def sample_greeks():
    """Generate sample Greeks for portfolio testing."""
    return {
        'delta': 0.45,
        'gamma': 0.02,
        'theta': -0.05,
        'vega': 0.15,
        'rho': 0.08,
    }


@pytest.fixture
def sample_portfolio():
    """Generate sample portfolio for risk testing."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'position': [100, 50, 30, 40, 60],
        'price': [175.0, 380.0, 140.0, 180.0, 500.0],
        'delta': [0.5, 0.45, 0.55, 0.48, 0.52],
        'gamma': [0.02, 0.015, 0.025, 0.018, 0.022],
        'vega': [0.12, 0.15, 0.10, 0.14, 0.11],
    })
