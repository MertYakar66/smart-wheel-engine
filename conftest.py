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
        "markers",
        "integration: cross-boundary tests (loopback HTTP server / subprocess spawns) — "
        "headless-CI-runnable; anchors the CI Integration Tests lane",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quant: marks tests as quantitative validation tests"
    )
    config.addinivalue_line(
        "markers",
        "backtest_regression: long-running backtest reproducers — excluded from per-PR CI; "
        "run via .claude/commands/backtest-regression.md",
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


@pytest.fixture(scope="session", autouse=True)
def _neutralize_option_premium_rail(tmp_path_factory):
    """Pin SWE_OPTION_PREMIUM_DIR to an empty dir for the whole suite (D4-2).

    The option-premium rail (gitignored ``data_processed/option_premium/``,
    absent in CI) is otherwise picked up via the DEFAULT-path fallback in
    ``MarketDataConnector.__init__`` — unset/empty env means "use the repo
    default dir", NOT "rail off" — and swaps the ranker's synthetic-BSM
    premium for real market mids, breaking exact-EV pins locally that are
    green in CI (the AAPL $5.35 F4 control was the live instance).

    Session scope is load-bearing: module/class-scoped runner fixtures
    (``test_w2_output_realism.ranked``, ``TestF4CasesRanker.runner``)
    construct the connector during fixture setup, before any function-scoped
    autouse fixture would run. Rail tests opt in by ``monkeypatch.setenv``-ing
    their own dir (function scope overrides this baseline and restores it on
    teardown). The env var is read once per connector construction — keep it
    that way; an import-time read would defeat this fixture.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv(
        "SWE_OPTION_PREMIUM_DIR",
        str(tmp_path_factory.mktemp("no_option_premium_rail")),
    )
    yield
    mp.undo()


def make_gbm_ohlcv(n=252, seed=42, start="2024-01-01", close_only=False):
    """Deterministic GBM OHLCV generator — a parametric HELPER, not a fixture.

    Mirrors the ``sample_ohlcv`` fixture's geometric-brownian process but takes
    explicit ``n`` / ``seed`` / ``start`` so a test can request its own series
    without a dedicated fixture. Self-contained (its own ``default_rng`` — never
    mutates the global numpy RNG). Returns the ``close`` Series when
    ``close_only=True``, else a full open/high/low/close/volume DataFrame indexed
    by business days. Provided for opt-in use; no existing test is migrated.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    if close_only:
        return pd.Series(close, index=dates, name="close")
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_prices = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n))
    open_prices[0] = 100.0
    volume = rng.lognormal(15, 0.5, n)
    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
