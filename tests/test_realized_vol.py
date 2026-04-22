"""Unit tests for realized-volatility estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.realized_vol import (
    close_to_close_vol,
    garman_klass_vol,
    parkinson_vol,
    realised_vol_bundle,
    rogers_satchell_vol,
    vol_risk_premium_bundle,
    yang_zhang_vol,
)


@pytest.fixture
def sample_ohlc() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=dates
    )


def test_close_to_close_positive(sample_ohlc):
    v = close_to_close_vol(sample_ohlc)
    assert 0 < v < 5.0, f"expected decimal vol, got {v}"


def test_parkinson_positive(sample_ohlc):
    assert parkinson_vol(sample_ohlc) > 0


def test_garman_klass_positive(sample_ohlc):
    assert garman_klass_vol(sample_ohlc) > 0


def test_rogers_satchell_positive(sample_ohlc):
    assert rogers_satchell_vol(sample_ohlc) >= 0


def test_yang_zhang_positive(sample_ohlc):
    assert yang_zhang_vol(sample_ohlc) > 0


def test_all_estimators_in_same_ballpark(sample_ohlc):
    """The five estimators should be within 3x of each other on a
    well-behaved sample — they estimate the same quantity."""
    b = realised_vol_bundle(sample_ohlc)
    vals = [v for v in b.values() if np.isfinite(v) and v > 0]
    assert len(vals) == 5
    ratio = max(vals) / min(vals)
    assert ratio < 3.0, f"estimator spread too high: {b}"


def test_empty_df_returns_nan():
    empty = pd.DataFrame(columns=["open", "high", "low", "close"])
    for fn in (close_to_close_vol, parkinson_vol, garman_klass_vol, rogers_satchell_vol, yang_zhang_vol):
        v = fn(empty)
        assert np.isnan(v), f"{fn.__name__} should return NaN on empty df"


def test_vrp_bundle_sign(sample_ohlc):
    """VRP = IV - RV. With IV=0.40 and realised ~0.10, VRP should be positive."""
    vrp = vol_risk_premium_bundle(sample_ohlc, iv_atm=0.40)
    assert vrp["iv_atm"] == 0.40
    assert np.isfinite(vrp["consensus_rv"])
    assert vrp["consensus_vrp"] > 0, "rich premium should give positive VRP"
