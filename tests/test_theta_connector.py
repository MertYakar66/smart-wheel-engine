"""
Smoke tests for ThetaConnector.

These tests run against the live ThetaTerminal on 127.0.0.1:25503 and
are skipped automatically when the Terminal is not running.  They verify
that the connector returns sensible data shapes and that unit invariants
(IV as decimal, rate as decimal) are enforced.

Run with the Terminal up:
    SWE_DATA_PROVIDER=theta python -m pytest tests/test_theta_connector.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.theta_connector import ThetaConnector


@pytest.fixture(scope="module")
def conn():
    c = ThetaConnector()
    if not c.is_terminal_alive():
        pytest.skip("ThetaTerminal not running on 127.0.0.1:25503")
    return c


# -----------------------------------------------------------------------
# Terminal health
# -----------------------------------------------------------------------

def test_terminal_alive(conn):
    assert conn.is_terminal_alive()


# -----------------------------------------------------------------------
# Option chain
# -----------------------------------------------------------------------

def test_option_chain_returns_dataframe(conn):
    df = conn.get_option_chain("SPY", dte_target=35)
    assert isinstance(df, pd.DataFrame), "get_option_chain must return a DataFrame"
    assert len(df) > 0, "SPY chain must be non-empty"


def test_option_chain_has_required_columns(conn):
    df = conn.get_option_chain("SPY", dte_target=35)
    for col in ("strike", "right", "iv"):
        assert col in df.columns, f"Missing column: {col}"


def test_option_chain_iv_is_decimal(conn):
    """IV must be a decimal (0 < iv <= 3.0), not a percent (>3)."""
    df = conn.get_option_chain("SPY", dte_target=35)
    iv = df["iv"].dropna()
    assert len(iv) > 0, "No IV values in chain"
    assert (iv > 0).all(), "All IVs must be positive"
    assert (iv <= 3.0).all(), (
        f"IV values look like percent not decimal — max={iv.max():.2f}. "
        "ThetaData returns IV as decimal (e.g. 0.2617 = 26.17%)."
    )


def test_option_chain_delta_range(conn):
    """Deltas must be in [-1, 1]. Puts are negative, calls positive."""
    df = conn.get_option_chain("SPY", dte_target=35)
    if "delta" not in df.columns:
        pytest.skip("delta column not in chain response")
    delta = df["delta"].dropna()
    assert (delta >= -1.0).all() and (delta <= 1.0).all(), (
        f"Delta out of [-1,1] range: min={delta.min():.3f}, max={delta.max():.3f}"
    )


def test_option_chain_cached(conn):
    """Second call with same params returns the same object (cache hit)."""
    df1 = conn.get_option_chain("SPY", dte_target=35)
    df2 = conn.get_option_chain("SPY", dte_target=35)
    assert df1 is df2, "Second call should hit cache and return same DataFrame object"


# -----------------------------------------------------------------------
# Fundamentals
# -----------------------------------------------------------------------

def test_fundamentals_has_live_iv(conn):
    """get_fundamentals should return an implied_vol_atm in decimal form."""
    f = conn.get_fundamentals("SPY")
    assert f is not None, "get_fundamentals returned None for SPY"
    iv = f.get("implied_vol_atm")
    assert iv is not None, "implied_vol_atm missing from fundamentals"
    assert 0 < float(iv) <= 3.0, (
        f"implied_vol_atm={iv} looks like percent not decimal"
    )


# -----------------------------------------------------------------------
# OHLCV
# -----------------------------------------------------------------------

def test_ohlcv_shape(conn):
    df = conn.get_ohlcv("SPY", start_date="2025-01-01")
    assert isinstance(df, pd.DataFrame)
    for col in ("open", "high", "low", "close"):
        assert col in df.columns, f"Missing OHLCV column: {col}"
    assert len(df) > 50, "Expected at least 50 trading days"


def test_ohlcv_high_ge_low(conn):
    df = conn.get_ohlcv("SPY", start_date="2025-01-01")
    df = df.dropna(subset=["high", "low"])
    bad = (df["high"] < df["low"]).sum()
    assert bad == 0, f"{bad} rows where high < low — OHLCV column mapping may be wrong"


# -----------------------------------------------------------------------
# IV rank
# -----------------------------------------------------------------------

def test_iv_rank_in_range(conn):
    rank = conn.get_iv_rank("SPY")
    assert isinstance(rank, float)
    assert 0.0 <= rank <= 1.0, f"IV rank out of [0,1]: {rank}"


# -----------------------------------------------------------------------
# VIX regime (live)
# -----------------------------------------------------------------------

def test_vix_regime_has_vix(conn):
    regime = conn.get_vix_regime()
    assert isinstance(regime, dict)
    vix = regime.get("vix")
    assert vix is not None
    assert not np.isnan(float(vix)), "VIX is NaN"
    assert 5 < float(vix) < 200, f"VIX level suspicious: {vix}"


# -----------------------------------------------------------------------
# Fallback: risk-free rate still comes from Bloomberg CSV (decimal)
# -----------------------------------------------------------------------

def test_risk_free_rate_is_decimal(conn):
    """get_risk_free_rate must return a decimal (e.g. 0.043), not percent (4.3)."""
    rate = conn.get_risk_free_rate()
    if np.isnan(rate):
        pytest.skip("Treasury CSV not present — skipping rate check")
    assert 0.0 < rate < 0.25, (
        f"Risk-free rate={rate} — expected decimal like 0.043, not percent like 4.3"
    )
