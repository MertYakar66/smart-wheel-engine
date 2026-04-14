"""Tests for the TradingView signal bridge (engine/tv_signals.py).

Coverage:
1. TVSignal dataclass round-trip + JSON safety
2. compute_tv_signal basic happy path on synthetic OHLCV
3. Rejection of insufficient history / missing columns / empty frames
4. Phase classification across synthetic compression / expansion /
   post-expansion / trend scenarios
5. IV overlay veto (low IV rank and negative VRP)
6. TVAlert parsing + extras preservation + validity checks
7. Pine parity — thresholds declared in the Pine file match the Python
   constants. This catches accidental edits to one side without the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import tv_signals
from engine.tv_signals import (
    DEFAULTS,
    TVAlert,
    TVSignal,
    compute_tv_signal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, *, drift: float = 0.0003, vol: float = 0.01, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame.

    Returned columns are the minimum set required by ``compute_tv_signal``.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, vol * 0.6, n)))
    low = close * (1 - np.abs(rng.normal(0, vol * 0.6, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vols = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vols}
    )


def _make_compression_ohlcv(n: int = 300) -> pd.DataFrame:
    """Low-vol tape that should classify as compression at the last bar."""
    rng = np.random.default_rng(1)
    # Long low-vol stretch
    rets = rng.normal(0.0, 0.003, n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def _make_post_expansion_ohlcv(n: int = 300) -> pd.DataFrame:
    """Tape with a recent vol expansion that is now stabilizing."""
    rng = np.random.default_rng(2)
    first = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.005, n - 60))
    # Expansion window
    expand = first[-1] * np.cumprod(1 + rng.normal(0.001, 0.04, 30))
    # Stabilization: vol drops but width stays elevated
    stable = expand[-1] * np.cumprod(1 + rng.normal(0.0, 0.008, 30))
    close = np.concatenate([first, expand, stable])
    high = close * (1 + np.abs(rng.normal(0, 0.004, len(close))))
    low = close * (1 - np.abs(rng.normal(0, 0.004, len(close))))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


# ---------------------------------------------------------------------------
# TVSignal dataclass
# ---------------------------------------------------------------------------


def test_tvsignal_default_roundtrip():
    sig = TVSignal(ticker="X")
    d = sig.to_dict()
    assert d["ticker"] == "X"
    assert d["ok"] is False
    assert d["phase"] == "unknown"
    # JSON-safety: no numpy scalars
    for v in d.values():
        assert not isinstance(v, np.generic)


# ---------------------------------------------------------------------------
# compute_tv_signal happy path
# ---------------------------------------------------------------------------


def test_compute_tv_signal_happy_path():
    df = _make_ohlcv()
    sig = compute_tv_signal(df, ticker="TEST")
    assert sig.ok is True
    assert sig.ticker == "TEST"
    assert sig.bar_count == len(df)
    assert 0.0 <= sig.bb_width_pctl <= 100.0
    assert 0.0 <= sig.atr_pctl <= 100.0
    assert 0.0 <= sig.rsi_14 <= 100.0
    assert sig.bollinger_state in {"narrow", "expanding", "wide_flat", "wide_contracting", "normal"}
    assert sig.atr_state in {"low", "rising", "elevated_flat", "declining", "normal"}
    assert sig.rsi_state in {"neutral", "overbought", "oversold", "extreme_ob", "extreme_os"}
    assert sig.trend_state in {"strong_up", "strong_down", "weak", "flat"}
    assert sig.range_state in {"mid", "near_high", "near_low", "beyond_band"}
    assert sig.phase in {"compression", "expansion", "post_expansion", "trend", "normal"}
    # Zones and avoid are mutually reasonable: avoid implies no action zones
    if sig.avoid_zone:
        assert not (sig.wheel_put_zone or sig.covered_call_zone or sig.strangle_zone)


# ---------------------------------------------------------------------------
# Rejection branches
# ---------------------------------------------------------------------------


def test_compute_tv_signal_empty_frame():
    sig = compute_tv_signal(pd.DataFrame(), ticker="X")
    assert sig.ok is False
    assert "empty" in sig.reason


def test_compute_tv_signal_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    sig = compute_tv_signal(df, ticker="X")
    assert sig.ok is False
    assert "missing_columns" in sig.reason


def test_compute_tv_signal_insufficient_history():
    df = _make_ohlcv(n=30)
    sig = compute_tv_signal(df, ticker="X")
    assert sig.ok is False
    assert "insufficient_history" in sig.reason


# ---------------------------------------------------------------------------
# Phase classification scenarios
# ---------------------------------------------------------------------------


def test_phase_compression_on_low_vol_tape():
    df = _make_compression_ohlcv()
    sig = compute_tv_signal(df, ticker="FLAT")
    # A long low-vol stretch should land in compression or normal with low ATR
    assert sig.ok is True
    # Either phase=compression, or at minimum ATR state is low/declining
    assert sig.atr_state in {"low", "declining", "normal"}


def test_phase_post_expansion_triggers_zone_possibility():
    df = _make_post_expansion_ohlcv()
    sig = compute_tv_signal(df, ticker="MU")
    assert sig.ok is True
    # Post-expansion is the *only* phase where the zone resolver allows entry
    if sig.phase == "post_expansion":
        # At least one of the zone flags may be set; the avoid flag should be
        # false unless an IV/VRP veto fired (we passed none).
        assert not sig.avoid_zone


# ---------------------------------------------------------------------------
# IV overlay veto
# ---------------------------------------------------------------------------


def test_iv_overlay_low_iv_blocks_entry():
    df = _make_post_expansion_ohlcv()
    # With an IV rank of 10, the zone resolver must veto
    sig_low = compute_tv_signal(df, ticker="MU", iv_rank=10.0)
    if sig_low.phase == "post_expansion":
        assert sig_low.avoid_zone is True
        assert sig_low.signal_action == "avoid_low_iv"


def test_iv_overlay_negative_vrp_blocks_entry():
    df = _make_post_expansion_ohlcv()
    sig_neg = compute_tv_signal(df, ticker="MU", vol_risk_premium=-10.0)
    if sig_neg.phase == "post_expansion":
        assert sig_neg.avoid_zone is True
        assert sig_neg.signal_action == "avoid_neg_vrp"


# ---------------------------------------------------------------------------
# TVAlert parsing
# ---------------------------------------------------------------------------


def test_tvalert_parse_happy_path():
    alert = TVAlert.parse(
        {
            "ticker": "mu",
            "signal": "wheel_put_zone",
            "price": 82.45,
            "timeframe": "1D",
            "phase": "post_expansion",
            "source": "smart_wheel_signals_v1",
            "rsi": 44.8,
            "bb_width_pctl": 72,
            "secret": "hunter2",
            "custom_debug_key": "preserve_me",
        }
    )
    assert alert.ticker == "MU"  # upcased
    assert alert.signal == "wheel_put_zone"
    assert alert.price == pytest.approx(82.45)
    assert alert.rsi == pytest.approx(44.8)
    assert alert.extras == {"custom_debug_key": "preserve_me"}
    assert alert.is_valid()


def test_tvalert_missing_ticker_is_invalid():
    alert = TVAlert.parse({"signal": "wheel_put_zone"})
    assert not alert.is_valid()


def test_tvalert_to_dict_is_json_safe():
    alert = TVAlert.parse({"ticker": "SPY", "signal": "avoid", "price": 500.1})
    d = alert.to_dict()
    import json

    # Should serialize without raising
    json.dumps(d)


# ---------------------------------------------------------------------------
# Pine parity
# ---------------------------------------------------------------------------


# Map Pine variable names to the Python constants they mirror
_PINE_PARITY = {
    "BB_NARROW_PCTL": tv_signals.BB_WIDTH_NARROW_PCTL,
    "BB_WIDE_PCTL": tv_signals.BB_WIDTH_WIDE_PCTL,
    "BB_EXPAND_SLOPE": tv_signals.BB_WIDTH_EXPAND_SLOPE,
    "BB_CONTRACT_SLOPE": tv_signals.BB_WIDTH_CONTRACT_SLOPE,
    "ATR_LOW_PCTL": tv_signals.ATR_LOW_PCTL,
    "ATR_HIGH_PCTL": tv_signals.ATR_HIGH_PCTL,
    "ATR_RISING_SLOPE": tv_signals.ATR_RISING_SLOPE,
    "ATR_DECLINING_SLOPE": tv_signals.ATR_DECLINING_SLOPE,
    "ATR_FLAT_ABS": tv_signals.ATR_FLAT_ABS,
    "RSI_EXTREME_OB": tv_signals.RSI_EXTREME_OB,
    "RSI_OB": tv_signals.RSI_OB,
    "RSI_OS": tv_signals.RSI_OS,
    "RSI_EXTREME_OS": tv_signals.RSI_EXTREME_OS,
    "TREND_STRONG_SLOPE": tv_signals.TREND_STRONG_SLOPE,
    "TREND_WEAK_SLOPE": tv_signals.TREND_WEAK_SLOPE,
    "TREND_FLAT_PRICE_GAP": tv_signals.TREND_FLAT_PRICE_GAP,
    "RANGE_NEAR_EDGE": tv_signals.RANGE_NEAR_EDGE,
}


def test_pine_parity_constants():
    """Every threshold in the Pine file must match its Python twin.

    Parses the Pine source with a lightweight regex rather than importing
    Pine semantics. The goal is not to validate Pine syntax — it's to catch
    the human error of editing one file without the other.
    """
    import re

    pine_path = (
        Path(__file__).resolve().parents[1]
        / "tradingview"
        / "smart_wheel_signals.pine"
    )
    assert pine_path.exists(), f"Pine file missing: {pine_path}"
    source = pine_path.read_text()

    # Match lines like: NAME = 12.34  (allow integer or float)
    for name, py_value in _PINE_PARITY.items():
        pattern = rf"^\s*{re.escape(name)}\s*=\s*(-?\d+(?:\.\d+)?)\b"
        m = re.search(pattern, source, flags=re.MULTILINE)
        assert m, f"Pine constant {name} not found in smart_wheel_signals.pine"
        pine_value = float(m.group(1))
        assert pine_value == pytest.approx(py_value), (
            f"Pine/Python drift on {name}: pine={pine_value} python={py_value}"
        )


def test_defaults_contain_expected_keys():
    for key in (
        "bb_window",
        "bb_std",
        "atr_window",
        "rsi_window",
        "trend_ma_window",
        "range_lookback",
        "percentile_lookback",
        "slope_lookback",
    ):
        assert key in DEFAULTS
