"""Tests for regime detection module."""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from engine.regime_detector import (
    RegimeDetector,
    RegimeState,
    VolatilityRegime,
    TrendRegime,
    VolTermStructure,
    calculate_regime_signals
)


class TestVolatilityRegime:
    """Test volatility regime classification."""

    def test_low_volatility_regime(self):
        """Low IV should be classified as LOW regime."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.1 for i in range(100)])

        regime = detector.detect_regime(
            current_iv=0.10,  # Very low IV
            prices=prices
        )

        assert regime.volatility_regime == VolatilityRegime.LOW
        assert regime.iv_percentile < 20

    def test_high_volatility_regime(self):
        """High IV should be classified as HIGH or CRISIS."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.1 for i in range(100)])

        regime = detector.detect_regime(
            current_iv=0.45,  # High IV
            prices=prices
        )

        assert regime.volatility_regime in [
            VolatilityRegime.HIGH,
            VolatilityRegime.CRISIS
        ]

    def test_normal_volatility_regime(self):
        """Normal IV should be classified as NORMAL."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.1 for i in range(100)])

        regime = detector.detect_regime(
            current_iv=0.18,  # Normal IV
            prices=prices
        )

        assert regime.volatility_regime in [
            VolatilityRegime.NORMAL,
            VolatilityRegime.ELEVATED
        ]


class TestTrendRegime:
    """Test trend regime classification."""

    def test_uptrend_detection(self):
        """Rising prices should detect uptrend."""
        detector = RegimeDetector()
        # Clear uptrend
        prices = pd.Series([100 + i * 0.5 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.20,
            prices=prices
        )

        assert regime.trend_regime in [
            TrendRegime.STRONG_UP,
            TrendRegime.WEAK_UP
        ]
        assert regime.trend_direction > 0

    def test_downtrend_detection(self):
        """Falling prices should detect downtrend."""
        detector = RegimeDetector()
        # Clear downtrend
        prices = pd.Series([100 - i * 0.5 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.20,
            prices=prices
        )

        assert regime.trend_regime in [
            TrendRegime.STRONG_DOWN,
            TrendRegime.WEAK_DOWN
        ]
        assert regime.trend_direction < 0

    def test_neutral_regime(self):
        """Sideways prices should detect neutral."""
        detector = RegimeDetector()
        # Sideways with noise
        np.random.seed(42)
        prices = pd.Series([100 + np.random.randn() * 0.5 for _ in range(50)])

        regime = detector.detect_regime(
            current_iv=0.20,
            prices=prices
        )

        # Should be neutral or weak trend
        assert regime.trend_regime in [
            TrendRegime.NEUTRAL,
            TrendRegime.WEAK_UP,
            TrendRegime.WEAK_DOWN
        ]


class TestTermStructure:
    """Test term structure classification."""

    def test_contango(self):
        """Front IV < back IV = contango."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.1 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.20,
            prices=prices,
            front_iv=0.18,
            back_iv=0.22
        )

        assert regime.term_structure in [
            VolTermStructure.CONTANGO,
            VolTermStructure.STEEP_CONTANGO
        ]

    def test_backwardation(self):
        """Front IV > back IV = backwardation."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.1 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.30,
            prices=prices,
            front_iv=0.35,
            back_iv=0.25
        )

        assert regime.term_structure in [
            VolTermStructure.BACKWARDATION,
            VolTermStructure.STEEP_BACKWARDATION
        ]


class TestStrategyAdjustments:
    """Test strategy adjustment recommendations."""

    def test_favorable_for_selling(self):
        """Elevated vol + uptrend + contango = favorable."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.3 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.28,
            prices=prices,
            front_iv=0.26,
            back_iv=0.30
        )

        # Should be favorable or at least not unfavorable
        assert regime.position_size_multiplier >= 0.5

    def test_unfavorable_for_selling(self):
        """Crisis vol + downtrend = unfavorable."""
        detector = RegimeDetector()
        prices = pd.Series([100 - i * 1.0 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.55,  # Crisis-level IV
            prices=prices
        )

        # Position size should be reduced
        assert regime.position_size_multiplier < 0.8

    def test_adjustments_dict(self):
        """Should return adjustment dictionary."""
        detector = RegimeDetector()
        prices = pd.Series([100 + i * 0.1 for i in range(50)])

        regime = detector.detect_regime(
            current_iv=0.20,
            prices=prices
        )

        adjustments = detector.get_strategy_adjustments(regime)

        assert 'position_size_mult' in adjustments
        assert 'delta_target' in adjustments
        assert 'reason' in adjustments


class TestRegimeSignals:
    """Test regime signal calculation."""

    def test_calculate_regime_signals(self):
        """Should calculate signals for DataFrame."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'close': [100 + i * 0.1 for i in range(100)],
            'iv': [0.20 + 0.001 * i for i in range(100)]
        }, index=dates)

        signals = calculate_regime_signals(prices)

        assert len(signals) == len(prices)
        assert 'vol_regime' in signals.columns
        assert 'trend_regime' in signals.columns
        assert 'position_mult' in signals.columns
