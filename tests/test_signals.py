"""Tests for signal generation module."""

import pytest
import numpy as np
import pandas as pd

from engine.signals import (
    SignalAggregator,
    Signal,
    CompositeSignal,
    SignalType,
    SignalStrength,
    IVRankSignal,
    TrendSignal,
    ProfitTargetSignal,
    StopLossSignal,
    DTESignal,
    EventFilterSignal,
    create_default_aggregator
)


class TestIVRankSignal:
    """Test IV rank signal generator."""

    def test_high_iv_rank_bullish(self):
        """High IV rank should be bullish for selling."""
        signal_gen = IVRankSignal(high_threshold=0.50)

        signal = signal_gen.generate({'iv_rank': 0.80})

        assert signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.WEAK_BUY]
        assert signal.value > 0

    def test_low_iv_rank_bearish(self):
        """Low IV rank should be bearish for selling."""
        signal_gen = IVRankSignal(low_threshold=0.20)

        signal = signal_gen.generate({'iv_rank': 0.10})

        assert signal.strength == SignalStrength.WEAK_SELL
        assert signal.value < 0

    def test_neutral_iv_rank(self):
        """Middle IV rank should be neutral."""
        signal_gen = IVRankSignal(high_threshold=0.50, low_threshold=0.20)

        signal = signal_gen.generate({'iv_rank': 0.35})

        assert signal.strength == SignalStrength.NEUTRAL
        assert signal.value == 0


class TestTrendSignal:
    """Test trend signal generator."""

    def test_uptrend_bullish(self):
        """Uptrend should be bullish."""
        signal_gen = TrendSignal()

        signal = signal_gen.generate({'trend_direction': 0.5})

        assert signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.WEAK_BUY]
        assert signal.value > 0

    def test_downtrend_bearish(self):
        """Downtrend should be bearish."""
        signal_gen = TrendSignal()

        signal = signal_gen.generate({'trend_direction': -0.5})

        assert signal.strength in [SignalStrength.STRONG_SELL, SignalStrength.WEAK_SELL]
        assert signal.value < 0

    def test_from_price_series(self):
        """Should compute trend from prices."""
        signal_gen = TrendSignal()

        # Uptrending prices
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        signal = signal_gen.generate({'prices': prices})

        assert signal.value > 0


class TestProfitTargetSignal:
    """Test profit target signal generator."""

    def test_profit_target_reached(self):
        """Should signal exit when profit target reached."""
        signal_gen = ProfitTargetSignal(target_pct=0.50)

        signal = signal_gen.generate({
            'entry_credit': 2.00,
            'current_value': 0.80  # 60% profit
        })

        assert signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.WEAK_BUY]
        assert "target reached" in signal.reason.lower() or "profit" in signal.reason.lower()

    def test_below_profit_target(self):
        """Should not signal exit when below target."""
        signal_gen = ProfitTargetSignal(target_pct=0.50)

        signal = signal_gen.generate({
            'entry_credit': 2.00,
            'current_value': 1.50  # Only 25% profit
        })

        assert signal.strength == SignalStrength.NEUTRAL


class TestStopLossSignal:
    """Test stop loss signal generator."""

    def test_stop_loss_triggered(self):
        """Should signal exit when stop triggered."""
        signal_gen = StopLossSignal(stop_multiplier=2.0)

        signal = signal_gen.generate({
            'entry_credit': 1.00,
            'current_value': 2.50  # 2.5x credit = stop triggered
        })

        assert signal.strength == SignalStrength.STRONG_BUY
        assert "stop" in signal.reason.lower()

    def test_within_stop(self):
        """Should not signal when within stop."""
        signal_gen = StopLossSignal(stop_multiplier=2.0)

        signal = signal_gen.generate({
            'entry_credit': 1.00,
            'current_value': 1.20  # 1.2x credit = OK
        })

        assert signal.strength == SignalStrength.NEUTRAL


class TestDTESignal:
    """Test DTE signal generator."""

    def test_exit_dte_reached(self):
        """Should signal exit at low DTE."""
        signal_gen = DTESignal(exit_dte=5)

        signal = signal_gen.generate({
            'dte': 3,
            'is_entry': False
        })

        assert signal.strength == SignalStrength.STRONG_BUY
        assert signal.signal_type == SignalType.EXIT

    def test_ideal_entry_dte(self):
        """Should signal entry at ideal DTE."""
        signal_gen = DTESignal(ideal_dte=35)

        signal = signal_gen.generate({
            'dte': 35,
            'is_entry': True
        })

        assert signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.WEAK_BUY]
        assert signal.signal_type == SignalType.ENTRY


class TestEventFilterSignal:
    """Test event filter signal generator."""

    def test_earnings_block(self):
        """Should block trade near earnings."""
        signal_gen = EventFilterSignal(earnings_buffer_days=5)

        signal = signal_gen.generate({
            'days_to_earnings': 3
        })

        assert signal.strength == SignalStrength.STRONG_SELL
        assert "block" in signal.reason.lower()

    def test_no_events(self):
        """Should not block when no events."""
        signal_gen = EventFilterSignal()

        signal = signal_gen.generate({
            'days_to_earnings': None,
            'days_to_fomc': None
        })

        assert signal.strength == SignalStrength.NEUTRAL


class TestSignalAggregator:
    """Test signal aggregation."""

    def test_evaluate_entry(self):
        """Should evaluate entry signals."""
        aggregator = SignalAggregator()

        context = {
            'iv_rank': 0.70,
            'trend_direction': 0.3,
            'dte': 35
        }

        composite = aggregator.evaluate_entry(context)

        assert isinstance(composite, CompositeSignal)
        assert len(composite.signals) > 0
        assert isinstance(composite.final_signal, SignalStrength)

    def test_evaluate_exit(self):
        """Should evaluate exit signals."""
        # Use lower threshold so profit target triggers action
        aggregator = SignalAggregator(exit_threshold=0.05)

        context = {
            'entry_credit': 2.00,
            'current_value': 0.80,  # 60% profit
            'dte': 20
        }

        composite = aggregator.evaluate_exit(context)

        assert isinstance(composite, CompositeSignal)
        # At least one exit signal should be actionable (profit target)
        actionable_signals = [s for s in composite.signals if s.is_actionable]
        assert len(actionable_signals) > 0
        # The profit target signal should be strong
        profit_signal = [s for s in composite.signals if 'profit' in s.name.lower()]
        assert len(profit_signal) > 0
        assert profit_signal[0].strength == SignalStrength.STRONG_BUY

    def test_filter_blocks_entry(self):
        """Filter signal should block entry."""
        aggregator = SignalAggregator()

        context = {
            'iv_rank': 0.90,  # Very favorable
            'trend_direction': 0.5,  # Uptrend
            'dte': 35,
            'days_to_earnings': 2  # But earnings coming!
        }

        composite = aggregator.evaluate_entry(context)

        assert not composite.action_recommended
        assert "block" in composite.explanation.lower()


class TestCreateDefaultAggregator:
    """Test default aggregator creation."""

    def test_creates_aggregator(self):
        """Should create working aggregator."""
        aggregator = create_default_aggregator()

        assert isinstance(aggregator, SignalAggregator)
        assert len(aggregator.entry_generators) > 0
        assert len(aggregator.exit_generators) > 0

    def test_default_aggregator_works(self):
        """Default aggregator should produce signals."""
        aggregator = create_default_aggregator()

        context = {
            'iv_rank': 0.60,
            'trend_direction': 0.2,
            'dte': 30,
            'entry_credit': 2.00,
            'current_value': 1.80
        }

        entry_signal = aggregator.evaluate_entry(context)
        exit_signal = aggregator.evaluate_exit(context)

        assert entry_signal is not None
        assert exit_signal is not None
