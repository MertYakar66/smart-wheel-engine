"""
Signal Generation Framework

Professional signal framework for systematic options trading:
- Entry signals (when to open new positions)
- Exit signals (profit targets, stop losses, roll signals)
- Filter signals (regime, event, liquidity)
- Composite signal aggregation

Key principle: Signals should be systematic, testable, and explainable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import date, datetime
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class SignalType(Enum):
    """Types of trading signals."""
    ENTRY = "entry"         # Open new position
    EXIT = "exit"           # Close existing position
    ROLL = "roll"           # Roll position to new expiry
    ADJUST = "adjust"       # Adjust position size
    FILTER = "filter"       # Pass/fail filter


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_SELL = -2
    WEAK_SELL = -1
    NEUTRAL = 0
    WEAK_BUY = 1
    STRONG_BUY = 2


@dataclass
class Signal:
    """Individual signal output."""
    name: str
    signal_type: SignalType
    strength: SignalStrength
    value: float              # Numeric value (-1 to 1 typically)
    confidence: float = 1.0   # 0 to 1
    metadata: Dict = field(default_factory=dict)
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        """Check if signal suggests action."""
        return self.strength not in [SignalStrength.NEUTRAL]

    @property
    def direction(self) -> int:
        """Get direction: -1 (sell/close), 0 (hold), +1 (buy/open)."""
        return self.strength.value

    def __str__(self) -> str:
        return f"{self.name}: {self.strength.name} ({self.value:.2f}) - {self.reason}"


@dataclass
class CompositeSignal:
    """Aggregated signal from multiple sources."""
    signals: List[Signal]
    final_signal: SignalStrength
    final_value: float
    final_confidence: float
    action_recommended: bool
    explanation: str

    def __str__(self) -> str:
        return (
            f"Composite: {self.final_signal.name} "
            f"(value={self.final_value:.2f}, conf={self.final_confidence:.0%})\n"
            f"Action: {'YES' if self.action_recommended else 'NO'}\n"
            f"Reason: {self.explanation}"
        )


class SignalGenerator(ABC):
    """Base class for signal generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal generator name."""
        pass

    @property
    @abstractmethod
    def signal_type(self) -> SignalType:
        """Type of signal generated."""
        pass

    @abstractmethod
    def generate(self, context: Dict) -> Signal:
        """Generate signal from context."""
        pass


class IVRankSignal(SignalGenerator):
    """
    Signal based on IV Rank.

    High IV rank = good time to sell premium.
    """

    @property
    def name(self) -> str:
        return "IV_Rank"

    @property
    def signal_type(self) -> SignalType:
        return SignalType.ENTRY

    def __init__(
        self,
        high_threshold: float = 0.50,  # Above 50th percentile = favorable
        low_threshold: float = 0.20    # Below 20th = unfavorable
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def generate(self, context: Dict) -> Signal:
        """
        Generate signal from IV rank.

        Context should contain:
        - iv_rank: Current IV percentile (0-1)
        """
        iv_rank = context.get('iv_rank', 0.5)

        if iv_rank >= self.high_threshold:
            if iv_rank >= 0.70:
                strength = SignalStrength.STRONG_BUY
            else:
                strength = SignalStrength.WEAK_BUY
            value = (iv_rank - self.high_threshold) / (1 - self.high_threshold)
            reason = f"IV rank {iv_rank:.0%} above threshold - favorable for selling"

        elif iv_rank <= self.low_threshold:
            strength = SignalStrength.WEAK_SELL
            value = -(self.low_threshold - iv_rank) / self.low_threshold
            reason = f"IV rank {iv_rank:.0%} below threshold - unfavorable for selling"

        else:
            strength = SignalStrength.NEUTRAL
            value = 0
            reason = f"IV rank {iv_rank:.0%} in neutral zone"

        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            strength=strength,
            value=value,
            confidence=0.8,
            metadata={'iv_rank': iv_rank},
            reason=reason
        )


class TrendSignal(SignalGenerator):
    """
    Signal based on price trend.

    Strong downtrend = avoid selling puts.
    """

    @property
    def name(self) -> str:
        return "Trend"

    @property
    def signal_type(self) -> SignalType:
        return SignalType.FILTER

    def __init__(
        self,
        lookback_days: int = 20,
        strength_threshold: float = 0.3
    ):
        self.lookback_days = lookback_days
        self.strength_threshold = strength_threshold

    def generate(self, context: Dict) -> Signal:
        """
        Generate trend signal.

        Context should contain:
        - prices: Recent price series
        - trend_direction: -1 to 1 (pre-computed) OR computed from prices
        """
        if 'trend_direction' in context:
            direction = context['trend_direction']
        elif 'prices' in context:
            prices = context['prices']
            if len(prices) >= 2:
                returns = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                direction = np.clip(returns * 10, -1, 1)
            else:
                direction = 0
        else:
            direction = 0

        if direction < -self.strength_threshold:
            strength = SignalStrength.STRONG_SELL
            value = direction
            reason = f"Strong downtrend ({direction:.2f}) - avoid selling puts"
        elif direction < 0:
            strength = SignalStrength.WEAK_SELL
            value = direction
            reason = f"Weak downtrend ({direction:.2f}) - caution"
        elif direction > self.strength_threshold:
            strength = SignalStrength.STRONG_BUY
            value = direction
            reason = f"Strong uptrend ({direction:.2f}) - favorable"
        elif direction > 0:
            strength = SignalStrength.WEAK_BUY
            value = direction
            reason = f"Weak uptrend ({direction:.2f}) - neutral to favorable"
        else:
            strength = SignalStrength.NEUTRAL
            value = 0
            reason = "No clear trend"

        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            strength=strength,
            value=value,
            confidence=0.7,
            metadata={'trend_direction': direction},
            reason=reason
        )


class ProfitTargetSignal(SignalGenerator):
    """
    Exit signal when profit target reached.
    """

    @property
    def name(self) -> str:
        return "Profit_Target"

    @property
    def signal_type(self) -> SignalType:
        return SignalType.EXIT

    def __init__(
        self,
        target_pct: float = 0.50  # 50% of max profit
    ):
        self.target_pct = target_pct

    def generate(self, context: Dict) -> Signal:
        """
        Generate profit target signal.

        Context should contain:
        - current_pnl_pct: Current P&L as % of max profit
        - entry_credit: Premium collected at entry
        - current_value: Current option value
        """
        entry_credit = context.get('entry_credit', 0)
        current_value = context.get('current_value', 0)

        if entry_credit > 0:
            pnl_pct = (entry_credit - current_value) / entry_credit
        else:
            pnl_pct = context.get('current_pnl_pct', 0)

        if pnl_pct >= self.target_pct:
            strength = SignalStrength.STRONG_BUY  # Strong signal to exit
            value = (pnl_pct - self.target_pct) / (1 - self.target_pct)
            reason = f"Profit target reached: {pnl_pct:.0%} >= {self.target_pct:.0%}"
        elif pnl_pct >= self.target_pct * 0.8:
            strength = SignalStrength.WEAK_BUY
            value = pnl_pct / self.target_pct - 0.8
            reason = f"Approaching profit target: {pnl_pct:.0%}"
        else:
            strength = SignalStrength.NEUTRAL
            value = 0
            reason = f"P&L {pnl_pct:.0%} below target {self.target_pct:.0%}"

        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            strength=strength,
            value=value,
            confidence=1.0,  # Objective signal
            metadata={'pnl_pct': pnl_pct, 'target_pct': self.target_pct},
            reason=reason
        )


class StopLossSignal(SignalGenerator):
    """
    Exit signal when stop loss breached.
    """

    @property
    def name(self) -> str:
        return "Stop_Loss"

    @property
    def signal_type(self) -> SignalType:
        return SignalType.EXIT

    def __init__(
        self,
        stop_multiplier: float = 2.0  # Stop at 2x premium (100% loss)
    ):
        self.stop_multiplier = stop_multiplier

    def generate(self, context: Dict) -> Signal:
        """
        Generate stop loss signal.

        Context should contain:
        - entry_credit: Premium collected
        - current_value: Current option value
        """
        entry_credit = context.get('entry_credit', 0)
        current_value = context.get('current_value', 0)

        if entry_credit > 0:
            loss_multiple = current_value / entry_credit
        else:
            loss_multiple = 1.0

        if loss_multiple >= self.stop_multiplier:
            strength = SignalStrength.STRONG_BUY  # Urgent exit
            value = 1.0
            reason = f"STOP LOSS: Current value {loss_multiple:.1f}x entry credit"
        elif loss_multiple >= self.stop_multiplier * 0.8:
            strength = SignalStrength.WEAK_BUY
            value = (loss_multiple - self.stop_multiplier * 0.8) / (self.stop_multiplier * 0.2)
            reason = f"Approaching stop: {loss_multiple:.1f}x credit"
        else:
            strength = SignalStrength.NEUTRAL
            value = 0
            reason = f"Within risk tolerance: {loss_multiple:.1f}x credit"

        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            strength=strength,
            value=value,
            confidence=1.0,
            metadata={'loss_multiple': loss_multiple},
            reason=reason
        )


class DTESignal(SignalGenerator):
    """
    Signal based on days to expiration.
    """

    @property
    def name(self) -> str:
        return "DTE"

    @property
    def signal_type(self) -> SignalType:
        return SignalType.EXIT

    def __init__(
        self,
        exit_dte: int = 5,      # Exit at 5 DTE
        ideal_dte: int = 35,    # Ideal entry DTE
        max_dte: int = 60       # Max DTE for entry
    ):
        self.exit_dte = exit_dte
        self.ideal_dte = ideal_dte
        self.max_dte = max_dte

    def generate(self, context: Dict) -> Signal:
        """
        Generate DTE signal.

        Context should contain:
        - dte: Days to expiration
        - is_entry: Whether evaluating for entry or exit
        """
        dte = context.get('dte', 30)
        is_entry = context.get('is_entry', False)

        if is_entry:
            # Entry evaluation
            if self.ideal_dte - 10 <= dte <= self.ideal_dte + 10:
                strength = SignalStrength.STRONG_BUY
                value = 1.0
                reason = f"Ideal DTE range: {dte} days"
            elif 20 <= dte <= self.max_dte:
                strength = SignalStrength.WEAK_BUY
                value = 0.5
                reason = f"Acceptable DTE: {dte} days"
            else:
                strength = SignalStrength.WEAK_SELL
                value = -0.5
                reason = f"DTE {dte} outside preferred range"
        else:
            # Exit evaluation
            if dte <= self.exit_dte:
                strength = SignalStrength.STRONG_BUY  # Exit
                value = 1.0
                reason = f"DTE {dte} at/below exit threshold {self.exit_dte}"
            elif dte <= self.exit_dte * 2:
                strength = SignalStrength.WEAK_BUY
                value = (self.exit_dte * 2 - dte) / self.exit_dte
                reason = f"Approaching exit DTE: {dte} days remaining"
            else:
                strength = SignalStrength.NEUTRAL
                value = 0
                reason = f"DTE {dte} - no exit pressure"

        return Signal(
            name=self.name,
            signal_type=SignalType.ENTRY if is_entry else SignalType.EXIT,
            strength=strength,
            value=value,
            confidence=1.0,
            metadata={'dte': dte, 'is_entry': is_entry},
            reason=reason
        )


class EventFilterSignal(SignalGenerator):
    """
    Filter signal based on upcoming events.
    """

    @property
    def name(self) -> str:
        return "Event_Filter"

    @property
    def signal_type(self) -> SignalType:
        return SignalType.FILTER

    def __init__(
        self,
        earnings_buffer_days: int = 5,
        fomc_buffer_days: int = 2
    ):
        self.earnings_buffer_days = earnings_buffer_days
        self.fomc_buffer_days = fomc_buffer_days

    def generate(self, context: Dict) -> Signal:
        """
        Generate event filter signal.

        Context should contain:
        - days_to_earnings: Days until earnings (None if no upcoming)
        - days_to_fomc: Days until FOMC (None if no upcoming)
        - expiry_date: Option expiration date
        """
        days_to_earnings = context.get('days_to_earnings')
        days_to_fomc = context.get('days_to_fomc')

        reasons = []
        block = False

        if days_to_earnings is not None:
            if days_to_earnings <= self.earnings_buffer_days:
                block = True
                reasons.append(f"Earnings in {days_to_earnings} days")

        if days_to_fomc is not None:
            if days_to_fomc <= self.fomc_buffer_days:
                reasons.append(f"FOMC in {days_to_fomc} days")
                # FOMC is warning, not block

        if block:
            strength = SignalStrength.STRONG_SELL
            value = -1.0
            reason = "BLOCKED: " + ", ".join(reasons)
        elif reasons:
            strength = SignalStrength.WEAK_SELL
            value = -0.5
            reason = "CAUTION: " + ", ".join(reasons)
        else:
            strength = SignalStrength.NEUTRAL
            value = 0
            reason = "No blocking events"

        return Signal(
            name=self.name,
            signal_type=self.signal_type,
            strength=strength,
            value=value,
            confidence=1.0,
            metadata={
                'days_to_earnings': days_to_earnings,
                'days_to_fomc': days_to_fomc
            },
            reason=reason
        )


class SignalAggregator:
    """
    Aggregate multiple signals into trading decision.
    """

    def __init__(
        self,
        entry_generators: Optional[List[SignalGenerator]] = None,
        exit_generators: Optional[List[SignalGenerator]] = None,
        filter_generators: Optional[List[SignalGenerator]] = None,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.5
    ):
        self.entry_generators = entry_generators or [
            IVRankSignal(),
            TrendSignal(),
            DTESignal()
        ]
        self.exit_generators = exit_generators or [
            ProfitTargetSignal(),
            StopLossSignal(),
            DTESignal()
        ]
        self.filter_generators = filter_generators or [
            EventFilterSignal()
        ]
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def evaluate_entry(self, context: Dict) -> CompositeSignal:
        """Evaluate whether to enter a position."""
        context['is_entry'] = True
        signals = []

        # Check filters first
        for gen in self.filter_generators:
            signal = gen.generate(context)
            signals.append(signal)
            if signal.strength == SignalStrength.STRONG_SELL:
                return CompositeSignal(
                    signals=signals,
                    final_signal=SignalStrength.STRONG_SELL,
                    final_value=-1.0,
                    final_confidence=1.0,
                    action_recommended=False,
                    explanation=f"Blocked by filter: {signal.reason}"
                )

        # Generate entry signals
        for gen in self.entry_generators:
            signal = gen.generate(context)
            signals.append(signal)

        # Aggregate
        return self._aggregate_signals(signals, self.entry_threshold, "entry")

    def evaluate_exit(self, context: Dict) -> CompositeSignal:
        """Evaluate whether to exit a position."""
        context['is_entry'] = False
        signals = []

        for gen in self.exit_generators:
            signal = gen.generate(context)
            signals.append(signal)

        return self._aggregate_signals(signals, self.exit_threshold, "exit")

    def _aggregate_signals(
        self,
        signals: List[Signal],
        threshold: float,
        action_type: str
    ) -> CompositeSignal:
        """Aggregate signals with weighted average."""
        if not signals:
            return CompositeSignal(
                signals=[],
                final_signal=SignalStrength.NEUTRAL,
                final_value=0,
                final_confidence=0,
                action_recommended=False,
                explanation="No signals to aggregate"
            )

        # Weight by confidence
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            weighted_value = 0
        else:
            weighted_value = sum(s.value * s.confidence for s in signals) / total_weight

        avg_confidence = np.mean([s.confidence for s in signals])

        # Determine final signal
        if weighted_value >= 0.6:
            final_signal = SignalStrength.STRONG_BUY
        elif weighted_value >= 0.2:
            final_signal = SignalStrength.WEAK_BUY
        elif weighted_value <= -0.6:
            final_signal = SignalStrength.STRONG_SELL
        elif weighted_value <= -0.2:
            final_signal = SignalStrength.WEAK_SELL
        else:
            final_signal = SignalStrength.NEUTRAL

        # Determine action
        action_recommended = weighted_value >= threshold

        # Build explanation
        contributing = [s for s in signals if s.is_actionable]
        if contributing:
            explanation = "; ".join([s.reason for s in contributing[:3]])
        else:
            explanation = "No strong signals"

        return CompositeSignal(
            signals=signals,
            final_signal=final_signal,
            final_value=weighted_value,
            final_confidence=avg_confidence,
            action_recommended=action_recommended,
            explanation=explanation
        )


def create_default_aggregator() -> SignalAggregator:
    """Create aggregator with default signal generators."""
    return SignalAggregator(
        entry_generators=[
            IVRankSignal(high_threshold=0.50, low_threshold=0.20),
            TrendSignal(lookback_days=20),
            DTESignal(exit_dte=5, ideal_dte=35)
        ],
        exit_generators=[
            ProfitTargetSignal(target_pct=0.50),
            StopLossSignal(stop_multiplier=2.0),
            DTESignal(exit_dte=5)
        ],
        filter_generators=[
            EventFilterSignal(earnings_buffer_days=5)
        ]
    )
