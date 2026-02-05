"""
Market Regime Detection Module

Professional regime classification for adaptive strategy behavior:
- Volatility regime (low/normal/high/crisis)
- Trend regime (strong up/weak up/neutral/weak down/strong down)
- Volatility term structure (contango/backwardation)
- Mean reversion vs momentum state
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats


class VolatilityRegime(Enum):
    """Volatility environment classification."""
    LOW = "low"           # IV < 15th percentile historically
    NORMAL = "normal"     # 15th-70th percentile
    ELEVATED = "elevated" # 70th-90th percentile
    HIGH = "high"         # 90th-97th percentile
    CRISIS = "crisis"     # > 97th percentile (VIX > 35 equivalent)


class TrendRegime(Enum):
    """Price trend classification."""
    STRONG_UP = "strong_up"       # Consistent uptrend, high ADX
    WEAK_UP = "weak_up"           # Uptrend but weakening
    NEUTRAL = "neutral"           # Range-bound, no clear direction
    WEAK_DOWN = "weak_down"       # Downtrend but weakening
    STRONG_DOWN = "strong_down"   # Consistent downtrend, high ADX


class VolTermStructure(Enum):
    """Volatility term structure state."""
    STEEP_CONTANGO = "steep_contango"  # Front IV << back IV (normal, bullish)
    CONTANGO = "contango"              # Front IV < back IV
    FLAT = "flat"                      # Front IV ~ back IV
    BACKWARDATION = "backwardation"    # Front IV > back IV (bearish signal)
    STEEP_BACKWARDATION = "steep_backwardation"  # Front IV >> back IV (crisis)


@dataclass
class RegimeState:
    """Complete market regime classification."""
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    term_structure: VolTermStructure

    # Numeric values
    current_iv: float  # Current implied volatility
    iv_percentile: float  # Historical percentile (0-100)
    realized_vol: float  # Realized volatility
    iv_rv_spread: float  # IV - RV (volatility risk premium)

    trend_strength: float  # ADX or equivalent (0-100)
    trend_direction: float  # -1 to +1

    # Derived signals
    vol_regime_score: float  # -1 (low vol) to +1 (high vol)
    trend_regime_score: float  # -1 (bearish) to +1 (bullish)

    # Confidence
    regime_confidence: float  # 0-1, how clear is the regime

    def __str__(self) -> str:
        return (
            f"Market Regime:\n"
            f"  Volatility: {self.volatility_regime.value} (IV={self.current_iv:.1%}, "
            f"percentile={self.iv_percentile:.0f})\n"
            f"  Trend: {self.trend_regime.value} (strength={self.trend_strength:.0f}, "
            f"direction={self.trend_direction:+.2f})\n"
            f"  Term Structure: {self.term_structure.value}\n"
            f"  IV-RV Spread: {self.iv_rv_spread:+.1%}\n"
            f"  Confidence: {self.regime_confidence:.0%}"
        )

    @property
    def is_favorable_for_selling(self) -> bool:
        """Is current regime favorable for selling options?"""
        # Good to sell: elevated vol + contango + not strong downtrend
        vol_ok = self.volatility_regime in [
            VolatilityRegime.ELEVATED,
            VolatilityRegime.NORMAL
        ]
        term_ok = self.term_structure in [
            VolTermStructure.CONTANGO,
            VolTermStructure.STEEP_CONTANGO,
            VolTermStructure.FLAT
        ]
        trend_ok = self.trend_regime not in [TrendRegime.STRONG_DOWN]

        return vol_ok and term_ok and trend_ok

    @property
    def position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on regime.

        Returns 0.0 to 1.5 multiplier for base position size.
        """
        # Base on volatility regime
        vol_mult = {
            VolatilityRegime.LOW: 0.7,      # Reduce size (less premium)
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.ELEVATED: 1.2,  # Increase size (more premium)
            VolatilityRegime.HIGH: 0.8,      # Reduce size (more risk)
            VolatilityRegime.CRISIS: 0.3     # Minimal size
        }.get(self.volatility_regime, 1.0)

        # Adjust for trend
        trend_adj = {
            TrendRegime.STRONG_UP: 1.2,
            TrendRegime.WEAK_UP: 1.1,
            TrendRegime.NEUTRAL: 1.0,
            TrendRegime.WEAK_DOWN: 0.9,
            TrendRegime.STRONG_DOWN: 0.5
        }.get(self.trend_regime, 1.0)

        return vol_mult * trend_adj


class RegimeDetector:
    """
    Market regime detection and classification.

    Uses multiple indicators to classify current market state
    and provide adaptive signals for strategy adjustment.
    """

    def __init__(
        self,
        vol_lookback: int = 252,  # 1 year for vol percentile
        trend_lookback: int = 20,  # 20 days for trend
        rv_window: int = 20,  # 20 days for realized vol
    ):
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.rv_window = rv_window

        # Historical data storage
        self.iv_history: List[float] = []
        self.price_history: List[float] = []
        self.regime_history: List[RegimeState] = []

    def detect_regime(
        self,
        current_iv: float,
        prices: pd.Series,
        iv_history: Optional[pd.Series] = None,
        front_iv: Optional[float] = None,
        back_iv: Optional[float] = None
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            current_iv: Current implied volatility
            prices: Historical price series
            iv_history: Historical IV series (optional)
            front_iv: Front-month IV (for term structure)
            back_iv: Back-month IV (for term structure)

        Returns:
            RegimeState with full classification
        """
        # Volatility regime
        vol_regime, iv_percentile = self._classify_volatility(current_iv, iv_history)

        # Realized volatility
        realized_vol = self._calculate_realized_vol(prices)
        iv_rv_spread = current_iv - realized_vol

        # Trend regime
        trend_regime, trend_strength, trend_direction = self._classify_trend(prices)

        # Term structure
        term_structure = self._classify_term_structure(front_iv, back_iv)

        # Compute scores
        vol_score = self._vol_to_score(vol_regime, iv_percentile)
        trend_score = self._trend_to_score(trend_regime, trend_direction)

        # Confidence based on consistency
        confidence = self._calculate_confidence(
            prices, vol_regime, trend_regime
        )

        regime = RegimeState(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            term_structure=term_structure,
            current_iv=current_iv,
            iv_percentile=iv_percentile,
            realized_vol=realized_vol,
            iv_rv_spread=iv_rv_spread,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            vol_regime_score=vol_score,
            trend_regime_score=trend_score,
            regime_confidence=confidence
        )

        self.regime_history.append(regime)
        return regime

    def _classify_volatility(
        self,
        current_iv: float,
        iv_history: Optional[pd.Series] = None
    ) -> Tuple[VolatilityRegime, float]:
        """Classify volatility regime based on percentile."""
        if iv_history is not None and len(iv_history) > 30:
            percentile = stats.percentileofscore(iv_history, current_iv)
        elif self.iv_history:
            percentile = stats.percentileofscore(self.iv_history, current_iv)
        else:
            # Use absolute thresholds if no history
            if current_iv < 0.12:
                percentile = 10
            elif current_iv < 0.18:
                percentile = 40
            elif current_iv < 0.25:
                percentile = 70
            elif current_iv < 0.35:
                percentile = 90
            else:
                percentile = 98

        # Store for future reference
        self.iv_history.append(current_iv)
        if len(self.iv_history) > self.vol_lookback:
            self.iv_history = self.iv_history[-self.vol_lookback:]

        # Classify
        if percentile < 15:
            regime = VolatilityRegime.LOW
        elif percentile < 70:
            regime = VolatilityRegime.NORMAL
        elif percentile < 90:
            regime = VolatilityRegime.ELEVATED
        elif percentile < 97:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.CRISIS

        return regime, percentile

    def _calculate_realized_vol(self, prices: pd.Series) -> float:
        """Calculate realized volatility from price series."""
        if len(prices) < 2:
            return 0.20  # Default

        returns = prices.pct_change().dropna()
        if len(returns) < self.rv_window:
            window = len(returns)
        else:
            window = self.rv_window

        recent_returns = returns.tail(window)
        daily_vol = recent_returns.std()

        # Annualize
        return daily_vol * np.sqrt(252)

    def _classify_trend(
        self,
        prices: pd.Series
    ) -> Tuple[TrendRegime, float, float]:
        """
        Classify trend regime using ADX-like logic.

        Returns:
            Tuple of (regime, strength 0-100, direction -1 to +1)
        """
        if len(prices) < self.trend_lookback:
            return TrendRegime.NEUTRAL, 0, 0

        # Calculate trend metrics
        recent = prices.tail(self.trend_lookback)

        # Direction: regression slope normalized
        x = np.arange(len(recent))
        slope, _, r_value, _, _ = stats.linregress(x, recent)

        # Normalize slope by price level and period
        avg_price = recent.mean()
        normalized_slope = (slope * len(recent)) / avg_price

        # Direction: -1 to +1
        direction = np.clip(normalized_slope * 10, -1, 1)

        # Strength: R-squared * 100 (how consistent is the trend)
        strength = abs(r_value) * 100

        # ADX-like adjustment: also consider ATR ratio
        high_low_range = (recent.max() - recent.min()) / avg_price
        if high_low_range > 0:
            consistency = abs(normalized_slope) / high_low_range
            strength = strength * min(consistency, 1.5)

        strength = min(strength, 100)

        # Classify
        if direction > 0.3 and strength > 40:
            regime = TrendRegime.STRONG_UP
        elif direction > 0.1:
            regime = TrendRegime.WEAK_UP
        elif direction < -0.3 and strength > 40:
            regime = TrendRegime.STRONG_DOWN
        elif direction < -0.1:
            regime = TrendRegime.WEAK_DOWN
        else:
            regime = TrendRegime.NEUTRAL

        return regime, strength, direction

    def _classify_term_structure(
        self,
        front_iv: Optional[float],
        back_iv: Optional[float]
    ) -> VolTermStructure:
        """Classify volatility term structure."""
        if front_iv is None or back_iv is None:
            return VolTermStructure.FLAT

        if back_iv <= 0:
            return VolTermStructure.FLAT

        ratio = front_iv / back_iv

        if ratio < 0.85:
            return VolTermStructure.STEEP_CONTANGO
        elif ratio < 0.97:
            return VolTermStructure.CONTANGO
        elif ratio < 1.03:
            return VolTermStructure.FLAT
        elif ratio < 1.15:
            return VolTermStructure.BACKWARDATION
        else:
            return VolTermStructure.STEEP_BACKWARDATION

    def _vol_to_score(
        self,
        regime: VolatilityRegime,
        percentile: float
    ) -> float:
        """Convert volatility regime to -1 to +1 score."""
        # Map percentile to score
        # 0 percentile = -1 (very low vol)
        # 50 percentile = 0 (normal)
        # 100 percentile = +1 (very high vol)
        return (percentile - 50) / 50

    def _trend_to_score(
        self,
        regime: TrendRegime,
        direction: float
    ) -> float:
        """Convert trend regime to -1 to +1 score."""
        base_score = {
            TrendRegime.STRONG_UP: 0.8,
            TrendRegime.WEAK_UP: 0.4,
            TrendRegime.NEUTRAL: 0.0,
            TrendRegime.WEAK_DOWN: -0.4,
            TrendRegime.STRONG_DOWN: -0.8
        }.get(regime, 0)

        # Blend with actual direction
        return (base_score + direction) / 2

    def _calculate_confidence(
        self,
        prices: pd.Series,
        vol_regime: VolatilityRegime,
        trend_regime: TrendRegime
    ) -> float:
        """Calculate confidence in regime classification."""
        confidence = 0.5  # Base

        # More data = more confidence
        data_points = len(prices)
        if data_points > 100:
            confidence += 0.2
        elif data_points > 50:
            confidence += 0.1

        # Extreme regimes are clearer
        if vol_regime in [VolatilityRegime.LOW, VolatilityRegime.CRISIS]:
            confidence += 0.1
        if trend_regime in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
            confidence += 0.1

        # Check regime stability (if we have history)
        if len(self.regime_history) >= 5:
            recent_vol_regimes = [r.volatility_regime for r in self.regime_history[-5:]]
            recent_trend_regimes = [r.trend_regime for r in self.regime_history[-5:]]

            vol_stable = len(set(recent_vol_regimes)) <= 2
            trend_stable = len(set(recent_trend_regimes)) <= 2

            if vol_stable:
                confidence += 0.1
            if trend_stable:
                confidence += 0.1

        return min(confidence, 1.0)

    def get_strategy_adjustments(
        self,
        regime: RegimeState
    ) -> Dict[str, any]:
        """
        Get recommended strategy adjustments based on regime.

        Returns dict with adjustment parameters.
        """
        adjustments = {
            'position_size_mult': regime.position_size_multiplier,
            'delta_target': 0.30,  # Base target
            'dte_preference': 'normal',
            'profit_target_mult': 1.0,
            'stop_loss_mult': 1.0,
            'new_positions_allowed': True,
            'reason': []
        }

        # Volatility adjustments
        if regime.volatility_regime == VolatilityRegime.LOW:
            adjustments['delta_target'] = 0.35  # Closer to ATM for more premium
            adjustments['dte_preference'] = 'longer'
            adjustments['reason'].append("Low vol: targeting closer strikes, longer DTE")

        elif regime.volatility_regime == VolatilityRegime.ELEVATED:
            adjustments['delta_target'] = 0.25  # Farther OTM
            adjustments['profit_target_mult'] = 0.8  # Take profits earlier
            adjustments['reason'].append("Elevated vol: wider strikes, earlier profit-taking")

        elif regime.volatility_regime == VolatilityRegime.HIGH:
            adjustments['delta_target'] = 0.20
            adjustments['stop_loss_mult'] = 1.5  # Wider stops
            adjustments['reason'].append("High vol: far OTM, wider stops")

        elif regime.volatility_regime == VolatilityRegime.CRISIS:
            adjustments['new_positions_allowed'] = False
            adjustments['reason'].append("Crisis: no new positions")

        # Trend adjustments
        if regime.trend_regime == TrendRegime.STRONG_DOWN:
            adjustments['delta_target'] *= 0.7  # Even farther OTM
            adjustments['new_positions_allowed'] = False
            adjustments['reason'].append("Strong downtrend: defensive mode")

        elif regime.trend_regime == TrendRegime.STRONG_UP:
            adjustments['profit_target_mult'] = 0.7  # Quick profits
            adjustments['reason'].append("Strong uptrend: quick profit-taking")

        # Term structure adjustments
        if regime.term_structure in [VolTermStructure.BACKWARDATION,
                                      VolTermStructure.STEEP_BACKWARDATION]:
            adjustments['dte_preference'] = 'shorter'
            adjustments['reason'].append("Backwardation: prefer shorter DTE")

        return adjustments


def calculate_regime_signals(
    prices: pd.DataFrame,
    iv_column: str = 'iv',
    close_column: str = 'close'
) -> pd.DataFrame:
    """
    Calculate regime signals for entire price DataFrame.

    Adds columns for regime classification at each point in time.
    """
    detector = RegimeDetector()

    results = []
    for i in range(len(prices)):
        row = prices.iloc[i]

        # Get historical data up to this point
        hist_prices = prices[close_column].iloc[:i + 1]
        hist_iv = prices[iv_column].iloc[:i + 1] if iv_column in prices.columns else None

        current_iv = row[iv_column] if iv_column in prices.columns else 0.20

        regime = detector.detect_regime(
            current_iv=current_iv,
            prices=hist_prices,
            iv_history=hist_iv
        )

        results.append({
            'date': prices.index[i],
            'vol_regime': regime.volatility_regime.value,
            'trend_regime': regime.trend_regime.value,
            'iv_percentile': regime.iv_percentile,
            'trend_strength': regime.trend_strength,
            'trend_direction': regime.trend_direction,
            'position_mult': regime.position_size_multiplier,
            'favorable_for_selling': regime.is_favorable_for_selling
        })

    return pd.DataFrame(results).set_index('date')
