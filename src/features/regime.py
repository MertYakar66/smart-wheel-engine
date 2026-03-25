"""
Market Regime Detection - Adapt strategy to market conditions

Same strategy behaves VERY differently across regimes:
- Bull market: wheel prints money
- Bear market: constant assignment, drawdowns
- High vol: huge premiums but huge risk
- Low vol: small premiums, safe

This module classifies regimes for strategy adaptation.
"""

import numpy as np
import pandas as pd
from typing import Literal
from enum import IntEnum


class MarketRegime(IntEnum):
    """Market regime classifications."""
    CRISIS = -2
    BEAR = -1
    SIDEWAYS = 0
    BULL = 1
    EUPHORIA = 2


class VolRegime(IntEnum):
    """Volatility regime classifications."""
    EXTREMELY_LOW = 0
    LOW = 1
    NORMAL = 2
    ELEVATED = 3
    CRISIS = 4


class LiquidityRegime(IntEnum):
    """Liquidity regime classifications."""
    FROZEN = 0
    TIGHT = 1
    NORMAL = 2
    ABUNDANT = 3


class RegimeDetector:
    """
    Detect market regimes for strategy adaptation.

    Regimes are classified along multiple dimensions:
    1. Trend regime (bull/bear/sideways)
    2. Volatility regime (low/normal/high/crisis)
    3. Liquidity regime (tight/normal/abundant)

    Each regime has implications for wheel strategy:
    - Position sizing
    - Strike selection
    - DTE selection
    - Entry/exit timing
    """

    # === TREND REGIME ===

    @staticmethod
    def trend_regime(
        price: pd.Series,
        short_window: int = 21,
        long_window: int = 200,
        momentum_window: int = 63,
    ) -> pd.Series:
        """
        Classify trend regime based on moving averages and momentum.

        Returns:
            -2 = Crisis (sharp decline)
            -1 = Bear (downtrend)
             0 = Sideways (range-bound)
             1 = Bull (uptrend)
             2 = Euphoria (parabolic)
        """
        # Moving averages
        sma_short = price.rolling(short_window).mean()
        sma_long = price.rolling(long_window).mean()

        # Momentum
        momentum = price.pct_change(momentum_window)

        # Trend direction
        above_long = price > sma_long
        short_above_long = sma_short > sma_long

        # Classify
        regime = pd.Series(MarketRegime.SIDEWAYS, index=price.index)

        # Bull conditions
        bull_mask = above_long & short_above_long & (momentum > 0)
        regime[bull_mask] = MarketRegime.BULL

        # Euphoria conditions (> 20% in 63 days while already bull)
        euphoria_mask = bull_mask & (momentum > 0.20)
        regime[euphoria_mask] = MarketRegime.EUPHORIA

        # Bear conditions
        bear_mask = ~above_long & ~short_above_long & (momentum < 0)
        regime[bear_mask] = MarketRegime.BEAR

        # Crisis conditions (> -15% in 63 days)
        crisis_mask = bear_mask & (momentum < -0.15)
        regime[crisis_mask] = MarketRegime.CRISIS

        return regime

    @staticmethod
    def trend_strength(
        price: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """
        Measure trend strength (0-1).

        Uses linear regression R-squared.
        High R² = strong trend
        Low R² = choppy/sideways
        """
        def r_squared(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation ** 2 if not np.isnan(correlation) else 0

        return price.rolling(window).apply(r_squared, raw=True)

    # === VOLATILITY REGIME ===

    @staticmethod
    def vol_regime(
        rv: pd.Series,
        vix: pd.Series | None = None,
    ) -> pd.Series:
        """
        Classify volatility regime.

        Uses realized volatility and optionally VIX.
        Auto-detects whether input is in decimal (e.g., 0.20) or point (e.g., 20) format.

        Returns:
            0 = Extremely low (< 10%)
            1 = Low (10-15%)
            2 = Normal (15-25%)
            3 = Elevated (25-35%)
            4 = Crisis (> 35%)
        """
        # Use VIX if available, otherwise RV
        vol = vix if vix is not None else rv

        # Auto-detect scale: if median > 1, assume point format (e.g., VIX = 20)
        # and convert to decimal (0.20) for consistent thresholding
        vol_clean = vol.dropna()
        if len(vol_clean) > 0 and vol_clean.median() > 1.0:
            # Input is in point format (e.g., VIX = 20), convert to decimal
            vol = vol / 100.0

        regime = pd.Series(VolRegime.NORMAL, index=vol.index)

        regime[vol < 0.10] = VolRegime.EXTREMELY_LOW
        regime[(vol >= 0.10) & (vol < 0.15)] = VolRegime.LOW
        regime[(vol >= 0.15) & (vol < 0.25)] = VolRegime.NORMAL
        regime[(vol >= 0.25) & (vol < 0.35)] = VolRegime.ELEVATED
        regime[vol >= 0.35] = VolRegime.CRISIS

        return regime

    @staticmethod
    def vol_regime_percentile(
        rv: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """
        Vol regime as percentile (0-100).

        Where is current vol vs history?
        """
        def percentile_rank(x):
            if len(x) < 2:
                return np.nan
            return (x < x.iloc[-1]).sum() / (len(x) - 1) * 100

        return rv.rolling(window).apply(percentile_rank, raw=False)

    @staticmethod
    def vix_term_structure_regime(
        vix_spot: pd.Series,
        vix_futures: pd.Series,
    ) -> pd.Series:
        """
        VIX term structure regime.

        Contango (spot < futures) = Normal, complacent
        Backwardation (spot > futures) = Fear, stress

        Returns:
            1 = Steep contango (> 5% below)
            0 = Flat
           -1 = Backwardation (> 5% above)
            NaN = Invalid data (vix_spot <= 0)
        """
        # Guard against division by zero or invalid VIX values
        spread = (vix_futures - vix_spot) / vix_spot.replace(0, np.nan)

        # Initialize with NaN to preserve invalid entries
        regime = pd.Series(np.nan, index=spread.index)

        # Only classify where spread is valid (not NaN)
        valid = spread.notna()
        regime[valid & (spread > 0.05)] = 1   # Contango
        regime[valid & (spread < -0.05)] = -1  # Backwardation
        regime[valid & (spread >= -0.05) & (spread <= 0.05)] = 0  # Flat

        return regime

    # === LIQUIDITY REGIME ===

    @staticmethod
    def liquidity_regime(
        volume: pd.Series,
        spread: pd.Series | None = None,
        window: int = 21,
    ) -> pd.Series:
        """
        Classify liquidity regime based on volume and spreads.

        Returns:
            0 = Frozen (extremely low liquidity)
            1 = Tight (below normal)
            2 = Normal
            3 = Abundant (high liquidity)
        """
        # Volume relative to average
        vol_ratio = volume / volume.rolling(window).mean()

        regime = pd.Series(LiquidityRegime.NORMAL, index=volume.index)

        # Volume-based classification
        regime[vol_ratio < 0.5] = LiquidityRegime.TIGHT
        regime[vol_ratio < 0.25] = LiquidityRegime.FROZEN
        regime[vol_ratio > 1.5] = LiquidityRegime.ABUNDANT

        # If spreads available, adjust
        if spread is not None:
            spread_ratio = spread / spread.rolling(window).mean()
            # Wide spreads indicate poor liquidity
            regime[spread_ratio > 2.0] = LiquidityRegime.TIGHT
            regime[spread_ratio > 3.0] = LiquidityRegime.FROZEN

        return regime

    # === COMPOSITE REGIME ===

    @staticmethod
    def composite_regime_score(
        trend_regime: pd.Series,
        vol_regime: pd.Series,
        liquidity_regime: pd.Series,
    ) -> pd.Series:
        """
        Composite regime score for wheel strategy.

        Higher score = better environment for wheel
        Lower score = caution needed

        Returns:
            Score from -100 to +100
        """
        # Trend component: bull is good, bear is bad
        trend_score = trend_regime.map({
            MarketRegime.CRISIS: -40,
            MarketRegime.BEAR: -20,
            MarketRegime.SIDEWAYS: 10,  # Sideways is good for wheel!
            MarketRegime.BULL: 20,
            MarketRegime.EUPHORIA: 0,  # Euphoria is risky
        })

        # Vol component: elevated vol is good (more premium), but crisis is bad
        vol_score = vol_regime.map({
            VolRegime.EXTREMELY_LOW: -10,
            VolRegime.LOW: 0,
            VolRegime.NORMAL: 20,
            VolRegime.ELEVATED: 30,  # Best for selling premium
            VolRegime.CRISIS: -20,  # Too risky
        })

        # Liquidity component
        liq_score = liquidity_regime.map({
            LiquidityRegime.FROZEN: -30,
            LiquidityRegime.TIGHT: -10,
            LiquidityRegime.NORMAL: 10,
            LiquidityRegime.ABUNDANT: 20,
        })

        return trend_score + vol_score + liq_score

    # === STRATEGY ADJUSTMENTS ===

    @staticmethod
    def regime_position_scalar(
        composite_score: pd.Series,
    ) -> pd.Series:
        """
        Position size scalar based on regime.

        Returns:
            0.0 = No positions
            0.5 = Half size
            1.0 = Full size
            1.5 = Overweight (great conditions)
        """
        scalar = pd.Series(1.0, index=composite_score.index)

        scalar[composite_score < -50] = 0.0   # Crisis - no new positions
        scalar[(composite_score >= -50) & (composite_score < -20)] = 0.5
        scalar[(composite_score >= -20) & (composite_score < 20)] = 0.75
        scalar[(composite_score >= 20) & (composite_score < 50)] = 1.0
        scalar[composite_score >= 50] = 1.25  # Great conditions

        return scalar

    @staticmethod
    def regime_delta_adjustment(
        vol_regime: pd.Series,
        base_delta: float = 0.30,
    ) -> pd.Series:
        """
        Adjust target delta based on vol regime.

        High vol = use lower delta (more OTM, safer)
        Low vol = use higher delta (more ITM, more premium)
        """
        adjustment = vol_regime.map({
            VolRegime.EXTREMELY_LOW: 0.05,   # Go more ITM
            VolRegime.LOW: 0.03,
            VolRegime.NORMAL: 0.00,
            VolRegime.ELEVATED: -0.05,
            VolRegime.CRISIS: -0.10,  # Go more OTM for safety
        })

        return base_delta + adjustment

    @staticmethod
    def regime_dte_adjustment(
        vol_regime: pd.Series,
        base_dte: int = 30,
    ) -> pd.Series:
        """
        Adjust target DTE based on vol regime.

        High vol = shorter DTE (capture theta faster)
        Low vol = longer DTE (more time for premium)
        """
        adjustment = vol_regime.map({
            VolRegime.EXTREMELY_LOW: 15,    # Longer DTE
            VolRegime.LOW: 10,
            VolRegime.NORMAL: 0,
            VolRegime.ELEVATED: -7,
            VolRegime.CRISIS: -14,  # Shorter DTE
        })

        return base_dte + adjustment

    def compute_all(
        self,
        price: pd.Series,
        rv: pd.Series,
        volume: pd.Series,
        vix: pd.Series | None = None,
        vix_futures: pd.Series | None = None,
        spread: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute all regime features.

        Args:
            price: Price series
            rv: Realized volatility
            volume: Trading volume
            vix: VIX index (optional)
            vix_futures: VIX futures (optional)
            spread: Bid-ask spread (optional)

        Returns:
            DataFrame with all regime features
        """
        result = pd.DataFrame(index=price.index)

        # Trend regime
        result["trend_regime"] = self.trend_regime(price)
        result["trend_strength"] = self.trend_strength(price)

        # Vol regime
        result["vol_regime"] = self.vol_regime(rv, vix)
        result["vol_regime_percentile"] = self.vol_regime_percentile(rv)

        if vix is not None and vix_futures is not None:
            result["vix_term_regime"] = self.vix_term_structure_regime(vix, vix_futures)

        # Liquidity regime
        result["liquidity_regime"] = self.liquidity_regime(volume, spread)

        # Composite
        result["regime_score"] = self.composite_regime_score(
            result["trend_regime"],
            result["vol_regime"],
            result["liquidity_regime"],
        )

        # Strategy adjustments
        result["position_scalar"] = self.regime_position_scalar(result["regime_score"])
        result["target_delta_adj"] = self.regime_delta_adjustment(result["vol_regime"])
        result["target_dte_adj"] = self.regime_dte_adjustment(result["vol_regime"])

        return result
