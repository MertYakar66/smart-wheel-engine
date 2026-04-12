"""
Short Strangle Entry Timing Framework

Volatility lifecycle timing model for systematic strangle entry:
- Detects volatility expansion exhaustion and post-expansion stabilization
- Scores entry quality from 0-100 using technical volatility proxies
- Classifies regime (Compression → Expansion → Post-Expansion → Trend)
- Integrates with signal framework for automated trading decisions

Core principle: Enter short strangles when the underlying has ALREADY
experienced a volatility expansion and is transitioning into stabilization.
This captures overpriced implied volatility while realized risk is declining.

Architecture:
    Layer 1 (this module): Technical Regime Engine — entry timing using
    Bollinger, ATR, RSI, trend, and range metrics as volatility proxies.
    Layer 2 (this module): Options IV Overlay — IV rank, vol risk premium,
    VIX regime, and term structure via MarketDataConnector.

Reference:
    This is NOT a "technical strategy." It is a volatility timing model
    using technical proxies to identify the lifecycle phase of volatility.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.features.technical import TechnicalFeatures

# =============================================================================
# Volatility Lifecycle Regime
# =============================================================================


class VolatilityPhase(Enum):
    """Phase in the volatility lifecycle."""

    COMPRESSION = "compression"  # Low vol, narrow bands → AVOID
    EXPANSION = "expansion"  # Active move, widening bands → WAIT
    POST_EXPANSION = "post_expansion"  # Elevated but flattening → ENTER
    TREND = "trend"  # Persistent direction → AVOID symmetric
    UNKNOWN = "unknown"


@dataclass
class StrangleRegime:
    """Complete volatility lifecycle state for strangle timing."""

    phase: VolatilityPhase
    confidence: float  # 0-1

    # Component states
    bollinger_state: str  # "narrow", "expanding", "wide_flat", "wide_contracting"
    atr_state: str  # "low", "rising", "elevated_flat", "declining"
    rsi_state: str  # "neutral", "overbought", "oversold", "extreme_ob", "extreme_os"
    trend_state: str  # "strong_up", "strong_down", "weak", "flat"
    range_state: str  # "mid", "near_high", "near_low", "beyond_band"

    # Raw metrics
    bb_width: float = 0.0
    bb_width_percentile: float = 0.0
    bb_pct_b: float = 0.5
    atr_14: float = 0.0
    atr_slope: float = 0.0  # ΔATR over recent period
    atr_percentile: float = 0.0
    rsi_14: float = 50.0
    rsi_2: float = 50.0
    ma_slope_20: float = 0.0
    dist_to_high_20: float = 0.0
    dist_to_low_20: float = 0.0

    def __str__(self) -> str:
        return (
            f"Strangle Regime: {self.phase.value} (conf={self.confidence:.0%})\n"
            f"  BB: {self.bollinger_state} (width_pctl={self.bb_width_percentile:.0f}, %B={self.bb_pct_b:.2f})\n"
            f"  ATR: {self.atr_state} (slope={self.atr_slope:+.4f}, pctl={self.atr_percentile:.0f})\n"
            f"  RSI: {self.rsi_state} (14d={self.rsi_14:.0f}, 2d={self.rsi_2:.0f})\n"
            f"  Trend: {self.trend_state} (MA slope={self.ma_slope_20:+.4f})\n"
            f"  Range: {self.range_state} (high={self.dist_to_high_20:.2%}, low={self.dist_to_low_20:.2%})"
        )


@dataclass
class StrangleEntryScore:
    """Weighted scoring for strangle entry quality."""

    total_score: float  # 0-100
    recommendation: str  # "strong_entry", "conditional", "avoid"

    # Component scores (each 0-100)
    bollinger_score: float = 0.0
    atr_score: float = 0.0
    rsi_score: float = 0.0
    trend_score: float = 0.0
    range_score: float = 0.0

    # Weights used
    weights: dict = field(default_factory=dict)

    # Flags
    compression_warning: bool = False
    expansion_active: bool = False
    strong_trend_warning: bool = False

    regime: StrangleRegime | None = None

    def __str__(self) -> str:
        flags = []
        if self.compression_warning:
            flags.append("⚠ COMPRESSION")
        if self.expansion_active:
            flags.append("⚠ EXPANSION")
        if self.strong_trend_warning:
            flags.append("⚠ TREND")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return (
            f"Entry Score: {self.total_score:.0f}/100 → {self.recommendation}{flag_str}\n"
            f"  BB={self.bollinger_score:.0f} ATR={self.atr_score:.0f} "
            f"RSI={self.rsi_score:.0f} Trend={self.trend_score:.0f} Range={self.range_score:.0f}"
        )


# =============================================================================
# Strangle Entry Timing Engine
# =============================================================================


class StrangleTimingEngine:
    """
    Volatility lifecycle timing engine for short strangle entry.

    Analyzes OHLCV data to classify the volatility phase and score
    entry quality. Does NOT predict direction — focuses exclusively
    on the lifecycle of volatility.

    Usage:
        engine = StrangleTimingEngine()
        score = engine.score_entry(ohlcv_df)
        if score.total_score >= 80:
            print("High-quality entry window")
    """

    # Default component weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        "bollinger": 0.25,
        "atr": 0.20,
        "rsi": 0.15,
        "trend": 0.20,
        "range": 0.20,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        bb_window: int = 20,
        bb_std: float = 2.0,
        atr_window: int = 14,
        rsi_window: int = 14,
        rsi_short_window: int = 2,
        trend_ma_window: int = 20,
        range_lookback: int = 20,
        bb_width_lookback: int = 100,
        atr_slope_lookback: int = 5,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.rsi_short_window = rsi_short_window
        self.trend_ma_window = trend_ma_window
        self.range_lookback = range_lookback
        self.bb_width_lookback = bb_width_lookback
        self.atr_slope_lookback = atr_slope_lookback
        self.tech = TechnicalFeatures()

    def classify_regime(self, df: pd.DataFrame) -> StrangleRegime:
        """
        Classify the current volatility lifecycle phase.

        Args:
            df: OHLCV DataFrame (needs at least 100 rows for robust classification)

        Returns:
            StrangleRegime with phase classification and metrics
        """
        if len(df) < self.bb_width_lookback:
            return StrangleRegime(
                phase=VolatilityPhase.UNKNOWN,
                confidence=0.0,
                bollinger_state="unknown",
                atr_state="unknown",
                rsi_state="unknown",
                trend_state="unknown",
                range_state="unknown",
            )

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # --- Bollinger Band State ---
        upper, middle, lower = self.tech.bollinger_bands(close, self.bb_window, self.bb_std)
        bb_width = (upper - lower) / middle
        bb_width_current = bb_width.iloc[-1]
        bb_width_hist = bb_width.dropna().tail(self.bb_width_lookback)
        bb_width_pctl = (
            (bb_width_hist < bb_width_current).mean() * 100 if len(bb_width_hist) > 0 else 50
        )
        bb_pct_b = (
            (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            if (upper.iloc[-1] - lower.iloc[-1]) > 0
            else 0.5
        )

        # BB width slope (expanding vs contracting)
        bb_width_slope = (
            (bb_width.iloc[-1] - bb_width.iloc[-self.atr_slope_lookback]) / self.atr_slope_lookback
            if len(bb_width) >= self.atr_slope_lookback
            else 0
        )

        if bb_width_pctl < 20:
            bollinger_state = "narrow"
        elif bb_width_slope > 0.001:
            bollinger_state = "expanding"
        elif bb_width_pctl > 60 and bb_width_slope < -0.0005:
            bollinger_state = "wide_contracting"
        elif bb_width_pctl > 60:
            bollinger_state = "wide_flat"
        else:
            bollinger_state = "normal"

        # --- ATR State ---
        atr = self.tech.atr(high, low, close, self.atr_window)
        atr_current = atr.iloc[-1]
        atr_hist = atr.dropna().tail(self.bb_width_lookback)
        atr_pctl = (atr_hist < atr_current).mean() * 100 if len(atr_hist) > 0 else 50
        atr_slope = (
            (atr.iloc[-1] - atr.iloc[-self.atr_slope_lookback]) / atr.iloc[-self.atr_slope_lookback]
            if atr.iloc[-self.atr_slope_lookback] > 0 and len(atr) >= self.atr_slope_lookback
            else 0
        )

        if atr_pctl < 25:
            atr_state = "low"
        elif atr_slope > 0.05:
            atr_state = "rising"
        elif atr_pctl > 60 and abs(atr_slope) < 0.03:
            atr_state = "elevated_flat"
        elif atr_slope < -0.03:
            atr_state = "declining"
        else:
            atr_state = "normal"

        # --- RSI State ---
        rsi_14 = self.tech.rsi(close, self.rsi_window).iloc[-1]
        rsi_2 = self.tech.rsi(close, self.rsi_short_window).iloc[-1]

        if rsi_14 > 80 or rsi_2 > 95:
            rsi_state = "extreme_ob"
        elif rsi_14 > 70:
            rsi_state = "overbought"
        elif rsi_14 < 20 or rsi_2 < 5:
            rsi_state = "extreme_os"
        elif rsi_14 < 30:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

        # --- Trend State ---
        ma_20 = self.tech.sma(close, self.trend_ma_window)
        ma_slope = (
            (ma_20.iloc[-1] - ma_20.iloc[-self.atr_slope_lookback])
            / ma_20.iloc[-self.atr_slope_lookback]
            if ma_20.iloc[-self.atr_slope_lookback] > 0 and len(ma_20) >= self.atr_slope_lookback
            else 0
        )
        price_vs_ma = (
            (close.iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1] if ma_20.iloc[-1] > 0 else 0
        )

        if abs(ma_slope) > 0.02 and abs(price_vs_ma) < 0.02:
            trend_state = "strong_up" if ma_slope > 0 else "strong_down"
        elif abs(ma_slope) > 0.005:
            trend_state = "weak"
        else:
            trend_state = "flat"

        # --- Range State ---
        rolling_high = high.rolling(self.range_lookback).max()
        rolling_low = low.rolling(self.range_lookback).min()
        dist_to_high = (rolling_high.iloc[-1] - close.iloc[-1]) / close.iloc[-1]
        dist_to_low = (close.iloc[-1] - rolling_low.iloc[-1]) / close.iloc[-1]

        if bb_pct_b > 1.0 or bb_pct_b < 0.0:
            range_state = "beyond_band"
        elif dist_to_high < 0.01:
            range_state = "near_high"
        elif dist_to_low < 0.01:
            range_state = "near_low"
        else:
            range_state = "mid"

        # --- Phase Classification ---
        phase, confidence = self._classify_phase(
            bollinger_state,
            atr_state,
            rsi_state,
            trend_state,
            bb_width_pctl,
            atr_pctl,
            rsi_14,
            ma_slope,
        )

        return StrangleRegime(
            phase=phase,
            confidence=confidence,
            bollinger_state=bollinger_state,
            atr_state=atr_state,
            rsi_state=rsi_state,
            trend_state=trend_state,
            range_state=range_state,
            bb_width=bb_width_current,
            bb_width_percentile=bb_width_pctl,
            bb_pct_b=bb_pct_b,
            atr_14=atr_current,
            atr_slope=atr_slope,
            atr_percentile=atr_pctl,
            rsi_14=rsi_14,
            rsi_2=rsi_2,
            ma_slope_20=ma_slope,
            dist_to_high_20=dist_to_high,
            dist_to_low_20=dist_to_low,
        )

    def _classify_phase(
        self,
        bb_state: str,
        atr_state: str,
        rsi_state: str,
        trend_state: str,
        bb_pctl: float,
        atr_pctl: float,
        rsi_14: float,
        ma_slope: float,
    ) -> tuple[VolatilityPhase, float]:
        """Determine volatility lifecycle phase from component states."""
        # Compression: low vol, narrow bands
        if bb_state == "narrow" and atr_state == "low":
            return VolatilityPhase.COMPRESSION, 0.85

        # Expansion: active move, widening bands, rising ATR
        if bb_state == "expanding" and atr_state == "rising":
            return VolatilityPhase.EXPANSION, 0.80

        # Strong trend: persistent directional bias
        if trend_state in ("strong_up", "strong_down") and bb_pctl > 40:
            return VolatilityPhase.TREND, 0.75

        # Post-expansion (TARGET): elevated vol but flattening/declining
        post_exp_signals = 0
        if bb_state in ("wide_flat", "wide_contracting"):
            post_exp_signals += 1
        if atr_state in ("elevated_flat", "declining"):
            post_exp_signals += 1
        if rsi_state in ("overbought", "oversold", "extreme_ob", "extreme_os"):
            post_exp_signals += 1
        if trend_state in ("flat", "weak"):
            post_exp_signals += 1

        if post_exp_signals >= 3:
            return VolatilityPhase.POST_EXPANSION, min(0.90, 0.60 + post_exp_signals * 0.10)

        if post_exp_signals >= 2 and atr_pctl > 50:
            return VolatilityPhase.POST_EXPANSION, 0.60

        # If ATR is elevated and expanding, still in expansion
        if atr_state == "rising":
            return VolatilityPhase.EXPANSION, 0.60

        # Default: check if it's more like compression or unknown
        if bb_pctl < 30 and atr_pctl < 30:
            return VolatilityPhase.COMPRESSION, 0.55

        return VolatilityPhase.UNKNOWN, 0.30

    def score_entry(self, df: pd.DataFrame) -> StrangleEntryScore:
        """
        Score entry quality for a short strangle (0-100).

        Scores each component and weights them:
        - Bollinger state (25%): wide+flat/contracting = high score
        - ATR regime (20%): elevated but flattening = high score
        - RSI exhaustion (15%): extreme readings = high score
        - Trend filter (20%): flat/weak = high score
        - Range positioning (20%): near extremes = high score

        Args:
            df: OHLCV DataFrame

        Returns:
            StrangleEntryScore with total score and component breakdown
        """
        regime = self.classify_regime(df)

        # --- Bollinger Score ---
        if regime.bollinger_state == "wide_contracting":
            bb_score = 90.0
        elif regime.bollinger_state == "wide_flat":
            bb_score = 80.0
        elif regime.bollinger_state == "expanding":
            bb_score = 30.0  # Active expansion — wait
        elif regime.bollinger_state == "narrow":
            bb_score = 10.0  # Compression — avoid
        else:
            bb_score = 50.0

        # Bonus for price at/beyond bands (mean-reversion opportunity)
        if regime.bb_pct_b > 1.0 or regime.bb_pct_b < 0.0:
            bb_score = min(100, bb_score + 15)
        elif regime.bb_pct_b > 0.9 or regime.bb_pct_b < 0.1:
            bb_score = min(100, bb_score + 10)

        # --- ATR Score ---
        if regime.atr_state == "elevated_flat":
            atr_score = 85.0
        elif regime.atr_state == "declining":
            atr_score = 90.0  # Vol declining from elevated = ideal
        elif regime.atr_state == "rising":
            atr_score = 20.0  # Active expansion — wait
        elif regime.atr_state == "low":
            atr_score = 15.0  # Low vol = underpriced premium
        else:
            atr_score = 50.0

        # --- RSI Score ---
        rsi = regime.rsi_14
        if regime.rsi_state in ("extreme_ob", "extreme_os"):
            rsi_score = 90.0  # Strong exhaustion signal
        elif regime.rsi_state in ("overbought", "oversold"):
            rsi_score = 75.0
        else:
            # Neutral RSI — mild positive (no extreme move to fade)
            rsi_score = 40.0 + 20.0 * (1.0 - abs(rsi - 50) / 50)  # Higher near 50

        # --- Trend Score ---
        if regime.trend_state == "flat":
            trend_score = 90.0  # Range-bound = ideal for strangles
        elif regime.trend_state == "weak":
            trend_score = 65.0  # Acceptable
        elif regime.trend_state in ("strong_up", "strong_down"):
            trend_score = 15.0  # Strong trend = dangerous for symmetric
        else:
            trend_score = 50.0

        # --- Range Score ---
        if regime.range_state == "beyond_band":
            range_score = 85.0  # Price extended beyond bands
        elif regime.range_state in ("near_high", "near_low"):
            range_score = 75.0  # Near recent extreme
        elif regime.range_state == "mid":
            range_score = 45.0  # Middle of range — less edge
        else:
            range_score = 50.0

        # --- Weighted Total ---
        w = self.weights
        total = (
            bb_score * w["bollinger"]
            + atr_score * w["atr"]
            + rsi_score * w["rsi"]
            + trend_score * w["trend"]
            + range_score * w["range"]
        )

        # --- Recommendation ---
        if total >= 80:
            recommendation = "strong_entry"
        elif total >= 60:
            recommendation = "conditional"
        else:
            recommendation = "avoid"

        # --- Warning Flags ---
        compression_warning = regime.phase == VolatilityPhase.COMPRESSION
        expansion_active = regime.phase == VolatilityPhase.EXPANSION
        strong_trend = regime.trend_state in ("strong_up", "strong_down")

        # Override recommendation on critical warnings
        if compression_warning:
            recommendation = "avoid"
        if expansion_active:
            recommendation = "avoid" if total < 70 else "conditional"

        return StrangleEntryScore(
            total_score=total,
            recommendation=recommendation,
            bollinger_score=bb_score,
            atr_score=atr_score,
            rsi_score=rsi_score,
            trend_score=trend_score,
            range_score=range_score,
            weights=w,
            compression_warning=compression_warning,
            expansion_active=expansion_active,
            strong_trend_warning=strong_trend,
            regime=regime,
        )

    def scan_universe(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        min_score: float = 60.0,
    ) -> pd.DataFrame:
        """
        Scan multiple symbols for strangle entry opportunities.

        Args:
            ohlcv_dict: Dict of {ticker: OHLCV DataFrame}
            min_score: Minimum score to include in results

        Returns:
            DataFrame with ticker, score, regime, recommendation
        """
        results = []
        for ticker, df in ohlcv_dict.items():
            try:
                score = self.score_entry(df)
                if score.total_score >= min_score:
                    results.append(
                        {
                            "ticker": ticker,
                            "score": score.total_score,
                            "recommendation": score.recommendation,
                            "phase": score.regime.phase.value if score.regime else "unknown",
                            "bb_score": score.bollinger_score,
                            "atr_score": score.atr_score,
                            "rsi_score": score.rsi_score,
                            "trend_score": score.trend_score,
                            "range_score": score.range_score,
                            "rsi_14": score.regime.rsi_14 if score.regime else 0,
                            "bb_pct_b": score.regime.bb_pct_b if score.regime else 0.5,
                            "atr_percentile": score.regime.atr_percentile if score.regime else 0,
                            "compression_warning": score.compression_warning,
                            "trend_warning": score.strong_trend_warning,
                        }
                    )
            except Exception:
                continue

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values("score", ascending=False)
        return df_results

    def compute_historical_scores(
        self,
        df: pd.DataFrame,
        lookback_required: int = 100,
        last_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute entry scores for every date in the history.

        Useful for backtesting strangle entry timing.

        Args:
            df: OHLCV DataFrame
            lookback_required: Minimum rows needed before scoring
            last_n: If set, only compute scores for the last N rows
                (still uses full history for indicator warm-up).
                Without this the loop is O(N) in df length, which
                can take >60 s for multi-year Bloomberg histories.

        Returns:
            DataFrame with date index and score columns
        """
        scores = []
        start_idx = lookback_required
        if last_n is not None and last_n > 0:
            start_idx = max(start_idx, len(df) - int(last_n))

        # Cap the scoring window so each iteration is O(1) in history size
        # rather than O(i). 250 rows comfortably covers the longest rolling
        # indicator (200-day BB) plus headroom.
        window_cap = 260

        for i in range(start_idx, len(df)):
            window_start = max(0, i + 1 - window_cap)
            window = df.iloc[window_start : i + 1]
            try:
                score = self.score_entry(window)
                scores.append(
                    {
                        "date": df.index[i] if hasattr(df.index[i], "date") else df.index[i],
                        "score": score.total_score,
                        "recommendation": score.recommendation,
                        "phase": score.regime.phase.value if score.regime else "unknown",
                        "bb_score": score.bollinger_score,
                        "atr_score": score.atr_score,
                        "rsi_score": score.rsi_score,
                        "trend_score": score.trend_score,
                        "range_score": score.range_score,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(scores)


# =============================================================================
# Layer 2: Options IV Overlay
# =============================================================================


class StrangleTimingWithIV(StrangleTimingEngine):
    """
    Layer 2: Strangle timing with real IV data overlay.

    Enhances Layer 1 technical regime with:
    - IV rank/percentile from historical data
    - Vol risk premium (IV - RV spread)
    - VIX regime context
    - IV term structure (contango/backwardation)

    Replaces proxy-based scoring with data-driven metrics when
    Bloomberg data is available via MarketDataConnector.
    """

    def __init__(
        self,
        data_connector=None,
        weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(weights=weights, **kwargs)
        self._data_connector = data_connector

    @property
    def data_connector(self):
        """Lazy access to data connector; import deferred to avoid circular imports."""
        if self._data_connector is None:
            from engine.data_connector import MarketDataConnector

            self._data_connector = MarketDataConnector()
        return self._data_connector

    # --------------------------------------------------------------------- #
    #  IV quality multiplier
    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_iv_multiplier(
        iv_rank: float,
        vol_risk_premium: float,
        vix_level: float,
        vix_contango: bool,
    ) -> float:
        """
        Compute an IV quality multiplier for the Layer 1 score.

        Args:
            iv_rank: IV rank 0-100 (percentile of current IV vs 1-year range).
            vol_risk_premium: IV minus RV as a percentage (e.g. 5.0 means 5%).
            vix_level: Current VIX spot level.
            vix_contango: True when VIX futures curve is in contango
                          (front < back — normal, benign).

        Returns:
            Multiplier clamped to [0.7, 1.3].
        """
        multiplier = 1.0

        # Elevated IV — premium is rich
        if iv_rank > 70:
            multiplier += 0.25  # 0.15 for >50 + 0.10 for >70
        elif iv_rank > 50:
            multiplier += 0.15

        # Rich vol risk premium
        if vol_risk_premium > 5.0:
            multiplier += 0.10

        # VIX term structure
        if vix_contango:
            multiplier += 0.05

        # VIX crisis regime — too chaotic for short premium
        if vix_level > 30:
            multiplier -= 0.10

        # Low IV — premium is underpriced
        if iv_rank < 20:
            multiplier -= 0.20

        return float(np.clip(multiplier, 0.7, 1.3))

    # --------------------------------------------------------------------- #
    #  Overridden score_entry with IV overlay
    # --------------------------------------------------------------------- #

    def score_entry(self, df: pd.DataFrame, iv_data: dict | None = None) -> StrangleEntryScore:
        """
        Score entry quality, optionally enhanced with IV data.

        When *iv_data* is supplied it must contain:
            iv_rank (float 0-100), vol_risk_premium (float %),
            vix_level (float), vix_contango (bool).

        If *iv_data* is ``None``, behaviour is identical to Layer 1.

        Args:
            df: OHLCV DataFrame.
            iv_data: Optional dict with IV metrics.

        Returns:
            StrangleEntryScore (total_score adjusted by IV multiplier when
            iv_data is present).
        """
        layer1_score = super().score_entry(df)

        if iv_data is None:
            return layer1_score

        iv_rank = iv_data.get("iv_rank", 50.0)
        vol_risk_premium = iv_data.get("vol_risk_premium", 0.0)
        vix_level = iv_data.get("vix_level", 20.0)
        vix_contango = iv_data.get("vix_contango", True)

        multiplier = self._compute_iv_multiplier(
            iv_rank,
            vol_risk_premium,
            vix_level,
            vix_contango,
        )
        adjusted_total = float(np.clip(layer1_score.total_score * multiplier, 0, 100))

        # Re-derive recommendation from adjusted score
        if adjusted_total >= 80:
            recommendation = "strong_entry"
        elif adjusted_total >= 60:
            recommendation = "conditional"
        else:
            recommendation = "avoid"

        # Preserve critical-warning overrides from Layer 1
        if layer1_score.compression_warning:
            recommendation = "avoid"
        if layer1_score.expansion_active:
            recommendation = "avoid" if adjusted_total < 70 else "conditional"

        return StrangleEntryScore(
            total_score=adjusted_total,
            recommendation=recommendation,
            bollinger_score=layer1_score.bollinger_score,
            atr_score=layer1_score.atr_score,
            rsi_score=layer1_score.rsi_score,
            trend_score=layer1_score.trend_score,
            range_score=layer1_score.range_score,
            weights=layer1_score.weights,
            compression_warning=layer1_score.compression_warning,
            expansion_active=layer1_score.expansion_active,
            strong_trend_warning=layer1_score.strong_trend_warning,
            regime=layer1_score.regime,
        )

    # --------------------------------------------------------------------- #
    #  Convenience: score a single ticker with real data from connector
    # --------------------------------------------------------------------- #

    def score_entry_with_iv(
        self,
        ticker: str,
        as_of: str | None = None,
    ) -> StrangleEntryScore:
        """
        Load data from the connector and return an IV-enhanced entry score.

        Args:
            ticker: Equity ticker symbol (e.g. "AAPL").
            as_of: Optional ISO-8601 date string to pin the scoring window.

        Returns:
            StrangleEntryScore with IV overlay applied.
        """
        connector = self.data_connector

        # Fetch OHLCV (at least 200 bars for robust BB-width percentile)
        ohlcv = connector.get_ohlcv(ticker, as_of=as_of, lookback=200)

        # Fetch IV metrics
        iv_rank = connector.get_iv_rank(ticker, as_of=as_of)
        rv = connector.get_realized_vol(ticker, as_of=as_of)
        iv_current = connector.get_current_iv(ticker, as_of=as_of)
        vol_risk_premium = iv_current - rv if (iv_current is not None and rv is not None) else 0.0

        vix_level = connector.get_vix_level(as_of=as_of)
        vix_contango = connector.get_vix_contango(as_of=as_of)

        iv_data = {
            "iv_rank": iv_rank if iv_rank is not None else 50.0,
            "vol_risk_premium": vol_risk_premium,
            "vix_level": vix_level if vix_level is not None else 20.0,
            "vix_contango": vix_contango if vix_contango is not None else True,
        }

        score = self.score_entry(ohlcv, iv_data=iv_data)

        # Attach IV metadata onto the regime for downstream consumers
        if score.regime is not None:
            score.regime.iv_rank = iv_data["iv_rank"]
            score.regime.vol_risk_premium = iv_data["vol_risk_premium"]
            score.regime.vix_level = iv_data["vix_level"]
            score.regime.vix_contango = iv_data["vix_contango"]

        return score

    # --------------------------------------------------------------------- #
    #  Universe scan with real IV data
    # --------------------------------------------------------------------- #

    def scan_universe_with_iv(
        self,
        tickers: list[str] | None = None,
        as_of: str | None = None,
        min_score: float = 60.0,
    ) -> pd.DataFrame:
        """
        Scan multiple tickers using real data from the connector.

        Args:
            tickers: List of ticker symbols. If ``None``, the connector's
                     default universe is used.
            as_of: Optional ISO-8601 date string.
            min_score: Minimum adjusted score to include.

        Returns:
            DataFrame sorted descending by score with columns:
            ticker, score, iv_rank, vol_risk_premium, vix_level,
            vix_contango, phase, recommendation.
        """
        connector = self.data_connector

        if tickers is None:
            tickers = connector.get_universe()

        results: list[dict] = []
        for ticker in tickers:
            try:
                entry = self.score_entry_with_iv(ticker, as_of=as_of)
                iv_rank = getattr(entry.regime, "iv_rank", None)
                vol_premium = getattr(entry.regime, "vol_risk_premium", None)
                vix_level = getattr(entry.regime, "vix_level", None)
                vix_contango = getattr(entry.regime, "vix_contango", None)

                if entry.total_score >= min_score:
                    results.append(
                        {
                            "ticker": ticker,
                            "score": entry.total_score,
                            "iv_rank": iv_rank,
                            "vol_risk_premium": vol_premium,
                            "vix_level": vix_level,
                            "vix_contango": vix_contango,
                            "phase": entry.regime.phase.value if entry.regime else "unknown",
                            "recommendation": entry.recommendation,
                            "bb_score": entry.bollinger_score,
                            "atr_score": entry.atr_score,
                            "rsi_score": entry.rsi_score,
                            "trend_score": entry.trend_score,
                            "range_score": entry.range_score,
                            "compression_warning": entry.compression_warning,
                            "trend_warning": entry.strong_trend_warning,
                        }
                    )
            except Exception:
                continue

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values("score", ascending=False).reset_index(drop=True)
        return df_results
