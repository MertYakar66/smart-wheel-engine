"""
Volatility Edge Features - The core of wheel strategy alpha

This is WHERE THE EDGE LIVES for selling options.

Key insight: You profit when IV > RV (volatility risk premium)
This module quantifies that edge.
"""


import numpy as np
import pandas as pd


class VolatilityEdge:
    """
    Compute volatility mispricing features.

    The wheel strategy profits from:
    1. Selling expensive volatility (IV > RV)
    2. Capturing theta decay
    3. Avoiding blow-ups (regime awareness)

    This module quantifies when options are mispriced.
    """

    @staticmethod
    def iv_rv_spread(iv: pd.Series, rv: pd.Series) -> pd.Series:
        """
        IV - RV spread (volatility risk premium).

        Positive = Options are expensive (good for selling)
        Negative = Options are cheap (bad for selling)
        Zero = Fair value

        This is your PRIMARY edge signal.
        """
        return iv - rv

    @staticmethod
    def iv_rv_ratio(iv: pd.Series, rv: pd.Series) -> pd.Series:
        """
        IV / RV ratio.

        > 1.0 = Options expensive
        < 1.0 = Options cheap
        = 1.0 = Fair value

        Ratio form is better for cross-asset comparison.
        """
        return iv / rv.replace(0, np.nan)

    @staticmethod
    def iv_rv_zscore(
        iv: pd.Series,
        rv: pd.Series,
        window: int = 63,
    ) -> pd.Series:
        """
        Z-score of IV-RV spread vs historical.

        High z-score = Unusually expensive (strong sell signal)
        Low z-score = Unusually cheap (avoid selling)
        """
        spread = iv - rv
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        # Guard against division by zero (std == 0 when no variance)
        return (spread - mean) / std.replace(0, np.nan)

    @staticmethod
    def forward_rv_vs_iv(
        iv: pd.Series,
        forward_rv: pd.Series,
    ) -> pd.Series:
        """
        Compare IV at time T with REALIZED vol over next N days.

        This is the TRUE measure of whether options were mispriced.
        Use for backtesting and label generation.

        Positive = You would have profited (IV was too high)
        Negative = You would have lost (IV was too low)
        """
        return iv - forward_rv

    @staticmethod
    def vrp_percentile(
        iv: pd.Series,
        rv: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """
        Volatility risk premium percentile.

        Where does current VRP (IV-RV spread) sit vs its own history?

        Interpretation:
        - > 80 = VRP is unusually high (rich premium, strong sell signal)
        - 50-80 = Above average VRP (good for selling)
        - 20-50 = Below average VRP (cautious)
        - < 20 = VRP is low (options are cheap, avoid selling)

        This is more robust than raw spread because it normalizes
        for different volatility regimes.
        """
        spread = iv - rv

        def percentile_rank(x):
            if len(x) < 2:
                return np.nan
            current = x.iloc[-1]
            # Proper percentile: count values strictly less than current
            below = (x.iloc[:-1] < current).sum()
            return below / (len(x) - 1) * 100

        return spread.rolling(window).apply(percentile_rank, raw=False)

    @staticmethod
    def vol_regime(
        rv: pd.Series,
        thresholds: tuple[float, float] = (0.15, 0.30),
    ) -> pd.Series:
        """
        Volatility regime classification.

        Args:
            rv: Realized volatility (annualized)
            thresholds: (low_threshold, high_threshold)

        Returns:
            0 = Low vol regime (< 15%)
            1 = Normal vol regime (15-30%)
            2 = High vol regime (> 30%)
        """
        low, high = thresholds
        regime = pd.Series(1, index=rv.index)  # Default normal
        regime[rv < low] = 0   # Low vol
        regime[rv > high] = 2  # High vol
        return regime

    @staticmethod
    def vol_regime_transition(
        rv: pd.Series,
        thresholds: tuple[float, float] = (0.15, 0.30),
    ) -> pd.Series:
        """
        Detect regime transitions.

        Returns:
            1 = Just entered higher regime
           -1 = Just entered lower regime
            0 = No transition
        """
        regime = VolatilityEdge.vol_regime(rv, thresholds)
        return regime.diff()

    @staticmethod
    def mean_reversion_signal(
        iv: pd.Series,
        long_window: int = 252,
        short_window: int = 21,
    ) -> pd.Series:
        """
        IV mean reversion signal.

        When short-term IV >> long-term mean, expect reversion down.
        Good for timing CSP entries.

        Returns z-score (positive = IV elevated, expect drop)
        """
        long_mean = iv.rolling(long_window).mean()
        long_std = iv.rolling(long_window).std()
        short_mean = iv.rolling(short_window).mean()

        return (short_mean - long_mean) / long_std

    @staticmethod
    def iv_term_premium(
        iv_30d: pd.Series,
        iv_60d: pd.Series,
        iv_90d: pd.Series,
    ) -> pd.Series:
        """
        Term premium in IV curve.

        Normal: longer term = higher IV (positive premium)
        Inverted: shorter term = higher IV (negative premium)

        Inverted curve often signals stress/opportunity.
        """
        # Average slope per 30 days
        slope_30_60 = (iv_60d - iv_30d) / 30
        slope_60_90 = (iv_90d - iv_60d) / 30
        return (slope_30_60 + slope_60_90) / 2

    @staticmethod
    def variance_swap_rate(
        iv_30d: pd.Series,
        rv_21d: pd.Series,
    ) -> pd.Series:
        """
        Simplified variance swap rate.

        Approximates the fair price of variance.
        Compare to IV^2 to find mispricing.
        """
        # IV is sqrt of expected variance
        # Compare IV^2 to RV^2
        return iv_30d ** 2 - rv_21d ** 2

    @staticmethod
    def edge_score(
        iv: pd.Series,
        rv: pd.Series,
        iv_rank: pd.Series,
        vrp_percentile: pd.Series,
    ) -> pd.Series:
        """
        Composite edge score for selling premium.

        Combines three signals with theoretical justification:

        1. VRP Percentile (40% weight):
           - Where is current IV-RV spread vs history?
           - High percentile = unusually rich premium = strong edge

        2. IV Rank (35% weight):
           - Where is IV relative to its own range?
           - High IV rank = options expensive vs own history

        3. IV/RV Ratio (25% weight):
           - Is IV correctly pricing forward RV?
           - Empirically, IV > RV ~85% of the time (VRP)
           - Ratio > 1.0 = options overpriced = edge exists

        Returns 0-100 score:
        - 80+ = Strong edge, consider larger position
        - 60-80 = Good edge, standard position
        - 40-60 = Moderate edge, smaller position
        - <40 = Weak/no edge, avoid or reduce

        This is your PRIMARY decision input for wheel entries.
        """
        # Component 1: VRP Percentile (already 0-100)
        vrp_component = vrp_percentile.clip(0, 100)

        # Component 2: IV Rank (already 0-100)
        iv_rank_component = iv_rank.clip(0, 100)

        # Component 3: IV/RV Ratio
        # Transform ratio to 0-100 scale using sigmoid-like function
        # Typical range: 0.8 to 1.5, center at 1.1 (median VRP)
        ratio = iv / rv.replace(0, np.nan)
        # Logistic transform: maps (0.7, 1.5) roughly to (10, 90)
        # Center at 1.1 (typical median ratio)
        ratio_normalized = 100 / (1 + np.exp(-5 * (ratio - 1.1)))
        ratio_component = ratio_normalized.clip(0, 100)

        # Weighted average with empirically-derived weights
        # VRP percentile most important (historical context)
        # IV rank second (own-history context)
        # Ratio provides cross-sectional signal
        score = (
            0.40 * vrp_component +
            0.35 * iv_rank_component +
            0.25 * ratio_component
        )

        return score

    def compute_all(
        self,
        df: pd.DataFrame,
        iv_col: str = "atm_iv",
        rv_col: str = "rv_21d",
        iv_30d_col: str | None = "iv_30d",
        iv_60d_col: str | None = "iv_60d",
        iv_90d_col: str | None = "iv_90d",
    ) -> pd.DataFrame:
        """
        Compute all volatility edge features.

        Args:
            df: DataFrame with IV and RV data

        Returns:
            DataFrame with edge features added
        """
        result = df.copy()

        iv = result[iv_col]
        rv = result[rv_col]

        # Core VRP metrics
        result["iv_rv_spread"] = self.iv_rv_spread(iv, rv)
        result["iv_rv_ratio"] = self.iv_rv_ratio(iv, rv)
        result["iv_rv_zscore"] = self.iv_rv_zscore(iv, rv)
        result["vrp_percentile"] = self.vrp_percentile(iv, rv)

        # Regime
        result["vol_regime"] = self.vol_regime(rv)
        result["vol_regime_transition"] = self.vol_regime_transition(rv)

        # Mean reversion
        result["iv_mean_reversion"] = self.mean_reversion_signal(iv)

        # Term structure (if available)
        if iv_30d_col and iv_60d_col and iv_90d_col:
            if all(col in result.columns for col in [iv_30d_col, iv_60d_col, iv_90d_col]):
                result["iv_term_premium"] = self.iv_term_premium(
                    result[iv_30d_col],
                    result[iv_60d_col],
                    result[iv_90d_col],
                )

        # Composite score
        if "iv_rank" in result.columns:
            result["edge_score"] = self.edge_score(
                iv, rv, result["iv_rank"], result["vrp_percentile"]
            )

        return result
