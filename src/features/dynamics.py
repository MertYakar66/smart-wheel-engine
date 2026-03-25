"""
Options Dynamics Features - Change-based signals

This module computes CHANGE features, not levels.
Markets react to changes, not states.

Key insight: ΔX is more predictive than X
"""

import numpy as np
import pandas as pd


class OptionsDynamics:
    """
    Compute change-based options features.

    These are the features that actually predict:
    - ΔOI (open interest changes)
    - ΔIV (implied volatility changes)
    - Δskew (put-call skew changes)
    - Δterm structure (term slope changes)
    """

    @staticmethod
    def oi_change(oi: pd.Series, periods: int = 1) -> pd.Series:
        """Daily change in open interest."""
        return oi.diff(periods)

    @staticmethod
    def oi_change_pct(oi: pd.Series, periods: int = 1) -> pd.Series:
        """Percentage change in open interest."""
        return oi.pct_change(periods)

    @staticmethod
    def oi_acceleration(oi: pd.Series) -> pd.Series:
        """Second derivative of OI (change in the change)."""
        return oi.diff().diff()

    @staticmethod
    def iv_change(iv: pd.Series, periods: int = 1) -> pd.Series:
        """Daily change in implied volatility (absolute)."""
        return iv.diff(periods)

    @staticmethod
    def iv_change_pct(iv: pd.Series, periods: int = 1) -> pd.Series:
        """Percentage change in IV."""
        return iv.pct_change(periods)

    @staticmethod
    def iv_velocity(iv: pd.Series, window: int = 5) -> pd.Series:
        """
        IV velocity: rate of IV change over window.

        Positive = IV expanding
        Negative = IV contracting
        """
        return iv.diff(window) / window

    @staticmethod
    def iv_acceleration(iv: pd.Series) -> pd.Series:
        """IV acceleration: change in IV velocity."""
        return iv.diff().diff()

    @staticmethod
    def skew_change(
        put_iv: pd.Series,
        call_iv: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """
        Change in put-call skew.

        Increasing skew = puts getting relatively more expensive
        Decreasing skew = calls catching up or puts cheapening
        """
        skew = put_iv - call_iv
        return skew.diff(periods)

    @staticmethod
    def term_structure_change(
        iv_front: pd.Series,
        iv_back: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """
        Change in term structure slope.

        iv_front: Near-term IV (e.g., 30d)
        iv_back: Far-term IV (e.g., 90d)

        Flattening = front IV rising relative to back
        Steepening = back IV rising relative to front
        """
        slope = iv_back - iv_front
        return slope.diff(periods)

    @staticmethod
    def term_structure_regime(
        iv_front: pd.Series,
        iv_back: pd.Series,
    ) -> pd.Series:
        """
        Term structure regime classification.

        Returns:
            1 = Contango (normal, front < back)
            0 = Flat
           -1 = Backwardation (inverted, front > back)
        """
        spread = iv_back - iv_front
        threshold = 0.02  # 2% threshold for "flat"

        regime = pd.Series(0, index=spread.index)
        regime[spread > threshold] = 1   # Contango
        regime[spread < -threshold] = -1  # Backwardation
        return regime

    @staticmethod
    def volume_oi_ratio_change(
        volume: pd.Series,
        oi: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """
        Change in volume/OI ratio.

        High ratio = new positions being opened
        Increasing = acceleration of positioning
        """
        ratio = volume / oi.replace(0, np.nan)
        return ratio.diff(periods)

    @staticmethod
    def put_call_ratio_change(
        put_volume: pd.Series,
        call_volume: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """Change in put/call volume ratio."""
        ratio = put_volume / call_volume.replace(0, np.nan)
        return ratio.diff(periods)

    @staticmethod
    def oi_price_divergence(
        oi_change: pd.Series,
        price_change: pd.Series,
    ) -> pd.Series:
        """
        OI-Price divergence signal.

        OI up + Price up = Strong bullish (new longs)
        OI up + Price down = Strong bearish (new shorts)
        OI down + Price up = Weak bullish (short covering)
        OI down + Price down = Weak bearish (long liquidation)

        Returns divergence score (-2 to +2)
        """
        oi_direction = np.sign(oi_change)
        price_direction = np.sign(price_change)

        # Aligned = strong signal
        # Diverged = weak signal
        return oi_direction + price_direction

    def compute_all(
        self,
        df: pd.DataFrame,
        iv_col: str = "atm_iv",
        call_oi_col: str = "call_oi",
        put_oi_col: str = "put_oi",
        call_vol_col: str = "call_volume",
        put_vol_col: str = "put_volume",
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Compute all dynamics features.

        Args:
            df: DataFrame with options data

        Returns:
            DataFrame with dynamics features added
        """
        result = df.copy()

        # Total OI
        total_oi = result[call_oi_col] + result[put_oi_col]
        total_volume = result[call_vol_col] + result[put_vol_col]

        # OI dynamics
        result["oi_change_1d"] = self.oi_change(total_oi, 1)
        result["oi_change_5d"] = self.oi_change(total_oi, 5)
        result["oi_change_pct_1d"] = self.oi_change_pct(total_oi, 1)
        result["oi_acceleration"] = self.oi_acceleration(total_oi)

        # Call/Put OI changes separately
        result["call_oi_change"] = self.oi_change(result[call_oi_col])
        result["put_oi_change"] = self.oi_change(result[put_oi_col])

        # IV dynamics
        if iv_col in result.columns:
            result["iv_change_1d"] = self.iv_change(result[iv_col], 1)
            result["iv_change_5d"] = self.iv_change(result[iv_col], 5)
            result["iv_velocity_5d"] = self.iv_velocity(result[iv_col], 5)
            result["iv_acceleration"] = self.iv_acceleration(result[iv_col])

        # Volume/OI dynamics
        result["vol_oi_ratio_change"] = self.volume_oi_ratio_change(
            total_volume, total_oi
        )

        # Put/Call ratio dynamics
        result["pc_ratio_change"] = self.put_call_ratio_change(
            result[put_vol_col], result[call_vol_col]
        )

        # OI-Price divergence
        if price_col in result.columns:
            result["oi_price_divergence"] = self.oi_price_divergence(
                result["oi_change_1d"],
                result[price_col].diff()
            )

        return result
