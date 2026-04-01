"""
Event Volatility Features - Earnings and macro event modeling

Events are where wheel strategy can win or lose big:
- Earnings: IV ramps up, then crushes
- Macro: FOMC, CPI cause vol spikes

This module models event behavior for optimal positioning.
"""

import numpy as np
import pandas as pd


class EventVolatility:
    """
    Model event-driven volatility behavior.

    Key patterns:
    1. Pre-event IV ramp
    2. Post-event IV crush
    3. Price gap distribution
    4. Event surprise magnitude

    Use these to decide:
    - Avoid selling into events (high gap risk)
    - Sell after events (capture crush)
    - Size positions around events
    """

    @staticmethod
    def days_to_event(
        dates: pd.DatetimeIndex,
        event_dates: pd.Series,
    ) -> pd.Series:
        """
        Calculate days until next event for each date.

        Vectorized implementation using searchsorted for O(N log M) complexity
        instead of O(N * M) loop-based approach.

        Returns:
            Series with days to next event (0 on event day, NaN after last event)
        """
        if len(event_dates) == 0:
            return pd.Series(np.nan, index=dates)

        # Convert to numpy arrays for vectorized operations
        dates_arr = pd.to_datetime(dates).values.astype("datetime64[D]")
        events_arr = pd.to_datetime(event_dates).sort_values().values.astype("datetime64[D]")

        # Find index of next event for each date using searchsorted
        # searchsorted returns index where element would be inserted to maintain order
        # 'left' means: index where date would go, so events_arr[idx] >= date
        next_event_idx = np.searchsorted(events_arr, dates_arr, side="left")

        # Initialize result with NaN
        result = np.full(len(dates), np.nan)

        # For valid indices (not past last event), compute days difference
        valid_mask = next_event_idx < len(events_arr)
        result[valid_mask] = (
            (events_arr[next_event_idx[valid_mask]] - dates_arr[valid_mask])
            .astype("timedelta64[D]")
            .astype(float)
        )

        return pd.Series(result, index=dates)

    @staticmethod
    def days_since_event(
        dates: pd.DatetimeIndex,
        event_dates: pd.Series,
    ) -> pd.Series:
        """
        Calculate days since last event for each date.

        Vectorized implementation using searchsorted for O(N log M) complexity
        instead of O(N * M) loop-based approach.

        Returns:
            Series with days since last event (0 on event day, NaN before first event)
        """
        if len(event_dates) == 0:
            return pd.Series(np.nan, index=dates)

        # Convert to numpy arrays for vectorized operations
        dates_arr = pd.to_datetime(dates).values.astype("datetime64[D]")
        events_arr = pd.to_datetime(event_dates).sort_values().values.astype("datetime64[D]")

        # Find index where date would be inserted (right side)
        # This gives us: events_arr[idx-1] <= date < events_arr[idx]
        # So the last event on or before date is at index (idx - 1)
        next_event_idx = np.searchsorted(events_arr, dates_arr, side="right")

        # The previous event is at index (next_event_idx - 1)
        prev_event_idx = next_event_idx - 1

        # Initialize result with NaN
        result = np.full(len(dates), np.nan)

        # For valid indices (not before first event), compute days difference
        valid_mask = prev_event_idx >= 0
        result[valid_mask] = (
            (dates_arr[valid_mask] - events_arr[prev_event_idx[valid_mask]])
            .astype("timedelta64[D]")
            .astype(float)
        )

        return pd.Series(result, index=dates)

    @staticmethod
    def iv_ramp(
        iv: pd.Series,
        days_to_event: pd.Series,
        ramp_window: int = 10,
    ) -> pd.Series:
        """
        Measure IV ramp into event.

        IV typically increases as event approaches.

        Returns:
            IV change over ramp window (positive = ramping up)
        """
        # Only measure when event is approaching
        mask = (days_to_event > 0) & (days_to_event <= ramp_window)
        ramp = iv.diff(ramp_window)
        ramp[~mask] = np.nan
        return ramp

    @staticmethod
    def iv_crush(
        iv: pd.Series,
        days_since_event: pd.Series,
        crush_window: int = 3,
    ) -> pd.Series:
        """
        Measure IV crush after event.

        IV typically drops sharply after event resolves.

        Returns:
            IV change post-event (negative = crush)
        """
        # Only measure right after event
        mask = (days_since_event >= 0) & (days_since_event <= crush_window)
        crush = iv.diff(crush_window)
        crush[~mask] = np.nan
        return crush

    @staticmethod
    def historical_iv_crush_magnitude(
        pre_event_iv: pd.Series,
        post_event_iv: pd.Series,
    ) -> dict:
        """
        Analyze historical IV crush patterns.

        Returns:
            Dict with crush statistics
        """
        crush = pre_event_iv - post_event_iv
        crush_pct = crush / pre_event_iv

        return {
            "avg_crush_abs": crush.mean(),
            "avg_crush_pct": crush_pct.mean(),
            "median_crush_pct": crush_pct.median(),
            "max_crush_pct": crush_pct.max(),
            "min_crush_pct": crush_pct.min(),
            "std_crush_pct": crush_pct.std(),
        }

    @staticmethod
    def price_gap(
        close: pd.Series,
        open_next: pd.Series,
    ) -> pd.Series:
        """
        Calculate overnight price gap.

        For earnings: gap between close before and open after announcement.
        """
        return (open_next - close) / close

    @staticmethod
    def historical_gap_distribution(
        gaps: pd.Series,
    ) -> dict:
        """
        Analyze historical gap distribution around events.

        Returns:
            Dict with gap statistics
        """
        abs_gaps = gaps.abs()

        return {
            "avg_gap_abs": abs_gaps.mean(),
            "median_gap_abs": abs_gaps.median(),
            "avg_gap_signed": gaps.mean(),
            "std_gap": gaps.std(),
            "max_gap_up": gaps.max(),
            "max_gap_down": gaps.min(),
            "pct_gap_up": (gaps > 0).mean(),
            "pct_gap_down": (gaps < 0).mean(),
            "pct_gap_large": (abs_gaps > 0.05).mean(),  # > 5% gap
        }

    @staticmethod
    def implied_vs_realized_move(
        implied_move: pd.Series,
        actual_move: pd.Series,
    ) -> pd.Series:
        """
        Compare implied move (from straddle) vs actual move.

        > 1.0 = Options overstated move (good for sellers)
        < 1.0 = Options understated move (bad for sellers)

        This is your POST-EVENT edge measure.
        """
        return implied_move / actual_move.abs().replace(0, np.nan)

    @staticmethod
    def event_premium_yield(
        iv_pre: pd.Series,
        iv_post: pd.Series,
        dte: int = 30,
    ) -> pd.Series:
        """
        Estimate premium yield from selling through event.

        Captures the IV crush component of premium decay.
        """
        # Approximate premium from IV difference
        # Premium ≈ Spot * IV * sqrt(T) * 0.4 (rough ATM approx)
        crush = iv_pre - iv_post
        time_factor = np.sqrt(dte / 365)
        premium_captured = crush * time_factor * 0.4

        return premium_captured

    @staticmethod
    def event_zone_flag(
        days_to_event: pd.Series,
        danger_zone: int = 7,
        opportunity_zone: int = 3,
    ) -> pd.Series:
        """
        Flag position relative to event.

        Returns:
            -1 = Danger zone (avoid new positions)
             0 = Normal
             1 = Opportunity zone (post-event, sell the crush)
        """
        flag = pd.Series(0, index=days_to_event.index)

        # Danger zone: event within N days
        flag[days_to_event <= danger_zone] = -1
        flag[days_to_event == 0] = -1  # Event day

        # Opportunity zone: right after event (use days_since)
        # This needs days_since_event, handled separately

        return flag

    @staticmethod
    def surprise_magnitude(
        actual: pd.Series,
        expected: pd.Series,
    ) -> pd.Series:
        """
        Calculate surprise magnitude for events.

        For earnings: (actual - estimate) / |estimate|

        Large surprises = large gaps
        """
        return (actual - expected) / expected.abs().replace(0, np.nan)

    @staticmethod
    def surprise_direction_streak(
        surprises: pd.Series,
    ) -> pd.Series:
        """
        Track streak of positive/negative surprises.

        Companies that consistently beat/miss tend to continue.
        """
        direction = np.sign(surprises)

        # Count consecutive same-direction surprises
        streak = pd.Series(0, index=surprises.index)
        current_streak = 0
        current_direction = 0

        for i, d in enumerate(direction):
            if pd.isna(d):
                continue
            if d == current_direction:
                current_streak += 1
            else:
                current_streak = 1
                current_direction = d
            streak.iloc[i] = current_streak * current_direction

        return streak

    @staticmethod
    def post_event_drift(
        returns: pd.Series,
        days_since_event: pd.Series,
        drift_window: int = 10,
    ) -> pd.Series:
        """
        Post-event price drift.

        Captures continuation after initial gap.
        PEAD (Post-Earnings Announcement Drift) is well-documented.
        """
        mask = (days_since_event >= 1) & (days_since_event <= drift_window)
        cumulative_return = returns.rolling(drift_window).sum()
        drift = cumulative_return.copy()
        drift[~mask] = np.nan
        return drift

    def compute_earnings_features(
        self,
        df: pd.DataFrame,
        earnings_dates: pd.Series,
        iv_col: str = "atm_iv",
        close_col: str = "close",
        open_col: str = "open",
    ) -> pd.DataFrame:
        """
        Compute all earnings-related features.

        Args:
            df: DataFrame with price and IV data
            earnings_dates: Series of earnings announcement dates
            iv_col: Column name for IV
            close_col: Column name for close price
            open_col: Column name for open price

        Returns:
            DataFrame with earnings features
        """
        result = df.copy()
        dates = pd.DatetimeIndex(result.index)

        # Days to/from earnings
        result["days_to_earnings"] = self.days_to_event(dates, earnings_dates)
        result["days_since_earnings"] = self.days_since_event(dates, earnings_dates)

        # IV dynamics around earnings
        result["earnings_iv_ramp"] = self.iv_ramp(
            result[iv_col], result["days_to_earnings"], ramp_window=10
        )
        result["earnings_iv_crush"] = self.iv_crush(
            result[iv_col], result["days_since_earnings"], crush_window=3
        )

        # Gap analysis
        result["overnight_gap"] = self.price_gap(result[close_col], result[open_col].shift(-1))

        # Event zone
        result["earnings_zone"] = self.event_zone_flag(
            result["days_to_earnings"], danger_zone=7, opportunity_zone=3
        )

        # Post-event drift
        returns = result[close_col].pct_change()
        result["post_earnings_drift"] = self.post_event_drift(
            returns, result["days_since_earnings"], drift_window=10
        )

        return result

    def compute_macro_event_features(
        self,
        df: pd.DataFrame,
        event_dates: pd.Series,
        event_type: str,
        iv_col: str = "atm_iv",
    ) -> pd.DataFrame:
        """
        Compute features for macro events (FOMC, CPI, NFP).

        Args:
            df: DataFrame with price and IV data
            event_dates: Series of event dates
            event_type: Type of event (for column naming)
            iv_col: Column name for IV

        Returns:
            DataFrame with macro event features
        """
        result = df.copy()
        dates = pd.DatetimeIndex(result.index)

        col_prefix = event_type.lower()

        # Days to/from event
        result[f"days_to_{col_prefix}"] = self.days_to_event(dates, event_dates)
        result[f"days_since_{col_prefix}"] = self.days_since_event(dates, event_dates)

        # IV dynamics
        result[f"{col_prefix}_iv_ramp"] = self.iv_ramp(
            result[iv_col], result[f"days_to_{col_prefix}"], ramp_window=5
        )
        result[f"{col_prefix}_iv_crush"] = self.iv_crush(
            result[iv_col], result[f"days_since_{col_prefix}"], crush_window=2
        )

        # Event zone
        result[f"{col_prefix}_zone"] = self.event_zone_flag(
            result[f"days_to_{col_prefix}"], danger_zone=3, opportunity_zone=2
        )

        return result
