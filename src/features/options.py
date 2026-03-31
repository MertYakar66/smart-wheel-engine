"""Options-specific features."""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Import canonical pricing functions to avoid duplication
from engine.option_pricer import black_scholes_delta as _bs_delta


class OptionsFeatures:
    """Compute options-specific features for wheel strategy."""

    @staticmethod
    def put_call_volume_ratio(put_volume: pd.Series, call_volume: pd.Series) -> pd.Series:
        """Put/Call volume ratio."""
        return put_volume / call_volume.replace(0, np.nan)

    @staticmethod
    def put_call_oi_ratio(put_oi: pd.Series, call_oi: pd.Series) -> pd.Series:
        """Put/Call open interest ratio."""
        return put_oi / call_oi.replace(0, np.nan)

    @staticmethod
    def oi_change(oi: pd.Series) -> pd.Series:
        """Daily change in open interest."""
        return oi.diff()

    @staticmethod
    def oi_change_pct(oi: pd.Series) -> pd.Series:
        """Percentage change in open interest."""
        return oi.pct_change()

    @staticmethod
    def unusual_volume_score(
        volume: pd.Series,
        window: int = 20,
        threshold: float = 2.0,
    ) -> pd.Series:
        """
        Unusual volume score.

        Score > threshold indicates unusual activity.

        Args:
            volume: Options volume
            window: Lookback for average
            threshold: Standard deviations for "unusual"

        Returns:
            Z-score of current volume (NaN when std == 0)
        """
        avg = volume.rolling(window).mean()
        std = volume.rolling(window).std()
        # Guard against division by zero
        return (volume - avg) / std.replace(0, np.nan)

    @staticmethod
    def iv_skew(iv_put: pd.Series, iv_call: pd.Series) -> pd.Series:
        """
        IV skew (put IV - call IV at same delta).

        Positive skew = puts more expensive (bearish sentiment).
        """
        return iv_put - iv_call

    @staticmethod
    def iv_crush_expected(
        iv_current: pd.Series,
        iv_historical_post_event: pd.Series,
    ) -> pd.Series:
        """
        Expected IV crush after event.

        Args:
            iv_current: Current IV
            iv_historical_post_event: Average post-event IV

        Returns:
            Expected IV drop
        """
        return iv_current - iv_historical_post_event

    @staticmethod
    def black_scholes_delta(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        is_call: bool = True,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Black-Scholes delta with continuous dividend yield.

        Delegates to the canonical implementation in engine.option_pricer
        to ensure consistency across the codebase.

        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate
            is_call: True for call, False for put
            dividend_yield: Continuous dividend yield

        Returns:
            Option delta
        """
        option_type = 'call' if is_call else 'put'
        return _bs_delta(
            S=spot,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type=option_type,
            q=dividend_yield
        )

    @staticmethod
    def probability_of_profit(
        spot: float,
        strike: float,
        premium: float,
        time_to_expiry: float,
        volatility: float,
        is_short_put: bool = True,
    ) -> float:
        """
        Probability of profit for option position.

        Args:
            spot: Current stock price
            strike: Option strike price
            premium: Premium received (for short) or paid (for long)
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            is_short_put: True for short put, False for long call

        Returns:
            Probability of profit (0-1)
        """
        if time_to_expiry <= 0:
            return 1.0 if is_short_put and spot >= strike else 0.0

        if is_short_put:
            # Short put profits if price stays above breakeven
            breakeven = strike - premium
            d2 = (np.log(spot / breakeven) + (-0.5 * volatility ** 2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry)
            )
            return norm.cdf(d2)
        else:
            # Long call profits if price goes above breakeven
            breakeven = strike + premium
            d2 = (np.log(spot / breakeven) + (-0.5 * volatility ** 2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry)
            )
            return norm.cdf(d2)

    @staticmethod
    def expected_move(
        spot: float,
        iv: float,
        days_to_expiry: int,
    ) -> float:
        """
        Expected move based on implied volatility.

        Args:
            spot: Current stock price
            iv: Implied volatility (annualized)
            days_to_expiry: Days to expiration

        Returns:
            Expected move in dollars
        """
        return spot * iv * np.sqrt(days_to_expiry / 365)

    @staticmethod
    def expected_move_pct(iv: float, days_to_expiry: int) -> float:
        """Expected move as percentage."""
        return iv * np.sqrt(days_to_expiry / 365)

    @staticmethod
    def premium_yield(
        premium: float,
        strike: float,
        days_to_expiry: int,
    ) -> float:
        """
        Annualized premium yield for CSP.

        Args:
            premium: Premium received
            strike: Strike price (capital at risk)
            days_to_expiry: Days to expiration

        Returns:
            Annualized yield (NaN if strike or days_to_expiry is zero)
        """
        if strike <= 0 or days_to_expiry <= 0:
            return np.nan
        return (premium / strike) * (365 / days_to_expiry)

    @staticmethod
    def risk_reward_ratio(
        premium: float,
        strike: float,
        expected_loss: float,
    ) -> float:
        """
        Risk/reward ratio for option position.

        Args:
            premium: Premium received
            strike: Strike price
            expected_loss: Expected loss if wrong

        Returns:
            Risk/reward ratio
        """
        return premium / max(expected_loss, 0.01)

    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute options flow features.

        Args:
            df: DataFrame with options data

        Returns:
            DataFrame with flow features added
        """
        result = df.copy()

        # Basic ratios
        result["pc_volume_ratio"] = self.put_call_volume_ratio(
            result["put_volume"], result["call_volume"]
        )
        result["pc_oi_ratio"] = self.put_call_oi_ratio(result["put_oi"], result["call_oi"])

        # OI changes
        result["call_oi_change"] = self.oi_change(result["call_oi"])
        result["put_oi_change"] = self.oi_change(result["put_oi"])
        result["call_oi_change_pct"] = self.oi_change_pct(result["call_oi"])
        result["put_oi_change_pct"] = self.oi_change_pct(result["put_oi"])

        # Unusual activity
        result["unusual_call_vol"] = self.unusual_volume_score(result["call_volume"])
        result["unusual_put_vol"] = self.unusual_volume_score(result["put_volume"])

        # Total options activity
        result["total_volume"] = result["call_volume"] + result["put_volume"]
        result["total_oi"] = result["call_oi"] + result["put_oi"]

        # Volume to OI ratio (indicates new positions vs closing)
        result["volume_oi_ratio"] = result["total_volume"] / result["total_oi"].replace(0, np.nan)

        return result

    def compute_pricing_features(
        self,
        spot: pd.Series,
        iv: pd.Series,
        days_to_expiry: int = 30,
    ) -> pd.DataFrame:
        """
        Compute pricing features for wheel strategy analysis.

        Args:
            spot: Stock prices
            iv: Implied volatility
            days_to_expiry: Target DTE

        Returns:
            DataFrame with pricing features
        """
        result = pd.DataFrame(index=spot.index)

        # Expected move
        result["expected_move"] = spot * iv * np.sqrt(days_to_expiry / 365)
        result["expected_move_pct"] = iv * np.sqrt(days_to_expiry / 365)

        # 1 standard deviation range
        result["upper_1sd"] = spot * (1 + result["expected_move_pct"])
        result["lower_1sd"] = spot * (1 - result["expected_move_pct"])

        # Approximate ATM premium (rough estimate)
        result["atm_premium_approx"] = spot * iv * np.sqrt(days_to_expiry / 365) * 0.4

        return result
