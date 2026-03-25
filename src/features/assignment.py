"""
Assignment Risk Features - Critical for wheel strategy

Your wheel strategy DEPENDS on understanding assignment risk:
- When will you get assigned?
- Should you roll or take assignment?
- What's the probability of touch?

This module quantifies assignment risk.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Literal


class AssignmentFeatures:
    """
    Compute assignment risk features for wheel strategy.

    Key metrics:
    - Probability of touch (will price reach strike?)
    - Early assignment likelihood (dividend + rate driven)
    - ITM depth and moneyness path
    - Roll vs assignment decision inputs
    """

    @staticmethod
    def probability_of_touch(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_put: bool = True,
    ) -> float:
        """
        Probability that price will touch strike at any point before expiry.

        This is DIFFERENT from delta (probability of finishing ITM).
        P(touch) ≈ 2 * delta for OTM options.

        Critical insight: Even a 0.30 delta put has ~60% chance of touching.

        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (annualized)
            is_put: True for put, False for call

        Returns:
            Probability of touch (0-1)
        """
        if time_to_expiry <= 0:
            return 1.0 if (is_put and spot <= strike) or (not is_put and spot >= strike) else 0.0

        # Use reflection principle approximation
        sigma_sqrt_t = volatility * np.sqrt(time_to_expiry)

        if is_put:
            # For put: probability of touching lower strike
            if spot <= strike:
                return 1.0  # Already touched

            log_ratio = np.log(spot / strike)
            # P(touch) ≈ 2 * N(-d2) for OTM put
            d2 = (log_ratio - 0.5 * volatility ** 2 * time_to_expiry) / sigma_sqrt_t
            p_touch = 2 * norm.cdf(-d2)
        else:
            # For call: probability of touching upper strike
            if spot >= strike:
                return 1.0  # Already touched

            log_ratio = np.log(strike / spot)
            d2 = (log_ratio - 0.5 * volatility ** 2 * time_to_expiry) / sigma_sqrt_t
            p_touch = 2 * norm.cdf(-d2)

        return min(p_touch, 1.0)

    @staticmethod
    def early_assignment_probability(
        spot: float,
        strike: float,
        time_to_expiry: float,
        dividend_yield: float,
        risk_free_rate: float,
        is_put: bool = True,
    ) -> float:
        """
        Estimate probability of early assignment.

        For puts: More likely when deeply ITM and rates are high
        For calls: More likely near ex-dividend when ITM

        Args:
            spot: Current stock price
            strike: Option strike
            time_to_expiry: Time to expiry in years
            dividend_yield: Annual dividend yield
            risk_free_rate: Risk-free rate
            is_put: True for put

        Returns:
            Early assignment probability estimate (0-1)
        """
        if time_to_expiry <= 0:
            return 0.0

        # Moneyness
        if is_put:
            moneyness = (strike - spot) / strike  # Positive when ITM
        else:
            moneyness = (spot - strike) / strike

        if moneyness <= 0:
            return 0.0  # OTM, no early exercise

        # Base probability from moneyness
        base_prob = min(moneyness * 2, 0.8)  # Deep ITM = higher prob

        # Adjustment factors
        if is_put:
            # Puts: early exercise more likely with high rates
            rate_factor = 1 + risk_free_rate * 2
            div_factor = 1 - dividend_yield  # High div reduces put exercise
        else:
            # Calls: early exercise more likely near ex-div
            div_factor = 1 + dividend_yield * 3
            rate_factor = 1 - risk_free_rate  # High rates reduce call exercise

        # Time factor: more likely as expiration approaches
        time_factor = 1 + (1 - time_to_expiry) * 0.5

        prob = base_prob * rate_factor * div_factor * time_factor
        return min(prob, 0.95)

    @staticmethod
    def moneyness(
        spot: pd.Series,
        strike: float,
        is_put: bool = True,
    ) -> pd.Series:
        """
        Moneyness: how far ITM/OTM the option is.

        For puts:
        - Positive = ITM (strike > spot)
        - Negative = OTM (strike < spot)

        For calls:
        - Positive = ITM (spot > strike)
        - Negative = OTM (spot < strike)
        """
        if is_put:
            return (strike - spot) / spot
        else:
            return (spot - strike) / spot

    @staticmethod
    def moneyness_path(
        spot: pd.Series,
        strike: float,
        window: int = 21,
        is_put: bool = True,
    ) -> dict:
        """
        Analyze moneyness path over time.

        Returns dict with:
        - min_moneyness: Closest to ITM
        - max_moneyness: Furthest OTM
        - days_itm: Number of days ITM
        - trend: Moneyness trend (positive = moving ITM)
        """
        moneyness = AssignmentFeatures.moneyness(spot, strike, is_put)
        recent = moneyness.tail(window)

        return {
            "min_moneyness": recent.min(),
            "max_moneyness": recent.max(),
            "current_moneyness": recent.iloc[-1] if len(recent) > 0 else np.nan,
            "days_itm": (recent > 0).sum(),
            "trend": recent.diff().mean() if len(recent) > 1 else 0,
        }

    @staticmethod
    def distance_to_strike_pct(
        spot: pd.Series,
        strike: float,
    ) -> pd.Series:
        """Distance to strike as percentage of spot."""
        return (strike - spot) / spot * 100

    @staticmethod
    def days_until_danger_zone(
        spot: float,
        strike: float,
        volatility: float,
        threshold_pct: float = 0.02,
    ) -> float:
        """
        Estimate days until price could reach "danger zone" near strike.

        Uses expected range based on volatility.

        Args:
            spot: Current price
            strike: Option strike
            volatility: Implied vol
            threshold_pct: How close is "danger zone" (default 2%)

        Returns:
            Estimated days (inf if very far OTM)
        """
        distance_pct = abs(strike - spot) / spot
        danger_zone = strike * (1 - threshold_pct) if spot > strike else strike * (1 + threshold_pct)
        distance_to_danger = abs(danger_zone - spot) / spot

        if distance_to_danger <= 0:
            return 0  # Already in danger zone

        # Expected daily move
        daily_vol = volatility / np.sqrt(252)

        # Days for 1 std move to reach danger zone
        days = (distance_to_danger / daily_vol) ** 2

        return min(days, 365)

    @staticmethod
    def roll_vs_assignment_score(
        spot: float,
        strike: float,
        current_premium: float,
        roll_premium: float,
        days_to_expiry: int,
        roll_dte: int,
        cost_of_carry: float = 0.0,
    ) -> dict:
        """
        Score whether to roll or take assignment.

        Returns dict with:
        - action: "ROLL" or "TAKE_ASSIGNMENT"
        - score: Confidence score (-1 to 1, positive = roll)
        - breakeven_improvement: Premium improvement from roll
        """
        # Current position value
        current_value = max(0, strike - spot)  # Intrinsic for put

        # Roll economics
        roll_credit = roll_premium - current_premium
        roll_yield = roll_credit / strike * (365 / roll_dte)

        # Assignment economics
        assignment_basis = strike - current_premium
        current_price_vs_basis = (spot - assignment_basis) / assignment_basis

        # Decision factors
        factors = {
            "roll_yield_annualized": roll_yield,
            "assignment_profit_pct": current_price_vs_basis,
            "days_extended": roll_dte - days_to_expiry,
            "premium_improvement": roll_credit,
        }

        # Simple scoring
        score = 0
        if roll_yield > 0.10:  # > 10% annualized roll yield
            score += 0.5
        if current_price_vs_basis < 0:  # Would take loss on assignment
            score += 0.3
        if roll_credit > 0:  # Positive roll credit
            score += 0.2

        if current_price_vs_basis > 0.05:  # > 5% profit on assignment
            score -= 0.5
        if roll_yield < 0.05:  # < 5% roll yield
            score -= 0.3

        action = "ROLL" if score > 0 else "TAKE_ASSIGNMENT"

        return {
            "action": action,
            "score": score,
            **factors,
        }

    @staticmethod
    def support_level_distance(
        spot: pd.Series,
        strike: float,
        lookback: int = 63,
    ) -> pd.Series:
        """
        Distance from strike to recent support level.

        If strike is below support, assignment less likely.
        If strike is above support, price could easily fall through.
        """
        support = spot.rolling(lookback).min()
        return (strike - support) / support

    def compute_all(
        self,
        spot: pd.Series,
        strike: float,
        iv: pd.Series,
        dte: int,
        dividend_yield: float = 0.02,
        risk_free_rate: float = 0.05,
        is_put: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all assignment features for a position.

        Args:
            spot: Price series
            strike: Option strike
            iv: Implied volatility series
            dte: Days to expiration
            dividend_yield: Annual dividend yield
            risk_free_rate: Risk-free rate
            is_put: True for put option

        Returns:
            DataFrame with assignment features
        """
        result = pd.DataFrame(index=spot.index)

        time_to_expiry = dte / 365

        # Current values (vectorized where possible)
        result["moneyness"] = self.moneyness(spot, strike, is_put)
        result["distance_to_strike_pct"] = self.distance_to_strike_pct(spot, strike)

        # Point-in-time calculations (last values)
        current_spot = spot.iloc[-1]
        current_iv = iv.iloc[-1]

        result["prob_touch"] = self.probability_of_touch(
            current_spot, strike, time_to_expiry, current_iv, is_put
        )
        result["early_assignment_prob"] = self.early_assignment_probability(
            current_spot, strike, time_to_expiry, dividend_yield, risk_free_rate, is_put
        )
        result["days_to_danger"] = self.days_until_danger_zone(
            current_spot, strike, current_iv
        )

        # Support level analysis
        result["support_distance"] = self.support_level_distance(spot, strike)

        # Moneyness path
        path = self.moneyness_path(spot, strike, window=21, is_put=is_put)
        for key, value in path.items():
            result[f"moneyness_{key}"] = value

        return result
