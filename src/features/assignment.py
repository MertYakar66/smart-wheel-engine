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
            d2 = (log_ratio - 0.5 * volatility**2 * time_to_expiry) / sigma_sqrt_t
            p_touch = 2 * norm.cdf(-d2)
        else:
            # For call: probability of touching upper strike
            if spot >= strike:
                return 1.0  # Already touched

            log_ratio = np.log(strike / spot)
            d2 = (log_ratio - 0.5 * volatility**2 * time_to_expiry) / sigma_sqrt_t
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

        Uses strike as denominator for consistency across the codebase
        and alignment with industry standard (S/K based measures).

        For puts:
        - Positive = ITM (strike > spot)
        - Negative = OTM (strike < spot)

        For calls:
        - Positive = ITM (spot > strike)
        - Negative = OTM (spot < strike)
        """
        if is_put:
            return (strike - spot) / strike
        else:
            return (spot - strike) / strike

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
            Estimated days (capped at 365 if very far OTM)
        """
        # AUDIT FIX: removed orphaned `abs(strike - spot) / spot` expression.
        # It was computed but never assigned — dead code that mis-signalled a
        # "distance_from_strike" variable that is not used by this function.
        danger_zone = (
            strike * (1 - threshold_pct) if spot > strike else strike * (1 + threshold_pct)
        )
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
        # AUDIT FIX: removed orphaned `max(0, strike - spot)` intrinsic-value
        # expression — it was computed but never assigned and not referenced
        # anywhere below. Intrinsic value is not actually needed for the roll
        # scoring logic (which uses assignment_basis + spot compare).

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

        AUDIT FIX (look-ahead bias): Previous implementation used
        ``spot.iloc[-1]`` and ``iv.iloc[-1]`` to compute ``prob_touch``,
        ``early_assignment_prob`` and ``days_to_danger`` as **scalar** values
        and broadcast them to every row of the historical feature frame.
        That meant a feature at date T was computed using spot/IV from the
        last row of the series — i.e. the future — a classic look-ahead
        bias that silently contaminates ML training labels. This version
        computes each feature per row using only that row's spot and IV.

        Args:
            spot: Price series (index is trade date)
            strike: Option strike
            iv: Implied volatility series (must be aligned with spot)
            dte: Days to expiration as of each row (constant here; caller
                 should pre-scale for decaying DTE if needed)
            dividend_yield: Annual dividend yield
            risk_free_rate: Risk-free rate
            is_put: True for put option

        Returns:
            DataFrame indexed by spot.index with per-row assignment features.
        """
        # Align iv to spot's index to avoid ffill/bfill surprises.
        iv_aligned = iv.reindex(spot.index)

        result = pd.DataFrame(index=spot.index)

        time_to_expiry = max(dte, 0) / 365.0

        # Row-wise vectorized features (no look-ahead).
        result["moneyness"] = self.moneyness(spot, strike, is_put)
        result["distance_to_strike_pct"] = self.distance_to_strike_pct(spot, strike)

        # Per-row prob_touch / early-assignment / days_to_danger using each
        # row's own spot and IV. Uses the existing _vectorized_* helpers which
        # accept arrays of strikes/ttes/ivs but take a scalar spot; we loop
        # over spot values with numpy broadcasting.
        spot_arr = spot.to_numpy(dtype=float)
        iv_arr = iv_aligned.to_numpy(dtype=float)
        n = len(spot_arr)
        strikes_arr = np.full(n, float(strike))
        ttes_arr = np.full(n, float(time_to_expiry))
        is_puts_arr = np.full(n, bool(is_put))

        prob_touch = np.full(n, np.nan)
        early_assign = np.full(n, np.nan)
        days_to_danger = np.full(n, np.nan)

        valid = ~(np.isnan(spot_arr) | np.isnan(iv_arr)) & (spot_arr > 0) & (iv_arr > 0)
        for i in np.where(valid)[0]:
            prob_touch[i] = self.probability_of_touch(
                float(spot_arr[i]), strike, time_to_expiry, float(iv_arr[i]), is_put
            )
            early_assign[i] = self.early_assignment_probability(
                float(spot_arr[i]),
                strike,
                time_to_expiry,
                dividend_yield,
                risk_free_rate,
                is_put,
            )
            days_to_danger[i] = self.days_until_danger_zone(
                float(spot_arr[i]), strike, float(iv_arr[i])
            )

        result["prob_touch"] = prob_touch
        result["early_assignment_prob"] = early_assign
        result["days_to_danger"] = days_to_danger

        # Support level analysis (already vectorized, PIT-safe via rolling)
        result["support_distance"] = self.support_level_distance(spot, strike)

        # Moneyness path — expanding window ending at each row (PIT-safe)
        path = self.moneyness_path(spot, strike, window=21, is_put=is_put)
        for key, value in path.items():
            result[f"moneyness_path_{key}"] = value

        return result

    @staticmethod
    def _vectorized_prob_touch(
        spot: float,
        strikes: np.ndarray,
        ttes: np.ndarray,
        ivs: np.ndarray,
        is_puts: np.ndarray,
    ) -> np.ndarray:
        """Vectorized probability of touch calculation."""
        n = len(strikes)
        result = np.zeros(n)

        sigma_sqrt_t = ivs * np.sqrt(ttes)

        # Handle puts
        put_mask = is_puts
        already_touched_put = put_mask & (spot <= strikes)
        otm_put = put_mask & ~already_touched_put

        if otm_put.any():
            log_ratio = np.log(spot / strikes[otm_put])
            d2 = (log_ratio - 0.5 * ivs[otm_put] ** 2 * ttes[otm_put]) / sigma_sqrt_t[otm_put]
            result[otm_put] = np.minimum(2 * norm.cdf(-d2), 1.0)

        result[already_touched_put] = 1.0

        # Handle calls
        call_mask = ~is_puts
        already_touched_call = call_mask & (spot >= strikes)
        otm_call = call_mask & ~already_touched_call

        if otm_call.any():
            log_ratio = np.log(strikes[otm_call] / spot)
            d2 = (log_ratio - 0.5 * ivs[otm_call] ** 2 * ttes[otm_call]) / sigma_sqrt_t[otm_call]
            result[otm_call] = np.minimum(2 * norm.cdf(-d2), 1.0)

        result[already_touched_call] = 1.0

        return result

    @staticmethod
    def _vectorized_early_assignment_prob(
        spot: float,
        strikes: np.ndarray,
        ttes: np.ndarray,
        dividend_yield: float,
        risk_free_rate: float,
        is_puts: np.ndarray,
    ) -> np.ndarray:
        """Vectorized early assignment probability calculation."""
        n = len(strikes)
        result = np.zeros(n)

        # Calculate moneyness (positive = ITM)
        moneyness = np.where(
            is_puts,
            (strikes - spot) / strikes,  # Put: ITM when strike > spot
            (spot - strikes) / strikes,  # Call: ITM when spot > strike
        )

        # Only ITM options can be early exercised
        itm_mask = moneyness > 0

        if not itm_mask.any():
            return result

        # Base probability from moneyness (capped at 0.8)
        base_prob = np.minimum(moneyness * 2, 0.8)

        # Adjustment factors
        rate_factor = np.where(is_puts, 1 + risk_free_rate * 2, 1 - risk_free_rate)
        div_factor = np.where(is_puts, 1 - dividend_yield, 1 + dividend_yield * 3)

        # Time factor: more likely as expiration approaches
        time_factor = 1 + (1 - ttes) * 0.5

        # Compute final probability
        prob = base_prob * rate_factor * div_factor * time_factor
        result = np.where(itm_mask, np.minimum(prob, 0.95), 0.0)

        return result

    @staticmethod
    def _vectorized_days_to_danger(
        spot: float,
        strikes: np.ndarray,
        ivs: np.ndarray,
        threshold_pct: float = 0.02,
    ) -> np.ndarray:
        """Vectorized days until danger zone calculation."""
        # Danger zone boundary
        danger_zone = np.where(
            spot > strikes, strikes * (1 - threshold_pct), strikes * (1 + threshold_pct)
        )

        distance_to_danger = np.abs(danger_zone - spot) / spot

        # Daily volatility
        daily_vol = ivs / np.sqrt(252)

        # Days for 1 std move to reach danger zone
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            days = np.where(
                distance_to_danger <= 0, 0.0, np.minimum((distance_to_danger / daily_vol) ** 2, 365)
            )

        return days

    def compute_for_chain(
        self,
        options_df: pd.DataFrame,
        spot_price: float,
        dividend_yield: float = 0.02,
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute assignment features for an options chain DataFrame.

        This method processes each row of an options chain and computes
        assignment risk metrics for each contract.

        Args:
            options_df: DataFrame with options chain data
                Required columns: strike, dte (or days_to_expiry), iv (or implied_volatility)
                Optional columns: option_type (defaults to 'put')
            spot_price: Current underlying price
            dividend_yield: Annual dividend yield
            risk_free_rate: Risk-free rate

        Returns:
            DataFrame with original data plus assignment features
        """
        result = options_df.copy()

        # Normalize column names
        strike_col = "strike" if "strike" in result.columns else "strike_price"
        dte_col = "dte" if "dte" in result.columns else "days_to_expiry"
        iv_col = "iv" if "iv" in result.columns else "implied_volatility"

        if strike_col not in result.columns:
            raise ValueError("Options DataFrame must have 'strike' column")

        # Default DTE to 30 if not available
        if dte_col not in result.columns:
            result[dte_col] = 30

        # Default IV to 0.25 if not available
        if iv_col not in result.columns:
            result[iv_col] = 0.25

        # Determine option type
        if "option_type" in result.columns:
            is_put = result["option_type"].str.lower() == "put"
        elif "type" in result.columns:
            is_put = result["type"].str.lower() == "put"
        else:
            is_put = pd.Series(True, index=result.index)  # Default to put

        # Compute assignment features for each row
        strikes = result[strike_col].values
        dtes = result[dte_col].values
        ivs = result[iv_col].values
        is_puts = is_put.values

        # Vectorized calculations where possible
        result["moneyness"] = np.where(
            is_puts,
            (strikes - spot_price) / strikes,  # Put: positive when ITM
            (spot_price - strikes) / strikes,  # Call: positive when ITM
        )

        result["distance_to_strike_pct"] = (strikes - spot_price) / spot_price * 100

        # Vectorized point-in-time calculations
        ttes = np.maximum(dtes / 365.0, 1e-6)  # Time to expiry in years

        # Vectorized probability of touch
        result["prob_touch"] = self._vectorized_prob_touch(spot_price, strikes, ttes, ivs, is_puts)

        # Vectorized early assignment probability
        result["early_assignment_prob"] = self._vectorized_early_assignment_prob(
            spot_price, strikes, ttes, dividend_yield, risk_free_rate, is_puts
        )

        # Vectorized days to danger zone
        result["days_to_danger"] = self._vectorized_days_to_danger(spot_price, strikes, ivs)

        return result
