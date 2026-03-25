"""
Label Generation - The Y in your ML models

Without correct labels, your model is useless.

This module generates labels for:
1. Option outcomes (win/loss/assignment)
2. Premium capture rates
3. Forward returns
4. Optimal actions (hindsight labels)
"""

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass
from enum import IntEnum


class OptionOutcome(IntEnum):
    """Outcome of a short option position."""
    FULL_WIN = 2       # Expired worthless, kept 100% premium
    PARTIAL_WIN = 1    # Closed early at profit
    SCRATCH = 0        # Breakeven
    PARTIAL_LOSS = -1  # Closed at loss but < premium
    FULL_LOSS = -2     # Lost more than premium collected
    ASSIGNED = -3      # Got assigned


@dataclass
class CSPOutcome:
    """Outcome of a cash-secured put position."""
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    ticker: str
    strike: float
    premium_collected: float
    entry_price: float
    exit_price: float
    outcome: OptionOutcome
    pnl: float
    pnl_pct: float
    assigned: bool
    days_held: int
    exit_reason: str


class LabelGenerator:
    """
    Generate labels for ML model training.

    Labels must be:
    1. Forward-looking (what happened AFTER entry)
    2. Point-in-time accurate (no lookahead bias)
    3. Aligned with strategy objectives
    """

    # === OPTION OUTCOME LABELS ===

    @staticmethod
    def csp_outcome(
        entry_price: float,
        strike: float,
        premium: float,
        prices_forward: pd.Series,
        dte: int,
    ) -> dict:
        """
        Determine outcome of a cash-secured put.

        Args:
            entry_price: Stock price at entry
            strike: Put strike price
            premium: Premium collected
            prices_forward: Price series from entry to expiry
            dte: Days to expiration at entry

        Returns:
            Dict with outcome details
        """
        if len(prices_forward) == 0:
            return {"outcome": None, "reason": "insufficient_data"}

        # Get expiry price (or last available)
        expiry_idx = min(dte, len(prices_forward) - 1)
        expiry_price = prices_forward.iloc[expiry_idx]
        min_price = prices_forward.iloc[:expiry_idx + 1].min()

        # Breakeven
        breakeven = strike - premium

        # Determine outcome
        if expiry_price >= strike:
            # Expired OTM - full win
            outcome = OptionOutcome.FULL_WIN
            pnl = premium
            assigned = False
            reason = "expired_otm"
        elif expiry_price >= breakeven:
            # ITM but above breakeven - partial loss but overall win
            outcome = OptionOutcome.PARTIAL_WIN
            pnl = premium - (strike - expiry_price)
            assigned = True
            reason = "assigned_above_breakeven"
        else:
            # Below breakeven - loss
            loss = strike - expiry_price - premium
            if loss < premium:
                outcome = OptionOutcome.PARTIAL_LOSS
            else:
                outcome = OptionOutcome.FULL_LOSS
            pnl = -loss
            assigned = True
            reason = "assigned_below_breakeven"

        # Capital at risk is strike (for CSP)
        pnl_pct = pnl / strike

        return {
            "outcome": outcome,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "assigned": assigned,
            "reason": reason,
            "expiry_price": expiry_price,
            "min_price": min_price,
            "max_drawdown": (entry_price - min_price) / entry_price,
        }

    @staticmethod
    def binary_win_label(
        entry_price: float,
        strike: float,
        premium: float,
        expiry_price: float,
    ) -> int:
        """
        Simple binary win/loss label.

        1 = Profitable (expired OTM or assigned above breakeven)
        0 = Loss
        """
        breakeven = strike - premium
        if expiry_price >= breakeven:
            return 1
        return 0

    @staticmethod
    def premium_capture_rate(
        premium_collected: float,
        premium_at_exit: float,
    ) -> float:
        """
        What percentage of premium was captured?

        1.0 = Full premium captured (expired worthless)
        0.5 = Closed at 50% profit
        0.0 = Breakeven
        -1.0 = Lost equal to premium
        """
        return (premium_collected - premium_at_exit) / premium_collected

    # === FORWARD RETURN LABELS ===

    @staticmethod
    def forward_return(
        price: pd.Series,
        periods: int,
    ) -> pd.Series:
        """
        N-period forward return.

        This is what you're trying to predict.
        """
        return price.shift(-periods) / price - 1

    @staticmethod
    def forward_return_binary(
        price: pd.Series,
        periods: int,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        Binary label: will price be above threshold in N periods?

        1 = Price went up (or stayed above threshold)
        0 = Price went down
        """
        fwd_return = LabelGenerator.forward_return(price, periods)
        return (fwd_return > threshold).astype(int)

    @staticmethod
    def forward_max_drawdown(
        price: pd.Series,
        periods: int,
    ) -> pd.Series:
        """
        Maximum drawdown over next N periods (vectorized).

        Important for CSP: max drop determines if you get assigned.

        Returns negative values (drawdowns are negative by convention).
        E.g., -0.10 means price dropped 10% from entry within the period.
        """
        # Vectorized: rolling min shifted back to align with entry point
        # For each point, find the minimum over the next N periods
        forward_min = price[::-1].rolling(periods + 1, min_periods=1).min()[::-1]

        # Shift to align: forward_min[i] should be min of price[i:i+periods+1]
        # The reversal + rolling already handles this
        drawdown = (forward_min - price) / price

        # Set last 'periods' values to NaN (insufficient forward data)
        drawdown.iloc[-periods:] = np.nan

        return drawdown

    @staticmethod
    def touch_strike_label(
        price: pd.Series,
        strike: float,
        periods: int,
    ) -> pd.Series:
        """
        Did price touch strike within N periods? (vectorized)

        For puts: did price drop to or below strike?
        For calls: would need to check if price rose above strike.

        1 = Yes (would have been ITM at some point)
        0 = No (stayed OTM)

        This is critical for probability of touch (PoT) validation.
        PoT >> delta, so this matters for assignment risk.
        """
        # Forward-looking minimum (vectorized)
        forward_min = price[::-1].rolling(periods + 1, min_periods=1).min()[::-1]
        touched = (forward_min <= strike).astype(int)
        # Set last 'periods' values to NaN
        touched.iloc[-periods:] = np.nan
        return touched

    # === VOLATILITY LABELS ===

    @staticmethod
    def forward_realized_vol(
        price: pd.Series,
        periods: int = 21,
    ) -> pd.Series:
        """
        Realized volatility over next N periods (vectorized).

        Compare to IV at entry to see if options were mispriced.

        This is the TRUE label for whether selling vol was profitable:
        - If IV at entry > forward RV: seller won (options were expensive)
        - If IV at entry < forward RV: seller lost (options were cheap)
        """
        log_returns = np.log(price / price.shift(1))

        # Vectorized forward-looking std using reversed rolling
        # Reverse, compute rolling std, reverse back
        forward_std = log_returns[::-1].rolling(periods, min_periods=periods).std()[::-1]

        # Shift by 1 because we want returns AFTER entry, not including entry day
        forward_rv = forward_std.shift(-1) * np.sqrt(252)

        # Last 'periods' values have insufficient data
        forward_rv.iloc[-periods:] = np.nan

        return forward_rv

    @staticmethod
    def iv_overpriced_label(
        iv: pd.Series,
        forward_rv: pd.Series,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        Was IV overpriced (good for selling)?

        1 = IV > RV (options were expensive, seller won)
        0 = IV <= RV (options were cheap, seller lost)
        """
        spread = iv - forward_rv
        return (spread > threshold).astype(int)

    # === OPTIMAL ACTION LABELS ===

    @staticmethod
    def optimal_entry_label(
        price: pd.Series,
        iv: pd.Series,
        forward_periods: int = 21,
        price_threshold: float = 0.05,
        iv_threshold: float = 0.8,
    ) -> pd.Series:
        """
        Hindsight-optimal entry timing.

        1 = Good entry (price didn't drop much, IV was high)
        0 = Bad entry (would have been better to wait)

        Use for training entry timing model.
        """
        fwd_drawdown = LabelGenerator.forward_max_drawdown(price, forward_periods)
        iv_rank = iv.rolling(252).apply(
            lambda x: (x < x.iloc[-1]).sum() / (len(x) - 1), raw=False
        )

        good_entry = (fwd_drawdown.abs() < price_threshold) & (iv_rank > iv_threshold)
        return good_entry.astype(int)

    @staticmethod
    def optimal_exit_label(
        premium_path: pd.Series,
        profit_target: float = 0.5,
        loss_limit: float = 2.0,
    ) -> pd.Series:
        """
        Optimal exit timing (with hindsight).

        Returns the best action at each point:
        1 = Should have closed (captured enough profit or avoiding larger loss)
        0 = Should have held

        premium_path: Premium value over time (starts at 1.0)
        """
        result = pd.Series(0, index=premium_path.index)

        # Identify points where closing was optimal
        # Close if premium dropped to profit target
        profit_points = premium_path <= (1 - profit_target)
        result[profit_points] = 1

        # Also identify where holding led to loss
        loss_points = premium_path >= (1 + loss_limit)
        result[loss_points] = 1

        return result

    # === CLASSIFICATION LABELS ===

    @staticmethod
    def multi_class_outcome(
        pnl_pct: pd.Series,
        bins: list[float] = [-np.inf, -0.05, 0, 0.02, 0.05, np.inf],
        labels: list[str] = ["big_loss", "small_loss", "small_win", "good_win", "great_win"],
    ) -> pd.Series:
        """
        Multi-class outcome labels.

        Bins PnL into categories for classification.
        """
        return pd.cut(pnl_pct, bins=bins, labels=labels)

    @staticmethod
    def regression_target(
        price: pd.Series,
        periods: int,
        target_type: Literal["return", "log_return", "rank"] = "return",
    ) -> pd.Series:
        """
        Regression target for continuous prediction.

        Args:
            price: Price series
            periods: Forward periods
            target_type: Type of target
                - return: Simple return
                - log_return: Log return
                - rank: Cross-sectional rank (for multiple stocks)
        """
        if target_type == "return":
            return LabelGenerator.forward_return(price, periods)
        elif target_type == "log_return":
            return np.log(price.shift(-periods) / price)
        elif target_type == "rank":
            fwd_return = LabelGenerator.forward_return(price, periods)
            return fwd_return.rank(pct=True)
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    def generate_training_labels(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        iv_col: str = "atm_iv",
        strike_pct: float = 0.95,
        dte: int = 30,
    ) -> pd.DataFrame:
        """
        Generate all training labels for wheel strategy.

        Args:
            df: DataFrame with price and IV data
            price_col: Price column name
            iv_col: IV column name
            strike_pct: Strike as percentage of spot (0.95 = 5% OTM)
            dte: Days to expiration

        Returns:
            DataFrame with all labels
        """
        result = df.copy()
        price = result[price_col]

        # Forward returns
        result["fwd_return_1d"] = self.forward_return(price, 1)
        result["fwd_return_5d"] = self.forward_return(price, 5)
        result["fwd_return_21d"] = self.forward_return(price, 21)
        result[f"fwd_return_{dte}d"] = self.forward_return(price, dte)

        # Forward drawdown (critical for CSP)
        result["fwd_max_drawdown_21d"] = self.forward_max_drawdown(price, 21)
        result[f"fwd_max_drawdown_{dte}d"] = self.forward_max_drawdown(price, dte)

        # Binary labels
        result["fwd_up_21d"] = self.forward_return_binary(price, 21)
        result[f"fwd_up_{dte}d"] = self.forward_return_binary(price, dte)

        # Realized vol (for IV comparison)
        result["fwd_rv_21d"] = self.forward_realized_vol(price, 21)

        # IV mispricing
        if iv_col in result.columns:
            result["iv_was_overpriced"] = self.iv_overpriced_label(
                result[iv_col], result["fwd_rv_21d"]
            )

        # Touch strike label (at various OTM levels)
        for otm in [0.95, 0.90, 0.85]:
            strike = price * otm
            col_name = f"touched_{int((1-otm)*100)}pct_otm"
            # This needs to be vectorized differently
            result[col_name] = (
                price.rolling(dte).min().shift(-dte) <= price * otm
            ).astype(int)

        return result
