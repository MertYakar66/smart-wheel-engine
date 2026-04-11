"""
Payoff Diagram and Strike Recommendation Engine

Generates interactive payoff curves and optimal strike recommendations
for wheel strategy structures:
- Cash-Secured Put (CSP) payoff
- Covered Call (CC) payoff
- Short Strangle payoff
- Expected move bands from IV
- Strike recommendation with probability and return analysis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class StrikeRecommendation:
    """Recommended strike for a wheel trade."""

    ticker: str
    strategy: str  # "csp" or "cc"
    strike: float
    premium_estimate: float
    delta: float
    probability_otm: float
    annualized_return: float  # Premium / capital at risk, annualized
    breakeven: float
    distance_from_spot_pct: float
    max_loss: float
    expected_value: float
    dte: int
    score: float  # 0-100 composite quality score


def compute_payoff(
    spot: float,
    strike: float,
    premium: float,
    strategy: str,
    contracts: int = 1,
    cost_basis: float | None = None,
    price_range_pct: float = 0.30,
    n_points: int = 100,
) -> list[dict]:
    """
    Compute payoff diagram data for a wheel strategy.

    Args:
        spot: Current underlying price
        strike: Option strike price
        premium: Premium received per share
        strategy: "csp", "cc", "short_strangle", "long_call", "long_put"
        contracts: Number of contracts
        cost_basis: Cost basis per share (for covered calls)
        price_range_pct: Range of prices to plot (±30% default)
        n_points: Number of price points

    Returns:
        List of dicts with {price, pnl, pnl_pct, breakeven_marker}
    """
    multiplier = contracts * 100
    prices = np.linspace(spot * (1 - price_range_pct), spot * (1 + price_range_pct), n_points)

    data = []
    for p in prices:
        price = float(p)

        if strategy == "csp":
            # Cash-secured put: profit = premium if above strike, loss below
            if price >= strike:
                pnl = premium * multiplier
            else:
                pnl = (premium - (strike - price)) * multiplier

        elif strategy == "cc":
            # Covered call: own shares + short call
            basis = cost_basis if cost_basis else spot
            share_pnl = (price - basis) * multiplier / 100  # Per share
            if price <= strike:
                pnl = share_pnl * 100 + premium * multiplier
            else:
                pnl = (strike - basis) * multiplier + premium * multiplier

        elif strategy == "short_strangle":
            # Short strangle: short put at strike, short call at strike + spread
            put_strike = strike
            call_strike = strike * 1.05  # 5% above for strangle
            put_pnl = premium / 2 if price >= put_strike else (premium / 2 - (put_strike - price))
            call_pnl = (
                premium / 2 if price <= call_strike else (premium / 2 - (price - call_strike))
            )
            pnl = (put_pnl + call_pnl) * multiplier

        elif strategy == "long_call":
            if price > strike:
                pnl = (price - strike - premium) * multiplier
            else:
                pnl = -premium * multiplier

        elif strategy == "long_put":
            if price < strike:
                pnl = (strike - price - premium) * multiplier
            else:
                pnl = -premium * multiplier

        else:
            pnl = 0

        capital_at_risk = strike * multiplier if strategy == "csp" else spot * multiplier
        pnl_pct = (pnl / capital_at_risk * 100) if capital_at_risk > 0 else 0

        data.append(
            {
                "price": round(price, 2),
                "pnl": round(pnl, 2),
                "pnlPct": round(pnl_pct, 2),
            }
        )

    return data


def compute_expected_move(
    spot: float,
    iv: float,
    dte: int,
    n_points: int = 100,
) -> dict:
    """
    Compute expected move bands from implied volatility.

    Returns price levels for 1σ, 1.5σ, and 2σ moves.
    """
    if iv <= 0 or dte <= 0:
        return {
            "spot": spot,
            "dte": dte,
            "iv": iv,
            "bands": [],
        }

    # IV is annualized, convert to period
    iv_decimal = iv / 100 if iv > 1 else iv
    period_vol = iv_decimal * np.sqrt(dte / 365)

    bands = []
    for sigma, label in [(1.0, "1σ"), (1.5, "1.5σ"), (2.0, "2σ")]:
        move = spot * period_vol * sigma
        prob = norm.cdf(sigma) - norm.cdf(-sigma)  # Probability of staying within
        bands.append(
            {
                "label": label,
                "upper": round(float(spot + move), 2),
                "lower": round(float(spot - move), 2),
                "move_pct": round(float(period_vol * sigma * 100), 2),
                "probability_within": round(float(prob * 100), 1),
            }
        )

    return {
        "spot": spot,
        "dte": dte,
        "iv": round(iv_decimal * 100, 1),
        "period_vol": round(float(period_vol * 100), 2),
        "bands": bands,
    }


def recommend_strikes(
    ticker: str,
    spot: float,
    iv: float,
    dte: int = 45,
    risk_free_rate: float = 0.04,
    strategy: str = "csp",
    n_candidates: int = 5,
) -> list[dict]:
    """
    Recommend optimal strikes for CSP or CC.

    Uses BSM delta-based strike selection with return optimization.
    """
    if spot <= 0 or iv <= 0 or dte <= 0:
        return []

    iv_decimal = iv / 100 if iv > 1 else iv
    T = dte / 365
    sqrt_T = np.sqrt(T)

    # Generate candidate strikes based on delta targets
    if strategy == "csp":
        # Short puts: target 20-40 delta (OTM puts)
        deltas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        candidates = []
        for target_delta in deltas:
            # Approximate strike from delta using BSM inverse
            # For puts: delta = -N(-d1), so d1 = N_inv(1 - |delta|)
            d1 = norm.ppf(1 - target_delta)
            k = np.log(spot) + (risk_free_rate + 0.5 * iv_decimal**2) * T - d1 * iv_decimal * sqrt_T
            strike = round(float(np.exp(k)), 0)

            # Estimate premium using BSM
            d1_actual = (np.log(spot / strike) + (risk_free_rate + 0.5 * iv_decimal**2) * T) / (
                iv_decimal * sqrt_T
            )
            d2 = d1_actual - iv_decimal * sqrt_T
            put_price = float(
                strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1_actual)
            )
            put_price = max(0.01, put_price)

            delta_actual = float(-norm.cdf(-d1_actual))
            prob_otm = float(norm.cdf(d2))  # Probability put expires OTM

            # Annualized return on capital
            capital = strike * 100
            annual_return = (put_price * 100 / capital) * (365 / dte) * 100

            # Breakeven
            breakeven = strike - put_price

            # Distance from spot
            dist_pct = (spot - strike) / spot * 100

            # Expected value (simplified)
            ev = prob_otm * put_price - (1 - prob_otm) * (strike - breakeven)

            # Score: weighted combination
            score = (
                prob_otm * 30  # Higher probability = safer
                + min(annual_return, 30) * 1.5  # Higher return = better (capped)
                + dist_pct * 2  # Further OTM = safer
                + (50 if 0.20 <= abs(delta_actual) <= 0.35 else 30) * 0.5  # Sweet spot delta
            )
            score = min(100, max(0, score))

            candidates.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "strike": strike,
                    "premium": round(put_price, 2),
                    "delta": round(delta_actual, 3),
                    "probabilityOtm": round(prob_otm * 100, 1),
                    "annualizedReturn": round(annual_return, 1),
                    "breakeven": round(breakeven, 2),
                    "distanceFromSpotPct": round(dist_pct, 1),
                    "maxLoss": round((strike - put_price) * 100, 0),
                    "expectedValue": round(ev, 2),
                    "dte": dte,
                    "score": round(score, 0),
                    "capitalRequired": round(capital, 0),
                }
            )

    elif strategy == "cc":
        # Covered calls: target 20-40 delta OTM calls
        deltas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        candidates = []
        for target_delta in deltas:
            d1 = norm.ppf(1 - target_delta)
            k = np.log(spot) + (risk_free_rate + 0.5 * iv_decimal**2) * T - d1 * iv_decimal * sqrt_T
            strike = round(float(np.exp(k)), 0)

            d1_actual = (np.log(spot / strike) + (risk_free_rate + 0.5 * iv_decimal**2) * T) / (
                iv_decimal * sqrt_T
            )
            d2 = d1_actual - iv_decimal * sqrt_T
            call_price = float(
                spot * norm.cdf(d1_actual) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
            )
            call_price = max(0.01, call_price)

            delta_actual = float(norm.cdf(d1_actual))
            prob_otm = float(1 - norm.cdf(d2))

            # Annualized premium yield
            annual_return = (call_price / spot) * (365 / dte) * 100

            # Max profit if called away
            max_profit = (strike - spot + call_price) * 100
            upside_sacrifice = (strike - spot) / spot * 100

            score = (
                prob_otm * 25
                + min(annual_return, 25) * 2
                + (40 if upside_sacrifice > 3 else 20) * 0.5
                + (50 if 0.20 <= delta_actual <= 0.35 else 30) * 0.5
            )
            score = min(100, max(0, score))

            candidates.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "strike": strike,
                    "premium": round(call_price, 2),
                    "delta": round(delta_actual, 3),
                    "probabilityOtm": round(prob_otm * 100, 1),
                    "annualizedReturn": round(annual_return, 1),
                    "breakeven": round(spot - call_price, 2),
                    "distanceFromSpotPct": round(upside_sacrifice, 1),
                    "maxProfit": round(max_profit, 0),
                    "upsideSacrifice": round(upside_sacrifice, 1),
                    "dte": dte,
                    "score": round(score, 0),
                }
            )

    else:
        return []

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:n_candidates]
