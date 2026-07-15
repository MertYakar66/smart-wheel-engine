"""
Reproduction test for held finding F2-dollar-gamma-100x.

FINDING: dollar-gamma P&L is 100x understated.

engine/risk_manager.py :: RiskManager.calculate_portfolio_greeks builds the
dollar-weighted greeks like this (around L362-363):

    greeks.delta_dollars += pos_greeks["delta"] * multiplier * spot
    greeks.gamma_dollars += pos_greeks["gamma"] * multiplier * spot * spot / 100
                                                                          ^^^^^^^  <-- spurious /100

where ``multiplier = direction * contracts * 100``.

Every gamma-P&L consumer then pairs ``gamma_dollars`` with a *fractional*
underlying move squared, e.g. historical VaR (L651):

    gamma_pnl = 0.5 * greeks.gamma_dollars * (hist_returns.values ** 2)

The delta term is internally consistent:
    delta_pnl = delta_dollars * ret
              = delta * (contracts*100) * spot * (dS/spot)
              = delta * (contracts*100) * dS         <-- correct: delta * shares * dS

For the gamma term to use the SAME fractional-return convention, the analytic
identity is:
    gamma_pnl = 0.5 * gamma * shares * dS**2
              = 0.5 * gamma * (contracts*100) * (ret * spot)**2
              = 0.5 * [gamma * (contracts*100) * spot**2] * ret**2

so the quantity multiplied by ``0.5 * ret**2`` MUST be
``gamma * multiplier * spot**2`` (NO ``/100``). The engine divides it by 100,
so the convexity term is exactly 100x too small.

VERIFIED numeric example (10-lot short $95 put, S=100, T=0.10, sigma=0.30,
r=0.0):
    gamma (analytic)          = 0.0353738725...
    correct gamma_dollars     = gamma * (-1*10*100) * 100^2      = -353,738.72
    engine  gamma_dollars     = gamma * (-1*10*100) * 100^2 /100 =  -3,537.39
    correct gamma P&L (-10%)  = 0.5 * (-353,738.72) * 0.10^2     = -1,768.69
    engine  gamma P&L (-10%)  = 0.5 * (  -3,537.39) * 0.10^2     =    -17.69   <-- 100x too small

This test asserts the CORRECT (bug-free) behavior and therefore FAILS on the
current engine code. It is marked xfail(strict=True): it reports 'xfailed' now
and will become an XPASS -> failure once the operator removes the /100.
"""

import os

import pytest

# Ensure a data provider is set before importing engine modules that may probe
# provider configuration at import time. Synthetic/bloomberg only — never Theta.
os.environ.setdefault("SWE_DATA_PROVIDER", "bloomberg")

from engine.option_pricer import black_scholes_all_greeks
from engine.risk_manager import RiskManager


# ---- Finding's exact book / market state ---------------------------------
SPOT = 100.0
STRIKE = 95.0
DTE_DAYS = 36.5          # 36.5 / 365 == 0.10 year, matching the finding's T=0.10
T_YEARS = DTE_DAYS / 365
SIGMA = 0.30
CONTRACTS = 10
RISK_FREE_RATE = 0.0     # match the finding's analytic target (r=0)
MOVE_FRACTION = -0.10    # a -10% underlying move

# Direction multiplier as computed inside calculate_portfolio_greeks:
#   direction = -1 (short); multiplier = direction * contracts * 100
MULTIPLIER = -1 * CONTRACTS * 100


def _analytic_gamma() -> float:
    """Analytic BS gamma for the finding's option, via the engine's own pricer."""
    greeks = black_scholes_all_greeks(
        S=SPOT,
        K=STRIKE,
        T=T_YEARS,
        r=RISK_FREE_RATE,
        sigma=SIGMA,
        option_type="put",
    )
    return greeks["gamma"]


@pytest.mark.xfail(
    reason="held finding F2-dollar-gamma-100x; drop xfail when fixed", strict=True
)
def test_dollar_gamma_pnl_not_understated_100x():
    """
    The gamma-P&L term the risk_manager uses (0.5 * gamma_dollars * move**2)
    MUST equal the analytic 0.5 * gamma * shares * (move*spot)**2. Equivalently,
    the engine's gamma_dollars MUST equal gamma * multiplier * spot**2 (no /100).

    Currently the engine divides by 100, so both assertions below fail (the
    computed gamma P&L is ~$17.69 instead of the analytic ~$1768.69).
    """
    rm = RiskManager(risk_free_rate=RISK_FREE_RATE)

    positions = [
        {
            "symbol": "TEST",
            "option_type": "put",
            "strike": STRIKE,
            "dte": DTE_DAYS,
            "iv": SIGMA,
            "contracts": CONTRACTS,
            "is_short": True,
        }
    ]
    spot_prices = {"TEST": SPOT}

    greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

    gamma = _analytic_gamma()

    # ---- Target 1: gamma_dollars must carry NO spurious /100 -------------
    # Correct convention: gamma_dollars = gamma * shares * spot**2
    # (shares == multiplier == direction * contracts * 100)
    expected_gamma_dollars = gamma * MULTIPLIER * SPOT * SPOT

    assert greeks.gamma_dollars == pytest.approx(expected_gamma_dollars, rel=1e-9), (
        f"gamma_dollars is understated: engine reported {greeks.gamma_dollars:,.2f} "
        f"but the /100-free convention requires {expected_gamma_dollars:,.2f} "
        f"(off by factor {expected_gamma_dollars / greeks.gamma_dollars:.1f}x)"
    )

    # ---- Target 2: the gamma P&L term for a -10% move must be ~ -$1768.69 -
    # This is exactly the expression used by historical VaR (L651),
    # stress tests (L1259/L1303/L1349/L1376): 0.5 * gamma_dollars * move**2.
    gamma_pnl_engine = 0.5 * greeks.gamma_dollars * (MOVE_FRACTION**2)

    # Analytic dollar-gamma P&L: 0.5 * gamma * shares * dS**2, dS = move*spot
    dS = MOVE_FRACTION * SPOT
    gamma_pnl_analytic = 0.5 * gamma * MULTIPLIER * (dS**2)

    # Sanity: the analytic target is the ~-$1768.69 from the finding (short
    # gamma => a loss on a large move), NOT the ~-$17.69 the engine yields.
    assert gamma_pnl_analytic == pytest.approx(-1768.69, abs=1.0)

    assert gamma_pnl_engine == pytest.approx(gamma_pnl_analytic, rel=1e-9), (
        f"gamma P&L for a {MOVE_FRACTION:.0%} move is understated ~100x: "
        f"engine computes {gamma_pnl_engine:,.2f} but analytic convexity P&L is "
        f"{gamma_pnl_analytic:,.2f}"
    )
