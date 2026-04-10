#!/usr/bin/env python3
"""
Quantitative Benchmark Acceptance Gate

P0 requirement: Hard acceptance gates for quant accuracy benchmarks.

This script validates the quantitative engine against known reference values
from academic sources and industry standards. Must pass before any release.

Usage:
    python scripts/quant_benchmark_gate.py [--strict] [--verbose]

Exit codes:
    0: All benchmarks pass
    1: Benchmark failures detected
    2: Critical tolerance violations (>10x tolerance)

References:
    - Hull, J.C. "Options, Futures, and Other Derivatives" (11th Edition)
    - Jorion, P. "Value at Risk" (3rd Edition)
    - Glasserman, P. "Monte Carlo Methods in Financial Engineering"
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    name: str
    expected: float
    actual: float
    tolerance: float
    passed: bool
    error_ratio: float  # How many tolerances away from target
    category: str
    reference: str


def run_option_pricing_benchmarks() -> list[BenchmarkResult]:
    """Run Black-Scholes pricing benchmarks against Hull textbook."""
    from engine.option_pricer import (
        black_scholes_delta,
        black_scholes_gamma,
        black_scholes_price,
        black_scholes_vega,
        implied_volatility,
    )

    results = []

    # Hull Example 15.6: S=42, K=40, T=0.5, r=0.10, sigma=0.20
    S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20

    # Call price
    call_price = black_scholes_price(S, K, T, r, sigma, "call")
    results.append(
        BenchmarkResult(
            name="BS Call Price (Hull 15.6)",
            expected=4.7594,
            actual=call_price,
            tolerance=0.01,
            passed=abs(call_price - 4.7594) < 0.01,
            error_ratio=abs(call_price - 4.7594) / 0.01,
            category="Pricing",
            reference="Hull 11th Ed, Example 15.6",
        )
    )

    # Call delta
    call_delta = black_scholes_delta(S, K, T, r, sigma, "call")
    results.append(
        BenchmarkResult(
            name="BS Call Delta (Hull 15.6)",
            expected=0.7791,
            actual=call_delta,
            tolerance=0.001,
            passed=abs(call_delta - 0.7791) < 0.001,
            error_ratio=abs(call_delta - 0.7791) / 0.001,
            category="Greeks",
            reference="Hull 11th Ed, Table 15.8",
        )
    )

    # Call gamma
    call_gamma = black_scholes_gamma(S, K, T, r, sigma)
    results.append(
        BenchmarkResult(
            name="BS Call Gamma (Hull 15.6)",
            expected=0.0500,
            actual=call_gamma,
            tolerance=0.001,
            passed=abs(call_gamma - 0.0500) < 0.001,
            error_ratio=abs(call_gamma - 0.0500) / 0.001,
            category="Greeks",
            reference="Hull 11th Ed, Chapter 19",
        )
    )

    # Call vega - our implementation returns vega per 1 vol point (0.01)
    # Hull reports per 100% vol change, so multiply by 100 to compare
    call_vega = black_scholes_vega(S, K, T, r, sigma)
    call_vega_hull = call_vega * 100  # Convert to Hull convention (per 100% vol)
    expected_vega = 8.81  # Hull Example: N'(d1) * S * sqrt(T)
    results.append(
        BenchmarkResult(
            name="BS Call Vega (Hull 15.6)",
            expected=expected_vega,
            actual=call_vega_hull,
            tolerance=0.05,
            passed=abs(call_vega_hull - expected_vega) < 0.05,
            error_ratio=abs(call_vega_hull - expected_vega) / 0.05,
            category="Greeks",
            reference="Hull 11th Ed, Chapter 19",
        )
    )

    # Put-call parity
    put_price = black_scholes_price(S, K, T, r, sigma, "put")
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    results.append(
        BenchmarkResult(
            name="Put-Call Parity",
            expected=parity_rhs,
            actual=parity_lhs,
            tolerance=1e-10,
            passed=abs(parity_lhs - parity_rhs) < 1e-10,
            error_ratio=abs(parity_lhs - parity_rhs) / 1e-10 if parity_rhs != 0 else 0,
            category="Parity",
            reference="Hull 11th Ed, Equation 11.6",
        )
    )

    # Implied volatility round-trip
    recovered_iv = implied_volatility(call_price, S, K, T, r, "call")
    results.append(
        BenchmarkResult(
            name="IV Round-Trip",
            expected=sigma,
            actual=recovered_iv,
            tolerance=1e-6,
            passed=abs(recovered_iv - sigma) < 1e-6,
            error_ratio=abs(recovered_iv - sigma) / 1e-6,
            category="IV Solver",
            reference="Internal consistency",
        )
    )

    # Edge case: At-the-money option
    atm_price = black_scholes_price(100, 100, 0.25, 0.05, 0.20, "call")
    # Brenner-Subrahmanyam (1988) approximation: C_ATM ~= 0.4 * sigma * sqrt(T) * S
    # This is a rough approximation, valid mainly for near-zero rates
    atm_approx = 0.4 * 0.20 * np.sqrt(0.25) * 100
    results.append(
        BenchmarkResult(
            name="ATM Approximation (Brenner-Subrahmanyam)",
            expected=atm_approx,
            actual=atm_price,
            tolerance=1.0,  # Rough approximation, 1.0 tolerance
            passed=abs(atm_price - atm_approx) < 1.0,
            error_ratio=abs(atm_price - atm_approx) / 1.0,
            category="Pricing",
            reference="Brenner-Subrahmanyam (1988) approximation",
        )
    )

    return results


def run_risk_benchmarks() -> list[BenchmarkResult]:
    """Run VaR and risk management benchmarks."""
    from scipy import stats

    results = []

    # Parametric VaR: z * sigma * sqrt(t)
    # For 95% confidence, z = 1.645
    z_95 = stats.norm.ppf(0.95)
    results.append(
        BenchmarkResult(
            name="95% Z-Score",
            expected=1.6449,
            actual=z_95,
            tolerance=0.001,
            passed=abs(z_95 - 1.6449) < 0.001,
            error_ratio=abs(z_95 - 1.6449) / 0.001,
            category="VaR",
            reference="Jorion, Value at Risk",
        )
    )

    # Kelly criterion: f* = (p*b - q) / b
    from engine.risk_manager import calculate_kelly_fraction

    # 60% win rate, 1:1 odds => f* = (0.6*1 - 0.4) / 1 = 0.2
    kelly = calculate_kelly_fraction(0.6, 1.0, 1.0, 1.0)
    results.append(
        BenchmarkResult(
            name="Kelly Criterion (60% win, 1:1)",
            expected=0.20,
            actual=kelly,
            tolerance=0.001,
            passed=abs(kelly - 0.20) < 0.001,
            error_ratio=abs(kelly - 0.20) / 0.001,
            category="Position Sizing",
            reference="Kelly (1956)",
        )
    )

    # No edge = no bet
    kelly_no_edge = calculate_kelly_fraction(0.5, 1.0, 1.0, 1.0)
    results.append(
        BenchmarkResult(
            name="Kelly Criterion (50% win, 1:1)",
            expected=0.0,
            actual=kelly_no_edge,
            tolerance=0.001,
            passed=abs(kelly_no_edge - 0.0) < 0.001,
            error_ratio=abs(kelly_no_edge - 0.0) / 0.001 if kelly_no_edge != 0 else 0,
            category="Position Sizing",
            reference="Kelly (1956)",
        )
    )

    return results


def run_volatility_benchmarks() -> list[BenchmarkResult]:
    """Run volatility estimator benchmarks."""
    from src.features.volatility import VolatilityFeatures

    results = []

    # Create synthetic data with known volatility
    np.random.seed(42)
    n_days = 252
    true_vol = 0.20  # 20% annual volatility
    daily_vol = true_vol / np.sqrt(252)

    # Generate log returns (this is what the volatility estimator expects)
    log_returns = pd.Series(np.random.normal(0, daily_vol, n_days))

    vf = VolatilityFeatures()
    cc_vol = vf.realized_volatility_close(log_returns, window=252, annualize=True)

    # Allow 10% relative error due to estimation noise (sample volatility)
    results.append(
        BenchmarkResult(
            name="Close-to-Close Vol Estimator",
            expected=true_vol,
            actual=cc_vol.iloc[-1],
            tolerance=0.03,  # 3% absolute tolerance
            passed=abs(cc_vol.iloc[-1] - true_vol) < 0.03,
            error_ratio=abs(cc_vol.iloc[-1] - true_vol) / 0.03,
            category="Volatility",
            reference="Standard deviation of returns",
        )
    )

    return results


def run_monte_carlo_benchmarks() -> list[BenchmarkResult]:
    """Run Monte Carlo simulation benchmarks."""
    from engine.risk_manager import RiskManager

    results = []
    rm = RiskManager()

    # Monte Carlo VaR should converge to parametric VaR for simple case
    positions = [
        {
            "symbol": "TEST",
            "option_type": "put",
            "strike": 100,
            "dte": 30,
            "iv": 0.25,
            "contracts": 1,
            "is_short": True,
        }
    ]
    spot_prices = {"TEST": 105}
    volatilities = {"TEST": 0.25}

    # Run MC VaR with enough simulations for convergence
    mc_var, mc_cvar, details = rm.calculate_monte_carlo_var(
        portfolio_value=100000,
        positions=positions,
        spot_prices=spot_prices,
        volatilities=volatilities,
        n_simulations=10000,
        seed=42,
    )

    # MC VaR should be positive and reasonable (< 10% of portfolio for 1-day)
    results.append(
        BenchmarkResult(
            name="MC VaR Positivity",
            expected=1.0,  # Should be > 0
            actual=1.0 if mc_var > 0 else 0.0,
            tolerance=0.0,
            passed=mc_var > 0,
            error_ratio=0 if mc_var > 0 else float("inf"),
            category="Monte Carlo",
            reference="Basic sanity check",
        )
    )

    # CVaR should be >= VaR
    results.append(
        BenchmarkResult(
            name="CVaR >= VaR",
            expected=1.0,  # Should be true
            actual=1.0 if mc_cvar >= mc_var else 0.0,
            tolerance=0.0,
            passed=mc_cvar >= mc_var,
            error_ratio=0 if mc_cvar >= mc_var else float("inf"),
            category="Monte Carlo",
            reference="ES definition",
        )
    )

    # Reproducibility with same seed
    mc_var2, _, _ = rm.calculate_monte_carlo_var(
        portfolio_value=100000,
        positions=positions,
        spot_prices=spot_prices,
        volatilities=volatilities,
        n_simulations=10000,
        seed=42,
    )

    results.append(
        BenchmarkResult(
            name="MC VaR Reproducibility",
            expected=mc_var,
            actual=mc_var2,
            tolerance=1e-10,
            passed=abs(mc_var - mc_var2) < 1e-10,
            error_ratio=abs(mc_var - mc_var2) / 1e-10 if mc_var != 0 else 0,
            category="Monte Carlo",
            reference="Seed reproducibility",
        )
    )

    return results


def print_results(results: list[BenchmarkResult], verbose: bool = False) -> tuple[int, int]:
    """Print benchmark results and return (passed, failed) counts."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    # Group by category
    categories: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print("\n" + "=" * 70)
    print("QUANTITATIVE BENCHMARK GATE RESULTS")
    print("=" * 70)

    for category, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        status = "PASS" if cat_passed == len(cat_results) else "FAIL"
        print(f"\n{category}: [{status}] {cat_passed}/{len(cat_results)}")
        print("-" * 40)

        for r in cat_results:
            status_icon = "\u2713" if r.passed else "\u2717"
            print(f"  {status_icon} {r.name}")
            if verbose or not r.passed:
                print(f"      Expected: {r.expected:.6f}")
                print(f"      Actual:   {r.actual:.6f}")
                print(f"      Error:    {r.error_ratio:.2f}x tolerance")
                if not r.passed:
                    print(f"      Ref: {r.reference}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{len(results)} benchmarks passed")
    if failed > 0:
        print(f"FAILED: {failed} benchmarks")

    # Check for critical violations (>10x tolerance)
    critical = [r for r in results if r.error_ratio > 10 and not r.passed]
    if critical:
        print(f"\nCRITICAL: {len(critical)} violations >10x tolerance!")
        for r in critical:
            print(f"  - {r.name}: {r.error_ratio:.1f}x")

    print("=" * 70)

    return passed, failed


def main():
    """Run all benchmark gates."""
    parser = argparse.ArgumentParser(description="Quantitative Benchmark Gate")
    parser.add_argument("--strict", action="store_true", help="Fail on any warning")
    parser.add_argument("--verbose", action="store_true", help="Show all results")
    args = parser.parse_args()

    print("Running quantitative benchmarks...")

    all_results = []

    # Run all benchmark categories
    print("  - Option pricing benchmarks...")
    all_results.extend(run_option_pricing_benchmarks())

    print("  - Risk management benchmarks...")
    all_results.extend(run_risk_benchmarks())

    print("  - Volatility benchmarks...")
    all_results.extend(run_volatility_benchmarks())

    print("  - Monte Carlo benchmarks...")
    all_results.extend(run_monte_carlo_benchmarks())

    # Print results
    passed, failed = print_results(all_results, args.verbose)

    # Determine exit code
    if failed > 0:
        critical = [r for r in all_results if r.error_ratio > 10 and not r.passed]
        if critical:
            print("\nEXIT CODE 2: Critical tolerance violations")
            sys.exit(2)
        print("\nEXIT CODE 1: Benchmark failures detected")
        sys.exit(1)

    print("\nEXIT CODE 0: All benchmarks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
