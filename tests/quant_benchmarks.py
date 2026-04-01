"""
Quantitative Benchmark Registry

Centralized tolerance thresholds for all quantitative validations.
Used as release gates - tests must pass these tolerances before deployment.

Reference Standards:
- Hull, Options, Futures, and Other Derivatives (10th Ed)
- Barone-Adesi & Whaley (1987) - American options
- Garman-Klass (1980) - Volatility estimators
- Yang-Zhang (2000) - Volatility estimators
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToleranceType(Enum):
    """Type of tolerance check."""
    ABSOLUTE = "absolute"       # |actual - expected| < tolerance
    RELATIVE = "relative"       # |actual - expected| / |expected| < tolerance
    BOUNDS = "bounds"           # lower <= actual <= upper
    RANGE = "range"             # actual in [min, max]
    NON_NEGATIVE = "non_neg"    # actual >= 0
    PROBABILITY = "probability" # actual in [0, 1]


@dataclass
class BenchmarkTolerance:
    """Tolerance specification for a benchmark."""
    name: str
    tolerance_type: ToleranceType
    value: float | tuple[float, float]
    reference: str
    description: str
    is_release_gate: bool = True  # If True, failing blocks release


# =============================================================================
# BLACK-SCHOLES PRICING BENCHMARKS
# =============================================================================

BLACKSCHOLES_BENCHMARKS = {
    # Hull Example 15.6: S=42, K=40, r=0.10, T=0.5, σ=0.20
    "hull_call_price": BenchmarkTolerance(
        name="Hull Ex 15.6 Call Price",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.01,
        reference="Hull Ch. 15, Example 15.6",
        description="European call option price accuracy",
    ),
    "hull_put_price": BenchmarkTolerance(
        name="Hull Ex 15.6 Put Price",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.01,
        reference="Hull Ch. 15, Example 15.6",
        description="European put option price accuracy",
    ),
    "put_call_parity": BenchmarkTolerance(
        name="Put-Call Parity",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=1e-10,
        reference="Mathematical identity",
        description="C - P = S*exp(-qT) - K*exp(-rT)",
    ),
}

# =============================================================================
# GREEKS BENCHMARKS
# =============================================================================

GREEKS_BENCHMARKS = {
    "delta_call_bounds": BenchmarkTolerance(
        name="Call Delta Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(0.0, 1.0),
        reference="Option theory",
        description="Call delta must be in [0, 1]",
    ),
    "delta_put_bounds": BenchmarkTolerance(
        name="Put Delta Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(-1.0, 0.0),
        reference="Option theory",
        description="Put delta must be in [-1, 0]",
    ),
    "gamma_non_negative": BenchmarkTolerance(
        name="Gamma Non-Negative",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Option theory",
        description="Gamma must be >= 0",
    ),
    "vega_non_negative": BenchmarkTolerance(
        name="Vega Non-Negative",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Option theory",
        description="Vega must be >= 0",
    ),
    "hull_delta": BenchmarkTolerance(
        name="Hull Delta Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.001,
        reference="Hull Ch. 15",
        description="Delta accuracy vs textbook",
    ),
    "hull_gamma": BenchmarkTolerance(
        name="Hull Gamma Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.001,
        reference="Hull Ch. 15",
        description="Gamma accuracy vs textbook",
    ),
    "hull_vega": BenchmarkTolerance(
        name="Hull Vega Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.05,
        reference="Hull Ch. 15",
        description="Vega accuracy vs textbook",
    ),
    "hull_theta": BenchmarkTolerance(
        name="Hull Theta Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.01,
        reference="Hull Ch. 15",
        description="Theta accuracy vs textbook",
    ),
    "hull_rho": BenchmarkTolerance(
        name="Hull Rho Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=0.05,
        reference="Hull Ch. 15",
        description="Rho accuracy vs textbook",
    ),
}

# =============================================================================
# AMERICAN OPTIONS BENCHMARKS
# =============================================================================

AMERICAN_BENCHMARKS = {
    "american_geq_european": BenchmarkTolerance(
        name="American >= European",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Option theory",
        description="American option value >= European value",
    ),
    "baw_accuracy": BenchmarkTolerance(
        name="BAW Approximation Accuracy",
        tolerance_type=ToleranceType.RELATIVE,
        value=0.01,  # 1% relative error
        reference="Barone-Adesi & Whaley (1987)",
        description="BAW accuracy vs binomial tree",
    ),
    "early_exercise_premium": BenchmarkTolerance(
        name="Early Exercise Premium",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Option theory",
        description="Early exercise premium must be >= 0",
    ),
}

# =============================================================================
# MONTE CARLO BENCHMARKS
# =============================================================================

MONTE_CARLO_BENCHMARKS = {
    "mc_convergence": BenchmarkTolerance(
        name="MC Convergence",
        tolerance_type=ToleranceType.RELATIVE,
        value=0.05,  # 5% relative error with 10k paths
        reference="MC theory",
        description="MC price should converge to BS",
    ),
    "mc_ci_coverage": BenchmarkTolerance(
        name="MC CI Coverage",
        tolerance_type=ToleranceType.BOUNDS,
        value=(0.90, 1.0),
        reference="Statistics",
        description="95% CI should cover true value ~95% of time",
    ),
    "var_monotonicity": BenchmarkTolerance(
        name="VaR Monotonicity",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Risk theory",
        description="VaR(99%) >= VaR(95%) >= VaR(90%)",
    ),
    "cvar_geq_var": BenchmarkTolerance(
        name="CVaR >= VaR",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Risk theory",
        description="CVaR must be >= VaR at same confidence",
    ),
}

# =============================================================================
# VOLATILITY ESTIMATOR BENCHMARKS
# =============================================================================

VOLATILITY_BENCHMARKS = {
    "realized_vol_non_negative": BenchmarkTolerance(
        name="Realized Vol Non-Negative",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Mathematics",
        description="Realized volatility must be >= 0",
    ),
    "garman_klass_accuracy": BenchmarkTolerance(
        name="Garman-Klass Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=1e-6,
        reference="Garman-Klass (1980)",
        description="GK formula implementation accuracy",
    ),
    "yang_zhang_accuracy": BenchmarkTolerance(
        name="Yang-Zhang Accuracy",
        tolerance_type=ToleranceType.ABSOLUTE,
        value=1e-6,
        reference="Yang-Zhang (2000)",
        description="YZ formula implementation accuracy",
    ),
    "iv_rank_bounds": BenchmarkTolerance(
        name="IV Rank Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(0.0, 100.0),
        reference="Technical analysis",
        description="IV Rank must be in [0, 100]",
    ),
}

# =============================================================================
# RISK METRICS BENCHMARKS
# =============================================================================

RISK_BENCHMARKS = {
    "sharpe_bounds": BenchmarkTolerance(
        name="Sharpe Ratio Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(-10.0, 10.0),
        reference="Finance theory",
        description="Sharpe ratio should be reasonable",
    ),
    "kelly_bounds": BenchmarkTolerance(
        name="Kelly Criterion Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(0.0, 0.25),
        reference="Kelly (1956), half-Kelly practice",
        description="Kelly fraction capped at 25%",
    ),
    "win_rate_bounds": BenchmarkTolerance(
        name="Win Rate Bounds",
        tolerance_type=ToleranceType.PROBABILITY,
        value=(0.0, 1.0),
        reference="Probability theory",
        description="Win rate must be in [0, 1]",
    ),
    "drawdown_bounds": BenchmarkTolerance(
        name="Max Drawdown Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(0.0, 1.0),
        reference="Finance",
        description="Drawdown must be in [0, 1] (0-100%)",
    ),
}

# =============================================================================
# TECHNICAL INDICATORS BENCHMARKS
# =============================================================================

TECHNICAL_BENCHMARKS = {
    "rsi_bounds": BenchmarkTolerance(
        name="RSI Bounds",
        tolerance_type=ToleranceType.BOUNDS,
        value=(0.0, 100.0),
        reference="Wilder (1978)",
        description="RSI must be in [0, 100]",
    ),
    "bollinger_width_positive": BenchmarkTolerance(
        name="Bollinger Width Positive",
        tolerance_type=ToleranceType.NON_NEGATIVE,
        value=0.0,
        reference="Bollinger (1980s)",
        description="Bollinger bandwidth must be positive",
    ),
}

# =============================================================================
# ALL BENCHMARKS REGISTRY
# =============================================================================

ALL_BENCHMARKS: dict[str, BenchmarkTolerance] = {
    **BLACKSCHOLES_BENCHMARKS,
    **GREEKS_BENCHMARKS,
    **AMERICAN_BENCHMARKS,
    **MONTE_CARLO_BENCHMARKS,
    **VOLATILITY_BENCHMARKS,
    **RISK_BENCHMARKS,
    **TECHNICAL_BENCHMARKS,
}

# Release gates - benchmarks that MUST pass for deployment
RELEASE_GATE_BENCHMARKS = {
    name: benchmark
    for name, benchmark in ALL_BENCHMARKS.items()
    if benchmark.is_release_gate
}


def check_tolerance(
    actual: float,
    expected: float | None,
    benchmark: BenchmarkTolerance
) -> tuple[bool, str]:
    """
    Check if actual value satisfies benchmark tolerance.

    Returns:
        Tuple of (passed: bool, message: str)
    """
    tol_type = benchmark.tolerance_type
    tol_value = benchmark.value

    if tol_type == ToleranceType.ABSOLUTE:
        if expected is None:
            return False, "Expected value required for absolute tolerance"
        error = abs(actual - expected)
        passed = error < tol_value
        return passed, f"Error {error:.2e} {'<' if passed else '>='} {tol_value:.2e}"

    elif tol_type == ToleranceType.RELATIVE:
        if expected is None or expected == 0:
            return False, "Expected value required for relative tolerance"
        error = abs(actual - expected) / abs(expected)
        passed = error < tol_value
        return passed, f"Relative error {error:.2%} {'<' if passed else '>='} {tol_value:.2%}"

    elif tol_type == ToleranceType.BOUNDS:
        lower, upper = tol_value
        passed = lower <= actual <= upper
        return passed, f"Value {actual:.4f} {'in' if passed else 'not in'} [{lower}, {upper}]"

    elif tol_type == ToleranceType.RANGE:
        lower, upper = tol_value
        passed = lower <= actual <= upper
        return passed, f"Value {actual:.4f} {'in' if passed else 'not in'} [{lower}, {upper}]"

    elif tol_type == ToleranceType.NON_NEGATIVE:
        passed = actual >= 0
        return passed, f"Value {actual:.4f} {'>=0' if passed else '<0'}"

    elif tol_type == ToleranceType.PROBABILITY:
        passed = 0 <= actual <= 1
        return passed, f"Value {actual:.4f} {'in' if passed else 'not in'} [0, 1]"

    return False, f"Unknown tolerance type: {tol_type}"


def get_benchmark(name: str) -> BenchmarkTolerance | None:
    """Get a benchmark by name."""
    return ALL_BENCHMARKS.get(name)


def list_benchmarks(category: str | None = None) -> list[str]:
    """List all benchmark names, optionally filtered by category."""
    categories = {
        "blackscholes": BLACKSCHOLES_BENCHMARKS,
        "greeks": GREEKS_BENCHMARKS,
        "american": AMERICAN_BENCHMARKS,
        "monte_carlo": MONTE_CARLO_BENCHMARKS,
        "volatility": VOLATILITY_BENCHMARKS,
        "risk": RISK_BENCHMARKS,
        "technical": TECHNICAL_BENCHMARKS,
    }

    if category and category in categories:
        return list(categories[category].keys())

    return list(ALL_BENCHMARKS.keys())


def generate_benchmark_report() -> dict[str, Any]:
    """Generate a summary report of all benchmarks."""
    report = {
        "total_benchmarks": len(ALL_BENCHMARKS),
        "release_gates": len(RELEASE_GATE_BENCHMARKS),
        "categories": {
            "blackscholes": len(BLACKSCHOLES_BENCHMARKS),
            "greeks": len(GREEKS_BENCHMARKS),
            "american": len(AMERICAN_BENCHMARKS),
            "monte_carlo": len(MONTE_CARLO_BENCHMARKS),
            "volatility": len(VOLATILITY_BENCHMARKS),
            "risk": len(RISK_BENCHMARKS),
            "technical": len(TECHNICAL_BENCHMARKS),
        },
        "benchmarks": [
            {
                "name": name,
                "type": b.tolerance_type.value,
                "value": b.value,
                "reference": b.reference,
                "is_release_gate": b.is_release_gate,
            }
            for name, b in ALL_BENCHMARKS.items()
        ]
    }
    return report
