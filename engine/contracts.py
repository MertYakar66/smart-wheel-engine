"""
Interface contracts for core engine modules.

Defines Protocol classes that document the expected interfaces for option pricers,
risk managers, and stress testers. Use these for type checking, testing, and
ensuring module interoperability.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd


# =============================================================================
# Contract Constants
# =============================================================================

REQUIRED_GREEK_KEYS: frozenset[str] = frozenset(
    {"price", "delta", "gamma", "theta", "vega", "rho"}
)

REQUIRED_LADDER_COLUMNS: frozenset[str] = frozenset(
    {"spot_change", "total_pnl", "delta_pnl", "gamma_pnl", "theta_pnl", "vega_pnl"}
)


# =============================================================================
# Protocol Definitions
# =============================================================================


@runtime_checkable
class PricerContract(Protocol):
    """Interface contract for option pricers (e.g. Black-Scholes, Monte Carlo)."""

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
    ) -> float:
        """Return the fair value of a European option."""
        ...

    def all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
    ) -> dict[str, float]:
        """
        Return a dict of Greeks for the option.

        The returned dict must contain at least the keys in REQUIRED_GREEK_KEYS:
        price, delta, gamma, theta, vega, rho.
        """
        ...


@runtime_checkable
class RiskContract(Protocol):
    """Interface contract for risk managers."""

    def calculate_position_size(
        self,
        portfolio_value: float,
        underlying_price: float,
        strike: float,
        iv: float,
        dte: int,
        win_probability: float = 0.70,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        existing_positions: int = 0,
        underlying_correlation: float = 0.0,
    ) -> tuple[int, str]:
        """
        Calculate the number of contracts to trade and a sizing rationale string.

        Returns:
            (num_contracts, rationale) where rationale explains the sizing decision.
        """
        ...

    def calculate_portfolio_greeks(
        self,
        positions: list[dict[str, Any]],
        spot_prices: dict[str, float],
    ) -> Any:
        """
        Aggregate Greeks across all portfolio positions.

        Args:
            positions: List of position dicts (symbol, option_type, strike, etc.).
            spot_prices: Mapping of symbol to current spot price.

        Returns:
            A PortfolioGreeks-compatible object with delta, gamma, theta, vega, rho.
        """
        ...

    def calculate_var(
        self,
        portfolio_value: float,
        positions: list[dict[str, Any]],
        spot_prices: dict[str, float],
        returns_data: pd.DataFrame | None = None,
        volatilities: dict[str, float] | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR.

        Returns:
            (var, cvar) as dollar amounts.
        """
        ...


@runtime_checkable
class StressTesterContract(Protocol):
    """Interface contract for stress testers."""

    def run_scenario(
        self,
        scenario: Any,
        positions: list[dict[str, Any]],
        spot_prices: dict[str, float],
        portfolio_value: float,
    ) -> Any:
        """
        Run a single stress scenario against the portfolio.

        Args:
            scenario: A Scenario instance defining the market shock.
            positions: Current portfolio positions.
            spot_prices: Current spot prices by symbol.
            portfolio_value: Current total portfolio value.

        Returns:
            A ScenarioResult with P&L breakdown and risk flags.
        """
        ...

    def greeks_stress_ladder(
        self,
        positions: list[dict[str, Any]],
        spot_prices: dict[str, float],
        portfolio_value: float,
        spot_range: tuple[float, float] = (-0.20, 0.20),
        n_steps: int = 21,
        iv_shock: float = 0.0,
        dte_decay: int = 0,
    ) -> pd.DataFrame:
        """
        Produce a P&L decomposition ladder across spot price changes.

        The returned DataFrame must contain at least the columns in
        REQUIRED_LADDER_COLUMNS: spot_change, total_pnl, delta_pnl,
        gamma_pnl, theta_pnl, vega_pnl.
        """
        ...


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_greeks_output(greeks: dict[str, float]) -> list[str]:
    """
    Validate a greeks dict against the contract.

    Returns:
        List of error strings. Empty list means the output is valid.
    """
    errors: list[str] = []

    if not isinstance(greeks, dict):
        return [f"Expected dict, got {type(greeks).__name__}"]

    missing = REQUIRED_GREEK_KEYS - greeks.keys()
    if missing:
        errors.append(f"Missing required keys: {sorted(missing)}")

    for key in REQUIRED_GREEK_KEYS & greeks.keys():
        val = greeks[key]
        if not isinstance(val, (int, float)):
            errors.append(f"Key '{key}' must be numeric, got {type(val).__name__}")
        elif isinstance(val, float) and (val != val):  # NaN check
            errors.append(f"Key '{key}' is NaN")

    return errors


def validate_ladder_output(df: pd.DataFrame) -> list[str]:
    """
    Validate a stress-ladder DataFrame against the contract.

    Returns:
        List of error strings. Empty list means the output is valid.
    """
    errors: list[str] = []

    if not isinstance(df, pd.DataFrame):
        return [f"Expected DataFrame, got {type(df).__name__}"]

    if df.empty:
        errors.append("DataFrame is empty")
        return errors

    missing = REQUIRED_LADDER_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    for col in REQUIRED_LADDER_COLUMNS & set(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")

    return errors


def validate_greeks_semantics(greeks: dict[str, float], option_type: str) -> list[str]:
    """
    Semantic validation of Greeks values (unit and sign constraints).

    Goes beyond structural checks to verify financial correctness:
    - Delta sign matches option type
    - Gamma is non-negative
    - Vega is non-negative
    - Theta convention is annual (magnitude check)
    - Price is non-negative

    Args:
        greeks: Dict from pricer with at least REQUIRED_GREEK_KEYS
        option_type: 'call' or 'put'

    Returns:
        List of semantic violation strings. Empty = valid.
    """
    errors: list[str] = []

    price = greeks.get("price", 0)
    delta = greeks.get("delta", 0)
    gamma = greeks.get("gamma", 0)
    vega = greeks.get("vega", 0)
    theta = greeks.get("theta", 0)

    # Price must be non-negative
    if price < -1e-10:
        errors.append(f"price={price:.6f} is negative")

    # Delta sign constraint
    if option_type == "call" and delta < -0.01:
        errors.append(f"call delta={delta:.6f} should be >= 0")
    if option_type == "put" and delta > 0.01:
        errors.append(f"put delta={delta:.6f} should be <= 0")

    # Delta magnitude
    if abs(delta) > 1.01:
        errors.append(f"|delta|={abs(delta):.6f} exceeds 1.0 bound")

    # Gamma non-negative (for vanilla options)
    if gamma < -1e-10:
        errors.append(f"gamma={gamma:.6f} is negative (should be >= 0 for vanilla)")

    # Vega non-negative (for vanilla options, per vol-point convention)
    if vega < -1e-10:
        errors.append(f"vega={vega:.6f} is negative (should be >= 0 for vanilla)")

    # Theta convention check: annual theta for ATM options is typically
    # in the range [-50, 0] for long options. If |theta| > 100,
    # it may indicate incorrect units (daily instead of annual).
    # This is a soft heuristic, not a hard constraint.
    if abs(theta) > 500 and abs(delta) > 0.1:
        errors.append(
            f"theta={theta:.2f} has unusually large magnitude — "
            f"verify annual convention (not daily)"
        )

    return errors
