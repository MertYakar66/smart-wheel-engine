"""
Stress Testing and Scenario Analysis Module

Professional risk scenarios for options portfolios:
- Historical stress tests (2008, 2020, etc.)
- Hypothetical scenarios (gap downs, IV spikes)
- Greeks-based sensitivity analysis
- Monte Carlo simulation
- Tail risk analysis

Key principle: Understand worst-case outcomes before they happen.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from .option_pricer import black_scholes_price


class ScenarioType(Enum):
    """Types of stress scenarios."""

    HISTORICAL = "historical"  # Based on actual historical events
    HYPOTHETICAL = "hypothetical"  # User-defined scenarios
    SENSITIVITY = "sensitivity"  # Greeks-based sensitivities
    MONTE_CARLO = "monte_carlo"  # Simulated scenarios


@dataclass
class Scenario:
    """Single stress scenario definition."""

    name: str
    scenario_type: ScenarioType
    description: str

    # Market shocks
    spot_change_pct: float = 0.0  # % change in underlying
    iv_change_pct: float = 0.0  # % change in IV (relative)
    iv_change_abs: float = 0.0  # Absolute change in IV
    rate_change_bps: float = 0.0  # Change in rates (basis points)
    time_decay_days: int = 0  # Days of time decay

    # Correlation assumptions
    correlation_shock: float = 1.0  # 1.0 = all assets move together

    def __str__(self) -> str:
        parts = [f"{self.name}:"]
        if self.spot_change_pct != 0:
            parts.append(f"Spot {self.spot_change_pct:+.1%}")
        if self.iv_change_pct != 0:
            parts.append(f"IV {self.iv_change_pct:+.1%}")
        if self.iv_change_abs != 0:
            parts.append(f"IV {self.iv_change_abs:+.0%} abs")
        if self.rate_change_bps != 0:
            parts.append(f"Rates {self.rate_change_bps:+.0f}bps")
        return " ".join(parts)


@dataclass
class ScenarioResult:
    """Result of applying a scenario."""

    scenario: Scenario
    portfolio_pnl: float  # Total P&L
    portfolio_pnl_pct: float  # P&L as % of portfolio
    position_pnls: dict[str, float]  # P&L by position
    new_portfolio_value: float
    new_greeks: dict | None = None

    # Risk metrics after scenario
    margin_call: bool = False
    max_loss_position: str | None = None
    max_loss_amount: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.scenario.name}: P&L ${self.portfolio_pnl:+,.0f} ({self.portfolio_pnl_pct:+.1%})"
        )


@dataclass
class StressTestReport:
    """Complete stress test report."""

    results: list[ScenarioResult]
    worst_case: ScenarioResult
    best_case: ScenarioResult
    expected_shortfall: float  # Average of worst 5%
    var_95: float  # 95% VaR from scenarios

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "Stress Test Report",
            "=" * 50,
            "",
            "Scenario Results:",
        ]

        for result in sorted(self.results, key=lambda r: r.portfolio_pnl):
            lines.append(f"  {result}")

        lines.extend(
            [
                "",
                f"Worst Case: {self.worst_case.scenario.name} "
                f"(${self.worst_case.portfolio_pnl:+,.0f})",
                f"Best Case:  {self.best_case.scenario.name} "
                f"(${self.best_case.portfolio_pnl:+,.0f})",
                f"95% VaR:    ${self.var_95:,.0f}",
                f"Exp. Shortfall: ${self.expected_shortfall:,.0f}",
            ]
        )

        return "\n".join(lines)


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = [
    Scenario(
        name="2008 Financial Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="Lehman collapse, Oct 2008",
        spot_change_pct=-0.20,
        iv_change_abs=0.40,  # VIX went to 80+
        correlation_shock=1.0,
    ),
    Scenario(
        name="2020 COVID Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="March 2020 COVID panic",
        spot_change_pct=-0.12,
        iv_change_abs=0.50,  # VIX hit 82
        correlation_shock=1.0,
    ),
    Scenario(
        name="2022 Rate Shock",
        scenario_type=ScenarioType.HISTORICAL,
        description="Fed aggressive rate hikes",
        spot_change_pct=-0.08,
        iv_change_abs=0.10,
        rate_change_bps=100,
    ),
    Scenario(
        name="Flash Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="May 2010 flash crash",
        spot_change_pct=-0.10,
        iv_change_abs=0.15,
        time_decay_days=0,  # Intraday
    ),
    Scenario(
        name="2011 Debt Ceiling",
        scenario_type=ScenarioType.HISTORICAL,
        description="US credit downgrade",
        spot_change_pct=-0.07,
        iv_change_abs=0.20,
    ),
    Scenario(
        name="2018 Volmageddon",
        scenario_type=ScenarioType.HISTORICAL,
        description="XIV collapse, Feb 2018",
        spot_change_pct=-0.04,
        iv_change_abs=0.25,  # VIX doubled
    ),
]

# Hypothetical scenarios for options selling
HYPOTHETICAL_SCENARIOS = [
    Scenario(
        name="Moderate Correction",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="5% pullback with IV spike",
        spot_change_pct=-0.05,
        iv_change_pct=0.30,
    ),
    Scenario(
        name="Sharp Selloff",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="10% drop, elevated IV",
        spot_change_pct=-0.10,
        iv_change_abs=0.25,
    ),
    Scenario(
        name="Crash Scenario",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="20% crash with extreme IV",
        spot_change_pct=-0.20,
        iv_change_abs=0.50,
    ),
    Scenario(
        name="Gap Down Assignment",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="15% overnight gap (earnings/event)",
        spot_change_pct=-0.15,
        iv_change_abs=0.20,
    ),
    Scenario(
        name="Slow Grind Down",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="8% decline over weeks, IV muted",
        spot_change_pct=-0.08,
        iv_change_pct=-0.10,
        time_decay_days=21,
    ),
    Scenario(
        name="IV Crush",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="IV collapses post-event",
        spot_change_pct=0.02,
        iv_change_pct=-0.40,
    ),
    Scenario(
        name="Rally",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="5% rally, IV declines",
        spot_change_pct=0.05,
        iv_change_pct=-0.20,
    ),
    Scenario(
        name="Strong Rally",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="10% rally",
        spot_change_pct=0.10,
        iv_change_pct=-0.30,
    ),
]


class StressTester:
    """
    Stress testing engine for options portfolios.

    Calculates P&L impact under various market scenarios.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        scenarios: list[Scenario] | None = None,
        residual_tolerance: float = 0.10,
    ):
        self.risk_free_rate = risk_free_rate
        self.scenarios = scenarios or (HISTORICAL_SCENARIOS + HYPOTHETICAL_SCENARIOS)
        self.residual_tolerance = residual_tolerance

    @classmethod
    def from_policy(cls) -> "StressTester":
        """Construct StressTester from centralized policy configuration."""
        from .policy_config import load_policy
        policy = load_policy()
        return cls(residual_tolerance=policy.greeks.decomposition_residual_tolerance)

    def run_scenario(
        self,
        scenario: Scenario,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
    ) -> ScenarioResult:
        """
        Run single scenario on portfolio.

        Args:
            scenario: Scenario to apply
            positions: List of position dicts with:
                - symbol, option_type, strike, dte, iv, contracts, is_short
            spot_prices: Current spot prices by symbol
            portfolio_value: Current portfolio value

        Returns:
            ScenarioResult with P&L details
        """
        total_pnl = 0.0
        position_pnls = {}

        for pos in positions:
            symbol = pos["symbol"]
            current_spot = spot_prices.get(symbol, pos.get("underlying_price", 100))

            # Apply scenario shocks
            new_spot = current_spot * (1 + scenario.spot_change_pct)

            current_iv = pos["iv"]
            if scenario.iv_change_abs != 0:
                new_iv = current_iv + scenario.iv_change_abs
            else:
                new_iv = current_iv * (1 + scenario.iv_change_pct)
            new_iv = max(0.01, new_iv)  # Floor at 1%

            new_rate = self.risk_free_rate + (scenario.rate_change_bps / 10000)

            # Adjust DTE for time decay
            new_dte = max(0, pos["dte"] - scenario.time_decay_days)

            # Calculate current value
            current_price = black_scholes_price(
                S=current_spot,
                K=pos["strike"],
                T=pos["dte"] / 365,
                r=self.risk_free_rate,
                sigma=current_iv,
                option_type=pos["option_type"],
                q=pos.get("dividend_yield", 0.0),
            )

            # Calculate new value under scenario
            if new_dte <= 0:
                # Expired - intrinsic value
                if pos["option_type"] == "put":
                    new_price = max(0, pos["strike"] - new_spot)
                else:
                    new_price = max(0, new_spot - pos["strike"])
            else:
                new_price = black_scholes_price(
                    S=new_spot,
                    K=pos["strike"],
                    T=new_dte / 365,
                    r=new_rate,
                    sigma=new_iv,
                    option_type=pos["option_type"],
                    q=pos.get("dividend_yield", 0.0),
                )

            # P&L calculation
            direction = -1 if pos.get("is_short", True) else 1
            price_change = new_price - current_price
            position_pnl = direction * price_change * pos["contracts"] * 100

            position_pnls[f"{symbol}_{pos['option_type']}_{pos['strike']}"] = position_pnl
            total_pnl += position_pnl

        # Find worst position
        if position_pnls:
            worst_pos = min(position_pnls.items(), key=lambda x: x[1])
            max_loss_position = worst_pos[0]
            max_loss_amount = worst_pos[1]
        else:
            max_loss_position = None
            max_loss_amount = 0.0

        # Check for margin call (simplified)
        new_value = portfolio_value + total_pnl
        margin_call = new_value < portfolio_value * 0.25  # Below 25% = margin call

        return ScenarioResult(
            scenario=scenario,
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=total_pnl / portfolio_value if portfolio_value > 0 else 0,
            position_pnls=position_pnls,
            new_portfolio_value=new_value,
            margin_call=margin_call,
            max_loss_position=max_loss_position,
            max_loss_amount=max_loss_amount,
        )

    def run_all_scenarios(
        self,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
        scenarios: list[Scenario] | None = None,
    ) -> StressTestReport:
        """Run all scenarios and generate report."""
        scenarios = scenarios or self.scenarios
        results = []

        for scenario in scenarios:
            result = self.run_scenario(scenario, positions, spot_prices, portfolio_value)
            results.append(result)

        # Sort by P&L
        sorted_results = sorted(results, key=lambda r: r.portfolio_pnl)

        # Calculate risk metrics
        pnls = [r.portfolio_pnl for r in results]
        worst_5_pct = int(max(1, len(pnls) * 0.05))
        expected_shortfall = np.mean(sorted(pnls)[:worst_5_pct])
        var_95 = np.percentile(pnls, 5)  # 5th percentile = 95% VaR

        return StressTestReport(
            results=results,
            worst_case=sorted_results[0],
            best_case=sorted_results[-1],
            expected_shortfall=abs(expected_shortfall),
            var_95=abs(var_95),
        )

    def sensitivity_analysis(
        self,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
        spot_range: tuple[float, float] = (-0.15, 0.15),
        iv_range: tuple[float, float] = (-0.30, 0.30),
        n_points: int = 11,
    ) -> pd.DataFrame:
        """
        Generate sensitivity grid for spot and IV changes.

        Returns DataFrame with P&L for each spot/IV combination.
        """
        spot_changes = np.linspace(spot_range[0], spot_range[1], n_points)
        iv_changes = np.linspace(iv_range[0], iv_range[1], n_points)

        results = []
        for spot_chg in spot_changes:
            for iv_chg in iv_changes:
                scenario = Scenario(
                    name=f"Spot {spot_chg:+.0%} IV {iv_chg:+.0%}",
                    scenario_type=ScenarioType.SENSITIVITY,
                    description="Sensitivity grid",
                    spot_change_pct=spot_chg,
                    iv_change_pct=iv_chg,
                )
                result = self.run_scenario(scenario, positions, spot_prices, portfolio_value)
                results.append(
                    {
                        "spot_change": spot_chg,
                        "iv_change": iv_chg,
                        "pnl": result.portfolio_pnl,
                        "pnl_pct": result.portfolio_pnl_pct,
                    }
                )

        return pd.DataFrame(results)

    def monte_carlo_stress(
        self,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
        n_simulations: int = 10000,
        horizon_days: int = 30,
        vol_of_vol: float = 0.50,
    ) -> dict[str, float]:
        """
        Monte Carlo stress testing.

        Simulates random market moves to estimate tail risk.
        """
        pnls = []

        for _ in range(n_simulations):
            # Simulate spot change (fat-tailed)
            # Use t-distribution for fatter tails
            df = 5  # degrees of freedom
            z = stats.t.rvs(df)
            avg_iv = np.mean([p["iv"] for p in positions]) if positions else 0.20
            daily_vol = avg_iv / np.sqrt(252)
            spot_change = z * daily_vol * np.sqrt(horizon_days)

            # Simulate IV change (correlated with spot move)
            iv_shock = -spot_change * 2  # IV typically moves opposite to spot
            iv_shock += np.random.normal(0, vol_of_vol * avg_iv)

            scenario = Scenario(
                name="MC",
                scenario_type=ScenarioType.MONTE_CARLO,
                description="Monte Carlo simulation",
                spot_change_pct=spot_change,
                iv_change_pct=iv_shock / avg_iv if avg_iv > 0 else 0,
                time_decay_days=horizon_days,
            )

            result = self.run_scenario(scenario, positions, spot_prices, portfolio_value)
            pnls.append(result.portfolio_pnl)

        pnls = np.array(pnls)

        return {
            "mean": np.mean(pnls),
            "std": np.std(pnls),
            "var_95": np.percentile(pnls, 5),
            "var_99": np.percentile(pnls, 1),
            "cvar_95": np.mean(pnls[pnls <= np.percentile(pnls, 5)]),
            "max_loss": np.min(pnls),
            "max_gain": np.max(pnls),
            "prob_loss": np.mean(pnls < 0),
            "prob_10pct_loss": np.mean(pnls < -portfolio_value * 0.10),
        }

    def greeks_stress_ladder(
        self,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
        spot_range: tuple[float, float] = (-0.20, 0.20),
        n_steps: int = 21,
        iv_shock: float = 0.0,
        dte_decay: int = 0,
    ) -> pd.DataFrame:
        """
        Greeks stress ladder: P&L decomposition across spot prices.

        Shows how each Greek contributes to P&L as the underlying moves.
        Essential for understanding portfolio risk profile.

        Args:
            positions: List of position dictionaries
            spot_prices: Current spot prices per symbol
            portfolio_value: Total portfolio value
            spot_range: Range of spot moves (default: -20% to +20%)
            n_steps: Number of price points in ladder
            iv_shock: Optional IV change to apply (default: 0)
            dte_decay: Days of time decay to apply (default: 0)

        Returns:
            DataFrame with columns: spot_change, total_pnl, delta_pnl,
            gamma_pnl, theta_pnl, vega_pnl, rho_pnl

        Example usage:
            ladder = tester.greeks_stress_ladder(positions, spots, 100000)
            print(ladder[['spot_change', 'total_pnl', 'delta_pnl', 'gamma_pnl']])
        """
        from .option_pricer import black_scholes_all_greeks

        spot_changes = np.linspace(spot_range[0], spot_range[1], n_steps)
        results = []

        for spot_chg in spot_changes:
            total_delta_pnl = 0.0
            total_gamma_pnl = 0.0
            total_theta_pnl = 0.0
            total_vega_pnl = 0.0
            total_rho_pnl = 0.0
            total_pnl = 0.0

            for pos in positions:
                symbol = pos["symbol"]
                spot = spot_prices.get(symbol, pos.get("underlying_price", 100))
                new_spot = spot * (1 + spot_chg)

                strike = pos["strike"]
                original_dte = pos["dte"]
                T = max(0.001, original_dte / 365)
                iv = pos["iv"] * (1 + iv_shock) if iv_shock else pos["iv"]
                option_type = pos["option_type"]
                contracts = pos["contracts"]
                is_short = pos.get("is_short", True)
                r = pos.get("rate", 0.05)
                q = pos.get("dividend_yield", 0.0)

                direction = -1 if is_short else 1
                multiplier = contracts * 100 * direction

                # Calculate current Greeks at original T (before any decay)
                greeks = black_scholes_all_greeks(
                    S=spot, K=strike, T=T, r=r, sigma=iv, option_type=option_type, q=q
                )

                # Calculate new price at shocked spot and decayed time
                decayed_dte = max(0.001, original_dte - dte_decay) if dte_decay else original_dte
                T_new = max(0.001, decayed_dte / 365)
                new_greeks = black_scholes_all_greeks(
                    S=new_spot, K=strike, T=T_new, r=r, sigma=iv, option_type=option_type, q=q
                )

                # P&L decomposition using Taylor expansion
                dS = new_spot - spot
                delta_pnl = greeks["delta"] * dS * multiplier
                gamma_pnl = 0.5 * greeks["gamma"] * dS**2 * multiplier
                # Theta from pricer is annual; convert to daily before scaling by days
                theta_pnl = (greeks["theta"] / 365) * dte_decay * multiplier if dte_decay else 0
                # Vega P&L: iv_shock is relative change (e.g., 0.15 = 15% increase in IV level)
                # Convert to vol points: base_iv * relative_change * 100
                # See docs/GREEKS_UNIT_CONTRACT.md for unit conventions
                iv_change_vol_points = pos["iv"] * iv_shock * 100 if iv_shock else 0
                vega_pnl = greeks["vega"] * iv_change_vol_points * multiplier
                rho_pnl = 0  # No rate shock in this scenario

                # Full repricing P&L (most accurate)
                old_price = greeks["price"]
                new_price = new_greeks["price"]
                actual_pnl = (new_price - old_price) * multiplier

                # Higher-order residual
                greeks_sum = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
                residual = actual_pnl - greeks_sum

                total_delta_pnl += delta_pnl
                total_gamma_pnl += gamma_pnl
                total_theta_pnl += theta_pnl
                total_vega_pnl += vega_pnl
                total_rho_pnl += rho_pnl
                total_pnl += actual_pnl

            # Decomposition residual: repricing P&L minus sum of Greek P&Ls
            greeks_total = (total_delta_pnl + total_gamma_pnl + total_theta_pnl
                           + total_vega_pnl + total_rho_pnl)
            decomp_residual = total_pnl - greeks_total
            residual_pct = (abs(decomp_residual) / abs(total_pnl)
                           if abs(total_pnl) > 1e-6 else 0.0)

            results.append({
                "spot_change": spot_chg,
                "spot_change_pct": f"{spot_chg:+.1%}",
                "total_pnl": total_pnl,
                "delta_pnl": total_delta_pnl,
                "gamma_pnl": total_gamma_pnl,
                "theta_pnl": total_theta_pnl,
                "vega_pnl": total_vega_pnl,
                "rho_pnl": total_rho_pnl,
                "decomp_residual": decomp_residual,
                "residual_pct": residual_pct,
                "pnl_pct": total_pnl / portfolio_value if portfolio_value > 0 else 0,
            })

        df = pd.DataFrame(results)

        # Alert if residual exceeds tolerance (default 10%)
        residual_tolerance = getattr(self, 'residual_tolerance', 0.10)
        high_residual = df[df["residual_pct"] > residual_tolerance]
        if not high_residual.empty:
            import warnings
            max_residual = high_residual["residual_pct"].max()
            warnings.warn(
                f"Greeks decomposition residual exceeds {residual_tolerance:.0%} tolerance "
                f"at {len(high_residual)} spot points (max={max_residual:.1%}). "
                f"Higher-order effects may be material.",
                stacklevel=2,
            )

        return df

    def greeks_scenario_matrix(
        self,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
        spot_shocks: list[float] | None = None,
        iv_shocks: list[float] | None = None,
        time_shocks: list[int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Comprehensive Greeks scenario matrix.

        Tests portfolio across multiple dimensions simultaneously:
        - Spot price shocks
        - IV shocks
        - Time decay scenarios

        Returns matrices for:
        - P&L surface (spot vs IV)
        - Greeks evolution (how Greeks change with spot)
        - Time decay profile

        Args:
            positions: List of position dictionaries
            spot_prices: Current spot prices per symbol
            portfolio_value: Total portfolio value
            spot_shocks: List of spot changes (default: -20% to +20%)
            iv_shocks: List of IV changes (default: -30% to +50%)
            time_shocks: List of DTE values (default: current, 7d, 14d, 30d decay)

        Returns:
            Dict with 'pnl_surface', 'greeks_surface', 'time_decay' DataFrames
        """
        from .option_pricer import black_scholes_all_greeks

        if spot_shocks is None:
            spot_shocks = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
        if iv_shocks is None:
            iv_shocks = [-0.30, -0.15, 0, 0.15, 0.30, 0.50]
        if time_shocks is None:
            time_shocks = [0, 7, 14, 30]

        # === P&L Surface (Spot x IV) ===
        pnl_rows = []
        for spot_chg in spot_shocks:
            row_data = {"spot_change": spot_chg}
            for iv_chg in iv_shocks:
                total_pnl = 0.0
                for pos in positions:
                    symbol = pos["symbol"]
                    spot = spot_prices.get(symbol, pos.get("underlying_price", 100))
                    new_spot = spot * (1 + spot_chg)
                    new_iv = pos["iv"] * (1 + iv_chg)

                    T = max(0.001, pos["dte"] / 365)
                    r = pos.get("rate", 0.05)
                    q = pos.get("dividend_yield", 0.0)

                    direction = -1 if pos.get("is_short", True) else 1
                    multiplier = pos["contracts"] * 100 * direction

                    old_greeks = black_scholes_all_greeks(
                        S=spot, K=pos["strike"], T=T, r=r,
                        sigma=pos["iv"], option_type=pos["option_type"], q=q
                    )
                    new_greeks = black_scholes_all_greeks(
                        S=new_spot, K=pos["strike"], T=T, r=r,
                        sigma=new_iv, option_type=pos["option_type"], q=q
                    )
                    total_pnl += (new_greeks["price"] - old_greeks["price"]) * multiplier

                row_data[f"iv_{iv_chg:+.0%}"] = total_pnl
            pnl_rows.append(row_data)

        pnl_surface = pd.DataFrame(pnl_rows)

        # === Greeks Surface (how Greeks change with spot) ===
        greeks_rows = []
        for spot_chg in spot_shocks:
            total_delta = 0.0
            total_gamma = 0.0
            total_theta = 0.0
            total_vega = 0.0

            for pos in positions:
                symbol = pos["symbol"]
                spot = spot_prices.get(symbol, pos.get("underlying_price", 100))
                new_spot = spot * (1 + spot_chg)

                T = max(0.001, pos["dte"] / 365)
                r = pos.get("rate", 0.05)
                q = pos.get("dividend_yield", 0.0)

                direction = -1 if pos.get("is_short", True) else 1
                multiplier = pos["contracts"] * 100 * direction

                greeks = black_scholes_all_greeks(
                    S=new_spot, K=pos["strike"], T=T, r=r,
                    sigma=pos["iv"], option_type=pos["option_type"], q=q
                )

                total_delta += greeks["delta"] * multiplier
                total_gamma += greeks["gamma"] * multiplier * new_spot
                total_theta += greeks["theta"] * multiplier
                total_vega += greeks["vega"] * multiplier

            greeks_rows.append({
                "spot_change": spot_chg,
                "delta": total_delta,
                "delta_dollars": total_delta * spot_prices.get(
                    positions[0]["symbol"], 100) if positions else 0,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega,
            })

        greeks_surface = pd.DataFrame(greeks_rows)

        # === Time Decay Profile ===
        time_rows = []
        for days in time_shocks:
            total_pnl = 0.0
            total_theta = 0.0

            for pos in positions:
                symbol = pos["symbol"]
                spot = spot_prices.get(symbol, pos.get("underlying_price", 100))
                dte_new = max(0.001, pos["dte"] - days)

                T_old = max(0.001, pos["dte"] / 365)
                T_new = dte_new / 365
                r = pos.get("rate", 0.05)
                q = pos.get("dividend_yield", 0.0)

                direction = -1 if pos.get("is_short", True) else 1
                multiplier = pos["contracts"] * 100 * direction

                old_greeks = black_scholes_all_greeks(
                    S=spot, K=pos["strike"], T=T_old, r=r,
                    sigma=pos["iv"], option_type=pos["option_type"], q=q
                )
                new_greeks = black_scholes_all_greeks(
                    S=spot, K=pos["strike"], T=T_new, r=r,
                    sigma=pos["iv"], option_type=pos["option_type"], q=q
                )

                total_pnl += (new_greeks["price"] - old_greeks["price"]) * multiplier
                total_theta += new_greeks["theta"] * multiplier

            time_rows.append({
                "days_elapsed": days,
                "cumulative_pnl": total_pnl,
                "pnl_pct": total_pnl / portfolio_value if portfolio_value > 0 else 0,
                "remaining_theta": total_theta,
            })

        time_decay = pd.DataFrame(time_rows)

        return {
            "pnl_surface": pnl_surface,
            "greeks_surface": greeks_surface,
            "time_decay": time_decay,
        }

    def extreme_greeks_scenarios(
        self,
        positions: list[dict],
        spot_prices: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, dict]:
        """
        Run extreme Greeks-focused stress scenarios.

        Tests portfolio under severe but plausible market conditions
        that stress specific Greek exposures.

        Scenarios:
        1. Black Monday (1987): -22% spot, +300% vol
        2. COVID Crash (2020): -35% over 4 weeks, +400% vol
        3. Flash Crash: -10% instant, +200% vol, immediate recovery
        4. Vol Crush: -50% vol (earnings resolution)
        5. Rate Shock: +200bps overnight
        6. Gamma Squeeze: +30% spot, +50% vol (meme stock style)
        7. Theta Burn: 30-day time decay, no spot/vol change
        8. Weekend Gap: -8% gap, +80% vol

        Returns:
            Dict of scenario_name -> detailed results including
            P&L breakdown by Greek
        """
        from .option_pricer import black_scholes_all_greeks

        scenarios = {
            "black_monday_1987": {
                "spot_change": -0.22,
                "iv_change": 3.0,  # 300% increase
                "rate_change": 0.0,
                "days_decay": 0,
                "description": "Black Monday 1987: -22% spot, IV triples",
            },
            "covid_crash_2020": {
                "spot_change": -0.35,
                "iv_change": 4.0,  # 400% increase (VIX hit 82)
                "rate_change": -0.01,  # Fed cut rates
                "days_decay": 20,
                "description": "COVID crash: -35%, IV quadruples over 20 days",
            },
            "flash_crash": {
                "spot_change": -0.10,
                "iv_change": 2.0,
                "rate_change": 0.0,
                "days_decay": 0,
                "description": "Flash crash: -10% instant, IV doubles",
            },
            "vol_crush": {
                "spot_change": 0.02,
                "iv_change": -0.50,  # 50% vol decrease
                "rate_change": 0.0,
                "days_decay": 1,
                "description": "Vol crush: earnings resolution, -50% IV",
            },
            "rate_shock_hawkish": {
                "spot_change": -0.03,
                "iv_change": 0.20,
                "rate_change": 0.02,  # +200bps
                "days_decay": 0,
                "description": "Hawkish Fed: +200bps rates, -3% spot",
            },
            "gamma_squeeze": {
                "spot_change": 0.30,
                "iv_change": 0.50,
                "rate_change": 0.0,
                "days_decay": 0,
                "description": "Gamma squeeze: +30% spot, +50% IV",
            },
            "theta_burn": {
                "spot_change": 0.0,
                "iv_change": 0.0,
                "rate_change": 0.0,
                "days_decay": 30,
                "description": "Pure theta: 30 days decay, no spot/vol change",
            },
            "weekend_gap": {
                "spot_change": -0.08,
                "iv_change": 0.80,
                "rate_change": 0.0,
                "days_decay": 2,
                "description": "Weekend gap: -8% gap, +80% IV",
            },
        }

        results = {}

        for scenario_name, params in scenarios.items():
            total_pnl = 0.0
            delta_pnl = 0.0
            gamma_pnl = 0.0
            theta_pnl = 0.0
            vega_pnl = 0.0
            rho_pnl = 0.0
            position_details = []

            for pos in positions:
                symbol = pos["symbol"]
                spot = spot_prices.get(symbol, pos.get("underlying_price", 100))
                new_spot = spot * (1 + params["spot_change"])
                new_iv = pos["iv"] * (1 + params["iv_change"])
                new_rate = pos.get("rate", 0.05) + params["rate_change"]

                T_old = max(0.001, pos["dte"] / 365)
                T_new = max(0.001, (pos["dte"] - params["days_decay"]) / 365)
                q = pos.get("dividend_yield", 0.0)

                direction = -1 if pos.get("is_short", True) else 1
                multiplier = pos["contracts"] * 100 * direction

                # Current Greeks
                old_greeks = black_scholes_all_greeks(
                    S=spot, K=pos["strike"], T=T_old, r=pos.get("rate", 0.05),
                    sigma=pos["iv"], option_type=pos["option_type"], q=q
                )

                # Stressed Greeks
                new_greeks = black_scholes_all_greeks(
                    S=new_spot, K=pos["strike"], T=T_new, r=new_rate,
                    sigma=new_iv, option_type=pos["option_type"], q=q
                )

                # Full repricing P&L
                pos_pnl = (new_greeks["price"] - old_greeks["price"]) * multiplier

                # Greeks decomposition
                # Per GREEKS_UNIT_CONTRACT.md:
                # - Theta: pricer returns annual theta, convert to daily with /365
                # - Vega: pricer returns per 1% (vol point), IV change in decimal needs *100
                # - Rho: pricer returns per 1%, rate change in decimal needs *100
                dS = new_spot - spot
                pos_delta_pnl = old_greeks["delta"] * dS * multiplier
                pos_gamma_pnl = 0.5 * old_greeks["gamma"] * dS**2 * multiplier
                pos_theta_pnl = (old_greeks["theta"] / 365) * params["days_decay"] * multiplier
                pos_vega_pnl = old_greeks["vega"] * (new_iv - pos["iv"]) * 100 * multiplier
                pos_rho_pnl = old_greeks["rho"] * params["rate_change"] * 100 * multiplier

                delta_pnl += pos_delta_pnl
                gamma_pnl += pos_gamma_pnl
                theta_pnl += pos_theta_pnl
                vega_pnl += pos_vega_pnl
                rho_pnl += pos_rho_pnl
                total_pnl += pos_pnl

                position_details.append({
                    "symbol": symbol,
                    "pnl": pos_pnl,
                    "delta_contrib": pos_delta_pnl,
                    "gamma_contrib": pos_gamma_pnl,
                    "theta_contrib": pos_theta_pnl,
                    "vega_contrib": pos_vega_pnl,
                })

            results[scenario_name] = {
                "description": params["description"],
                "total_pnl": total_pnl,
                "pnl_pct": total_pnl / portfolio_value if portfolio_value > 0 else 0,
                "greek_attribution": {
                    "delta_pnl": delta_pnl,
                    "gamma_pnl": gamma_pnl,
                    "theta_pnl": theta_pnl,
                    "vega_pnl": vega_pnl,
                    "rho_pnl": rho_pnl,
                    "higher_order": total_pnl - (delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl),
                },
                "position_details": position_details,
                "params": params,
            }

        return results


def quick_stress_test(
    positions: list[dict], spot_prices: dict[str, float], portfolio_value: float
) -> str:
    """
    Quick stress test with standard scenarios.

    Returns formatted summary string.
    """
    tester = StressTester()
    report = tester.run_all_scenarios(positions, spot_prices, portfolio_value)
    return report.summary()


def calculate_max_loss(positions: list[dict], spot_prices: dict[str, float]) -> float:
    """
    Calculate theoretical maximum loss for portfolio.

    For short puts: max loss = strike * 100 * contracts (stock goes to 0)
    For short calls: max loss = unlimited (capped at reasonable level)
    """
    max_loss = 0.0

    for pos in positions:
        if pos.get("is_short", True):
            if pos["option_type"] == "put":
                # Max loss: stock goes to 0
                pos_max_loss = pos["strike"] * 100 * pos["contracts"]
            else:
                # Short call: cap at 5x current price
                current_price = spot_prices.get(pos["symbol"], pos["strike"])
                pos_max_loss = current_price * 5 * 100 * pos["contracts"]
        else:
            # Long positions: max loss = premium paid (need entry price)
            pos_max_loss = pos.get("entry_price", 0) * 100 * pos["contracts"]

        max_loss += pos_max_loss

    return max_loss
