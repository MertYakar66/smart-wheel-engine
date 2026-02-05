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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

from .option_pricer import black_scholes_price, black_scholes_all_greeks


class ScenarioType(Enum):
    """Types of stress scenarios."""
    HISTORICAL = "historical"       # Based on actual historical events
    HYPOTHETICAL = "hypothetical"   # User-defined scenarios
    SENSITIVITY = "sensitivity"     # Greeks-based sensitivities
    MONTE_CARLO = "monte_carlo"     # Simulated scenarios


@dataclass
class Scenario:
    """Single stress scenario definition."""
    name: str
    scenario_type: ScenarioType
    description: str

    # Market shocks
    spot_change_pct: float = 0.0      # % change in underlying
    iv_change_pct: float = 0.0        # % change in IV (relative)
    iv_change_abs: float = 0.0        # Absolute change in IV
    rate_change_bps: float = 0.0      # Change in rates (basis points)
    time_decay_days: int = 0          # Days of time decay

    # Correlation assumptions
    correlation_shock: float = 1.0    # 1.0 = all assets move together

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
    portfolio_pnl: float              # Total P&L
    portfolio_pnl_pct: float          # P&L as % of portfolio
    position_pnls: Dict[str, float]   # P&L by position
    new_portfolio_value: float
    new_greeks: Optional[Dict] = None

    # Risk metrics after scenario
    margin_call: bool = False
    max_loss_position: Optional[str] = None
    max_loss_amount: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.scenario.name}: P&L ${self.portfolio_pnl:+,.0f} "
            f"({self.portfolio_pnl_pct:+.1%})"
        )


@dataclass
class StressTestReport:
    """Complete stress test report."""
    results: List[ScenarioResult]
    worst_case: ScenarioResult
    best_case: ScenarioResult
    expected_shortfall: float         # Average of worst 5%
    var_95: float                     # 95% VaR from scenarios

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

        lines.extend([
            "",
            f"Worst Case: {self.worst_case.scenario.name} "
            f"(${self.worst_case.portfolio_pnl:+,.0f})",
            f"Best Case:  {self.best_case.scenario.name} "
            f"(${self.best_case.portfolio_pnl:+,.0f})",
            f"95% VaR:    ${self.var_95:,.0f}",
            f"Exp. Shortfall: ${self.expected_shortfall:,.0f}",
        ])

        return "\n".join(lines)


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = [
    Scenario(
        name="2008 Financial Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="Lehman collapse, Oct 2008",
        spot_change_pct=-0.20,
        iv_change_abs=0.40,  # VIX went to 80+
        correlation_shock=1.0
    ),
    Scenario(
        name="2020 COVID Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="March 2020 COVID panic",
        spot_change_pct=-0.12,
        iv_change_abs=0.50,  # VIX hit 82
        correlation_shock=1.0
    ),
    Scenario(
        name="2022 Rate Shock",
        scenario_type=ScenarioType.HISTORICAL,
        description="Fed aggressive rate hikes",
        spot_change_pct=-0.08,
        iv_change_abs=0.10,
        rate_change_bps=100
    ),
    Scenario(
        name="Flash Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="May 2010 flash crash",
        spot_change_pct=-0.10,
        iv_change_abs=0.15,
        time_decay_days=0  # Intraday
    ),
    Scenario(
        name="2011 Debt Ceiling",
        scenario_type=ScenarioType.HISTORICAL,
        description="US credit downgrade",
        spot_change_pct=-0.07,
        iv_change_abs=0.20
    ),
    Scenario(
        name="2018 Volmageddon",
        scenario_type=ScenarioType.HISTORICAL,
        description="XIV collapse, Feb 2018",
        spot_change_pct=-0.04,
        iv_change_abs=0.25  # VIX doubled
    ),
]

# Hypothetical scenarios for options selling
HYPOTHETICAL_SCENARIOS = [
    Scenario(
        name="Moderate Correction",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="5% pullback with IV spike",
        spot_change_pct=-0.05,
        iv_change_pct=0.30
    ),
    Scenario(
        name="Sharp Selloff",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="10% drop, elevated IV",
        spot_change_pct=-0.10,
        iv_change_abs=0.25
    ),
    Scenario(
        name="Crash Scenario",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="20% crash with extreme IV",
        spot_change_pct=-0.20,
        iv_change_abs=0.50
    ),
    Scenario(
        name="Gap Down Assignment",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="15% overnight gap (earnings/event)",
        spot_change_pct=-0.15,
        iv_change_abs=0.20
    ),
    Scenario(
        name="Slow Grind Down",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="8% decline over weeks, IV muted",
        spot_change_pct=-0.08,
        iv_change_pct=-0.10,
        time_decay_days=21
    ),
    Scenario(
        name="IV Crush",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="IV collapses post-event",
        spot_change_pct=0.02,
        iv_change_pct=-0.40
    ),
    Scenario(
        name="Rally",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="5% rally, IV declines",
        spot_change_pct=0.05,
        iv_change_pct=-0.20
    ),
    Scenario(
        name="Strong Rally",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="10% rally",
        spot_change_pct=0.10,
        iv_change_pct=-0.30
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
        scenarios: Optional[List[Scenario]] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.scenarios = scenarios or (HISTORICAL_SCENARIOS + HYPOTHETICAL_SCENARIOS)

    def run_scenario(
        self,
        scenario: Scenario,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        portfolio_value: float
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
            symbol = pos['symbol']
            current_spot = spot_prices.get(symbol, pos.get('underlying_price', 100))

            # Apply scenario shocks
            new_spot = current_spot * (1 + scenario.spot_change_pct)

            current_iv = pos['iv']
            if scenario.iv_change_abs != 0:
                new_iv = current_iv + scenario.iv_change_abs
            else:
                new_iv = current_iv * (1 + scenario.iv_change_pct)
            new_iv = max(0.01, new_iv)  # Floor at 1%

            new_rate = self.risk_free_rate + (scenario.rate_change_bps / 10000)

            # Adjust DTE for time decay
            new_dte = max(0, pos['dte'] - scenario.time_decay_days)

            # Calculate current value
            current_price = black_scholes_price(
                S=current_spot,
                K=pos['strike'],
                T=pos['dte'] / 365,
                r=self.risk_free_rate,
                sigma=current_iv,
                option_type=pos['option_type'],
                q=pos.get('dividend_yield', 0.0)
            )

            # Calculate new value under scenario
            if new_dte <= 0:
                # Expired - intrinsic value
                if pos['option_type'] == 'put':
                    new_price = max(0, pos['strike'] - new_spot)
                else:
                    new_price = max(0, new_spot - pos['strike'])
            else:
                new_price = black_scholes_price(
                    S=new_spot,
                    K=pos['strike'],
                    T=new_dte / 365,
                    r=new_rate,
                    sigma=new_iv,
                    option_type=pos['option_type'],
                    q=pos.get('dividend_yield', 0.0)
                )

            # P&L calculation
            direction = -1 if pos.get('is_short', True) else 1
            price_change = new_price - current_price
            position_pnl = direction * price_change * pos['contracts'] * 100

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
            max_loss_amount=max_loss_amount
        )

    def run_all_scenarios(
        self,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        portfolio_value: float,
        scenarios: Optional[List[Scenario]] = None
    ) -> StressTestReport:
        """Run all scenarios and generate report."""
        scenarios = scenarios or self.scenarios
        results = []

        for scenario in scenarios:
            result = self.run_scenario(
                scenario, positions, spot_prices, portfolio_value
            )
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
            var_95=abs(var_95)
        )

    def sensitivity_analysis(
        self,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        portfolio_value: float,
        spot_range: Tuple[float, float] = (-0.15, 0.15),
        iv_range: Tuple[float, float] = (-0.30, 0.30),
        n_points: int = 11
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
                    iv_change_pct=iv_chg
                )
                result = self.run_scenario(
                    scenario, positions, spot_prices, portfolio_value
                )
                results.append({
                    'spot_change': spot_chg,
                    'iv_change': iv_chg,
                    'pnl': result.portfolio_pnl,
                    'pnl_pct': result.portfolio_pnl_pct
                })

        return pd.DataFrame(results)

    def monte_carlo_stress(
        self,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        portfolio_value: float,
        n_simulations: int = 10000,
        horizon_days: int = 30,
        vol_of_vol: float = 0.50
    ) -> Dict[str, float]:
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
            avg_iv = np.mean([p['iv'] for p in positions]) if positions else 0.20
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
                time_decay_days=horizon_days
            )

            result = self.run_scenario(
                scenario, positions, spot_prices, portfolio_value
            )
            pnls.append(result.portfolio_pnl)

        pnls = np.array(pnls)

        return {
            'mean': np.mean(pnls),
            'std': np.std(pnls),
            'var_95': np.percentile(pnls, 5),
            'var_99': np.percentile(pnls, 1),
            'cvar_95': np.mean(pnls[pnls <= np.percentile(pnls, 5)]),
            'max_loss': np.min(pnls),
            'max_gain': np.max(pnls),
            'prob_loss': np.mean(pnls < 0),
            'prob_10pct_loss': np.mean(pnls < -portfolio_value * 0.10)
        }


def quick_stress_test(
    positions: List[Dict],
    spot_prices: Dict[str, float],
    portfolio_value: float
) -> str:
    """
    Quick stress test with standard scenarios.

    Returns formatted summary string.
    """
    tester = StressTester()
    report = tester.run_all_scenarios(
        positions, spot_prices, portfolio_value
    )
    return report.summary()


def calculate_max_loss(
    positions: List[Dict],
    spot_prices: Dict[str, float]
) -> float:
    """
    Calculate theoretical maximum loss for portfolio.

    For short puts: max loss = strike * 100 * contracts (stock goes to 0)
    For short calls: max loss = unlimited (capped at reasonable level)
    """
    max_loss = 0.0

    for pos in positions:
        if pos.get('is_short', True):
            if pos['option_type'] == 'put':
                # Max loss: stock goes to 0
                pos_max_loss = pos['strike'] * 100 * pos['contracts']
            else:
                # Short call: cap at 5x current price
                current_price = spot_prices.get(pos['symbol'], pos['strike'])
                pos_max_loss = current_price * 5 * 100 * pos['contracts']
        else:
            # Long positions: max loss = premium paid (need entry price)
            pos_max_loss = pos.get('entry_price', 0) * 100 * pos['contracts']

        max_loss += pos_max_loss

    return max_loss
