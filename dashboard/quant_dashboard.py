"""
Smart Wheel Engine - Quantitative Trading Dashboard

Professional dashboard for institutional-grade options analysis:
- Option pricing (European & American)
- Full Greeks (first, second, third-order)
- Multi-asset portfolio risk (VaR, CVaR)
- Stress testing scenarios

Usage:
    from dashboard.quant_dashboard import QuantDashboard
    dash = QuantDashboard()
    dash.run()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime

from engine.option_pricer import (
    black_scholes_price,
    black_scholes_all_greeks,
    black_scholes_speed,
    black_scholes_color,
    black_scholes_ultima,
    american_option_price,
    american_option_greeks,
    implied_volatility,
)
from engine.risk_manager import (
    RiskManager,
    RiskLimits,
    PortfolioGreeks,
    calculate_kelly_fraction,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OptionInput:
    """Standard option input parameters."""
    spot: float
    strike: float
    dte: int  # Days to expiration
    rate: float = 0.05
    volatility: float = 0.25
    dividend_yield: float = 0.0
    option_type: Literal['call', 'put'] = 'put'

    @property
    def T(self) -> float:
        return self.dte / 365.0


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    option_type: Literal['call', 'put']
    strike: float
    dte: int
    iv: float
    contracts: int
    is_short: bool = True
    underlying_price: Optional[float] = None
    dividend_yield: float = 0.0


@dataclass
class PortfolioInput:
    """Portfolio configuration."""
    positions: List[Position] = field(default_factory=list)
    spot_prices: Dict[str, float] = field(default_factory=dict)
    volatilities: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    portfolio_value: float = 100_000


# =============================================================================
# Dashboard Core
# =============================================================================

class QuantDashboard:
    """
    Professional quantitative trading dashboard.

    Features:
    - Option pricing (European & American)
    - Greeks analysis (1st, 2nd, 3rd order)
    - Portfolio risk metrics (VaR, CVaR)
    - Stress testing
    - Position sizing (Kelly criterion)
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.risk_manager = RiskManager(risk_free_rate=risk_free_rate)
        self._portfolio = PortfolioInput()

    # =========================================================================
    # Option Pricing
    # =========================================================================

    def price_european(self, opt: OptionInput) -> Dict:
        """
        Price European option with full Greeks.

        Returns dict with: price, delta, gamma, theta, vega, rho,
                          vanna, charm, volga, speed, color, ultima
        """
        result = black_scholes_all_greeks(
            S=opt.spot,
            K=opt.strike,
            T=opt.T,
            r=opt.rate,
            sigma=opt.volatility,
            option_type=opt.option_type,
            q=opt.dividend_yield,
            include_second_order=True
        )
        result['model'] = 'Black-Scholes-Merton'
        result['style'] = 'European'
        return result

    def price_american(self, opt: OptionInput) -> Dict:
        """
        Price American option using Barone-Adesi-Whaley approximation.

        Returns dict with: price, delta, gamma, theta, vega, rho
        """
        result = american_option_greeks(
            S=opt.spot,
            K=opt.strike,
            T=opt.T,
            r=opt.rate,
            sigma=opt.volatility,
            option_type=opt.option_type,
            q=opt.dividend_yield
        )

        # Also compute European for comparison
        european = black_scholes_price(
            opt.spot, opt.strike, opt.T, opt.rate,
            opt.volatility, opt.option_type, opt.dividend_yield
        )

        result['model'] = 'Barone-Adesi-Whaley'
        result['style'] = 'American'
        result['european_price'] = european
        result['early_exercise_premium'] = result['price'] - european
        return result

    def compare_pricing(self, opt: OptionInput) -> pd.DataFrame:
        """Compare European vs American pricing."""
        european = self.price_european(opt)
        american = self.price_american(opt)

        data = {
            'Metric': ['Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
            'European': [
                european['price'], european['delta'], european['gamma'],
                european['theta'], european['vega'], european['rho']
            ],
            'American': [
                american['price'], american['delta'], american['gamma'],
                american['theta'], american['vega'], american['rho']
            ],
        }
        df = pd.DataFrame(data)
        df['Difference'] = df['American'] - df['European']
        return df

    # =========================================================================
    # Greeks Analysis
    # =========================================================================

    def analyze_greeks(self, opt: OptionInput) -> Dict:
        """
        Comprehensive Greeks analysis including all orders.

        First-order:  Delta, Theta, Vega, Rho
        Second-order: Gamma, Vanna, Charm, Volga
        Third-order:  Speed, Color, Ultima
        """
        greeks = self.price_european(opt)

        return {
            'first_order': {
                'delta': greeks['delta'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho'],
            },
            'second_order': {
                'gamma': greeks['gamma'],
                'vanna': greeks['vanna'],
                'charm': greeks['charm'],
                'volga': greeks['volga'],
            },
            'third_order': {
                'speed': greeks['speed'],
                'color': greeks['color'],
                'ultima': greeks['ultima'],
            },
            'summary': {
                'price': greeks['price'],
                'model': greeks['model'],
            }
        }

    def greeks_surface(
        self,
        base_opt: OptionInput,
        spot_range: Tuple[float, float] = (0.8, 1.2),
        vol_range: Tuple[float, float] = (0.1, 0.5),
        greek: str = 'delta',
        grid_size: int = 20
    ) -> pd.DataFrame:
        """
        Generate Greeks surface for visualization.

        Args:
            base_opt: Base option parameters
            spot_range: (min, max) as fraction of current spot
            vol_range: (min, max) volatility range
            greek: Which Greek to compute
            grid_size: Number of points per dimension
        """
        spots = np.linspace(
            base_opt.spot * spot_range[0],
            base_opt.spot * spot_range[1],
            grid_size
        )
        vols = np.linspace(vol_range[0], vol_range[1], grid_size)

        results = []
        for spot in spots:
            for vol in vols:
                opt = OptionInput(
                    spot=spot,
                    strike=base_opt.strike,
                    dte=base_opt.dte,
                    rate=base_opt.rate,
                    volatility=vol,
                    dividend_yield=base_opt.dividend_yield,
                    option_type=base_opt.option_type
                )
                greeks = self.price_european(opt)
                results.append({
                    'spot': spot,
                    'volatility': vol,
                    greek: greeks.get(greek, 0)
                })

        return pd.DataFrame(results).pivot(
            index='spot', columns='volatility', values=greek
        )

    # =========================================================================
    # Implied Volatility
    # =========================================================================

    def solve_iv(
        self,
        market_price: float,
        opt: OptionInput
    ) -> Optional[float]:
        """
        Solve for implied volatility from market price.

        Returns IV or None if no solution found.
        """
        return implied_volatility(
            market_price=market_price,
            S=opt.spot,
            K=opt.strike,
            T=opt.T,
            r=opt.rate,
            option_type=opt.option_type,
            q=opt.dividend_yield
        )

    def iv_surface(
        self,
        market_prices: pd.DataFrame,
        spot: float,
        rate: float = 0.05,
        dividend_yield: float = 0.0
    ) -> pd.DataFrame:
        """
        Build IV surface from market prices.

        Args:
            market_prices: DataFrame with columns [strike, dte, price, option_type]
            spot: Current spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield

        Returns:
            DataFrame with IV surface
        """
        results = []
        for _, row in market_prices.iterrows():
            opt = OptionInput(
                spot=spot,
                strike=row['strike'],
                dte=int(row['dte']),
                rate=rate,
                dividend_yield=dividend_yield,
                option_type=row['option_type']
            )
            iv = self.solve_iv(row['price'], opt)
            results.append({
                'strike': row['strike'],
                'dte': row['dte'],
                'moneyness': row['strike'] / spot,
                'iv': iv,
                'option_type': row['option_type']
            })

        return pd.DataFrame(results)

    # =========================================================================
    # Portfolio Management
    # =========================================================================

    def add_position(self, position: Position) -> None:
        """Add position to portfolio."""
        self._portfolio.positions.append(position)

        # Auto-update spot prices and volatilities
        if position.underlying_price:
            self._portfolio.spot_prices[position.symbol] = position.underlying_price
        self._portfolio.volatilities[position.symbol] = position.iv

    def clear_portfolio(self) -> None:
        """Clear all positions."""
        self._portfolio = PortfolioInput()

    def set_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """Set correlation matrix for multi-asset VaR."""
        self._portfolio.correlation_matrix = corr_matrix

    def set_portfolio_value(self, value: float) -> None:
        """Set portfolio value."""
        self._portfolio.portfolio_value = value

    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Calculate aggregate portfolio Greeks."""
        positions = self._positions_to_dicts()
        return self.risk_manager.calculate_portfolio_greeks(
            positions, self._portfolio.spot_prices
        )

    def _positions_to_dicts(self) -> List[Dict]:
        """Convert Position objects to dicts for risk manager."""
        return [
            {
                'symbol': p.symbol,
                'option_type': p.option_type,
                'strike': p.strike,
                'dte': p.dte,
                'iv': p.iv,
                'contracts': p.contracts,
                'is_short': p.is_short,
                'dividend_yield': p.dividend_yield,
            }
            for p in self._portfolio.positions
        ]

    # =========================================================================
    # Risk Metrics
    # =========================================================================

    def calculate_var(
        self,
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> Dict:
        """
        Calculate VaR and CVaR for portfolio.

        Uses multi-asset covariance VaR if correlation matrix provided,
        otherwise falls back to parametric VaR.
        """
        positions = self._positions_to_dicts()

        if self._portfolio.correlation_matrix is not None:
            var, cvar, components = self.risk_manager.calculate_covariance_var(
                portfolio_value=self._portfolio.portfolio_value,
                positions=positions,
                spot_prices=self._portfolio.spot_prices,
                volatilities=self._portfolio.volatilities,
                correlation_matrix=self._portfolio.correlation_matrix,
                confidence=confidence,
                horizon_days=horizon_days
            )
            return {
                'var': var,
                'cvar': cvar,
                'var_pct': var / self._portfolio.portfolio_value,
                'cvar_pct': cvar / self._portfolio.portfolio_value,
                'method': 'Multi-Asset Covariance',
                'confidence': confidence,
                'horizon_days': horizon_days,
                'components': components
            }
        else:
            var, cvar = self.risk_manager.calculate_var(
                portfolio_value=self._portfolio.portfolio_value,
                positions=positions,
                spot_prices=self._portfolio.spot_prices,
                confidence=confidence,
                horizon_days=horizon_days
            )
            return {
                'var': var,
                'cvar': cvar,
                'var_pct': var / self._portfolio.portfolio_value,
                'cvar_pct': cvar / self._portfolio.portfolio_value,
                'method': 'Parametric (Single-Factor)',
                'confidence': confidence,
                'horizon_days': horizon_days,
            }

    def run_stress_tests(
        self,
        custom_scenarios: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive stress tests on portfolio.

        Returns DataFrame with scenario results.
        """
        positions = self._positions_to_dicts()
        results = self.risk_manager.run_stress_tests(
            portfolio_value=self._portfolio.portfolio_value,
            positions=positions,
            spot_prices=self._portfolio.spot_prices,
            custom_scenarios=custom_scenarios
        )

        # Convert to DataFrame
        rows = []
        for name, data in results.items():
            rows.append({
                'Scenario': name,
                'P&L': data['pnl'],
                'P&L %': data['pct_loss'] * 100,
                'Description': data['description']
            })

        return pd.DataFrame(rows).sort_values('P&L')

    # =========================================================================
    # Position Sizing
    # =========================================================================

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5
    ) -> Dict:
        """
        Calculate Kelly criterion position size.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average profit on winning trade
            avg_loss: Average loss on losing trade (positive number)
            fraction: Kelly fraction (0.5 = half-Kelly, recommended)

        Returns:
            Dict with kelly_fraction, recommended_allocation, edge
        """
        kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss, fraction)

        # Calculate edge
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss

        return {
            'kelly_fraction': kelly,
            'recommended_allocation': kelly * self._portfolio.portfolio_value,
            'edge': edge,
            'edge_pct': edge / avg_loss if avg_loss > 0 else 0,
            'win_rate': win_rate,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
        }

    def optimal_position_size(
        self,
        opt: OptionInput,
        win_probability: float = 0.70,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
    ) -> Dict:
        """
        Calculate optimal position size for new trade.
        """
        contracts, reasoning = self.risk_manager.calculate_position_size(
            portfolio_value=self._portfolio.portfolio_value,
            underlying_price=opt.spot,
            strike=opt.strike,
            iv=opt.volatility,
            dte=opt.dte,
            win_probability=win_probability,
            avg_win=avg_win,
            avg_loss=avg_loss,
            existing_positions=len(self._portfolio.positions)
        )

        return {
            'contracts': contracts,
            'notional': contracts * opt.strike * 100,
            'notional_pct': (contracts * opt.strike * 100) / self._portfolio.portfolio_value,
            'reasoning': reasoning
        }

    # =========================================================================
    # Reports
    # =========================================================================

    def portfolio_summary(self) -> Dict:
        """Generate comprehensive portfolio summary."""
        greeks = self.get_portfolio_greeks()
        var_metrics = self.calculate_var()

        return {
            'portfolio_value': self._portfolio.portfolio_value,
            'num_positions': len(self._portfolio.positions),
            'greeks': {
                'delta': greeks.delta,
                'delta_dollars': greeks.delta_dollars,
                'gamma': greeks.gamma,
                'gamma_dollars': greeks.gamma_dollars,
                'theta_daily': greeks.theta,
                'vega': greeks.vega,
                'rho': greeks.rho,
            },
            'risk': {
                'var_95': var_metrics['var'],
                'var_95_pct': var_metrics['var_pct'],
                'cvar_95': var_metrics['cvar'],
                'cvar_95_pct': var_metrics['cvar_pct'],
                'method': var_metrics['method'],
            },
            'timestamp': datetime.now().isoformat()
        }

    def option_report(self, opt: OptionInput, style: str = 'european') -> str:
        """Generate formatted option analysis report."""
        if style == 'american':
            result = self.price_american(opt)
        else:
            result = self.price_european(opt)

        greeks = self.analyze_greeks(opt)

        report = f"""
{'='*60}
OPTION ANALYSIS REPORT
{'='*60}

INPUT PARAMETERS
----------------
Spot Price:      ${opt.spot:,.2f}
Strike Price:    ${opt.strike:,.2f}
Days to Expiry:  {opt.dte}
Risk-Free Rate:  {opt.rate:.2%}
Volatility:      {opt.volatility:.2%}
Dividend Yield:  {opt.dividend_yield:.2%}
Option Type:     {opt.option_type.upper()}
Pricing Model:   {result['model']}

PRICING
-------
Option Price:    ${result['price']:,.4f}
"""
        if style == 'american':
            report += f"""European Price:  ${result['european_price']:,.4f}
Early Ex. Prem:  ${result['early_exercise_premium']:,.4f}
"""

        report += f"""
FIRST-ORDER GREEKS
------------------
Delta:           {greeks['first_order']['delta']:+.4f}
Theta (annual):  {greeks['first_order']['theta']:+.4f}
Theta (daily):   {greeks['first_order']['theta']/365:+.4f}
Vega:            {greeks['first_order']['vega']:+.4f}
Rho:             {greeks['first_order']['rho']:+.4f}

SECOND-ORDER GREEKS
-------------------
Gamma:           {greeks['second_order']['gamma']:+.6f}
Vanna:           {greeks['second_order']['vanna']:+.6f}
Charm:           {greeks['second_order']['charm']:+.6f}
Volga:           {greeks['second_order']['volga']:+.6f}

THIRD-ORDER GREEKS
------------------
Speed:           {greeks['third_order']['speed']:+.8f}
Color:           {greeks['third_order']['color']:+.8f}
Ultima:          {greeks['third_order']['ultima']:+.8f}

{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
"""
        return report

    def risk_report(self) -> str:
        """Generate formatted portfolio risk report."""
        summary = self.portfolio_summary()
        stress = self.run_stress_tests()

        report = f"""
{'='*60}
PORTFOLIO RISK REPORT
{'='*60}

PORTFOLIO OVERVIEW
------------------
Total Value:     ${summary['portfolio_value']:,.2f}
Positions:       {summary['num_positions']}

AGGREGATE GREEKS
----------------
Delta:           {summary['greeks']['delta']:+.2f} shares
Delta ($):       ${summary['greeks']['delta_dollars']:+,.0f}
Gamma:           {summary['greeks']['gamma']:+.4f}
Gamma ($):       ${summary['greeks']['gamma_dollars']:+,.0f}
Theta (daily):   ${summary['greeks']['theta_daily']:+,.2f}
Vega:            ${summary['greeks']['vega']:+,.2f}
Rho:             ${summary['greeks']['rho']:+,.2f}

VALUE AT RISK ({summary['risk']['method']})
{'-'*40}
95% 1-Day VaR:   ${summary['risk']['var_95']:,.2f} ({summary['risk']['var_95_pct']:.2%})
95% 1-Day CVaR:  ${summary['risk']['cvar_95']:,.2f} ({summary['risk']['cvar_95_pct']:.2%})

STRESS TEST RESULTS
-------------------
"""
        for _, row in stress.head(10).iterrows():
            report += f"{row['Scenario']:20s} ${row['P&L']:>12,.0f} ({row['P&L %']:>6.1f}%)\n"

        report += f"""
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
"""
        return report

    # =========================================================================
    # Interactive CLI
    # =========================================================================

    def run(self):
        """Run interactive dashboard CLI."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║           SMART WHEEL ENGINE - QUANT DASHBOARD               ║
║                    Version 2.0.0                             ║
╚══════════════════════════════════════════════════════════════╝
        """)

        while True:
            print("""
┌────────────────────────────────────────────────────────────┐
│  MAIN MENU                                                 │
├────────────────────────────────────────────────────────────┤
│  1. Option Pricing (European)                              │
│  2. Option Pricing (American)                              │
│  3. Greeks Analysis                                        │
│  4. Implied Volatility Solver                              │
│  5. Portfolio Management                                   │
│  6. Risk Analysis (VaR/CVaR)                               │
│  7. Stress Testing                                         │
│  8. Position Sizing (Kelly)                                │
│  9. Generate Reports                                       │
│  0. Exit                                                   │
└────────────────────────────────────────────────────────────┘
            """)

            try:
                choice = input("Select option [0-9]: ").strip()

                if choice == '0':
                    print("\nExiting dashboard. Goodbye!")
                    break
                elif choice == '1':
                    self._menu_european_pricing()
                elif choice == '2':
                    self._menu_american_pricing()
                elif choice == '3':
                    self._menu_greeks_analysis()
                elif choice == '4':
                    self._menu_iv_solver()
                elif choice == '5':
                    self._menu_portfolio()
                elif choice == '6':
                    self._menu_risk_analysis()
                elif choice == '7':
                    self._menu_stress_testing()
                elif choice == '8':
                    self._menu_kelly()
                elif choice == '9':
                    self._menu_reports()
                else:
                    print("Invalid option. Please try again.")

            except KeyboardInterrupt:
                print("\n\nExiting dashboard. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

    def _get_option_input(self) -> OptionInput:
        """Helper to get option parameters from user."""
        print("\n--- Enter Option Parameters ---")
        spot = float(input("Spot price [$]: ") or "100")
        strike = float(input("Strike price [$]: ") or "100")
        dte = int(input("Days to expiration: ") or "30")
        rate = float(input("Risk-free rate [0.05]: ") or "0.05")
        vol = float(input("Volatility [0.25]: ") or "0.25")
        div = float(input("Dividend yield [0.0]: ") or "0.0")
        opt_type = input("Option type [put/call]: ").lower() or "put"

        return OptionInput(
            spot=spot, strike=strike, dte=dte,
            rate=rate, volatility=vol, dividend_yield=div,
            option_type=opt_type
        )

    def _menu_european_pricing(self):
        opt = self._get_option_input()
        print(self.option_report(opt, style='european'))

    def _menu_american_pricing(self):
        opt = self._get_option_input()
        print(self.option_report(opt, style='american'))

        print("\n--- European vs American Comparison ---")
        print(self.compare_pricing(opt).to_string(index=False))

    def _menu_greeks_analysis(self):
        opt = self._get_option_input()
        greeks = self.analyze_greeks(opt)

        print("\n" + "="*50)
        print("GREEKS ANALYSIS")
        print("="*50)

        for order, values in greeks.items():
            if order != 'summary':
                print(f"\n{order.upper().replace('_', ' ')}:")
                for name, val in values.items():
                    print(f"  {name:12s}: {val:+.6f}")

    def _menu_iv_solver(self):
        opt = self._get_option_input()
        price = float(input("Market price [$]: "))

        iv = self.solve_iv(price, opt)
        if iv:
            print(f"\n>>> Implied Volatility: {iv:.2%}")
        else:
            print("\n>>> No valid IV found (check price bounds)")

    def _menu_portfolio(self):
        print(f"\nCurrent positions: {len(self._portfolio.positions)}")
        print("""
  1. Add position
  2. View positions
  3. Clear portfolio
  4. Set portfolio value
  5. Back to main menu
        """)

        choice = input("Select [1-5]: ").strip()

        if choice == '1':
            symbol = input("Symbol: ").upper()
            opt = self._get_option_input()
            contracts = int(input("Contracts: ") or "1")
            is_short = input("Short position? [y/n]: ").lower() == 'y'

            pos = Position(
                symbol=symbol,
                option_type=opt.option_type,
                strike=opt.strike,
                dte=opt.dte,
                iv=opt.volatility,
                contracts=contracts,
                is_short=is_short,
                underlying_price=opt.spot,
                dividend_yield=opt.dividend_yield
            )
            self.add_position(pos)
            print(f">>> Added {contracts} {symbol} {opt.strike} {opt.option_type}")

        elif choice == '2':
            if not self._portfolio.positions:
                print("No positions in portfolio.")
            else:
                for i, p in enumerate(self._portfolio.positions, 1):
                    direction = "Short" if p.is_short else "Long"
                    print(f"{i}. {direction} {p.contracts}x {p.symbol} ${p.strike} {p.option_type} ({p.dte}d)")

        elif choice == '3':
            self.clear_portfolio()
            print(">>> Portfolio cleared")

        elif choice == '4':
            value = float(input("Portfolio value [$]: "))
            self.set_portfolio_value(value)
            print(f">>> Portfolio value set to ${value:,.2f}")

    def _menu_risk_analysis(self):
        if not self._portfolio.positions:
            print("No positions in portfolio. Add positions first.")
            return

        print(self.risk_report())

    def _menu_stress_testing(self):
        if not self._portfolio.positions:
            print("No positions in portfolio. Add positions first.")
            return

        stress = self.run_stress_tests()
        print("\n" + "="*60)
        print("STRESS TEST RESULTS")
        print("="*60)
        print(stress.to_string(index=False))

    def _menu_kelly(self):
        print("\n--- Kelly Criterion Calculator ---")
        win_rate = float(input("Win rate (0-1) [0.70]: ") or "0.70")
        avg_win = float(input("Average win [$]: ") or "100")
        avg_loss = float(input("Average loss [$]: ") or "200")

        result = self.calculate_kelly(win_rate, avg_win, avg_loss)

        print(f"""
>>> Kelly Results:
    Kelly Fraction:     {result['kelly_fraction']:.2%}
    Recommended Size:   ${result['recommended_allocation']:,.2f}
    Edge:               ${result['edge']:,.2f} ({result['edge_pct']:.2%})
    Win/Loss Ratio:     {result['win_loss_ratio']:.2f}
        """)

    def _menu_reports(self):
        print("""
  1. Option Analysis Report
  2. Portfolio Risk Report
  3. Back to main menu
        """)

        choice = input("Select [1-3]: ").strip()

        if choice == '1':
            opt = self._get_option_input()
            style = input("Style [european/american]: ").lower() or "european"
            print(self.option_report(opt, style))
        elif choice == '2':
            print(self.risk_report())


# =============================================================================
# Quick Functions (Standalone Usage)
# =============================================================================

def quick_price(
    spot: float,
    strike: float,
    dte: int,
    volatility: float = 0.25,
    option_type: str = 'put',
    style: str = 'european'
) -> float:
    """Quick option pricing."""
    if style == 'american':
        return american_option_price(
            spot, strike, dte/365, 0.05, volatility, option_type
        )
    return black_scholes_price(
        spot, strike, dte/365, 0.05, volatility, option_type
    )


def quick_greeks(
    spot: float,
    strike: float,
    dte: int,
    volatility: float = 0.25,
    option_type: str = 'put'
) -> Dict:
    """Quick Greeks calculation."""
    return black_scholes_all_greeks(
        spot, strike, dte/365, 0.05, volatility, option_type
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    dashboard = QuantDashboard()
    dashboard.run()
