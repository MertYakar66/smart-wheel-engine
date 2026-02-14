"""
Risk Management Module for Wheel Strategy

Institutional-grade risk controls:
- Position sizing (Kelly, fixed-fractional, volatility-scaled)
- Portfolio Greeks aggregation
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Concentration limits
- Drawdown-based position scaling
- Correlation-aware diversification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

from .option_pricer import black_scholes_all_greeks


class PositionSizingMethod(Enum):
    """Position sizing methodologies."""
    FIXED_FRACTIONAL = "fixed_fractional"  # Fixed % of capital per trade
    KELLY = "kelly"  # Kelly criterion based
    VOLATILITY_SCALED = "volatility_scaled"  # Size inversely to volatility
    EQUAL_RISK = "equal_risk"  # Equal dollar risk per position
    MAX_LOSS = "max_loss"  # Size based on max acceptable loss


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio-level Greeks."""
    delta: float = 0.0  # Net delta exposure in shares
    gamma: float = 0.0  # Net gamma (delta sensitivity)
    theta: float = 0.0  # Daily time decay ($/day)
    vega: float = 0.0   # Volatility sensitivity ($/1% IV move)
    rho: float = 0.0    # Interest rate sensitivity

    # Normalized metrics
    delta_dollars: float = 0.0  # Delta * underlying price * 100
    gamma_dollars: float = 0.0  # Gamma * underlying price^2 * 100 / 100

    def __str__(self) -> str:
        return (
            f"Portfolio Greeks:\n"
            f"  Delta: {self.delta:+.2f} ({self.delta_dollars:+,.0f} $-delta)\n"
            f"  Gamma: {self.gamma:+.4f} ({self.gamma_dollars:+,.0f} $-gamma)\n"
            f"  Theta: {self.theta:+,.2f} $/day\n"
            f"  Vega:  {self.vega:+,.2f} $/1% IV\n"
            f"  Rho:   {self.rho:+,.2f} $/1% rate"
        )


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    # Position limits
    max_positions: int = 10
    max_single_position_pct: float = 0.20  # Max 20% in single underlying
    max_sector_pct: float = 0.40  # Max 40% in single sector

    # Greeks limits (as % of portfolio)
    max_portfolio_delta: float = 0.50  # Max 50% net delta
    max_portfolio_gamma_dollars: float = 50000  # Max gamma dollar exposure
    max_portfolio_vega: float = 10000  # Max vega exposure

    # Loss limits
    max_drawdown_pct: float = 0.20  # Max 20% drawdown
    daily_loss_limit_pct: float = 0.03  # Max 3% daily loss

    # VaR limits
    max_var_95_pct: float = 0.05  # Max 5% 1-day 95% VaR
    max_cvar_95_pct: float = 0.08  # Max 8% 1-day 95% CVaR

    # Concentration limits
    min_positions_for_full_size: int = 5  # Need 5+ positions for full sizing
    correlation_penalty_threshold: float = 0.70  # Reduce size if corr > 70%


@dataclass
class RiskMetrics:
    """Computed risk metrics for portfolio."""
    # VaR metrics
    var_95_1d: float = 0.0  # 1-day 95% VaR
    var_99_1d: float = 0.0  # 1-day 99% VaR
    cvar_95_1d: float = 0.0  # 1-day 95% CVaR (Expected Shortfall)

    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0

    # Concentration metrics
    herfindahl_index: float = 0.0  # Position concentration (0-1)
    largest_position_pct: float = 0.0

    # Greeks summary
    portfolio_greeks: PortfolioGreeks = field(default_factory=PortfolioGreeks)

    # Risk-adjusted capacity
    available_risk_budget: float = 1.0  # 0-1, how much more risk can we take


class RiskManager:
    """
    Institutional risk management for options portfolio.

    Handles:
    - Position sizing with multiple methodologies
    - Portfolio Greeks calculation and limits
    - VaR/CVaR estimation
    - Drawdown monitoring
    - Concentration limits
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_SCALED,
        risk_free_rate: float = 0.05
    ):
        self.limits = limits or RiskLimits()
        self.sizing_method = sizing_method
        self.risk_free_rate = risk_free_rate

        # Track historical data for risk calcs
        self.portfolio_values: List[float] = []
        self.peak_value: float = 0.0
        self.returns_history: List[float] = []

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
        underlying_correlation: float = 0.0
    ) -> Tuple[int, str]:
        """
        Calculate optimal position size (number of contracts).

        Returns:
            Tuple of (contracts, reasoning)
        """
        if portfolio_value <= 0:
            return 0, "No capital available"

        # Base sizing by method
        if self.sizing_method == PositionSizingMethod.FIXED_FRACTIONAL:
            base_pct = 0.05  # 5% per position

        elif self.sizing_method == PositionSizingMethod.KELLY:
            # Kelly: f* = (p*b - q) / b where b = win/loss ratio
            if avg_loss <= 0:
                base_pct = 0.05
            else:
                b = avg_win / avg_loss
                p = win_probability
                q = 1 - p
                kelly = (p * b - q) / b if b > 0 else 0
                # Half-Kelly for safety
                base_pct = max(0, min(kelly * 0.5, 0.25))

        elif self.sizing_method == PositionSizingMethod.VOLATILITY_SCALED:
            # Higher IV = smaller position
            # Target: position risk ~ constant across different IV environments
            base_vol = 0.20  # Baseline IV
            vol_scalar = base_vol / max(iv, 0.05)
            base_pct = 0.05 * min(vol_scalar, 2.0)  # Cap at 2x normal

        elif self.sizing_method == PositionSizingMethod.EQUAL_RISK:
            # Size so max loss is equal across positions
            max_loss_per_position = portfolio_value * 0.02  # 2% max loss
            notional_at_risk = strike * 100  # Full assignment risk
            contracts_by_risk = max_loss_per_position / notional_at_risk
            base_pct = (contracts_by_risk * strike * 100) / portfolio_value

        elif self.sizing_method == PositionSizingMethod.MAX_LOSS:
            # Size based on worst-case assignment
            max_acceptable_loss = portfolio_value * 0.05  # 5% max loss per trade
            # Worst case: stock goes to 0 after assignment
            max_loss_per_contract = strike * 100
            contracts = max_acceptable_loss / max_loss_per_contract
            base_pct = (contracts * strike * 100) / portfolio_value
        else:
            base_pct = 0.05

        # Apply concentration penalty
        concentration_scalar = 1.0
        if existing_positions < self.limits.min_positions_for_full_size:
            # Reduce size when concentrated
            concentration_scalar = existing_positions / self.limits.min_positions_for_full_size
            concentration_scalar = max(0.5, concentration_scalar)  # Floor at 50%

        # Apply correlation penalty
        correlation_scalar = 1.0
        if underlying_correlation > self.limits.correlation_penalty_threshold:
            excess_corr = underlying_correlation - self.limits.correlation_penalty_threshold
            correlation_scalar = 1.0 - (excess_corr * 2)  # Reduce by 2x excess correlation
            correlation_scalar = max(0.3, correlation_scalar)  # Floor at 30%

        # Apply drawdown scaling
        drawdown_scalar = self._get_drawdown_scalar()

        # Final position percentage
        final_pct = base_pct * concentration_scalar * correlation_scalar * drawdown_scalar
        final_pct = min(final_pct, self.limits.max_single_position_pct)

        # Convert to contracts
        capital_for_position = portfolio_value * final_pct
        notional_per_contract = strike * 100
        contracts = int(capital_for_position / notional_per_contract)

        # Ensure at least 1 contract if any allocation
        if contracts == 0 and final_pct > 0.01:
            contracts = 1

        # Cap by max positions
        if existing_positions >= self.limits.max_positions:
            return 0, f"Max positions ({self.limits.max_positions}) reached"

        reasoning = (
            f"Method: {self.sizing_method.value}, "
            f"Base: {base_pct:.1%}, "
            f"Scalars: conc={concentration_scalar:.2f}, corr={correlation_scalar:.2f}, dd={drawdown_scalar:.2f}, "
            f"Final: {final_pct:.1%} = {contracts} contracts"
        )

        return contracts, reasoning

    def _get_drawdown_scalar(self) -> float:
        """Scale position size based on current drawdown."""
        if not self.portfolio_values:
            return 1.0

        current_value = self.portfolio_values[-1]
        if self.peak_value <= 0:
            return 1.0

        drawdown = (self.peak_value - current_value) / self.peak_value

        # Linear scaling: at max_drawdown, size = 25%
        if drawdown <= 0:
            return 1.0
        elif drawdown >= self.limits.max_drawdown_pct:
            return 0.25  # Minimum 25% sizing in max drawdown
        else:
            # Linear interpolation
            return 1.0 - (0.75 * drawdown / self.limits.max_drawdown_pct)

    def calculate_portfolio_greeks(
        self,
        positions: List[Dict],
        spot_prices: Dict[str, float]
    ) -> PortfolioGreeks:
        """
        Calculate aggregate portfolio Greeks.

        Args:
            positions: List of position dicts with keys:
                - symbol, option_type, strike, dte, iv, contracts, is_short
            spot_prices: Dict of symbol -> current price
        """
        greeks = PortfolioGreeks()

        for pos in positions:
            symbol = pos['symbol']
            spot = spot_prices.get(symbol, pos.get('underlying_price', 100))

            pos_greeks = black_scholes_all_greeks(
                S=spot,
                K=pos['strike'],
                T=pos['dte'] / 365,
                r=self.risk_free_rate,
                sigma=pos['iv'],
                option_type=pos['option_type'],
                q=pos.get('dividend_yield', 0.0)
            )

            # Direction multiplier (short = negative Greeks exposure)
            direction = -1 if pos.get('is_short', True) else 1
            multiplier = direction * pos['contracts'] * 100

            greeks.delta += pos_greeks['delta'] * multiplier
            greeks.gamma += pos_greeks['gamma'] * multiplier
            greeks.theta += pos_greeks['theta'] * multiplier
            greeks.vega += pos_greeks['vega'] * multiplier
            greeks.rho += pos_greeks['rho'] * multiplier

            # Dollar-weighted metrics
            greeks.delta_dollars += pos_greeks['delta'] * multiplier * spot
            greeks.gamma_dollars += pos_greeks['gamma'] * multiplier * spot * spot / 100

        return greeks

    def calculate_var(
        self,
        portfolio_value: float,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        returns_data: Optional[pd.DataFrame] = None,
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR.

        Uses delta-normal approximation when no returns data available,
        historical simulation when returns data provided.

        Returns:
            Tuple of (VaR, CVaR) as positive dollar amounts
        """
        if portfolio_value <= 0:
            return 0.0, 0.0

        greeks = self.calculate_portfolio_greeks(positions, spot_prices)

        if returns_data is not None and len(returns_data) > 30:
            # Historical simulation
            var, cvar = self._historical_var(
                portfolio_value, greeks, returns_data, confidence, horizon_days
            )
        else:
            # Delta-normal approximation
            var, cvar = self._parametric_var(
                portfolio_value, greeks, spot_prices, confidence, horizon_days
            )

        return var, cvar

    def _parametric_var(
        self,
        portfolio_value: float,
        greeks: PortfolioGreeks,
        spot_prices: Dict[str, float],
        confidence: float,
        horizon_days: int
    ) -> Tuple[float, float]:
        """Delta-normal VaR approximation."""
        # Assume 20% annualized vol if no data
        avg_vol = 0.20
        daily_vol = avg_vol / np.sqrt(252)

        # Portfolio volatility from delta exposure
        # Simplified: assume all underlyings move together
        portfolio_daily_vol = abs(greeks.delta_dollars) * daily_vol / portfolio_value

        # Scale to horizon
        horizon_vol = portfolio_daily_vol * np.sqrt(horizon_days)

        # VaR
        z_score = stats.norm.ppf(confidence)
        var = portfolio_value * horizon_vol * z_score

        # CVaR (expected shortfall) for normal distribution
        pdf_at_z = stats.norm.pdf(z_score)
        cvar = portfolio_value * horizon_vol * pdf_at_z / (1 - confidence)

        return abs(var), abs(cvar)

    def _historical_var(
        self,
        portfolio_value: float,
        greeks: PortfolioGreeks,
        returns_data: pd.DataFrame,
        confidence: float,
        horizon_days: int
    ) -> Tuple[float, float]:
        """Historical simulation VaR."""
        # Get portfolio returns using delta approximation
        if 'returns' in returns_data.columns:
            hist_returns = returns_data['returns'].dropna()
        else:
            # Assume first column is price, compute returns
            hist_returns = returns_data.iloc[:, 0].pct_change().dropna()

        # Scale returns to horizon
        if horizon_days > 1:
            # Rolling sum for multi-day returns
            hist_returns = hist_returns.rolling(horizon_days).sum().dropna()

        # Portfolio P&L scenarios
        pnl_scenarios = greeks.delta_dollars * hist_returns

        # Sort for percentile calculation
        sorted_pnl = np.sort(pnl_scenarios)

        # VaR
        var_idx = int(len(sorted_pnl) * (1 - confidence))
        var = -sorted_pnl[var_idx] if var_idx < len(sorted_pnl) else 0

        # CVaR (mean of losses beyond VaR)
        tail_losses = sorted_pnl[:var_idx + 1]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var

        return max(0, var), max(0, cvar)

    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value history for drawdown tracking."""
        self.portfolio_values.append(value)

        if value > self.peak_value:
            self.peak_value = value

        if len(self.portfolio_values) > 1:
            ret = (value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns_history.append(ret)

    def get_risk_metrics(
        self,
        portfolio_value: float,
        positions: List[Dict],
        spot_prices: Dict[str, float]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        metrics = RiskMetrics()

        # Greeks
        metrics.portfolio_greeks = self.calculate_portfolio_greeks(positions, spot_prices)

        # VaR
        metrics.var_95_1d, metrics.cvar_95_1d = self.calculate_var(
            portfolio_value, positions, spot_prices, confidence=0.95
        )
        metrics.var_99_1d, _ = self.calculate_var(
            portfolio_value, positions, spot_prices, confidence=0.99
        )

        # Drawdown
        if self.peak_value > 0:
            metrics.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            metrics.max_drawdown = max(
                metrics.current_drawdown,
                self._calculate_max_drawdown()
            )

        # Concentration (Herfindahl Index)
        if positions:
            position_values = []
            for pos in positions:
                value = pos.get('market_value', pos['strike'] * 100 * pos['contracts'])
                position_values.append(abs(value))
            total = sum(position_values)
            if total > 0:
                weights = [v / total for v in position_values]
                metrics.herfindahl_index = sum(w ** 2 for w in weights)
                metrics.largest_position_pct = max(weights)

        # Risk budget
        metrics.available_risk_budget = self._calculate_risk_budget(
            portfolio_value, metrics
        )

        return metrics

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from history."""
        if not self.portfolio_values:
            return 0.0

        values = np.array(self.portfolio_values)
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks
        return float(np.max(drawdowns))

    def _calculate_risk_budget(
        self,
        portfolio_value: float,
        metrics: RiskMetrics
    ) -> float:
        """Calculate remaining risk budget (0-1)."""
        budget = 1.0

        # Drawdown constraint
        dd_usage = metrics.current_drawdown / self.limits.max_drawdown_pct
        budget = min(budget, 1.0 - dd_usage)

        # VaR constraint
        if portfolio_value > 0:
            var_pct = metrics.var_95_1d / portfolio_value
            var_usage = var_pct / self.limits.max_var_95_pct
            budget = min(budget, 1.0 - var_usage)

        # Greeks constraints
        greeks = metrics.portfolio_greeks
        if portfolio_value > 0:
            delta_usage = abs(greeks.delta_dollars / portfolio_value) / self.limits.max_portfolio_delta
            budget = min(budget, 1.0 - delta_usage)

        return max(0.0, budget)

    def check_limits(
        self,
        portfolio_value: float,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        proposed_trade: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits.

        Returns:
            Tuple of (is_within_limits, list of violations)
        """
        violations = []
        metrics = self.get_risk_metrics(portfolio_value, positions, spot_prices)

        # Position count
        if len(positions) > self.limits.max_positions:
            violations.append(
                f"Position count {len(positions)} > limit {self.limits.max_positions}"
            )

        # Concentration
        if metrics.largest_position_pct > self.limits.max_single_position_pct:
            violations.append(
                f"Single position {metrics.largest_position_pct:.1%} > limit {self.limits.max_single_position_pct:.1%}"
            )

        # Drawdown
        if metrics.current_drawdown > self.limits.max_drawdown_pct:
            violations.append(
                f"Drawdown {metrics.current_drawdown:.1%} > limit {self.limits.max_drawdown_pct:.1%}"
            )

        # VaR
        if portfolio_value > 0:
            var_pct = metrics.var_95_1d / portfolio_value
            if var_pct > self.limits.max_var_95_pct:
                violations.append(
                    f"95% VaR {var_pct:.1%} > limit {self.limits.max_var_95_pct:.1%}"
                )

        # Greeks
        greeks = metrics.portfolio_greeks
        if portfolio_value > 0:
            delta_pct = abs(greeks.delta_dollars / portfolio_value)
            if delta_pct > self.limits.max_portfolio_delta:
                violations.append(
                    f"Delta exposure {delta_pct:.1%} > limit {self.limits.max_portfolio_delta:.1%}"
                )

        if abs(greeks.gamma_dollars) > self.limits.max_portfolio_gamma_dollars:
            violations.append(
                f"Gamma ${abs(greeks.gamma_dollars):,.0f} > limit ${self.limits.max_portfolio_gamma_dollars:,.0f}"
            )

        if abs(greeks.vega) > self.limits.max_portfolio_vega:
            violations.append(
                f"Vega ${abs(greeks.vega):,.0f} > limit ${self.limits.max_portfolio_vega:,.0f}"
            )

        return len(violations) == 0, violations


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.5
) -> float:
    """
    Calculate Kelly criterion bet size.

    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average profit on winning trade
        avg_loss: Average loss on losing trade (positive number)
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)

    Returns:
        Optimal fraction of capital to risk
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    b = avg_win / avg_loss  # Win/loss ratio
    p = win_rate
    q = 1 - p

    # Kelly formula: f* = (p*b - q) / b
    full_kelly = (p * b - q) / b

    # Apply fraction and bounds
    sized_kelly = full_kelly * kelly_fraction
    return max(0.0, min(sized_kelly, 0.25))  # Cap at 25%


def calculate_optimal_contracts(
    capital: float,
    strike: float,
    max_risk_pct: float = 0.05,
    margin_requirement: float = 0.20
) -> int:
    """
    Calculate maximum contracts given capital and risk constraints.

    Args:
        capital: Available capital
        strike: Option strike price
        max_risk_pct: Maximum percentage of capital to risk
        margin_requirement: Margin requirement as fraction of notional

    Returns:
        Number of contracts
    """
    notional_per_contract = strike * 100
    margin_per_contract = notional_per_contract * margin_requirement

    # Risk-based limit
    max_risk_capital = capital * max_risk_pct
    contracts_by_risk = int(max_risk_capital / notional_per_contract)

    # Margin-based limit
    contracts_by_margin = int(capital / margin_per_contract)

    return max(1, min(contracts_by_risk, contracts_by_margin))


# =============================================================================
# Sector Exposure Management
# =============================================================================

# GICS Sector mappings - SP500 constituents only
DEFAULT_SECTOR_MAP: Dict[str, str] = {
    # Information Technology
    'AAPL': 'Information Technology', 'MSFT': 'Information Technology',
    'NVDA': 'Information Technology', 'AMD': 'Information Technology',
    'INTC': 'Information Technology', 'CRM': 'Information Technology',
    'ADBE': 'Information Technology', 'AVGO': 'Information Technology',
    'ORCL': 'Information Technology', 'CSCO': 'Information Technology',
    'ACN': 'Information Technology', 'TXN': 'Information Technology',
    'QCOM': 'Information Technology', 'INTU': 'Information Technology',
    'AMAT': 'Information Technology', 'MU': 'Information Technology',
    'LRCX': 'Information Technology', 'KLAC': 'Information Technology',
    'NOW': 'Information Technology', 'PANW': 'Information Technology',

    # Communication Services
    'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
    'META': 'Communication Services', 'NFLX': 'Communication Services',
    'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
    'TMUS': 'Communication Services', 'VZ': 'Communication Services',
    'T': 'Communication Services', 'EA': 'Communication Services',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary',
    'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',
    'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'CL': 'Consumer Staples', 'MDLZ': 'Consumer Staples',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
    'AXP': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
    'PGR': 'Financials', 'CB': 'Financials', 'CME': 'Financials',
    'ICE': 'Financials', 'COF': 'Financials', 'USB': 'Financials',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'LLY': 'Healthcare',
    'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'ISRG': 'Healthcare', 'CVS': 'Healthcare', 'CI': 'Healthcare',
    'ELV': 'Healthcare', 'VRTX': 'Healthcare', 'REGN': 'Healthcare',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
    'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy',
    'WMB': 'Energy', 'HAL': 'Energy', 'DVN': 'Energy',

    # Industrials
    'CAT': 'Industrials', 'BA': 'Industrials', 'HON': 'Industrials',
    'UPS': 'Industrials', 'GE': 'Industrials', 'RTX': 'Industrials',
    'DE': 'Industrials', 'LMT': 'Industrials', 'UNP': 'Industrials',
    'MMM': 'Industrials', 'WM': 'Industrials', 'FDX': 'Industrials',
    'NOC': 'Industrials', 'GD': 'Industrials', 'CSX': 'Industrials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'D': 'Utilities', 'AEP': 'Utilities', 'SRE': 'Utilities',
    'EXC': 'Utilities', 'XEL': 'Utilities',

    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate',
    'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'O': 'Real Estate',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'FCX': 'Materials', 'NEM': 'Materials', 'ECL': 'Materials',
    'DOW': 'Materials', 'NUE': 'Materials',
}


@dataclass
class SectorExposure:
    """Sector-level exposure metrics."""
    sector: str
    position_count: int
    notional_exposure: float
    exposure_pct: float  # As % of portfolio
    symbols: List[str]

    @property
    def is_concentrated(self) -> bool:
        """Check if sector is over-concentrated (>25%)."""
        return self.exposure_pct > 0.25


class SectorExposureManager:
    """
    Manage sector-level portfolio exposure.

    Prevents over-concentration in correlated sectors.
    """

    def __init__(
        self,
        sector_map: Optional[Dict[str, str]] = None,
        max_sector_pct: float = 0.25
    ):
        """
        Args:
            sector_map: Dict of symbol -> sector
            max_sector_pct: Maximum allocation per sector
        """
        self.sector_map = sector_map or DEFAULT_SECTOR_MAP
        self.max_sector_pct = max_sector_pct

    def get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        return self.sector_map.get(symbol, 'Unknown')

    def calculate_sector_exposures(
        self,
        positions: List[Dict],
        portfolio_value: float
    ) -> Dict[str, SectorExposure]:
        """
        Calculate exposure by sector.

        Args:
            positions: List of position dicts with 'symbol', 'strike', 'contracts'
            portfolio_value: Total portfolio value

        Returns:
            Dict of sector -> SectorExposure
        """
        sector_data: Dict[str, Dict] = {}

        for pos in positions:
            symbol = pos['symbol']
            sector = self.get_sector(symbol)
            notional = pos['strike'] * 100 * pos['contracts']

            if sector not in sector_data:
                sector_data[sector] = {
                    'notional': 0,
                    'count': 0,
                    'symbols': []
                }

            sector_data[sector]['notional'] += notional
            sector_data[sector]['count'] += 1
            if symbol not in sector_data[sector]['symbols']:
                sector_data[sector]['symbols'].append(symbol)

        # Convert to SectorExposure objects
        exposures = {}
        for sector, data in sector_data.items():
            exposure_pct = data['notional'] / portfolio_value if portfolio_value > 0 else 0

            exposures[sector] = SectorExposure(
                sector=sector,
                position_count=data['count'],
                notional_exposure=data['notional'],
                exposure_pct=exposure_pct,
                symbols=data['symbols']
            )

        return exposures

    def check_sector_limit(
        self,
        symbol: str,
        proposed_notional: float,
        positions: List[Dict],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Check if adding position would breach sector limit.

        Returns:
            (is_allowed, reason)
        """
        sector = self.get_sector(symbol)
        exposures = self.calculate_sector_exposures(positions, portfolio_value)

        current_exposure = exposures.get(sector, SectorExposure(
            sector=sector, position_count=0, notional_exposure=0,
            exposure_pct=0, symbols=[]
        ))

        new_exposure_pct = (
            (current_exposure.notional_exposure + proposed_notional) / portfolio_value
            if portfolio_value > 0 else 0
        )

        if new_exposure_pct > self.max_sector_pct:
            return False, (
                f"Sector '{sector}' would be {new_exposure_pct:.1%} "
                f"(limit: {self.max_sector_pct:.1%}). "
                f"Current positions: {current_exposure.symbols}"
            )

        return True, f"Sector '{sector}' at {new_exposure_pct:.1%} (limit: {self.max_sector_pct:.1%})"

    def get_sector_violations(
        self,
        positions: List[Dict],
        portfolio_value: float
    ) -> List[str]:
        """Get list of sector limit violations."""
        violations = []
        exposures = self.calculate_sector_exposures(positions, portfolio_value)

        for sector, exposure in exposures.items():
            if exposure.exposure_pct > self.max_sector_pct:
                violations.append(
                    f"Sector '{sector}' at {exposure.exposure_pct:.1%} > "
                    f"limit {self.max_sector_pct:.1%} "
                    f"(symbols: {', '.join(exposure.symbols)})"
                )

        return violations

    def suggest_diversification(
        self,
        positions: List[Dict],
        portfolio_value: float,
        available_symbols: List[str]
    ) -> List[str]:
        """
        Suggest symbols from under-represented sectors.

        Returns list of symbols to consider for diversification.
        """
        exposures = self.calculate_sector_exposures(positions, portfolio_value)

        # Find sectors with room
        sectors_with_room = {}
        for symbol in available_symbols:
            sector = self.get_sector(symbol)
            current = exposures.get(sector, SectorExposure(
                sector=sector, position_count=0, notional_exposure=0,
                exposure_pct=0, symbols=[]
            ))

            if current.exposure_pct < self.max_sector_pct * 0.5:  # Less than 50% of limit
                if sector not in sectors_with_room:
                    sectors_with_room[sector] = []
                sectors_with_room[sector].append(symbol)

        # Flatten and return
        suggestions = []
        for sector in sorted(sectors_with_room.keys()):
            suggestions.extend(sectors_with_room[sector][:3])  # Max 3 per sector

        return suggestions


# =============================================================================
# Hierarchical Risk Parity (HRP) Portfolio Optimization
# =============================================================================

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimization.

    Advantages over Mean-Variance Optimization:
    - Doesn't require expected return estimates
    - More stable out-of-sample
    - Handles correlated assets better

    Based on López de Prado (2016).
    """

    def __init__(self, linkage_method: str = 'ward'):
        """
        Args:
            linkage_method: Hierarchical clustering method
                ('ward', 'single', 'complete', 'average')
        """
        self.linkage_method = linkage_method

    def fit(
        self,
        returns: pd.DataFrame,
        covariance: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate HRP weights.

        Args:
            returns: DataFrame of asset returns (T x N)
            covariance: Optional pre-computed covariance matrix

        Returns:
            Dict of symbol -> weight
        """
        if covariance is None:
            covariance = returns.cov()

        # Step 1: Tree clustering
        corr = returns.corr()
        dist = self._correlation_distance(corr)
        link = self._hierarchical_cluster(dist)

        # Step 2: Quasi-diagonalization
        sorted_idx = self._quasi_diagonalize(link)
        sorted_symbols = [returns.columns[i] for i in sorted_idx]

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(covariance, sorted_symbols)

        return weights

    def _correlation_distance(self, corr: pd.DataFrame) -> np.ndarray:
        """Convert correlation matrix to distance matrix."""
        # Distance = sqrt(0.5 * (1 - correlation))
        dist = np.sqrt(0.5 * (1 - corr.values))
        return dist

    def _hierarchical_cluster(self, dist: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering."""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        # Convert to condensed form for linkage
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)
        return link

    def _quasi_diagonalize(self, link: np.ndarray) -> List[int]:
        """
        Reorganize items to quasi-diagonal form.

        Returns sorted indices.
        """
        from scipy.cluster.hierarchy import leaves_list
        return list(leaves_list(link))

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_symbols: List[str]
    ) -> Dict[str, float]:
        """
        Recursive bisection for weight allocation.

        Allocates weights inversely proportional to cluster variance.
        """
        weights = {s: 1.0 for s in sorted_symbols}
        clusters = [sorted_symbols]

        while len(clusters) > 0:
            # Split each cluster
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # Split in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Calculate cluster variances
                left_var = self._cluster_variance(cov, left)
                right_var = self._cluster_variance(cov, right)

                # Allocate inversely to variance
                total_var = left_var + right_var
                if total_var > 0:
                    left_weight = 1 - left_var / total_var
                    right_weight = 1 - right_var / total_var
                else:
                    left_weight = right_weight = 0.5

                # Update weights
                for s in left:
                    weights[s] *= left_weight
                for s in right:
                    weights[s] *= right_weight

                # Add sub-clusters for next iteration
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _cluster_variance(
        self,
        cov: pd.DataFrame,
        symbols: List[str]
    ) -> float:
        """Calculate variance of equal-weighted cluster."""
        if len(symbols) == 0:
            return 0.0

        sub_cov = cov.loc[symbols, symbols]
        n = len(symbols)
        w = np.ones(n) / n

        var = float(np.dot(w, np.dot(sub_cov.values, w)))
        return var


def calculate_hrp_weights(
    returns_df: pd.DataFrame,
    target_symbols: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Convenience function to calculate HRP weights.

    Args:
        returns_df: DataFrame with daily returns (columns = symbols)
        target_symbols: Optional subset of symbols to include

    Returns:
        Dict of symbol -> weight
    """
    if target_symbols:
        returns_df = returns_df[target_symbols]

    hrp = HierarchicalRiskParity()
    return hrp.fit(returns_df)


def optimize_position_weights(
    symbols: List[str],
    returns_data: pd.DataFrame,
    max_weight: float = 0.20,
    min_weight: float = 0.02
) -> Dict[str, float]:
    """
    Optimize position weights for Wheel strategy.

    Combines HRP with practical constraints.

    Args:
        symbols: Symbols to include
        returns_data: Historical returns DataFrame
        max_weight: Maximum weight per symbol
        min_weight: Minimum weight per symbol

    Returns:
        Dict of symbol -> constrained weight
    """
    # Filter to available symbols
    available = [s for s in symbols if s in returns_data.columns]

    if len(available) < 2:
        # Not enough for optimization
        return {s: 1.0 / len(symbols) for s in symbols}

    # Get HRP weights
    hrp_weights = calculate_hrp_weights(returns_data, available)

    # Apply constraints
    constrained = {}
    for symbol in symbols:
        if symbol in hrp_weights:
            w = hrp_weights[symbol]
            w = max(min_weight, min(max_weight, w))
            constrained[symbol] = w
        else:
            constrained[symbol] = min_weight

    # Normalize
    total = sum(constrained.values())
    if total > 0:
        constrained = {s: w / total for s, w in constrained.items()}

    return constrained
