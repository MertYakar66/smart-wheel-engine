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
        limits: RiskLimits | None = None,
        sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_SCALED,
        risk_free_rate: float = 0.05
    ):
        self.limits = limits or RiskLimits()
        self.sizing_method = sizing_method
        self.risk_free_rate = risk_free_rate

        # Track historical data for risk calcs
        self.portfolio_values: list[float] = []
        self.peak_value: float = 0.0
        self.returns_history: list[float] = []

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
    ) -> tuple[int, str]:
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
        positions: list[dict],
        spot_prices: dict[str, float]
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
            # Convert annual theta to daily theta (pricer returns per-year)
            greeks.theta += (pos_greeks['theta'] / 365) * multiplier
            greeks.vega += pos_greeks['vega'] * multiplier
            greeks.rho += pos_greeks['rho'] * multiplier

            # Dollar-weighted metrics
            greeks.delta_dollars += pos_greeks['delta'] * multiplier * spot
            greeks.gamma_dollars += pos_greeks['gamma'] * multiplier * spot * spot / 100

        return greeks

    def calculate_var(
        self,
        portfolio_value: float,
        positions: list[dict],
        spot_prices: dict[str, float],
        returns_data: pd.DataFrame | None = None,
        volatilities: dict[str, float] | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR.

        Method selection (in order of preference):
        1. Multi-asset covariance VaR if correlation_matrix provided
        2. Historical simulation if returns_data provided (>30 days)
        3. Delta-normal approximation (single-factor) as fallback

        Args:
            portfolio_value: Total portfolio value
            positions: List of position dictionaries
            spot_prices: Current spot prices per symbol
            returns_data: Historical returns for historical VaR
            volatilities: Per-asset volatilities for covariance VaR
            correlation_matrix: Asset correlation matrix for covariance VaR
            confidence: VaR confidence level (default 0.95)
            horizon_days: VaR horizon in days

        Returns:
            Tuple of (VaR, CVaR) as positive dollar amounts
        """
        if portfolio_value <= 0:
            return 0.0, 0.0

        greeks = self.calculate_portfolio_greeks(positions, spot_prices)

        # Priority 1: Multi-asset covariance VaR (most accurate for multi-asset)
        if correlation_matrix is not None and volatilities is not None:
            var, cvar, _ = self.calculate_covariance_var(
                portfolio_value, positions, spot_prices,
                volatilities, correlation_matrix,
                confidence, horizon_days
            )
            return var, cvar

        # Priority 2: Historical simulation
        if returns_data is not None and len(returns_data) > 30:
            var, cvar = self._historical_var(
                portfolio_value, greeks, returns_data, confidence, horizon_days
            )
            return var, cvar

        # Priority 3: Simple delta-normal approximation
        var, cvar = self._parametric_var(
            portfolio_value, greeks, spot_prices, confidence, horizon_days
        )

        return var, cvar

    def _parametric_var(
        self,
        portfolio_value: float,
        greeks: PortfolioGreeks,
        spot_prices: dict[str, float],
        confidence: float,
        horizon_days: int,
        volatilities: dict[str, float] | None = None,
        correlations: pd.DataFrame | None = None,  # Reserved for future use
        vol_of_vol: float = 0.05,
    ) -> tuple[float, float]:
        """
        Delta-gamma-vega parametric VaR.

        Current implementation:
        1. Uses aggregate portfolio delta (single-factor approximation)
        2. Average volatility across assets
        3. Gamma P&L adjustment: 0.5 * gamma_dollars * (dS)^2
        4. Vega P&L from vol-of-vol shock

        Limitations:
        - Does NOT use per-asset delta decomposition
        - Correlation matrix parameter is reserved but not yet implemented
        - For concentrated/correlated books, use historical VaR instead

        Args:
            portfolio_value: Total portfolio value
            greeks: Aggregated portfolio Greeks
            spot_prices: Current spot prices per symbol
            confidence: VaR confidence level (e.g., 0.95)
            horizon_days: VaR horizon
            volatilities: Dict of symbol -> annualized volatility
            correlations: Reserved for future multi-asset covariance
            vol_of_vol: Volatility of volatility for vega shock (default 5%)

        Returns:
            Tuple of (VaR, CVaR) as positive dollar amounts
        """
        if portfolio_value <= 0:
            return 0.0, 0.0

        # Use provided volatilities or estimate from spot prices
        if volatilities is None:
            # Default to 25% annualized vol (market average)
            avg_vol = 0.25
            volatilities = dict.fromkeys(spot_prices, avg_vol)

        # Calculate portfolio variance using delta exposures
        symbols = list(spot_prices.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return 0.0, 0.0

        # Build delta vector (per asset)
        # Note: For full multi-asset, we'd need per-position Greeks
        # Here we use aggregate delta_dollars as proxy
        delta_dollars = greeks.delta_dollars

        # Average volatility weighted by exposure
        avg_vol = np.mean([volatilities.get(s, 0.25) for s in symbols])
        daily_vol = avg_vol / np.sqrt(252)

        # Scale to horizon
        horizon_vol = daily_vol * np.sqrt(horizon_days)
        z_score = stats.norm.ppf(confidence)

        # === Delta P&L Component ===
        # Portfolio P&L from delta: delta_dollars * return
        # Variance: (delta_dollars)^2 * vol^2
        delta_var_component = (delta_dollars * horizon_vol) ** 2

        # === Gamma P&L Component ===
        # Second-order: 0.5 * gamma_dollars * (dS)^2
        # Expected value of dS^2 under normal: vol^2
        # This adds convexity (positive gamma = good for long options)
        gamma_dollars = greeks.gamma_dollars
        # For short options (negative gamma), this is a risk
        # Variance contribution from gamma: (0.5 * gamma * vol^2)^2
        # Simplified: gamma impact on expected shortfall
        gamma_expected_loss = 0.5 * abs(gamma_dollars) * (horizon_vol ** 2)
        if gamma_dollars < 0:
            # Short gamma: losses accelerate as market moves
            gamma_adjustment = gamma_expected_loss * z_score
        else:
            # Long gamma: gains accelerate (reduces VaR)
            gamma_adjustment = -gamma_expected_loss * 0.5  # Partial credit

        # === Vega P&L Component ===
        # P&L from vol change: vega * dVol
        # dVol ~ Normal(0, vol_of_vol * sqrt(horizon))
        vega = greeks.vega
        horizon_vol_of_vol = vol_of_vol * np.sqrt(horizon_days)
        vega_var_component = (vega * horizon_vol_of_vol) ** 2

        # === Combined VaR ===
        # Total variance (assuming independence between price and vol shocks)
        # In reality they're negatively correlated (leverage effect), but this is conservative
        total_variance = delta_var_component + vega_var_component
        total_std = np.sqrt(total_variance)

        # VaR = z * portfolio_std + gamma adjustment
        var = total_std * z_score + gamma_adjustment

        # CVaR (Expected Shortfall) for delta-gamma-vega portfolio
        # For normal distribution: CVaR = σ * φ(z_α) / (1 - α)
        # Reference: Rockafellar & Uryasev (2000)
        pdf_at_z = stats.norm.pdf(z_score)
        base_cvar = total_std * pdf_at_z / (1 - confidence)

        # Gamma adjustment for CVaR:
        # CVaR considers the entire tail beyond VaR, where gamma effects compound.
        # The hazard rate h(z) = φ(z)/(1-Φ(z)) scales the tail impact.
        # For 99% confidence, h(z_0.99) ≈ 5.6, so tail losses are amplified.
        # We use a conservative multiplier of φ(z)/(1-α) / z ≈ 1.3 for gamma scaling.
        hazard_ratio = pdf_at_z / ((1 - confidence) * z_score) if z_score > 0 else 1.0
        gamma_cvar_multiplier = min(2.0, max(1.0, hazard_ratio))  # Clamp to [1.0, 2.0]
        cvar = base_cvar + gamma_adjustment * gamma_cvar_multiplier

        return abs(var), abs(cvar)

    def _historical_var(
        self,
        portfolio_value: float,
        greeks: PortfolioGreeks,
        returns_data: pd.DataFrame,
        confidence: float,
        horizon_days: int,
        vol_returns: pd.Series | None = None,
    ) -> tuple[float, float]:
        """
        Historical simulation VaR with delta-gamma-vega.

        Upgrades from simple delta-only:
        1. Multi-asset P&L using weighted returns (if multi-column)
        2. Gamma P&L: 0.5 * gamma * (dS)^2 for each scenario
        3. Vega P&L: vega * dVol for each scenario (if vol_returns provided)
        4. Proper compounding for multi-day horizons

        Args:
            portfolio_value: Total portfolio value
            greeks: Aggregated portfolio Greeks
            returns_data: DataFrame with returns (single or multi-column)
            confidence: VaR confidence level
            horizon_days: VaR horizon
            vol_returns: Optional series of IV changes for vega P&L

        Returns:
            Tuple of (VaR, CVaR) as positive dollar amounts
        """
        # Handle multi-column returns (multiple assets)
        if isinstance(returns_data, pd.DataFrame):
            if 'returns' in returns_data.columns:
                hist_returns = returns_data['returns'].dropna()
            elif returns_data.shape[1] == 1:
                hist_returns = returns_data.iloc[:, 0].dropna()
            else:
                # Multi-asset: use equal-weighted portfolio return
                # In production, would use actual position weights
                hist_returns = returns_data.mean(axis=1).dropna()
        else:
            hist_returns = returns_data.dropna()

        # Compute returns if we have prices instead of returns
        if hist_returns.max() > 1:  # Likely prices, not returns
            hist_returns = hist_returns.pct_change().dropna()

        # Scale returns to horizon using proper compounding
        if horizon_days > 1:
            compound_factor = (1 + hist_returns).rolling(horizon_days).apply(
                lambda x: x.prod(), raw=True
            ).dropna() - 1
            hist_returns = compound_factor

        n_scenarios = len(hist_returns)
        if n_scenarios == 0:
            return 0.0, 0.0

        # === Delta P&L ===
        # P&L = delta_dollars * return
        delta_pnl = greeks.delta_dollars * hist_returns.values

        # === Gamma P&L ===
        # P&L = 0.5 * gamma_dollars * return^2 * spot^2 / spot^2
        # Simplified: 0.5 * gamma_dollars * return^2 (since gamma_dollars already scaled)
        # Note: For negative gamma (short options), this adds to losses
        gamma_pnl = 0.5 * greeks.gamma_dollars * (hist_returns.values ** 2)

        # === Vega P&L ===
        vega_pnl = np.zeros(n_scenarios)
        if vol_returns is not None and len(vol_returns) >= n_scenarios:
            # P&L = vega * dVol
            vol_changes = vol_returns.values[-n_scenarios:]
            vega_pnl = greeks.vega * vol_changes
        else:
            # Estimate vol shocks from return magnitude (leverage effect)
            # When returns are negative, vol typically increases
            # Approximate: dVol ~ -0.5 * return (simplified leverage effect)
            implied_vol_change = -0.5 * hist_returns.values
            vega_pnl = greeks.vega * implied_vol_change * 0.01  # Scale to 1% vol change

        # === Total P&L Scenarios ===
        total_pnl = delta_pnl + gamma_pnl + vega_pnl

        # Sort for percentile calculation (ascending, so losses are first)
        sorted_pnl = np.sort(total_pnl)

        # === VaR with Linear Interpolation ===
        # Use proper quantile interpolation for accuracy with small samples
        # Reference: Hyndman & Fan (1996), Type 7 (R/Python default)
        alpha = 1 - confidence  # e.g., 0.05 for 95% VaR
        var = -np.percentile(sorted_pnl, alpha * 100, method='linear')

        # === CVaR (Expected Shortfall) with proper weighting ===
        # CVaR is the conditional expectation E[X | X <= VaR]
        # For finite samples, use weighted average including fractional observation
        tail_threshold = -var
        tail_observations = sorted_pnl[sorted_pnl <= tail_threshold]

        if len(tail_observations) > 0:
            # Include partial weight for the observation at the quantile boundary
            # This gives a more accurate CVaR estimate for small samples
            exact_idx = alpha * (n_scenarios - 1)
            floor_idx = int(np.floor(exact_idx))
            frac = exact_idx - floor_idx

            if floor_idx > 0:
                # Full weight for observations strictly below VaR
                strict_tail = sorted_pnl[:floor_idx]
                # Fractional weight for boundary observation
                boundary_contrib = sorted_pnl[floor_idx] * frac if frac > 0 else 0

                total_weight = floor_idx + frac
                if total_weight > 0:
                    weighted_sum = np.sum(strict_tail) + boundary_contrib
                    cvar = -weighted_sum / total_weight
                else:
                    cvar = var
            else:
                # Very high confidence or small sample: use first observation
                cvar = -sorted_pnl[0]
        else:
            cvar = var

        return max(0.0, var), max(0.0, cvar)

    def calculate_covariance_var(
        self,
        portfolio_value: float,
        positions: list[dict],
        spot_prices: dict[str, float],
        volatilities: dict[str, float],
        correlation_matrix: pd.DataFrame,
        confidence: float = 0.95,
        horizon_days: int = 1,
        vol_of_vol: float = 0.05,
    ) -> tuple[float, float, dict]:
        """
        Multi-asset covariance VaR using full correlation structure.

        This is the proper institutional approach for portfolios with
        multiple correlated assets. Uses the delta-normal method with
        full covariance matrix.

        Formula: VaR = z_α × √(δ' Σ δ)

        where:
        - δ = vector of dollar deltas per asset
        - Σ = covariance matrix (correlation × outer(vol, vol))
        - z_α = confidence z-score

        Args:
            portfolio_value: Total portfolio value
            positions: List of position dicts with 'symbol', 'delta_dollars'
            spot_prices: Current spot prices per symbol
            volatilities: Dict of symbol -> annualized volatility
            correlation_matrix: DataFrame with symbols as index/columns
            confidence: VaR confidence level (default 0.95)
            horizon_days: VaR horizon in days
            vol_of_vol: Volatility of volatility for vega component

        Returns:
            Tuple of (VaR, CVaR, component_breakdown)

        Reference:
            Jorion, P. "Value at Risk", Chapter 7: Portfolio Risk
        """
        if portfolio_value <= 0:
            return 0.0, 0.0, {}

        # Build per-asset delta exposures from positions
        symbol_deltas: dict[str, float] = {}
        symbol_gammas: dict[str, float] = {}
        symbol_vegas: dict[str, float] = {}

        for pos in positions:
            symbol = pos['symbol']
            spot = spot_prices.get(symbol, pos.get('underlying_price', 100))

            # Calculate or use provided Greeks
            if 'delta_dollars' in pos:
                delta_d = pos['delta_dollars']
            else:
                # Calculate from position
                pos_greeks = black_scholes_all_greeks(
                    S=spot,
                    K=pos['strike'],
                    T=pos['dte'] / 365,
                    r=self.risk_free_rate,
                    sigma=pos['iv'],
                    option_type=pos['option_type'],
                    q=pos.get('dividend_yield', 0.0)
                )
                direction = -1 if pos.get('is_short', True) else 1
                multiplier = direction * pos['contracts'] * 100
                delta_d = pos_greeks['delta'] * multiplier * spot
                gamma_d = pos_greeks['gamma'] * multiplier * spot * spot / 100
                vega_d = pos_greeks['vega'] * multiplier

                symbol_gammas[symbol] = symbol_gammas.get(symbol, 0.0) + gamma_d
                symbol_vegas[symbol] = symbol_vegas.get(symbol, 0.0) + vega_d

            symbol_deltas[symbol] = symbol_deltas.get(symbol, 0.0) + delta_d

        # Get list of symbols with exposure
        symbols = list(symbol_deltas.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return 0.0, 0.0, {}

        # Build delta vector and volatility vector
        delta_vec = np.array([symbol_deltas[s] for s in symbols])
        vol_vec = np.array([volatilities.get(s, 0.25) for s in symbols])

        # Convert to daily volatility and scale to horizon
        daily_vol_vec = vol_vec / np.sqrt(252)
        horizon_vol_vec = daily_vol_vec * np.sqrt(horizon_days)

        # Build covariance matrix
        # Σ = diag(vol) × corr × diag(vol)
        if n_assets == 1:
            # Single asset case
            covariance = np.array([[horizon_vol_vec[0] ** 2]])
        else:
            # Extract correlation submatrix for our symbols
            corr_subset = correlation_matrix.loc[symbols, symbols].values

            # Ensure it's symmetric
            corr_subset = (corr_subset + corr_subset.T) / 2
            np.fill_diagonal(corr_subset, 1.0)

            # PSD repair: eigenvalue flooring to ensure positive semi-definiteness
            # This handles noisy estimated correlations that may not be valid
            eigenvalues, eigenvectors = np.linalg.eigh(corr_subset)
            min_eigenvalue = np.min(eigenvalues)
            if min_eigenvalue < 0:
                # Floor negative eigenvalues to small positive value
                eigenvalues = np.maximum(eigenvalues, 1e-8)
                # Reconstruct correlation matrix
                corr_subset = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                # Re-normalize to ensure diagonal = 1
                d = np.sqrt(np.diag(corr_subset))
                corr_subset = corr_subset / np.outer(d, d)

            # Build covariance: Σ_ij = σ_i × σ_j × ρ_ij
            covariance = np.outer(horizon_vol_vec, horizon_vol_vec) * corr_subset

        # === Delta VaR Component ===
        # Portfolio variance: δ' Σ δ
        delta_variance = delta_vec @ covariance @ delta_vec
        delta_std = np.sqrt(max(0, delta_variance))

        z_score = stats.norm.ppf(confidence)
        delta_var = delta_std * z_score

        # === Gamma Component ===
        # For each asset: gamma_loss = 0.5 * gamma * (horizon_vol * S)^2
        # This is a simplification - full treatment would use non-central chi-square
        gamma_var = 0.0
        if symbol_gammas:
            for s in symbols:
                gamma_d = symbol_gammas.get(s, 0.0)
                spot = spot_prices.get(s, 100)
                vol_h = horizon_vol_vec[symbols.index(s)]
                # Expected loss from gamma (for short gamma positions)
                gamma_expected = 0.5 * abs(gamma_d) * (vol_h ** 2)
                if gamma_d < 0:  # Short gamma
                    gamma_var += gamma_expected * z_score
                else:  # Long gamma (benefit)
                    gamma_var -= gamma_expected * 0.5

        # === Vega Component ===
        # Vol-of-vol impact on vega positions
        vega_var = 0.0
        if symbol_vegas:
            total_vega = sum(symbol_vegas.values())
            horizon_vol_of_vol = vol_of_vol * np.sqrt(horizon_days)
            vega_var = abs(total_vega * horizon_vol_of_vol * z_score)

        # === Combined VaR ===
        # Assuming independence between spot and vol shocks (conservative)
        total_var = np.sqrt(delta_var**2 + vega_var**2) + gamma_var

        # === CVaR (Expected Shortfall) ===
        pdf_at_z = stats.norm.pdf(z_score)
        base_cvar = delta_std * pdf_at_z / (1 - confidence)

        # Gamma scaling for CVaR tail
        hazard_ratio = pdf_at_z / ((1 - confidence) * z_score) if z_score > 0 else 1.0
        gamma_cvar_multiplier = min(2.0, max(1.0, hazard_ratio))
        total_cvar = np.sqrt(base_cvar**2 + vega_var**2) + gamma_var * gamma_cvar_multiplier

        # Component breakdown for risk attribution
        components = {
            'delta_var': abs(delta_var),
            'gamma_var': abs(gamma_var),
            'vega_var': abs(vega_var),
            'delta_std': delta_std,
            'per_asset_contribution': {},
        }

        # Marginal VaR per asset (∂VaR/∂w_i)
        if delta_variance > 0:
            marginal_var = (covariance @ delta_vec) / delta_std * z_score
            for i, s in enumerate(symbols):
                components['per_asset_contribution'][s] = {
                    'delta_dollars': delta_vec[i],
                    'marginal_var': marginal_var[i],
                    'component_var': delta_vec[i] * marginal_var[i] / delta_var if delta_var > 0 else 0,
                }

        return abs(total_var), abs(total_cvar), components

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
        positions: list[dict],
        spot_prices: dict[str, float]
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

    def run_stress_tests(
        self,
        portfolio_value: float,
        positions: list[dict],
        spot_prices: dict[str, float],
        custom_scenarios: list[dict] | None = None,
    ) -> dict[str, dict]:
        """
        Run comprehensive stress scenarios on the portfolio.

        Standard scenarios:
        1. Market crash: -10%, -20%, -30% spot moves with vol spike
        2. Vol explosion: +50%, +100% IV increase
        3. Gap down: Overnight -5%, -10% gap
        4. Correlation breakdown: All correlations go to 1
        5. Rate shock: +100bps, +200bps rate increase

        Args:
            portfolio_value: Current portfolio value
            positions: List of position dictionaries
            spot_prices: Current spot prices
            custom_scenarios: Optional list of custom scenario dicts

        Returns:
            Dict of scenario_name -> {pnl, pct_loss, description}
        """
        greeks = self.calculate_portfolio_greeks(positions, spot_prices)
        results = {}

        # === Standard Scenarios ===

        # 1. Market Crash Scenarios
        crash_scenarios = [
            (-0.10, 0.30, "Moderate crash: -10% spot, +30% vol"),
            (-0.20, 0.50, "Severe crash: -20% spot, +50% vol"),
            (-0.30, 1.00, "Extreme crash: -30% spot, +100% vol"),
        ]

        for spot_move, vol_move, desc in crash_scenarios:
            # Delta P&L
            delta_pnl = greeks.delta_dollars * spot_move
            # Gamma P&L (second order): 0.5 * gamma * (dS)^2
            gamma_pnl = 0.5 * greeks.gamma_dollars * (spot_move ** 2)
            # Vega P&L from vol increase
            vega_pnl = greeks.vega * vol_move * 100  # vega is per 1%, vol_move is decimal

            total_pnl = delta_pnl + gamma_pnl + vega_pnl

            scenario_name = f"crash_{int(abs(spot_move)*100)}pct"
            results[scenario_name] = {
                "pnl": total_pnl,
                "pct_loss": total_pnl / portfolio_value if portfolio_value > 0 else 0,
                "description": desc,
                "components": {
                    "delta_pnl": delta_pnl,
                    "gamma_pnl": gamma_pnl,
                    "vega_pnl": vega_pnl,
                }
            }

        # 2. Vol Explosion Scenarios (market rallies or crashes)
        vol_scenarios = [
            (0.50, "Vol spike: +50% IV"),
            (1.00, "Vol explosion: +100% IV"),
            (-0.30, "Vol crush: -30% IV"),
        ]

        for vol_move, desc in vol_scenarios:
            vega_pnl = greeks.vega * vol_move * 100

            scenario_name = f"vol_{'+' if vol_move > 0 else ''}{int(vol_move*100)}pct"
            results[scenario_name] = {
                "pnl": vega_pnl,
                "pct_loss": vega_pnl / portfolio_value if portfolio_value > 0 else 0,
                "description": desc,
                "components": {"vega_pnl": vega_pnl}
            }

        # 3. Gap Down Scenarios (overnight moves)
        gap_scenarios = [
            (-0.05, 0.20, "Gap down: -5% overnight, +20% vol"),
            (-0.10, 0.40, "Large gap: -10% overnight, +40% vol"),
        ]

        for spot_move, vol_move, desc in gap_scenarios:
            delta_pnl = greeks.delta_dollars * spot_move
            gamma_pnl = 0.5 * greeks.gamma_dollars * (spot_move ** 2)
            vega_pnl = greeks.vega * vol_move * 100
            # Theta benefit from passage of time (1 day)
            theta_pnl = greeks.theta  # Daily theta

            total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

            scenario_name = f"gap_{int(abs(spot_move)*100)}pct"
            results[scenario_name] = {
                "pnl": total_pnl,
                "pct_loss": total_pnl / portfolio_value if portfolio_value > 0 else 0,
                "description": desc,
                "components": {
                    "delta_pnl": delta_pnl,
                    "gamma_pnl": gamma_pnl,
                    "vega_pnl": vega_pnl,
                    "theta_pnl": theta_pnl,
                }
            }

        # 4. Rate Shock Scenarios
        rate_scenarios = [
            (0.01, "Rate hike: +100bps"),
            (0.02, "Large rate hike: +200bps"),
        ]

        for rate_move, desc in rate_scenarios:
            # Rho is sensitivity to 1% rate change
            rho_pnl = greeks.rho * rate_move * 100

            scenario_name = f"rate_{int(rate_move*10000)}bps"
            results[scenario_name] = {
                "pnl": rho_pnl,
                "pct_loss": rho_pnl / portfolio_value if portfolio_value > 0 else 0,
                "description": desc,
                "components": {"rho_pnl": rho_pnl}
            }

        # 5. Combined worst case: crash + vol + correlation spike
        worst_case = {
            "spot_move": -0.25,
            "vol_move": 0.80,
            "theta_days": 5,  # 5 days of theta decay
        }

        delta_pnl = greeks.delta_dollars * worst_case["spot_move"]
        gamma_pnl = 0.5 * greeks.gamma_dollars * (worst_case["spot_move"] ** 2)
        vega_pnl = greeks.vega * worst_case["vol_move"] * 100
        theta_pnl = greeks.theta * worst_case["theta_days"]

        total_worst = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

        results["worst_case"] = {
            "pnl": total_worst,
            "pct_loss": total_worst / portfolio_value if portfolio_value > 0 else 0,
            "description": "Worst case: -25% spot, +80% vol, 5d theta",
            "components": {
                "delta_pnl": delta_pnl,
                "gamma_pnl": gamma_pnl,
                "vega_pnl": vega_pnl,
                "theta_pnl": theta_pnl,
            }
        }

        # 6. Custom scenarios
        if custom_scenarios:
            for i, scenario in enumerate(custom_scenarios):
                spot_move = scenario.get("spot_move", 0)
                vol_move = scenario.get("vol_move", 0)
                rate_move = scenario.get("rate_move", 0)
                desc = scenario.get("description", f"Custom scenario {i+1}")

                delta_pnl = greeks.delta_dollars * spot_move
                gamma_pnl = 0.5 * greeks.gamma_dollars * (spot_move ** 2)
                vega_pnl = greeks.vega * vol_move * 100
                rho_pnl = greeks.rho * rate_move * 100

                total_pnl = delta_pnl + gamma_pnl + vega_pnl + rho_pnl

                results[f"custom_{i+1}"] = {
                    "pnl": total_pnl,
                    "pct_loss": total_pnl / portfolio_value if portfolio_value > 0 else 0,
                    "description": desc,
                    "components": {
                        "delta_pnl": delta_pnl,
                        "gamma_pnl": gamma_pnl,
                        "vega_pnl": vega_pnl,
                        "rho_pnl": rho_pnl,
                    }
                }

        return results

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
        positions: list[dict],
        spot_prices: dict[str, float],
        proposed_trade: dict | None = None
    ) -> tuple[bool, list[str]]:
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
        win_rate: Probability of winning (must be in range [0, 1])
        avg_win: Average profit on winning trade
        avg_loss: Average loss on losing trade (positive number)
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)

    Returns:
        Optimal fraction of capital to risk (0 if inputs invalid)
    """
    # Validate win_rate is a probability in [0, 1]
    if not (0.0 <= win_rate <= 1.0):
        return 0.0

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
    margin_requirement: float = 0.20,
    stress_loss_pct: float = 0.25,
    premium_per_share: float = 0.0
) -> int:
    """
    Calculate maximum contracts given capital and risk constraints.

    Uses stress-loss model rather than full notional for risk calculation.
    For a cash-secured put, the stress loss is the expected drawdown from
    strike to stress level, minus premium received.

    Args:
        capital: Available capital
        strike: Option strike price
        max_risk_pct: Maximum percentage of capital to risk (per position)
        margin_requirement: Margin requirement as fraction of notional
        stress_loss_pct: Expected max drawdown in stress scenario (default 25%)
        premium_per_share: Premium received per share (reduces risk)

    Returns:
        Number of contracts (0 if constraints cannot be satisfied)
    """
    if capital <= 0 or strike <= 0:
        return 0

    notional_per_contract = strike * 100
    margin_per_contract = notional_per_contract * margin_requirement

    # Risk-based limit using stress loss model
    # Loss per contract = (strike * stress_loss_pct - premium) * 100
    # This is more realistic than assuming stock goes to $0
    stress_loss_per_contract = max(0, (strike * stress_loss_pct - premium_per_share) * 100)

    # If stress loss is very small (high premium relative to stress), use a floor
    min_loss_per_contract = strike * 0.10 * 100  # Floor at 10% loss
    loss_per_contract = max(stress_loss_per_contract, min_loss_per_contract)

    max_risk_capital = capital * max_risk_pct
    contracts_by_risk = int(max_risk_capital / loss_per_contract) if loss_per_contract > 0 else 0

    # Margin-based limit
    contracts_by_margin = int(capital / margin_per_contract)

    # Return 0 if constraints cannot be satisfied (do NOT force minimum of 1)
    return max(0, min(contracts_by_risk, contracts_by_margin))


# =============================================================================
# Sector Exposure Management
# =============================================================================

# GICS Sector mappings - SP500 constituents only
DEFAULT_SECTOR_MAP: dict[str, str] = {
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
    symbols: list[str]

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
        sector_map: dict[str, str] | None = None,
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
        positions: list[dict],
        portfolio_value: float
    ) -> dict[str, SectorExposure]:
        """
        Calculate exposure by sector.

        Args:
            positions: List of position dicts with 'symbol', 'strike', 'contracts'
            portfolio_value: Total portfolio value

        Returns:
            Dict of sector -> SectorExposure
        """
        sector_data: dict[str, dict] = {}

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
        positions: list[dict],
        portfolio_value: float
    ) -> tuple[bool, str]:
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
        positions: list[dict],
        portfolio_value: float
    ) -> list[str]:
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
        positions: list[dict],
        portfolio_value: float,
        available_symbols: list[str]
    ) -> list[str]:
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
        covariance: pd.DataFrame | None = None
    ) -> dict[str, float]:
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

    def _quasi_diagonalize(self, link: np.ndarray) -> list[int]:
        """
        Reorganize items to quasi-diagonal form.

        Returns sorted indices.
        """
        from scipy.cluster.hierarchy import leaves_list
        return list(leaves_list(link))

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_symbols: list[str]
    ) -> dict[str, float]:
        """
        Recursive bisection for weight allocation.

        Allocates weights inversely proportional to cluster variance.
        """
        weights = dict.fromkeys(sorted_symbols, 1.0)
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
        symbols: list[str]
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
    target_symbols: list[str] | None = None
) -> dict[str, float]:
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
    symbols: list[str],
    returns_data: pd.DataFrame,
    max_weight: float = 0.20,
    min_weight: float = 0.02
) -> dict[str, float]:
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
