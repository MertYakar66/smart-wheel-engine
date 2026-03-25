"""
Monte Carlo Simulation Module for Options Portfolios

Three professional-grade simulation engines:

1. Block Bootstrap: Resample historical returns in blocks to preserve
   autocorrelation and volatility clustering. Generates synthetic equity
   curves for strategy robustness testing.

2. Jump Diffusion (Merton 1976): GBM + Poisson jumps for realistic
   equity paths. Includes "Bagholder Probability" — the probability of
   being stuck holding assigned stock for >N months.

3. Least-Squares Monte Carlo (Longstaff-Schwartz 2001): Price American
   options and estimate early assignment probability, critical for
   dividend-paying SP500 stocks.

All simulators target SP500 / MAG7 equities exclusively.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .option_pricer import black_scholes_price


# ─────────────────────────────────────────────────────────────────────
# 1. Block Bootstrap Monte Carlo
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """Result of block bootstrap simulation."""
    # Equity curves: (n_simulations, n_days) array
    equity_curves: np.ndarray
    # Terminal wealth distribution
    terminal_values: np.ndarray
    # Strategy statistics across simulations
    median_return: float
    mean_return: float
    cvar_5: float                    # Expected shortfall at 5%
    var_5: float                     # 5th percentile return
    prob_loss: float                 # P(terminal < initial)
    prob_severe_loss: float          # P(terminal < 0.8 * initial)
    max_drawdown_dist: np.ndarray    # Max drawdown per simulation
    sharpe_dist: np.ndarray          # Sharpe ratio per simulation
    # Confidence intervals
    return_ci_95: Tuple[float, float]
    return_ci_99: Tuple[float, float]

    def summary(self) -> str:
        """Formatted summary report."""
        mdd_median = np.median(self.max_drawdown_dist)
        sharpe_median = np.median(self.sharpe_dist)
        lines = [
            "Block Bootstrap Monte Carlo Report",
            "=" * 50,
            f"Simulations:        {len(self.terminal_values):,}",
            f"Median Return:      {self.median_return:+.2%}",
            f"Mean Return:        {self.mean_return:+.2%}",
            f"95% CI:             [{self.return_ci_95[0]:+.2%}, {self.return_ci_95[1]:+.2%}]",
            f"5% VaR:             {self.var_5:+.2%}",
            f"5% CVaR (ES):       {self.cvar_5:+.2%}",
            f"P(Loss):            {self.prob_loss:.1%}",
            f"P(Severe Loss>20%): {self.prob_severe_loss:.1%}",
            f"Median Max DD:      {mdd_median:.2%}",
            f"Median Sharpe:      {sharpe_median:.2f}",
        ]
        return "\n".join(lines)


class BlockBootstrap:
    """
    Block bootstrap Monte Carlo for strategy equity curves.

    Resamples historical daily P&L in contiguous blocks to preserve
    autocorrelation, volatility clustering, and mean-reversion patterns
    observed in real SP500 returns.

    Reference: Politis & Romano (1994), "The Stationary Bootstrap"
    """

    def __init__(
        self,
        block_size: int = 21,          # ~1 month of trading days
        n_simulations: int = 10000,
        seed: Optional[int] = 42
    ):
        self.block_size = block_size
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        daily_returns: np.ndarray,
        n_days: int = 252,
        initial_capital: float = 100000.0
    ) -> BootstrapResult:
        """
        Generate synthetic equity curves via block bootstrap.

        Args:
            daily_returns: Historical daily strategy returns (not prices).
                           Shape: (n_historical_days,)
            n_days: Number of days per simulation path.
            initial_capital: Starting capital.

        Returns:
            BootstrapResult with full distribution of outcomes.
        """
        n_hist = len(daily_returns)
        if n_hist < self.block_size:
            raise ValueError(
                f"Need at least {self.block_size} historical returns, "
                f"got {n_hist}"
            )

        # Pre-compute blocks count needed per simulation
        n_blocks = int(np.ceil(n_days / self.block_size))

        # Maximum valid start index for a block
        max_start = n_hist - self.block_size

        # Generate all random block starts at once for efficiency
        # Shape: (n_simulations, n_blocks)
        block_starts = self.rng.integers(
            0, max_start + 1,
            size=(self.n_simulations, n_blocks)
        )

        # Build synthetic return series
        equity_curves = np.zeros((self.n_simulations, n_days))

        for sim_idx in range(self.n_simulations):
            # Concatenate blocks
            synth_returns = np.concatenate([
                daily_returns[start:start + self.block_size]
                for start in block_starts[sim_idx]
            ])[:n_days]

            # Build equity curve
            equity_curves[sim_idx] = initial_capital * np.cumprod(1 + synth_returns)

        # Terminal values
        terminal_values = equity_curves[:, -1]
        total_returns = (terminal_values / initial_capital) - 1

        # Max drawdown per simulation
        max_drawdowns = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            curve = equity_curves[i]
            running_max = np.maximum.accumulate(curve)
            drawdowns = (curve - running_max) / running_max
            max_drawdowns[i] = np.min(drawdowns)  # Most negative

        # Annualized Sharpe per simulation
        sharpe_dist = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            # Recover daily returns from equity curve
            daily_r = np.diff(equity_curves[i]) / equity_curves[i, :-1]
            if np.std(daily_r) > 0:
                sharpe_dist[i] = (np.mean(daily_r) / np.std(daily_r)) * np.sqrt(252)

        # Risk metrics
        var_5 = np.percentile(total_returns, 5)
        tail_returns = total_returns[total_returns <= var_5]
        cvar_5 = np.mean(tail_returns) if len(tail_returns) > 0 else var_5

        return BootstrapResult(
            equity_curves=equity_curves,
            terminal_values=terminal_values,
            median_return=float(np.median(total_returns)),
            mean_return=float(np.mean(total_returns)),
            cvar_5=float(cvar_5),
            var_5=float(var_5),
            prob_loss=float(np.mean(total_returns < 0)),
            prob_severe_loss=float(np.mean(total_returns < -0.20)),
            max_drawdown_dist=max_drawdowns,
            sharpe_dist=sharpe_dist,
            return_ci_95=(
                float(np.percentile(total_returns, 2.5)),
                float(np.percentile(total_returns, 97.5))
            ),
            return_ci_99=(
                float(np.percentile(total_returns, 0.5)),
                float(np.percentile(total_returns, 99.5))
            )
        )

    def simulate_stationary(
        self,
        daily_returns: np.ndarray,
        n_days: int = 252,
        initial_capital: float = 100000.0,
        mean_block_size: Optional[int] = None
    ) -> BootstrapResult:
        """
        Stationary bootstrap with random block lengths.

        Block lengths are drawn from a geometric distribution, making the
        resampled series strictly stationary. More robust than fixed-block
        for varying market regimes.

        Args:
            daily_returns: Historical daily strategy returns.
            n_days: Number of days per simulation.
            initial_capital: Starting capital.
            mean_block_size: Expected block length (default: self.block_size).
        """
        n_hist = len(daily_returns)
        if n_hist < 2:
            raise ValueError("Need at least 2 historical returns")

        mean_bs = mean_block_size or self.block_size
        # Probability of ending a block (geometric distribution)
        p_end = 1.0 / mean_bs

        equity_curves = np.zeros((self.n_simulations, n_days))

        for sim_idx in range(self.n_simulations):
            synth_returns = np.zeros(n_days)
            idx = self.rng.integers(0, n_hist)  # Random start

            for day in range(n_days):
                synth_returns[day] = daily_returns[idx]

                # With probability p_end, jump to a new random position
                if self.rng.random() < p_end:
                    idx = self.rng.integers(0, n_hist)
                else:
                    idx = (idx + 1) % n_hist  # Continue block

            equity_curves[sim_idx] = initial_capital * np.cumprod(1 + synth_returns)

        # Reuse metrics computation
        terminal_values = equity_curves[:, -1]
        total_returns = (terminal_values / initial_capital) - 1

        max_drawdowns = np.zeros(self.n_simulations)
        sharpe_dist = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            curve = equity_curves[i]
            running_max = np.maximum.accumulate(curve)
            drawdowns = (curve - running_max) / running_max
            max_drawdowns[i] = np.min(drawdowns)
            daily_r = np.diff(curve) / curve[:-1]
            if np.std(daily_r) > 0:
                sharpe_dist[i] = (np.mean(daily_r) / np.std(daily_r)) * np.sqrt(252)

        var_5 = np.percentile(total_returns, 5)
        tail_returns = total_returns[total_returns <= var_5]
        cvar_5 = np.mean(tail_returns) if len(tail_returns) > 0 else var_5

        return BootstrapResult(
            equity_curves=equity_curves,
            terminal_values=terminal_values,
            median_return=float(np.median(total_returns)),
            mean_return=float(np.mean(total_returns)),
            cvar_5=float(cvar_5),
            var_5=float(var_5),
            prob_loss=float(np.mean(total_returns < 0)),
            prob_severe_loss=float(np.mean(total_returns < -0.20)),
            max_drawdown_dist=max_drawdowns,
            sharpe_dist=sharpe_dist,
            return_ci_95=(
                float(np.percentile(total_returns, 2.5)),
                float(np.percentile(total_returns, 97.5))
            ),
            return_ci_99=(
                float(np.percentile(total_returns, 0.5)),
                float(np.percentile(total_returns, 99.5))
            )
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Jump Diffusion (Merton 1976) with Bagholder Probability
# ─────────────────────────────────────────────────────────────────────

@dataclass
class JumpDiffusionParams:
    """Parameters for Merton jump-diffusion model."""
    mu: float = 0.08              # Annualized drift (SP500 ~8%)
    sigma: float = 0.20           # Annualized diffusion volatility
    jump_intensity: float = 2.0   # Expected jumps per year (lambda)
    jump_mean: float = -0.05      # Mean jump size (log, negative = crashes)
    jump_std: float = 0.10        # Jump size standard deviation
    dividend_yield: float = 0.02  # Continuous dividend yield

    @classmethod
    def from_historical(
        cls,
        daily_returns: np.ndarray,
        threshold_sigma: float = 3.0
    ) -> 'JumpDiffusionParams':
        """
        Calibrate jump-diffusion parameters from historical returns.

        Identifies jumps as returns exceeding threshold_sigma standard
        deviations, then fits GBM + Poisson components separately.
        """
        mu_daily = np.mean(daily_returns)
        sigma_daily = np.std(daily_returns)

        # Identify jumps (returns beyond threshold)
        threshold = threshold_sigma * sigma_daily
        jump_mask = np.abs(daily_returns) > threshold
        normal_mask = ~jump_mask

        n_jumps = np.sum(jump_mask)
        n_total = len(daily_returns)

        # Jump parameters
        jump_intensity = (n_jumps / n_total) * 252  # Annualized
        if n_jumps > 0:
            jump_returns = daily_returns[jump_mask]
            jump_mean = float(np.mean(jump_returns))
            jump_std = float(np.std(jump_returns)) if n_jumps > 1 else 0.05
        else:
            jump_mean = -0.05
            jump_std = 0.10

        # Diffusion parameters (from non-jump returns)
        if np.sum(normal_mask) > 1:
            sigma = float(np.std(daily_returns[normal_mask])) * np.sqrt(252)
        else:
            sigma = sigma_daily * np.sqrt(252)

        mu = float(mu_daily * 252)

        return cls(
            mu=mu,
            sigma=sigma,
            jump_intensity=max(0.1, jump_intensity),
            jump_mean=jump_mean,
            jump_std=max(0.01, jump_std)
        )


@dataclass
class JumpDiffusionResult:
    """Result of jump-diffusion simulation."""
    # Price paths: (n_simulations, n_steps + 1) starting with S0
    paths: np.ndarray
    # Terminal prices
    terminal_prices: np.ndarray
    # Bagholder analysis
    bagholder_probability: float     # P(stuck holding > threshold months)
    median_recovery_days: float      # Median days to recover from assignment
    prob_never_recover: float        # P(never recovers above strike within horizon)
    # Distribution stats
    mean_terminal: float
    median_terminal: float
    var_5_terminal: float            # 5th percentile terminal price
    prob_below_strike: float         # P(S_T < K)
    expected_loss_if_below: float    # E[K - S_T | S_T < K]
    # Jump diagnostics
    avg_jumps_per_path: float
    max_single_day_drop: float

    def summary(self, strike: float) -> str:
        """Formatted summary with bagholder analysis."""
        lines = [
            "Jump Diffusion Simulation Report",
            "=" * 50,
            f"Paths Simulated:       {len(self.terminal_prices):,}",
            f"Strike Price:          ${strike:,.2f}",
            "",
            "--- Terminal Price Distribution ---",
            f"Mean Terminal:         ${self.mean_terminal:,.2f}",
            f"Median Terminal:       ${self.median_terminal:,.2f}",
            f"5% VaR Terminal:       ${self.var_5_terminal:,.2f}",
            f"P(Below Strike):       {self.prob_below_strike:.1%}",
            f"E[Loss | Below]:       ${self.expected_loss_if_below:,.2f}",
            "",
            "--- Bagholder Analysis ---",
            f"P(Stuck > threshold):  {self.bagholder_probability:.1%}",
            f"Median Recovery Days:  {self.median_recovery_days:.0f}",
            f"P(Never Recover):      {self.prob_never_recover:.1%}",
            "",
            "--- Jump Diagnostics ---",
            f"Avg Jumps/Path:        {self.avg_jumps_per_path:.1f}",
            f"Max Single-Day Drop:   {self.max_single_day_drop:.1%}",
        ]
        return "\n".join(lines)


class JumpDiffusionSimulator:
    """
    Merton (1976) jump-diffusion process for equity path simulation.

    dS/S = (mu - lambda*k)dt + sigma*dW + J*dN

    where:
        mu    = drift
        sigma = diffusion volatility
        dW    = Wiener process
        J     = jump size ~ N(jump_mean, jump_std^2)
        dN    = Poisson process with intensity lambda
        k     = E[e^J - 1] = compensator

    Designed for SP500 constituents with empirically calibrated
    jump parameters.
    """

    def __init__(
        self,
        params: Optional[JumpDiffusionParams] = None,
        n_simulations: int = 50000,
        seed: Optional[int] = 42
    ):
        self.params = params or JumpDiffusionParams()
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_paths(
        self,
        S0: float,
        n_days: int = 252,
        dt: float = 1.0 / 252
    ) -> np.ndarray:
        """
        Simulate price paths under jump-diffusion.

        Args:
            S0: Initial stock price.
            n_days: Number of trading days to simulate.
            dt: Time step in years (default: 1 trading day).

        Returns:
            Price paths array of shape (n_simulations, n_days + 1).
        """
        p = self.params
        n_sims = self.n_simulations

        # Compensator: k = E[e^J - 1]
        k = np.exp(p.jump_mean + 0.5 * p.jump_std**2) - 1

        # Adjusted drift (remove jump compensation + dividend)
        drift = (p.mu - p.dividend_yield - p.jump_intensity * k
                 - 0.5 * p.sigma**2) * dt

        # Pre-generate all random components
        # Diffusion (Brownian)
        Z = self.rng.standard_normal(size=(n_sims, n_days))
        diffusion = p.sigma * np.sqrt(dt) * Z

        # Poisson jump counts
        N_jumps = self.rng.poisson(
            lam=p.jump_intensity * dt,
            size=(n_sims, n_days)
        )

        # Jump sizes (compound Poisson: sum of N_jumps normal jumps)
        # For each time step, total jump = sum of N individual jumps
        # E[sum] = N * jump_mean, Var[sum] = N * jump_std^2
        jump_sizes = np.zeros((n_sims, n_days))
        max_jumps = int(N_jumps.max()) if N_jumps.max() > 0 else 0
        for j in range(1, max_jumps + 1):
            mask = N_jumps >= j
            jump_sizes[mask] += self.rng.normal(
                p.jump_mean, p.jump_std, size=np.sum(mask)
            )

        # Log returns: drift + diffusion + jumps
        log_returns = drift + diffusion + jump_sizes

        # Build price paths
        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def bagholder_analysis(
        self,
        S0: float,
        strike: float,
        n_days: int = 504,             # 2 years horizon
        stuck_threshold_days: int = 252,  # 1 year = "stuck"
        paths: Optional[np.ndarray] = None
    ) -> JumpDiffusionResult:
        """
        Simulate paths and compute "bagholder probability."

        Scenario: You sold a put at strike K. The stock drops and you
        get assigned at K. How often are you stuck holding the stock
        below your cost basis for more than `stuck_threshold_days`?

        Args:
            S0: Current stock price.
            strike: Put strike price (your cost basis if assigned).
            n_days: Simulation horizon in trading days.
            stuck_threshold_days: Days below strike to count as "stuck."
            paths: Pre-computed paths (optional, for reuse).

        Returns:
            JumpDiffusionResult with bagholder metrics.
        """
        if paths is None:
            paths = self.simulate_paths(S0, n_days)

        n_sims = paths.shape[0]
        terminal_prices = paths[:, -1]

        # --- Bagholder probability ---
        # For each path, find the first day price recovers above strike
        below_strike = paths[:, 1:] < strike  # (n_sims, n_days)

        recovery_days = np.full(n_sims, np.inf)
        stuck_count = 0
        never_recover_count = 0

        for i in range(n_sims):
            # Find consecutive days below strike from day 1
            below = below_strike[i]
            if not below[0]:
                # Never dropped below strike after assignment
                recovery_days[i] = 0
                continue

            # Find first day above strike
            above_indices = np.where(~below)[0]
            if len(above_indices) == 0:
                # Never recovered within horizon
                recovery_days[i] = np.inf
                never_recover_count += 1
                stuck_count += 1
            else:
                first_recovery = above_indices[0]
                recovery_days[i] = first_recovery
                if first_recovery > stuck_threshold_days:
                    stuck_count += 1

        bagholder_prob = stuck_count / n_sims
        prob_never_recover = never_recover_count / n_sims

        # Median recovery days (excluding never-recovered)
        finite_recovery = recovery_days[np.isfinite(recovery_days)]
        median_recovery = float(np.median(finite_recovery)) if len(finite_recovery) > 0 else float('inf')

        # --- Terminal price distribution ---
        below_mask = terminal_prices < strike
        prob_below = np.mean(below_mask)
        expected_loss_below = float(
            np.mean(strike - terminal_prices[below_mask])
        ) if np.any(below_mask) else 0.0

        # --- Jump diagnostics ---
        # Re-simulate to count jumps (use log returns)
        if paths.shape[1] > 1:
            log_rets = np.diff(np.log(paths), axis=1)
            daily_vol = self.params.sigma / np.sqrt(252)
            jump_threshold = 3.0 * daily_vol
            jump_counts = np.sum(np.abs(log_rets) > jump_threshold, axis=1)
            avg_jumps = float(np.mean(jump_counts))

            # Worst single-day drop
            min_daily = np.min(log_rets)
            max_single_drop = float(np.exp(min_daily) - 1)
        else:
            avg_jumps = 0.0
            max_single_drop = 0.0

        return JumpDiffusionResult(
            paths=paths,
            terminal_prices=terminal_prices,
            bagholder_probability=bagholder_prob,
            median_recovery_days=median_recovery,
            prob_never_recover=prob_never_recover,
            mean_terminal=float(np.mean(terminal_prices)),
            median_terminal=float(np.median(terminal_prices)),
            var_5_terminal=float(np.percentile(terminal_prices, 5)),
            prob_below_strike=float(prob_below),
            expected_loss_if_below=expected_loss_below,
            avg_jumps_per_path=avg_jumps,
            max_single_day_drop=max_single_drop
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Least-Squares Monte Carlo (Longstaff-Schwartz 2001)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LSMResult:
    """Result from Least-Squares Monte Carlo pricing."""
    # Prices
    american_price: float            # American option price
    european_price: float            # European (BSM) benchmark
    early_exercise_premium: float    # American - European
    # Assignment probabilities
    prob_early_exercise: float       # P(exercised before expiry)
    prob_exercise_by_day: np.ndarray # Cumulative exercise probability
    optimal_exercise_boundary: np.ndarray  # Strike-relative boundary
    # Ex-dividend analysis
    prob_exercise_pre_dividend: float   # P(exercised before div date)
    expected_exercise_day: float        # E[exercise day | exercised early]
    # Standard errors
    price_std_error: float

    def summary(self) -> str:
        """Formatted summary."""
        lines = [
            "Least-Squares Monte Carlo Report",
            "=" * 50,
            f"American Price:       ${self.american_price:.4f}",
            f"European Price:       ${self.european_price:.4f}",
            f"Early Exercise Prem:  ${self.early_exercise_premium:.4f}",
            f"P(Early Exercise):    {self.prob_early_exercise:.2%}",
            f"P(Exercise Pre-Div):  {self.prob_exercise_pre_dividend:.2%}",
            f"E[Exercise Day]:      {self.expected_exercise_day:.1f}",
            f"Std Error:            ${self.price_std_error:.6f}",
        ]
        return "\n".join(lines)


class LSMPricer:
    """
    Longstaff-Schwartz (2001) Least-Squares Monte Carlo for American options.

    Prices American puts and calls on dividend-paying SP500 stocks.
    Critical for the Wheel strategy: if you sell a put, your counterparty
    holds an American put — knowing the early exercise probability helps
    predict assignment risk.

    For covered calls on dividend-paying stocks, early exercise is common
    just before ex-dividend dates.

    Key insight for Wheel traders:
    - American put assignment risk increases as stock drops below strike
    - American call assignment risk spikes before large dividends
    - LSM quantifies both probabilities precisely

    Reference: Longstaff & Schwartz (2001), "Valuing American Options
    by Simulation: A Simple Least-Squares Approach", RFS.
    """

    def __init__(
        self,
        n_paths: int = 50000,
        n_steps_per_day: int = 1,
        polynomial_degree: int = 3,
        seed: Optional[int] = 42
    ):
        self.n_paths = n_paths
        self.n_steps_per_day = n_steps_per_day
        self.poly_degree = polynomial_degree
        self.rng = np.random.default_rng(seed)

    def _generate_gbm_paths(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        q: float,
        n_steps: int,
        dividend_dates: Optional[List[int]] = None,
        dividend_amounts: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate GBM paths under risk-neutral measure.

        Handles discrete dividends by reducing stock price on ex-dates.

        Args:
            S0: Initial stock price.
            r: Risk-free rate.
            sigma: Volatility.
            T: Time to expiration (years).
            q: Continuous dividend yield.
            n_steps: Number of time steps.
            dividend_dates: Time step indices of ex-dividend dates.
            dividend_amounts: Dollar dividend amounts.

        Returns:
            Paths array of shape (n_paths, n_steps + 1).
        """
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        Z = self.rng.standard_normal(size=(self.n_paths, n_steps))
        log_returns = drift + vol * Z

        log_paths = np.zeros((self.n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(S0)
        log_paths[:, 1:] = np.cumsum(log_returns, axis=1) + np.log(S0)

        paths = np.exp(log_paths)

        # Apply discrete dividends (subtract dollar amount on ex-dates)
        if dividend_dates and dividend_amounts:
            for div_step, div_amount in zip(dividend_dates, dividend_amounts):
                if 0 < div_step <= n_steps:
                    # Stock drops by dividend amount on ex-date
                    ratio = np.maximum(
                        0.01,
                        (paths[:, div_step] - div_amount) / paths[:, div_step]
                    )
                    paths[:, div_step:] *= ratio[:, np.newaxis]

        return paths

    def price(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'put',
        q: float = 0.0,
        dividend_dates: Optional[List[int]] = None,
        dividend_amounts: Optional[List[float]] = None
    ) -> LSMResult:
        """
        Price an American option using LSM.

        Args:
            S0: Current stock price.
            K: Strike price.
            T: Time to expiration (years).
            r: Risk-free rate.
            sigma: Implied volatility.
            option_type: 'put' or 'call'.
            q: Continuous dividend yield.
            dividend_dates: Time step indices for discrete dividends.
            dividend_amounts: Dollar dividend amounts at each date.

        Returns:
            LSMResult with price, exercise probabilities, and boundary.
        """
        n_steps = max(int(T * 252 * self.n_steps_per_day), 10)
        dt = T / n_steps

        # Generate paths under risk-neutral measure
        paths = self._generate_gbm_paths(
            S0, r, sigma, T, q, n_steps,
            dividend_dates, dividend_amounts
        )

        # Payoff function
        if option_type == 'put':
            payoff_fn = lambda S: np.maximum(K - S, 0)
        else:
            payoff_fn = lambda S: np.maximum(S - K, 0)

        # --- Backward induction (Longstaff-Schwartz) ---
        # Cash flows from exercise (initialized at expiration)
        cashflows = payoff_fn(paths[:, -1])
        exercise_time = np.full(self.n_paths, n_steps)

        # Store exercise boundary (stock price where exercise = continue)
        exercise_boundary = np.full(n_steps + 1, np.nan)
        exercise_boundary[-1] = K  # At expiry, boundary = strike

        # Exercise probability by time step
        exercise_at_step = np.zeros(n_steps + 1)
        exercise_at_step[-1] = np.mean(payoff_fn(paths[:, -1]) > 0)

        # Backward sweep from T-1 to 1
        for t in range(n_steps - 1, 0, -1):
            S_t = paths[:, t]
            immediate_payoff = payoff_fn(S_t)

            # Only consider in-the-money paths for regression
            itm_mask = immediate_payoff > 0
            n_itm = np.sum(itm_mask)

            if n_itm < self.poly_degree + 1:
                continue

            # Discounted future cashflows for ITM paths
            future_cf = cashflows[itm_mask] * np.exp(
                -r * dt * (exercise_time[itm_mask] - t)
            )

            # Regression basis: Laguerre polynomials (standard in LSM)
            S_itm = S_t[itm_mask]
            X = self._laguerre_basis(S_itm / K, self.poly_degree)

            # Least-squares regression: E[continuation | S_t]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, future_cf, rcond=None)
                continuation_value = X @ coeffs
            except np.linalg.LinAlgError:
                continue

            # Exercise decision: exercise if immediate > continuation
            exercise_mask_itm = immediate_payoff[itm_mask] > continuation_value

            # Update exercise times and cashflows
            exercise_indices = np.where(itm_mask)[0][exercise_mask_itm]
            cashflows[exercise_indices] = immediate_payoff[exercise_indices]
            exercise_time[exercise_indices] = t

            # Exercise boundary: stock price where exercise ~ continuation
            if n_itm > 0 and np.any(exercise_mask_itm):
                exercised_prices = S_itm[exercise_mask_itm]
                if option_type == 'put':
                    exercise_boundary[t] = np.max(exercised_prices)
                else:
                    exercise_boundary[t] = np.min(exercised_prices)

        # --- Compute results ---
        # American price = discounted expected cashflows
        discount_factors = np.exp(-r * dt * exercise_time)
        discounted_cf = cashflows * discount_factors
        american_price = float(np.mean(discounted_cf))
        price_std_error = float(np.std(discounted_cf) / np.sqrt(self.n_paths))

        # European price (BSM benchmark)
        european_price = black_scholes_price(S0, K, T, r, sigma, option_type, q)

        # Early exercise premium
        early_ex_premium = max(0, american_price - european_price)

        # Exercise probability
        early_exercise_mask = exercise_time < n_steps
        prob_early = float(np.mean(early_exercise_mask))

        # Cumulative exercise probability by day
        step_to_day = n_steps / (T * 252) if T > 0 else 1
        n_cal_days = int(T * 252)
        prob_by_day = np.zeros(max(n_cal_days, 1))
        for d in range(n_cal_days):
            step = int(d * step_to_day)
            prob_by_day[d] = float(np.mean(exercise_time <= step))

        # Expected exercise day (conditional on early exercise)
        if prob_early > 0:
            early_times = exercise_time[early_exercise_mask]
            expected_day = float(np.mean(early_times / step_to_day))
        else:
            expected_day = float(n_cal_days)

        # Pre-dividend exercise probability
        prob_pre_div = 0.0
        if dividend_dates:
            for div_step in dividend_dates:
                # Count exercises in the step just before dividend
                window_start = max(0, div_step - max(int(step_to_day), 1))
                pre_div_mask = (
                    (exercise_time >= window_start)
                    & (exercise_time < div_step)
                )
                prob_pre_div += float(np.mean(pre_div_mask))

        return LSMResult(
            american_price=american_price,
            european_price=european_price,
            early_exercise_premium=early_ex_premium,
            prob_early_exercise=prob_early,
            prob_exercise_by_day=prob_by_day,
            optimal_exercise_boundary=exercise_boundary,
            prob_exercise_pre_dividend=prob_pre_div,
            expected_exercise_day=expected_day,
            price_std_error=price_std_error
        )

    def assignment_risk(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'put',
        q: float = 0.0,
        dividend_dates: Optional[List[int]] = None,
        dividend_amounts: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Quick assignment risk assessment for Wheel strategy.

        Returns probability of being assigned at various time horizons.
        Used to decide whether to sell a particular option.
        """
        result = self.price(
            S0, K, T, r, sigma, option_type, q,
            dividend_dates, dividend_amounts
        )

        n_days = int(T * 252)
        risk = {
            'prob_early_assignment': result.prob_early_exercise,
            'prob_pre_dividend_assignment': result.prob_exercise_pre_dividend,
            'expected_assignment_day': result.expected_exercise_day,
            'early_exercise_premium_pct': (
                result.early_exercise_premium / result.american_price * 100
                if result.american_price > 0 else 0
            ),
            'american_price': result.american_price,
            'european_price': result.european_price,
        }

        # Assignment probability at key horizons
        for days in [7, 14, 21, 30]:
            if days < n_days and days < len(result.prob_exercise_by_day):
                risk[f'prob_assigned_within_{days}d'] = float(
                    result.prob_exercise_by_day[days]
                )

        return risk

    @staticmethod
    def _laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
        """
        Weighted Laguerre polynomial basis for LSM regression.

        Standard basis from Longstaff-Schwartz (2001):
        L0(x) = exp(-x/2)
        L1(x) = exp(-x/2) * (1 - x)
        L2(x) = exp(-x/2) * (1 - 2x + x^2/2)
        L3(x) = exp(-x/2) * (1 - 3x + 3x^2/2 - x^3/6)
        """
        n = len(x)
        X = np.ones((n, degree + 1))

        # Use simple polynomial basis (more numerically stable for
        # typical moneyness ranges)
        for d in range(1, degree + 1):
            X[:, d] = x ** d

        return X


# ─────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────

def run_bootstrap_analysis(
    daily_returns: np.ndarray,
    n_simulations: int = 10000,
    n_days: int = 252,
    initial_capital: float = 100000.0,
    block_size: int = 21,
    seed: Optional[int] = 42
) -> BootstrapResult:
    """
    Quick block bootstrap analysis.

    Args:
        daily_returns: Historical daily strategy returns.
        n_simulations: Number of synthetic equity curves.
        n_days: Days per simulation.
        initial_capital: Starting capital.
        block_size: Bootstrap block size in days.
        seed: Random seed.

    Returns:
        BootstrapResult with full distribution analysis.
    """
    bootstrap = BlockBootstrap(
        block_size=block_size,
        n_simulations=n_simulations,
        seed=seed
    )
    return bootstrap.simulate(daily_returns, n_days, initial_capital)


def run_bagholder_analysis(
    S0: float,
    strike: float,
    sigma: float = 0.30,
    n_simulations: int = 50000,
    horizon_days: int = 504,
    stuck_threshold_days: int = 252,
    params: Optional[JumpDiffusionParams] = None,
    seed: Optional[int] = 42
) -> JumpDiffusionResult:
    """
    Quick bagholder probability analysis for a put sale.

    "If I sell a put at this strike, how often will I be stuck
    holding the stock for over a year?"

    Args:
        S0: Current stock price.
        strike: Put strike (= assignment cost basis).
        sigma: Stock volatility (overrides params.sigma if no params).
        n_simulations: Number of paths.
        horizon_days: Trading days to simulate (504 = 2 years).
        stuck_threshold_days: Days below strike to count as "stuck."
        params: Full JumpDiffusionParams (optional).
        seed: Random seed.

    Returns:
        JumpDiffusionResult with bagholder metrics.
    """
    if params is None:
        params = JumpDiffusionParams(sigma=sigma)

    sim = JumpDiffusionSimulator(
        params=params,
        n_simulations=n_simulations,
        seed=seed
    )
    return sim.bagholder_analysis(
        S0, strike, horizon_days, stuck_threshold_days
    )


def price_american_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'put',
    q: float = 0.0,
    n_paths: int = 50000,
    seed: Optional[int] = 42
) -> LSMResult:
    """
    Quick American option pricing via LSM.

    Args:
        S0: Stock price.
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'put' or 'call'.
        q: Continuous dividend yield.
        n_paths: Number of simulation paths.
        seed: Random seed.

    Returns:
        LSMResult with price and exercise analysis.
    """
    pricer = LSMPricer(n_paths=n_paths, seed=seed)
    return pricer.price(S0, K, T, r, sigma, option_type, q)
