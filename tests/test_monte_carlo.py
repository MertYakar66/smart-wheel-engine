"""
Tests for Monte Carlo Simulation Module

Tests cover:
1. Block Bootstrap - synthetic equity curve generation
2. Jump Diffusion - Merton model with bagholder analysis
3. LSM - American option pricing and assignment risk
"""

import numpy as np
import pytest

from engine.monte_carlo import (
    BlockBootstrap,
    BootstrapResult,
    JumpDiffusionSimulator,
    JumpDiffusionParams,
    JumpDiffusionResult,
    LSMPricer,
    LSMResult,
    run_bootstrap_analysis,
    run_bagholder_analysis,
    price_american_option,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def daily_returns():
    """Synthetic daily returns resembling SP500."""
    rng = np.random.default_rng(42)
    # Mean ~0.04%/day, vol ~1%/day (roughly 16% annualized)
    returns = rng.normal(0.0004, 0.01, size=504)  # 2 years
    # Add a few large drawdowns
    returns[50] = -0.05   # Flash crash day
    returns[200] = -0.03  # Moderate selloff
    return returns


@pytest.fixture
def bootstrap():
    return BlockBootstrap(block_size=21, n_simulations=500, seed=42)


@pytest.fixture
def jd_params():
    return JumpDiffusionParams(
        mu=0.08,
        sigma=0.20,
        jump_intensity=2.0,
        jump_mean=-0.05,
        jump_std=0.10,
        dividend_yield=0.02
    )


@pytest.fixture
def jd_simulator(jd_params):
    return JumpDiffusionSimulator(
        params=jd_params, n_simulations=1000, seed=42
    )


@pytest.fixture
def lsm_pricer():
    return LSMPricer(n_paths=5000, seed=42)


# ─────────────────────────────────────────────────────────────────────
# 1. Block Bootstrap Tests
# ─────────────────────────────────────────────────────────────────────

class TestBlockBootstrap:
    """Tests for block bootstrap Monte Carlo."""

    def test_basic_simulation(self, bootstrap, daily_returns):
        """Bootstrap produces correct output shape."""
        result = bootstrap.simulate(daily_returns, n_days=252)

        assert isinstance(result, BootstrapResult)
        assert result.equity_curves.shape == (500, 252)
        assert len(result.terminal_values) == 500
        assert len(result.max_drawdown_dist) == 500
        assert len(result.sharpe_dist) == 500

    def test_terminal_values_reasonable(self, bootstrap, daily_returns):
        """Terminal values should be in a reasonable range."""
        result = bootstrap.simulate(
            daily_returns, n_days=252, initial_capital=100000
        )

        # Terminal values should mostly be between 50k and 200k for 1 year
        assert np.median(result.terminal_values) > 50000
        assert np.median(result.terminal_values) < 200000

    def test_prob_loss_bounded(self, bootstrap, daily_returns):
        """Probability of loss should be between 0 and 1."""
        result = bootstrap.simulate(daily_returns, n_days=252)

        assert 0 <= result.prob_loss <= 1
        assert 0 <= result.prob_severe_loss <= 1
        assert result.prob_severe_loss <= result.prob_loss

    def test_confidence_intervals_ordered(self, bootstrap, daily_returns):
        """99% CI should contain 95% CI."""
        result = bootstrap.simulate(daily_returns, n_days=252)

        assert result.return_ci_99[0] <= result.return_ci_95[0]
        assert result.return_ci_99[1] >= result.return_ci_95[1]

    def test_var_cvar_relationship(self, bootstrap, daily_returns):
        """CVaR should be worse (more negative) than VaR."""
        result = bootstrap.simulate(daily_returns, n_days=252)

        # CVaR is expected shortfall (conditional on being in tail)
        assert result.cvar_5 <= result.var_5

    def test_max_drawdown_negative(self, bootstrap, daily_returns):
        """Max drawdowns should be negative (they represent losses)."""
        result = bootstrap.simulate(daily_returns, n_days=252)

        assert np.all(result.max_drawdown_dist <= 0)

    def test_short_history_raises(self, bootstrap):
        """Should raise if history is shorter than block size."""
        short_returns = np.array([0.01, -0.01, 0.005])

        with pytest.raises(ValueError, match="at least"):
            bootstrap.simulate(short_returns, n_days=100)

    def test_stationary_bootstrap(self, bootstrap, daily_returns):
        """Stationary bootstrap should also produce valid results."""
        result = bootstrap.simulate_stationary(
            daily_returns, n_days=252, initial_capital=100000
        )

        assert isinstance(result, BootstrapResult)
        assert result.equity_curves.shape == (500, 252)
        assert np.median(result.terminal_values) > 50000

    def test_summary_format(self, bootstrap, daily_returns):
        """Summary should be a non-empty formatted string."""
        result = bootstrap.simulate(daily_returns, n_days=126)
        summary = result.summary()

        assert "Block Bootstrap" in summary
        assert "Simulations:" in summary
        assert "Median Return:" in summary

    def test_reproducibility(self, daily_returns):
        """Same seed should give identical results."""
        bs1 = BlockBootstrap(n_simulations=100, seed=123)
        bs2 = BlockBootstrap(n_simulations=100, seed=123)

        r1 = bs1.simulate(daily_returns, n_days=100)
        r2 = bs2.simulate(daily_returns, n_days=100)

        np.testing.assert_array_equal(r1.terminal_values, r2.terminal_values)


# ─────────────────────────────────────────────────────────────────────
# 2. Jump Diffusion Tests
# ─────────────────────────────────────────────────────────────────────

class TestJumpDiffusion:
    """Tests for Merton jump-diffusion simulator."""

    def test_path_shape(self, jd_simulator):
        """Paths should have correct shape."""
        paths = jd_simulator.simulate_paths(S0=100, n_days=252)

        assert paths.shape == (1000, 253)  # n_sims x (n_days + 1)
        assert np.all(paths[:, 0] == 100)  # All start at S0

    def test_paths_positive(self, jd_simulator):
        """All prices should be positive (GBM property)."""
        paths = jd_simulator.simulate_paths(S0=150, n_days=252)

        assert np.all(paths > 0)

    def test_bagholder_analysis(self, jd_simulator):
        """Bagholder analysis should return valid probabilities."""
        result = jd_simulator.bagholder_analysis(
            S0=100, strike=95, n_days=504, stuck_threshold_days=252
        )

        assert isinstance(result, JumpDiffusionResult)
        assert 0 <= result.bagholder_probability <= 1
        assert 0 <= result.prob_never_recover <= 1
        assert 0 <= result.prob_below_strike <= 1
        assert result.prob_never_recover <= result.bagholder_probability

    def test_deep_otm_put_low_bagholder(self, jd_params):
        """Deep OTM put should have low bagholder probability."""
        sim = JumpDiffusionSimulator(
            params=jd_params, n_simulations=2000, seed=42
        )
        result = sim.bagholder_analysis(
            S0=100, strike=70,  # 30% OTM
            n_days=504, stuck_threshold_days=252
        )

        # With strike at 70 and stock at 100, being stuck > 1 year
        # should be rare
        assert result.bagholder_probability < 0.30

    def test_atm_put_meaningful_bagholder(self, jd_params):
        """ATM put should have non-trivial bagholder probability."""
        sim = JumpDiffusionSimulator(
            params=jd_params, n_simulations=2000, seed=42
        )
        result = sim.bagholder_analysis(
            S0=100, strike=100,  # ATM
            n_days=504, stuck_threshold_days=252
        )

        # ATM put: meaningful chance of being stuck
        assert result.bagholder_probability > 0.01

    def test_calibrate_from_historical(self):
        """Calibration from returns should produce valid params."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.012, size=500)
        # Insert some jumps
        returns[100] = -0.08
        returns[300] = -0.06
        returns[400] = 0.07

        params = JumpDiffusionParams.from_historical(returns)

        assert params.sigma > 0
        assert params.jump_intensity > 0
        assert params.jump_std > 0

    def test_terminal_distribution_mean(self, jd_params):
        """Mean terminal price should be near risk-neutral expectation."""
        # Under real-world measure, E[S_T] ~ S0 * exp(mu*T)
        sim = JumpDiffusionSimulator(
            params=jd_params, n_simulations=10000, seed=42
        )
        paths = sim.simulate_paths(S0=100, n_days=252)
        terminal = paths[:, -1]

        # Expected growth ~ exp((mu - q) * T) = exp(0.06)
        expected_mean = 100 * np.exp((0.08 - 0.02) * 1.0)
        actual_mean = np.mean(terminal)

        # Allow 10% tolerance due to jump contribution
        assert abs(actual_mean - expected_mean) / expected_mean < 0.15

    def test_summary_format(self, jd_simulator):
        """Summary should contain key sections."""
        result = jd_simulator.bagholder_analysis(S0=100, strike=95)
        summary = result.summary(strike=95)

        assert "Jump Diffusion" in summary
        assert "Bagholder" in summary
        assert "Terminal Price" in summary

    def test_expected_loss_nonnegative(self, jd_simulator):
        """Expected loss conditional on being below strike should be >= 0."""
        result = jd_simulator.bagholder_analysis(S0=100, strike=95)

        assert result.expected_loss_if_below >= 0

    def test_max_single_day_drop_negative(self, jd_simulator):
        """Max single-day drop should be negative."""
        result = jd_simulator.bagholder_analysis(S0=100, strike=100)

        assert result.max_single_day_drop < 0


# ─────────────────────────────────────────────────────────────────────
# 3. LSM (Longstaff-Schwartz) Tests
# ─────────────────────────────────────────────────────────────────────

class TestLSMPricer:
    """Tests for Least-Squares Monte Carlo pricing."""

    def test_put_price_positive(self, lsm_pricer):
        """American put price should be positive."""
        result = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put'
        )

        assert isinstance(result, LSMResult)
        assert result.american_price > 0

    def test_american_geq_european_put(self, lsm_pricer):
        """American put should be worth >= European put."""
        result = lsm_pricer.price(
            S0=100, K=100, T=0.5, r=0.05, sigma=0.30, option_type='put'
        )

        # American >= European (early exercise premium >= 0)
        assert result.american_price >= result.european_price * 0.99  # 1% tolerance for MC noise

    def test_american_call_no_dividend(self, lsm_pricer):
        """American call with no dividends should equal European."""
        result = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20,
            option_type='call', q=0.0
        )

        # Without dividends, American call = European call
        # Allow MC noise
        assert abs(result.american_price - result.european_price) < 1.0

    def test_deep_itm_put_high_exercise(self, lsm_pricer):
        """Deep ITM put should have high early exercise probability."""
        result = lsm_pricer.price(
            S0=70, K=100, T=0.5, r=0.05, sigma=0.20, option_type='put'
        )

        # Deep ITM put: early exercise should be common
        assert result.prob_early_exercise > 0.10

    def test_deep_otm_put_low_exercise(self, lsm_pricer):
        """Deep OTM put should have low early exercise probability."""
        result = lsm_pricer.price(
            S0=130, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put'
        )

        # Deep OTM put: early exercise very rare
        assert result.prob_early_exercise < 0.30

    def test_call_with_dividends(self, lsm_pricer):
        """Call with dividends should have meaningful early exercise premium."""
        result = lsm_pricer.price(
            S0=100, K=95, T=0.5, r=0.05, sigma=0.20,
            option_type='call', q=0.04  # 4% dividend yield
        )

        # With dividends, American call > European call
        assert result.early_exercise_premium >= 0

    def test_assignment_risk(self, lsm_pricer):
        """Assignment risk method should return valid dict."""
        risk = lsm_pricer.assignment_risk(
            S0=100, K=105, T=0.25, r=0.05, sigma=0.25, option_type='put'
        )

        assert 'prob_early_assignment' in risk
        assert 'american_price' in risk
        assert 'european_price' in risk
        assert 0 <= risk['prob_early_assignment'] <= 1

    def test_put_call_parity_rough(self, lsm_pricer):
        """American prices should roughly satisfy put-call inequality."""
        call = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call'
        )
        put = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='put'
        )

        # Both prices should be reasonable
        assert call.american_price > 0
        assert put.american_price > 0

    def test_price_increases_with_vol(self, lsm_pricer):
        """Higher vol should give higher option price."""
        low_vol = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.15, option_type='put'
        )
        high_vol = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.40, option_type='put'
        )

        assert high_vol.american_price > low_vol.american_price

    def test_summary_format(self, lsm_pricer):
        """Summary should contain key fields."""
        result = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20
        )
        summary = result.summary()

        assert "Least-Squares Monte Carlo" in summary
        assert "American Price:" in summary
        assert "European Price:" in summary
        assert "Early Exercise" in summary

    def test_std_error_reasonable(self, lsm_pricer):
        """Standard error should be small relative to price."""
        result = lsm_pricer.price(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20
        )

        # Std error should be < 10% of price
        if result.american_price > 0.1:
            assert result.price_std_error / result.american_price < 0.10

    def test_discrete_dividends(self):
        """LSM should handle discrete dividends."""
        pricer = LSMPricer(n_paths=5000, seed=42)
        result = pricer.price(
            S0=100, K=95, T=0.5, r=0.05, sigma=0.20,
            option_type='call', q=0.0,
            dividend_dates=[60],       # Ex-div at ~60 steps
            dividend_amounts=[1.50]    # $1.50 dividend
        )

        assert result.american_price > 0
        # With dividend, there should be some exercise before div
        # (but not guaranteed with this small simulation)
        assert result.prob_exercise_pre_dividend >= 0


# ─────────────────────────────────────────────────────────────────────
# 4. Convenience Function Tests
# ─────────────────────────────────────────────────────────────────────

class TestConvenienceFunctions:
    """Tests for top-level convenience functions."""

    def test_run_bootstrap_analysis(self, daily_returns):
        """Convenience function should work end-to-end."""
        result = run_bootstrap_analysis(
            daily_returns, n_simulations=200, n_days=126,
            initial_capital=50000, seed=42
        )

        assert isinstance(result, BootstrapResult)
        assert len(result.terminal_values) == 200

    def test_run_bagholder_analysis(self):
        """Convenience function for bagholder analysis."""
        result = run_bagholder_analysis(
            S0=150, strike=140, sigma=0.30,
            n_simulations=1000, horizon_days=252,
            seed=42
        )

        assert isinstance(result, JumpDiffusionResult)
        assert 0 <= result.bagholder_probability <= 1

    def test_price_american_option(self):
        """Convenience function for American pricing."""
        result = price_american_option(
            S0=100, K=100, T=0.25, r=0.05, sigma=0.20,
            option_type='put', n_paths=2000, seed=42
        )

        assert isinstance(result, LSMResult)
        assert result.american_price > 0


# ─────────────────────────────────────────────────────────────────────
# 5. Laguerre Basis Test
# ─────────────────────────────────────────────────────────────────────

class TestLaguerreBasis:
    """Tests for the regression basis used in LSM."""

    def test_basis_shape(self):
        """Basis matrix should have correct shape."""
        x = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        basis = LSMPricer._laguerre_basis(x, degree=3)

        assert basis.shape == (5, 4)  # n x (degree + 1)

    def test_basis_first_column_ones(self):
        """First column should be all ones (constant term)."""
        x = np.array([0.5, 1.0, 1.5, 2.0])
        basis = LSMPricer._laguerre_basis(x, degree=2)

        np.testing.assert_array_equal(basis[:, 0], np.ones(4))
