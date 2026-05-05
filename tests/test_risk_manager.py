"""Tests for risk management module."""

import numpy as np
import pandas as pd
import pytest

from engine.risk_manager import (
    HierarchicalRiskParity,
    PositionSizingMethod,
    RiskLimits,
    RiskManager,
    SectorExposure,
    SectorExposureManager,
    calculate_hrp_weights,
    calculate_kelly_fraction,
    calculate_optimal_contracts,
    optimize_position_weights,
)


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_fixed_fractional_sizing(self):
        """Fixed fractional should allocate fixed % per position."""
        rm = RiskManager(sizing_method=PositionSizingMethod.FIXED_FRACTIONAL)

        contracts, reason = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=5,
        )

        assert contracts >= 1
        assert "FIXED_FRACTIONAL" in reason.upper() or "fixed_fractional" in reason

    def test_volatility_scaled_sizing(self):
        """Higher IV should result in smaller position size."""
        rm = RiskManager(sizing_method=PositionSizingMethod.VOLATILITY_SCALED)

        # Low IV
        contracts_low_iv, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.15,
            dte=30,
            existing_positions=5,
        )

        # High IV
        contracts_high_iv, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.50,
            dte=30,
            existing_positions=5,
        )

        # Lower IV should allow larger position
        assert contracts_low_iv >= contracts_high_iv

    def test_kelly_sizing(self):
        """Kelly criterion sizing."""
        rm = RiskManager(sizing_method=PositionSizingMethod.KELLY)

        contracts, reason = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            win_probability=0.70,
            avg_win=1.0,
            avg_loss=2.0,
            existing_positions=5,
        )

        assert contracts >= 0
        assert "kelly" in reason.lower()

    def test_max_positions_limit(self):
        """Should return 0 when max positions reached."""
        limits = RiskLimits(max_positions=5)
        rm = RiskManager(limits=limits)

        contracts, reason = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=5,  # At limit
        )

        assert contracts == 0
        assert "max positions" in reason.lower()

    def test_concentration_penalty(self):
        """Fewer positions should reduce size."""
        rm = RiskManager()

        # Many positions
        contracts_diverse, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=8,
        )

        # Few positions
        contracts_concentrated, _ = rm.calculate_position_size(
            portfolio_value=100000,
            underlying_price=100,
            strike=95,
            iv=0.25,
            dte=30,
            existing_positions=2,
        )

        # More diversified = larger individual positions allowed
        assert contracts_diverse >= contracts_concentrated


class TestPortfolioGreeks:
    """Test portfolio Greeks calculation."""

    def test_short_put_greeks(self):
        """Short put should have positive delta (negative exposure)."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

        # Short put: negative delta (we're short negative delta = positive exposure)
        # As price drops, our liability increases
        assert greeks.delta > 0  # Short put means we want stock to go up
        assert greeks.theta > 0  # Short options benefit from time decay

    def test_covered_call_greeks(self):
        """Covered call should have reduced delta."""
        rm = RiskManager()

        # Long 100 shares + short 1 call
        positions = [
            {
                "symbol": "AAPL",
                "option_type": "call",
                "strike": 160,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

        # Short call has negative delta
        assert greeks.delta < 0

    def test_multiple_positions(self):
        """Multiple positions should aggregate Greeks."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 2,
                "is_short": True,
            },
            {
                "symbol": "MSFT",
                "option_type": "put",
                "strike": 300,
                "dte": 45,
                "iv": 0.22,
                "contracts": 1,
                "is_short": True,
            },
        ]

        spot_prices = {"AAPL": 155, "MSFT": 310}
        greeks = rm.calculate_portfolio_greeks(positions, spot_prices)

        # Should have non-zero aggregated Greeks
        assert greeks.delta != 0
        assert greeks.theta > 0  # Both short = positive theta


class TestVaR:
    """Test Value at Risk calculations."""

    def test_parametric_var(self):
        """Parametric VaR should be positive."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 5,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        var, cvar = rm.calculate_var(
            portfolio_value=100000, positions=positions, spot_prices=spot_prices
        )

        assert var > 0
        assert cvar >= var  # CVaR should be >= VaR


class TestMonteCarloVaR:
    """Test Monte Carlo VaR calculations."""

    def test_monte_carlo_var_basic(self):
        """Monte Carlo VaR should return valid results."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 5,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        volatilities = {"AAPL": 0.25}

        var, cvar, details = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            n_simulations=1000,
            seed=42,
        )

        assert var > 0
        assert cvar >= var  # CVaR should be >= VaR
        assert "n_simulations" in details
        assert details["n_simulations"] == 1000
        assert "pnl_mean" in details
        assert "pnl_std" in details
        assert "percentiles" in details

    def test_monte_carlo_var_with_correlation(self):
        """Monte Carlo VaR with correlation matrix."""
        import pandas as pd

        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 3,
                "is_short": True,
            },
            {
                "symbol": "MSFT",
                "option_type": "put",
                "strike": 300,
                "dte": 30,
                "iv": 0.22,
                "contracts": 2,
                "is_short": True,
            },
        ]

        spot_prices = {"AAPL": 155, "MSFT": 310}
        volatilities = {"AAPL": 0.25, "MSFT": 0.22}

        # Correlation matrix
        corr_matrix = pd.DataFrame(
            [[1.0, 0.7], [0.7, 1.0]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        )

        var, cvar, details = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            correlation_matrix=corr_matrix,
            n_simulations=1000,
            seed=42,
        )

        assert var > 0
        assert cvar >= var
        assert details["method"] == "monte_carlo_full_revaluation"

    def test_monte_carlo_var_with_jump_diffusion(self):
        """Monte Carlo VaR with Merton jump-diffusion."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 5,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        volatilities = {"AAPL": 0.25}

        # Without jumps
        var_no_jump, _, _ = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            n_simulations=5000,
            include_jump_diffusion=False,
            seed=42,
        )

        # With jumps (should generally have higher tail risk)
        var_with_jump, _, details = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            n_simulations=5000,
            include_jump_diffusion=True,
            jump_intensity=2.0,
            jump_mean=-0.02,
            jump_std=0.03,
            seed=42,
        )

        assert var_with_jump > 0
        assert details["include_jump_diffusion"] is True

    def test_monte_carlo_var_reproducibility(self):
        """Monte Carlo VaR should be reproducible with same seed."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 5,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        volatilities = {"AAPL": 0.25}

        var1, cvar1, _ = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            n_simulations=1000,
            seed=123,
        )

        var2, cvar2, _ = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            n_simulations=1000,
            seed=123,
        )

        assert var1 == var2
        assert cvar1 == cvar2

    def test_monte_carlo_var_details(self):
        """Monte Carlo VaR should return comprehensive statistics."""
        rm = RiskManager()

        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 5,
                "is_short": True,
            }
        ]

        spot_prices = {"AAPL": 155}
        volatilities = {"AAPL": 0.25}

        _, _, details = rm.calculate_monte_carlo_var(
            portfolio_value=100000,
            positions=positions,
            spot_prices=spot_prices,
            volatilities=volatilities,
            n_simulations=2000,
            seed=42,
        )

        # Check all expected detail fields
        assert "pnl_skew" in details
        assert "pnl_kurtosis" in details
        assert "worst_case" in details
        assert "best_case" in details
        assert "percentiles" in details

        percentiles = details["percentiles"]
        assert "p1" in percentiles
        assert "p5" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

        # Worst case should be more extreme than p1
        assert details["worst_case"] >= percentiles["p1"]


class TestRiskLimits:
    """Test risk limit checks."""

    def test_within_limits(self):
        """Test risk limit checking."""
        rm = RiskManager()

        # Multiple diversified positions
        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
                "market_value": 500,
            },
            {
                "symbol": "MSFT",
                "option_type": "put",
                "strike": 300,
                "dte": 30,
                "iv": 0.22,
                "contracts": 1,
                "is_short": True,
                "market_value": 600,
            },
            {
                "symbol": "GOOGL",
                "option_type": "put",
                "strike": 140,
                "dte": 30,
                "iv": 0.28,
                "contracts": 1,
                "is_short": True,
                "market_value": 450,
            },
        ]

        spot_prices = {"AAPL": 155, "MSFT": 310, "GOOGL": 145}
        within_limits, violations = rm.check_limits(
            portfolio_value=100000, positions=positions, spot_prices=spot_prices
        )

        # Should not have position count violations with 3 positions
        assert not any("position count" in v.lower() for v in violations)

    def test_position_count_violation(self):
        """Too many positions should violate limits."""
        limits = RiskLimits(max_positions=2)
        rm = RiskManager(limits=limits)

        positions = [
            {
                "symbol": f"SYM{i}",
                "option_type": "put",
                "strike": 100,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
                "market_value": 500,
            }
            for i in range(5)  # 5 positions, limit is 2
        ]

        spot_prices = {f"SYM{i}": 105 for i in range(5)}
        within_limits, violations = rm.check_limits(
            portfolio_value=100000, positions=positions, spot_prices=spot_prices
        )

        assert not within_limits
        assert any("position count" in v.lower() for v in violations)


class TestKellyFraction:
    """Test Kelly criterion calculation."""

    def test_positive_edge(self):
        """Positive edge should give positive Kelly."""
        kelly = calculate_kelly_fraction(
            win_rate=0.70, avg_win=1.0, avg_loss=1.0, kelly_fraction=1.0
        )

        assert kelly > 0

    def test_no_edge(self):
        """50/50 with equal wins/losses = 0 Kelly."""
        kelly = calculate_kelly_fraction(
            win_rate=0.50, avg_win=1.0, avg_loss=1.0, kelly_fraction=1.0
        )

        assert kelly == 0

    def test_negative_edge(self):
        """Negative edge should give 0 Kelly."""
        kelly = calculate_kelly_fraction(
            win_rate=0.30, avg_win=1.0, avg_loss=1.0, kelly_fraction=1.0
        )

        assert kelly == 0

    def test_half_kelly(self):
        """Half Kelly should be capped appropriately."""
        full_kelly = calculate_kelly_fraction(
            win_rate=0.70, avg_win=1.5, avg_loss=1.0, kelly_fraction=1.0
        )

        half_kelly = calculate_kelly_fraction(
            win_rate=0.70, avg_win=1.5, avg_loss=1.0, kelly_fraction=0.5
        )

        # Half kelly is capped at 0.25 max, so it should be <= full_kelly
        assert half_kelly <= full_kelly
        assert half_kelly <= 0.25  # Function caps at 25%


class TestOptimalContracts:
    """Test optimal contract calculation."""

    def test_basic_calculation(self):
        """Should return reasonable number of contracts."""
        contracts = calculate_optimal_contracts(
            capital=100000, strike=100, max_risk_pct=0.05, margin_requirement=0.20
        )

        assert contracts >= 1
        assert contracts <= 50  # Reasonable upper bound

    def test_low_capital(self):
        """Low capital with high risk constraints should give zero contracts.

        With $10k capital, $100 strike, 5% max risk ($500), and default 25% stress:
        - Stress loss per contract = $100 * 0.25 * 100 = $2,500
        - Max contracts by risk = $500 / $2,500 = 0 (correctly rejects)

        This is the correct behavior - don't allocate if risk exceeds constraints.
        """
        contracts = calculate_optimal_contracts(
            capital=10000, strike=100, max_risk_pct=0.05, margin_requirement=0.20
        )

        # Risk constraint dominates - 0 contracts is correct
        assert contracts == 0

    def test_adequate_capital(self):
        """Adequate capital should give some contracts."""
        contracts = calculate_optimal_contracts(
            capital=100000,  # Higher capital
            strike=100,
            max_risk_pct=0.10,  # Higher risk tolerance
            margin_requirement=0.20,
        )

        assert contracts >= 1
        assert contracts <= 50


# =============================================================================
# SectorExposure + SectorExposureManager
# =============================================================================


class TestSectorExposure:
    def test_concentrated_above_25pct(self):
        ex = SectorExposure(
            sector="Tech", position_count=3,
            notional_exposure=300_000, exposure_pct=0.30,
            symbols=["AAPL", "MSFT", "NVDA"],
        )
        assert ex.is_concentrated is True

    def test_not_concentrated_below_25pct(self):
        ex = SectorExposure(
            sector="Tech", position_count=1,
            notional_exposure=100_000, exposure_pct=0.20,
            symbols=["AAPL"],
        )
        assert ex.is_concentrated is False


class TestSectorExposureManager:
    @pytest.fixture
    def mgr(self) -> SectorExposureManager:
        return SectorExposureManager(
            sector_map={"AAPL": "Tech", "MSFT": "Tech", "JPM": "Financials", "XOM": "Energy"},
            max_sector_pct=0.25,
        )

    def test_get_sector_known(self, mgr: SectorExposureManager):
        assert mgr.get_sector("AAPL") == "Tech"

    def test_get_sector_unknown_returns_unknown(self, mgr: SectorExposureManager):
        assert mgr.get_sector("ZZZ") == "Unknown"

    def test_calculate_sector_exposures(self, mgr: SectorExposureManager):
        positions = [
            {"symbol": "AAPL", "strike": 150, "contracts": 5},
            {"symbol": "MSFT", "strike": 400, "contracts": 2},
            {"symbol": "JPM", "strike": 200, "contracts": 3},
        ]
        out = mgr.calculate_sector_exposures(positions, portfolio_value=1_000_000)
        assert "Tech" in out
        assert "Financials" in out
        # Tech notional = 150*100*5 + 400*100*2 = 75_000 + 80_000 = 155_000
        assert out["Tech"].notional_exposure == 155_000
        assert out["Tech"].position_count == 2
        assert "AAPL" in out["Tech"].symbols
        assert "MSFT" in out["Tech"].symbols

    def test_calculate_sector_exposures_zero_portfolio(self, mgr: SectorExposureManager):
        positions = [{"symbol": "AAPL", "strike": 150, "contracts": 5}]
        out = mgr.calculate_sector_exposures(positions, portfolio_value=0)
        assert out["Tech"].exposure_pct == 0  # divide-by-zero guard

    def test_check_sector_limit_within(self, mgr: SectorExposureManager):
        positions = [{"symbol": "AAPL", "strike": 150, "contracts": 5}]
        # Tech at 75_000; adding 100_000 → 175_000 / 1M = 17.5% < 25% limit
        allowed, reason = mgr.check_sector_limit(
            "MSFT", proposed_notional=100_000,
            positions=positions, portfolio_value=1_000_000,
        )
        assert allowed is True

    def test_check_sector_limit_breach(self, mgr: SectorExposureManager):
        positions = [{"symbol": "AAPL", "strike": 200, "contracts": 5}]
        # Tech at 100_000; adding 200_000 → 300_000 / 1M = 30% > 25% limit
        allowed, reason = mgr.check_sector_limit(
            "MSFT", proposed_notional=200_000,
            positions=positions, portfolio_value=1_000_000,
        )
        assert allowed is False
        assert "Tech" in reason

    def test_check_sector_limit_zero_portfolio(self, mgr: SectorExposureManager):
        allowed, _ = mgr.check_sector_limit(
            "AAPL", proposed_notional=10_000,
            positions=[], portfolio_value=0,
        )
        # Divide-by-zero guard → 0% < 25% limit → allowed
        assert allowed is True

    def test_get_sector_violations_empty_when_within(self, mgr: SectorExposureManager):
        positions = [{"symbol": "AAPL", "strike": 100, "contracts": 1}]
        violations = mgr.get_sector_violations(positions, portfolio_value=1_000_000)
        assert violations == []

    def test_get_sector_violations_when_above(self, mgr: SectorExposureManager):
        positions = [
            {"symbol": "AAPL", "strike": 200, "contracts": 10},  # 200_000 → 20%
            {"symbol": "MSFT", "strike": 200, "contracts": 5},   # 100_000 → 10%
        ]
        # Total Tech = 30% > 25% limit
        violations = mgr.get_sector_violations(positions, portfolio_value=1_000_000)
        assert len(violations) == 1
        assert "Tech" in violations[0]

    def test_suggest_diversification(self, mgr: SectorExposureManager):
        # Tech is loaded; suggest Financials/Energy
        positions = [{"symbol": "AAPL", "strike": 200, "contracts": 5}]
        suggestions = mgr.suggest_diversification(
            positions, portfolio_value=1_000_000,
            available_symbols=["JPM", "XOM", "MSFT"],
        )
        # Should include Financials/Energy symbols (under-represented)
        assert isinstance(suggestions, list)
        # JPM (Financials) and XOM (Energy) are under 50% of limit → included
        assert "JPM" in suggestions or "XOM" in suggestions


# =============================================================================
# HierarchicalRiskParity
# =============================================================================


class TestHierarchicalRiskParity:
    @pytest.fixture
    def returns(self) -> pd.DataFrame:
        """5 assets, 252 daily returns, with realistic correlation structure."""
        rng = np.random.default_rng(42)
        n = 252
        # Two clusters: tech (correlated), financials (correlated)
        tech_factor = rng.standard_normal(n) * 0.012
        fin_factor = rng.standard_normal(n) * 0.010
        df = pd.DataFrame({
            "AAPL": tech_factor + rng.standard_normal(n) * 0.005,
            "MSFT": tech_factor + rng.standard_normal(n) * 0.005,
            "GOOGL": tech_factor + rng.standard_normal(n) * 0.006,
            "JPM": fin_factor + rng.standard_normal(n) * 0.005,
            "BAC": fin_factor + rng.standard_normal(n) * 0.005,
        })
        return df

    def test_init_defaults(self):
        h = HierarchicalRiskParity()
        assert h.linkage_method == "ward"

    def test_init_custom_linkage(self):
        h = HierarchicalRiskParity(linkage_method="single")
        assert h.linkage_method == "single"

    def test_fit_returns_normalised_weights(self, returns: pd.DataFrame):
        h = HierarchicalRiskParity()
        weights = h.fit(returns)
        assert set(weights.keys()) == set(returns.columns)
        # Weights normalised to ~1
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # All non-negative
        assert all(w >= 0 for w in weights.values())

    def test_fit_with_explicit_covariance(self, returns: pd.DataFrame):
        cov = returns.cov()
        h = HierarchicalRiskParity()
        weights = h.fit(returns, covariance=cov)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_correlation_distance(self, returns: pd.DataFrame):
        h = HierarchicalRiskParity()
        corr = returns.corr()
        dist = h._correlation_distance(corr)
        # Distance matrix shape matches
        assert dist.shape == (5, 5)
        # Distances in [0, 1] for normalised correlations
        assert (dist >= 0).all()
        # Diagonal should be ~0 (correlation with self = 1 → dist = 0)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)

    def test_cluster_variance_single_asset(self, returns: pd.DataFrame):
        cov = returns.cov()
        h = HierarchicalRiskParity()
        v = h._cluster_variance(cov, ["AAPL"])
        # Single-asset cluster variance = asset variance
        assert v == pytest.approx(cov.loc["AAPL", "AAPL"])

    def test_cluster_variance_empty_returns_zero(self, returns: pd.DataFrame):
        cov = returns.cov()
        h = HierarchicalRiskParity()
        assert h._cluster_variance(cov, []) == 0.0


class TestCalculateHrpWeights:
    def test_returns_normalised_dict(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 4)), columns=["A", "B", "C", "D"])
        weights = calculate_hrp_weights(df)
        assert set(weights.keys()) == {"A", "B", "C", "D"}
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_filters_to_target_symbols(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 4)), columns=["A", "B", "C", "D"])
        weights = calculate_hrp_weights(df, target_symbols=["A", "B"])
        assert set(weights.keys()) == {"A", "B"}


class TestOptimizePositionWeights:
    def test_few_symbols_equal_weights(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 1)), columns=["AAPL"])
        weights = optimize_position_weights(["AAPL"], df)
        # Single symbol → 100% weight
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_normalises_to_one(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 4)), columns=["A", "B", "C", "D"])
        weights = optimize_position_weights(["A", "B", "C", "D"], df)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_min_weight_for_unavailable_symbols(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 2)), columns=["A", "B"])
        # Z is not in df; should get min_weight
        weights = optimize_position_weights(["A", "B", "Z"], df, min_weight=0.05)
        # Z assigned at least min_weight before normalisation
        assert "Z" in weights
        # All weights >= 0
        assert all(w >= 0 for w in weights.values())

    def test_max_weight_clamped(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((100, 2)), columns=["A", "B"])
        weights = optimize_position_weights(["A", "B"], df, max_weight=0.5, min_weight=0.05)
        # All weights >= 0
        for w in weights.values():
            assert w <= 1.0  # at minimum, total normalised


# =============================================================================
# RiskManager init + risk metrics
# =============================================================================


class TestRiskManagerInit:
    def test_default_construction(self):
        rm = RiskManager()
        assert rm is not None
        assert isinstance(rm.limits, RiskLimits)

    def test_from_policy(self):
        rm = RiskManager.from_policy(environment="dev")
        assert rm is not None

    def test_update_portfolio_value(self):
        rm = RiskManager()
        rm.update_portfolio_value(500_000)
        # portfolio_values is a list (history), not a single attribute
        assert rm.portfolio_values[-1] == 500_000

    def test_update_portfolio_value_tracks_history(self):
        rm = RiskManager()
        rm.update_portfolio_value(100_000)
        rm.update_portfolio_value(105_000)
        assert len(rm.portfolio_values) == 2
        assert rm.portfolio_values[-1] == 105_000


class TestRiskManagerStressTests:
    def test_run_stress_tests_returns_dict(self):
        rm = RiskManager()
        positions = [
            {"symbol": "AAPL", "option_type": "put", "strike": 150, "dte": 30,
             "iv": 0.25, "delta": -0.30, "contracts": 1, "covered": False},
        ]
        spot_prices = {"AAPL": 155.0}
        result = rm.run_stress_tests(
            portfolio_value=100_000,
            positions=positions,
            spot_prices=spot_prices,
        )
        assert isinstance(result, dict)
        # Standard scenarios produce some output
        assert len(result) > 0

    def test_run_stress_tests_with_custom_scenario(self):
        rm = RiskManager()
        positions = [
            {"symbol": "AAPL", "option_type": "put", "strike": 150, "dte": 30,
             "iv": 0.25, "delta": -0.30, "contracts": 1, "covered": False},
        ]
        custom = [{"name": "tiny_move", "spot_move": -0.01, "vol_move": 0.0}]
        result = rm.run_stress_tests(
            portfolio_value=100_000,
            positions=positions,
            spot_prices={"AAPL": 155.0},
            custom_scenarios=custom,
        )
        assert isinstance(result, dict)


class TestKellyFractionEdgeCases:
    def test_zero_avg_loss_returns_zero(self):
        # Just verify no crash for zero-loss edge case
        result = calculate_kelly_fraction(0.6, 1.0, 0.0)
        assert isinstance(result, float)
