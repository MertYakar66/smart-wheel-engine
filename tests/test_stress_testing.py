"""Tests for stress testing module."""

import pandas as pd

from engine.stress_testing import (
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
    Scenario,
    ScenarioResult,
    ScenarioType,
    StressTester,
    StressTestReport,
    calculate_max_loss,
    quick_stress_test,
)


class TestScenarios:
    """Test scenario definitions."""

    def test_historical_scenarios_exist(self):
        """Should have predefined historical scenarios."""
        assert len(HISTORICAL_SCENARIOS) > 0
        for scenario in HISTORICAL_SCENARIOS:
            assert scenario.scenario_type == ScenarioType.HISTORICAL

    def test_hypothetical_scenarios_exist(self):
        """Should have predefined hypothetical scenarios."""
        assert len(HYPOTHETICAL_SCENARIOS) > 0
        for scenario in HYPOTHETICAL_SCENARIOS:
            assert scenario.scenario_type == ScenarioType.HYPOTHETICAL

    def test_scenario_str(self):
        """Scenario string representation."""
        scenario = Scenario(
            name="Test",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="Test scenario",
            spot_change_pct=-0.10,
            iv_change_abs=0.20,
        )

        str_repr = str(scenario)
        assert "Test" in str_repr
        assert "Spot" in str_repr


class TestStressTester:
    """Test stress testing engine."""

    def setup_method(self):
        """Set up test positions."""
        self.positions = [
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
        self.spot_prices = {"AAPL": 155}
        self.portfolio_value = 100000

    def test_run_single_scenario(self):
        """Should run single scenario."""
        tester = StressTester()
        scenario = Scenario(
            name="Test Drop",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="10% drop",
            spot_change_pct=-0.10,
            iv_change_abs=0.15,
        )

        result = tester.run_scenario(
            scenario=scenario,
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
        )

        assert isinstance(result, ScenarioResult)
        assert result.portfolio_pnl < 0  # Should lose money on drop
        assert result.scenario == scenario

    def test_crash_scenario_loss(self):
        """Crash scenario should show significant loss."""
        tester = StressTester()
        scenario = Scenario(
            name="Crash",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="20% crash",
            spot_change_pct=-0.20,
            iv_change_abs=0.40,
        )

        result = tester.run_scenario(
            scenario=scenario,
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
        )

        # Short puts lose big on crash
        assert result.portfolio_pnl < -1000
        assert result.portfolio_pnl_pct < -0.01

    def test_rally_scenario_profit(self):
        """Rally scenario should show profit for short puts."""
        tester = StressTester()
        scenario = Scenario(
            name="Rally",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="10% rally",
            spot_change_pct=0.10,
            iv_change_pct=-0.30,
        )

        result = tester.run_scenario(
            scenario=scenario,
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
        )

        # Short puts profit on rally
        assert result.portfolio_pnl > 0

    def test_run_all_scenarios(self):
        """Should run all scenarios and generate report."""
        tester = StressTester()

        report = tester.run_all_scenarios(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
        )

        assert isinstance(report, StressTestReport)
        assert len(report.results) > 0
        assert report.worst_case is not None
        assert report.best_case is not None
        assert report.worst_case.portfolio_pnl <= report.best_case.portfolio_pnl

    def test_sensitivity_analysis(self):
        """Should generate sensitivity grid."""
        tester = StressTester()

        grid = tester.sensitivity_analysis(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            n_points=5,
        )

        assert isinstance(grid, pd.DataFrame)
        assert "spot_change" in grid.columns
        assert "iv_change" in grid.columns
        assert "pnl" in grid.columns
        assert len(grid) == 25  # 5x5 grid

    def test_monte_carlo_stress(self):
        """Should run Monte Carlo simulation."""
        tester = StressTester()

        results = tester.monte_carlo_stress(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            n_simulations=1000,
            horizon_days=30,
        )

        assert "mean" in results
        assert "var_95" in results
        assert "cvar_95" in results
        assert results["var_95"] <= 0  # Should be a loss at 95% VaR


class TestQuickStressTest:
    """Test quick stress test function."""

    def test_quick_stress_test(self):
        """Should return formatted summary."""
        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 2,
                "is_short": True,
            }
        ]
        spot_prices = {"AAPL": 155}

        summary = quick_stress_test(
            positions=positions, spot_prices=spot_prices, portfolio_value=100000
        )

        assert isinstance(summary, str)
        assert "Stress Test" in summary
        assert "Worst Case" in summary


class TestMaxLoss:
    """Test maximum loss calculation."""

    def test_short_put_max_loss(self):
        """Short put max loss = strike * 100 * contracts."""
        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "contracts": 5,
                "is_short": True,
            }
        ]
        spot_prices = {"AAPL": 155}

        max_loss = calculate_max_loss(positions, spot_prices)

        # Max loss = 150 * 100 * 5 = $75,000
        assert max_loss == 75000

    def test_multiple_positions(self):
        """Multiple positions should sum max losses."""
        positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "contracts": 2,
                "is_short": True,
            },
            {
                "symbol": "MSFT",
                "option_type": "put",
                "strike": 300,
                "contracts": 1,
                "is_short": True,
            },
        ]
        spot_prices = {"AAPL": 155, "MSFT": 310}

        max_loss = calculate_max_loss(positions, spot_prices)

        # AAPL: 150 * 100 * 2 = 30000
        # MSFT: 300 * 100 * 1 = 30000
        # Total: 60000
        assert max_loss == 60000


class TestGreeksStressTesting:
    """Test Greeks stress-testing scenarios."""

    def setup_method(self):
        """Set up test positions."""
        self.positions = [
            {
                "symbol": "AAPL",
                "option_type": "put",
                "strike": 150,
                "dte": 30,
                "iv": 0.25,
                "contracts": 5,
                "is_short": True,
            },
            {
                "symbol": "AAPL",
                "option_type": "call",
                "strike": 160,
                "dte": 30,
                "iv": 0.22,
                "contracts": 3,
                "is_short": True,
            },
        ]
        self.spot_prices = {"AAPL": 155}
        self.portfolio_value = 100000

    def test_greeks_stress_ladder(self):
        """Should generate Greeks stress ladder with P&L decomposition."""
        tester = StressTester()

        ladder = tester.greeks_stress_ladder(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            spot_range=(-0.10, 0.10),
            n_steps=11,
        )

        assert isinstance(ladder, pd.DataFrame)
        assert "spot_change" in ladder.columns
        assert "total_pnl" in ladder.columns
        assert "delta_pnl" in ladder.columns
        assert "gamma_pnl" in ladder.columns
        assert "theta_pnl" in ladder.columns
        assert "vega_pnl" in ladder.columns
        assert len(ladder) == 11

        # P&L should be negative for downside moves (short puts)
        down_rows = ladder[ladder["spot_change"] < -0.05]
        assert all(down_rows["total_pnl"] < 0)

    def test_greeks_stress_ladder_with_iv_shock(self):
        """Ladder with IV shock should show vega impact."""
        tester = StressTester()

        ladder = tester.greeks_stress_ladder(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            spot_range=(-0.05, 0.05),
            n_steps=5,
            iv_shock=0.50,  # 50% IV increase
        )

        # Should have non-zero vega P&L
        assert any(ladder["vega_pnl"] != 0)

    def test_greeks_scenario_matrix(self):
        """Should generate comprehensive scenario matrices."""
        tester = StressTester()

        results = tester.greeks_scenario_matrix(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            spot_shocks=[-0.10, 0, 0.10],
            iv_shocks=[-0.20, 0, 0.20],
            time_shocks=[0, 7, 14],
        )

        assert "pnl_surface" in results
        assert "greeks_surface" in results
        assert "time_decay" in results

        # P&L surface should have spot changes as rows
        pnl_surface = results["pnl_surface"]
        assert "spot_change" in pnl_surface.columns
        assert len(pnl_surface) == 3  # 3 spot shocks

        # Greeks surface should show delta, gamma, etc.
        greeks_surface = results["greeks_surface"]
        assert "delta" in greeks_surface.columns
        assert "gamma" in greeks_surface.columns
        assert "theta" in greeks_surface.columns
        assert "vega" in greeks_surface.columns

        # Time decay should show theta decay
        time_decay = results["time_decay"]
        assert "days_elapsed" in time_decay.columns
        assert "cumulative_pnl" in time_decay.columns

    def test_extreme_greeks_scenarios(self):
        """Should run extreme historical scenarios with Greeks attribution."""
        tester = StressTester()

        results = tester.extreme_greeks_scenarios(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
        )

        # Should have all predefined scenarios
        assert "black_monday_1987" in results
        assert "covid_crash_2020" in results
        assert "flash_crash" in results
        assert "vol_crush" in results
        assert "gamma_squeeze" in results
        assert "theta_burn" in results

        # Each scenario should have Greeks attribution
        for name, result in results.items():
            assert "total_pnl" in result
            assert "greek_attribution" in result
            attribution = result["greek_attribution"]
            assert "delta_pnl" in attribution
            assert "gamma_pnl" in attribution
            assert "theta_pnl" in attribution
            assert "vega_pnl" in attribution

        # Black Monday should show large loss for short puts
        black_monday = results["black_monday_1987"]
        assert black_monday["total_pnl"] < -5000  # Significant loss

        # Vol crush should benefit short options (collect premium)
        vol_crush = results["vol_crush"]
        assert vol_crush["greek_attribution"]["vega_pnl"] > 0  # Vega profit

        # Theta burn should show theta decay profit for short options
        theta_burn = results["theta_burn"]
        assert theta_burn["greek_attribution"]["theta_pnl"] > 0
