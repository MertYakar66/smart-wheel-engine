"""Tests for stress testing module."""

import pytest
import numpy as np
import pandas as pd

from engine.stress_testing import (
    StressTester,
    Scenario,
    ScenarioResult,
    StressTestReport,
    ScenarioType,
    quick_stress_test,
    calculate_max_loss,
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS
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
            iv_change_abs=0.20
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
                'symbol': 'AAPL',
                'option_type': 'put',
                'strike': 150,
                'dte': 30,
                'iv': 0.25,
                'contracts': 5,
                'is_short': True
            }
        ]
        self.spot_prices = {'AAPL': 155}
        self.portfolio_value = 100000

    def test_run_single_scenario(self):
        """Should run single scenario."""
        tester = StressTester()
        scenario = Scenario(
            name="Test Drop",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="10% drop",
            spot_change_pct=-0.10,
            iv_change_abs=0.15
        )

        result = tester.run_scenario(
            scenario=scenario,
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value
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
            iv_change_abs=0.40
        )

        result = tester.run_scenario(
            scenario=scenario,
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value
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
            iv_change_pct=-0.30
        )

        result = tester.run_scenario(
            scenario=scenario,
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value
        )

        # Short puts profit on rally
        assert result.portfolio_pnl > 0

    def test_run_all_scenarios(self):
        """Should run all scenarios and generate report."""
        tester = StressTester()

        report = tester.run_all_scenarios(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value
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
            n_points=5
        )

        assert isinstance(grid, pd.DataFrame)
        assert 'spot_change' in grid.columns
        assert 'iv_change' in grid.columns
        assert 'pnl' in grid.columns
        assert len(grid) == 25  # 5x5 grid

    def test_monte_carlo_stress(self):
        """Should run Monte Carlo simulation."""
        tester = StressTester()

        results = tester.monte_carlo_stress(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            n_simulations=1000,
            horizon_days=30
        )

        assert 'mean' in results
        assert 'var_95' in results
        assert 'cvar_95' in results
        assert results['var_95'] <= 0  # Should be a loss at 95% VaR


class TestQuickStressTest:
    """Test quick stress test function."""

    def test_quick_stress_test(self):
        """Should return formatted summary."""
        positions = [
            {
                'symbol': 'AAPL',
                'option_type': 'put',
                'strike': 150,
                'dte': 30,
                'iv': 0.25,
                'contracts': 2,
                'is_short': True
            }
        ]
        spot_prices = {'AAPL': 155}

        summary = quick_stress_test(
            positions=positions,
            spot_prices=spot_prices,
            portfolio_value=100000
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
                'symbol': 'AAPL',
                'option_type': 'put',
                'strike': 150,
                'contracts': 5,
                'is_short': True
            }
        ]
        spot_prices = {'AAPL': 155}

        max_loss = calculate_max_loss(positions, spot_prices)

        # Max loss = 150 * 100 * 5 = $75,000
        assert max_loss == 75000

    def test_multiple_positions(self):
        """Multiple positions should sum max losses."""
        positions = [
            {
                'symbol': 'AAPL',
                'option_type': 'put',
                'strike': 150,
                'contracts': 2,
                'is_short': True
            },
            {
                'symbol': 'MSFT',
                'option_type': 'put',
                'strike': 300,
                'contracts': 1,
                'is_short': True
            }
        ]
        spot_prices = {'AAPL': 155, 'MSFT': 310}

        max_loss = calculate_max_loss(positions, spot_prices)

        # AAPL: 150 * 100 * 2 = 30000
        # MSFT: 300 * 100 * 1 = 30000
        # Total: 60000
        assert max_loss == 60000
