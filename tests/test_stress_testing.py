"""Tests for stress testing module."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import engine.stress_testing as st
from engine.option_pricer import black_scholes_all_greeks
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

    def test_iv_shock_is_reflected_in_full_repricing_total_pnl(self):
        """O4: with no spot move and no decay, the only P&L source is the vol
        shock, so the full-repricing total_pnl must materially reflect it. The
        pre-fix code priced the base leg at the SHOCKED iv, cancelling the vol
        shock out of (new_price - base_price) -> total_pnl ~ 0 despite a clearly
        non-zero vega_pnl in the decomposition."""
        tester = StressTester()
        ladder = tester.greeks_stress_ladder(
            positions=self.positions,
            spot_prices=self.spot_prices,
            portfolio_value=self.portfolio_value,
            spot_range=(0.0, 0.0),  # single row, no spot move
            n_steps=1,
            iv_shock=0.30,
            dte_decay=0,
        )
        row = ladder.iloc[0]
        assert abs(row["vega_pnl"]) > 0
        # Full-repricing total_pnl must reflect the vol shock, not collapse to ~0.
        assert abs(row["total_pnl"]) > 0.5 * abs(row["vega_pnl"])
        # Short vega (short put + short call): a +IV shock is a loss.
        assert row["total_pnl"] < 0

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
        for _name, result in results.items():
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


# ----------------------------------------------------------------------
# CMD 7 — stress_testing correctness fixes (risk-REPORTING module; not the
# EV ranker/trio): t-dist variance normalization, per-name dollar-delta spot,
# per-position rate in run_scenario.
# ----------------------------------------------------------------------


def test_monte_carlo_returns_are_variance_normalized(monkeypatch):
    """Bug 1: the simulated returns' std must match the IV-implied target
    (daily_vol * sqrt(horizon)), NOT the ~1.29x an un-normalized Student-t(5)
    produced (t(5) std = sqrt(5/3) ≈ 1.29). We capture the spot_change_pct each
    sim feeds run_scenario."""
    tester = StressTester()
    positions = [
        {
            "symbol": "AAPL",
            "option_type": "put",
            "strike": 150,
            "dte": 30,
            "iv": 0.30,
            "contracts": 1,
            "is_short": True,
        }
    ]
    captured = []

    def _spy(scenario, *a, **k):
        captured.append(scenario.spot_change_pct)
        return SimpleNamespace(portfolio_pnl=0.0)

    monkeypatch.setattr(tester, "run_scenario", _spy)
    tester.monte_carlo_stress(
        positions=positions,
        spot_prices={"AAPL": 155},
        portfolio_value=100000,
        n_simulations=20000,
        horizon_days=1,
    )
    sc = np.array(captured)
    target = 0.30 / np.sqrt(252) * np.sqrt(1)  # daily_vol * sqrt(horizon_days)
    assert sc.std() == pytest.approx(target, rel=0.08)
    assert sc.std() < 1.15 * target  # explicitly rejects the ~1.29x inflation


def test_greeks_matrix_dollar_delta_uses_each_names_spot():
    """Bug 2: aggregate delta_dollars = sum_i(delta_i * mult_i * spot_i), each
    name at its OWN spot — not pooled_delta * the FIRST symbol's spot."""
    tester = StressTester()
    positions = [
        {
            "symbol": "AAPL",
            "option_type": "put",
            "strike": 100,
            "dte": 30,
            "iv": 0.25,
            "contracts": 1,
            "is_short": True,
        },
        {
            "symbol": "BIGCO",
            "option_type": "put",
            "strike": 1000,
            "dte": 30,
            "iv": 0.25,
            "contracts": 1,
            "is_short": True,
        },
    ]
    spot_prices = {"AAPL": 100.0, "BIGCO": 1000.0}
    res = tester.greeks_scenario_matrix(
        positions=positions,
        spot_prices=spot_prices,
        portfolio_value=100000,
        spot_shocks=[0.0],
        iv_shocks=[0.0],
        time_shocks=[0],
    )
    row = res["greeks_surface"].iloc[0]

    expected = 0.0
    for p in positions:
        s = spot_prices[p["symbol"]]
        g = black_scholes_all_greeks(
            S=s,
            K=p["strike"],
            T=p["dte"] / 365,
            r=p.get("rate", 0.05),
            sigma=p["iv"],
            option_type=p["option_type"],
            q=0.0,
        )
        mult = p["contracts"] * 100 * (-1 if p["is_short"] else 1)
        expected += g["delta"] * mult * s
    assert row["delta_dollars"] == pytest.approx(expected)
    # ... and NOT the old pooled_delta * first-symbol spot (materially different
    # here because BIGCO trades at 10x AAPL).
    assert row["delta_dollars"] != pytest.approx(row["delta"] * spot_prices["AAPL"])


def test_run_scenario_honors_per_position_rate(monkeypatch):
    """Bug 3: run_scenario prices with the position's own rate, not the tester's
    risk_free_rate. A pos with rate=0.03 must price both legs at r=0.03."""
    tester = StressTester(risk_free_rate=0.05)
    seen_r = []
    orig = st.black_scholes_price

    def _spy(**kw):
        seen_r.append(round(kw.get("r"), 6))
        return orig(**kw)

    monkeypatch.setattr(st, "black_scholes_price", _spy)
    pos = {
        "symbol": "AAPL",
        "option_type": "put",
        "strike": 150,
        "dte": 30,
        "iv": 0.25,
        "contracts": 1,
        "is_short": True,
        "rate": 0.03,
    }
    scenario = Scenario(
        name="flat",
        scenario_type=ScenarioType.HYPOTHETICAL,
        description="",
        spot_change_pct=0.0,
    )
    tester.run_scenario(scenario, [pos], {"AAPL": 155}, 100000)
    assert 0.03 in seen_r, f"expected the position rate 0.03 to price a leg; saw {seen_r}"
    assert 0.05 not in seen_r, f"must not fall back to risk_free_rate; saw {seen_r}"


class TestGreeksScenarioMatrixThetaUnits:
    """Proposal #8 — greeks_scenario_matrix must emit DAILY theta (annual/365),
    matching GREEKS_UNIT_CONTRACT.md and every other Greeks surface. Value-level
    regression: the emitted theta equals the pricer's annual theta / 365, not the
    raw annual value (which was ~365x too large pre-fix)."""

    def _position(self):
        return {
            "symbol": "AAPL",
            "option_type": "put",
            "strike": 150,
            "dte": 30,
            "iv": 0.25,
            "contracts": 5,
            "is_short": True,
        }

    def test_greeks_surface_theta_is_daily(self):
        from engine.option_pricer import black_scholes_all_greeks

        pos = self._position()
        spot = 155.0
        matrix = StressTester().greeks_scenario_matrix(
            positions=[pos],
            spot_prices={"AAPL": spot},
            portfolio_value=100_000,
            spot_shocks=[0.0],
            iv_shocks=[0.0],
            time_shocks=[0],
        )
        surface = matrix["greeks_surface"]
        emitted = float(surface.loc[surface["spot_change"] == 0.0, "theta"].iloc[0])

        # replicate the loop math at spot_chg == 0 (new_spot == spot)
        multiplier = pos["contracts"] * 100 * -1  # is_short => direction -1
        greeks = black_scholes_all_greeks(
            S=spot,
            K=pos["strike"],
            T=max(0.001, pos["dte"] / 365),
            r=pos.get("rate", 0.05),
            sigma=pos["iv"],
            option_type=pos["option_type"],
            q=pos.get("dividend_yield", 0.0),
        )
        annual_theta = greeks["theta"]
        expected_daily = (annual_theta / 365) * multiplier

        assert emitted == pytest.approx(expected_daily, rel=1e-9), (
            f"greeks_surface theta {emitted} must equal annual/365 * mult {expected_daily}"
        )
        # and NOT the ~365x-larger annual value (proves the /365 conversion is present)
        assert emitted != pytest.approx(annual_theta * multiplier, rel=1e-2), (
            "emitted theta must be daily, not the annual value"
        )

    def test_time_decay_remaining_theta_is_daily(self):
        from engine.option_pricer import black_scholes_all_greeks

        pos = self._position()
        spot = 155.0
        days = 7
        matrix = StressTester().greeks_scenario_matrix(
            positions=[pos],
            spot_prices={"AAPL": spot},
            portfolio_value=100_000,
            spot_shocks=[0.0],
            iv_shocks=[0.0],
            time_shocks=[days],
        )
        decay = matrix["time_decay"]
        emitted = float(decay.loc[decay["days_elapsed"] == days, "remaining_theta"].iloc[0])

        multiplier = pos["contracts"] * 100 * -1  # is_short => direction -1
        dte_new = max(0.001, pos["dte"] - days)
        new_greeks = black_scholes_all_greeks(
            S=spot,
            K=pos["strike"],
            T=dte_new / 365,
            r=pos.get("rate", 0.05),
            sigma=pos["iv"],
            option_type=pos["option_type"],
            q=pos.get("dividend_yield", 0.0),
        )
        annual_theta = new_greeks["theta"]
        expected_daily = (annual_theta / 365) * multiplier

        assert emitted == pytest.approx(expected_daily, rel=1e-9), (
            f"time_decay remaining_theta {emitted} must equal annual/365 * mult {expected_daily}"
        )
        assert emitted != pytest.approx(annual_theta * multiplier, rel=1e-2), (
            "remaining_theta must be daily, not the annual value"
        )
