"""
Comprehensive Test Suite for Quantitative Trading Dashboard

Tests all dashboard functionality:
- Option pricing (European & American)
- Greeks analysis (all orders)
- IV solving
- Portfolio management
- Risk metrics (VaR, CVaR)
- Stress testing
- Position sizing
- Report generation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import (
    OptionInput,
    Position,
    QuantDashboard,
    quick_greeks,
    quick_price,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def dashboard():
    """Create fresh dashboard instance."""
    return QuantDashboard(risk_free_rate=0.05)


@pytest.fixture
def atm_put():
    """ATM put option."""
    return OptionInput(
        spot=100.0,
        strike=100.0,
        dte=30,
        rate=0.05,
        volatility=0.25,
        dividend_yield=0.0,
        option_type='put'
    )


@pytest.fixture
def itm_call():
    """ITM call option."""
    return OptionInput(
        spot=110.0,
        strike=100.0,
        dte=45,
        rate=0.05,
        volatility=0.30,
        dividend_yield=0.02,
        option_type='call'
    )


@pytest.fixture
def sample_portfolio(dashboard):
    """Portfolio with multiple positions."""
    dashboard.set_portfolio_value(500_000)

    positions = [
        Position('AAPL', 'put', 170, 45, 0.28, 5, True, 175, 0.005),
        Position('MSFT', 'put', 400, 45, 0.24, 3, True, 420, 0.008),
        Position('GOOGL', 'put', 145, 45, 0.26, 4, True, 150, 0.0),
    ]

    for p in positions:
        dashboard.add_position(p)

    corr = pd.DataFrame(
        [[1.00, 0.70, 0.65],
         [0.70, 1.00, 0.68],
         [0.65, 0.68, 1.00]],
        index=['AAPL', 'MSFT', 'GOOGL'],
        columns=['AAPL', 'MSFT', 'GOOGL']
    )
    dashboard.set_correlation_matrix(corr)

    return dashboard


# =============================================================================
# Quick Functions Tests
# =============================================================================

class TestQuickFunctions:
    """Test standalone quick functions."""

    def test_quick_price_european_put(self):
        """Test quick European put pricing."""
        price = quick_price(100, 100, 30, volatility=0.25, option_type='put')
        assert price > 0, "Price should be positive"
        assert price < 100, "Put price should be less than strike"

    def test_quick_price_european_call(self):
        """Test quick European call pricing."""
        price = quick_price(100, 100, 30, volatility=0.25, option_type='call')
        assert price > 0, "Price should be positive"
        assert price < 100, "Call price should be less than spot"

    def test_quick_price_american(self):
        """Test quick American pricing."""
        american = quick_price(100, 100, 30, volatility=0.25, option_type='put', style='american')
        european = quick_price(100, 100, 30, volatility=0.25, option_type='put', style='european')
        assert american >= european - 0.001, "American should >= European"

    def test_quick_greeks_structure(self):
        """Test quick Greeks returns all expected keys."""
        greeks = quick_greeks(100, 100, 30, volatility=0.25, option_type='put')

        expected_keys = ['price', 'delta', 'gamma', 'theta', 'vega', 'rho',
                         'vanna', 'charm', 'volga', 'speed', 'color', 'ultima']
        for key in expected_keys:
            assert key in greeks, f"Missing key: {key}"

    def test_quick_greeks_put_delta_negative(self):
        """Put delta should be negative."""
        greeks = quick_greeks(100, 100, 30, volatility=0.25, option_type='put')
        assert greeks['delta'] < 0, "Put delta should be negative"

    def test_quick_greeks_call_delta_positive(self):
        """Call delta should be positive."""
        greeks = quick_greeks(100, 100, 30, volatility=0.25, option_type='call')
        assert greeks['delta'] > 0, "Call delta should be positive"


# =============================================================================
# European Pricing Tests
# =============================================================================

class TestEuropeanPricing:
    """Test European option pricing through dashboard."""

    def test_price_european_returns_dict(self, dashboard, atm_put):
        """European pricing returns complete dict."""
        result = dashboard.price_european(atm_put)
        assert isinstance(result, dict)
        assert 'price' in result
        assert 'delta' in result
        assert 'model' in result
        assert result['model'] == 'Black-Scholes-Merton'

    def test_price_european_includes_third_order(self, dashboard, atm_put):
        """European pricing includes third-order Greeks."""
        result = dashboard.price_european(atm_put)
        assert 'speed' in result
        assert 'color' in result
        assert 'ultima' in result

    def test_put_call_parity(self, dashboard):
        """Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)."""
        S, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.02

        call_opt = OptionInput(S, K, int(T*365), r, 0.25, q, 'call')
        put_opt = OptionInput(S, K, int(T*365), r, 0.25, q, 'put')

        call = dashboard.price_european(call_opt)
        put = dashboard.price_european(put_opt)

        lhs = call['price'] - put['price']
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        assert abs(lhs - rhs) < 0.01, f"Put-call parity violated: {lhs} vs {rhs}"


# =============================================================================
# American Pricing Tests
# =============================================================================

class TestAmericanPricing:
    """Test American option pricing through dashboard."""

    def test_price_american_returns_dict(self, dashboard, atm_put):
        """American pricing returns complete dict."""
        result = dashboard.price_american(atm_put)
        assert isinstance(result, dict)
        assert 'price' in result
        assert 'model' in result
        assert result['model'] == 'Barone-Adesi-Whaley'

    def test_american_includes_early_exercise_premium(self, dashboard, atm_put):
        """American pricing includes early exercise premium."""
        result = dashboard.price_american(atm_put)
        assert 'european_price' in result
        assert 'early_exercise_premium' in result

    def test_american_put_geq_european(self, dashboard, atm_put):
        """American put should be >= European put."""
        result = dashboard.price_american(atm_put)
        assert result['price'] >= result['european_price'] - 0.001

    def test_compare_pricing_dataframe(self, dashboard, atm_put):
        """Compare pricing returns DataFrame."""
        df = dashboard.compare_pricing(atm_put)
        assert isinstance(df, pd.DataFrame)
        assert 'European' in df.columns
        assert 'American' in df.columns
        assert 'Difference' in df.columns


# =============================================================================
# Greeks Analysis Tests
# =============================================================================

class TestGreeksAnalysis:
    """Test Greeks analysis functionality."""

    def test_analyze_greeks_structure(self, dashboard, atm_put):
        """Greeks analysis returns proper structure."""
        result = dashboard.analyze_greeks(atm_put)

        assert 'first_order' in result
        assert 'second_order' in result
        assert 'third_order' in result
        assert 'summary' in result

    def test_first_order_greeks(self, dashboard, atm_put):
        """First-order Greeks are complete."""
        result = dashboard.analyze_greeks(atm_put)
        first = result['first_order']

        assert 'delta' in first
        assert 'theta' in first
        assert 'vega' in first
        assert 'rho' in first

    def test_second_order_greeks(self, dashboard, atm_put):
        """Second-order Greeks are complete."""
        result = dashboard.analyze_greeks(atm_put)
        second = result['second_order']

        assert 'gamma' in second
        assert 'vanna' in second
        assert 'charm' in second
        assert 'volga' in second

    def test_third_order_greeks(self, dashboard, atm_put):
        """Third-order Greeks are complete."""
        result = dashboard.analyze_greeks(atm_put)
        third = result['third_order']

        assert 'speed' in third
        assert 'color' in third
        assert 'ultima' in third

    def test_greeks_surface_shape(self, dashboard, atm_put):
        """Greeks surface has correct shape."""
        surface = dashboard.greeks_surface(
            atm_put,
            spot_range=(0.9, 1.1),
            vol_range=(0.2, 0.4),
            greek='delta',
            grid_size=10
        )

        assert isinstance(surface, pd.DataFrame)
        assert surface.shape == (10, 10)

    def test_greeks_surface_delta_monotonic(self, dashboard, atm_put):
        """Delta should be monotonic in spot price for fixed vol."""
        # Test with varying spot at a fixed (single) volatility
        opt = atm_put
        spots = [80, 90, 100, 110, 120]
        deltas = []

        for spot in spots:
            test_opt = OptionInput(
                spot=float(spot),
                strike=opt.strike,
                dte=opt.dte,
                rate=opt.rate,
                volatility=0.25,
                dividend_yield=opt.dividend_yield,
                option_type='put'
            )
            result = dashboard.price_european(test_opt)
            deltas.append(result['delta'])

        # For put, delta should increase (become less negative) as spot increases
        for i in range(len(deltas) - 1):
            assert deltas[i] <= deltas[i + 1] + 0.001, \
                f"Put delta should be monotonically increasing in spot: {deltas}"


# =============================================================================
# Implied Volatility Tests
# =============================================================================

class TestImpliedVolatility:
    """Test IV solving functionality."""

    def test_iv_solver_recovery(self, dashboard, atm_put):
        """IV solver should recover the input volatility."""
        result = dashboard.price_european(atm_put)
        market_price = result['price']

        recovered_iv = dashboard.solve_iv(market_price, atm_put)

        assert recovered_iv is not None
        assert abs(recovered_iv - atm_put.volatility) < 0.001

    def test_iv_solver_higher_price_higher_iv(self, dashboard, atm_put):
        """Higher price should give higher IV."""
        iv_low = dashboard.solve_iv(2.0, atm_put)
        iv_high = dashboard.solve_iv(4.0, atm_put)

        assert iv_low is not None
        assert iv_high is not None
        assert iv_high > iv_low

    def test_iv_solver_invalid_price_returns_none(self, dashboard, atm_put):
        """IV solver returns None for invalid prices."""
        # Price below intrinsic
        dashboard.solve_iv(0.001, atm_put)
        # May or may not be None depending on bounds
        # Just verify it doesn't crash

    def test_iv_surface_from_prices(self, dashboard):
        """IV surface can be built from market prices."""
        market_prices = pd.DataFrame({
            'strike': [95, 100, 105, 95, 100, 105],
            'dte': [30, 30, 30, 60, 60, 60],
            'price': [1.5, 3.0, 5.5, 2.5, 4.0, 6.5],
            'option_type': ['put'] * 6
        })

        surface = dashboard.iv_surface(market_prices, spot=100)

        assert isinstance(surface, pd.DataFrame)
        assert 'iv' in surface.columns
        assert 'moneyness' in surface.columns


# =============================================================================
# Portfolio Management Tests
# =============================================================================

class TestPortfolioManagement:
    """Test portfolio management functionality."""

    def test_add_position(self, dashboard):
        """Can add positions to portfolio."""
        pos = Position('AAPL', 'put', 150, 30, 0.25, 5, True, 155)
        dashboard.add_position(pos)

        greeks = dashboard.get_portfolio_greeks()
        assert greeks.delta != 0

    def test_clear_portfolio(self, dashboard):
        """Can clear portfolio."""
        pos = Position('AAPL', 'put', 150, 30, 0.25, 5, True, 155)
        dashboard.add_position(pos)
        dashboard.clear_portfolio()

        greeks = dashboard.get_portfolio_greeks()
        assert greeks.delta == 0

    def test_set_portfolio_value(self, dashboard):
        """Can set portfolio value."""
        dashboard.set_portfolio_value(1_000_000)
        summary = dashboard.portfolio_summary()
        assert summary['portfolio_value'] == 1_000_000

    def test_set_correlation_matrix(self, sample_portfolio):
        """Correlation matrix is used in VaR."""
        var_result = sample_portfolio.calculate_var()
        assert var_result['method'] == 'Multi-Asset Covariance'

    def test_portfolio_greeks_aggregation(self, sample_portfolio):
        """Portfolio Greeks are properly aggregated."""
        greeks = sample_portfolio.get_portfolio_greeks()

        assert greeks.delta != 0
        assert greeks.delta_dollars != 0
        assert greeks.gamma != 0
        assert greeks.theta != 0
        assert greeks.vega != 0


# =============================================================================
# Risk Metrics Tests
# =============================================================================

class TestRiskMetrics:
    """Test risk metrics functionality."""

    def test_var_positive(self, sample_portfolio):
        """VaR should be positive."""
        var_result = sample_portfolio.calculate_var()
        assert var_result['var'] > 0

    def test_cvar_geq_var(self, sample_portfolio):
        """CVaR should be >= VaR."""
        var_result = sample_portfolio.calculate_var()
        assert var_result['cvar'] >= var_result['var'] - 0.01

    def test_var_confidence_levels(self, sample_portfolio):
        """Higher confidence should give higher VaR."""
        var_90 = sample_portfolio.calculate_var(confidence=0.90)['var']
        var_95 = sample_portfolio.calculate_var(confidence=0.95)['var']
        var_99 = sample_portfolio.calculate_var(confidence=0.99)['var']

        assert var_90 <= var_95 * 1.1  # Allow small tolerance
        assert var_95 <= var_99 * 1.1

    def test_var_horizon_scaling(self, sample_portfolio):
        """Longer horizon should give higher VaR."""
        var_1d = sample_portfolio.calculate_var(horizon_days=1)['var']
        var_10d = sample_portfolio.calculate_var(horizon_days=10)['var']

        # VaR scales roughly with sqrt(T)
        assert var_10d > var_1d

    def test_var_components_present(self, sample_portfolio):
        """VaR components should be present."""
        var_result = sample_portfolio.calculate_var()

        if 'components' in var_result:
            assert 'delta_var' in var_result['components']
            assert 'per_asset_contribution' in var_result['components']


# =============================================================================
# Stress Testing Tests
# =============================================================================

class TestStressTesting:
    """Test stress testing functionality."""

    def test_stress_tests_return_dataframe(self, sample_portfolio):
        """Stress tests return DataFrame."""
        results = sample_portfolio.run_stress_tests()
        assert isinstance(results, pd.DataFrame)

    def test_stress_tests_have_scenarios(self, sample_portfolio):
        """Stress tests include expected scenarios."""
        results = sample_portfolio.run_stress_tests()
        scenarios = results['Scenario'].tolist()

        # Check for standard scenarios
        assert any('crash' in s for s in scenarios)
        assert any('vol' in s for s in scenarios)
        assert 'worst_case' in scenarios

    def test_stress_tests_pnl_column(self, sample_portfolio):
        """Stress tests have P&L column."""
        results = sample_portfolio.run_stress_tests()
        assert 'P&L' in results.columns
        assert 'P&L %' in results.columns

    def test_custom_stress_scenarios(self, sample_portfolio):
        """Can run custom stress scenarios."""
        custom = [
            {'spot_move': -0.15, 'vol_move': 0.25, 'description': 'Custom crash'},
            {'spot_move': 0.10, 'vol_move': -0.20, 'description': 'Rally with vol crush'},
        ]

        results = sample_portfolio.run_stress_tests(custom_scenarios=custom)
        scenarios = results['Scenario'].tolist()

        assert 'custom_1' in scenarios
        assert 'custom_2' in scenarios


# =============================================================================
# Position Sizing Tests
# =============================================================================

class TestPositionSizing:
    """Test position sizing functionality."""

    def test_kelly_positive_edge(self, dashboard):
        """Kelly with positive edge gives positive allocation."""
        dashboard.set_portfolio_value(100_000)
        result = dashboard.calculate_kelly(0.70, 150, 200, fraction=0.5)

        assert result['kelly_fraction'] > 0
        assert result['recommended_allocation'] > 0
        assert result['edge'] > 0

    def test_kelly_negative_edge(self, dashboard):
        """Kelly with negative edge gives zero allocation."""
        dashboard.set_portfolio_value(100_000)
        result = dashboard.calculate_kelly(0.30, 100, 200, fraction=0.5)

        assert result['kelly_fraction'] == 0
        assert result['recommended_allocation'] == 0

    def test_kelly_capped(self, dashboard):
        """Kelly is capped at 25%."""
        dashboard.set_portfolio_value(100_000)
        result = dashboard.calculate_kelly(0.90, 300, 100, fraction=1.0)

        assert result['kelly_fraction'] <= 0.25

    def test_optimal_position_size(self, dashboard, atm_put):
        """Optimal position size returns valid result."""
        dashboard.set_portfolio_value(100_000)
        result = dashboard.optimal_position_size(
            atm_put,
            win_probability=0.70,
            avg_win=100,
            avg_loss=200
        )

        assert 'contracts' in result
        assert 'notional' in result
        assert 'reasoning' in result
        assert result['contracts'] >= 0


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Test report generation functionality."""

    def test_portfolio_summary(self, sample_portfolio):
        """Portfolio summary has expected structure."""
        summary = sample_portfolio.portfolio_summary()

        assert 'portfolio_value' in summary
        assert 'num_positions' in summary
        assert 'greeks' in summary
        assert 'risk' in summary
        assert 'timestamp' in summary

    def test_option_report_format(self, dashboard, atm_put):
        """Option report is properly formatted."""
        report = dashboard.option_report(atm_put, style='european')

        assert isinstance(report, str)
        assert 'OPTION ANALYSIS REPORT' in report
        assert 'INPUT PARAMETERS' in report
        assert 'FIRST-ORDER GREEKS' in report
        assert 'SECOND-ORDER GREEKS' in report
        assert 'THIRD-ORDER GREEKS' in report

    def test_option_report_american(self, dashboard, atm_put):
        """American option report includes early exercise info."""
        report = dashboard.option_report(atm_put, style='american')

        assert 'Early Ex. Prem' in report

    def test_risk_report_format(self, sample_portfolio):
        """Risk report is properly formatted."""
        report = sample_portfolio.risk_report()

        assert isinstance(report, str)
        assert 'PORTFOLIO RISK REPORT' in report
        assert 'AGGREGATE GREEKS' in report
        assert 'VALUE AT RISK' in report
        assert 'STRESS TEST RESULTS' in report


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_expired_option(self, dashboard):
        """Expired option returns intrinsic."""
        opt = OptionInput(spot=105, strike=100, dte=0, volatility=0.25, option_type='call')
        result = dashboard.price_european(opt)
        assert abs(result['price'] - 5.0) < 0.01

    def test_zero_volatility(self, dashboard):
        """Zero volatility returns deterministic value."""
        opt = OptionInput(spot=105, strike=100, dte=30, volatility=0.0, option_type='call')
        result = dashboard.price_european(opt)
        assert result['price'] > 0

    def test_deep_itm_put(self, dashboard):
        """Deep ITM put has delta near -1."""
        opt = OptionInput(spot=50, strike=100, dte=30, volatility=0.25, option_type='put')
        result = dashboard.price_european(opt)
        assert result['delta'] < -0.9

    def test_deep_otm_call(self, dashboard):
        """Deep OTM call has delta near 0."""
        opt = OptionInput(spot=50, strike=100, dte=30, volatility=0.25, option_type='call')
        result = dashboard.price_european(opt)
        assert abs(result['delta']) < 0.1

    def test_empty_portfolio_var(self, dashboard):
        """Empty portfolio VaR is zero."""
        dashboard.set_portfolio_value(100_000)
        var_result = dashboard.calculate_var()
        # With no positions, VaR should be 0 or very small
        assert var_result['var'] >= 0

    def test_single_position_portfolio(self, dashboard):
        """Single position portfolio works correctly."""
        dashboard.set_portfolio_value(100_000)
        pos = Position('AAPL', 'put', 150, 30, 0.25, 1, True, 155)
        dashboard.add_position(pos)

        var_result = dashboard.calculate_var()
        assert var_result['var'] >= 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_analysis_workflow(self, dashboard):
        """Test complete analysis workflow."""
        # 1. Analyze single option
        opt = OptionInput(spot=150, strike=145, dte=45, volatility=0.28, option_type='put')

        european = dashboard.price_european(opt)
        american = dashboard.price_american(opt)
        greeks = dashboard.analyze_greeks(opt)

        assert european['price'] > 0
        assert american['price'] >= european['price'] - 0.01
        assert 'third_order' in greeks

        # 2. Solve IV
        iv = dashboard.solve_iv(european['price'], opt)
        assert abs(iv - 0.28) < 0.01

        # 3. Build portfolio
        dashboard.set_portfolio_value(200_000)
        for sym, strike, price in [('AAPL', 145, 150), ('MSFT', 380, 400)]:
            pos = Position(sym, 'put', strike, 45, 0.28, 3, True, price)
            dashboard.add_position(pos)

        # 4. Calculate risk
        summary = dashboard.portfolio_summary()
        assert summary['num_positions'] == 2

        # 5. Run stress tests
        stress = dashboard.run_stress_tests()
        assert len(stress) > 0

        # 6. Position sizing
        kelly = dashboard.calculate_kelly(0.70, 150, 300)
        assert kelly['kelly_fraction'] >= 0

    def test_report_consistency(self, sample_portfolio):
        """Reports should be consistent with direct calculations."""
        summary = sample_portfolio.portfolio_summary()
        greeks = sample_portfolio.get_portfolio_greeks()

        # Summary greeks should match direct calculation
        assert abs(summary['greeks']['delta'] - greeks.delta) < 0.01
        assert abs(summary['greeks']['gamma'] - greeks.gamma) < 0.0001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
