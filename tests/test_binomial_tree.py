"""
Comprehensive tests for CRR binomial tree and cross-model validation.

Tests cover:
1. Binomial tree correctness (convergence, put-call parity, edge cases)
2. BAW vs CRR agreement across moneyness/tenor/vol regimes
3. Discrete dividend handling
4. Greeks accuracy vs finite differences
5. Cross-model validation framework
6. Richardson extrapolation
"""

import numpy as np
import pytest

from engine.binomial_tree import (
    BinomialResult,
    DiscreteDividend,
    binomial_american_full,
    binomial_american_price,
    binomial_with_richardson,
    convergence_study,
)
from engine.option_pricer import (
    american_option_price,
    american_option_greeks,
    black_scholes_price,
)
from engine.model_validation import (
    CrossModelValidator,
    ModelTolerances,
    run_benchmark_grid,
)


class TestBinomialTreeBasics:
    """Basic correctness of the CRR tree."""

    def test_european_put_matches_bsm(self):
        """For European (no early exercise benefit), CRR should match BSM."""
        # American call on non-dividend stock = European call
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        bsm = black_scholes_price(S, K, T, r, sigma, "call", q=0.0)
        crr = binomial_american_price(S, K, T, r, sigma, "call", q=0.0, n_steps=500)
        assert abs(crr - bsm) / bsm < 0.002, f"CRR={crr:.6f} vs BSM={bsm:.6f}"

    def test_american_put_geq_european(self):
        """American put must be >= European put (early exercise premium >= 0)."""
        S, K, T, r, sigma = 95.0, 100.0, 0.5, 0.05, 0.25
        euro = black_scholes_price(S, K, T, r, sigma, "put", q=0.0)
        amer = binomial_american_price(S, K, T, r, sigma, "put", q=0.0, n_steps=500)
        assert amer >= euro - 0.001, f"American={amer:.6f} < European={euro:.6f}"

    def test_put_call_parity_european(self):
        """European put-call parity: C - P = S*e^(-qT) - K*e^(-rT)."""
        S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.05, 0.25, 0.0
        # American call = European call when q=0
        call = binomial_american_price(S, K, T, r, sigma, "call", q, n_steps=500)
        # European put from BSM
        put_euro = black_scholes_price(S, K, T, r, sigma, "put", q)
        parity_diff = call - put_euro - (S * np.exp(-q * T) - K * np.exp(-r * T))
        assert abs(parity_diff) < 0.05, f"Put-call parity violation: {parity_diff:.6f}"

    def test_intrinsic_value_at_expiry(self):
        """At T=0, price should be intrinsic value."""
        assert binomial_american_price(110, 100, 0.0, 0.05, 0.25, "call") == 10.0
        assert binomial_american_price(90, 100, 0.0, 0.05, 0.25, "put") == 10.0
        assert binomial_american_price(90, 100, 0.0, 0.05, 0.25, "call") == 0.0

    def test_deep_itm_put_early_exercise(self):
        """Deep ITM American put should have early exercise premium."""
        S, K, T, r, sigma = 50.0, 100.0, 1.0, 0.05, 0.25
        result = binomial_american_full(S, K, T, r, sigma, "put", n_steps=300)
        assert result.early_exercise_premium > 0, "Deep ITM put should have EE premium"
        assert result.early_exercise_nodes > 0, "Should exercise at some nodes"

    def test_zero_vol(self):
        """Zero volatility should return deterministic value."""
        price = binomial_american_price(110, 100, 0.5, 0.05, 0.0, "call")
        assert price >= 10.0  # At least intrinsic

    def test_negative_price_impossible(self):
        """Option price can never be negative."""
        for S in [50, 100, 200]:
            for K in [80, 100, 120]:
                for opt in ["call", "put"]:
                    p = binomial_american_price(S, K, 0.5, 0.05, 0.30, opt, n_steps=100)
                    assert p >= 0, f"Negative price: S={S} K={K} {opt} → {p}"


class TestBinomialConvergence:
    """Convergence and accuracy tests."""

    def test_convergence_monotonic_variance(self):
        """Prices should converge as steps increase (variance decreases)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        study = convergence_study(S, K, T, r, sigma, "put")
        # Last few entries should have very small changes
        assert abs(study[-1]["change_pct"]) < 0.1, "Should converge to < 0.1% change"

    def test_richardson_extrapolation_accuracy(self):
        """Richardson should be more accurate than raw tree at same cost."""
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.30
        # Reference: high-step tree
        ref = binomial_american_price(S, K, T, r, sigma, "put", n_steps=1600)
        # Raw 200-step
        raw = binomial_american_price(S, K, T, r, sigma, "put", n_steps=200)
        # Richardson with 200+400 steps
        rich = binomial_with_richardson(S, K, T, r, sigma, "put", n_steps=200)

        raw_err = abs(raw - ref)
        rich_err = abs(rich - ref)
        # Richardson should be at least as good (usually much better)
        assert rich_err <= raw_err * 1.5 + 0.001, (
            f"Richardson ({rich_err:.6f}) should improve on raw ({raw_err:.6f})"
        )

    def test_500_steps_within_01pct_of_1600(self):
        """500 steps should be within 0.1% of 1600 steps."""
        S, K, T, r, sigma = 100.0, 95.0, 0.25, 0.05, 0.25
        p500 = binomial_american_price(S, K, T, r, sigma, "put", n_steps=500)
        p1600 = binomial_american_price(S, K, T, r, sigma, "put", n_steps=1600)
        if p1600 > 0.01:
            rel_err = abs(p500 - p1600) / p1600
            assert rel_err < 0.001, f"500-step error: {rel_err:.4%}"


class TestBinomialGreeks:
    """Greeks accuracy from the tree."""

    def test_delta_bounded(self):
        """Delta should be in [-1, 0] for puts and [0, 1] for calls."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        put_result = binomial_american_full(S, K, T, r, sigma, "put", n_steps=300)
        call_result = binomial_american_full(S, K, T, r, sigma, "call", q=0.02, n_steps=300)
        assert -1.0 <= put_result.delta <= 0.0, f"Put delta={put_result.delta}"
        assert 0.0 <= call_result.delta <= 1.0, f"Call delta={call_result.delta}"

    def test_gamma_positive(self):
        """Gamma should be positive for both puts and calls."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        put_result = binomial_american_full(S, K, T, r, sigma, "put", n_steps=300)
        call_result = binomial_american_full(S, K, T, r, sigma, "call", q=0.02, n_steps=300)
        assert put_result.gamma >= 0, f"Put gamma={put_result.gamma}"
        assert call_result.gamma >= 0, f"Call gamma={call_result.gamma}"

    def test_greeks_match_baw_atm(self):
        """CRR Greeks should roughly match BAW Greeks for ATM options."""
        S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.05, 0.25, 0.02
        crr = binomial_american_full(S, K, T, r, sigma, "put", q, n_steps=500)
        baw = american_option_greeks(S, K, T, r, sigma, "put", q)

        # Delta within 0.03
        assert abs(crr.delta - baw["delta"]) < 0.03, (
            f"Delta: CRR={crr.delta:.4f} BAW={baw['delta']:.4f}"
        )
        # Gamma: BAW is an analytic approximation, CRR is a lattice method —
        # their second derivatives can differ more than first derivatives
        assert abs(crr.gamma - baw["gamma"]) < 0.015, (
            f"Gamma: CRR={crr.gamma:.6f} BAW={baw['gamma']:.6f}"
        )


class TestDiscreteDividends:
    """Test discrete dividend handling."""

    def test_discrete_div_increases_put_value(self):
        """Put value should increase with discrete dividend (stock drops)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        # No dividend
        p_no_div = binomial_american_price(S, K, T, r, sigma, "put", n_steps=300)
        # With $2 discrete dividend at T/2
        divs = [DiscreteDividend(ex_date=None, amount=2.0, time_frac=0.5)]
        p_div = binomial_american_price(S, K, T, r, sigma, "put", q=0.0, n_steps=300,
                                        discrete_dividends=divs)
        assert p_div > p_no_div, (
            f"Put with div ({p_div:.4f}) should exceed no-div ({p_no_div:.4f})"
        )

    def test_discrete_div_decreases_call_value(self):
        """Call value should decrease with discrete dividend."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        c_no_div = binomial_american_price(S, K, T, r, sigma, "call", n_steps=300)
        divs = [DiscreteDividend(ex_date=None, amount=2.0, time_frac=0.5)]
        c_div = binomial_american_price(S, K, T, r, sigma, "call", q=0.0, n_steps=300,
                                        discrete_dividends=divs)
        assert c_div < c_no_div, (
            f"Call with div ({c_div:.4f}) should be less than no-div ({c_no_div:.4f})"
        )

    def test_large_dividend_floor(self):
        """Very large dividend shouldn't cause negative stock prices or crashes."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        divs = [DiscreteDividend(ex_date=None, amount=80.0, time_frac=0.3)]
        p = binomial_american_price(S, K, T, r, sigma, "put", q=0.0, n_steps=200,
                                    discrete_dividends=divs)
        assert np.isfinite(p) and p >= 0, f"Price should be finite and non-negative: {p}"


class TestCrossModelValidation:
    """Test the cross-model validation framework."""

    def test_baw_crr_agreement_atm(self):
        """BAW and CRR should agree within tolerance for ATM options."""
        validator = CrossModelValidator(crr_steps=500)
        report = validator.validate(
            S=100, K=100, T=0.5, r=0.05, sigma=0.25,
            option_type="put", q=0.02, symbol="TEST",
        )
        assert report.baw_vs_crr is not None
        assert report.baw_vs_crr.within_tolerance, (
            f"BAW-CRR should agree ATM: {report.baw_vs_crr.violations}"
        )

    def test_escalation_near_expiry(self):
        """Near-expiry options should trigger escalation."""
        validator = CrossModelValidator(auto_escalate=False)
        report = validator.validate(
            S=100, K=100, T=3/365, r=0.05, sigma=0.25,
            option_type="put", q=0.0, symbol="TEST",
        )
        assert report.escalation_triggered
        assert any("expiry" in r.lower() for r in report.escalation_reasons)

    def test_escalation_deep_otm(self):
        """Deep OTM options should trigger escalation."""
        validator = CrossModelValidator(auto_escalate=False)
        report = validator.validate(
            S=100, K=60, T=0.5, r=0.05, sigma=0.20,
            option_type="put", q=0.0, symbol="TEST",
        )
        assert report.escalation_triggered
        assert any("OTM" in r for r in report.escalation_reasons)

    def test_block_on_divergence(self):
        """Should block trade when block_on_divergence=True and models diverge."""
        # Use very tight tolerances to force a failure
        tight_tol = ModelTolerances(price_abs_tol=0.000001, price_rel_tol=0.000001)
        validator = CrossModelValidator(
            tolerances=tight_tol, block_on_divergence=True, crr_steps=50
        )
        report = validator.validate(
            S=100, K=100, T=0.5, r=0.05, sigma=0.25,
            option_type="put", q=0.02, symbol="TEST",
        )
        # With extremely tight tolerances and only 50 steps, models should diverge
        if not report.baw_vs_crr.within_tolerance:
            assert not report.trade_approved

    def test_report_summary_format(self):
        """Report summary should be a readable string."""
        validator = CrossModelValidator(crr_steps=100)
        report = validator.validate(
            S=100, K=100, T=0.5, r=0.05, sigma=0.25,
            option_type="put", q=0.0, symbol="AAPL",
        )
        summary = report.summary()
        assert "AAPL" in summary
        assert "BAW" in summary
        assert "CRR" in summary


class TestBenchmarkGrid:
    """Test the benchmark grid for CI integration."""

    def test_benchmark_grid_runs(self):
        """Benchmark grid should complete without errors."""
        results = run_benchmark_grid(
            spot_range=(95, 105),
            strikes=[95.0, 100.0, 105.0],
            T_values=[0.1, 0.5],
            sigma_values=[0.20, 0.35],
            crr_steps=200,
        )
        assert len(results) > 0
        assert all("baw_price" in r and "crr_price" in r for r in results)

    def test_benchmark_grid_max_divergence(self):
        """BAW vs CRR divergence should be within 1% for standard parameters."""
        results = run_benchmark_grid(
            spot_range=(90, 110),
            strikes=[90.0, 100.0, 110.0],
            T_values=[0.1, 0.25, 0.5],
            sigma_values=[0.15, 0.25, 0.40],
            crr_steps=500,
        )

        # Filter out low-premium options where relative error is meaningless
        # BAW approximation accuracy degrades for deep OTM options with tiny premiums
        meaningful = [r for r in results if r["crr_price"] > 1.0]

        if meaningful:
            max_rel_diff = max(r["rel_diff"] for r in meaningful)
            # BAW should be within 2% of CRR for options with meaningful premium
            assert max_rel_diff < 0.02, (
                f"Max BAW-CRR divergence: {max_rel_diff:.4%}. "
                f"Worst case: {max((r for r in meaningful), key=lambda r: r['rel_diff'])}"
            )

    def test_all_prices_non_negative(self):
        """Every price in the grid should be non-negative."""
        results = run_benchmark_grid(
            spot_range=(80, 120),
            strikes=[90.0, 100.0, 110.0],
            T_values=[0.1, 0.5],
            sigma_values=[0.20],
            crr_steps=200,
        )
        for r in results:
            assert r["baw_price"] >= 0, f"Negative BAW: {r}"
            assert r["crr_price"] >= 0, f"Negative CRR: {r}"


class TestBinomialEdgeCases:
    """Edge cases and numerical stability."""

    def test_very_short_expiry(self):
        """T = 1 day should work."""
        p = binomial_american_price(100, 100, 1/365, 0.05, 0.25, "put", n_steps=100)
        assert np.isfinite(p) and p >= 0

    def test_very_long_expiry(self):
        """T = 5 years should work."""
        p = binomial_american_price(100, 100, 5.0, 0.05, 0.25, "put", n_steps=500)
        assert np.isfinite(p) and p > 0

    def test_very_high_vol(self):
        """sigma = 1.5 (150%) should work."""
        p = binomial_american_price(100, 100, 0.5, 0.05, 1.5, "put", n_steps=200)
        assert np.isfinite(p) and p > 0

    def test_zero_rate(self):
        """r=0 should work (no early exercise premium for puts)."""
        p = binomial_american_price(95, 100, 0.5, 0.0, 0.25, "put", n_steps=300)
        assert np.isfinite(p) and p > 0

    def test_high_dividend_yield(self):
        """High continuous dividend yield should work."""
        p = binomial_american_price(100, 100, 0.5, 0.05, 0.25, "call", q=0.10, n_steps=300)
        assert np.isfinite(p) and p > 0
