"""Tests for engine/tail_risk.py — POT-GPD tail estimation."""

from __future__ import annotations

import numpy as np
import pytest

from engine.tail_risk import (
    GPDTailFit,
    fit_gpd_tail,
    gpd_var_cvar,
    pot_gpd_cvar,
    select_threshold,
    tail_regime_flag,
)


class TestSelectThreshold:
    def test_default_95th_percentile(self):
        rng = np.random.default_rng(0)
        losses = np.abs(rng.standard_normal(1000))
        u = select_threshold(losses)
        # Roughly 5% of losses exceed u
        frac = (losses > u).mean()
        assert 0.03 <= frac <= 0.07

    def test_explicit_percentile(self):
        losses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        u = select_threshold(losses, percentile=90.0)
        # 90th percentile of 1..10 is ~9.1
        assert 8.5 <= u <= 9.5

    def test_empty_array_returns_zero(self):
        assert select_threshold(np.array([])) == 0.0

    def test_drops_non_finite(self):
        losses = np.array([1.0, 2.0, np.nan, np.inf, 3.0, 4.0, 5.0])
        u = select_threshold(losses, percentile=50.0)
        # 50th percentile of finite values [1,2,3,4,5] is 3.0
        assert u == pytest.approx(3.0)


class TestFitGpdTail:
    def test_fit_returns_gpd_tailfit(self):
        rng = np.random.default_rng(42)
        # Generate exponential losses (xi=0 limit)
        losses = rng.exponential(scale=0.02, size=5000)
        fit = fit_gpd_tail(losses)
        assert isinstance(fit, GPDTailFit)
        assert fit.n_total == 5000
        assert fit.n_exceedances > 0
        assert fit.scale_beta > 0
        assert fit.converged

    def test_explicit_threshold_used(self):
        losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100)
        fit = fit_gpd_tail(losses, threshold=0.3)
        assert fit.threshold == pytest.approx(0.3)
        # Exceedances are losses > 0.3 (so 0.4 and 0.5 → 200 of them)
        assert fit.n_exceedances == 200

    def test_too_few_exceedances_returns_unconverged(self):
        # Only 5 exceedances above threshold, below default min_exceedances=15
        losses = np.array([0.01] * 100 + [0.99] * 5)
        fit = fit_gpd_tail(losses, threshold=0.5)
        assert fit.converged is False
        assert fit.n_exceedances == 5

    def test_empty_array_returns_sentinel(self):
        fit = fit_gpd_tail(np.array([]))
        assert fit.n_total == 0
        assert fit.n_exceedances == 0
        assert fit.converged is False

    def test_drops_non_finite(self):
        # 5 values per repeat × 100 repeats = 500; nan+inf account for 2/5,
        # leaving 3/5 × 500 = 300 finite values.
        losses = np.array([0.1, 0.2, np.nan, np.inf, 0.3] * 100)
        fit = fit_gpd_tail(losses)
        assert fit.n_total == 300
        # Exact converged-vs-not depends on threshold + exceedance count;
        # we only require that beta is positive (sentinel still sets it).
        assert fit.scale_beta > 0


class TestGpdVarCvar:
    def test_unconverged_fit_returns_zero(self):
        empty_fit = GPDTailFit(0.0, 0, 0, 0.0, 1e-6, 0.0, False, 0.0)
        var, cvar = gpd_var_cvar(empty_fit, confidence=0.99)
        assert var == 0.0
        assert cvar == 0.0

    def test_xi_zero_branch(self):
        # Exponential tail → xi ≈ 0 → uses logarithmic VaR formula
        rng = np.random.default_rng(7)
        losses = rng.exponential(scale=0.02, size=10000)
        fit = fit_gpd_tail(losses)
        var, cvar = gpd_var_cvar(fit, confidence=0.99)
        assert var > 0
        assert cvar >= var  # CVaR is always >= VaR

    def test_xi_positive_branch(self):
        # Pareto tail (xi > 0)
        rng = np.random.default_rng(11)
        losses = rng.pareto(2.0, size=5000) * 0.01
        fit = fit_gpd_tail(losses)
        var, cvar = gpd_var_cvar(fit, confidence=0.99)
        assert var > 0
        assert cvar >= var

    def test_extreme_xi_caps_cvar(self):
        # Synthesise xi >= 1 fit; check the practical-bound clause runs.
        fit = GPDTailFit(
            threshold=0.05,
            n_exceedances=100,
            n_total=1000,
            shape_xi=1.2,
            scale_beta=0.02,
            tail_fraction=0.1,
            converged=True,
            log_likelihood=0.0,
        )
        var, cvar = gpd_var_cvar(fit, confidence=0.99)
        assert cvar == pytest.approx(var * 3.0)

    def test_confidence_below_tail_prob_returns_empirical(self):
        # When 1-alpha > tail_prob the function shortcircuits to a
        # quasi-empirical bound (u, u + beta).
        fit = GPDTailFit(
            threshold=0.05,
            n_exceedances=100,
            n_total=1000,
            shape_xi=0.2,
            scale_beta=0.02,
            tail_fraction=0.1,
            converged=True,
            log_likelihood=0.0,
        )
        # tail_prob = 0.1; 1 - 0.5 = 0.5 > 0.1 → shortcircuits
        var, cvar = gpd_var_cvar(fit, confidence=0.5)
        assert var == 0.05
        assert cvar == 0.07  # u + beta

    def test_n_total_zero_returns_zero(self):
        empty = GPDTailFit(0.0, 0, 0, 0.0, 1e-6, 0.0, True, 0.0)
        var, cvar = gpd_var_cvar(empty, confidence=0.99)
        assert var == 0.0
        assert cvar == 0.0


class TestTailRegimeFlag:
    def test_unconverged_returns_false(self):
        fit = GPDTailFit(0.0, 0, 0, 0.5, 1.0, 0.0, False, 0.0)
        assert tail_regime_flag(fit) is False

    def test_xi_above_threshold_flags(self):
        fit = GPDTailFit(0.0, 100, 1000, 0.5, 1.0, 0.1, True, 0.0)
        assert tail_regime_flag(fit) is True
        assert tail_regime_flag(fit, heavy_tail_threshold=0.6) is False

    def test_xi_below_threshold_does_not_flag(self):
        fit = GPDTailFit(0.0, 100, 1000, 0.1, 1.0, 0.1, True, 0.0)
        assert tail_regime_flag(fit) is False


class TestPotGpdCvarWrapper:
    def test_returns_full_diagnostic_dict(self):
        rng = np.random.default_rng(99)
        losses = rng.exponential(scale=0.02, size=2000)
        result = pot_gpd_cvar(losses)
        for key in (
            "var",
            "cvar",
            "xi",
            "beta",
            "threshold",
            "n_exceedances",
            "n_total",
            "tail_fraction",
            "heavy_tail",
            "fit_quality",
            "converged",
        ):
            assert key in result

    def test_quality_classification_thin_tail(self):
        rng = np.random.default_rng(1)
        losses = rng.exponential(scale=0.01, size=5000)
        result = pot_gpd_cvar(losses)
        assert result["fit_quality"] in {"thin_tail", "finite_variance"}

    def test_quality_classification_heavy_tail(self):
        # Pareto with xi > 0.3 → heavy_tail
        rng = np.random.default_rng(3)
        losses = rng.pareto(1.5, size=5000) * 0.01
        result = pot_gpd_cvar(losses)
        # Pareto tail index 1.5 → xi = 1/1.5 ≈ 0.67 → extreme_heavy_tail
        assert result["fit_quality"] in {"heavy_tail", "extreme_heavy_tail"}
        assert result["heavy_tail"] is True

    def test_quality_insufficient_data(self):
        # Tiny sample → unconverged
        result = pot_gpd_cvar(np.array([0.01, 0.02, 0.99]))
        assert result["fit_quality"] == "insufficient_data"
        assert result["converged"] is False

    def test_var_cvar_positive_for_normal_losses(self):
        rng = np.random.default_rng(13)
        losses = np.abs(rng.standard_normal(2000)) * 0.02
        result = pot_gpd_cvar(losses, confidence=0.99)
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]
