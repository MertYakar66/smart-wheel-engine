"""Edge-path coverage for ``engine/portfolio_copula.py``.

The 2026-05-08 audit (DECISIONS.md D10) found this module at 78% line
coverage — below the 80% floor. ``tests/test_quant_upgrades.py`` covers
the happy paths (Gaussian/Student-t marginal preservation, t-copula
tail dependence, end-to-end CVaR). The lines this file targets are:

* PSD repair when eigenvalues fall below ``min_eig`` (L51-55).
* Cholesky LinAlgError → eigen fallback (L68-71).
* ``_empirical_quantile`` on an empty values array (L84).
* Empty-marginals short-circuit in both copula simulators
  (L112, L138).
* ``portfolio_cvar_copula`` length-mismatch ``ValueError`` (L187).
* The four verdict-ladder branches in ``portfolio_cvar_copula``
  (negligible / mild / material / critical at L207-214).

All tests are hermetic — pure numpy/pandas, no Theta Terminal, no
network. They do not change runtime behaviour; they only exercise
existing public/internal surface.

The verdict-ladder branches are driven via ``monkeypatch`` on
``gaussian_copula_simulation`` and ``student_t_copula_simulation``
because the natural amplification produced by realistic equity
marginals tops out around 1.2-1.4 — landing in the ``>1.5`` "critical"
band requires data shapes that don't otherwise occur in the wheel
universe. Patching the two simulators with deterministic synthetic
arrays lets us pin each branch without depending on noisy stochastic
output.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine import portfolio_copula as pc
from engine.portfolio_copula import (
    _correlation_psd_repair,
    _empirical_quantile,
    _sample_correlated_normals,
    gaussian_copula_simulation,
    portfolio_cvar_copula,
    student_t_copula_simulation,
)


# =========================================================================
# 1. _correlation_psd_repair — eigenvalue clamping branch
# =========================================================================
class TestCorrelationPsdRepair:
    """Cover the ``np.min(eigvals) < min_eig`` branch (L51-55)."""

    def test_negative_eigenvalue_is_clamped_and_diagonal_renormalised(self):
        # Off-diagonals of magnitude > 1 force the smallest eigenvalue
        # negative — the repair branch must fire.
        bad = np.array(
            [
                [1.0, 1.4, 1.4],
                [1.4, 1.0, 1.4],
                [1.4, 1.4, 1.0],
            ]
        )
        pre_eigvals = np.linalg.eigvalsh(bad)
        assert pre_eigvals.min() < 0.0

        repaired = _correlation_psd_repair(bad, min_eig=1e-8)

        # Post-conditions: PSD (all eigenvalues >= min_eig within tol)
        # and diagonal renormalised to 1.0.
        post_eigvals = np.linalg.eigvalsh(repaired)
        assert post_eigvals.min() >= 1e-9
        np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-10)
        np.testing.assert_allclose(repaired, repaired.T, atol=1e-12)

    def test_already_psd_matrix_passes_through_unchanged(self):
        # Identity is already PSD — the if-branch is skipped, so the
        # output equals the symmetrised input (still identity).
        good = np.eye(3)
        out = _correlation_psd_repair(good, min_eig=1e-8)
        np.testing.assert_allclose(out, good, atol=1e-12)


# =========================================================================
# 2. _sample_correlated_normals — Cholesky failure → eigen fallback
# =========================================================================
class TestSampleCorrelatedNormalsCholeskyFallback:
    """Cover the ``np.linalg.LinAlgError`` branch (L68-71).

    The function tries ``np.linalg.cholesky`` first; on a degenerate
    (rank-deficient or non-PSD) correlation it falls back to the eigen
    decomposition. Constructing a non-PSD matrix triggers the fallback
    deterministically.
    """

    def test_singular_correlation_falls_back_to_eigen_decomposition(self):
        # Rank-deficient (zero eigenvalue) — Cholesky raises, eigen path
        # produces a valid sample matrix of shape (n_samples, n_assets).
        # Two perfectly-correlated assets ⇒ correlation [[1,1],[1,1]].
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        # Sanity: confirm Cholesky would in fact fail.
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(corr)

        rng = np.random.default_rng(123)
        z = _sample_correlated_normals(n_samples=500, corr=corr, rng=rng)

        assert z.shape == (500, 2)
        # The fallback uses eigenvalue clipping so finite values come out.
        assert np.all(np.isfinite(z))


# =========================================================================
# 3. _empirical_quantile — empty values array branch
# =========================================================================
class TestEmpiricalQuantileEmpty:
    """Cover the ``n == 0`` branch (L84)."""

    def test_empty_values_returns_zeros_with_u_shape(self):
        u = np.array([0.1, 0.5, 0.9])
        out = _empirical_quantile(np.array([]), u)
        assert out.shape == u.shape
        np.testing.assert_array_equal(out, np.zeros_like(u))


# =========================================================================
# 4. gaussian / student_t copula — N==0 marginals short-circuit
# =========================================================================
class TestCopulaEmptyMarginals:
    """Cover the empty-marginals short-circuits (L112 and L138)."""

    def test_gaussian_empty_marginals_returns_empty_columns(self):
        out = gaussian_copula_simulation(
            marginals=[],
            correlation=np.zeros((0, 0)),
            n_samples=128,
            seed=0,
        )
        assert out.shape == (128, 0)

    def test_student_t_empty_marginals_returns_empty_columns(self):
        out = student_t_copula_simulation(
            marginals=[],
            correlation=np.zeros((0, 0)),
            df=5.0,
            n_samples=64,
            seed=0,
        )
        assert out.shape == (64, 0)


# =========================================================================
# 5. portfolio_cvar_copula — weight-length validation
# =========================================================================
class TestPortfolioCvarWeightValidation:
    """Cover the ``len(w) != N`` raise (L187)."""

    def test_weights_length_mismatch_raises_value_error(self):
        rng = np.random.default_rng(0)
        marg = [rng.normal(0, 0.01, 200) for _ in range(2)]
        corr = np.eye(2)
        with pytest.raises(ValueError, match="weights length"):
            portfolio_cvar_copula(
                marginals=marg,
                correlation=corr,
                # Three weights for two marginals ⇒ contract violation.
                weights=np.array([0.5, 0.3, 0.2]),
                confidence=0.95,
                n_samples=200,
                seed=0,
            )


# =========================================================================
# 6. portfolio_cvar_copula — verdict ladder (L207-214)
# =========================================================================
#
# The verdict is decided by ``amp = t_cvar / gaussian_cvar``:
#
#   amp > 1.5   → "critical_tail_dependence"
#   amp > 1.3   → "material_tail_dependence"
#   amp > 1.1   → "mild_tail_dependence"
#   otherwise   → "negligible_tail_dependence"
#
# Realistic equity marginals top out around amp ≈ 1.2-1.4 in this
# function — never landing in the >1.5 "critical" band reliably under
# any seed. We patch the two simulator functions with deterministic
# synthetic arrays so the weighted-sum percentile produces the target
# (g_cvar, t_cvar) pair on the nose. Each branch is covered exactly
# once with an explicit verdict assertion.
# =========================================================================
def _arrays_with_target_cvar(
    g_cvar_target: float, t_cvar_target: float, n_samples: int = 2000
):
    """Return (gauss_arr, t_arr) shaped (n_samples, 1) so that the
    weighted port-sum (with weight 1.0) at confidence=0.99 yields
    g_cvar = g_cvar_target and t_cvar = t_cvar_target.

    Trick: at 99% confidence, the bottom 1% of n_samples samples is the
    tail. We set the bottom 1% to ``-target`` and the rest to 0, so
    np.percentile at 1% is ``-target`` and CVaR (mean of tail) is also
    ``-target`` ⇒ ``-(-target) = target``.
    """
    one_pct = max(1, int(0.01 * n_samples))

    def _build(target):
        a = np.zeros((n_samples, 1))
        a[:one_pct, 0] = -target
        return a

    return _build(g_cvar_target), _build(t_cvar_target)


class TestPortfolioCvarVerdictLadder:
    @pytest.mark.parametrize(
        "g_cvar,t_cvar,expected_verdict,expected_amp",
        [
            (0.01, 0.01, "negligible_tail_dependence", 1.0),     # amp <= 1.1 → else (L214)
            (0.01, 0.012, "mild_tail_dependence", 1.2),          # 1.1 < amp <= 1.3 → L212
            (0.01, 0.014, "material_tail_dependence", 1.4),      # 1.3 < amp <= 1.5 → L210
            (0.01, 0.017, "critical_tail_dependence", 1.7),      # amp > 1.5 → L208
        ],
        ids=["negligible", "mild", "material", "critical"],
    )
    def test_verdict_branches(
        self,
        monkeypatch,
        g_cvar: float,
        t_cvar: float,
        expected_verdict: str,
        expected_amp: float,
    ):
        n_samples = 2000
        gauss_arr, t_arr = _arrays_with_target_cvar(g_cvar, t_cvar, n_samples)

        def fake_gauss(*_args, **_kwargs):
            return gauss_arr

        def fake_t(*_args, **_kwargs):
            return t_arr

        monkeypatch.setattr(pc, "gaussian_copula_simulation", fake_gauss)
        monkeypatch.setattr(pc, "student_t_copula_simulation", fake_t)

        res = portfolio_cvar_copula(
            marginals=[np.array([0.0])],  # ignored by patched simulators
            correlation=np.array([[1.0]]),
            weights=np.array([1.0]),
            confidence=0.99,
            n_samples=n_samples,
            seed=0,
            t_copula_df=5.0,
        )

        assert res["verdict"] == expected_verdict
        # Allow tiny rounding drift from percentile/mean operations.
        assert abs(res["tail_amplification"] - expected_amp) < 1e-9


# =========================================================================
# 7. portfolio_cvar_copula — g_cvar==0 fallback (amp = 1.0)
# =========================================================================
class TestPortfolioCvarZeroGaussianCvar:
    """Cover the ``amp = ... if g_cvar > 0 else 1.0`` else-branch (L206).

    When the Gaussian-copula port-sum produces no negative-percentile
    mass (e.g. all-zero synthetic marginals), the engine returns the
    fallback amp=1.0 instead of dividing by zero.
    """

    def test_zero_gaussian_cvar_yields_amp_one_negligible_verdict(
        self, monkeypatch
    ):
        n_samples = 1000
        zeros = np.zeros((n_samples, 1))

        # Both copulas return all-zeros ⇒ g_cvar = t_cvar = 0.
        # The function should take the fallback amp=1.0 path.
        monkeypatch.setattr(
            pc, "gaussian_copula_simulation", lambda *a, **k: zeros
        )
        monkeypatch.setattr(
            pc, "student_t_copula_simulation", lambda *a, **k: zeros
        )

        res = portfolio_cvar_copula(
            marginals=[np.array([0.0])],
            correlation=np.array([[1.0]]),
            weights=np.array([1.0]),
            confidence=0.99,
            n_samples=n_samples,
            seed=0,
            t_copula_df=5.0,
        )
        assert res["gaussian_cvar"] == 0.0
        assert res["t_cvar"] == 0.0
        assert res["tail_amplification"] == 1.0
        assert res["verdict"] == "negligible_tail_dependence"


# =========================================================================
# 8. Public-contract smoke: returned dict shape stays stable
# =========================================================================
class TestReturnedDictContract:
    """Lock the public-contract surface so a silent rename
    (e.g. ``t_var`` → ``student_t_var``) breaks tests rather than
    callers."""

    def test_returned_dict_has_every_documented_key(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.01, 800)
        b = rng.normal(0, 0.01, 800)
        res = portfolio_cvar_copula(
            marginals=[a, b],
            correlation=np.eye(2),
            weights=np.array([0.6, 0.4]),
            confidence=0.95,
            n_samples=800,
            seed=0,
            t_copula_df=5.0,
        )
        assert set(res.keys()) == {
            "gaussian_var",
            "gaussian_cvar",
            "t_var",
            "t_cvar",
            "tail_amplification",
            "verdict",
            "confidence",
            "n_samples",
            "t_copula_df",
        }
        # Four scalar risk numbers are finite floats.
        for k in ("gaussian_var", "gaussian_cvar", "t_var", "t_cvar"):
            assert isinstance(res[k], float)
            assert np.isfinite(res[k])
