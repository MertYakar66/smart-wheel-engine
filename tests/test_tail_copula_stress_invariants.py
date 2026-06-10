"""Tail-risk, copula-CVaR and stress invariants — quant-layer test audit round 2
(W44-W48), PR-2.

Pins the risk-math contracts that feed EVEngine.evaluate's heavy-tail penalty
(``engine/tail_risk.py``), reviewer R8's vol-spike drawdown trigger
(``engine/stress_testing.py``), and the portfolio tail-dependence CVaR
(``engine/portfolio_copula.py``).

Behaviour-pinning (the #366 lesson): real numeric invariants, never a shape proxy.
In particular W45 runs the copula simulators FOR REAL — the existing verdict-ladder
tests monkeypatch both simulators away, so the percentile->tail-mean math that must
satisfy CVaR>=VaR was never actually exercised by an assertion. No §2 surface is
touched. The t-copula ``df`` bound guard is an engine change tracked as (E) #384,
not grabbed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.portfolio_copula import portfolio_cvar_copula, student_t_copula_simulation
from engine.stress_testing import Scenario, ScenarioType, StressTester
from engine.tail_risk import GPDTailFit, gpd_var_cvar

# ---------------------------------------------------------------------------
# W44 — gpd_var_cvar CVaR>=VaR on the xi<0 (thin-tail) branch. The existing tests
# only fit Gaussian/exponential/Pareto data (xi>=~0); the distinct xi<0 CVaR
# formula cvar=(var+beta-xi*u)/(1-xi) (tail_risk.py:217) is never exercised, nor is
# the abs(xi)<1e-8 exponential-branch boundary.
# ---------------------------------------------------------------------------


class TestGpdCvarNegativeXi:
    @pytest.mark.parametrize("xi", [-0.3, -1e-9])
    def test_cvar_ge_var_for_non_positive_xi(self, xi):
        # tail_prob = n_u/n = 0.1 > (1 - 0.99) so the POT formula path runs.
        fit = GPDTailFit(
            threshold=0.05,
            n_exceedances=100,
            n_total=1000,
            shape_xi=xi,
            scale_beta=0.02,
            tail_fraction=0.1,
            converged=True,
            log_likelihood=0.0,
        )
        var, cvar = gpd_var_cvar(fit, confidence=0.99)
        assert var > 0 and cvar > 0, (
            f"xi={xi}: expected positive loss magnitudes, got {var}, {cvar}"
        )
        assert cvar >= var, (
            f"xi={xi}: CVaR {cvar} must be >= VaR {var} (McNeil tail-mean invariant)"
        )


# ---------------------------------------------------------------------------
# W45 — portfolio_cvar_copula CVaR>=VaR on the REAL simulators. The coverage tests
# (test_portfolio_copula_coverage.py) monkeypatch gaussian_copula_simulation and
# student_t_copula_simulation, so the real percentile->tail-mean computation
# (portfolio_copula.py:193-204) — the code that MUST satisfy CVaR>=VaR — is never
# asserted. Run it for real and pin the invariant on both the Gaussian and t legs.
# ---------------------------------------------------------------------------


class TestCopulaCvarRealMath:
    def test_real_copula_cvar_ge_var_both_legs(self):
        rng = np.random.default_rng(0)
        marginals = [rng.normal(0.0, 0.02, 6000) for _ in range(3)]
        correlation = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
        weights = np.array([1.0, 1.0, 1.0])
        res = portfolio_cvar_copula(
            marginals, correlation, weights, confidence=0.95, n_samples=6000, seed=7
        )
        for leg in ("gaussian", "t"):
            var, cvar = res[f"{leg}_var"], res[f"{leg}_cvar"]
            assert np.isfinite(var) and np.isfinite(cvar), f"{leg}: non-finite VaR/CVaR"
            assert var > 0 and cvar > 0, (
                f"{leg}: expected positive loss magnitudes, got {var}, {cvar}"
            )
            assert cvar >= var - 1e-9, (
                f"{leg}: CVaR {cvar} must be >= VaR {var} (tail mean of the same cut)"
            )


# ---------------------------------------------------------------------------
# W46 — monte_carlo_stress seed reproducibility. The docstring claims "Uses seeded
# RNG for reproducibility" and the result carries _seed/_run_id, but no test passes
# a seed and asserts identical risk numbers — the contract was shape-checked (key
# presence), a #366 false-green. Same seed -> identical; different seed -> different.
# ---------------------------------------------------------------------------

_POS = [
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
_SPOT = {"AAPL": 155}
_NAV = 100_000


class TestMonteCarloStressDeterminism:
    def test_same_seed_reproducible_different_seed_differs(self):
        st = StressTester()
        a = st.monte_carlo_stress(_POS, _SPOT, _NAV, n_simulations=500, seed=123)
        b = st.monte_carlo_stress(_POS, _SPOT, _NAV, n_simulations=500, seed=123)
        for key in ("mean", "var_95", "var_99", "cvar_95"):
            assert a[key] == b[key], (
                f"{key} not reproducible under a fixed seed ({a[key]} != {b[key]})"
            )
        c = st.monte_carlo_stress(_POS, _SPOT, _NAV, n_simulations=500, seed=124)
        assert a["var_95"] != c["var_95"], "a different seed must change the draw (seed is plumbed)"


# ---------------------------------------------------------------------------
# W47 — monte_carlo_stress cvar_95 <= var_95. Both are negative-P&L numbers
# (stress_testing.py:494/:496); the conditional-tail mean must be at least as
# negative as the 5% quantile. Only key-presence + var_95<=0 were pinned.
# ---------------------------------------------------------------------------


class TestMonteCarloStressTailOrdering:
    def test_cvar95_at_most_var95(self):
        st = StressTester()
        r = st.monte_carlo_stress(_POS, _SPOT, _NAV, n_simulations=2000, seed=42)
        assert np.isfinite(r["var_95"]) and np.isfinite(r["cvar_95"])
        assert r["cvar_95"] <= r["var_95"], (
            f"cvar_95 {r['cvar_95']} must be <= var_95 {r['var_95']}"
        )
        assert r["var_95"] <= 0, (
            f"a short-put loss-tail book should have var_95 <= 0, got {r['var_95']}"
        )


# ---------------------------------------------------------------------------
# W48 — run_scenario expired-option intrinsic branch (new_dte<=0 -> max(0,K-S) /
# max(0,S-K), stress_testing.py:315-320). The existing near-expiry test uses dte=1
# with time_decay_days=0 (stays on the BSM path); the intrinsic payoff that drives
# R8 drawdown for assigned near-dated books has no value/sign assertion.
# ---------------------------------------------------------------------------


class TestRunScenarioIntrinsic:
    def _expire(self, spot_change_pct):
        # time_decay_days (40) > dte (30) forces new_dte<=0 -> the intrinsic branch.
        sc = Scenario(
            name="expiry",
            scenario_type=ScenarioType.MONTE_CARLO,
            description="forced expiry",
            spot_change_pct=spot_change_pct,
            time_decay_days=40,
        )
        return StressTester().run_scenario(sc, _POS, _SPOT, _NAV).portfolio_pnl

    def test_expired_itm_short_put_is_a_finite_loss(self):
        # spot 155 -> 124 (< strike 150): short put assigned ITM -> intrinsic 26 -> loss.
        pnl = self._expire(-0.20)
        assert np.isfinite(pnl), "expired-ITM intrinsic P&L must be finite"
        assert pnl < 0, f"an assigned (ITM) short put at expiry is a loss, got {pnl}"

    def test_expired_otm_short_put_is_a_finite_gain(self):
        # spot 155 -> 186 (> strike 150): put expires worthless -> short keeps value -> gain.
        pnl = self._expire(0.20)
        assert np.isfinite(pnl), "expired-OTM intrinsic P&L must be finite"
        assert pnl > 0, f"a worthless (OTM) short put at expiry is a gain, got {pnl}"


# ---------------------------------------------------------------------------
# t_copula_df bound guard (closes (E) #384). The t-copula df parameter fed
# rng.chisquare(df) and stats.t.cdf(df) with NO bound check: df<=0 hit numpy's
# terse "df <= 0" ValueError, and 0<df<=2 ran with INFINITE variance (a fragile
# CVaR) silently. _validate_t_copula_df now raises a clear domain error on df<=0
# (and on non-finite df) and warns on 0<df<=2; df>2 is unchanged. Pinned on the
# real simulator AND through portfolio_cvar_copula (which delegates to it).
# ---------------------------------------------------------------------------

_MARGINALS = [np.linspace(-0.1, 0.1, 200) for _ in range(3)]
_CORR = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.25], [0.2, 0.25, 1.0]])
_WEIGHTS = np.array([1.0, 1.0, 1.0])


class TestTCopulaDfGuard:
    @pytest.mark.parametrize("bad_df", [0.0, -1.0, np.nan, np.inf])
    def test_non_positive_or_nonfinite_df_raises(self, bad_df):
        # df<=0 is invalid (Student-t requires positive dof); non-finite is invalid.
        # Raised BEFORE rng.chisquare, so the message is the clear domain one.
        with pytest.raises(ValueError, match="t_copula_df"):
            student_t_copula_simulation(_MARGINALS, _CORR, df=bad_df, n_samples=500, seed=1)

    def test_non_positive_df_raises_through_portfolio_cvar(self):
        # portfolio_cvar_copula delegates to the simulator, so the guard fires there too.
        with pytest.raises(ValueError, match="t_copula_df"):
            portfolio_cvar_copula(
                _MARGINALS, _CORR, _WEIGHTS, t_copula_df=0.0, n_samples=500, seed=1
            )

    @pytest.mark.parametrize("frag_df", [1.0, 2.0])
    def test_infinite_variance_df_warns_but_runs(self, frag_df):
        # 0<df<=2 is mathematically valid (infinite variance) -> warn, don't block;
        # the simulation still returns a well-shaped, finite draw.
        with pytest.warns(RuntimeWarning, match="infinite variance"):
            out = student_t_copula_simulation(_MARGINALS, _CORR, df=frag_df, n_samples=500, seed=1)
        assert out.shape == (500, 3)
        assert np.isfinite(out).all()

    def test_safe_df_does_not_warn(self):
        # df=5 (the default, moderate tail dependence) must NOT emit the fragility
        # warning. simplefilter("error") turns any RuntimeWarning into a failure.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = student_t_copula_simulation(_MARGINALS, _CORR, df=5.0, n_samples=500, seed=1)
        assert out.shape == (500, 3)
