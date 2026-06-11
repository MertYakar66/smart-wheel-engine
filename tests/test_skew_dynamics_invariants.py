"""Skew-dynamics invariants — quant-layer test audit round 2 (W56-W59), PR-4.

Pins the standalone skew-math primitives in ``engine/skew_dynamics.py``: the
Nelson-Siegel term-structure fail-fast + degenerate-fit contract, the skew-momentum
degenerate-history behaviour, and the surface-dislocation composite bound.

Scope note: the LIVE ``skew_mult`` mapping (``clip(1.0-0.5*slope, 0.85, 1.08)``,
wheel_runner.py:1516) is in the decision trio AND dormant on the Bloomberg path
(no put/call skew -> slope==0 -> mult==1.0; see memory bloomberg-iv-no-skew), and the
"a regime/skew boost cannot rescue a negative EV" §2 invariant is already covered by
the generic regime-multiplier test — so neither is re-pinned here. These are the
behaviour-pinning primitive contracts (the #366 lesson); no §2 surface is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.skew_dynamics import (
    NelsonSiegelTermStructure,
    ivs_dislocation_score,
    skew_momentum,
)

# ---------------------------------------------------------------------------
# W56 — NelsonSiegelTermStructure fail-fast. iv_at / factor_loadings raise
# RuntimeError before fit (a deliberate not-silently-0.20 contract, the D9 spirit).
# Never asserted (every existing use fits first).
# ---------------------------------------------------------------------------


class TestNelsonSiegelUnfitGuards:
    def test_iv_at_and_factor_loadings_raise_before_fit(self):
        ns = NelsonSiegelTermStructure()
        with pytest.raises(RuntimeError):
            ns.iv_at(0.5)
        with pytest.raises(RuntimeError):
            ns.factor_loadings()


# ---------------------------------------------------------------------------
# W57 — NS degenerate-fit branches. n<2 (skew_dynamics.py:77-87) is a distinct path:
# n==1 -> beta0 = the single observed IV, slope/curvature 0, converged False,
# n_points 1; n==0 (all points filtered by the finite & T>0 & y>0 mask) -> the 0.20
# sentinel level. Unpinned, so a regression mishandling a one-tenor term structure
# would be invisible.
# ---------------------------------------------------------------------------


class TestNelsonSiegelDegenerateFit:
    def test_single_point_fit_is_level_only(self):
        ns = NelsonSiegelTermStructure()
        fit = ns.fit(np.array([0.25]), np.array([0.30]))
        assert fit.n_points == 1
        assert fit.converged is False
        assert fit.beta0 == pytest.approx(0.30)
        assert fit.beta1 == 0.0 and fit.beta2 == 0.0
        # A level-only fit evaluates to beta0 at any tenor.
        assert ns.iv_at(0.5) == pytest.approx(0.30)

    def test_all_points_filtered_falls_back_to_sentinel_level(self):
        # T <= 0 is masked out -> n == 0 -> the 0.20 sentinel beta0.
        ns = NelsonSiegelTermStructure()
        fit = ns.fit(np.array([-1.0]), np.array([0.30]))
        assert fit.n_points == 0
        assert fit.converged is False
        assert fit.beta0 == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# W58 — skew_momentum degenerate history. Empty -> all-NaN with steepening False;
# a history no longer than short_window collapses both rolling slices to the same
# values -> momentum == 0 and steepening False (a silent "no signal" callers rely on).
# Existing tests use long clean histories only.
# ---------------------------------------------------------------------------


class TestSkewMomentumDegenerate:
    def test_empty_history_is_nan_and_not_steepening(self):
        m = skew_momentum(np.array([]))
        assert np.isnan(m["momentum"]) and np.isnan(m["current_skew"])
        assert m["steepening"] is False

    def test_short_history_yields_zero_momentum_no_signal(self):
        # n == 1 and n (=3) <= short_window (5): arr[-short:] == arr[-long:] -> momentum 0.
        for hist in (np.array([-0.05]), np.array([-0.05, -0.06, -0.04])):
            m = skew_momentum(hist, short_window=5, long_window=21)
            assert m["momentum"] == 0.0, (
                f"short history should give 0 momentum, got {m['momentum']}"
            )
            assert m["steepening"] is False


# ---------------------------------------------------------------------------
# W59 — ivs_dislocation_score composite bound. The docstring promises a composite in
# [-1, +1] (the normalised rich/cheap signal callers rely on); pin the bound holds on
# both a normal and a stressed (extreme-outlier) term structure.
# ---------------------------------------------------------------------------


class TestIvsDislocationBound:
    @pytest.mark.parametrize(
        "ivs",
        [
            np.array([0.28, 0.25, 0.23, 0.22, 0.21]),  # normal downward term structure
            np.array([0.20, 0.20, 0.20, 0.20, 5.0]),  # extreme outlier tenor
        ],
    )
    def test_composite_score_within_unit_band(self, ivs):
        tenors = np.array([0.08, 0.25, 0.5, 1.0, 2.0])
        score = ivs_dislocation_score(tenors, ivs)["composite_score"]
        assert -1.0 <= score <= 1.0, f"composite_score {score} escaped the documented [-1, 1] band"
