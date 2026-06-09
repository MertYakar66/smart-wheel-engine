"""HMM regime-multiplier invariants — quant-layer test audit round 2 (W50-W55), PR-3.

The 4-state Gaussian HMM (``engine/regime_hmm.py``) produces the regime multiplier
that scales ``ev_dollars`` (``hmm_mult`` -> ``combined_regime_mult`` -> EVResult);
``engine/regime_detector.py`` is the older heuristic sizing path. The fit guards
(``T<K*3``, near-constant) are already covered by test_quant_upgrades; this pins the
GAPS: the multiplier envelope's true sup/inf, same-seed determinism (the cache + the
backtest fingerprint depend on it), the unfit-guard RuntimeError, and the
RegimeDetector degenerate fallback.

Behaviour-pinning (the #366 lesson): real numeric invariants, never a shape proxy.
§2-relevant: the multiplier only SCALES a >=0 EV envelope (it cannot flip sign), so
these assert the documented [0.2, 1.25] envelope — never weaken it. ``fit`` failing
to guard non-finite input (returns a NaN multiplier) is an engine change tracked as
(E) #386, pinned here with xfail(strict) asserting the desired raise — not grabbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.regime_detector import RegimeDetector
from engine.regime_hmm import GaussianHMM

# Per-label weights (regime_hmm.py:288-293) — the documented envelope.
_WEIGHTS = {"crisis": 0.2, "bear": 0.5, "normal": 1.0, "bull_quiet": 1.25}


def _two_regime_returns(seed: int = 0) -> np.ndarray:
    """A calm/crash/calm log-return series that fits cleanly into 4 states."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0005, 0.008, 400)
    crash = rng.normal(-0.01, 0.04, 120)
    return np.concatenate([calm, crash, calm])


# ---------------------------------------------------------------------------
# W50 — position_multiplier envelope sup/inf. The existing test asserts
# 0<=mult<=1.25 on ONE diffuse random posterior — it passes for any weights <=~1.25
# and never reaches the extremes. Pin the achieved bounds: a one-hot on each label
# returns exactly that label's weight, so the sup is 1.25 (bull_quiet) and the inf
# is 0.2 (crisis), and every simplex point lands inside [0.2, 1.25].
# ---------------------------------------------------------------------------


class TestHmmMultiplierEnvelope:
    def test_one_hot_maps_to_label_weight_and_extremes(self):
        hmm = GaussianHMM(n_states=4, random_state=42)
        hmm.fit(_two_regime_returns())
        labels = hmm.fit_result.state_labels
        assert set(labels) == set(_WEIGHTS), f"unexpected labels {labels}"
        mults = []
        for i, lbl in enumerate(labels):
            one_hot = np.zeros(4)
            one_hot[i] = 1.0
            m = hmm.position_multiplier(one_hot)
            assert m == pytest.approx(_WEIGHTS[lbl]), (
                f"one-hot {lbl} -> {m}, expected {_WEIGHTS[lbl]}"
            )
            mults.append(m)
        assert max(mults) == pytest.approx(1.25), "bull_quiet one-hot must be the 1.25 supremum"
        assert min(mults) == pytest.approx(0.2), "crisis one-hot must be the 0.2 infimum"

    def test_every_simplex_point_inside_envelope(self):
        hmm = GaussianHMM(n_states=4, random_state=42)
        hmm.fit(_two_regime_returns())
        rng = np.random.default_rng(1)
        for _ in range(200):
            v = rng.dirichlet([1, 1, 1, 1])
            m = hmm.position_multiplier(v)
            assert 0.2 - 1e-9 <= m <= 1.25 + 1e-9, f"multiplier {m} escaped [0.2, 1.25]"


# ---------------------------------------------------------------------------
# W51 — same-seed determinism. random_state is documented for reproducibility and
# the production ranker caches the fit + the backtest fingerprint gate assumes it,
# but no test asserts two fits of the same data with the same seed give identical
# means / labels / multiplier. An EM-init change could silently shift the multiplier.
# ---------------------------------------------------------------------------


class TestHmmDeterminism:
    def test_same_seed_same_fit_and_multiplier(self):
        x = _two_regime_returns()
        h1 = GaussianHMM(n_states=4, random_state=42)
        h1.fit(x)
        h2 = GaussianHMM(n_states=4, random_state=42)
        h2.fit(x)
        np.testing.assert_allclose(h1.fit_result.means, h2.fit_result.means)
        assert h1.fit_result.state_labels == h2.fit_result.state_labels
        m1 = h1.position_multiplier(h1.predict_proba(x)[-1])
        m2 = h2.position_multiplier(h2.predict_proba(x)[-1])
        assert m1 == m2, f"same-seed multiplier not reproducible: {m1} != {m2}"


# ---------------------------------------------------------------------------
# W53 — unfit-guard RuntimeError. predict_proba / viterbi / position_multiplier each
# raise RuntimeError when fit_result is None (regime_hmm.py:239/:253/:285). This is
# the guard the ranker's broad try/except relies on to fall back to neutral 1.0; a
# refactor returning garbage instead of raising would silently feed an undefined
# multiplier. Never asserted (every existing test calls .fit() first).
# ---------------------------------------------------------------------------


class TestHmmUnfitGuards:
    def test_methods_raise_before_fit(self):
        hmm = GaussianHMM(n_states=4)
        with pytest.raises(RuntimeError):
            hmm.predict_proba(np.zeros(10))
        with pytest.raises(RuntimeError):
            hmm.viterbi(np.zeros(10))
        with pytest.raises(RuntimeError):
            hmm.position_multiplier(np.array([1.0, 0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# W54 — RegimeDetector degenerate realized-vol fallback. _calculate_realized_vol
# returns a 0.20 default when there are <2 returns (regime_detector.py:251-252) —
# the existing detector tests all use >=50-point clean series, so the guard that
# stops a NaN realized_vol from poisoning iv_rv_spread / sizing is unpinned.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::FutureWarning")  # pct_change fill_method deprecation
class TestRegimeDetectorDegenerate:
    def test_realized_vol_defaults_on_too_few_points(self):
        rd = RegimeDetector()
        assert rd._calculate_realized_vol(pd.Series([100.0])) == 0.20
        assert rd._calculate_realized_vol(pd.Series([], dtype=float)) == 0.20


# ---------------------------------------------------------------------------
# W55 — (E) #386. GaussianHMM.fit does NOT guard non-finite observations: np.nanstd
# ignores NaNs so the std gate passes, and the M-step produces NaN means -> a NaN
# position_multiplier (probed). The ranker try/except only catches a RAISE (it does
# not here), so only the downstream ev_engine non-finite clamp saves EV. DESIRED:
# fit raises on non-finite input so the neutral-1.0 fallback engages cleanly.
# xfail(strict) flips green when the (E) guard lands.
# ---------------------------------------------------------------------------


class TestHmmNonFiniteInput:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.xfail(
        reason="(E) #386: fit does not guard non-finite input; returns NaN multiplier instead of raising",
        strict=True,
    )
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_input_should_raise(self, bad):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.01, 200)
        x[50] = bad
        with pytest.raises(ValueError):
            GaussianHMM(n_states=4, random_state=42).fit(x)
