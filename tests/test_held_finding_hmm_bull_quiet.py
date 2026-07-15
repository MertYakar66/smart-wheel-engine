"""Held-finding reproduction — F3-hmm-bull-quiet.

FINDING 3: HMM 'bull_quiet' label is assigned by within-window RANK, not by the
absolute sign of the state's mean return, so a steady low-vol DECLINE (all four
states have NEGATIVE mean returns) still labels the least-negative state
'bull_quiet' and up-sizes the position 1.25x on a down regime.

Root cause (engine/regime_hmm.py):
  * _label_states(means, stds, K)  L357-369: for K==4 it *unconditionally*
    returns ["crisis", "bear", "normal", "bull_quiet"] — pure positional lookup,
    NO reference to the sign of any state's mean return.
  * fit()  L225-232: scores = means[:,0] - 0.5*stds[:,0]; order = argsort(scores)
    (ascending), so the last state index (K-1) is always the highest-scored
    state and thus always gets the 'bull_quiet' label — even when its mean is
    negative.
  * position_multiplier(...)  L292-312: maps 'bull_quiet' -> 1.25 by pure label
    lookup, with NO sign gate.

These tests assert the CORRECT (bug-free) behavior:
  1. A 4-state configuration whose top-ranked state (index K-1 after the engine's
     ascending sort) has a NEGATIVE mean must NOT label that state 'bull_quiet'.
  2. Through the full engine path (fit on a synthetic steady low-vol DECLINE),
     the position multiplier for the current down regime must be <= 1.0, never
     the 1.25x up-size the engine currently emits.

Both currently FAIL on the real engine code, proving the bug. They are marked
xfail(strict=True) so they are green (xfailed) now and flip to a hard failure
(xpass) the moment the bug is fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.regime_hmm import GaussianHMM


# ---------------------------------------------------------------------------
# Test 1 — direct, deterministic: call _label_states on a synthetic 4-state
# configuration that is a steady low-vol DECLINE (all means negative), laid out
# in ascending-score order exactly as fit() would hand it to _label_states.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="held finding F3-hmm-bull-quiet; drop xfail when fixed", strict=True
)
def test_negative_mean_top_state_is_not_bull_quiet():
    K = 4

    # A steady, low-volatility DECLINE: every state has a NEGATIVE mean return.
    # Column 0 is the primary (log-return) feature that fit() sorts / labels on;
    # column 1 is a realised-vol feature (unused by the label logic but present
    # because fit() emits (T, 2) observations).
    #
    # Ordered ascending by the engine's score = means[:,0] - 0.5*stds[:,0] so
    # index K-1 is the "best" (least-negative) state — the one fit() would label
    # 'bull_quiet'. Its mean is -0.0005 (a ~-0.05%/day drift): still a DOWN
    # regime, not a bull.
    means = np.array(
        [
            [-0.0200, 0.030],  # most negative  -> crisis
            [-0.0100, 0.020],  # negative       -> bear
            [-0.0040, 0.012],  # negative       -> normal
            [-0.0005, 0.006],  # least negative -> engine mislabels 'bull_quiet'
        ]
    )
    stds = np.array(
        [
            [0.030, 0.010],
            [0.020, 0.008],
            [0.012, 0.006],
            [0.006, 0.004],
        ]
    )

    # Sanity: confirm the array is in the ascending-score order fit() produces,
    # so index K-1 really is the state the engine will call 'bull_quiet'.
    scores = means[:, 0] - 0.5 * stds[:, 0]
    assert np.all(np.diff(scores) > 0), "test setup: states must be score-ascending"

    labels = GaussianHMM._label_states(means, stds, K)

    top_idx = K - 1
    top_mean = means[top_idx, 0]
    assert top_mean < 0.0, "test setup: top-ranked state must have a negative mean"

    # CORRECT behavior: a state with a NEGATIVE mean return must NOT be labeled
    # 'bull_quiet'. The engine returns ["crisis","bear","normal","bull_quiet"]
    # unconditionally, so labels[top_idx] == "bull_quiet" and this FAILS.
    assert labels[top_idx] != "bull_quiet", (
        f"State {top_idx} has a NEGATIVE mean ({top_mean:+.4f}) but was labeled "
        f"'{labels[top_idx]}'. A down regime must never be labeled 'bull_quiet'. "
        f"Full labels: {labels}"
    )


# ---------------------------------------------------------------------------
# Test 2 — full engine path: fit() on a synthetic steady low-vol DECLINE, then
# check the position multiplier for the current (down) regime. It must be
# <= 1.0, never the 1.25x up-size 'bull_quiet' produces.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="held finding F3-hmm-bull-quiet; drop xfail when fixed", strict=True
)
def test_down_regime_position_multiplier_not_upsized():
    # Build a steady low-volatility DECLINE with four distinct sub-regimes, all
    # DOWN, laid out so the tail (the "current" regime) is the least-negative,
    # lowest-vol cluster — i.e. the state the engine ranks top and mislabels
    # 'bull_quiet'. The drift dominates the vol so every day is a down day
    # (fraction positive == 0), which forces EM to fit four states that ALL have
    # negative mean returns (a plain Gaussian around a small negative drift would
    # otherwise carve out a spurious positive-mean up-day cluster). Deterministic
    # RNG so the fit is reproducible; std >> the engine's 1e-7 degenerate guard
    # and the series is finite, so fit() proceeds normally.
    rng = np.random.default_rng(7)
    n = 480
    idx = np.arange(n)
    # Progressively less-steep, lower-vol declines; the final block (tail) is the
    # least-negative + lowest-vol regime -> engine's top-ranked 'bull_quiet'.
    drift = np.select(
        [idx < 120, idx < 240, idx < 360],
        [-0.0060, -0.0045, -0.0030],
        default=-0.0015,
    )
    vol = np.select(
        [idx < 120, idx < 240, idx < 360],
        [0.0016, 0.0011, 0.0007],
        default=0.0004,
    )
    log_returns = drift + rng.normal(0.0, 1.0, n) * vol
    assert (log_returns > 0).mean() == 0.0, "test setup: expected a pure decline"

    # Two features: (log_return, rolling realised vol) — matches how the engine
    # is fed in practice. Rolling std over a short window; fill the head.
    win = 10
    rv = np.array(
        [log_returns[max(0, i - win) : i + 1].std() for i in range(n)]
    )
    rv[:win] = rv[win]
    obs = np.column_stack([log_returns, rv])

    hmm = GaussianHMM(n_states=4, random_state=42)
    fit = hmm.fit(obs)

    # Every fitted state's mean return should be negative on a pure decline —
    # confirm the fixture actually produced a down regime (not a knife-edge).
    assert np.all(fit.means[:, 0] < 0.0), (
        f"test setup: expected all state means negative on a decline, got "
        f"{fit.means[:, 0]}"
    )

    # Current filtered posterior over states.
    probs = hmm.predict_proba(obs)[-1]
    mult = hmm.position_multiplier(probs)

    # CORRECT behavior: no state in an all-negative-mean (down) regime should
    # carry a >1.0 weight, so the blended multiplier must be <= 1.0. The engine
    # labels the least-negative state 'bull_quiet' (weight 1.25), so whenever the
    # current posterior puts meaningful mass on that state the multiplier exceeds
    # 1.0 and this FAILS.
    assert mult <= 1.0 + 1e-9, (
        f"Down regime (all state means negative: {fit.means[:, 0]}) produced a "
        f"position multiplier of {mult:.4f} > 1.0 — the engine up-sized on a "
        f"decline because it labeled the least-negative state 'bull_quiet'. "
        f"State labels: {fit.state_labels}; posterior: {probs}"
    )
