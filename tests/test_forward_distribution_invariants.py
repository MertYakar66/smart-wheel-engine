"""Forward-distribution cascade + realized-vol invariants — quant-layer test audit
round 2 (W38-W43).

The 2026-06-09 data-test audit (W14-W37, ``test_data_to_engine.py`` /
``test_data_integrity_bloomberg.py``) covered DATA accessors -> ranker OUTPUT. This
round pins the **math in between**: the empirical-non-overlapping -> overlapping ->
block-bootstrap -> HAR-RV cascade (``engine/forward_distribution.py``) that produces
the forward log-returns ``EVEngine.evaluate`` integrates over, plus the realized-vol
estimators (``engine/realized_vol.py``).

Behaviour-pinning (the #366 lesson): every test asserts real numeric behaviour, never
a shape/membership proxy. No §2 surface is touched — these assert the EV integrand's
documented contracts, never weaken them. A confirmed-incomplete guard is pinned with
``xfail(strict=True)`` asserting the DESIRED behaviour (flips when the engine fix lands),
never asserting the buggy output as correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.forward_distribution import (
    best_available_forward_distribution,
    block_bootstrap_log_returns,
    empirical_forward_log_returns,
    har_rv_conditional_distribution,
)
from engine.realized_vol import (
    close_to_close_vol,
    garman_klass_vol,
    parkinson_vol,
    rogers_satchell_vol,
    yang_zhang_vol,
)

# Pinned, deterministic frontier (matches the data-test suite; no today()).
FRONTIER = "2026-06-04"

# Column order used by _series + the iloc poisoning below.
_OPEN, _HIGH, _LOW, _CLOSE, _VOLUME = 0, 1, 2, 3, 4


def _series(n_bars: int, sigma: float = 0.012, seed: int = 0, end: str = FRONTIER) -> pd.DataFrame:
    """``n_bars`` business-day OHLCV ending at ``end``, GBM close, VALID OHLC bars
    (high >= max(open, close), low <= min(open, close)) so every estimator is well
    defined on the clean control."""
    dates = pd.bdate_range(end=end, periods=n_bars)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n_bars)))
    openp = close * (1.0 + rng.normal(0.0, sigma / 4, n_bars))
    high = np.maximum(openp, close) * (1.0 + np.abs(rng.normal(0.0, sigma / 4, n_bars)))
    low = np.minimum(openp, close) * (1.0 - np.abs(rng.normal(0.0, sigma / 4, n_bars)))
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=dates,
    )


# ---------------------------------------------------------------------------
# W38 — cascade tier SELECTION (not just membership). The only prior cascade test
# (test_audit_improvements.py::test_cascading_fallback_picks_best) asserts
# ``method in (4-tuple)`` — a false-green that passes for ANY tier. Pin the EXACT
# tier the documented precedence (forward_distribution.py:342-386) returns at each
# history depth: a regression that skipped the statistically-honest empirical tiers
# (and silently flipped is_iid_forward_source / the prob_profit Wilson-CI honesty)
# would survive the membership check but fail here.
# ---------------------------------------------------------------------------


class TestCascadeTierSelection:
    def test_deep_history_selects_non_overlapping(self):
        rets, tier = best_available_forward_distribution(
            _series(600), horizon_days=20, as_of=FRONTIER
        )
        assert tier == "empirical_non_overlapping"
        assert len(rets) >= 20

    def test_medium_history_selects_overlapping(self):
        # NOS count (9) < min_empirical_samples (20); overlapping count (180) >= 60.
        rets, tier = best_available_forward_distribution(
            _series(200), horizon_days=20, as_of=FRONTIER
        )
        assert tier == "empirical_overlapping"
        assert len(rets) > 0

    def test_short_history_selects_block_bootstrap(self):
        # h=55, N=112: NOS (2) and overlapping (57 < 60) both fail; block_bootstrap
        # fires because len(prices)=112 >= max(2*h, 100)=110 and n_rets=111 >= h+5.
        rets, tier = best_available_forward_distribution(
            _series(112), horizon_days=55, as_of=FRONTIER
        )
        assert tier == "block_bootstrap"
        assert len(rets) > 0

    def test_tiny_history_selects_har_rv(self):
        # h=20, N=70: NOS/overlapping fail AND block_bootstrap fails (needs >=100
        # prices); har_rv fires (needs >=60 prices, len(y)>=30).
        rets, tier = best_available_forward_distribution(
            _series(70), horizon_days=20, as_of=FRONTIER
        )
        assert tier == "har_rv"
        assert len(rets) > 0

    def test_empty_history_terminal_none(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rets, tier = best_available_forward_distribution(empty, horizon_days=20, as_of=FRONTIER)
        assert tier == "none"
        assert len(rets) == 0


# ---------------------------------------------------------------------------
# W39 — sampler determinism. block_bootstrap_log_returns and
# har_rv_conditional_distribution feed EVEngine.evaluate; a non-deterministic draw
# would make ev_dollars/prob_profit non-reproducible across identical ranker calls
# (and break the backtest fingerprint determinism gate). Same seed -> identical
# array; a DIFFERENT seed -> different array (so the test pins seed plumbing, not a
# frozen constant). Only the monte_carlo.BlockBootstrap class had a determinism
# test; these forward_distribution samplers on the EV path did not.
# ---------------------------------------------------------------------------


class TestSamplerDeterminism:
    def test_block_bootstrap_same_seed_reproducible(self):
        df = _series(600)
        a = block_bootstrap_log_returns(df, horizon_days=20, as_of=FRONTIER)
        b = block_bootstrap_log_returns(df, horizon_days=20, as_of=FRONTIER)
        np.testing.assert_array_equal(a, b)
        c = block_bootstrap_log_returns(df, horizon_days=20, as_of=FRONTIER, seed=99)
        assert not np.array_equal(a, c), "different seed must change the draw (seed is plumbed)"

    def test_har_rv_same_seed_reproducible(self):
        df = _series(600)
        a = har_rv_conditional_distribution(df, horizon_days=20, as_of=FRONTIER)
        b = har_rv_conditional_distribution(df, horizon_days=20, as_of=FRONTIER)
        np.testing.assert_array_equal(a, b)
        c = har_rv_conditional_distribution(df, horizon_days=20, as_of=FRONTIER, seed=99)
        assert not np.array_equal(a, c), "different seed must change the draw (seed is plumbed)"


# ---------------------------------------------------------------------------
# W41 — empirical_forward_log_returns boundary + finite-filter behaviour. The
# min_samples / horizon guards and the np.isfinite filter (forward_distribution.py
# :123/:133/:135) gate whether the EV engine gets an empirical distribution at all;
# none were pinned at the boundary.
# ---------------------------------------------------------------------------


class TestEmpiricalBoundary:
    def test_exact_min_samples_boundary(self):
        # h=5, N=101 -> exactly 20 non-overlapping forward returns.
        df = _series(101)
        at = empirical_forward_log_returns(df, horizon_days=5, as_of=FRONTIER, min_samples=20)
        assert len(at) == 20, "exactly min_samples should pass the gate"
        below = empirical_forward_log_returns(df, horizon_days=5, as_of=FRONTIER, min_samples=21)
        assert len(below) == 0, "min_samples-1 short of the count must return empty (fall back)"

    def test_n_le_horizon_returns_empty(self):
        # n == horizon_days -> empty (the n<=horizon guard); n == horizon_days+1 ->
        # one return survives with min_samples=1.
        assert (
            len(
                empirical_forward_log_returns(
                    _series(5), horizon_days=5, as_of=FRONTIER, min_samples=1
                )
            )
            == 0
        )
        assert (
            len(
                empirical_forward_log_returns(
                    _series(6), horizon_days=5, as_of=FRONTIER, min_samples=1
                )
            )
            == 1
        )

    def test_nonfinite_returns_are_filtered_not_passed_to_ev(self):
        # A zero close mid-series makes several forward log-returns +/-inf; the
        # isfinite filter must drop them so EV never integrates an inf.
        df = _series(300)
        df.iloc[150, _CLOSE] = 0.0
        rets = empirical_forward_log_returns(
            df, horizon_days=5, as_of=FRONTIER, min_samples=1, non_overlapping=False
        )
        assert len(rets) > 0
        assert np.all(np.isfinite(rets)), (
            "isfinite filter must drop the inf returns the zero price induced"
        )


# ---------------------------------------------------------------------------
# W40 — realized_vol._log non-positive guard. CLEAN cases are (T) and pinned here;
# the INCOMPLETE-guard cases (a zero DENOMINATOR leaks +inf through the OHLC-ratio
# estimators; an all-negative bar is silently swallowed to ~0) are pinned with
# xfail(strict) asserting the DESIRED NaN, and tracked as an (E) engine fix (harden
# _log to clamp raw prices / handle zero denominators). NOTE: the connector's
# OHLCV-positivity integrity tests prevent this input upstream today, so the leak is
# latent/defensive — but the guard's documented "non-negativity enforced" contract
# is not actually met for the ratio estimators.
# ---------------------------------------------------------------------------

_ALL_ESTIMATORS = [
    close_to_close_vol,
    parkinson_vol,
    garman_klass_vol,
    rogers_satchell_vol,
    yang_zhang_vol,
]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # degenerate inputs intentionally trip numpy
class TestRealizedVolNonPositiveGuard:
    @pytest.mark.parametrize("est", _ALL_ESTIMATORS, ids=lambda f: f.__name__)
    def test_fully_zero_bar_yields_nan_sentinel(self, est):
        # A fully-degenerate (all-OHLC-zero) bar -> NaN for every estimator (the
        # honest sentinel), never a finite fabricated vol and never inf.
        df = _series(40)
        df.iloc[20, [_OPEN, _HIGH, _LOW, _CLOSE]] = 0.0
        v = est(df, window=20)
        assert np.isnan(v), f"{est.__name__} should be NaN on a fully-zero bar, got {v}"

    def test_close_to_close_negative_close_is_nan(self):
        # close_to_close logs the RAW close, so the _log(<=0)->NaN guard fully
        # protects it: a negative close yields NaN, not a swallowed/inf value.
        df = _series(40)
        df.iloc[20, _CLOSE] = -1.0
        assert np.isnan(close_to_close_vol(df, window=20))

    @pytest.mark.xfail(
        reason="(E) #382 incomplete _log guard: zero DENOMINATOR leaks +inf through the OHLC-ratio estimators; should be NaN. Tracked as an engine fix.",
        strict=True,
    )
    @pytest.mark.parametrize("est", [parkinson_vol, garman_klass_vol], ids=lambda f: f.__name__)
    def test_zero_low_should_not_leak_inf(self, est):
        # DESIRED contract: a non-positive price anywhere (here low=0, a denominator)
        # yields NaN, never +inf. TODAY parkinson/garman_klass return +inf because the
        # guard sits on the post-division ratio, not the raw price. Flips green when
        # the (E) guard hardening lands.
        df = _series(40)
        df.iloc[20, _LOW] = 0.0
        v = est(df, window=20)
        assert not np.isinf(v), f"{est.__name__} leaked inf on a zero-low bar"


# ---------------------------------------------------------------------------
# W42 — variance-floor on degenerate bars. garman_klass / rogers_satchell apply
# max(var, 0.0) before sqrt (realized_vol.py:72/:86) so a degenerate bar that drives
# the variance estimate negative yields a real (finite, non-negative) vol instead of
# NaN-from-sqrt-of-negative. Never exercised: the only tests feed well-behaved data
# where the variance is comfortably positive.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # degenerate inputs intentionally trip numpy
class TestVarianceFloor:
    def test_floor_engages_on_negative_variance_bars(self):
        # high == low == close (zero intraday range) with a gapped open drives the GK
        # variance term negative and the RS term to 0; the floor must yield a finite
        # >= 0 vol (here exactly 0.0), not NaN.
        df = _series(40)
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["open"] = df["close"] * 1.05
        gk = garman_klass_vol(df, window=20)
        rs = rogers_satchell_vol(df, window=20)
        yz = yang_zhang_vol(df, window=20)
        assert np.isfinite(gk) and gk >= 0.0 and gk == 0.0, f"GK floor should yield 0.0, got {gk}"
        assert np.isfinite(rs) and rs >= 0.0 and rs == 0.0, f"RS floor should yield 0.0, got {rs}"
        assert np.isfinite(yz) and yz >= 0.0, f"YZ should stay finite >= 0, got {yz}"
