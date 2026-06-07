"""Small-sample honesty for ``prob_profit`` (2026-06-01).

``EVResult.prob_profit`` is a ``k / N`` binomial frequency over the
forward-scenario set — and on the default empirical non-overlapping path
``N`` is only ~30-35 (35 at the typical 35-DTE). Reported to 4 decimals
with no ``N`` and no interval, it reads as exact precision when its true
95% interval is ~20 percentage points wide (e.g. ``30/35 = 0.857`` →
Wilson 95% ``[0.706, 0.937]``). This is the engine's headline
reliability gap (heavy-verify 2026-05-31, "where you cannot trust it").

The reliable move — since making the estimate *more accurate*
(recalibration) is GATED on a leave-one-crisis-out test it does not pass
— is to make its UNCERTAINTY honest. So the engine now surfaces, as
ADDITIVE fields, the sample size ``n_scenarios`` and the Wilson 95% CI
``(prob_profit_ci_low, prob_profit_ci_high)`` on :class:`EVResult` and as
columns on the ranker frame. ``prob_profit`` itself is unchanged.

Pinned here:
  1. The Wilson math is correct (known cells, edge cases, clamping).
  2. ``EVResult`` carries ``n_scenarios`` == the forward sample size and a
     CI that brackets ``prob_profit`` and stays in ``[0, 1]``.
  3. The fix is ADDITIVE: ``prob_profit`` equals the raw ``k / N`` frequency
     (the CI annotates it, never moves it).
  4. The ranker frame emits the three columns, in the CORE row (present
     even when ``include_diagnostic_fields=False``), and the CI brackets
     ``prob_profit`` on every survivor row.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.ev_engine import EVEngine, ShortOptionTrade, _wilson_score_interval
from engine.forward_distribution import is_iid_forward_source


# ======================================================================
# 1. Wilson score interval — pure math
# ======================================================================
class TestWilsonScoreInterval:
    def test_known_cell_30_of_35(self):
        lo, hi = _wilson_score_interval(30, 35)
        # Canonical Wilson 95% for 30/35 ≈ [0.706, 0.937].
        assert lo == pytest.approx(0.706, abs=0.01)
        assert hi == pytest.approx(0.937, abs=0.01)

    def test_known_cell_29_of_35(self):
        lo, hi = _wilson_score_interval(29, 35)
        assert lo == pytest.approx(0.673, abs=0.01)
        assert hi == pytest.approx(0.919, abs=0.01)

    def test_interval_brackets_point_estimate(self):
        for k, n in [(1, 4), (5, 10), (17, 20), (30, 35), (50, 100)]:
            lo, hi = _wilson_score_interval(k, n)
            p = k / n
            assert lo <= p <= hi, f"{k}/{n}: {lo} <= {p} <= {hi} failed"
            assert 0.0 <= lo <= hi <= 1.0

    def test_extremes_are_clamped(self):
        # p = 1 (all wins): high clamps at 1.0, low strictly < 1.
        lo, hi = _wilson_score_interval(35, 35)
        assert hi <= 1.0
        assert lo < 1.0
        # p = 0 (no wins): low clamps at 0.0, high strictly > 0.
        lo0, hi0 = _wilson_score_interval(0, 35)
        assert lo0 >= 0.0
        assert hi0 > 0.0

    def test_zero_n_is_nan(self):
        lo, hi = _wilson_score_interval(0, 0)
        assert math.isnan(lo) and math.isnan(hi)
        lo2, hi2 = _wilson_score_interval(0, -5)
        assert math.isnan(lo2) and math.isnan(hi2)

    def test_smaller_n_widens_the_interval(self):
        """Honesty property: the SAME proportion at a smaller sample is a
        wider interval (less certain)."""
        lo_small, hi_small = _wilson_score_interval(6, 7)  # ~0.857, N=7
        lo_big, hi_big = _wilson_score_interval(60, 70)  # ~0.857, N=70
        assert (hi_small - lo_small) > (hi_big - lo_big)


# ======================================================================
# 2/3. EVResult surfaces N + CI; prob_profit unchanged (additive)
# ======================================================================
def _put(spot: float = 100.0, strike: float = 95.0, dte: int = 35) -> ShortOptionTrade:
    return ShortOptionTrade(
        option_type="put",
        underlying="TEST",
        spot=spot,
        strike=strike,
        premium=2.0,
        dte=dte,
        iv=0.25,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        contracts=1,
        bid=1.9,
        ask=2.1,
        open_interest=1000,
        regime_multiplier=1.0,
    )


class TestEVResultCarriesCI:
    def test_n_scenarios_matches_forward_sample_size(self):
        # 40 controlled forward log-returns → n_scenarios == 40.
        rng = np.random.default_rng(7)
        flr = rng.normal(0.0003, 0.02, 40)
        res = EVEngine().evaluate(_put(), forward_log_returns=flr)
        assert res.n_scenarios == 40

    def test_ci_brackets_prob_profit_and_in_unit_interval(self):
        rng = np.random.default_rng(11)
        flr = rng.normal(0.0005, 0.018, 35)
        res = EVEngine().evaluate(_put(), forward_log_returns=flr)
        assert 0.0 <= res.prob_profit_ci_low <= res.prob_profit <= res.prob_profit_ci_high <= 1.0

    def test_ci_equals_wilson_of_the_same_k_n(self):
        """The surfaced CI is exactly the Wilson interval of the engine's
        own (k, N) — k = round(prob_profit * N)."""
        rng = np.random.default_rng(3)
        flr = rng.normal(0.0, 0.02, 50)
        res = EVEngine().evaluate(_put(), forward_log_returns=flr)
        k = round(res.prob_profit * res.n_scenarios)
        lo, hi = _wilson_score_interval(k, res.n_scenarios)
        assert res.prob_profit_ci_low == pytest.approx(lo, abs=1e-9)
        assert res.prob_profit_ci_high == pytest.approx(hi, abs=1e-9)

    def test_prob_profit_is_unchanged_raw_frequency(self):
        """Additive guarantee: prob_profit is still the raw k/N frequency;
        the CI annotates it, never moves it. k is recoverable as
        round(prob_profit * N) and must reproduce prob_profit exactly."""
        rng = np.random.default_rng(5)
        flr = rng.normal(0.001, 0.02, 35)
        res = EVEngine().evaluate(_put(), forward_log_returns=flr)
        k = round(res.prob_profit * res.n_scenarios)
        assert res.prob_profit == pytest.approx(k / res.n_scenarios, abs=1e-9)


# ======================================================================
# 4. Ranker frame emits the columns (CORE — present without diagnostics)
# ======================================================================
_AS_OF = "2026-03-20"
_CI_COLS = ["n_scenarios", "prob_profit_ci_low", "prob_profit_ci_high"]


class TestRankerEmitsCI:
    def test_columns_present_even_without_diagnostic_fields(self):
        """The CI travels with prob_profit in the CORE row, so it is
        present even on the lean (include_diagnostic_fields=False) path —
        the honesty annotation is never hidden behind a flag."""
        from engine.wheel_runner import WheelRunner

        df = WheelRunner().rank_candidates_by_ev(
            tickers=["AAPL", "MSFT"],
            top_n=5,
            min_ev_dollars=-1e9,
            as_of=_AS_OF,
            include_diagnostic_fields=False,
        )
        for col in ["prob_profit", *_CI_COLS]:
            assert col in df.columns, f"ranker frame missing {col!r}"

    def test_ci_brackets_prob_profit_on_survivor_rows(self):
        from engine.wheel_runner import WheelRunner

        df = WheelRunner().rank_candidates_by_ev(
            tickers=["AAPL", "MSFT"],
            top_n=5,
            min_ev_dollars=-1e9,
            as_of=_AS_OF,
            include_diagnostic_fields=True,
        )
        if df.empty:
            pytest.skip(f"no survivor rows at as_of={_AS_OF} — likely all event-gated")
        # The put ranker targets a fixed ~35 DTE, whose trailing window
        # yields enough non-overlapping forward windows that survivor rows
        # are on the IID empirical_non_overlapping tier — so the CI is
        # emitted and brackets prob_profit. (The tier gate itself is
        # exercised directly in section 5; here we only require that the
        # honest CI reaches the rows it should.)
        for _, r in df.iterrows():
            if r["distribution_source"] != "empirical_non_overlapping":
                continue  # non-IID rows carry a suppressed CI by design (D4)
            pp = r["prob_profit"]
            n = r["n_scenarios"]
            lo, hi = r["prob_profit_ci_low"], r["prob_profit_ci_high"]
            # Survivor rows were evaluated → real N and a real bracket.
            assert n is not None and n > 0
            assert lo is not None and hi is not None
            assert 0.0 <= lo <= pp <= hi <= 1.0, f"{r['ticker']}: {lo} <= {pp} <= {hi} failed"


# ======================================================================
# 5. Tier gate — the Wilson CI is honest ONLY on the IID forward tier (D4)
# ======================================================================
# prob_profit's Wilson CI is a binomial interval; it is an honest *sampling*
# spread only when n_scenarios is a count of INDEPENDENT trials — the
# empirical non-overlapping tier. The overlapping (autocorrelated),
# block_bootstrap / har_rv (synthetic, n~5000) and lognormal_fallback
# (n~20000) tiers report an N that is NOT an independent-trial count, so a
# Wilson CI over them is deceptively tight (false precision). The rankers gate
# CI emission on engine.forward_distribution.is_iid_forward_source so the only
# interval a trader ever sees is a genuine sampling spread.
class TestIsIidForwardSource:
    def test_only_non_overlapping_is_iid(self):
        assert is_iid_forward_source("empirical_non_overlapping") is True

    def test_non_iid_tiers_are_rejected(self):
        for src in [
            "empirical_overlapping",
            "block_bootstrap",
            "har_rv",
            "lognormal_fallback",
            "price_scenarios",
            "none",
        ]:
            assert is_iid_forward_source(src) is False, src

    def test_none_and_unknown_are_rejected(self):
        assert is_iid_forward_source(None) is False
        assert is_iid_forward_source("") is False
        assert is_iid_forward_source("something_new") is False


class TestRankerGatesCiOffIidTier:
    """End-to-end: when the forward tier is NOT the IID non-overlapping one,
    the put ranker suppresses the CI bundle (n_scenarios + CI → None) while
    prob_profit itself is unchanged. The tier is forced by RELABELLING the
    cascade output (keeping the real returns, so candidates survive and
    prob_profit is identical); the ranker's function-scope import of
    ``best_available_forward_distribution`` picks up the patched attribute."""

    def _relabelled_df(self, monkeypatch, label):
        import engine.forward_distribution as fd

        real = fd.best_available_forward_distribution

        def fake(*a, **k):
            rets, _src = real(*a, **k)
            return rets, label

        monkeypatch.setattr(fd, "best_available_forward_distribution", fake)
        from engine.wheel_runner import WheelRunner

        return WheelRunner().rank_candidates_by_ev(
            tickers=["AAPL", "MSFT"],
            top_n=5,
            min_ev_dollars=-1e9,
            as_of=_AS_OF,
            include_diagnostic_fields=True,
        )

    def test_non_iid_tier_suppresses_ci_keeps_prob_profit(self, monkeypatch):
        df = self._relabelled_df(monkeypatch, "block_bootstrap")
        if df.empty:
            pytest.skip(f"no survivor rows at as_of={_AS_OF}")
        assert (df["distribution_source"] == "block_bootstrap").all()
        # prob_profit is unchanged: still a finite frequency in [0, 1].
        assert df["prob_profit"].notna().all()
        assert ((df["prob_profit"] >= 0.0) & (df["prob_profit"] <= 1.0)).all()
        # The CI bundle is suppressed → every gated column is null.
        for col in _CI_COLS:
            assert df[col].isna().all(), f"{col} must be null on a non-IID tier"

    def test_iid_tier_emits_ci(self, monkeypatch):
        # Control: forcing the IID label keeps the CI present + bracketing,
        # proving the suppression above is the tier gate, not a side effect.
        df = self._relabelled_df(monkeypatch, "empirical_non_overlapping")
        if df.empty:
            pytest.skip(f"no survivor rows at as_of={_AS_OF}")
        for col in _CI_COLS:
            assert df[col].notna().all(), f"{col} must be finite on the IID tier"
        assert (df["prob_profit_ci_low"] <= df["prob_profit"]).all()
        assert (df["prob_profit"] <= df["prob_profit_ci_high"]).all()
