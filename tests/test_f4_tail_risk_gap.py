"""Regression tests pinning the F4 tail-risk gap (PR #178 / PR #184).

The S22 backtest (`docs/ENGINE_BACKTEST_2022_2024.md`, PR #178) flagged the
single highest-leverage engineering gap in the current decision layer:
``WheelRunner.rank_candidates_by_ev`` produced ``+EV`` and
``prob_profit ≈ 0.83`` for short puts on COST entered April 2022 and UNH
entered November 2024, despite each subsequently realising a 19-24% drop
over the 35-day holding period. PR #184 (S27 IV-PIT re-run) confirmed
the gap is real (same ``prob_profit=0.833`` and same -$7,500 mean
realized loss on both pre-fix and post-fix engines).

Two hypotheses from PR #178:

* **H1 — 504-day empirical lookback dilution.** ``empirical_forward_log_returns``
  defaults to ``lookback_years=5.0`` (~1260 trading days, well over the
  504 cited in the doc). A 30-day vol spike at the end of a multi-year
  history is averaged out by the much larger volume of low-vol windows.
* **H2 — POT-GPD threshold too high.** ``fit_gpd_tail`` uses the 95th
  percentile of losses as the GPD threshold; mild-persistent tail
  events (15-25% drops over 35 days, recurring once a year or so) sit
  in the body of the empirical distribution rather than the extreme
  tail, so the shape parameter ``xi`` doesn't exceed the 0.3
  heavy-tail flag threshold.

What this file pins:

1. The two real-world failure modes — COST 2022-04-04 (35-day realized
   -23.89%) and UNH 2024-11-11 (-19.31%) — surface on the production
   path: tail metrics on the ranker's diagnostic row don't widen to
   reflect the realized drops.
2. The H1 dilution mechanism in isolation, using synthetic two-regime
   OHLCV so the failure is reproducible without depending on specific
   ticker history.
3. The fix direction (shorter lookback captures recent vol) is
   demonstrated by the comparison-pair test.
4. The ``heavy_tail`` flag's current False-on-realized-heavy-tail
   behavior is tracked via :py:func:`pytest.mark.xfail` (``strict=False``)
   so a future fix that flips the flag to True will register as an
   ``XPASS`` without breaking CI.

Tests are read-only on the decision layer (board #113 rule 3); the F4
gap is documented here, not fixed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.forward_distribution import best_available_forward_distribution
from engine.wheel_runner import WheelRunner

_COST_AS_OF = "2022-04-04"  # 35-day realized: -23.89% (verified from CSV)
_UNH_AS_OF = "2024-11-11"  # 35-day realized: -19.31% (verified from CSV)


# ----------------------------------------------------------------------
# Shared fixtures — one ranker call per ticker, reused across tests
# ----------------------------------------------------------------------
def _real_ranker_row(ticker: str, as_of: str) -> dict:
    """Pull the top row from a real ``rank_candidates_by_ev`` call.

    Uses the real :class:`MarketDataConnector` + the real Bloomberg
    CSVs. Tests assert on the row's diagnostic fields directly — the
    ranker is the production entry point that drives the full
    ``forward_distribution`` + ``EVEngine.evaluate`` + POT-GPD pipeline
    F4 implicates.
    """
    runner = WheelRunner()
    df = runner.rank_candidates_by_ev(
        tickers=[ticker],
        top_n=1,
        as_of=as_of,
        min_ev_dollars=-1e9,  # keep -EV rows so the test isn't gated on engine verdict
        include_diagnostic_fields=True,
    )
    assert not df.empty, (
        f"expected a row for {ticker} at as_of={as_of}; drops={df.attrs.get('drops', [])}"
    )
    return df.iloc[0].to_dict()


@pytest.fixture(scope="module")
def cost_2022_04_row() -> dict:
    return _real_ranker_row("COST", _COST_AS_OF)


@pytest.fixture(scope="module")
def unh_2024_11_row() -> dict:
    return _real_ranker_row("UNH", _UNH_AS_OF)


@pytest.fixture(scope="module")
def two_regime_ohlcv() -> pd.DataFrame:
    """Synthetic 534-day OHLCV: 504 low-vol days + 30 high-vol days.

    Designed to isolate H1: the 30-day vol spike at the tail of a
    long history is the smaller sample, so the forward distribution
    is dominated by the 504-day low-vol regime.
    """
    rng = np.random.default_rng(0)
    low_vol = rng.normal(0.0003, 0.005, 504)  # ~daily 0.5% vol
    high_vol = rng.normal(-0.002, 0.04, 30)  # ~daily 4% vol — the spike
    log_returns = np.concatenate([low_vol, high_vol])
    log_prices = np.cumsum(log_returns)
    prices = 100.0 * np.exp(log_prices)
    idx = pd.date_range("2022-01-03", periods=534, freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


# ======================================================================
# 1-3. Real-world F4 cases — COST 2022-04 + UNH 2024-11
# ======================================================================
class TestF4RealWorldGapOnProductionRanker:
    """Replay the two cases PR #178 named as F4. Today's engine produces
    high prob_profit and narrow tail metrics on these dates; tomorrow's
    fix should flip the failing assertions in this class.
    """

    def test_cost_2022_04_forward_distribution_does_not_widen_for_tail(
        self, cost_2022_04_row: dict
    ):
        """The forward distribution's CVaR-5 (worst-5% conditional loss)
        does not reach the realized -23.89% drop. Pins F4's "forward
        distribution did not widen" claim — H1 evidence on real data.
        """
        spot = float(cost_2022_04_row["spot"])
        cvar_5 = float(cost_2022_04_row["cvar_5"])
        # cvar_5 is a signed P&L number (negative = loss in dollars on
        # the position, not per-share log return). Normalise to a
        # fraction-of-collateral so the threshold is interpretable.
        # Collateral for one short put = strike * 100.
        collateral = float(cost_2022_04_row["strike"]) * 100.0
        cvar_5_pct_of_collateral = cvar_5 / collateral
        # Engine's worst-5% loss is well shy of the realized 23.89%
        # drop in spot. F4 expectation: not nearly enough.
        assert cvar_5_pct_of_collateral > -0.10, (
            f"COST 2022-04 cvar_5 = ${cvar_5:.2f} = "
            f"{cvar_5_pct_of_collateral * 100:.2f}% of collateral. "
            f"F4 narrative: the engine's 5%-CVaR should not have reached "
            f"the realized -23.89% drop. If this assertion now fails, "
            f"the engine likely was fixed — flip the test to assert "
            f"cvar_5_pct_of_collateral < -0.15 instead. spot={spot:.2f}"
        )

    def test_cost_2022_04_pot_gpd_does_not_flag_heavy_tail(self, cost_2022_04_row: dict):
        """POT-GPD shape parameter ``xi`` stays at-or-below the 0.3
        heavy-tail threshold (or is unpopulated, which is an even
        stronger F4 signal — the fit may not have fired at all).
        Pins F4's "POT-GPD threshold too high" claim — H2 evidence on
        real data."""
        tail_xi = cost_2022_04_row["tail_xi"]
        # The row converts NaN floats to None in some pandas->dict
        # paths, so handle both. If ``tail_xi`` is None or NaN, the
        # GPD fit never produced a finite shape parameter — that's an
        # even stronger version of the F4 finding ("POT-GPD didn't fit
        # the tail at all") so the test passes trivially.
        if tail_xi is None or (isinstance(tail_xi, float) and math.isnan(tail_xi)):
            # F4 strengthened: not just below threshold, but no fit.
            return
        tail_xi_f = float(tail_xi)
        assert tail_xi_f <= 0.3, (
            f"COST 2022-04 tail_xi = {tail_xi_f:.4f}. F4 narrative: xi "
            f"should not exceed the 0.3 heavy-tail threshold despite "
            f"the realized -23.89% drop. If this now fails, the "
            f"POT-GPD threshold may have been lowered — verify the fix."
        )

    def test_unh_2024_11_same_gap_shape(self, unh_2024_11_row: dict):
        """UNH November 2024 (realized -19.31%): same combined shape as
        COST tests above. Cross-ticker replication of the F4 finding."""
        spot = float(unh_2024_11_row["spot"])
        cvar_5 = float(unh_2024_11_row["cvar_5"])
        collateral = float(unh_2024_11_row["strike"]) * 100.0
        cvar_5_pct_of_collateral = cvar_5 / collateral
        heavy_tail = bool(unh_2024_11_row["heavy_tail"])
        assert cvar_5_pct_of_collateral > -0.10, (
            f"UNH 2024-11 cvar_5 = ${cvar_5:.2f} = "
            f"{cvar_5_pct_of_collateral * 100:.2f}% of collateral. "
            f"F4 narrative: engine's 5%-CVaR doesn't reach the realized "
            f"-19.31% drop. spot={spot:.2f}"
        )
        assert heavy_tail is False, (
            f"UNH 2024-11 heavy_tail = {heavy_tail}. F4 narrative: flag "
            f"stays False despite the realized 19.31% drop."
        )


# ======================================================================
# 4-5. Synthetic isolation — H1 lookback dilution + fix direction
# ======================================================================
class TestF4LookbackDilutionMechanism:
    """Demonstrate H1 (lookback dilution) reproducibly using synthetic
    OHLCV. Tests 4+5 form a comparison pair: full 504-day lookback
    dilutes the recent vol spike; truncating to a 60-day window
    captures it. Production currently uses the 5-year default, putting
    real tickers in the test-4 regime.
    """

    def test_504d_lookback_with_recent_vol_spike_does_not_widen_forward_dist(
        self, two_regime_ohlcv: pd.DataFrame
    ):
        """504 low-vol + 30 high-vol → forward distribution std is
        closer to the pure low-vol expected std than the high-vol's.
        H1 in isolation."""
        arr, label = best_available_forward_distribution(two_regime_ohlcv, horizon_days=35)
        assert len(arr) > 0, f"forward distribution returned empty; label={label}"
        std = float(np.std(arr))
        # Expected std under pure low-vol regime: sigma_daily * sqrt(35)
        # = 0.005 * sqrt(35) ≈ 0.0296.
        # Expected std under pure high-vol regime: 0.04 * sqrt(35) ≈ 0.237.
        # H1 says: 30 recent high-vol days don't widen materially.
        # Threshold of 0.05 sits cleanly between the two regimes; F4
        # gap means std stays below.
        assert std < 0.05, (
            f"forward dist std = {std:.4f} (label={label}). Expected H1 "
            f"behavior: std stays closer to pure low-vol "
            f"(~0.030) than pure high-vol (~0.237). If std now exceeds "
            f"0.05, the engine may have been re-weighted toward recent "
            f"history — verify the lookback fix."
        )

    def test_60d_lookback_captures_recent_vol_spike(self, two_regime_ohlcv: pd.DataFrame):
        """Same two-regime OHLCV but truncated to the last 60 days
        (30 low-vol + 30 high-vol). Forward distribution std widens
        materially. Demonstrates the fix direction.
        """
        truncated = two_regime_ohlcv.iloc[-60:]
        arr_short, label_short = best_available_forward_distribution(truncated, horizon_days=35)
        assert len(arr_short) > 0, (
            f"short-lookback forward distribution returned empty; label={label_short}"
        )
        std_short = float(np.std(arr_short))

        # Same baseline call as test 4 for the comparison.
        arr_long, _ = best_available_forward_distribution(two_regime_ohlcv, horizon_days=35)
        std_long = float(np.std(arr_long))

        assert std_short > 2 * std_long, (
            f"short-lookback std {std_short:.4f} should be > 2x full-lookback "
            f"std {std_long:.4f}; short-lookback captures the recent "
            f"vol spike, full-lookback dilutes it. Pins the H1 fix "
            f"direction."
        )


# ======================================================================
# 6. Regression-watch (xfail strict=False) — heavy_tail should fire
# ======================================================================
class TestF4HeavyTailFlagShouldFireOnRealizedHeavyTailCases:
    """When the engine is fixed (forward dist widens OR POT-GPD
    threshold lowered OR both), ``heavy_tail`` should fire True on the
    COST 2022-04 and UNH 2024-11 entries — both realized 19-24% drops
    are heavy-tail events by any reasonable definition.

    Today the flag is False on both. Test 6 is :py:func:`pytest.mark.xfail`
    with ``strict=False`` so:

    * **Today (gap unfixed):** test runs, assertion fails, recorded as
      XFAIL, suite stays green.
    * **Post-fix (flag flips):** test runs, assertion passes, recorded
      as XPASS in the report. ``strict=False`` avoids CI red during
      the transition window of a partial fix.

    After the F4 fix lands and stabilises, tighten to ``strict=True``
    so any subsequent regression that re-breaks the flag fails the
    build.
    """

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "F4 known gap from PR #178 / #184: engine does not fire "
            "heavy_tail on 19-24% realized drops (COST 2022-04, UNH "
            "2024-11). Tests track until the forward-distribution + "
            "POT-GPD pipeline is fixed."
        ),
    )
    def test_cost_2022_04_heavy_tail_should_be_true(self, cost_2022_04_row: dict):
        assert bool(cost_2022_04_row["heavy_tail"]) is True

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "F4 known gap from PR #178 / #184: same as COST test above, cross-ticker replication."
        ),
    )
    def test_unh_2024_11_heavy_tail_should_be_true(self, unh_2024_11_row: dict):
        assert bool(unh_2024_11_row["heavy_tail"]) is True
