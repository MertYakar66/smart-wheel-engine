"""Tests for the premium-correction pilot's split-adjustment layer.

The post-split pilot band (2024-09 onward) exercises only the identity path,
so the split layer's *non-trivial* behaviour must be validated against a
known historical split. AAPL's 4:1 forward split (ex-date 2020-08-31) is the
anchor: a raw pre-split strike of $300 must map to an adjusted strike of $75,
and the bogus join the pilot caught (adjusted $74 ↔ raw $75) must NOT recur.
"""

from __future__ import annotations

from datetime import date

import pytest

from studies.premium_correction.splits import (
    adjusted_to_raw_strike,
    cumulative_factor_after,
    raw_to_adjusted_price,
    raw_to_adjusted_strike,
)

# --------------------------------------------------------------------------
# AAPL 4:1, ex-date 2020-08-31 — the anchor split.
# --------------------------------------------------------------------------


def test_aapl_factor_before_split_is_four():
    # Any date strictly before the ex-date sees the 4:1 split ahead of it.
    assert cumulative_factor_after("AAPL", date(2020, 1, 8)) == 4.0
    assert cumulative_factor_after("AAPL", "2020-08-30") == 4.0


def test_aapl_factor_on_or_after_split_is_one():
    # Ex-date itself and anything after is post-split — identity.
    assert cumulative_factor_after("AAPL", date(2020, 8, 31)) == 1.0
    assert cumulative_factor_after("AAPL", "2024-12-01") == 1.0


def test_aapl_adjusted_strike_maps_to_raw_pre_split():
    # The exact bug the pilot caught: an adjusted ~$75 strike must map to the
    # raw ~$300 contract that actually listed on 2020-01-08, NOT to a raw $75.
    assert adjusted_to_raw_strike("AAPL", "2020-01-08", 75.0) == 300.0
    # And the inverse round-trips.
    assert raw_to_adjusted_strike("AAPL", "2020-01-08", 300.0) == 75.0


def test_aapl_premium_scales_with_split():
    # A raw pre-split premium is divided by the factor to reach adjusted space.
    assert raw_to_adjusted_price("AAPL", "2020-01-08", 4.0) == 1.0


# --------------------------------------------------------------------------
# TSLA 5:1 (2020-08-31) then 3:1 (2022-08-25) — cumulative factor.
# --------------------------------------------------------------------------


def test_tsla_cumulative_factor_compounds():
    # Before both splits: 5 * 3 = 15.
    assert cumulative_factor_after("TSLA", "2019-06-01") == 15.0
    # Between the two splits: only the 3:1 is still ahead.
    assert cumulative_factor_after("TSLA", "2021-01-01") == 3.0
    # After both: identity.
    assert cumulative_factor_after("TSLA", "2024-12-01") == 1.0


# --------------------------------------------------------------------------
# NVDA 4:1 (2021-07-20) then 10:1 (2024-06-10) — pilot band is post-both.
# --------------------------------------------------------------------------


def test_nvda_cumulative_factor():
    assert cumulative_factor_after("NVDA", "2020-01-01") == 40.0
    assert cumulative_factor_after("NVDA", "2022-01-01") == 10.0
    # The 10:1 ex-date is 2024-06-10, so the 2024-09 pilot band is identity.
    assert cumulative_factor_after("NVDA", "2024-09-01") == 1.0


# --------------------------------------------------------------------------
# Pilot-band invariant: every pilot name is identity across the band, so the
# harness join is split-free there (the reason the band was chosen).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", ["TSLA", "NVDA", "AAPL"])
@pytest.mark.parametrize("as_of", ["2024-09-01", "2025-06-15", "2026-03-01"])
def test_pilot_band_is_split_free(ticker, as_of):
    assert cumulative_factor_after(ticker, as_of) == 1.0
    assert adjusted_to_raw_strike(ticker, as_of, 123.5) == 123.5
    assert raw_to_adjusted_price(ticker, as_of, 4.25) == 4.25


def test_unknown_ticker_is_identity():
    assert cumulative_factor_after("ZZZZ", "2019-01-01") == 1.0


# --------------------------------------------------------------------------
# Refinement-2 axis: realized-outcome resolution (physical-vs-physical).
# The deliverable axis is engine-predicted vs *realized* assignment, NOT the
# risk-neutral-vs-physical wedge (Q − P) — see the review that drove this.
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.premium_correction.pilot import (  # noqa: E402
    MIN_BIN_N,
    MIN_CLUSTERS,
    _binned,
    _terminal_close,
    _wilson_ci,
)


def _synth_ohlcv():
    idx = pd.to_datetime(pd.date_range("2024-10-01", "2024-10-31", freq="B"))
    return pd.DataFrame({"close": np.linspace(220.0, 230.0, len(idx))}, index=idx)


def test_terminal_close_nearest_trading_day_on_or_before():
    o = _synth_ohlcv()
    # 2024-10-11 is a Friday (trading day) — exact match.
    assert _terminal_close(o, date(2024, 10, 11)) == pytest.approx(
        float(o.loc["2024-10-11", "close"])
    )


def test_terminal_close_backfills_weekend():
    o = _synth_ohlcv()
    # 2024-10-12/13 is a weekend — must fall back to Friday's close.
    assert _terminal_close(o, date(2024, 10, 13)) == pytest.approx(
        float(o.loc["2024-10-11", "close"])
    )


def test_terminal_close_unavailable_is_nan():
    o = _synth_ohlcv()
    # Far before the series start, outside the tolerance window → NaN.
    assert np.isnan(_terminal_close(o, date(2024, 1, 1)))


def test_realized_assignment_is_terminal_close_below_strike():
    o = _synth_ohlcv()
    tc = _terminal_close(o, date(2024, 10, 31))  # ~230
    assert float(tc < 235.0) == 1.0  # finishes ITM for a 235 put
    assert float(tc < 225.0) == 0.0  # not ITM for a 225 put


def test_binned_reports_predicted_vs_realized_gap():
    # Construct a frame where high-correction rows have realized >> predicted
    # (engine under-sees realized risk) and low-correction rows are calibrated.
    rng = np.random.default_rng(0)
    n = 200
    corr = np.concatenate([rng.uniform(0, 0.1, n), rng.uniform(0.3, 0.6, n)])
    pred = np.full(2 * n, 0.2)
    # low-correction: realized ~ predicted; high-correction: realized ~ 0.5
    realized = np.concatenate(
        [(rng.uniform(size=n) < 0.2).astype(float), (rng.uniform(size=n) < 0.5).astype(float)]
    )
    df = pd.DataFrame({"correction_pct": corr, "eng_prob_itm": pred, "realized_itm": realized})
    out = _binned(df, "correction_pct", q=4)
    assert not out.empty
    # The highest-correction bin must show a clearly positive realized−predicted gap.
    assert out.iloc[-1]["gap_realized_minus_pred"] > out.iloc[0]["gap_realized_minus_pred"]


# --------------------------------------------------------------------------
# Small-sample statistical guard: Wilson score interval + n-based flagging.
# Realized assignment frequency is a small binomial proportion — every point
# carries a Wilson CI and thin bins must be flagged, never read as signal.
# --------------------------------------------------------------------------


def test_wilson_ci_zero_events_clamps_low_and_is_positive_upper():
    lo, hi = _wilson_ci(0, 30)
    assert lo == 0.0  # clamped at 0
    assert 0.08 < hi < 0.14  # upper bound is meaningfully above the 0 estimate


def test_wilson_ci_half_is_symmetric_and_wide_for_small_n():
    lo, hi = _wilson_ci(15, 30)
    assert lo == pytest.approx(0.331, abs=0.01)
    assert hi == pytest.approx(0.669, abs=0.01)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-9)  # symmetric at p=0.5


def test_wilson_ci_all_events_clamps_upper_at_one():
    lo, hi = _wilson_ci(5, 5)
    assert hi == 1.0
    assert lo < 1.0  # not a degenerate point


def test_wilson_ci_zero_n_is_nan():
    lo, hi = _wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_binned_refuses_below_honest_floor():
    # Fewer than 2*MIN_BIN_N resolved contracts → no bins (don't bin dishonestly).
    df = pd.DataFrame(
        {
            "correction_pct": np.linspace(0, 0.3, 2 * MIN_BIN_N - 1),
            "eng_prob_itm": 0.2,
            "realized_itm": 0.0,
        }
    )
    assert _binned(df, "correction_pct").empty


def test_binned_emits_wilson_ci_and_trust_columns():
    n = 6 * MIN_BIN_N
    df = pd.DataFrame(
        {
            "correction_pct": np.linspace(0, 0.5, n),
            "eng_prob_itm": np.full(n, 0.2),
            "realized_itm": (np.arange(n) % 4 == 0).astype(float),
        }
    )
    out = _binned(df, "correction_pct")
    assert not out.empty
    for col in ("realized_lo", "realized_hi", "gap_lo", "gap_hi", "trustworthy", "n", "n_clusters"):
        assert col in out.columns
    # No ticker/as_of columns ⇒ each row is its own cluster ⇒ many clusters ⇒
    # trustworthy, and the cluster-robust CI brackets the point estimate.
    assert out["trustworthy"].all()
    assert (out["realized_lo"] <= out["realized"] + 1e-9).all()
    assert (out["realized"] <= out["realized_hi"] + 1e-9).all()


def test_binned_flags_pseudo_replicated_bin_as_untrustworthy():
    # Large n but only a few independent (ticker, as_of) clusters — the exact
    # trap the preliminary exposed: 150 contracts, 4 clusters. Must NOT be
    # trustworthy despite n >> MIN_BIN_N, because n_clusters < MIN_CLUSTERS.
    rng = np.random.default_rng(1)
    n = 6 * MIN_BIN_N
    # 4 clusters total (2 tickers x 2 as_of); all correction values span the
    # range so every bin inherits the same 4 clusters.
    tickers = np.array(["AAA", "BBB"])[rng.integers(0, 2, n)]
    as_ofs = np.array(["2025-01-01", "2025-02-01"])[rng.integers(0, 2, n)]
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "as_of": as_ofs,
            "correction_pct": np.linspace(0, 0.5, n),
            "eng_prob_itm": np.full(n, 0.2),
            "realized_itm": (rng.uniform(size=n) < 0.4).astype(float),
        }
    )
    out = _binned(df, "correction_pct")
    assert not out.empty
    assert (out["n"] >= MIN_BIN_N).all()  # plenty of contracts
    assert (out["n_clusters"] < MIN_CLUSTERS).all()  # but too few independent events
    assert not out["trustworthy"].any()  # ⇒ never read as signal
