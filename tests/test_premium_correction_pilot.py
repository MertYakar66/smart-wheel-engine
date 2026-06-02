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
