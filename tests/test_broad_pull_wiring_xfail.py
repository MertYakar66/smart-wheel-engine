"""Acceptance scaffolds for the supervised Phase 1-3 broad-pull wiring.

These are **xfail(strict) behaviour pins** (the #366 discipline: pin behaviour,
not signature). Each encodes the observable acceptance criterion for a wiring
step that `docs/WIRING_CAMPAIGN.md` schedules as EV-moving / supervised — work
that is **out of scope** for Phase 0B. They FAIL today (the wiring is absent),
so they `xfail`; when a step lands they XPASS, and `strict=True` turns that into
a failure that forces the implementer to drop the marker and adopt the test as a
live behaviour gate.

Only steps whose contract is **explicit** in the runbook / audit are scaffolded
here (not speculation):

* **#354 / W-2 / Phase 3G** — `get_fundamentals` and `get_credit_risk` gain an
  `as_of` so historical backtests read point-in-time values, not the 2026
  snapshot. (Runbook Phase 5 #354: "add `as_of` to `get_fundamentals` /
  `get_credit_risk` and thread it from `wheel_runner`.")
  - **Status:** the `get_fundamentals` connector half **LANDED 2026-06-26** —
    `as_of` resolves the dividend yield PIT from the dated `dividend_pit` panel
    (backward-compatible: `as_of=None` is unchanged, so the ranked path and the
    S27/S32/S34/S35 baselines do not move). Its xfail is dropped and the test is
    a live gate. STILL PENDING (supervised): (a) threading `as_of` from
    `wheel_runner` into the ranked path — the EV-moving / re-baseline-coupled
    half; (b) `get_credit_risk` `as_of` — there is **no dated credit source**
    (ratings are a single 2026 `snapshot_bdp` as-of), so its xfail stays put
    rather than be flipped by a signature-only no-op (the false-green trap).
* **#372 / W-17 / C2 / Phase 1** — the R9 sector grouping resolves a served
  name's **real GICS sector**, not the hardcoded `DEFAULT_SECTOR_MAP` `'Unknown'`
  bucket. (Audit C2; runbook 2A.)

The remaining steps (skew→pricer, macro-calendar→event_gate, #369/#378 IV gates,
ratings reviewer, short-interest R10) have their acceptance criteria recorded in
`docs/worklog/phase0b-broad-pull-loaders.md` — their consumer contracts are
decided during the supervised design, so committing a concrete behaviour test
now would risk encoding the wrong contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.broad_pull_loaders import BroadPullLoader

BROAD_PULL_DIR = Path("data/bloomberg/broad_pull")
DATA_DIR = Path("data/bloomberg")
HAS_BROAD_PULL_DATA = (BROAD_PULL_DIR / "macro_vol" / "sp500_vol_indices.csv").exists()
HAS_BLOOMBERG_DATA = (DATA_DIR / "sp500_ohlcv.csv").exists()

needs_data = pytest.mark.skipif(
    not (HAS_BROAD_PULL_DATA and HAS_BLOOMBERG_DATA),
    reason="needs both data/bloomberg and data/bloomberg/broad_pull",
)


@needs_data
def test_get_fundamentals_is_point_in_time():
    """ACCEPTANCE (#354 connector half — LANDED 2026-06-26): `get_fundamentals(
    ticker, as_of=...)` returns a point-in-time dividend yield — historical !=
    current snapshot — sourced from the dated `broad_pull/dividend_pit` panel.

    The connector PIT capability landed (xfail dropped); this is now a live
    behaviour gate. The EV-moving step that *threads* `as_of` from `wheel_runner`
    into the ranked path (so historical backtests price BSM `q` PIT) remains the
    supervised Phase-3G change; the connector contract this test pins is met.
    """
    from engine.data_connector import MarketDataConnector

    # Data-driven pick: a ticker whose 12m dividend yield moved the most between
    # 2016 and 2024 (so the PIT difference is unambiguous), from the integrated
    # dividend-PIT panel itself.
    div = BroadPullLoader().load("dividend_pit")
    assert div is not None
    early_win = div[div["date"].dt.year == 2016]
    late_win = div[div["date"].dt.year == 2024]
    e = early_win.groupby("ticker_normalized")["dvd_yld_12m"].mean()
    le = late_win.groupby("ticker_normalized")["dvd_yld_12m"].mean()
    common = e.index.intersection(le.index)
    delta = (le[common] - e[common]).abs().dropna()
    ticker = str(delta.idxmax())

    conn = MarketDataConnector()
    early = conn.get_fundamentals(ticker, as_of="2016-06-30")
    late = conn.get_fundamentals(ticker, as_of="2024-06-30")
    assert early is not None and late is not None
    assert early.get("dividend_yield") != late.get("dividend_yield"), (
        "dividend yield must be point-in-time, not the current snapshot"
    )


@needs_data
@pytest.mark.xfail(strict=True, reason="#354/Phase-3G: get_credit_risk not yet as_of-aware (PIT)")
def test_get_credit_risk_is_point_in_time():
    """ACCEPTANCE (#354): `get_credit_risk(ticker, as_of=...)` accepts an as-of
    and returns a PIT rating. Today no `as_of` -> TypeError -> xfail.
    """
    from engine.data_connector import MarketDataConnector

    conn = MarketDataConnector()
    out = conn.get_credit_risk("AAPL", as_of="2018-06-30")
    assert out is not None


@needs_data
@pytest.mark.xfail(
    strict=True, reason="#372/C2: R9 sector grouping still uses DEFAULT_SECTOR_MAP, not real GICS"
)
def test_r9_sector_grouping_uses_real_gics():
    """ACCEPTANCE (#372): the R9 sector manager resolves a served name's real
    GICS sector instead of collapsing it to `'Unknown'`.

    Today `SectorExposureManager` binds `DEFAULT_SECTOR_MAP` (132/511 names), so
    a name outside the map -> `'Unknown'` (audit C2). We pick such a name that
    *does* carry a real GICS sector in the broad-pull snapshot and assert the
    grouping equals it. Fails today (`'Unknown'` != real) -> xfail; flips when
    #372 wires `gics_sector_name` as the R9 sector source.

    If #372 lands the GICS map at the `check_sector_cap` call site rather than in
    the default `SectorExposureManager()` construction, update this test to query
    that path when removing the marker (see `docs/WIRING_CAMPAIGN.md` Phase 1).
    """
    from engine.risk_manager import DEFAULT_SECTOR_MAP, SectorExposureManager

    snap = BroadPullLoader().load("snapshot_bdp")
    assert snap is not None
    # served names with a real GICS sector but absent from DEFAULT_SECTOR_MAP
    cand = snap[snap["gics_sector"].notna()]
    gap = cand[~cand["ticker_normalized"].isin(DEFAULT_SECTOR_MAP.keys())]
    assert not gap.empty, "expected names outside DEFAULT_SECTOR_MAP (the C2 gap)"
    row = gap.iloc[0]
    name = str(row["ticker_normalized"])
    real_gics = str(row["gics_sector"]).strip()

    mgr = SectorExposureManager()
    resolved = str(mgr.get_sector(name)).strip()
    assert resolved == real_gics, (
        f"{name}: R9 sector {resolved!r} should be the real GICS {real_gics!r}, not 'Unknown'"
    )


def test_scaffolds_are_xfail_not_skipped_when_data_present():
    """Meta-guard: when the data is present these acceptance tests must be live
    xfails (not silently skipped), so a landed wiring step trips strict-xpass.
    """
    if not (HAS_BROAD_PULL_DATA and HAS_BLOOMBERG_DATA):
        pytest.skip("data absent; acceptance scaffolds skip by design")
    # Sanity: the source-of-truth panels the scaffolds rely on are loadable.
    loader = BroadPullLoader()
    assert loader.load("dividend_pit") is not None
    assert loader.load("snapshot_bdp") is not None
    assert isinstance(pd.Timestamp("2016-06-30"), pd.Timestamp)
