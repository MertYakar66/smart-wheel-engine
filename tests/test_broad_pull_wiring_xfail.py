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
@pytest.mark.xfail(strict=True, reason="#354/Phase-3G: get_fundamentals not yet as_of-aware (PIT)")
def test_get_fundamentals_is_point_in_time():
    """ACCEPTANCE (#354): `get_fundamentals(ticker, as_of=...)` returns a
    point-in-time dividend yield — historical != current snapshot.

    Today the accessor has no `as_of` (the lookahead, audit W2): the call raises
    TypeError -> xfail. When #354 wires the dated dividend-yield panel (Phase 3G)
    + threads `as_of`, the two dates differ -> XPASS -> drop the marker.
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
def test_r9_sector_grouping_uses_real_gics():
    """ACCEPTANCE (#372, LANDED): R9 resolves a served name's real GICS sector
    via the call-site ``sector_map`` (``resolve_sector`` / ``build_gics_sector_map``),
    instead of collapsing off-``DEFAULT_SECTOR_MAP`` names to ``'Unknown'``.

    #372 wired GICS at the ``check_sector_cap`` call site (not the default
    ``SectorExposureManager()`` construction), so this queries that path — per
    the marker-removal note this scaffold carried. The bare manager remains the
    documented legacy fallback.
    """
    from engine.risk_manager import (
        DEFAULT_SECTOR_MAP,
        GICS_11,
        SectorExposureManager,
        build_gics_sector_map,
        resolve_sector,
    )

    snap = BroadPullLoader().load("snapshot_bdp")
    assert snap is not None
    # served names with a real (canonical) GICS sector but absent from DEFAULT_SECTOR_MAP
    cand = snap[snap["gics_sector"].notna()]
    gap = cand[~cand["ticker_normalized"].isin(DEFAULT_SECTOR_MAP.keys())]
    gap = gap[gap["gics_sector"].astype(str).str.strip().isin(GICS_11)]
    assert not gap.empty, "expected canonical-GICS names outside DEFAULT_SECTOR_MAP (the C2 gap)"
    row = gap.iloc[0]
    name = str(row["ticker_normalized"])
    real_gics = str(row["gics_sector"]).strip()

    # resolver: GICS-primary beats the bare-map 'Unknown'
    assert resolve_sector(name, real_gics) == real_gics

    # call-site path: a connector-built sector_map threads the real GICS into the
    # gate's SectorExposureManager (the seam #372 uses).
    class _Stub:
        def get_fundamentals(self, t):
            return {"sector": real_gics} if t == name else None

    sector_map = build_gics_sector_map(_Stub(), [name])
    assert sector_map[name] == real_gics
    assert SectorExposureManager(sector_map=sector_map).get_sector(name) == real_gics


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
