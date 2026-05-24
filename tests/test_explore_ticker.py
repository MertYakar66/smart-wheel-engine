"""Real-data smoke test for ``WheelRunner.explore_ticker``.

Investor-scenario 2 logged that ``rank_candidates_by_ev`` emits a single
``(delta_target=0.25, dte_target=35)`` cell per ticker, while the
investor's natural follow-up is "show me the strike/DTE surface for this
ticker." ``explore_ticker`` is that follow-up: a grid sweep that reuses
the production EV path one cell at a time.

This test asserts (a) the grid shape matches ``len(deltas) × len(dtes)``,
(b) the contract columns are present, (c) the frame is sorted
descending by ``ev_dollars``, and (d) the new ``delta_target`` column
appears immediately after ``ticker``. Skipped cleanly when the
Bloomberg CSVs are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.wheel_runner import WheelRunner

DATA_DIR = Path("data/bloomberg")

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "sp500_ohlcv.csv").exists()
    or not (DATA_DIR / "sp500_fundamentals.csv").exists()
    or not (DATA_DIR / "treasury_yields.csv").exists(),
    reason="Bloomberg CSVs not present — real-data smoke only",
)

# Small grid keeps the suite fast (2 deltas × 2 DTEs = 4 inner ranker
# calls on a one-ticker universe; ~3-5 s total).
DELTAS = (0.20, 0.25)
DTES = (28, 35)
TICKER = "XOM"

# Columns the investor / AI explanation layer is contractually
# guaranteed to see. The provenance columns (``oi_source`` /
# ``premium_source``) from PR #160 are intentionally NOT in this list
# until #160 lands on main; once it does, ``explore_ticker`` inherits
# them automatically via the inner ranker.
REQUIRED_COLUMNS = {
    "ticker",
    "delta_target",
    "dte",
    "strike",
    "premium",
    "ev_dollars",
    "ev_raw",
    "prob_profit",
    "prob_assignment",
    "ev_per_day",
    "cvar_5",
}


class TestExploreTickerGrid:
    def test_shape_matches_grid(self):
        runner = WheelRunner()
        df = runner.explore_ticker(
            TICKER,
            deltas=DELTAS,
            dtes=DTES,
            as_of="2025-06-01",
        )
        # No cell should be dropped on a recent as_of for a major name
        # like XOM (history > 504d, no earnings inside 28-35 DTE window
        # for a 2025-06-01 cutoff). If a cell IS legitimately dropped
        # the .attrs["drops"] gives the reason; surface it in the
        # assertion message rather than a silent shape mismatch.
        drops = df.attrs.get("drops", [])
        assert len(df) == len(DELTAS) * len(DTES), (
            f"expected {len(DELTAS) * len(DTES)} cells, got {len(df)}. Drops: {drops}"
        )

    def test_contract_columns_present(self):
        runner = WheelRunner()
        df = runner.explore_ticker(TICKER, deltas=DELTAS, dtes=DTES, as_of="2025-06-01")
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"missing required columns: {missing}"

    def test_delta_target_column_is_second(self):
        runner = WheelRunner()
        df = runner.explore_ticker(TICKER, deltas=DELTAS, dtes=DTES, as_of="2025-06-01")
        # delta_target inserted right after ticker so the grid reads
        # left-to-right: ticker, delta_target, dte, strike, ...
        assert list(df.columns)[:2] == ["ticker", "delta_target"]

    def test_each_cell_unique_delta_dte_pair(self):
        runner = WheelRunner()
        df = runner.explore_ticker(TICKER, deltas=DELTAS, dtes=DTES, as_of="2025-06-01")
        pairs = list(zip(df["delta_target"], df["dte"], strict=True))
        assert len(set(pairs)) == len(pairs), f"duplicate (delta, dte) cells in grid: {pairs}"

    def test_sorted_descending_by_ev_dollars(self):
        runner = WheelRunner()
        df = runner.explore_ticker(TICKER, deltas=DELTAS, dtes=DTES, as_of="2025-06-01")
        ev = df["ev_dollars"].to_list()
        assert ev == sorted(ev, reverse=True), (
            f"explore_ticker output must be sorted descending by ev_dollars; got {ev}"
        )

    def test_drops_attr_carries_cell_tags(self):
        # Force-trigger the strike-out-of-range drop by asking for a
        # 0.99 delta — that's deep ITM, the BSM delta solver returns a
        # strike >= spot and the gate at wheel_runner.py:973 drops it.
        runner = WheelRunner()
        df = runner.explore_ticker(TICKER, deltas=(0.99,), dtes=(35,), as_of="2025-06-01")
        drops = df.attrs.get("drops", [])
        assert len(drops) >= 1, "expected a drop for delta=0.99"
        # Each drop must carry the cell tags so a caller can attribute
        # gaps in the grid to specific (delta, dte) pairs.
        assert all("delta_target" in d and "dte_target" in d for d in drops), drops
