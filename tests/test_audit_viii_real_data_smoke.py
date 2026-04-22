"""
AUDIT-VIII real-data smoke test.

Asserts that ``WheelRunner.rank_candidates_by_ev`` produces non-empty
output on the bundled Bloomberg CSVs. This is the one invariant that
would have caught the unit-conversion bugs surfaced in this audit:
the engine had 1067 component tests passing while the authoritative
EV path was returning zero rows on real data because of IV and
risk-free-rate percent/decimal confusion.

The test is marked ``smoke`` so CI can opt in and skipped cleanly when
the CSVs are absent (e.g. on a shallow clone without the data dir).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.data_connector import MarketDataConnector
from engine.wheel_runner import WheelRunner


DATA_DIR = Path("data/bloomberg")


pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "sp500_ohlcv.csv").exists()
    or not (DATA_DIR / "sp500_fundamentals.csv").exists()
    or not (DATA_DIR / "treasury_yields.csv").exists(),
    reason="Bloomberg CSVs not present — run real-data smoke in CI",
)


class TestRealDataEVSmoke:
    def test_ev_ranker_returns_rows_on_real_bloomberg_data(self):
        runner = WheelRunner()
        df = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "JPM", "JNJ", "KO", "V", "PG"],
            dte_target=35,
            delta_target=0.25,
            top_n=10,
            min_ev_dollars=-1e9,
            as_of="2025-06-01",
        )
        assert df is not None, "EV ranker returned None"
        assert len(df) > 0, (
            "EV ranker returned zero rows on real Bloomberg data — the "
            "unit-conversion guards in rank_candidates_by_ev are not "
            "working. This is the regression that caused the audit-VIII "
            "P0 launch blocker."
        )
        # Sanity: first row carries the basic EV fields.
        row = df.iloc[0]
        assert "ev_dollars" in row
        assert "prob_profit" in row
        assert "distribution_source" in row
        # IV should be a decimal (0 < iv < 3), not a percent.
        assert 0.0 < float(row["iv"]) < 3.0
        # Premium should be a non-trivial dollar amount.
        assert float(row["premium"]) > 0.05

    def test_connector_risk_free_rate_is_decimal(self):
        """The connector fix must return a decimal rate now. If this
        regresses back to a percent, downstream BSM math breaks
        catastrophically (the original audit-VIII P0)."""
        conn = MarketDataConnector()
        rate = conn.get_risk_free_rate("2025-06-01")
        # Historical 3M Treasury has been between 0% and 6% over the
        # bundled range; anything over 1.0 means we regressed to percent.
        assert 0.0 <= rate <= 0.20, f"risk-free rate out of decimal range: {rate}"
