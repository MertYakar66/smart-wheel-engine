"""R3 + R6 survivorship harness tests — docs/DATA_LAYER_DEEP_READ_DESIGN.md Part B.

All gated on ``SWE_DEEP_TEST_DATA`` pointing at a materialized data dir (refresh
monoliths + deep/ + delisted gz + sp500_index_membership.csv) — they need the
deep + delisted panels, which are not committed. They skip in CI.

R3 (universe): the point-in-time universe on a 2008 date includes the names that
later delisted (Lehman, WaMu) and excludes names that joined after 2008; size ≈ 500.

R6 (the proof): a 2008 survivorship backtest where a delisted name (Lehman) flows
through the ranker and its loss is REALIZED at the delisting price — not silently
dropped (which a plain on/after spot lookup would do past the delisting date).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

_DEEP = os.environ.get("SWE_DEEP_TEST_DATA")

deep_data = pytest.mark.skipif(
    not (
        _DEEP
        and (Path(_DEEP) / "deep" / "sp500_ohlcv__delisted.csv.gz").exists()
        and (Path(_DEEP) / "sp500_index_membership.csv").exists()
    ),
    reason="set SWE_DEEP_TEST_DATA to a dir with deep/ slices + membership to run survivorship tests",
)


# --------------------------------------------------------------------------
# R3 — point-in-time universe selection
# --------------------------------------------------------------------------


@deep_data
def test_pit_universe_2008_includes_delisted_names():
    from backtests.survivorship import pit_universe

    u = set(pit_universe("2008-07-01", data_dir=_DEEP))
    assert "LEHMQ" in u, "Lehman (LEHMQ) should be in the 2008-07-01 PIT universe"
    assert "WAMUQ" in u, "Washington Mutual (WAMUQ) should be in the 2008-07-01 PIT universe"


@deep_data
def test_pit_universe_2008_excludes_post_2008_names():
    from backtests.survivorship import pit_universe

    u = set(pit_universe("2008-07-01", data_dir=_DEEP))
    for late in ("META", "ABNB", "GEHC", "CEG"):
        assert late not in u, f"{late} joined after 2008 and must not be in the 2008 PIT universe"


@deep_data
def test_pit_universe_size_circa_500():
    from backtests.survivorship import pit_universe

    n = len(pit_universe("2008-07-01", data_dir=_DEEP))
    assert 490 <= n <= 510, f"PIT universe size {n} not ~500"


# --------------------------------------------------------------------------
# R6 — delisting flows through, loss realized (no silent drop)
# --------------------------------------------------------------------------


@deep_data
def test_terminal_spot_delisted_returns_delisting_close_not_none():
    """A post-delisting expiry must value Lehman at its last close (~3.65),
    flagged delisted — NOT return None (which would silently drop the loss)."""
    from backtests.survivorship import make_deep_connector, terminal_spot

    conn = make_deep_connector(_DEEP)
    spot, delisted = terminal_spot(conn, "LEHMQ", date(2008, 10, 17))
    assert delisted is True
    assert spot == pytest.approx(3.65, abs=0.01)
    # A live name on a normal trading day resolves on/after (not delisted-path).
    spot2, delisted2 = terminal_spot(conn, "AAPL", date(2020, 6, 15))
    assert spot2 is not None and spot2 > 0 and delisted2 is False


@deep_data
def test_survivorship_window_assertion_accepts_pre_2018_start():
    """assert_data_window_available (R3 extension) must accept a pre-2018 start
    once the deep OHLCV slice floors are supplied."""
    from backtests.regression._common import assert_data_window_available

    base = Path(_DEEP)
    # Without the deep floors, a 2008 start is rejected (monolith starts 2018).
    with pytest.raises(RuntimeError):
        assert_data_window_available(
            "2008-08-01", "2008-12-31", ohlcv_path=base / "sp500_ohlcv.csv"
        )
    # With them, it passes.
    assert_data_window_available(
        "2008-08-01",
        "2008-12-31",
        ohlcv_path=base / "sp500_ohlcv.csv",
        extra_floor_paths=[
            base / "deep/sp500_ohlcv__1994_2018.csv.gz",
            base / "deep/sp500_ohlcv__delisted.csv.gz",
        ],
    )
