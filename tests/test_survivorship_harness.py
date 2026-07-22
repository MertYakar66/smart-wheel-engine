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

import pandas as pd
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


# --------------------------------------------------------------------------
# CMD 5 — UNGATED synthetic-connector coverage for the survivorship guard.
#
# The @deep_data tests above skip in normal CI (they need the uncommitted deep +
# delisted panels), so terminal_spot() / pit_universe() — which carry the
# load-bearing survivorship guarantee — had ZERO CI coverage. These tests use
# in-memory stubs (no SWE_DEEP_TEST_DATA, no deep data) so they run everywhere,
# and they would FAIL if terminal_spot regressed to a plain on/after lookup
# (which returns None past the delisting date, silently dropping the loss).
# --------------------------------------------------------------------------


class _StubConn:
    """Connector stub: per-ticker ascending ``(iso_date, close)`` series, sliced
    by ``start_date`` / ``end_date`` exactly like the ``get_ohlcv`` contract that
    ``_spot_on_or_after`` (first row on/after) and ``terminal_spot``'s fallback
    (last row on/before) rely on. ISO date strings compare lexicographically,
    matching the ``.isoformat()`` bounds the harness passes."""

    def __init__(self, series: dict[str, list[tuple[str, float]]]):
        self._series = series

    def get_ohlcv(self, ticker, start_date=None, end_date=None, **_kw):
        rows = self._series.get(ticker, [])
        if start_date is not None:
            rows = [r for r in rows if r[0] >= start_date]
        if end_date is not None:
            rows = [r for r in rows if r[0] <= end_date]
        return pd.DataFrame({"close": [c for _, c in rows]}, index=[d for d, _ in rows])


def test_terminal_spot_live_name_resolves_on_or_after():
    """A name trading on/after expiry resolves to that close, delisted=False."""
    from backtests.survivorship import terminal_spot

    conn = _StubConn({"LIVE": [("2023-03-10", 90.0), ("2023-03-16", 100.0)]})
    spot, delisted = terminal_spot(conn, "LIVE", date(2023, 3, 15))
    assert spot == pytest.approx(100.0)
    assert delisted is False


def test_terminal_spot_delisted_returns_last_close_never_none():
    """A name whose history ENDS before expiry (no on/after bar) values at the
    last close on/before expiry (the delisting price), flagged delisted — and
    NEVER None. This is the survivorship guarantee."""
    from backtests.regression._common import _spot_on_or_after
    from backtests.survivorship import terminal_spot

    conn = _StubConn({"SIVB": [("2023-02-01", 250.0), ("2023-03-10", 30.0)]})
    spot, delisted = terminal_spot(conn, "SIVB", date(2023, 3, 15))
    assert spot is not None, "terminal_spot must never return None for a name with history"
    assert spot == pytest.approx(30.0)  # last close on/before expiry
    assert delisted is True

    # Guard has teeth: a PLAIN on/after lookup (what terminal_spot guards
    # against) returns None for this exact input — so this test would fail if
    # terminal_spot regressed to that behaviour.
    assert _spot_on_or_after(conn, "SIVB", date(2023, 3, 15)) is None


def test_terminal_spot_no_history_is_realized_total_loss():
    """No history at all → (0.0, True): a total loss, still REALIZED not None."""
    from backtests.survivorship import terminal_spot

    conn = _StubConn({})  # unknown ticker → empty everywhere
    spot, delisted = terminal_spot(conn, "GONE", date(2023, 3, 15))
    assert spot == 0.0
    assert delisted is True


class _StubLoader:
    """Loader stub exposing only ``get_universe_as_of`` — mirrors
    ``ConsolidatedBloombergLoader``'s PIT accessor."""

    def __init__(self, universe):
        self._u = list(universe)

    def get_universe_as_of(self, as_of):  # noqa: ARG002
        return list(self._u)


def test_pit_universe_returns_exact_pit_membership():
    """pit_universe returns exactly the loader's PIT set — including a
    since-delisted name and excluding a later joiner — with no survivor
    filtering applied."""
    from backtests.survivorship import pit_universe

    # SIVB = a 2022 member that later delisted (must be INCLUDED); GEHC joined
    # 2023 and is (by construction) absent from the 2022 snapshot.
    pit = ["AAPL", "MSFT", "SIVB"]
    result = pit_universe("2022-06-30", loader=_StubLoader(pit))
    assert result == pit
    assert "SIVB" in result  # since-delisted name preserved
    assert "GEHC" not in result  # later joiner absent
