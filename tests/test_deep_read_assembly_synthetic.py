"""R2 deep-read assembly — synthetic-fixture unit tests (CI-runnable).

The real assembly tests in ``test_deep_read_connector.py`` need the multi-GB deep
panels and skip in CI. These build tiny gz fixtures in ``tmp_path`` so the
load-bearing logic — **dedup precedence** (recent > deep-current > delisted),
multi-slice concat, missing-slice degrade, and the default-OFF gate — is covered
in CI without any committed data.

The discriminator is the ``volume`` column (not rotated by ``get_ohlcv``), so the
assertions are independent of the OHLC rename.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector

_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _row(ticker: str, d: str, volume: int) -> dict:
    # Valid rotated bar (stored open=row-max, low=row-min) so the invariant check
    # stays quiet; prices are irrelevant to these assertions.
    return {
        "date": d,
        "ticker": ticker,
        "open": 110.0,
        "high": 100.0,
        "low": 90.0,
        "close": 95.0,
        "volume": volume,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_COLS).to_csv(path, index=False)


def _write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_COLS).to_csv(path, index=False, compression="gzip")


@pytest.fixture
def deep_dir(tmp_path):
    """A tiny data dir: monolith + deep-current + delisted, with deliberate
    (ticker, date) overlaps to exercise precedence."""
    base = tmp_path / "bloomberg"
    _write_csv(
        base / "sp500_ohlcv.csv",
        [_row("AAA", "2018-01-02", 1000), _row("AAA", "2020-01-02", 1001)],
    )
    _write_gz(
        base / "deep" / "sp500_ohlcv__1994_2018.csv.gz",
        [
            _row("AAA", "2015-01-02", 2000),
            _row("AAA", "2018-01-02", 2001),  # overlaps the monolith -> recent wins
        ],
    )
    _write_gz(
        base / "deep" / "sp500_ohlcv__delisted.csv.gz",
        [
            _row("AAA", "2015-01-02", 3000),  # overlaps deep-current -> deep-current wins
            _row("DEAD", "2010-01-04", 5000),  # delisted-only name
        ],
    )
    return base


def test_assembly_spans_all_sources(deep_dir):
    df = MarketDataConnector(str(deep_dir), deep_history=True).get_ohlcv("AAA")
    assert df.index.min() == pd.Timestamp("2015-01-02")
    assert df.index.max() == pd.Timestamp("2020-01-02")
    assert df.index.is_monotonic_increasing


def test_precedence_recent_over_deep(deep_dir):
    df = MarketDataConnector(str(deep_dir), deep_history=True).get_ohlcv("AAA")
    # 2018-01-02 exists in BOTH monolith (vol 1000) and deep-current (vol 2001):
    # recent monolith must win.
    assert df.loc["2018-01-02", "volume"] == 1000


def test_precedence_deep_over_delisted(deep_dir):
    df = MarketDataConnector(str(deep_dir), deep_history=True).get_ohlcv("AAA")
    # 2015-01-02 exists in deep-current (vol 2000) and delisted (vol 3000):
    # deep-current must win.
    assert df.loc["2015-01-02", "volume"] == 2000


def test_delisted_only_name_present_when_on(deep_dir):
    conn = MarketDataConnector(str(deep_dir), deep_history=True)
    dead = conn.get_ohlcv("DEAD")
    assert not dead.empty
    assert pd.Timestamp("2010-01-04") in dead.index


def test_default_off_is_monolith_only(deep_dir):
    conn = MarketDataConnector(str(deep_dir), deep_history=False)
    aapl = conn.get_ohlcv("AAA")
    assert aapl.index.min() == pd.Timestamp("2018-01-02")  # no deep rows
    assert conn.get_ohlcv("DEAD").empty


def test_missing_slice_degrades(tmp_path):
    """Only the monolith + deep-current present (no delisted gz): assembles what
    is there, skips the missing slice, no crash, no DEAD name."""
    base = tmp_path / "bloomberg"
    _write_csv(base / "sp500_ohlcv.csv", [_row("AAA", "2020-01-02", 1001)])
    _write_gz(
        base / "deep" / "sp500_ohlcv__1994_2018.csv.gz",
        [_row("AAA", "2015-01-02", 2000)],
    )
    conn = MarketDataConnector(str(base), deep_history=True)
    df = conn.get_ohlcv("AAA")
    assert df.index.min() == pd.Timestamp("2015-01-02")
    assert df.index.max() == pd.Timestamp("2020-01-02")
    assert conn.get_ohlcv("DEAD").empty  # delisted slice absent -> skipped


def test_non_deep_key_not_assembled(deep_dir):
    """A key absent from _DEEP_SLICES uses the normal _load even with deep ON."""
    conn = MarketDataConnector(str(deep_dir), deep_history=True)
    # fundamentals has no file here -> empty frame, and no assembly attempted.
    assert "fundamentals" not in MarketDataConnector._DEEP_SLICES
    assert conn.get_fundamentals("AAA") is None
