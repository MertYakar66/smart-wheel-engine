"""Equivalence tests for the cached `_filter_ticker` + `_load` unique-map.

These pin that the connector's ticker-filtering speed-up
(`engine.data_connector`, PR: connector-ticker-filter-perf) is **output-identical**
to the prior naive `df[df["ticker"] == ticker]` boolean-mask + `.apply`
normalization — i.e. the engine sees byte-identical data, so the optimization is
§2-neutral. Synthetic data only; no dependency on the large Bloomberg CSVs.
"""

from __future__ import annotations

import pandas as pd

from engine.data_connector import MarketDataConnector, normalize_ticker


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "AAPL", "JPM", "MSFT", "AAPL", "XOM"],
            "date": pd.to_datetime(["2020-01-02"] * 7),
            "close": [10.0, 20.0, 11.0, 40.0, 21.0, 12.0, 70.0],
        }
    )


def test_filter_ticker_matches_boolean_mask():
    """Optimized `_filter_ticker` == naive `df[df["ticker"] == t]` for every
    ticker, including multi-occurrence names and an absent ticker."""
    conn = MarketDataConnector()
    df = _sample_frame()
    for t in ["AAPL", "MSFT", "JPM", "XOM", "ABSENT"]:
        naive = df[df["ticker"] == t]
        opt = conn._filter_ticker(df, t)
        pd.testing.assert_frame_equal(opt, naive)


def test_filter_ticker_cache_is_built_and_reused():
    conn = MarketDataConnector()
    df = _sample_frame()
    first = conn._filter_ticker(df, "AAPL")
    assert id(df) in conn._ticker_groups  # index built on first call
    second = conn._filter_ticker(df, "AAPL")
    pd.testing.assert_frame_equal(first, second)
    # other tickers served from the same cached index, still correct
    pd.testing.assert_frame_equal(conn._filter_ticker(df, "MSFT"), df[df["ticker"] == "MSFT"])


def test_absent_ticker_returns_empty_same_schema():
    conn = MarketDataConnector()
    df = _sample_frame()
    out = conn._filter_ticker(df, "NOPE")
    assert out.empty
    assert list(out.columns) == list(df.columns)


def test_empty_or_tickerless_frame_passthrough():
    conn = MarketDataConnector()
    empty = pd.DataFrame()
    assert conn._filter_ticker(empty, "AAPL").empty
    no_tk = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "close": [1.0]})
    pd.testing.assert_frame_equal(conn._filter_ticker(no_tk, "AAPL"), no_tk)


def test_load_unique_map_normalizes_like_apply(tmp_path):
    """`_load`'s unique-map normalization == the prior `.apply(normalize_ticker)`
    (same values), and downstream `_filter_ticker` resolves the normalized name."""
    raw = pd.DataFrame(
        {
            "ticker": ["AAPL UW Equity", "MSFT UW Equity", "AAPL UW Equity", "A UN"],
            "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-02"],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    (tmp_path / "sp500_ohlcv.csv").write_text(raw.to_csv(index=False), encoding="utf-8")

    conn = MarketDataConnector(data_dir=str(tmp_path))
    loaded = conn._load("ohlcv")

    expected = raw["ticker"].apply(normalize_ticker).tolist()
    assert loaded["ticker"].tolist() == expected == ["AAPL", "MSFT", "AAPL", "A"]
    # filtering on the normalized symbol returns both AAPL rows
    aapl = conn._filter_ticker(loaded, "AAPL")
    assert aapl["close"].tolist() == [1.0, 3.0]
