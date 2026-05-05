"""Tests for engine/external_data/yfinance_adapter.py with HTTP mocking."""

from __future__ import annotations

import math
import re

import pytest
import requests_mock as rm_module

from engine.external_data.yfinance_adapter import (
    CROSS_ASSET_PACK,
    SECTOR_ETFS,
    YFinanceAdapter,
)


@pytest.fixture
def mock():
    with rm_module.Mocker() as m:
        yield m


def _ohlcv_csv(rows: list[tuple[str, float, float, float, float, int]] | None = None) -> str:
    """Build a Yahoo-style OHLCV CSV."""
    rows = rows or [
        ("2026-01-02", 100.0, 101.5, 99.5, 101.0, 1_000_000),
        ("2026-01-03", 101.0, 102.5, 100.5, 102.0, 1_100_000),
    ]
    header = "Date,Open,High,Low,Close,Adj Close,Volume\n"
    body = "\n".join(
        f"{d},{o},{h},{l},{c},{c},{v}" for d, o, h, l, c, v in rows
    )
    return header + body


YF_RE = re.compile(r"https://query1\.finance\.yahoo\.com/v7/finance/download/.*")


class TestGetOhlcv:
    def test_happy_path(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv())
        adapter = YFinanceAdapter()
        df = adapter.get_ohlcv("AAPL")
        assert not df.empty
        assert {"open", "high", "low", "close", "volume"} <= set(df.columns)
        assert len(df) == 2

    def test_404_returns_empty(self, mock: rm_module.Mocker):
        mock.get(YF_RE, status_code=404)
        adapter = YFinanceAdapter()
        assert adapter.get_ohlcv("AAPL").empty

    def test_500_returns_empty(self, mock: rm_module.Mocker):
        mock.get(YF_RE, status_code=500)
        adapter = YFinanceAdapter()
        assert adapter.get_ohlcv("AAPL").empty

    def test_network_error_returns_empty(self, mock: rm_module.Mocker):
        mock.get(YF_RE, exc=ConnectionError("boom"))
        adapter = YFinanceAdapter()
        assert adapter.get_ohlcv("AAPL").empty

    def test_empty_csv_returns_empty(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text="")
        adapter = YFinanceAdapter()
        assert adapter.get_ohlcv("AAPL").empty

    def test_csv_without_date_column_returns_empty(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text="Close\n100\n101\n")
        adapter = YFinanceAdapter()
        assert adapter.get_ohlcv("AAPL").empty

    def test_partial_columns_filtered(self, mock: rm_module.Mocker):
        # Date + Close only — adapter returns just close
        text = "Date,Close\n2026-01-02,100.0\n2026-01-03,101.0\n"
        mock.get(YF_RE, text=text)
        adapter = YFinanceAdapter()
        df = adapter.get_ohlcv("AAPL")
        assert "close" in df.columns
        assert "open" not in df.columns

    def test_caching_avoids_second_call(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv())
        adapter = YFinanceAdapter()
        adapter.get_ohlcv("AAPL")
        adapter.get_ohlcv("AAPL")
        # Cached on first hit
        assert mock.call_count == 1

    def test_period_days_override(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv())
        adapter = YFinanceAdapter(lookback_days=365)
        df = adapter.get_ohlcv("AAPL", period_days=30)
        assert not df.empty
        # The request URL params include period1/period2 — captured in mock
        sent_url = mock.last_request.url
        assert "period1=" in sent_url
        assert "period2=" in sent_url


class TestLatestClose:
    def test_returns_last_close(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv())
        adapter = YFinanceAdapter()
        # Last close = 102.0 from fixture
        assert adapter.latest_close("AAPL") == pytest.approx(102.0)

    def test_returns_nan_on_404(self, mock: rm_module.Mocker):
        mock.get(YF_RE, status_code=404)
        adapter = YFinanceAdapter()
        assert math.isnan(adapter.latest_close("AAPL"))

    def test_returns_nan_when_no_close_column(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text="Date,Open\n2026-01-02,100.0\n")
        adapter = YFinanceAdapter()
        assert math.isnan(adapter.latest_close("AAPL"))


class TestCrossAssetSnapshot:
    def test_full_pack_returns_price_and_ret_1d(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv([
            ("2026-01-02", 100.0, 101.5, 99.5, 100.0, 1_000_000),
            ("2026-01-03", 101.0, 102.5, 100.5, 102.0, 1_100_000),
        ]))
        adapter = YFinanceAdapter()
        snap = adapter.cross_asset_snapshot()
        assert set(snap.keys()) == set(CROSS_ASSET_PACK.keys())
        for entry in snap.values():
            assert "price" in entry
            assert "ret_1d" in entry
        # 1-day return: (102 - 100) / 100 = 0.02
        for entry in snap.values():
            if entry["price"] == entry["price"]:  # not NaN
                assert entry["ret_1d"] == pytest.approx(0.02)

    def test_handles_missing_data_per_symbol(self, mock: rm_module.Mocker):
        # All symbols 404
        mock.get(YF_RE, status_code=404)
        adapter = YFinanceAdapter()
        snap = adapter.cross_asset_snapshot()
        for entry in snap.values():
            assert math.isnan(entry["price"])
            assert math.isnan(entry["ret_1d"])

    def test_single_row_no_prev_close_yields_nan_ret(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv([
            ("2026-01-02", 100.0, 101.5, 99.5, 100.0, 1_000_000),
        ]))
        adapter = YFinanceAdapter()
        snap = adapter.cross_asset_snapshot()
        for entry in snap.values():
            # < 2 rows → both fields NaN
            assert math.isnan(entry["price"])
            assert math.isnan(entry["ret_1d"])


class TestSectorReturns:
    def test_full_sector_pack_returns_returns(self, mock: rm_module.Mocker):
        mock.get(YF_RE, text=_ohlcv_csv([
            ("2026-01-02", 100.0, 101.0, 99.0, 100.0, 1_000_000),
            ("2026-01-30", 105.0, 106.0, 104.0, 105.0, 1_100_000),
        ]))
        adapter = YFinanceAdapter()
        rets = adapter.sector_returns(period_days=30)
        assert set(rets.keys()) == set(SECTOR_ETFS.keys())
        # 5% return: (105 - 100) / 100 = 0.05
        for v in rets.values():
            if v == v:  # not NaN
                assert v == pytest.approx(0.05)

    def test_handles_missing_data_per_sector(self, mock: rm_module.Mocker):
        mock.get(YF_RE, status_code=404)
        adapter = YFinanceAdapter()
        rets = adapter.sector_returns()
        for v in rets.values():
            assert math.isnan(v)

    def test_zero_first_close_yields_nan(self, mock: rm_module.Mocker):
        # Pathological zero-first-close → divide-by-zero guard returns NaN
        mock.get(YF_RE, text=_ohlcv_csv([
            ("2026-01-02", 0.0, 0.0, 0.0, 0.0, 0),
            ("2026-01-30", 100.0, 101.0, 99.0, 100.0, 1_000_000),
        ]))
        adapter = YFinanceAdapter()
        rets = adapter.sector_returns()
        for v in rets.values():
            assert math.isnan(v)
