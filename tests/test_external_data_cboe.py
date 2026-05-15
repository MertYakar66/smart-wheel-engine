"""Tests for engine/external_data/cboe_adapter.py with HTTP mocking."""

from __future__ import annotations

import math

import pytest
import requests_mock as rm_module

from engine.external_data.cboe_adapter import _CBOE_URL, CBOEAdapter


def _csv(symbol: str = "VIX", rows: list[tuple[str, float, float, float, float]] | None = None) -> str:
    """Build a CBOE-style CSV (Date, OPEN, HIGH, LOW, CLOSE)."""
    rows = rows or [
        ("2026-01-02", 14.5, 15.1, 14.2, 14.8),
        ("2026-01-03", 14.8, 15.5, 14.6, 15.2),
        ("2026-01-04", 15.2, 16.0, 15.0, 15.7),
    ]
    header = "DATE,OPEN,HIGH,LOW,CLOSE\n"
    body = "\n".join(f"{d},{o},{h},{l},{c}" for d, o, h, l, c in rows)
    return header + body


@pytest.fixture
def mock():
    with rm_module.Mocker() as m:
        yield m


class TestGetSeries:
    def test_happy_path_returns_dataframe(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text=_csv("VIX"))
        adapter = CBOEAdapter()
        df = adapter.get_series("VIX")
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close"]
        assert len(df) == 3
        # Series sorted ascending by date
        assert df.index[0] < df.index[-1]

    def test_404_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="WIBBLE"), status_code=404)
        adapter = CBOEAdapter()
        assert adapter.get_series("WIBBLE").empty

    def test_5xx_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), status_code=503)
        adapter = CBOEAdapter()
        assert adapter.get_series("VIX").empty

    def test_network_error_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), exc=ConnectionError("boom"))
        adapter = CBOEAdapter()
        assert adapter.get_series("VIX").empty

    def test_malformed_csv_returns_empty_or_filtered(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text="not a csv at all")
        adapter = CBOEAdapter()
        df = adapter.get_series("VIX")
        # pandas may parse this as a 1-column DF; either way no OHLC cols -> empty
        assert df.empty or not {"open", "high", "low", "close"} & set(df.columns)

    def test_empty_csv_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text="")
        adapter = CBOEAdapter()
        assert adapter.get_series("VIX").empty

    def test_csv_with_only_header_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text="DATE,OPEN,HIGH,LOW,CLOSE\n")
        adapter = CBOEAdapter()
        assert adapter.get_series("VIX").empty

    def test_csv_with_partial_columns_filters_to_present(self, mock: rm_module.Mocker):
        # Only DATE + CLOSE — adapter should return just close
        text = "DATE,CLOSE\n2026-01-02,14.5\n2026-01-03,15.0\n"
        mock.get(_CBOE_URL.format(sym="VIX"), text=text)
        adapter = CBOEAdapter()
        df = adapter.get_series("VIX")
        assert "close" in df.columns
        assert "open" not in df.columns

    def test_caching_avoids_second_call(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text=_csv("VIX"))
        adapter = CBOEAdapter()
        adapter.get_series("VIX")
        adapter.get_series("VIX")
        assert mock.call_count == 1  # served from lru_cache

    def test_symbol_uppercased(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text=_csv())
        adapter = CBOEAdapter()
        df = adapter.get_series("vix")  # lowercase
        assert not df.empty


class TestLatest:
    def test_returns_last_close(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="VIX"), text=_csv())
        adapter = CBOEAdapter()
        # Fixture's last close is 15.7
        assert adapter.latest("VIX") == pytest.approx(15.7)

    def test_returns_nan_on_404(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="WIBBLE"), status_code=404)
        adapter = CBOEAdapter()
        assert math.isnan(adapter.latest("WIBBLE"))

    def test_returns_nan_when_no_close_column(self, mock: rm_module.Mocker):
        text = "DATE,OPEN\n2026-01-02,14.5\n"
        mock.get(_CBOE_URL.format(sym="VIX"), text=text)
        adapter = CBOEAdapter()
        assert math.isnan(adapter.latest("VIX"))


class TestVixTermStructure:
    def test_full_pack_contango(self, mock: rm_module.Mocker):
        # VIX < VIX3M → contango (slope_30d_3m > 0)
        mock.get(_CBOE_URL.format(sym="VIX"), text=_csv("VIX", [("2026-01-02", 13, 14, 13, 14.0)]))
        mock.get(_CBOE_URL.format(sym="VIX9D"), text=_csv("VIX9D", [("2026-01-02", 12, 13, 12, 12.5)]))
        mock.get(_CBOE_URL.format(sym="VIX3M"), text=_csv("VIX3M", [("2026-01-02", 16, 17, 16, 16.5)]))
        mock.get(_CBOE_URL.format(sym="VIX6M"), text=_csv("VIX6M", [("2026-01-02", 17, 18, 17, 17.8)]))
        adapter = CBOEAdapter()
        out = adapter.vix_term_structure()
        assert out["vix"] == pytest.approx(14.0)
        assert out["vix9d"] == pytest.approx(12.5)
        assert out["vix3m"] == pytest.approx(16.5)
        assert out["slope_30d_3m"] == pytest.approx(2.5)
        assert out["contango"] is True

    def test_backwardation_not_flagged(self, mock: rm_module.Mocker):
        # VIX > VIX3M → backwardation (slope_30d_3m < 0)
        mock.get(_CBOE_URL.format(sym="VIX"), text=_csv("VIX", [("2026-01-02", 30, 31, 29, 30.0)]))
        mock.get(_CBOE_URL.format(sym="VIX9D"), text=_csv("VIX9D", [("2026-01-02", 28, 29, 28, 28.5)]))
        mock.get(_CBOE_URL.format(sym="VIX3M"), text=_csv("VIX3M", [("2026-01-02", 25, 26, 25, 25.5)]))
        mock.get(_CBOE_URL.format(sym="VIX6M"), text=_csv("VIX6M", [("2026-01-02", 23, 24, 23, 23.5)]))
        adapter = CBOEAdapter()
        out = adapter.vix_term_structure()
        assert out["contango"] is False

    def test_missing_data_yields_nan_slope(self, mock: rm_module.Mocker):
        # All four endpoints 404
        for sym in ("VIX", "VIX9D", "VIX3M", "VIX6M"):
            mock.get(_CBOE_URL.format(sym=sym), status_code=404)
        adapter = CBOEAdapter()
        out = adapter.vix_term_structure()
        assert math.isnan(out["vix"])
        assert math.isnan(out["slope_9d_30d"])
        assert math.isnan(out["slope_30d_3m"])
        assert out["contango"] is False  # NaN comparison defaults to False


class TestSkewIndex:
    def test_returns_level_and_percentile(self, mock: rm_module.Mocker):
        # Build 60 rows so the percentile branch (>= 50) runs
        rows = [(f"2026-01-{i+1:02d}" if i < 31 else f"2026-02-{i-30:02d}",
                 100.0, 105.0, 100.0, 130.0 + (i % 5)) for i in range(60)]
        mock.get(_CBOE_URL.format(sym="SKEW"), text=_csv("SKEW", rows))
        adapter = CBOEAdapter()
        out = adapter.skew_index()
        assert "skew" in out
        assert "skew_pct_1y" in out
        assert 0.0 <= out["skew_pct_1y"] <= 1.0

    def test_short_history_returns_nan_percentile(self, mock: rm_module.Mocker):
        # Fewer than 50 rows → percentile is NaN
        rows = [("2026-01-01", 100.0, 105.0, 100.0, 130.0)]
        mock.get(_CBOE_URL.format(sym="SKEW"), text=_csv("SKEW", rows))
        adapter = CBOEAdapter()
        out = adapter.skew_index()
        assert out["skew"] == pytest.approx(130.0)
        assert math.isnan(out["skew_pct_1y"])

    def test_404_returns_both_nan(self, mock: rm_module.Mocker):
        mock.get(_CBOE_URL.format(sym="SKEW"), status_code=404)
        adapter = CBOEAdapter()
        out = adapter.skew_index()
        assert math.isnan(out["skew"])
        assert math.isnan(out["skew_pct_1y"])


class TestSnapshot:
    def test_combines_all_signals(self, mock: rm_module.Mocker):
        for sym in ("VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX"):
            mock.get(_CBOE_URL.format(sym=sym), text=_csv(sym, [("2026-01-02", 15, 16, 15, 15.5)]))
        mock.get(_CBOE_URL.format(sym="SKEW"), text=_csv("SKEW", [("2026-01-02", 100, 105, 100, 130.0)]))
        adapter = CBOEAdapter()
        snap = adapter.snapshot()
        for key in ("vix", "vix9d", "vix3m", "vix6m", "vvix", "skew", "skew_pct_1y", "contango"):
            assert key in snap
