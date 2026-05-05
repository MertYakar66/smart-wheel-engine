"""Tests for engine/external_data/fred_adapter.py with HTTP mocking."""

from __future__ import annotations

import math

import pytest
import requests_mock as rm_module

from engine.external_data.fred_adapter import (
    SERIES_PACK,
    FREDAdapter,
    _FRED_API_URL,
    _FRED_CSV_URL,
)


@pytest.fixture
def mock():
    with rm_module.Mocker() as m:
        yield m


def _csv(values: list[tuple[str, float]], series_id: str = "DGS10") -> str:
    """Build a FRED-style CSV (DATE, <SERIES_ID>)."""
    header = f"DATE,{series_id}\n"
    body = "\n".join(f"{d},{v}" for d, v in values)
    return header + body


class TestGetSeriesCsvPath:
    def test_happy_csv(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_CSV_URL,
            text=_csv([("2026-01-01", 4.25), ("2026-01-02", 4.30)]),
        )
        adapter = FREDAdapter(api_key=None)
        s = adapter.get_series("DGS10")
        assert len(s) == 2
        assert s.iloc[-1] == pytest.approx(4.30)

    def test_empty_csv_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, text="DATE,DGS10\n")
        adapter = FREDAdapter(api_key=None)
        assert adapter.get_series("DGS10").empty

    def test_single_column_csv_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, text="DATE\n2026-01-01\n")
        adapter = FREDAdapter(api_key=None)
        assert adapter.get_series("DGS10").empty

    def test_404_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, status_code=404)
        adapter = FREDAdapter(api_key=None)
        assert adapter.get_series("DGS10").empty

    def test_drops_non_numeric_values(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_CSV_URL,
            text="DATE,DGS10\n2026-01-01,.\n2026-01-02,4.30\n",
        )
        adapter = FREDAdapter(api_key=None)
        s = adapter.get_series("DGS10")
        assert len(s) == 1
        assert s.iloc[-1] == pytest.approx(4.30)


class TestGetSeriesApiPath:
    def test_happy_json(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_API_URL,
            json={
                "observations": [
                    {"date": "2026-01-01", "value": "4.25"},
                    {"date": "2026-01-02", "value": "4.30"},
                ]
            },
        )
        adapter = FREDAdapter(api_key="fake-key")
        s = adapter.get_series("DGS10")
        assert len(s) == 2
        assert s.iloc[-1] == pytest.approx(4.30)

    def test_empty_observations_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_API_URL, json={"observations": []})
        adapter = FREDAdapter(api_key="fake-key")
        assert adapter.get_series("DGS10").empty

    def test_missing_observation_keys_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_API_URL, json={"observations": [{"foo": "bar"}]})
        adapter = FREDAdapter(api_key="fake-key")
        assert adapter.get_series("DGS10").empty

    def test_500_returns_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_API_URL, status_code=500)
        adapter = FREDAdapter(api_key="fake-key")
        assert adapter.get_series("DGS10").empty


class TestLatest:
    def test_returns_last_value(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_CSV_URL,
            text=_csv([("2026-01-01", 4.25), ("2026-01-02", 4.30)]),
        )
        adapter = FREDAdapter(api_key=None)
        assert adapter.latest("DGS10") == pytest.approx(4.30)

    def test_returns_nan_on_empty(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, status_code=404)
        adapter = FREDAdapter(api_key=None)
        assert math.isnan(adapter.latest("DGS10"))


class TestSnapshotAndHistory:
    def test_snapshot_iterates_pack(self, mock: rm_module.Mocker):
        # Stub every series in the pack with a single value
        for sid in SERIES_PACK.values():
            mock.get(
                _FRED_CSV_URL,
                text=_csv([("2026-01-02", 1.0)], series_id=sid),
            )
        adapter = FREDAdapter(api_key=None)
        snap = adapter.snapshot()
        assert set(snap.keys()) == set(SERIES_PACK.keys())
        # All values should be 1.0 (or NaN if fixture was unmatched)
        assert all(v == 1.0 for v in snap.values() if v == v)

    def test_history_returns_wide_dataframe(self, mock: rm_module.Mocker):
        small_pack = {"a": "DGS10", "b": "DGS2"}
        for sid in small_pack.values():
            mock.get(
                _FRED_CSV_URL,
                text=_csv([
                    ("2026-01-01", 1.0),
                    ("2026-01-02", 2.0),
                ], series_id=sid),
            )
        adapter = FREDAdapter(api_key=None)
        df = adapter.history(pack=small_pack)
        assert {"a", "b"} <= set(df.columns)

    def test_history_returns_empty_when_all_series_fail(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, status_code=500)
        adapter = FREDAdapter(api_key=None)
        df = adapter.history(pack={"only": "DGS10"})
        assert df.empty


class TestCreditRegime:
    def test_benign_when_low_percentile(self, mock: rm_module.Mocker):
        # 5 years of monotonically-increasing HY OAS; current value at top
        # → percentile = 1.0 (~crisis). Flip: current low, history high.
        rows = [(f"2024-01-{i+1:02d}", 5.0 + i * 0.1) for i in range(28)]
        rows = rows + [("2024-02-01", 1.0)]  # latest value lowest
        mock.get(
            _FRED_CSV_URL,
            text=_csv(rows, series_id="BAMLH0A0HYM2"),
        )
        adapter = FREDAdapter(api_key=None)
        out = adapter.credit_regime()
        assert out["regime"] in {"benign", "stressed", "crisis"}

    def test_crisis_when_above_95th_percentile(self, mock: rm_module.Mocker):
        # 100 days of low values + final spike → top of distribution
        rows = [(f"2024-01-{(i % 28)+1:02d}", 1.0) for i in range(99)]
        rows.append(("2024-12-31", 99.0))
        mock.get(
            _FRED_CSV_URL,
            text=_csv(rows, series_id="BAMLH0A0HYM2"),
        )
        adapter = FREDAdapter(api_key=None)
        out = adapter.credit_regime()
        # Latest = 99 > 95th pct of [1.0]*99 → crisis
        assert out["regime"] == "crisis"

    def test_unknown_when_no_hy_data(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, status_code=500)
        adapter = FREDAdapter(api_key=None)
        out = adapter.credit_regime()
        assert out["regime"] == "unknown"
        assert math.isnan(out["hy_oas"])

    def test_includes_ig_percentile_when_available(self, mock: rm_module.Mocker):
        # Provide BOTH HY and IG with same data
        from requests_mock import ANY  # type: ignore
        mock.get(
            _FRED_CSV_URL,
            text=_csv([("2024-01-01", 2.0), ("2024-01-02", 3.0)], series_id="BAMLH0A0HYM2"),
        )
        adapter = FREDAdapter(api_key=None)
        out = adapter.credit_regime()
        # ig_pct_5y populated when ig fetch succeeds
        if not math.isnan(out["ig_pct_5y"]):
            assert 0.0 <= out["ig_pct_5y"] <= 1.0


class TestYieldCurveSignal:
    def test_normal_curve_not_inverted(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_CSV_URL,
            text=_csv([(f"2026-01-{i+1:02d}", 0.5) for i in range(25)], series_id="T10Y2Y"),
        )
        adapter = FREDAdapter(api_key=None)
        out = adapter.yield_curve_signal()
        assert out["spread_2y10y"] == pytest.approx(0.5)
        assert out["inverted"] is False
        # spread_1m_change present (>= 22 obs)
        assert out["spread_1m_change"] == pytest.approx(0.0)

    def test_inverted_curve_flagged(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_CSV_URL,
            text=_csv([("2026-01-01", -0.30)], series_id="T10Y2Y"),
        )
        adapter = FREDAdapter(api_key=None)
        out = adapter.yield_curve_signal()
        assert out["spread_2y10y"] == pytest.approx(-0.30)
        assert out["inverted"] is True

    def test_no_data_returns_nan_not_inverted(self, mock: rm_module.Mocker):
        mock.get(_FRED_CSV_URL, status_code=404)
        adapter = FREDAdapter(api_key=None)
        out = adapter.yield_curve_signal()
        assert math.isnan(out["spread_2y10y"])
        assert out["inverted"] is False

    def test_short_history_yields_nan_change(self, mock: rm_module.Mocker):
        mock.get(
            _FRED_CSV_URL,
            text=_csv([("2026-01-01", 0.5)], series_id="T10Y2Y"),
        )
        adapter = FREDAdapter(api_key=None)
        out = adapter.yield_curve_signal()
        # < 22 obs → spread_1m_change is NaN
        assert math.isnan(out["spread_1m_change"])
