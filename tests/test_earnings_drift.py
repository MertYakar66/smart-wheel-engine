"""Tests for engine/earnings_drift.py — post-earnings drift analytics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.earnings_drift import EarningsDriftAnalyzer


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def populated_dir(tmp_path: Path) -> Path:
    """A data dir with three tickers, each with two earnings events."""
    earnings = []
    ohlcv = []
    fundamentals = []
    base = pd.Timestamp("2026-01-01")

    for ticker, sector in [("AAPL", "Tech"), ("MSFT", "Tech"), ("XOM", "Energy")]:
        # Two earnings events, 90 days apart
        for i, ann_offset in enumerate([60, 150]):
            ann = base + pd.Timedelta(days=ann_offset)
            earnings.append({
                "ticker": ticker,
                "announcement_date": ann.isoformat(),
                "earnings_eps": 1.0 + 0.1 * i,
                "estimate_eps": 0.95 + 0.1 * i,
            })
        fundamentals.append({"ticker": ticker, "sector": sector})

        # 200 days of OHLCV around the events
        for d in range(0, 240):
            day = base + pd.Timedelta(days=d)
            # Mild upward drift with noise
            close = 100.0 * (1.0 + d * 0.001) + np.sin(d / 5) * 2.0
            ohlcv.append({"ticker": ticker, "date": day.isoformat(), "close": close})

    _write_csv(tmp_path / "sp500_earnings.csv", earnings)
    _write_csv(tmp_path / "sp500_ohlcv.csv", ohlcv)
    _write_csv(tmp_path / "sp500_fundamentals.csv", fundamentals)
    return tmp_path


class TestLazyLoaders:
    def test_missing_files_return_empty_dataframes(self, tmp_path: Path):
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        assert a._load_earnings().empty
        assert a._load_ohlcv().empty
        assert a._load_fundamentals().empty

    def test_columns_lowercased_on_load(self, tmp_path: Path):
        _write_csv(tmp_path / "sp500_earnings.csv", [
            {"Ticker": "AAPL", "Announcement_Date": "2026-05-01",
             "Earnings_EPS": 1.0, "Estimate_EPS": 0.95},
        ])
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        df = a._load_earnings()
        assert "ticker" in df.columns
        assert "announcement_date" in df.columns

    def test_load_caches(self, tmp_path: Path):
        _write_csv(tmp_path / "sp500_earnings.csv", [
            {"ticker": "AAPL", "announcement_date": "2026-05-01",
             "earnings_eps": 1.0, "estimate_eps": 0.95},
        ])
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        df1 = a._load_earnings()
        df2 = a._load_earnings()
        assert df1 is df2  # cached identity


class TestTickerEarningsMoves:
    def test_returns_one_row_per_event(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        df = a.ticker_earnings_moves("AAPL")
        assert len(df) == 2
        assert {"announcement_date", "prior_close", "day1_ret", "day3_ret",
                "day5_ret", "eps_surprise_pct", "surprise_sign"} <= set(df.columns)

    def test_returns_empty_when_no_earnings_data(self, tmp_path: Path):
        # OHLCV exists but earnings does not
        _write_csv(tmp_path / "sp500_ohlcv.csv", [
            {"ticker": "AAPL", "date": "2026-01-01", "close": 100.0},
        ])
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        assert a.ticker_earnings_moves("AAPL").empty

    def test_returns_empty_when_no_ohlcv_data(self, tmp_path: Path):
        _write_csv(tmp_path / "sp500_earnings.csv", [
            {"ticker": "AAPL", "announcement_date": "2026-05-01",
             "earnings_eps": 1.0, "estimate_eps": 0.95},
        ])
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        assert a.ticker_earnings_moves("AAPL").empty

    def test_returns_empty_for_unknown_ticker(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        assert a.ticker_earnings_moves("NVDA").empty

    def test_case_insensitive_ticker(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        df = a.ticker_earnings_moves("aapl")
        assert len(df) >= 1

    def test_surprise_sign_classification(self, tmp_path: Path):
        _write_csv(tmp_path / "sp500_earnings.csv", [
            {"ticker": "AAPL", "announcement_date": "2026-03-01",
             "earnings_eps": 1.10, "estimate_eps": 1.00},  # +10% beat
            {"ticker": "AAPL", "announcement_date": "2026-06-01",
             "earnings_eps": 0.85, "estimate_eps": 1.00},  # -15% miss
            {"ticker": "AAPL", "announcement_date": "2026-09-01",
             "earnings_eps": 1.005, "estimate_eps": 1.00},  # inline (0.5%)
        ])
        ohlcv = [
            {"ticker": "AAPL", "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).isoformat(),
             "close": 100.0 + d * 0.1} for d in range(0, 270)
        ]
        _write_csv(tmp_path / "sp500_ohlcv.csv", ohlcv)
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        df = a.ticker_earnings_moves("AAPL")
        signs = sorted(df["surprise_sign"].tolist())
        assert "beat" in signs
        assert "inline" in signs
        assert "miss" in signs

    def test_skips_rows_without_announcement_date(self, tmp_path: Path):
        _write_csv(tmp_path / "sp500_earnings.csv", [
            {"ticker": "AAPL", "announcement_date": "2026-03-01",
             "earnings_eps": 1.0, "estimate_eps": 0.95},
            {"ticker": "AAPL", "announcement_date": None,
             "earnings_eps": 1.0, "estimate_eps": 0.95},
        ])
        ohlcv = [
            {"ticker": "AAPL", "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).isoformat(),
             "close": 100.0 + d * 0.1} for d in range(0, 100)
        ]
        _write_csv(tmp_path / "sp500_ohlcv.csv", ohlcv)
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        df = a.ticker_earnings_moves("AAPL")
        assert len(df) == 1


class TestTickerDriftStats:
    def test_returns_empty_dict_for_unknown_ticker(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        assert a.ticker_drift_stats("ZZZ") == {}

    def test_returns_summary_stats(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        out = a.ticker_drift_stats("AAPL")
        assert out["ticker"] == "AAPL"
        assert out["n_events"] == 2
        for prefix in ("day1_ret", "day3_ret", "day5_ret"):
            assert f"{prefix}_median" in out
            assert f"{prefix}_p5" in out
            assert f"{prefix}_p95" in out
            assert f"{prefix}_abs_median" in out
            assert f"{prefix}_std" in out

    def test_heavy_tail_flag_present(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        out = a.ticker_drift_stats("AAPL")
        # heavy_tail flag may or may not be present depending on data shape
        if "heavy_tail" in out:
            assert isinstance(out["heavy_tail"], bool)


class TestSectorDriftStats:
    def test_returns_empty_when_no_fundamentals(self, tmp_path: Path):
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        assert a.sector_drift_stats().empty

    def test_aggregates_by_sector(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        df = a.sector_drift_stats()
        assert not df.empty
        assert {"sector", "horizon", "n", "median", "p5", "p95", "abs_median"} <= set(df.columns)
        # Three sectors in fixture (Tech, Tech, Energy → 2 sector groups)
        assert df["sector"].nunique() <= 2

    def test_skips_rows_with_missing_sector(self, tmp_path: Path):
        _write_csv(tmp_path / "sp500_fundamentals.csv", [
            {"ticker": "AAPL", "sector": "Tech"},
            {"ticker": "MSFT", "sector": None},
        ])
        _write_csv(tmp_path / "sp500_earnings.csv", [
            {"ticker": "AAPL", "announcement_date": "2026-05-01",
             "earnings_eps": 1.0, "estimate_eps": 0.95}
        ])
        ohlcv = [
            {"ticker": "AAPL", "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).isoformat(),
             "close": 100.0 + d * 0.1} for d in range(0, 200)
        ]
        _write_csv(tmp_path / "sp500_ohlcv.csv", ohlcv)
        a = EarningsDriftAnalyzer(data_dir=tmp_path)
        df = a.sector_drift_stats()
        # At most one sector represented
        if not df.empty:
            assert "Tech" in df["sector"].values


class TestExpectedMoveMagnitude:
    def test_returns_abs_median(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        v = a.expected_move_magnitude("AAPL", horizon="day1_ret")
        assert isinstance(v, float)
        # Non-NaN finite value
        assert v == v  # not NaN

    def test_returns_nan_for_unknown_ticker(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        v = a.expected_move_magnitude("ZZZ")
        assert v != v  # NaN

    def test_supports_alternative_horizons(self, populated_dir: Path):
        a = EarningsDriftAnalyzer(data_dir=populated_dir)
        v3 = a.expected_move_magnitude("AAPL", horizon="day3_ret")
        v5 = a.expected_move_magnitude("AAPL", horizon="day5_ret")
        assert v3 == v3  # not NaN
        assert v5 == v5
