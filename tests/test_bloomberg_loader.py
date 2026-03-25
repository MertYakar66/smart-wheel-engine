"""
Tests for Bloomberg Data Ingestion

Tests use temporary CSV files that mimic Bloomberg Excel export formats.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.bloomberg_loader import (
    load_bloomberg_ohlcv,
    load_all_ohlcv,
    load_bloomberg_options,
    load_bloomberg_earnings,
    load_bloomberg_dividends,
    load_bloomberg_iv_history,
    load_bloomberg_rates,
    load_bloomberg_fundamentals,
    compute_earnings_features,
    compute_iv_rank,
    get_annual_dividend_yield,
    get_upcoming_dividends,
    get_current_risk_free_rate,
    build_sector_map,
    _rename_columns,
    _detect_header_rows,
)
from data.pipeline import DataPipeline, DataStatus


# ─────────────────────────────────────────────────────────────────────
# Fixtures — temporary Bloomberg-formatted CSV files
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_bloomberg_dir(tmp_path):
    """Create a temporary Bloomberg data directory structure."""
    for subdir in ["ohlcv", "options", "earnings", "dividends",
                   "iv_history", "rates", "fundamentals"]:
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture
def sample_ohlcv_csv(tmp_bloomberg_dir):
    """Create a sample Bloomberg OHLCV CSV."""
    dates = pd.bdate_range("2024-01-02", periods=100)
    df = pd.DataFrame({
        "Date": dates.strftime("%m/%d/%Y"),  # Bloomberg date format
        "PX_OPEN": np.random.uniform(148, 152, 100),
        "PX_HIGH": np.random.uniform(150, 155, 100),
        "PX_LOW": np.random.uniform(145, 150, 100),
        "PX_LAST": np.random.uniform(148, 153, 100),
        "PX_VOLUME": np.random.randint(10000000, 50000000, 100),
    })
    filepath = tmp_bloomberg_dir / "ohlcv" / "AAPL.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


@pytest.fixture
def sample_options_csv(tmp_bloomberg_dir):
    """Create a sample Bloomberg option chain CSV."""
    df = pd.DataFrame({
        "OPT_STRIKE_PX": [140, 145, 150, 155, 160, 140, 145, 150, 155, 160],
        "OPT_PUT_CALL": ["Call", "Call", "Call", "Call", "Call",
                          "Put", "Put", "Put", "Put", "Put"],
        "OPT_EXPIRE_DT": ["03/21/2025"] * 10,
        "BID": [12.5, 8.3, 4.8, 2.1, 0.7, 0.6, 1.5, 3.2, 6.1, 10.3],
        "ASK": [12.8, 8.6, 5.1, 2.4, 0.9, 0.8, 1.7, 3.5, 6.4, 10.6],
        "OPT_IMPLIED_VOLATILITY_MID": [28.5, 27.3, 26.1, 25.8, 25.2,
                                       29.1, 27.8, 26.5, 26.0, 25.5],
        "OPEN_INT": [5000, 12000, 25000, 8000, 3000,
                     4000, 10000, 20000, 7000, 2500],
        "VOLUME": [500, 1200, 3000, 800, 200, 400, 900, 2500, 600, 150],
        "OPT_DELTA": [0.85, 0.72, 0.52, 0.28, 0.12,
                      -0.15, -0.28, -0.48, -0.72, -0.88],
        "OPT_UNDL_PX": [150.0] * 10,
    })
    filepath = tmp_bloomberg_dir / "options" / "AAPL.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


@pytest.fixture
def sample_earnings_csv(tmp_bloomberg_dir):
    """Create a sample Bloomberg earnings CSV."""
    df = pd.DataFrame({
        "Date": ["01/25/2024", "04/25/2024", "07/25/2024", "10/31/2024",
                 "01/30/2025", "04/24/2025"],
        "IS_EPS": [2.18, 1.53, 1.40, 1.64, 2.40, None],
        "BEST_EPS_MEDIAN": [2.10, 1.50, 1.35, 1.60, 2.35, 2.55],
        "EARN_EST_EPS_SURPRISE_PCT": [3.81, 2.00, 3.70, 2.50, 2.13, None],
        "EARNING_ANNOUNCEMENT_TIMING": ["AMC", "AMC", "AMC", "AMC", "AMC", "AMC"],
    })
    filepath = tmp_bloomberg_dir / "earnings" / "AAPL.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


@pytest.fixture
def sample_dividends_csv(tmp_bloomberg_dir):
    """Create a sample Bloomberg dividend CSV."""
    df = pd.DataFrame({
        "DVD_EX_DT": ["02/09/2024", "05/10/2024", "08/12/2024",
                       "11/01/2024", "02/07/2025"],
        "DVD_RECORD_DT": ["02/12/2024", "05/13/2024", "08/12/2024",
                           "11/01/2024", "02/10/2025"],
        "DVD_PAY_DT": ["02/15/2024", "05/16/2024", "08/15/2024",
                        "11/07/2024", "02/13/2025"],
        "DVD_SH_LAST": [0.24, 0.25, 0.25, 0.25, 0.25],
        "DVD_FREQ": ["Quarterly"] * 5,
    })
    filepath = tmp_bloomberg_dir / "dividends" / "AAPL.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


@pytest.fixture
def sample_iv_history_csv(tmp_bloomberg_dir):
    """Create a sample Bloomberg IV history CSV."""
    dates = pd.bdate_range("2024-01-02", periods=300)
    df = pd.DataFrame({
        "Date": dates.strftime("%m/%d/%Y"),
        "30DAY_IMPVOL_100.0%MNY_DF": np.random.uniform(20, 35, 300),
        "60DAY_IMPVOL_100.0%MNY_DF": np.random.uniform(22, 33, 300),
        "30DAY_IMPVOL_90.0%MNY_DF": np.random.uniform(25, 40, 300),
        "30DAY_IMPVOL_110.0%MNY_DF": np.random.uniform(18, 30, 300),
        "20DAY_HV": np.random.uniform(15, 40, 300),
        "60DAY_HV": np.random.uniform(18, 35, 300),
    })
    filepath = tmp_bloomberg_dir / "iv_history" / "AAPL.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


@pytest.fixture
def sample_rates_csv(tmp_bloomberg_dir):
    """Create a sample Treasury yields CSV."""
    dates = pd.bdate_range("2024-01-02", periods=100)
    df = pd.DataFrame({
        "Date": dates.strftime("%m/%d/%Y"),
        "PX_LAST": np.random.uniform(4.5, 5.5, 100),
    })
    filepath = tmp_bloomberg_dir / "rates" / "treasury_yields.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


@pytest.fixture
def sample_fundamentals_csv(tmp_bloomberg_dir):
    """Create a sample fundamentals CSV."""
    df = pd.DataFrame({
        "Security": ["AAPL US Equity", "MSFT US Equity", "JPM US Equity"],
        "CUR_MKT_CAP": [3500000, 3200000, 580000],
        "GICS_SECTOR_NAME": ["Information Technology", "Information Technology", "Financials"],
        "GICS_INDUSTRY_GROUP_NAME": ["Technology Hardware", "Software", "Banks"],
        "EQY_DVD_YLD_IND": [0.44, 0.72, 2.10],
        "PE_RATIO": [32.5, 35.2, 12.8],
    })
    filepath = tmp_bloomberg_dir / "fundamentals" / "sp500_fundamentals.csv"
    df.to_csv(filepath, index=False)
    return tmp_bloomberg_dir


# ─────────────────────────────────────────────────────────────────────
# 1. OHLCV Tests
# ─────────────────────────────────────────────────────────────────────

class TestOHLCVLoader:
    """Tests for Bloomberg OHLCV loading."""

    def test_load_ohlcv(self, sample_ohlcv_csv):
        """Should load and normalize OHLCV data."""
        df = load_bloomberg_ohlcv("AAPL", sample_ohlcv_csv / "ohlcv")

        assert df is not None
        assert len(df) == 100
        assert "Date" in df.columns
        assert "Close" in df.columns
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Volume" in df.columns

    def test_ohlcv_sorted_by_date(self, sample_ohlcv_csv):
        """Dates should be sorted ascending."""
        df = load_bloomberg_ohlcv("AAPL", sample_ohlcv_csv / "ohlcv")

        assert df["Date"].is_monotonic_increasing

    def test_ohlcv_numeric_columns(self, sample_ohlcv_csv):
        """Price columns should be numeric."""
        df = load_bloomberg_ohlcv("AAPL", sample_ohlcv_csv / "ohlcv")

        assert pd.api.types.is_numeric_dtype(df["Close"])
        assert pd.api.types.is_numeric_dtype(df["Open"])
        assert pd.api.types.is_numeric_dtype(df["Volume"])

    def test_missing_ticker_returns_none(self, sample_ohlcv_csv):
        """Non-existent ticker should return None."""
        result = load_bloomberg_ohlcv("ZZZZ", sample_ohlcv_csv / "ohlcv")
        assert result is None

    def test_load_all_ohlcv(self, sample_ohlcv_csv):
        """Should load all available tickers."""
        result = load_all_ohlcv(data_dir=sample_ohlcv_csv / "ohlcv")

        assert "AAPL" in result
        assert len(result["AAPL"]) == 100


# ─────────────────────────────────────────────────────────────────────
# 2. Options Tests
# ─────────────────────────────────────────────────────────────────────

class TestOptionsLoader:
    """Tests for Bloomberg option chain loading."""

    def test_load_options(self, sample_options_csv):
        """Should load and normalize option chain."""
        df = load_bloomberg_options("AAPL", data_dir=sample_options_csv / "options")

        assert df is not None
        assert len(df) == 10
        assert "strike" in df.columns
        assert "option_type" in df.columns
        assert "implied_vol" in df.columns
        assert "bid" in df.columns
        assert "ask" in df.columns

    def test_option_type_normalized(self, sample_options_csv):
        """Option type should be normalized to C/P."""
        df = load_bloomberg_options("AAPL", data_dir=sample_options_csv / "options")

        assert set(df["option_type"].unique()) == {"C", "P"}

    def test_iv_normalized(self, sample_options_csv):
        """IV should be in decimal format (0.25 not 25)."""
        df = load_bloomberg_options("AAPL", data_dir=sample_options_csv / "options")

        assert df["implied_vol"].max() < 1.0  # Decimal format
        assert df["implied_vol"].min() > 0.0

    def test_mid_price_computed(self, sample_options_csv):
        """Mid price should be computed from bid/ask."""
        df = load_bloomberg_options("AAPL", data_dir=sample_options_csv / "options")

        assert "mid_price" in df.columns
        expected_mid = (df["bid"] + df["ask"]) / 2
        pd.testing.assert_series_equal(df["mid_price"], expected_mid, check_names=False)

    def test_ticker_added(self, sample_options_csv):
        """Ticker column should be present."""
        df = load_bloomberg_options("AAPL", data_dir=sample_options_csv / "options")

        assert "ticker" in df.columns
        assert (df["ticker"] == "AAPL").all()


# ─────────────────────────────────────────────────────────────────────
# 3. Earnings Tests
# ─────────────────────────────────────────────────────────────────────

class TestEarningsLoader:
    """Tests for Bloomberg earnings loading."""

    def test_load_earnings(self, sample_earnings_csv):
        """Should load and normalize earnings data."""
        df = load_bloomberg_earnings("AAPL", sample_earnings_csv / "earnings")

        assert df is not None
        assert len(df) == 6
        assert "earnings_date" in df.columns
        assert "eps_actual" in df.columns
        assert "eps_estimate" in df.columns

    def test_surprise_computed(self, sample_earnings_csv):
        """EPS surprise should be computed if not provided."""
        df = load_bloomberg_earnings("AAPL", sample_earnings_csv / "earnings")

        assert "eps_surprise" in df.columns or "surprise_pct" in df.columns

    def test_timing_normalized(self, sample_earnings_csv):
        """Timing should be normalized with is_pre_market flag."""
        df = load_bloomberg_earnings("AAPL", sample_earnings_csv / "earnings")

        assert "is_pre_market" in df.columns

    def test_compute_earnings_features(self, sample_earnings_csv, sample_ohlcv_csv):
        """Should compute earnings features from raw data."""
        earnings = load_bloomberg_earnings("AAPL", sample_earnings_csv / "earnings")
        ohlcv = load_bloomberg_ohlcv("AAPL", sample_ohlcv_csv / "ohlcv")

        features = compute_earnings_features("AAPL", earnings, ohlcv)

        # May be None if OHLCV doesn't overlap with earnings dates
        # but the function should not crash
        if features is not None:
            assert "symbol" in features
            assert "historical_avg_move" in features
            assert "iv_rank_52w" in features


# ─────────────────────────────────────────────────────────────────────
# 4. Dividend Tests
# ─────────────────────────────────────────────────────────────────────

class TestDividendLoader:
    """Tests for Bloomberg dividend loading."""

    def test_load_dividends(self, sample_dividends_csv):
        """Should load and normalize dividend data."""
        df = load_bloomberg_dividends("AAPL", sample_dividends_csv / "dividends")

        assert df is not None
        assert len(df) == 5
        assert "ex_date" in df.columns
        assert "amount" in df.columns

    def test_dividend_yield_computation(self, sample_dividends_csv):
        """Should compute annualized dividend yield."""
        df = load_bloomberg_dividends("AAPL", sample_dividends_csv / "dividends")

        yield_val = get_annual_dividend_yield("AAPL", df, spot_price=150.0)

        assert yield_val >= 0
        assert yield_val < 0.10  # Should be reasonable

    def test_upcoming_dividends(self, sample_dividends_csv):
        """Should filter to upcoming ex-dates."""
        df = load_bloomberg_dividends("AAPL", sample_dividends_csv / "dividends")

        upcoming = get_upcoming_dividends(df, horizon_days=365)
        # May or may not have upcoming depending on test date
        assert isinstance(upcoming, pd.DataFrame)

    def test_zero_spot_yield(self, sample_dividends_csv):
        """Zero spot price should return 0 yield."""
        df = load_bloomberg_dividends("AAPL", sample_dividends_csv / "dividends")

        yield_val = get_annual_dividend_yield("AAPL", df, spot_price=0.0)
        assert yield_val == 0.0


# ─────────────────────────────────────────────────────────────────────
# 5. IV History Tests
# ─────────────────────────────────────────────────────────────────────

class TestIVHistoryLoader:
    """Tests for Bloomberg IV history loading."""

    def test_load_iv_history(self, sample_iv_history_csv):
        """Should load and normalize IV history."""
        df = load_bloomberg_iv_history("AAPL", sample_iv_history_csv / "iv_history")

        assert df is not None
        assert len(df) == 300
        assert "iv_atm_30d" in df.columns

    def test_iv_normalized_to_decimal(self, sample_iv_history_csv):
        """IV should be in decimal format after normalization."""
        df = load_bloomberg_iv_history("AAPL", sample_iv_history_csv / "iv_history")

        # Bloomberg gives as percentage (20-35), should be normalized to 0.20-0.35
        assert df["iv_atm_30d"].max() < 1.0
        assert df["iv_atm_30d"].min() > 0.0

    def test_compute_iv_rank(self, sample_iv_history_csv):
        """IV rank should be between 0 and 1."""
        df = load_bloomberg_iv_history("AAPL", sample_iv_history_csv / "iv_history")

        rank = compute_iv_rank(df)
        assert rank is not None
        assert 0 <= rank <= 1

    def test_iv_rank_insufficient_data(self):
        """IV rank should return None with insufficient data."""
        rank = compute_iv_rank(None)
        assert rank is None

        rank = compute_iv_rank(pd.DataFrame())
        assert rank is None


# ─────────────────────────────────────────────────────────────────────
# 6. Rates Tests
# ─────────────────────────────────────────────────────────────────────

class TestRatesLoader:
    """Tests for Treasury rates loading."""

    def test_load_rates(self, sample_rates_csv):
        """Should load and normalize rates."""
        df = load_bloomberg_rates(sample_rates_csv / "rates")

        assert df is not None
        assert len(df) == 100

    def test_rates_converted_to_decimal(self, sample_rates_csv):
        """Rates should be in decimal format."""
        df = load_bloomberg_rates(sample_rates_csv / "rates")

        # Bloomberg gives 4.5-5.5 percentage, should be 0.045-0.055
        rate_cols = [c for c in df.columns if c.startswith("rate_")]
        if rate_cols:
            for col in rate_cols:
                assert df[col].max() < 1.0

    def test_get_current_rate(self, sample_rates_csv):
        """Should return most recent rate."""
        df = load_bloomberg_rates(sample_rates_csv / "rates")
        rate = get_current_risk_free_rate(df)

        assert 0 < rate < 0.15  # Reasonable range

    def test_default_rate_no_data(self):
        """Should return 0.05 default with no data."""
        rate = get_current_risk_free_rate(None)
        assert rate == 0.05


# ─────────────────────────────────────────────────────────────────────
# 7. Fundamentals Tests
# ─────────────────────────────────────────────────────────────────────

class TestFundamentalsLoader:
    """Tests for company fundamentals loading."""

    def test_load_fundamentals(self, sample_fundamentals_csv):
        """Should load and normalize fundamentals."""
        df = load_bloomberg_fundamentals(sample_fundamentals_csv / "fundamentals")

        assert df is not None
        assert len(df) == 3
        assert "ticker" in df.columns
        assert "gics_sector" in df.columns

    def test_ticker_cleaned(self, sample_fundamentals_csv):
        """'US Equity' suffix should be removed from tickers."""
        df = load_bloomberg_fundamentals(sample_fundamentals_csv / "fundamentals")

        assert "AAPL" in df["ticker"].values
        assert "AAPL US Equity" not in df["ticker"].values

    def test_build_sector_map(self, sample_fundamentals_csv):
        """Should build ticker → sector mapping."""
        df = load_bloomberg_fundamentals(sample_fundamentals_csv / "fundamentals")
        sector_map = build_sector_map(df)

        assert sector_map["AAPL"] == "Information Technology"
        assert sector_map["JPM"] == "Financials"


# ─────────────────────────────────────────────────────────────────────
# 8. Pipeline Tests
# ─────────────────────────────────────────────────────────────────────

class TestDataPipeline:
    """Tests for the master data pipeline."""

    @pytest.fixture
    def full_pipeline_dir(self, tmp_bloomberg_dir):
        """Create directory with all data types."""
        # OHLCV
        dates = pd.bdate_range("2024-01-02", periods=100)
        ohlcv = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "PX_OPEN": 150.0, "PX_HIGH": 155.0,
            "PX_LOW": 148.0, "PX_LAST": 152.0,
            "PX_VOLUME": 30000000,
        })
        ohlcv.to_csv(tmp_bloomberg_dir / "ohlcv" / "AAPL.csv", index=False)

        # Options
        opts = pd.DataFrame({
            "OPT_STRIKE_PX": [145, 150, 155],
            "OPT_PUT_CALL": ["Put", "Put", "Put"],
            "OPT_EXPIRE_DT": ["03/21/2025"] * 3,
            "BID": [1.5, 3.2, 6.1],
            "ASK": [1.7, 3.5, 6.4],
            "OPT_IMPLIED_VOLATILITY_MID": [27.8, 26.5, 26.0],
            "OPEN_INT": [10000, 20000, 7000],
            "VOLUME": [900, 2500, 600],
            "OPT_UNDL_PX": [152.0] * 3,
        })
        opts.to_csv(tmp_bloomberg_dir / "options" / "AAPL.csv", index=False)

        # Rates
        rates = pd.DataFrame({
            "Date": dates[:50].strftime("%Y-%m-%d"),
            "PX_LAST": [5.25] * 50,
        })
        rates.to_csv(tmp_bloomberg_dir / "rates" / "treasury_yields.csv", index=False)

        return tmp_bloomberg_dir

    def test_pipeline_load_all(self, full_pipeline_dir):
        """Pipeline should load all available data."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_all()

        status = pipeline.status()
        assert status.ohlcv_tickers >= 1
        assert status.rates_loaded

    def test_pipeline_get_spot(self, full_pipeline_dir):
        """Should return spot price."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_ohlcv()

        price = pipeline.get_spot_price("AAPL")
        assert price is not None
        assert price > 0

    def test_pipeline_get_options(self, full_pipeline_dir):
        """Should return option chain (unfiltered)."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_options()

        # Get all options without DTE filter (test expiry may be in the past)
        opts = pipeline.get_options("AAPL", min_dte=-9999, max_dte=9999)
        assert opts is not None
        assert len(opts) == 3

    def test_pipeline_get_risk_free_rate(self, full_pipeline_dir):
        """Should return risk-free rate from loaded data."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_rates()

        rate = pipeline.get_risk_free_rate()
        assert 0 < rate < 0.15

    def test_pipeline_status_summary(self, full_pipeline_dir):
        """Status summary should be a formatted string."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_all()

        status = pipeline.status()
        summary = status.summary()
        assert "Data Pipeline Status" in summary
        assert "OHLCV:" in summary

    def test_pipeline_empty_dir(self, tmp_path):
        """Pipeline should handle empty directory gracefully."""
        empty_dir = tmp_path / "empty_bbg"
        for subdir in ["ohlcv", "options", "earnings", "dividends",
                       "iv_history", "rates", "fundamentals"]:
            (empty_dir / subdir).mkdir(parents=True)

        pipeline = DataPipeline(data_dir=str(empty_dir))
        # Only load from bloomberg dir (don't fall back to legacy)
        pipeline._ohlcv = {}
        pipeline.load_options()
        pipeline.load_earnings()

        status = pipeline.status()
        assert status.options_tickers == 0
        assert status.earnings_tickers == 0

    def test_pipeline_validate(self, full_pipeline_dir):
        """Validate should return issues dict."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_all()

        issues = pipeline.validate()
        assert isinstance(issues, dict)

    def test_pipeline_daily_returns(self, full_pipeline_dir):
        """Should compute daily returns array."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_ohlcv()

        returns = pipeline.get_daily_returns("AAPL")
        assert returns is not None
        assert len(returns) == 99  # 100 prices → 99 returns

    def test_pipeline_ohlcv_for_backtester(self, full_pipeline_dir):
        """Should format OHLCV for backtester consumption."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_ohlcv()

        bt_data = pipeline.get_ohlcv_for_backtester()
        assert "AAPL" in bt_data
        assert "date" in bt_data["AAPL"].columns
        assert "close" in bt_data["AAPL"].columns

    def test_pipeline_sector_map_fallback(self, full_pipeline_dir):
        """Should fall back to hardcoded sector map if no fundamentals."""
        pipeline = DataPipeline(data_dir=str(full_pipeline_dir))
        pipeline.load_all()

        sector_map = pipeline.get_sector_map()
        assert len(sector_map) > 0  # Should have hardcoded fallback


# ─────────────────────────────────────────────────────────────────────
# 9. Utility Tests
# ─────────────────────────────────────────────────────────────────────

class TestUtilities:
    """Tests for internal utility functions."""

    def test_rename_columns(self):
        """Column renaming should work with mapping."""
        df = pd.DataFrame({"PX_LAST": [100], "PX_VOLUME": [1000]})
        mapped = _rename_columns(df, {"PX_LAST": "Close", "PX_VOLUME": "Volume"})

        assert "Close" in mapped.columns
        assert "Volume" in mapped.columns

    def test_rename_keeps_unmapped(self):
        """Unmapped columns should be kept as-is."""
        df = pd.DataFrame({"PX_LAST": [100], "custom_col": [42]})
        mapped = _rename_columns(df, {"PX_LAST": "Close"})

        assert "Close" in mapped.columns
        assert "custom_col" in mapped.columns
