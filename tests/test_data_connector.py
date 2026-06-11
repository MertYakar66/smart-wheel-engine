"""
Tests for engine/data_connector.py — the MarketDataConnector Bloomberg-CSV
provider.

Coverage backfill: data_connector.py had no dedicated test file and was
exercised only indirectly. These tests drive every query method against
small synthetic CSVs written to a temp data_dir, covering the
data-present, data-absent, and edge-case branches with real assertions.
No production code is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector, normalize_ticker


def _csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(autouse=True)
def _reset_ohlcv_invariant_flag():
    """`_ohlcv_invariant_warned` is class-level state — reset around each
    test so the once-per-process warning gate does not leak between tests.
    """
    MarketDataConnector._ohlcv_invariant_warned = False
    yield
    MarketDataConnector._ohlcv_invariant_warned = False


@pytest.fixture
def data_dir(tmp_path):
    """A data_dir populated with small synthetic Bloomberg CSVs.

    OHLCV note: get_ohlcv renames open->high, high->close, close->open
    (the Bloomberg CSV ships with labels rotated one position). The rows
    here are written in that rotated on-disk layout so the post-rename
    OHLC is internally consistent — e.g. post-rename AAPL row 1 is
    open=100, high=106, low=97, close=103.
    """
    d = tmp_path / "bloomberg"
    d.mkdir()

    _csv(
        d / "sp500_ohlcv.csv",
        [
            {
                "date": "2024-01-02",
                "ticker": "AAPL UW Equity",
                "open": 106,
                "high": 103,
                "low": 97,
                "close": 100,
                "volume": 1_000_000,
            },
            {
                "date": "2024-01-03",
                "ticker": "AAPL UW Equity",
                "open": 109,
                "high": 105,
                "low": 99,
                "close": 101,
                "volume": 1_100_000,
            },
            {
                "date": "2024-01-04",
                "ticker": "MSFT UW Equity",
                "open": 212,
                "high": 205,
                "low": 198,
                "close": 200,
                "volume": 900_000,
            },
        ],
    )
    _csv(
        d / "sp500_vol_iv_full.csv",
        [
            {
                "date": f"2024-01-0{i}",
                "ticker": "AAPL UW Equity",
                "hist_put_imp_vol": v,
                "hist_call_imp_vol": v,
                "volatility_30d": 30.0,
                "volatility_60d": 31.0,
                "volatility_90d": 32.0,
                "volatility_260d": 33.0,
            }
            # IV in PERCENT (Bloomberg contract). The connector's R7/W8 gate
            # nulls sub-3.0 implied-vol cells, so synthetic vol_iv must be percent.
            for i, v in enumerate([10.0, 20.0, 30.0, 40.0, 50.0], start=1)
        ],
    )
    _csv(
        d / "treasury_yields.csv",
        [
            {
                "date": "2024-01-02",
                "rate_3m": 5.20,
                "rate_6m": 5.10,
                "rate_2y": 4.30,
                "rate_10y": 4.00,
            },
            {
                "date": "2024-01-03",
                "rate_3m": 5.25,
                "rate_6m": 5.15,
                "rate_2y": 4.35,
                "rate_10y": 4.05,
            },
        ],
    )
    _csv(
        d / "vix_term_structure.csv",
        [
            {"date": "2024-01-02", "vix": 14.0, "vix_3m": 16.0, "vix_6m": 17.0},
            {"date": "2024-01-03", "vix": 15.0, "vix_3m": 16.5, "vix_6m": 17.5},
        ],
    )
    _csv(
        d / "sp500_earnings.csv",
        [
            {
                "ticker": "AAPL UW Equity",
                "year/period": "2023 Q4",
                "announcement_date": "2024-02-01",
                "announcement_time": "AMC",
                "earnings_eps": 2.18,
                "comparable_eps": 2.10,
                "estimate_eps": 2.11,
            },
            {
                "ticker": "AAPL UW Equity",
                "year/period": "2024 Q1",
                "announcement_date": "2024-05-01",
                "announcement_time": "AMC",
                "earnings_eps": np.nan,
                "comparable_eps": 1.50,
                "estimate_eps": 1.52,
            },
        ],
    )
    _csv(
        d / "sp500_dividends.csv",
        [
            {
                "ticker": "AAPL UW Equity",
                "declared_date": "2024-01-25",
                "ex_date": "2024-02-09",
                "record_date": "2024-02-12",
                "payable_date": "2024-02-15",
                "dividend_amount": 0.24,
                "dividend_frequency": "Quarterly",
                "dividend_type": "Regular Cash",
            },
            {
                "ticker": "AAPL UW Equity",
                "declared_date": "2024-04-25",
                "ex_date": "2024-05-10",
                "record_date": "2024-05-13",
                "payable_date": "2024-05-16",
                "dividend_amount": 0.25,
                "dividend_frequency": "Quarterly",
                "dividend_type": "Regular Cash",
            },
        ],
    )
    _csv(
        d / "sp500_fundamentals.csv",
        [
            {
                "ticker": "AAPL UW Equity",
                "pe_ratio": 28.0,
                "best_pe_ratio": 26.0,
                "beta_raw_overridable": 1.2,
                "cur_mkt_cap": 3.0e12,
                "eqy_dvd_yld_12m": 0.5,
                "free_cash_flow_yield": 3.1,
                "return_com_eqy": 150.0,
                "tot_debt_to_tot_eqy": 140.0,
                "gics_sector_name": "Information Technology",
                "gics_industry_group_name": "Technology Hardware",
                "volatility_30d": 0.30,
                "30day_impvol_100.0%mny_df": 0.31,
            },
            {
                "ticker": "XOM UN Equity",
                "pe_ratio": 12.0,
                "best_pe_ratio": 11.0,
                "beta_raw_overridable": 0.9,
                "cur_mkt_cap": 4.0e11,
                "eqy_dvd_yld_12m": 3.4,
                "free_cash_flow_yield": 6.0,
                "return_com_eqy": 20.0,
                "tot_debt_to_tot_eqy": 25.0,
                "gics_sector_name": "Energy",
                "gics_industry_group_name": "Energy",
                "volatility_30d": 0.22,
                "30day_impvol_100.0%mny_df": 0.24,
            },
        ],
    )
    _csv(
        d / "sp500_credit_risk.csv",
        [
            {
                "ticker": "AAPL UW Equity",
                "altman_z_score": 8.1,
                "interest_coverage_ratio": 30.0,
                "rtg_sp_lt_lc_issuer_credit": "AA+",
            },
        ],
    )
    _csv(
        d / "sp500_liquidity.csv",
        [
            {
                "date": "2024-01-02",
                "ticker": "AAPL UW Equity",
                "avg_vol_30d": 5.5e7,
                "turnover": 0.9,
                "shares_out": 1.5e10,
            },
        ],
    )
    return d


# =====================================================================
# 1. normalize_ticker
# =====================================================================
class TestNormalizeTicker:
    def test_strips_equity_and_exchange_suffix(self):
        assert normalize_ticker("AAPL UW Equity") == "AAPL"
        assert normalize_ticker("A UN") == "A"

    def test_already_standard_unchanged(self):
        assert normalize_ticker("AAPL") == "AAPL"

    def test_dotted_root_preserved(self):
        # BRK/B keeps the slash; only the exchange suffix is stripped.
        assert normalize_ticker("BRK/B UN Equity") == "BRK/B"

    def test_unknown_suffix_kept(self):
        # "ZZ" is not a known exchange suffix — leave the string intact.
        assert normalize_ticker("AAPL ZZ") == "AAPL ZZ"

    def test_non_string_input_coerced(self):
        assert normalize_ticker(123) == "123"

    def test_static_method_delegates(self):
        assert MarketDataConnector.normalize_ticker("MSFT UW Equity") == "MSFT"


# =====================================================================
# 2. _load — lazy load, caching, missing / unreadable files
# =====================================================================
class TestLoad:
    def test_missing_file_returns_empty_and_caches(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        df = conn._load("ohlcv")
        assert df.empty
        # Cached: a second call returns the same cached object.
        assert conn._load("ohlcv") is df

    def test_unreadable_file_returns_empty(self, tmp_path):
        # A zero-byte CSV makes pandas raise EmptyDataError -> except branch.
        (tmp_path / "sp500_ohlcv.csv").write_text("")
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn._load("ohlcv").empty

    def test_load_normalizes_tickers_and_parses_dates(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn._load("ohlcv")
        assert "AAPL" in set(df["ticker"])
        assert "AAPL UW Equity" not in set(df["ticker"])
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_is_cached(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn._load("vix") is conn._load("vix")


# =====================================================================
# 3. OHLCV + the column-rename invariant guard
# =====================================================================
class TestOHLCV:
    def test_get_ohlcv_renames_and_indexes(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn.get_ohlcv("AAPL")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "date"
        # Row 1 post-rename: open=100, high=106, low=97, close=103.
        first = df.iloc[0]
        assert first["open"] == 100
        assert first["high"] == 106
        assert first["low"] == 97
        assert first["close"] == 103
        # high is the per-row max, low the per-row min (rename is correct).
        assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all()

    def test_get_ohlcv_unknown_ticker_empty(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_ohlcv("NOSUCH").empty

    def test_get_ohlcv_date_filter(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn.get_ohlcv("AAPL", start_date="2024-01-03")
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2024-01-03")

    def test_invariant_guard_quiet_on_clean_data(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        conn.get_ohlcv("AAPL")
        assert MarketDataConnector._ohlcv_invariant_warned is False

    def test_invariant_guard_fires_on_rotated_data(self, tmp_path, caplog):
        # Write 6 rows whose post-rename layout violates high >= max(o,c,l):
        # CSV 'open' (-> high) is tiny while CSV 'close' (-> open) is huge.
        rows = [
            {
                "date": f"2024-02-{i:02d}",
                "ticker": "BAD UW Equity",
                "open": 1,
                "high": 150,
                "low": 0.5,
                "close": 200,
                "volume": 1,
            }
            for i in range(1, 7)
        ]
        _csv(tmp_path / "sp500_ohlcv.csv", rows)
        conn = MarketDataConnector(data_dir=str(tmp_path))
        with caplog.at_level("CRITICAL"):
            conn.get_ohlcv("BAD")
        assert MarketDataConnector._ohlcv_invariant_warned is True
        assert any("OHLCV invariant violation" in r.message for r in caplog.records)


# =====================================================================
# 4. IV history, rank, percentile, vol-risk-premium
# =====================================================================
class TestVolatilityIV:
    def test_get_iv_history_columns(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        iv = conn.get_iv_history("AAPL")
        assert "hist_put_imp_vol" in iv.columns
        assert iv.index.name == "date"
        assert len(iv) == 5

    def test_get_iv_history_unknown_ticker_empty(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_iv_history("NOSUCH").empty

    def test_iv_rank_exact(self, data_dir):
        # Composite IV series = [10, 20, 30, 40, 50] (PERCENT), current 50.
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_iv_rank("AAPL") == pytest.approx(1.0)

    def test_iv_percentile_exact(self, data_dir):
        # 4 of 5 days strictly below the current 50 -> 0.8.
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_iv_percentile("AAPL") == pytest.approx(0.8)

    def test_iv_rank_with_as_of(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        rank = conn.get_iv_rank("AAPL", as_of="2024-01-05")
        assert 0.0 <= rank <= 1.0

    def test_iv_rank_nan_when_no_data(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert np.isnan(conn.get_iv_rank("NOSUCH"))

    def test_iv_rank_half_when_flat(self, tmp_path):
        # All-equal IV -> hi == lo -> 0.5 by definition.
        _csv(
            tmp_path / "sp500_vol_iv_full.csv",
            [
                {
                    "date": f"2024-01-0{i}",
                    "ticker": "FLAT",
                    "hist_put_imp_vol": 25.0,
                    "hist_call_imp_vol": 25.0,
                }
                for i in range(1, 4)
            ],
        )
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_iv_rank("FLAT") == 0.5

    def test_iv_percentile_nan_when_no_data(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert np.isnan(conn.get_iv_percentile("NOSUCH"))

    def test_vol_risk_premium(self, data_dir):
        # Last row: IV 50.0%, realized vol_30d 30.0% -> premium 20.0 points
        # (vol_risk_premium is IV - RV in the file's native PERCENT units).
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_vol_risk_premium("AAPL") == pytest.approx(20.0)

    def test_vol_risk_premium_nan_when_no_data(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert np.isnan(conn.get_vol_risk_premium("NOSUCH"))


# =====================================================================
# 4b. vol_iv implausible-IV cleaning gate (R7 / audit W1+W8 / #356 / #360)
# =====================================================================
class TestVolIvCleaningGate:
    """The connector NULLs implausible vol_iv implied-vol cells on the
    (default) monolith read — the sub-3.0 low-tail garbage and the ~134217.7
    high sentinel — keeping the row (and its realized-vol columns). So every
    served IV is unambiguous PERCENT (> 3.0), which makes the downstream
    ``if iv > 3.0`` percent->decimal conversion in wheel_runner / wheel_tracker
    always correct without a decision-trio edit (#356, #360)."""

    def _conn(self, tmp_path, rows):
        _csv(tmp_path / "sp500_vol_iv_full.csv", rows)
        return MarketDataConnector(data_dir=str(tmp_path))

    def test_sub_floor_implied_vol_nulled_row_and_realized_kept(self, tmp_path):
        conn = self._conn(
            tmp_path,
            [
                {
                    "date": "2024-01-01",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 28.0,
                    "hist_call_imp_vol": 28.0,
                    "volatility_30d": 26.0,
                },
                {
                    "date": "2024-01-02",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 0.01,
                    "hist_call_imp_vol": 0.01,
                    "volatility_30d": 25.0,
                },
            ],
        )
        iv = conn.get_iv_history("ZZ")
        assert len(iv) == 2  # the sub-3.0 row is KEPT, not dropped
        assert iv["hist_put_imp_vol"].iloc[0] == pytest.approx(28.0)  # normal kept
        assert pd.isna(iv["hist_put_imp_vol"].iloc[1])  # 0.01 implied nulled
        assert pd.isna(iv["hist_call_imp_vol"].iloc[1])
        assert iv["volatility_30d"].iloc[1] == pytest.approx(25.0)  # realized survives

    def test_high_sentinel_nulled_on_monolith(self, tmp_path):
        conn = self._conn(
            tmp_path,
            [
                {
                    "date": "2024-01-01",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 134217.7,
                    "hist_call_imp_vol": 134217.7,
                    "volatility_30d": 26.0,
                }
            ],
        )
        iv = conn.get_iv_history("ZZ")
        assert len(iv) == 1
        assert pd.isna(iv["hist_put_imp_vol"].iloc[0])

    def test_band_boundaries(self, tmp_path):
        conn = self._conn(
            tmp_path,
            [
                {
                    "date": "2024-01-01",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 3.0,
                    "hist_call_imp_vol": 3.0,
                },
                {
                    "date": "2024-01-02",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 3.01,
                    "hist_call_imp_vol": 3.01,
                },
                {
                    "date": "2024-01-03",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 10000.0,
                    "hist_call_imp_vol": 10000.0,
                },
                {
                    "date": "2024-01-04",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 10000.1,
                    "hist_call_imp_vol": 10000.1,
                },
            ],
        )
        vals = conn.get_iv_history("ZZ")["hist_put_imp_vol"].tolist()
        assert pd.isna(vals[0])  # 3.0 nulled (<= low floor)
        assert vals[1] == pytest.approx(3.01)  # just above floor: kept
        assert vals[2] == pytest.approx(10000.0)  # at sentinel floor: kept
        assert pd.isna(vals[3])  # above sentinel floor: nulled

    def test_normal_percent_iv_untouched(self, tmp_path):
        conn = self._conn(
            tmp_path,
            [
                {
                    "date": "2024-01-01",
                    "ticker": "ZZ",
                    "hist_put_imp_vol": 30.79,
                    "hist_call_imp_vol": 30.79,
                }
            ],
        )
        assert conn.get_iv_history("ZZ")["hist_put_imp_vol"].iloc[0] == pytest.approx(30.79)


# =====================================================================
# 5. Earnings + dividends
# =====================================================================
class TestEvents:
    def test_get_earnings_renames_year_period(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn.get_earnings("AAPL")
        assert "year_period" in df.columns
        assert "year/period" not in df.columns
        assert len(df) == 2

    def test_get_earnings_unknown_ticker_empty(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_earnings("NOSUCH").empty

    def test_get_next_earnings_future(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        nxt = conn.get_next_earnings("AAPL", as_of="2024-01-15")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2024-02-01")
        assert nxt["year_period"] == "2023 Q4"

    def test_get_next_earnings_none_when_all_past(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_next_earnings("AAPL", as_of="2030-01-01") is None

    def test_get_next_earnings_none_when_no_file(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_next_earnings("AAPL") is None

    def test_get_dividends(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn.get_dividends("AAPL")
        assert len(df) == 2
        assert "ex_date" in df.columns

    def test_get_next_dividend_future(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        nxt = conn.get_next_dividend("AAPL", as_of="2024-03-01")
        assert nxt is not None
        assert nxt["ex_date"] == pd.Timestamp("2024-05-10")
        assert nxt["dividend_amount"] == pytest.approx(0.25)

    def test_get_next_dividend_none_when_all_past(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_next_dividend("AAPL", as_of="2030-01-01") is None


# =====================================================================
# 6. Risk-free rate (percent <-> decimal normalisation)
# =====================================================================
class TestRiskFreeRate:
    def test_percent_form_normalised_to_decimal(self, data_dir):
        # CSV stores 5.25 (percent) -> connector returns 0.0525 (decimal).
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_risk_free_rate(tenor="rate_3m") == pytest.approx(0.0525)

    def test_sub_one_percent_rate_normalised_as_percent(self, tmp_path):
        # D20 regression: the treasury CSV is authoritatively PERCENT, so a
        # sub-1% value (a ZIRP-era T-bill) must be divided by 100, NOT returned
        # as-is. The pre-fix `/100 if > 1 else rate` heuristic returned 0.045
        # (=4.5%) for a 0.045% rate — a 100x error across 2011-2022.
        _csv(tmp_path / "treasury_yields.csv", [{"date": "2015-06-30", "rate_3m": 0.045}])
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_risk_free_rate(tenor="rate_3m") == pytest.approx(0.00045)

    def test_low_rate_era_within_percent_series(self, tmp_path):
        # The exact failure mode: a sub-1% percent row sitting in a percent
        # series. Each row is /100 regardless of magnitude.
        _csv(
            tmp_path / "treasury_yields.csv",
            [
                {"date": "2015-06-30", "rate_3m": 0.04},  # 0.04% (ZIRP)
                {"date": "2020-06-30", "rate_3m": 0.13},  # 0.13%
                {"date": "2023-06-30", "rate_3m": 5.20},  # 5.20%
            ],
        )
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_risk_free_rate(as_of="2015-07-01", tenor="rate_3m") == pytest.approx(0.0004)
        assert conn.get_risk_free_rate(as_of="2023-07-01", tenor="rate_3m") == pytest.approx(0.052)

    def test_as_of_picks_latest_on_or_before(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        # Only the 2024-01-02 row is on/before this as_of -> rate 5.20%.
        assert conn.get_risk_free_rate(as_of="2024-01-02", tenor="rate_3m") == pytest.approx(0.052)

    def test_unknown_tenor_returns_nan(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert np.isnan(conn.get_risk_free_rate(tenor="rate_30y"))

    def test_missing_file_returns_nan(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert np.isnan(conn.get_risk_free_rate())

    def test_as_of_before_all_data_returns_nan(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert np.isnan(conn.get_risk_free_rate(as_of="2000-01-01"))


# =====================================================================
# 7. VIX + VIX regime
# =====================================================================
class TestVIX:
    def test_get_vix(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn.get_vix()
        assert list(df.columns) == ["vix", "vix_3m", "vix_6m"]
        assert len(df) == 2

    def test_vix_regime_contango(self, data_dir):
        # Latest row: vix 15 < vix_3m 16.5 -> contango.
        conn = MarketDataConnector(data_dir=str(data_dir))
        r = conn.get_vix_regime()
        assert r["term_structure"] == "contango"
        assert r["vix"] == pytest.approx(15.0)
        assert 0.0 <= r["vix_percentile"] <= 1.0

    def test_vix_regime_backwardation(self, tmp_path):
        _csv(
            tmp_path / "vix_term_structure.csv",
            [{"date": "2024-01-02", "vix": 30.0, "vix_3m": 24.0, "vix_6m": 22.0}],
        )
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_vix_regime()["term_structure"] == "backwardation"

    def test_vix_regime_unknown_when_no_data(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        r = conn.get_vix_regime()
        assert r["term_structure"] == "unknown"
        assert np.isnan(r["vix"])

    def test_vix_regime_as_of_before_data_is_unknown(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_vix_regime(as_of="2000-01-01")["term_structure"] == "unknown"


# =====================================================================
# 8. Fundamentals + credit risk
# =====================================================================
class TestFundamentals:
    def test_get_fundamentals(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        f = conn.get_fundamentals("AAPL")
        assert f is not None
        assert f["sector"] == "Information Technology"
        assert f["pe_ratio"] == pytest.approx(28.0)
        assert f["beta"] == pytest.approx(1.2)

    def test_get_fundamentals_unknown_ticker_none(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_fundamentals("NOSUCH") is None

    def test_get_fundamentals_missing_file_none(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_fundamentals("AAPL") is None

    def test_get_credit_risk(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        c = conn.get_credit_risk("AAPL")
        assert c is not None
        assert c["sp_rating"] == "AA+"

    def test_get_credit_risk_unknown_ticker_none(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_credit_risk("NOSUCH") is None


# =====================================================================
# 9. Universe + screening
# =====================================================================
class TestUniverseScreen:
    def test_get_universe_union_sorted(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        uni = conn.get_universe()
        # Union of ohlcv (AAPL, MSFT) + fundamentals (AAPL, XOM) + vol_iv (AAPL).
        assert uni == sorted(uni)
        assert {"AAPL", "MSFT", "XOM"}.issubset(set(uni))

    def test_screen_universe_market_cap_filter(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        out = conn.screen_universe(min_market_cap=1.0e12)
        # Only AAPL (3e12) clears 1e12; XOM (4e11) does not.
        assert set(out["ticker"]) == {"AAPL"}

    def test_screen_universe_pe_and_sector_filter(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        out = conn.screen_universe(max_pe=15.0, sectors=["Energy"])
        assert set(out["ticker"]) == {"XOM"}

    def test_screen_universe_beta_filter(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        out = conn.screen_universe(max_beta=1.0)
        assert set(out["ticker"]) == {"XOM"}

    def test_screen_universe_min_iv_rank_adds_column(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        out = conn.screen_universe(min_iv_rank=0.0)
        assert "iv_rank" in out.columns
        # AAPL has IV data (rank computable); XOM has none (NaN -> filtered).
        assert "AAPL" in set(out["ticker"])

    def test_screen_universe_empty_when_no_fundamentals(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.screen_universe().empty


# =====================================================================
# 10. Liquidity
# =====================================================================
class TestLiquidity:
    def test_get_liquidity(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        df = conn.get_liquidity("AAPL")
        assert "avg_vol_30d" in df.columns
        assert df.index.name == "date"
        assert len(df) == 1

    def test_get_liquidity_unknown_ticker_empty(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_liquidity("NOSUCH").empty


# =====================================================================
# 11. Data frontier (M3 — as_of=None staleness gate helper)
# =====================================================================
class TestDataFrontier:
    """MarketDataConnector.get_data_frontier() — the M3 staleness-gate helper.

    Pins that the method (a) equals the underlying _load max-date,
    (b) is >= EXPECTED_FRONTIER on the committed Bloomberg CSVs,
    (c) works for the vol_iv dataset,
    (d) returns None on an empty/missing table,
    (e) clamps a future-dated corrupt row to at most today.
    """

    # Mirror the bump-on-refresh constant from test_preflight_environment.py.
    # When the OHLCV CSVs are refreshed, this constant must be bumped too.
    EXPECTED_FRONTIER = pd.Timestamp("2026-06-04")

    def test_frontier_returns_ohlcv_date_max(self, data_dir):
        """get_data_frontier() equals _load('ohlcv')['date'].max() (clamped)."""
        conn = MarketDataConnector(data_dir=str(data_dir))
        frontier = conn.get_data_frontier()
        assert frontier is not None
        # The synthetic data_dir fixture has dates up to 2024-01-04 (MSFT row).
        assert isinstance(frontier, pd.Timestamp)
        assert frontier == pd.Timestamp("2024-01-04")

    def test_real_frontier_ge_expected(self):
        """Real Bloomberg CSVs: frontier >= EXPECTED_FRONTIER (2026-06-04).

        Skips when the real data dir is unavailable (e.g. CI sandbox).
        """
        import os

        data_dir_real = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg")
        if not os.path.isdir(data_dir_real):
            pytest.skip("Real bloomberg data_dir unavailable")
        conn = MarketDataConnector(data_dir=data_dir_real)
        frontier = conn.get_data_frontier()
        assert frontier is not None, "Frontier must not be None on real data"
        assert frontier >= self.EXPECTED_FRONTIER, (
            f"OHLCV frontier {frontier.date()} is before EXPECTED_FRONTIER "
            f"{self.EXPECTED_FRONTIER.date()} — bump EXPECTED_FRONTIER if "
            "you refreshed the Bloomberg CSVs"
        )

    def test_vol_iv_dataset_returns_nonnone(self, data_dir):
        """get_data_frontier('vol_iv') returns a non-None timestamp."""
        conn = MarketDataConnector(data_dir=str(data_dir))
        frontier = conn.get_data_frontier(dataset="vol_iv")
        assert frontier is not None
        assert isinstance(frontier, pd.Timestamp)

    def test_empty_table_returns_none(self, tmp_path):
        """Empty OHLCV table -> None (gate skips gracefully)."""
        d = tmp_path / "bloomberg"
        d.mkdir()
        # Write an empty CSV (header only).
        pd.DataFrame(columns=["date", "ticker", "close"]).to_csv(d / "sp500_ohlcv.csv", index=False)
        conn = MarketDataConnector(data_dir=str(d))
        assert conn.get_data_frontier() is None

    def test_missing_table_returns_none(self, tmp_path):
        """Missing dataset table -> None (no crash)."""
        d = tmp_path / "bloomberg"
        d.mkdir()
        conn = MarketDataConnector(data_dir=str(d))
        assert conn.get_data_frontier() is None

    def test_future_dated_corrupt_row_clamped_to_today(self, tmp_path):
        """A future-dated corrupt row is clamped to today, not inflated."""
        import datetime as _dt

        d = tmp_path / "bloomberg"
        d.mkdir()
        _csv(
            d / "sp500_ohlcv.csv",
            [
                {
                    "date": "2024-01-02",
                    "ticker": "AAPL UW Equity",
                    "open": 106,
                    "high": 103,
                    "low": 97,
                    "close": 100,
                    "volume": 1_000_000,
                },
                {
                    "date": "2099-01-01",  # corrupt future row
                    "ticker": "CORRUPT UW Equity",
                    "open": 999,
                    "high": 999,
                    "low": 999,
                    "close": 999,
                    "volume": 1,
                },
            ],
        )
        conn = MarketDataConnector(data_dir=str(d))
        frontier = conn.get_data_frontier()
        assert frontier is not None
        today_ts = pd.Timestamp(_dt.date.today())
        assert frontier <= today_ts, (
            f"Frontier {frontier.date()} must not exceed today {today_ts.date()} "
            "even when a corrupt future-dated row is present"
        )
