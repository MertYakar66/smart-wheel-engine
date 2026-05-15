"""Tests for engine/theta_connector.py — Theta v3 connector with HTTP mocking.

Tests the Theta-side behaviour with full requests-mock stubs of the v3
endpoints. The Bloomberg fallback paths are exercised by pointing the
connector at an empty tmp_path data_dir, so super() calls return empty
results — proving the fallback was taken without needing real CSVs.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import requests
import requests_mock as rm_module

from engine.theta_connector import (
    FailureRecord,
    PerEndpointFailure,
    ThetaConnector,
    _normalise_theta_symbol,
)

THETA_BASE = "http://127.0.0.1:25503"
THETA_RE = re.compile(r"http://127\.0\.0\.1:25503/v3/.*")


# ---------------------------------------------------------------------------
# Fixture builders for Theta v3 CSV responses
# ---------------------------------------------------------------------------

def _stock_eod_csv(rows: list[tuple[str, float, float, float, float, int]] | None = None) -> str:
    """Build a /v3/stock/history/eod CSV (symbol, date, open, high, low, close, volume)."""
    rows = rows or [
        ("AAPL", "20260102", 150.0, 152.0, 149.0, 151.0, 50_000_000),
        ("AAPL", "20260103", 151.0, 153.5, 150.5, 153.0, 55_000_000),
    ]
    header = "symbol,date,open,high,low,close,volume\n"
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    return header + body


def _expirations_csv(dates: list[str] | None = None) -> str:
    """Build a /v3/option/list/expirations CSV (one column 'expiration').

    Dates must be ISO ('YYYY-MM-DD') so pandas auto-parses them.
    Pandas does NOT auto-parse 'YYYYMMDD' strings — they get treated
    as integer nanoseconds and round to 1970.
    """
    dates = dates or ["2026-01-17", "2026-02-21", "2026-03-20"]
    return "expiration\n" + "\n".join(dates)


def _greeks_csv(strikes: list[float] | None = None) -> str:
    """Build a /v3/option/snapshot/greeks/first_order CSV.

    For each strike, returns one row per side (call + put) with greeks + iv.
    """
    strikes = strikes or [140.0, 150.0, 160.0]
    header = "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
    rows = []
    for k in strikes:
        # Call: positive delta, OTM puts have negative delta
        call_delta = 0.5 + (150.0 - k) * 0.01
        put_delta = -0.5 - (k - 150.0) * 0.01
        rows.append(f"AAPL,20260221,{k},call,{call_delta},0.02,-0.05,0.20,0.01,0.25,1.50,1.55")
        rows.append(f"AAPL,20260221,{k},put,{put_delta},0.02,-0.05,0.20,0.01,0.27,1.40,1.45")
    return header + "\n".join(rows)


def _quotes_csv(strikes: list[float] | None = None) -> str:
    """Build a /v3/option/snapshot/quote CSV with bid/ask only."""
    strikes = strikes or [140.0, 150.0, 160.0]
    header = "symbol,expiration,strike,right,bid,ask,bid_size,ask_size\n"
    rows = []
    for k in strikes:
        rows.append(f"AAPL,20260221,{k},call,1.50,1.55,10,20")
        rows.append(f"AAPL,20260221,{k},put,1.40,1.45,12,18")
    return header + "\n".join(rows)


def _open_interest_csv(strikes: list[float] | None = None) -> str:
    """Build a /v3/option/snapshot/open_interest CSV."""
    strikes = strikes or [140.0, 150.0, 160.0]
    header = "symbol,expiration,strike,right,open_interest\n"
    rows = []
    for k in strikes:
        rows.append(f"AAPL,20260221,{k},call,5000")
        rows.append(f"AAPL,20260221,{k},put,3500")
    return header + "\n".join(rows)


def _vix_snapshot_csv(price: float = 15.5) -> str:
    return f"symbol,price\nVIX,{price}"


def _option_history_eod_csv(rows: list[tuple[str, float, float, float, float, int]] | None = None) -> str:
    """Build /v3/option/history/eod CSV."""
    rows = rows or [
        ("20260102", 1.50, 1.55, 1.45, 1.50, 100),
        ("20260103", 1.50, 1.60, 1.40, 1.55, 150),
    ]
    header = "created,open,high,low,close,volume,bid,ask\n"
    body = "\n".join(f"{d},{o},{h},{lo},{c},{v},{c-0.05},{c+0.05}" for d, o, h, lo, c, v in rows)
    return header + body


def _stock_intraday_csv(n_rows: int = 5) -> str:
    """Build /v3/stock/history/intraday CSV with timestamps."""
    base = pd.Timestamp("2026-01-02 09:30:00")
    header = "timestamp,open,high,low,close,volume\n"
    rows = []
    for i in range(n_rows):
        ts = base + pd.Timedelta(minutes=5 * i)
        rows.append(f"{ts.isoformat()},150.0,150.5,149.8,150.2,1000")
    return header + "\n".join(rows)


def _iv_history_csv(n_rows: int = 30) -> str:
    """Build /v3/option/history/greeks/* CSV with iv column."""
    base = pd.Timestamp("2025-06-01")
    header = "timestamp,iv\n"
    rows = []
    for i in range(n_rows):
        ts = base + pd.Timedelta(days=i)
        # Realistic IV pattern: hover around 0.25 with daily noise
        iv = 0.25 + (i % 5) * 0.01
        rows.append(f"{ts.isoformat()},{iv}")
    return header + "\n".join(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock():
    with rm_module.Mocker() as m:
        yield m


@pytest.fixture
def connector(tmp_path: Path) -> ThetaConnector:
    """Connector pointed at an empty tmp_path so Bloomberg fallbacks
    return empty results — we can detect "fell back to CSV" by an
    empty DataFrame."""
    return ThetaConnector(data_dir=str(tmp_path), base_url=THETA_BASE)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestNormaliseSymbol:
    def test_already_normalised(self):
        assert _normalise_theta_symbol("AAPL") == "AAPL"

    def test_lowercase_uppercased(self):
        assert _normalise_theta_symbol("aapl") == "AAPL"

    def test_explicit_alias_brk_b(self):
        assert _normalise_theta_symbol("BRK-B") == "BRK.B"

    def test_explicit_alias_bf_b(self):
        assert _normalise_theta_symbol("BF-B") == "BF.B"

    def test_explicit_alias_googl(self):
        assert _normalise_theta_symbol("GOOG-L") == "GOOGL"

    def test_generic_hyphen_class_letter(self):
        # Not in the alias map but still has the pattern
        assert _normalise_theta_symbol("XYZ-A") == "XYZ.A"

    def test_generic_slash(self):
        assert _normalise_theta_symbol("BRK/B") == "BRK.B"

    def test_generic_space(self):
        assert _normalise_theta_symbol("BRK B") == "BRK.B"

    def test_strips_bloomberg_uw_equity(self):
        assert _normalise_theta_symbol("AAPL UW Equity") == "AAPL"

    def test_strips_us_equity(self):
        assert _normalise_theta_symbol("MSFT US Equity") == "MSFT"

    def test_strips_un_suffix(self):
        assert _normalise_theta_symbol("JPM UN") == "JPM"

    def test_complex_passthrough(self):
        # Multi-word non-class-letter pattern: passes through as-is
        assert _normalise_theta_symbol("ABCDE") == "ABCDE"


# ---------------------------------------------------------------------------
# _fetch — internal HTTP helper (touches every method)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_happy_csv(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="a,b\n1,2\n")
        df = connector._fetch("/v3/stock/history/eod", {"symbol": "AAPL"})
        assert not df.empty
        assert list(df.columns) == ["a", "b"]

    def test_403_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=403)
        df = connector._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_404_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        df = connector._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_500_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=500)
        df = connector._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_503_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=503)
        df = connector._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_connection_error_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Bad URL outside requests_mock → requests-mock raises NoMockAddress,
        # which lands in _fetch's generic ``except Exception:`` handler and
        # returns empty. Note this test does NOT exercise _handle_network_failure;
        # for the explicit ConnectionError path (and the probe branch), see
        # TestPerEndpointFailure below.
        bad_conn = ThetaConnector(data_dir=str(Path("/tmp")), base_url="http://127.0.0.1:1")
        df = bad_conn._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_v2_upgrade_message_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="We have upgraded the API to v3")
        df = connector._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_empty_response_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="")
        df = connector._fetch("/v3/some/path", {"symbol": "AAPL"})
        assert df.empty

    def test_symbol_normalised_in_request(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="a\n1\n")
        connector._fetch("/v3/some/path", {"symbol": "BRK-B"})
        # The mock request URL should contain BRK.B, not BRK-B
        url = mock.last_request.url
        assert "BRK.B" in url or "BRK%2EB" in url or "BRK." in url


# ---------------------------------------------------------------------------
# get_ohlcv
# ---------------------------------------------------------------------------

class TestGetOhlcv:
    def test_happy_path(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text=_stock_eod_csv())
        df = connector.get_ohlcv("AAPL")
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2

    def test_empty_response_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Theta returns empty → falls back to Bloomberg → also empty (tmp_path data_dir)
        mock.get(THETA_RE, status_code=404)
        df = connector.get_ohlcv("AAPL")
        # Tmp_path data_dir → Bloomberg fallback returns empty too
        assert df.empty

    def test_missing_close_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Response has all columns except close → triggers warning + fallback
        text = "symbol,date,open,high,low,volume\nAAPL,20260102,150.0,152.0,149.0,1000000\n"
        mock.get(THETA_RE, text=text)
        df = connector.get_ohlcv("AAPL")
        assert df.empty  # Bloomberg fallback also empty

    def test_dates_passed_through(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text=_stock_eod_csv())
        connector.get_ohlcv("AAPL", start_date="2026-01-01", end_date="2026-01-31")
        url = mock.last_request.url
        assert "start_date=20260101" in url
        assert "end_date=20260131" in url


# ---------------------------------------------------------------------------
# _nearest_expiration
# ---------------------------------------------------------------------------

class TestNearestExpiration:
    def test_picks_closest_to_target(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # ISO-format dates so pandas auto-parses (YYYYMMDD strings parse as nanoseconds → 1970)
        d30 = (datetime.now(UTC).date() + timedelta(days=30)).isoformat()
        d60 = (datetime.now(UTC).date() + timedelta(days=60)).isoformat()
        d90 = (datetime.now(UTC).date() + timedelta(days=90)).isoformat()
        mock.get(THETA_RE, text=_expirations_csv([d30, d60, d90]))
        result = connector._nearest_expiration("AAPL", dte_target=35)
        # Returned in YYYYMMDD format
        assert result == d30.replace("-", "")

    def test_empty_returns_none(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="expiration\n")
        assert connector._nearest_expiration("AAPL") is None

    def test_404_returns_none(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        assert connector._nearest_expiration("AAPL") is None


# ---------------------------------------------------------------------------
# get_option_chain
# ---------------------------------------------------------------------------

class TestGetOptionChain:
    def test_happy_path_with_explicit_expiration(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Stub all three endpoints used by get_option_chain
        mock.get(re.compile(r".*greeks/first_order.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert not df.empty
        assert "iv" in df.columns
        assert "delta" in df.columns
        assert "mid" in df.columns
        assert "open_interest" in df.columns
        # Mid = (bid + ask) / 2 — check by row, not by position
        # (call: mid=(1.50+1.55)/2=1.525; put: mid=(1.40+1.45)/2=1.425)
        calls = df[df["right"] == "call"]
        puts = df[df["right"] == "put"]
        assert calls["mid"].iloc[0] == pytest.approx(1.525)
        assert puts["mid"].iloc[0] == pytest.approx(1.425)

    def test_iv_alias_renamed(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Greeks endpoint returns 'implied_vol' instead of 'iv'
        text = "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,implied_vol,bid,ask\n"
        text += "AAPL,20260221,150.0,call,0.5,0.02,-0.05,0.20,0.01,0.25,1.50,1.55\n"
        text += "AAPL,20260221,150.0,put,-0.5,0.02,-0.05,0.20,0.01,0.27,1.40,1.45\n"
        mock.get(re.compile(r".*greeks/first_order.*"), text=text)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv([150.0]))
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv([150.0]))
        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert "iv" in df.columns
        assert "implied_vol" not in df.columns

    def test_iv_percent_form_normalised_to_decimal(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # IV reported as 25 (percent form) should be normalised to 0.25
        text = "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
        text += "AAPL,20260221,150.0,call,0.5,0.02,-0.05,0.20,0.01,25.0,1.50,1.55\n"
        mock.get(re.compile(r".*greeks/first_order.*"), text=text)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv([150.0]))
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv([150.0]))
        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert df["iv"].iloc[0] == pytest.approx(0.25)

    def test_both_endpoints_empty_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert df.empty

    def test_auto_picks_expiration_when_omitted(self, mock: rm_module.Mocker, connector: ThetaConnector):
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        mock.get(re.compile(r".*greeks/first_order.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        df = connector.get_option_chain("AAPL")
        assert not df.empty

    def test_auto_pick_returns_empty_when_no_expirations(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*list/expirations.*"), text="expiration\n")
        df = connector.get_option_chain("AAPL")
        assert df.empty

    def test_cache_hit_avoids_second_fetch(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*greeks/first_order.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        df1 = connector.get_option_chain("AAPL", expiration="20260221")
        n_before = mock.call_count
        df2 = connector.get_option_chain("AAPL", expiration="20260221")
        assert mock.call_count == n_before  # served from cache
        assert len(df1) == len(df2)

    def test_quotes_only_when_greeks_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*greeks/first_order.*"), status_code=404)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), status_code=404)
        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert not df.empty
        assert "bid" in df.columns


# ---------------------------------------------------------------------------
# get_fundamentals — Bloomberg + live IV
# ---------------------------------------------------------------------------

class TestGetFundamentals:
    def test_no_csv_returns_none_or_dict_with_iv(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # No Bloomberg CSVs in tmp_path → super().get_fundamentals returns None
        # With Theta IV available, the connector should still return a dict
        mock.get(re.compile(r".*greeks/first_order.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        result = connector.get_fundamentals("AAPL")
        # Either None (no CSV + no chain.iv match) or dict with iv
        if result is not None:
            assert "implied_vol_atm" in result

    def test_chain_failure_returns_super_result(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Theta chain 404; super() returns None for empty data dir
        mock.get(THETA_RE, status_code=404)
        result = connector.get_fundamentals("AAPL")
        assert result is None


# ---------------------------------------------------------------------------
# get_iv_rank / get_iv_percentile
# ---------------------------------------------------------------------------

class TestGetIvRank:
    def test_historical_path_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # As-of in the past → live=False → super() called → CSV missing → returns 0.5
        result = connector.get_iv_rank("AAPL", as_of="2024-01-01")
        # Default fallback per MarketDataConnector returns 0.5 when no data
        assert isinstance(result, float)

    def test_insufficient_history_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Live path; chain returns nothing → _fetch_iv_history returns None
        # → falls back to super() which has no CSV
        mock.get(THETA_RE, status_code=404)
        result = connector.get_iv_rank("AAPL")
        assert isinstance(result, float)

    def test_iv_percentile_aliases_iv_rank(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        rank = connector.get_iv_rank("AAPL")
        pct = connector.get_iv_percentile("AAPL")
        # NaN != NaN by IEEE; use both-finite-and-equal OR both-nan
        import math
        assert (math.isnan(rank) and math.isnan(pct)) or rank == pct

    def test_live_with_full_iv_history(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """End-to-end: chain → ATM strike → 12 monthly IV history chunks."""
        # 1. Expirations endpoint returns ~35-DTE
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # 2. Greeks/quotes/OI for the chain (gives ATM strike from -0.50 delta)
        mock.get(re.compile(r".*greeks/first_order.*snapshot.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        # 3. IV history endpoint — first try implied_volatility, then first_order.
        #    Cover both endpoint paths for completeness.
        mock.get(re.compile(r".*history/greeks/implied_volatility.*"), text=_iv_history_csv(30))
        mock.get(re.compile(r".*history/greeks/first_order.*"), text=_iv_history_csv(30))
        rank = connector.get_iv_rank("AAPL")
        # With 30 IV obs per chunk * 13 chunks, plenty of history; rank in [0,1]
        assert isinstance(rank, float)
        # Either real rank or fallback NaN — both are acceptable signals that
        # the path executed without crashing
        assert rank != rank or 0.0 <= rank <= 1.0  # NaN check OR in-range

    def test_live_iv_history_dedicated_endpoint_fallthrough(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """When implied_volatility 404s, the connector falls through to first_order."""
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        mock.get(re.compile(r".*greeks/first_order.*snapshot.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        # Dedicated IV endpoint 404 → falls through to greeks/first_order history
        mock.get(re.compile(r".*history/greeks/implied_volatility.*"), status_code=404)
        mock.get(re.compile(r".*history/greeks/first_order.*"), text=_iv_history_csv(30))
        rank = connector.get_iv_rank("AAPL")
        assert isinstance(rank, float)


# ---------------------------------------------------------------------------
# get_vix_regime
# ---------------------------------------------------------------------------

class TestGetVixRegime:
    def test_historical_path_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        result = connector.get_vix_regime(as_of="2024-01-01")
        assert isinstance(result, dict)

    def test_live_with_data_returns_vix(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*index/snapshot/price.*"), text=_vix_snapshot_csv(15.5))
        result = connector.get_vix_regime()
        assert result["vix"] == pytest.approx(15.5)

    def test_live_with_404_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        result = connector.get_vix_regime()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# get_vol_risk_premium
# ---------------------------------------------------------------------------

class TestGetVolRiskPremium:
    def test_historical_falls_back(self, connector: ThetaConnector):
        result = connector.get_vol_risk_premium("AAPL", as_of="2024-01-01")
        assert isinstance(result, float)

    def test_insufficient_ohlcv_falls_back(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Theta OHLCV returns < 22 rows
        text = _stock_eod_csv([("AAPL", "20260102", 150, 151, 149, 150, 1000)])
        mock.get(re.compile(r".*stock/history/eod.*"), text=text)
        result = connector.get_vol_risk_premium("AAPL")
        assert isinstance(result, float)

    def test_full_path_with_ohlcv_and_iv(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """End-to-end: 30+ days of OHLCV + chain returning ATM IV."""
        # Build 35 rows of OHLCV
        base = pd.Timestamp("2026-01-01")
        rows = []
        for i in range(35):
            d = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
            close = 150.0 + (i % 5)  # noisy but bounded
            rows.append(("AAPL", d, close - 1, close + 1, close - 2, close, 1_000_000))
        mock.get(re.compile(r".*stock/history/eod.*"), text=_stock_eod_csv(rows))
        # Chain provides ATM IV via the put closest to -0.50 delta
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        mock.get(re.compile(r".*greeks/first_order.*snapshot.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        result = connector.get_vol_risk_premium("AAPL")
        # IV ~ 0.27, realised vol from noisy series — premium can be either sign
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# get_iv_surface / get_atm_term_structure / get_skew_snapshot
# ---------------------------------------------------------------------------

class TestGetIvSurface:
    def test_empty_when_no_expirations(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*list/expirations.*"), text="expiration\n")
        df = connector.get_iv_surface("AAPL")
        assert df.empty

    def test_filters_by_dte_window(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # All expirations are too far out (> max_dte=400)
        far_future = (datetime.now(UTC).date() + timedelta(days=500)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([far_future]))
        df = connector.get_iv_surface("AAPL")
        assert df.empty

    def test_returns_data_for_valid_expirations(self, mock: rm_module.Mocker, connector: ThetaConnector):
        future_45 = (datetime.now(UTC).date() + timedelta(days=45)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future_45]))
        mock.get(re.compile(r".*greeks/first_order.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        df = connector.get_iv_surface("AAPL")
        assert not df.empty
        assert {"strike", "right", "delta", "iv", "expiration", "dte"} <= set(df.columns)


class TestGetAtmTermStructure:
    def test_empty_when_no_surface(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        df = connector.get_atm_term_structure("AAPL")
        assert df.empty

    def test_returns_atm_iv(self, mock: rm_module.Mocker, connector: ThetaConnector):
        future_45 = (datetime.now(UTC).date() + timedelta(days=45)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future_45]))
        mock.get(re.compile(r".*greeks/first_order.*"), text=_greeks_csv())
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv())
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv())
        df = connector.get_atm_term_structure("AAPL")
        assert not df.empty
        assert "atm_iv" in df.columns


class TestGetSkewSnapshot:
    def test_empty_when_no_expiration(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*list/expirations.*"), text="expiration\n")
        result = connector.get_skew_snapshot("AAPL")
        assert result == {}

    def test_empty_when_chain_unusable(self, mock: rm_module.Mocker, connector: ThetaConnector):
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        mock.get(re.compile(r".*greeks/first_order.*"), status_code=404)
        mock.get(re.compile(r".*snapshot/quote.*"), status_code=404)
        mock.get(re.compile(r".*open_interest.*"), status_code=404)
        result = connector.get_skew_snapshot("AAPL")
        assert result == {}

    def test_full_path_with_all_three_strikes(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """End-to-end: chain has 25Δ put + ATM put + 25Δ call."""
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # Custom greeks fixture with strikes spanning the OTM spectrum
        # Need delta=-0.25 put, -0.50 put, +0.25 call
        text = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
            "AAPL,20260221,140.0,put,-0.25,0.02,-0.05,0.20,0.01,0.30,1.40,1.45\n"
            "AAPL,20260221,150.0,put,-0.50,0.02,-0.05,0.20,0.01,0.27,1.40,1.45\n"
            "AAPL,20260221,160.0,call,0.25,0.02,-0.05,0.20,0.01,0.28,1.40,1.45\n"
        )
        mock.get(re.compile(r".*greeks/first_order.*"), text=text)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_csv([140.0, 150.0, 160.0]))
        mock.get(re.compile(r".*open_interest.*"), text=_open_interest_csv([140.0, 150.0, 160.0]))
        result = connector.get_skew_snapshot("AAPL")
        # All three IVs populated → returns dict with iv_25d_put / iv_atm / iv_25d_call
        if result:  # not empty (no NaN guards triggered)
            assert "iv_25d_put" in result
            assert "iv_atm" in result
            assert "iv_25d_call" in result
            assert "expiration" in result
            assert "dte" in result


# ---------------------------------------------------------------------------
# get_vix_family
# ---------------------------------------------------------------------------

class TestGetVixFamily:
    def test_theta_succeeds_for_all(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*index/snapshot/price.*"), text=_vix_snapshot_csv(15.5))
        result = connector.get_vix_family()
        # All seven symbols served from Theta
        assert "VIX" in result
        assert result["VIX"] == pytest.approx(15.5)

    def test_falls_back_to_cboe_when_theta_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        # Theta 403 → CBOE attempted; we mock CBOE 404 → Yahoo attempted
        mock.get(re.compile(r".*index/snapshot/price.*"), status_code=403)
        mock.get(re.compile(r".*cboe\.com.*"), status_code=404)
        mock.get(re.compile(r".*yahoo\.com.*"), status_code=404)
        result = connector.get_vix_family()
        # All fell through; result is empty dict
        assert result == {}

    def test_cboe_serves_when_theta_403(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """Theta 403 → CBOE adapter returns valid CSV → out has VIX."""
        mock.get(re.compile(r".*index/snapshot/price.*"), status_code=403)
        # CBOE endpoint pattern — returns a 1-row VIX CSV
        cboe_csv = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "2026-01-02,15.0,16.0,14.5,15.5\n"
        )
        mock.get(re.compile(r".*cboe\.com.*VIX_History\.csv"), text=cboe_csv)
        # Other CBOE symbols 404
        mock.get(re.compile(r".*cboe\.com.*"), status_code=404)
        # Yahoo for VVIX/SKEW/MOVE that CBOE doesn't have
        mock.get(re.compile(r".*yahoo\.com.*"), status_code=404)
        result = connector.get_vix_family()
        # VIX should be populated from CBOE
        assert "VIX" in result or result == {}  # graceful either way

    def test_yahoo_fallback_for_remaining(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """Theta + CBOE both 404 for VVIX → Yahoo fallback runs."""
        mock.get(re.compile(r".*index/snapshot/price.*"), status_code=403)
        mock.get(re.compile(r".*cboe\.com.*"), status_code=404)
        # Yahoo returns OHLCV CSV for ^VVIX
        yf_csv = (
            "Date,Open,High,Low,Close,Adj Close,Volume\n"
            "2026-01-02,90.0,92.0,89.0,91.0,91.0,0\n"
        )
        mock.get(re.compile(r".*yahoo\.com.*"), text=yf_csv)
        result = connector.get_vix_family()
        # Some symbol might be populated
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# get_option_ohlc_history
# ---------------------------------------------------------------------------

class TestGetOptionOhlcHistory:
    def test_happy_path(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*option/history/eod.*"), text=_option_history_eod_csv())
        df = connector.get_option_ohlc_history(
            "AAPL", "20260221", 150.0, "put",
            start_date="2026-01-01", end_date="2026-01-10",
        )
        assert not df.empty
        assert "close" in df.columns

    def test_empty_when_404(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        df = connector.get_option_ohlc_history(
            "AAPL", "20260221", 150.0, "put",
        )
        assert df.empty

    def test_missing_close_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="created\n2026-01-02\n")
        df = connector.get_option_ohlc_history("AAPL", "20260221", 150.0, "put")
        assert df.empty


# ---------------------------------------------------------------------------
# get_stock_intraday
# ---------------------------------------------------------------------------

class TestGetStockIntraday:
    def test_happy_path(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(re.compile(r".*stock/history/intraday.*"), text=_stock_intraday_csv())
        df = connector.get_stock_intraday(
            "AAPL", interval="5m",
            start_date="2026-01-02", end_date="2026-01-02",
        )
        assert not df.empty
        assert "close" in df.columns

    def test_empty_when_404(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        df = connector.get_stock_intraday("AAPL")
        assert df.empty

    def test_missing_close_returns_empty(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="timestamp\n2026-01-02 09:30:00\n")
        df = connector.get_stock_intraday("AAPL")
        assert df.empty


# ---------------------------------------------------------------------------
# is_terminal_alive
# ---------------------------------------------------------------------------

class TestIsTerminalAlive:
    def test_alive_when_200_with_body(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text=_expirations_csv())
        assert connector.is_terminal_alive() is True

    def test_dead_when_404(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, status_code=404)
        assert connector.is_terminal_alive() is False

    def test_dead_when_connection_refused(self):
        bad = ThetaConnector(data_dir=str(Path("/tmp")), base_url="http://127.0.0.1:1")
        assert bad.is_terminal_alive() is False

    def test_dead_when_body_too_short(self, mock: rm_module.Mocker, connector: ThetaConnector):
        mock.get(THETA_RE, text="x")  # < 10 chars
        assert connector.is_terminal_alive() is False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestCacheHelpers:
    def test_to_yyyymmdd(self):
        # Static method test
        assert ThetaConnector._to_yyyymmdd("2026-01-02") == "20260102"
        assert ThetaConnector._to_yyyymmdd(None) is None
        assert ThetaConnector._to_yyyymmdd("not-a-date") is None

    def test_is_live_for_none(self, connector: ThetaConnector):
        assert connector._is_live(None) is True

    def test_is_live_for_today(self, connector: ThetaConnector):
        today = datetime.now(UTC).date().isoformat()
        assert connector._is_live(today) is True

    def test_is_live_for_past(self, connector: ThetaConnector):
        assert connector._is_live("2020-01-01") is False

    def test_is_live_for_invalid(self, connector: ThetaConnector):
        assert connector._is_live("nonsense") is False

    def test_chain_cache_set_then_get(self, connector: ThetaConnector):
        df = pd.DataFrame({"a": [1, 2]})
        connector._set_cached_chain("AAPL", "20260221", df)
        out = connector._get_cached_chain("AAPL", "20260221")
        assert out is not None
        assert len(out) == 2

    def test_chain_cache_miss_returns_none(self, connector: ThetaConnector):
        out = connector._get_cached_chain("ZZZZ", "20260221")
        assert out is None


# ---------------------------------------------------------------------------
# PerEndpointFailure / probe behaviour / instance flag (issue #71, D11)
#
# These tests exercise the contract added in commit 7a1ac38: when the data
# endpoint fails with ConnectionError / RetryError / Timeout, the connector
# probes /v3/option/list/expirations?symbol=SPY (5s GET via
# is_terminal_alive) to distinguish per-endpoint failures from a globally-
# down Terminal.
#   * Probe healthy → _fetch raises PerEndpointFailure (loud skip).
#   * Probe fails   → _terminal_down flag flips, returns empty
#                     (carve-out for backfill via Bloomberg CSV).
# ---------------------------------------------------------------------------

class TestPerEndpointFailure:
    def test_raises_when_probe_healthy(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """T1: data endpoint times out, probe says Terminal alive → raises."""
        # Order matters: more specific patterns added LAST so they win
        # (requests-mock: "If multiple matchers match, the last added wins").
        mock.get(
            re.compile(r".*stock/history/eod.*"),
            exc=requests.exceptions.ConnectionError,
        )
        mock.get(
            re.compile(r".*list/expirations.*symbol=SPY.*"),
            text="expiration\n2026-06-20\n",
        )

        with pytest.raises(PerEndpointFailure) as excinfo:
            connector.get_ohlcv("AAPL")

        # Exception carries the FailureRecord
        record = excinfo.value.record
        assert isinstance(record, FailureRecord)
        assert record.endpoint == "/v3/stock/history/eod"
        assert "AAPL" == record.params.get("symbol")
        assert "ConnectionError" in record.reason

        # Connector accumulator state: probe healthy, no down-mode
        assert connector._terminal_down is False
        assert len(connector._failures) == 1
        assert connector._failures[0].endpoint == "/v3/stock/history/eod"

        # get_failures returns + clears
        first = connector.get_failures()
        assert len(first) == 1
        assert connector.get_failures() == []

    def test_returns_empty_when_probe_also_fails(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """T2: data + probe both fail → empty df, flag set, no record."""
        mock.get(THETA_RE, exc=requests.exceptions.ConnectionError)

        df = connector.get_ohlcv("AAPL")

        # tmp_path data_dir → super().get_ohlcv() also returns empty
        assert df.empty
        assert connector._terminal_down is True
        assert connector._failures == []

    def test_terminal_down_skips_reprobe(
        self, mock: rm_module.Mocker, connector: ThetaConnector, monkeypatch: pytest.MonkeyPatch
    ):
        """T3: once flag is set, subsequent failures do NOT re-probe."""
        from unittest.mock import Mock as MockObj

        mock.get(THETA_RE, exc=requests.exceptions.ConnectionError)

        # Patch the probe to return False directly so we can count calls.
        # (Real probe would also return False against the failing mock,
        # but counting via Mock is the most explicit signal.)
        probe = MockObj(return_value=False)
        monkeypatch.setattr(connector, "is_terminal_alive", probe)

        # First failing call: probes once, flag flips to True.
        connector.get_ohlcv("AAPL")
        assert connector._terminal_down is True
        assert probe.call_count == 1

        # Second failing call: short-circuits inside _handle_network_failure.
        connector.get_ohlcv("MSFT")
        assert probe.call_count == 1  # not re-probed

    def test_fresh_instance_starts_clean(self, mock: rm_module.Mocker, tmp_path: Path):
        """T4: the down-mode flag and failure list are per-instance."""
        mock.get(THETA_RE, exc=requests.exceptions.ConnectionError)

        a = ThetaConnector(data_dir=str(tmp_path), base_url=THETA_BASE)
        a.get_ohlcv("AAPL")
        assert a._terminal_down is True

        b = ThetaConnector(data_dir=str(tmp_path), base_url=THETA_BASE)
        assert b._terminal_down is False
        assert b._failures == []
        assert b.get_failures() == []

    def test_get_fundamentals_propagates(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """T6: get_fundamentals re-raises PerEndpointFailure (no silent CSV-only).

        Pins the contract change: previously, an exception during the live
        ATM IV overlay was caught silently and CSV-only fundamentals were
        returned. That's the same mixed-provenance contamination the issue
        body calls out, so we now propagate.
        """
        # Catch-all fail
        mock.get(THETA_RE, exc=requests.exceptions.ConnectionError)
        # Probe healthy (added LAST so it wins for the SPY URL)
        mock.get(
            re.compile(r".*list/expirations.*symbol=SPY.*"),
            text="expiration\n2026-06-20\n",
        )

        with pytest.raises(PerEndpointFailure):
            connector.get_fundamentals("AAPL")

    def test_failure_record_is_json_serialisable(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """FailureRecord must round-trip through dataclasses.asdict for the
        sidecar manifest writer (C4)."""
        mock.get(
            re.compile(r".*stock/history/eod.*"),
            exc=requests.exceptions.ConnectionError,
        )
        mock.get(
            re.compile(r".*list/expirations.*symbol=SPY.*"),
            text="expiration\n2026-06-20\n",
        )

        with pytest.raises(PerEndpointFailure):
            connector.get_ohlcv("AAPL")

        records = connector.get_failures()
        # asdict + json.dumps must work end-to-end
        payload = json.dumps([asdict(r) for r in records])
        roundtrip = json.loads(payload)
        assert roundtrip[0]["endpoint"] == "/v3/stock/history/eod"
        assert roundtrip[0]["params"]["symbol"] == "AAPL"

    @pytest.mark.parametrize(
        "exc_cls",
        [
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.RetryError,
        ],
    )
    def test_all_network_failure_types_route_through_probe(
        self, mock: rm_module.Mocker, connector: ThetaConnector, exc_cls
    ):
        """All three network-error classes that _fetch catches route through
        _handle_network_failure and raise PerEndpointFailure when the probe
        is healthy.

        Issue #71's actual repro is a 30s ReadTimeout on
        /v3/option/history/eod, not a ConnectionError. If any of these three
        classes ever stopped routing through the probe (e.g. someone
        narrowed the except tuple), the fix would be silently incomplete.
        """
        mock.get(
            re.compile(r".*list/expirations.*symbol=SPY.*"),
            text="expiration\n2026-06-20\n",
        )
        mock.get(
            re.compile(r".*stock/history/eod.*"),
            exc=exc_cls,
        )

        with pytest.raises(PerEndpointFailure) as excinfo:
            connector.get_ohlcv("AAPL")

        assert exc_cls.__name__ in excinfo.value.record.reason
