"""Tests for engine/external_data/edgar_adapter.py with HTTP mocking."""

from __future__ import annotations

import pandas as pd
import pytest
import requests_mock as rm_module

from engine.external_data.edgar_adapter import EDGARAdapter, _EDGAR_BASE


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@pytest.fixture
def mock():
    with rm_module.Mocker() as m:
        yield m


def _ticker_index() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    }


def _submissions_payload(filing_dates: list[str], forms: list[str]) -> dict:
    return {
        "filings": {
            "recent": {
                "filingDate": filing_dates,
                "accessionNumber": [f"acc-{i}" for i in range(len(filing_dates))],
                "form": forms,
                "primaryDocument": [f"doc-{i}.htm" for i in range(len(filing_dates))],
            }
        }
    }


class TestCikForTicker:
    def test_resolves_known_ticker(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        cik = adapter.cik_for_ticker("AAPL")
        assert cik == "0000320193"  # 10-digit zero-padded

    def test_case_insensitive_match(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        cik = adapter.cik_for_ticker("aapl")
        assert cik == "0000320193"

    def test_unknown_ticker_returns_none(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.cik_for_ticker("ZZZZ") is None

    def test_network_error_returns_none(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, exc=ConnectionError("boom"))
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.cik_for_ticker("AAPL") is None

    def test_500_returns_none(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, status_code=500)
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.cik_for_ticker("AAPL") is None

    def test_caching_avoids_second_call(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        adapter.cik_for_ticker("AAPL")
        adapter.cik_for_ticker("AAPL")
        # Cached on first hit
        assert mock.call_count == 1


class TestRecentInsiderTrades:
    def _setup(self, mock: rm_module.Mocker, ticker: str = "AAPL"):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        return EDGARAdapter(min_interval_s=0.0)

    def test_unknown_ticker_returns_empty_df(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.recent_insider_trades("ZZZZ").empty

    def test_filters_by_form4_and_window(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        # Mix of form 4 (insider) and form 10-K (annual report)
        recent_date = (pd.Timestamp.now() - pd.Timedelta(days=10)).date().isoformat()
        old_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).date().isoformat()
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_payload(
                [recent_date, recent_date, old_date],
                ["4", "10-K", "4"],
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.recent_insider_trades("AAPL", n_days=90)
        # Only the 1 recent form-4 row should remain
        assert len(df) == 1
        assert df.iloc[0]["form"] == "4"

    def test_empty_recent_returns_empty(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json={"filings": {"recent": {}}},
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.recent_insider_trades("AAPL").empty

    def test_network_error_returns_empty(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            exc=ConnectionError("boom"),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.recent_insider_trades("AAPL").empty

    def test_no_filings_in_window(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        old_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).date().isoformat()
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_payload([old_date], ["4"]),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.recent_insider_trades("AAPL", n_days=30)
        assert df.empty


class TestInsiderActivitySignal:
    def test_no_recent_filings(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        old_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).date().isoformat()
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_payload([old_date], ["4"]),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        sig = adapter.insider_activity_signal("AAPL")
        assert sig == {"filings_count": 0, "last_filing_date": None, "active": False}

    def test_active_with_recent(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        recent = (pd.Timestamp.now() - pd.Timedelta(days=5)).date().isoformat()
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_payload([recent, recent], ["4", "4"]),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        sig = adapter.insider_activity_signal("AAPL", n_days=30)
        assert sig["filings_count"] == 2
        assert sig["active"] is True
        assert sig["last_filing_date"] == recent

    def test_unknown_ticker_returns_inactive(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        sig = adapter.insider_activity_signal("ZZZZ")
        assert sig["active"] is False


class TestRateLimiting:
    def test_min_interval_sleep_runs(self, mock: rm_module.Mocker):
        # Just verify the _sleep_if_needed branch is exercised; we keep
        # the interval tiny so this stays fast.
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.001)
        adapter._sleep_if_needed()
        adapter._sleep_if_needed()  # second call hits the sleep branch
