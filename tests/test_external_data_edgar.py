"""Tests for engine/external_data/edgar_adapter.py with HTTP mocking."""

from __future__ import annotations

import pandas as pd
import pytest
import requests_mock as rm_module

from engine.external_data.edgar_adapter import _EDGAR_BASE, EDGARAdapter

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


# =========================================================================
# 8-K filings (earnings releases via Item 2.02) — campaign PR3/9
# =========================================================================
def _submissions_with_items(
    filing_dates: list[str],
    forms: list[str],
    items: list[str],
) -> dict:
    """Submissions payload variant including the per-filing ``items``
    field that 8-K filings populate (and other forms leave empty)."""
    return {
        "filings": {
            "recent": {
                "filingDate": filing_dates,
                "accessionNumber": [f"acc-{i}" for i in range(len(filing_dates))],
                "form": forms,
                "items": items,
                "primaryDocument": [f"doc-{i}.htm" for i in range(len(filing_dates))],
            }
        }
    }


class TestItemsContains:
    """Pin the items-field parser tolerances: whitespace and the
    occasional ``Item `` prefix the SEC emits."""

    def test_comma_separated_match(self):
        assert EDGARAdapter._items_contains("1.01,2.02,9.01", "2.02") is True

    def test_single_item_match(self):
        assert EDGARAdapter._items_contains("2.02", "2.02") is True

    def test_whitespace_tolerant(self):
        assert EDGARAdapter._items_contains("1.01, 2.02 , 9.01", "2.02") is True

    def test_item_prefix_stripped(self):
        assert EDGARAdapter._items_contains("Item 2.02, Item 9.01", "2.02") is True

    def test_no_match(self):
        assert EDGARAdapter._items_contains("1.01,9.01", "2.02") is False

    def test_empty_field(self):
        assert EDGARAdapter._items_contains("", "2.02") is False

    def test_none_safe(self):
        # The DataFrame can carry None for the items field on non-8-K rows.
        assert EDGARAdapter._items_contains(None, "2.02") is False  # type: ignore[arg-type]


class Test8KFilingsFilter:
    def test_returns_only_8k_with_target_item(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        # Mix: an earnings 8-K, a non-earnings 8-K, a 10-K, an 8-K with
        # multiple items including 2.02.
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=[
                    "2026-04-30",  # 8-K with 2.02 — keep
                    "2026-03-15",  # 8-K without 2.02 — drop
                    "2026-02-01",  # 10-K — drop
                    "2026-01-30",  # 8-K with 2.02 in a multi-item list — keep
                ],
                forms=["8-K", "8-K", "10-K", "8-K"],
                items=["2.02", "1.01", "", "1.01,2.02,9.01"],
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.recent_8k_filings("AAPL", items_filter="2.02")
        assert len(df) == 2
        assert set(df["filing_date"].dt.strftime("%Y-%m-%d")) == {"2026-04-30", "2026-01-30"}
        # Sorted descending
        assert df.iloc[0]["filing_date"] > df.iloc[1]["filing_date"]

    def test_items_filter_none_returns_all_8k(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=["2026-04-30", "2026-03-15", "2026-02-01"],
                forms=["8-K", "8-K", "10-K"],
                items=["2.02", "1.01", ""],
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.recent_8k_filings("AAPL", items_filter=None)
        # Both 8-Ks; the 10-K is excluded by the form filter.
        assert len(df) == 2
        assert (df["form"] == "8-K").all()

    def test_since_filter(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=["2026-04-30", "2024-01-15", "2020-04-30"],
                forms=["8-K", "8-K", "8-K"],
                items=["2.02", "2.02", "2.02"],
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.recent_8k_filings("AAPL", items_filter="2.02", since="2023-01-01")
        assert len(df) == 2
        assert df["filing_date"].min() >= pd.Timestamp("2023-01-01")

    def test_unknown_ticker_returns_empty(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.recent_8k_filings("ZZZZ").empty

    def test_network_error_returns_empty(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            exc=ConnectionError("boom"),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.recent_8k_filings("AAPL").empty

    def test_empty_recent_block_returns_empty(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json={"filings": {"recent": {}}},
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.recent_8k_filings("AAPL").empty


class TestEarningsHistory:
    """``earnings_history`` is the Item-2.02 convenience wrapper, sorted
    ascending so callers can compute deltas directly."""

    def test_sorted_ascending(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=["2026-04-30", "2026-01-30", "2025-10-30", "2025-07-30"],
                forms=["8-K"] * 4,
                items=["2.02"] * 4,
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.earnings_history("AAPL")
        assert list(df["filing_date"].dt.strftime("%Y-%m-%d")) == [
            "2025-07-30",
            "2025-10-30",
            "2026-01-30",
            "2026-04-30",
        ]

    def test_only_2_02_filings(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=["2026-04-30", "2026-03-15"],
                forms=["8-K", "8-K"],
                items=["2.02", "1.01"],  # only the first is an earnings release
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        df = adapter.earnings_history("AAPL")
        assert len(df) == 1
        assert df.iloc[0]["filing_date"] == pd.Timestamp("2026-04-30")


class TestProjectNextEarnings:
    """The PIT-correct projection: median inter-filing delta added to
    the most recent known filing on or before ``as_of``."""

    def _quarterly_history(self) -> dict:
        # Four quarters of releases, ~91 days apart (roughly real-world
        # AAPL cadence).
        return _submissions_with_items(
            filing_dates=["2026-04-30", "2026-01-30", "2025-10-30", "2025-07-30"],
            forms=["8-K"] * 4,
            items=["2.02"] * 4,
        )

    def test_projects_forward_from_last_known(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(f"{_EDGAR_BASE}/submissions/CIK0000320193.json", json=self._quarterly_history())
        adapter = EDGARAdapter(min_interval_s=0.0)
        # as_of after the last known filing — projection should land
        # ~92 days after 2026-04-30, i.e. around late July 2026.
        res = adapter.project_next_earnings("AAPL", as_of="2026-05-15")
        assert res is not None
        assert res["source"] == "edgar_projection"
        assert res["n_history"] == 4
        # Median quarter is ~92 days. Projection from 2026-04-30 +92d
        # ≈ 2026-07-31. Allow ±5 days slack for median-vs-mean fudge.
        proj = res["announcement_date"]
        expected = pd.Timestamp("2026-07-31")
        assert abs((proj - expected).days) <= 5

    def test_pit_uses_only_filings_before_as_of(self, mock: rm_module.Mocker):
        """Calling project_next_earnings with a historical as_of must
        only feed filings whose ``filing_date <= as_of`` into the
        projection — no leak from the future."""
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(f"{_EDGAR_BASE}/submissions/CIK0000320193.json", json=self._quarterly_history())
        adapter = EDGARAdapter(min_interval_s=0.0)
        # Historical as_of: 2026-01-31 — only the three filings on or
        # before that date are eligible.
        res = adapter.project_next_earnings("AAPL", as_of="2026-01-31")
        assert res is not None
        assert res["n_history"] == 3
        # The projection should be after as_of and roughly +1 quarter
        # from the 2026-01-30 filing, NOT influenced by the future
        # 2026-04-30 filing (which would have nudged the median).
        proj = res["announcement_date"]
        assert proj > pd.Timestamp("2026-01-31")

    def test_below_min_history_returns_none(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        # Only two filings — below the default min_history=3.
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=["2026-04-30", "2026-01-30"],
                forms=["8-K"] * 2,
                items=["2.02"] * 2,
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.project_next_earnings("AAPL") is None

    def test_no_filings_returns_none(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json={"filings": {"recent": {}}},
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.project_next_earnings("AAPL") is None

    def test_unknown_ticker_returns_none(self, mock: rm_module.Mocker):
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        adapter = EDGARAdapter(min_interval_s=0.0)
        assert adapter.project_next_earnings("ZZZZ") is None

    def test_returned_dict_matches_get_next_earnings_shape(self, mock: rm_module.Mocker):
        """The return shape must be drop-in compatible with
        ``MarketDataConnector.get_next_earnings`` so a future wire-up
        in ``wheel_runner.py`` can swap source without restructuring
        the consumer."""
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        mock.get(f"{_EDGAR_BASE}/submissions/CIK0000320193.json", json=self._quarterly_history())
        adapter = EDGARAdapter(min_interval_s=0.0)
        res = adapter.project_next_earnings("AAPL", as_of="2026-05-15")
        assert res is not None
        # The consumer in wheel_runner.py reads .get("announcement_date")
        # and that's the field the EventGate consumes. Everything else
        # is optional metadata.
        assert "announcement_date" in res
        assert "announcement_time" in res
        assert "estimate_eps" in res
        assert "year_period" in res

    def test_projection_always_strictly_after_as_of(self, mock: rm_module.Mocker):
        """When the projected next-filing date computed from the
        median delta lands on or before ``as_of`` (e.g. the historical
        cadence is short and the last known filing is old), the
        projection must roll forward in delta-sized steps until it's
        strictly after ``as_of`` — otherwise the EventGate would never
        consider the projected event 'upcoming' and the lockout
        would silently no-op."""
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        # Tight 7-day cadence, last filing 2026-01-01. With as_of in
        # mid-2026, the naive +7 days lands in January — roll forward.
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=["2026-01-01", "2025-12-25", "2025-12-18", "2025-12-11"],
                forms=["8-K"] * 4,
                items=["2.02"] * 4,
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        res = adapter.project_next_earnings("AAPL", as_of="2026-05-15")
        assert res is not None
        assert res["announcement_date"] > pd.Timestamp("2026-05-15")

    def test_same_date_filings_do_not_stall_projection(self, mock: rm_module.Mocker):
        """Two Item 2.02 8-Ks on the same calendar date (an earnings
        release plus a same-day restated-guidance amendment, say) would
        inject a 0-day delta into the median; before the guard, that
        could collapse the median to 0 and the roll-forward loop would
        spin forever. After the fix the duplicate is dropped pre-median
        and the projection lands as if the duplicate had never been
        filed; n_history reflects the deduped count."""
        mock.get(COMPANY_TICKERS_URL, json=_ticker_index())
        # 5 raw filings → 4 unique dates after dedupe. Quarterly cadence
        # of ~91 days is unchanged because the same-date pair collapses
        # to one entry instead of injecting a stray 0.
        mock.get(
            f"{_EDGAR_BASE}/submissions/CIK0000320193.json",
            json=_submissions_with_items(
                filing_dates=[
                    "2026-04-30",
                    "2026-04-30",  # same-date duplicate
                    "2026-01-30",
                    "2025-10-30",
                    "2025-07-30",
                ],
                forms=["8-K"] * 5,
                items=["2.02"] * 5,
            ),
        )
        adapter = EDGARAdapter(min_interval_s=0.0)
        res = adapter.project_next_earnings("AAPL", as_of="2026-05-15")
        assert res is not None
        # Unique filing dates feed the projection — 4, not 5.
        assert res["n_history"] == 4
        # Median must be the cadence (~91 d), not 0.
        assert res["median_quarter_days"] > 0
        # Projection still lands ~92 days after 2026-04-30, i.e. late
        # July 2026 — the duplicate had no effect on the forward jump.
        proj = res["announcement_date"]
        expected = pd.Timestamp("2026-07-31")
        assert abs((proj - expected).days) <= 5


# =========================================================================
# Puller helper — scripts/pull_edgar_earnings.py
# =========================================================================
class TestMergeWithExisting:
    """Pin the parquet-merge contract for ``scripts/pull_edgar_earnings``:
    append-only mode concatenates and dedupes on ``(ticker, accession)``;
    refresh mode drops only the rows for successfully-refreshed tickers
    so a partial-refresh run never silently destroys prior data."""

    def _row(self, ticker: str, accession: str, day: str) -> dict:
        return {
            "ticker": ticker,
            "filing_date": pd.Timestamp(day),
            "accession": accession,
            "items": "2.02",
            "primary_document": f"{accession}.htm",
            "pulled_at": pd.Timestamp("2026-05-29"),
        }

    def test_refresh_preserves_other_tickers(self):
        """``--refresh`` with only AAPL in ``new_df`` must keep MSFT's
        prior rows untouched — the pre-fix behaviour overwrote the whole
        parquet with just AAPL."""
        from scripts.pull_edgar_earnings import merge_with_existing

        old_df = pd.DataFrame(
            [
                self._row("AAPL", "acc-aapl-1", "2026-01-30"),
                self._row("MSFT", "acc-msft-1", "2026-01-25"),
                self._row("MSFT", "acc-msft-2", "2026-04-25"),
            ]
        )
        new_df = pd.DataFrame(
            [
                self._row("AAPL", "acc-aapl-2", "2026-04-30"),
            ]
        )
        combined = merge_with_existing(old_df, new_df, refresh=True)
        # MSFT rows preserved.
        msft = combined[combined["ticker"] == "MSFT"]
        assert len(msft) == 2
        # AAPL's old accession dropped (re-pulled this run), new one in.
        aapl = combined[combined["ticker"] == "AAPL"]
        assert list(aapl["accession"]) == ["acc-aapl-2"]

    def test_refresh_failed_ticker_keeps_prior_rows(self):
        """If a refresh asks for AAPL+MSFT but only MSFT actually
        returns rows (AAPL hit a transient SEC failure / empty), AAPL's
        old rows must stay in the parquet — losing them would be a
        silent data destruction the operator wouldn't notice until the
        next backtest. (MSFT is fully replaced by the new pull because
        the SEC's recent-filings block is authoritative for any ticker
        we successfully re-pulled.)"""
        from scripts.pull_edgar_earnings import merge_with_existing

        old_df = pd.DataFrame(
            [
                self._row("AAPL", "acc-aapl-1", "2026-01-30"),
                self._row("MSFT", "acc-msft-1", "2026-01-25"),
            ]
        )
        # Refresh round only produced MSFT (AAPL failed → not in new_df).
        new_df = pd.DataFrame([self._row("MSFT", "acc-msft-2", "2026-04-25")])
        combined = merge_with_existing(old_df, new_df, refresh=True)
        # AAPL's prior row preserved despite AAPL being on the refresh list.
        aapl = combined[combined["ticker"] == "AAPL"]
        assert list(aapl["accession"]) == ["acc-aapl-1"]
        # MSFT was successfully refreshed → its prior row is dropped and
        # only the new pull's row remains.
        msft = combined[combined["ticker"] == "MSFT"]
        assert list(msft["accession"]) == ["acc-msft-2"]

    def test_append_mode_dedupes_on_ticker_accession(self):
        """Append-only mode is the default: concatenate then drop
        duplicates on ``(ticker, accession)`` keeping the latest row.
        The new pull's ``pulled_at`` should win for shared accessions."""
        from scripts.pull_edgar_earnings import merge_with_existing

        old_df = pd.DataFrame([self._row("AAPL", "acc-1", "2026-01-30")])
        new_df = pd.DataFrame(
            [
                self._row("AAPL", "acc-1", "2026-01-30"),  # same accession
                self._row("AAPL", "acc-2", "2026-04-30"),  # new accession
            ]
        )
        combined = merge_with_existing(old_df, new_df, refresh=False)
        assert sorted(combined["accession"].tolist()) == ["acc-1", "acc-2"]
