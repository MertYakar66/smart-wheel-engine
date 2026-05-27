"""
SEC EDGAR adapter.

EDGAR is free (SEC-mandated public access). The SEC requires a
``User-Agent`` header identifying the caller — we set a generic one by
default, override via env var ``SWE_EDGAR_UA``.

Data we pull
------------
- Form 4 (insider transactions) via the EDGAR full-text search and
  per-company submissions JSON.
- Form 8-K Item 2.02 (Results of Operations and Financial Condition) —
  the PIT-correct earnings-release event marker. See
  ``recent_8k_filings`` / ``earnings_history`` / ``project_next_earnings``.
  Wired in by the news-architecture redesign campaign (DECISIONS.md D18
  and the EDGAR campaign PR).
- Form 13F (institutional holdings, quarterly).
- Short interest: SEC short-sale transactions are reported through FINRA
  and published on ``regsho.finra.org`` — covered in a separate branch.

Rate limits
-----------
SEC requires no more than 10 requests/second. We use a small sleep
between calls and cache aggressively.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"

_UA = os.environ.get("SWE_EDGAR_UA", "smart-wheel-engine research@example.com")


class EDGARAdapter:
    def __init__(self, timeout: int = 15, min_interval_s: float = 0.12) -> None:
        self.timeout = timeout
        self.min_interval_s = min_interval_s
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _UA,
                "Accept": "application/json, text/html",
                "Host-Accept": "data.sec.gov",
            }
        )
        self._last_call_at = 0.0

    def _sleep_if_needed(self):
        delta = time.time() - self._last_call_at
        if delta < self.min_interval_s:
            time.sleep(self.min_interval_s - delta)
        self._last_call_at = time.time()

    @lru_cache(maxsize=512)  # noqa: B019 — short-lived adapter; per-instance cache is intended
    def cik_for_ticker(self, ticker: str) -> str | None:
        """Resolve a ticker to a zero-padded CIK (10-digit string)."""
        self._sleep_if_needed()
        try:
            resp = self._session.get(
                "https://www.sec.gov/files/company_tickers.json",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception:
            return None

        for entry in raw.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        return None

    @lru_cache(maxsize=256)  # noqa: B019 — short-lived adapter; per-instance cache is intended
    def recent_insider_trades(self, ticker: str, n_days: int = 90) -> pd.DataFrame:
        """Recent insider trades (Form 4) for a ticker.

        Returns DataFrame with columns: filing_date, accession, form,
        filer_name, primaryDocument. Empty on error.
        """
        cik = self.cik_for_ticker(ticker)
        if cik is None:
            return pd.DataFrame()

        self._sleep_if_needed()
        url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "filing_date": recent.get("filingDate", []),
                "accession": recent.get("accessionNumber", []),
                "form": recent.get("form", []),
                "primaryDocument": recent.get("primaryDocument", []),
            }
        )
        if df.empty:
            return df
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.dropna(subset=["filing_date"])
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
        df = df[(df["filing_date"] >= cutoff) & (df["form"] == "4")]
        return df.sort_values("filing_date", ascending=False).reset_index(drop=True)

    def insider_activity_signal(self, ticker: str, n_days: int = 30) -> dict:
        """Summarise insider activity over the last N days.

        Returns dict with 'filings_count', 'last_filing_date', 'active'.
        'active' = True if any Form 4 filed within the window.
        """
        df = self.recent_insider_trades(ticker, n_days=n_days)
        if df.empty:
            return {
                "filings_count": 0,
                "last_filing_date": None,
                "active": False,
            }
        return {
            "filings_count": int(len(df)),
            "last_filing_date": df["filing_date"].iloc[0].date().isoformat(),
            "active": True,
        }

    # ------------------------------------------------------------------
    # 8-K filings (earnings releases via Item 2.02)
    # ------------------------------------------------------------------
    def _fetch_submissions(self, ticker: str) -> dict | None:
        """Internal helper — fetch the raw SEC submissions JSON for ``ticker``.

        Returns the full payload as a dict, or ``None`` on CIK miss /
        network error. Callers handle the slicing.
        """
        cik = self.cik_for_ticker(ticker)
        if cik is None:
            return None
        self._sleep_if_needed()
        url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _items_contains(items_field: str, target: str) -> bool:
        """Whether ``target`` (e.g. ``"2.02"``) is in an 8-K filing's
        comma-separated ``items`` field. Tolerant of whitespace and the
        ``Item `` prefix the SEC occasionally emits."""
        if not items_field:
            return False
        target = target.strip()
        for raw in str(items_field).split(","):
            tok = raw.strip()
            # SEC sometimes prefixes "Item " — strip it.
            if tok.lower().startswith("item "):
                tok = tok[5:].strip()
            if tok == target:
                return True
        return False

    def recent_8k_filings(
        self,
        ticker: str,
        items_filter: str | None = "2.02",
        since: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Recent Form 8-K filings for a ticker.

        Args:
            ticker: equity ticker (case-insensitive).
            items_filter: SEC 8-K item code to filter on. Default
                ``"2.02"`` = "Results of Operations and Financial
                Condition" — the earnings-release marker. Pass ``None``
                to return all 8-Ks regardless of items.
            since: lower-bound filing date (``YYYY-MM-DD`` string or
                Timestamp). ``None`` = no lower bound (returns the full
                ``recent`` block, typically the last ~1000 filings the
                SEC index serves).

        Returns DataFrame with columns: ``filing_date`` (Timestamp),
        ``accession`` (str), ``form`` (str), ``items`` (str — comma-
        separated 8-K item codes, ``""`` if none), ``primary_document``
        (str — relative URL within the filing). Sorted by ``filing_date``
        descending. Empty DataFrame on CIK miss / network error.
        """
        data = self._fetch_submissions(ticker)
        if data is None:
            return pd.DataFrame()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "filing_date": recent.get("filingDate", []),
                "accession": recent.get("accessionNumber", []),
                "form": recent.get("form", []),
                "items": recent.get("items", []),
                "primary_document": recent.get("primaryDocument", []),
            }
        )
        if df.empty:
            return df
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.dropna(subset=["filing_date"])
        df = df[df["form"] == "8-K"]
        if since is not None:
            df = df[df["filing_date"] >= pd.Timestamp(since)]
        if items_filter is not None:
            df = df[df["items"].apply(lambda v: self._items_contains(v, items_filter))]
        return df.sort_values("filing_date", ascending=False).reset_index(drop=True)

    def earnings_history(
        self,
        ticker: str,
        since: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Historical earnings-release filings (8-K Item 2.02) for a ticker.

        Thin convenience wrapper around :meth:`recent_8k_filings` with
        the earnings-release item code. The returned DataFrame is
        sorted **ascending** (oldest first) so callers can compute
        inter-filing deltas directly without re-sorting.
        """
        df = self.recent_8k_filings(ticker, items_filter="2.02", since=since)
        if df.empty:
            return df
        return df.sort_values("filing_date", ascending=True).reset_index(drop=True)

    def project_next_earnings(
        self,
        ticker: str,
        as_of: str | pd.Timestamp | None = None,
        min_history: int = 3,
    ) -> dict | None:
        """Project the next expected earnings-release date for ``ticker``.

        EDGAR is a record of past filings — there is no native
        "next earnings calendar" endpoint. We project forward by taking
        the median inter-filing delta of the most recent ``min_history``+
        Item 2.02 filings on or before ``as_of`` and adding it to the
        last known filing date. This is the same heuristic professional
        data vendors (Zacks, Earnings Whispers) use under the hood — the
        S&P 500 earnings cadence is well-behaved (quarterly ±10 days).

        Args:
            ticker: equity ticker.
            as_of: cutoff timestamp; only filings ``<= as_of`` feed the
                projection (PIT discipline). ``None`` = wall-clock now.
            min_history: minimum number of historical Item 2.02 filings
                required before a projection is returned. Default 3
                (one year of quarterly cadence).

        Returns a dict matching the
        ``MarketDataConnector.get_next_earnings`` shape:
            {
                "announcement_date": pd.Timestamp,
                "announcement_time": None,
                "estimate_eps": None,
                "year_period": None,
                "source": "edgar_projection",
                "n_history": int,
                "median_quarter_days": float,
            }
        Returns ``None`` if fewer than ``min_history`` filings are on or
        before ``as_of``.
        """
        ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
        hist = self.earnings_history(ticker)
        if hist.empty:
            return None
        hist = hist[hist["filing_date"] <= ref]
        if len(hist) < min_history:
            return None

        # Inter-filing deltas in days; median is robust to occasional
        # delayed filings (e.g. when an 8-K Item 2.02 also bundles
        # restated guidance, the SEC accept timestamp may slip).
        deltas = hist["filing_date"].diff().dropna().dt.days
        if deltas.empty:
            return None
        median_days = float(deltas.median())
        last_filing = hist["filing_date"].iloc[-1]
        projected = last_filing + pd.Timedelta(days=int(round(median_days)))
        # Don't project a date that's already past — clamp forward to
        # the next multiple of the median delta beyond ``ref``.
        while projected <= ref:
            projected = projected + pd.Timedelta(days=int(round(median_days)))

        return {
            "announcement_date": projected,
            "announcement_time": None,
            "estimate_eps": None,
            "year_period": None,
            "source": "edgar_projection",
            "n_history": int(len(hist)),
            "median_quarter_days": median_days,
        }
