"""
SEC EDGAR adapter.

EDGAR is free (SEC-mandated public access). The SEC requires a
``User-Agent`` header identifying the caller — we set a generic one by
default, override via env var ``SWE_EDGAR_UA``.

Data we pull
------------
- Form 4 (insider transactions) via the EDGAR full-text search and
  per-company submissions JSON.
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

    @lru_cache(maxsize=512)
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

    @lru_cache(maxsize=256)
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
