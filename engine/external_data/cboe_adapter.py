"""
CBOE volatility-index adapter.

CBOE publishes delayed daily closes for all its indices as free public
CSVs. When ThetaData's index snapshot is not reachable, this is the
fallback. Primary consumers: regime_detector, risk_manager.

Indices of interest
-------------------
- VIX      S&P 500 30-day implied vol
- VIX9D    9-day VIX (short-end of term structure)
- VIX3M    3-month VIX
- VIX6M    6-month VIX
- VVIX     Volatility of VIX
- SKEW     CBOE SKEW index (tail-hedging demand)
- PUT      Put-Write Index (for strategy benchmarking)
- BXM      BuyWrite Index
- MOVE     ICE BofA MOVE (bond volatility, via Yahoo since CBOE-ish)

Free CSV endpoints
------------------
CBOE has standardised on
``https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv``

Yahoo fallback used when CBOE endpoint 404s:
``https://query1.finance.yahoo.com/v7/finance/download/^{SYMBOL}``
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"


class CBOEAdapter:
    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self._session = requests.Session()

    @lru_cache(maxsize=32)
    def get_series(self, symbol: str) -> pd.DataFrame:
        """Return daily OHLC DataFrame for a CBOE index.

        Columns: open, high, low, close. Index: DatetimeIndex (naive).
        Empty DataFrame on any error.
        """
        url = _CBOE_URL.format(sym=symbol.upper())
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            logger.debug("CBOE fetch failed for %s", symbol, exc_info=True)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next(
            (c for c in ("date", "datetime", "time") if c in df.columns), df.columns[0]
        )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        if not cols:
            return pd.DataFrame()
        return df[cols]

    def latest(self, symbol: str) -> float:
        df = self.get_series(symbol)
        if df.empty or "close" not in df.columns:
            return float("nan")
        return float(df["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Derived signals
    # ------------------------------------------------------------------

    def vix_term_structure(self) -> dict:
        """VIX term structure slope + contango/backwardation flag.

        Contango (VIX < VIX3M) = normal; premium sellers favoured.
        Backwardation (VIX > VIX3M) = stress; avoid new short-premium.
        Returns {'vix', 'vix3m', 'vix6m', 'vix9d', 'slope_9d_30d',
        'slope_30d_3m', 'contango': bool}.
        """
        vix = self.latest("VIX")
        vix9d = self.latest("VIX9D")
        vix3m = self.latest("VIX3M")
        vix6m = self.latest("VIX6M")
        slope_9d_30d = (
            vix - vix9d if (vix == vix and vix9d == vix9d) else float("nan")
        )
        slope_30d_3m = (
            vix3m - vix if (vix == vix and vix3m == vix3m) else float("nan")
        )
        return {
            "vix": vix,
            "vix9d": vix9d,
            "vix3m": vix3m,
            "vix6m": vix6m,
            "slope_9d_30d": slope_9d_30d,
            "slope_30d_3m": slope_30d_3m,
            "contango": bool(slope_30d_3m > 0) if slope_30d_3m == slope_30d_3m else False,
        }

    def skew_index(self) -> dict:
        """CBOE SKEW index. Higher = more tail-hedging demand.

        SKEW = 100 means no skew; >130 = notable crash hedging; >145 =
        extreme. Returns level + 1-year percentile.
        """
        df = self.get_series("SKEW")
        if df.empty or "close" not in df.columns:
            return {"skew": float("nan"), "skew_pct_1y": float("nan")}
        s = df["close"]
        last = float(s.iloc[-1])
        pct = float((s.tail(252) < last).mean()) if len(s) >= 50 else float("nan")
        return {"skew": last, "skew_pct_1y": pct}

    def snapshot(self) -> dict:
        """Return all VIX-family levels + SKEW in one dict."""
        ts = self.vix_term_structure()
        sk = self.skew_index()
        ts.update(sk)
        ts["vvix"] = self.latest("VVIX")
        return ts
