"""
Yahoo Finance adapter via the public download CSV endpoint.

No API key required. Works for any Yahoo-listed symbol including:
- DX-Y.NYB / UUP (Dollar index)
- CL=F / USO  (WTI crude)
- GC=F / GLD  (Gold)
- BTC-USD     (Bitcoin spot)
- ^TNX / ^FVX (Treasury proxies)
- Sector ETFs (XLK, XLF, XLE, ...)

Primary consumers:
- portfolio_copula (sector beta)
- regime_detector (commodity + FX regime)
- tail_risk (crypto correlation as risk-proxy)
"""

from __future__ import annotations

import io
import logging
import time
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_YF_URL = "https://query1.finance.yahoo.com/v7/finance/download/{sym}"
_YF_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Symbols the engine cares about for macro/cross-asset context
CROSS_ASSET_PACK = {
    "dxy": "DX-Y.NYB",
    "wti_crude": "CL=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "nat_gas": "NG=F",
    "btc_usd": "BTC-USD",
    "eth_usd": "ETH-USD",
    "eur_usd": "EURUSD=X",
    "usd_jpy": "USDJPY=X",
}

SECTOR_ETFS = {
    "tech": "XLK",
    "financials": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "consumer_disc": "XLY",
    "consumer_staples": "XLP",
    "utilities": "XLU",
    "industrials": "XLI",
    "materials": "XLB",
    "real_estate": "XLRE",
    "communications": "XLC",
}


class YFinanceAdapter:
    def __init__(self, timeout: int = 15, lookback_days: int = 365) -> None:
        self.timeout = timeout
        self.lookback_days = lookback_days
        self._session = requests.Session()
        self._session.headers.update(_YF_HEADERS)

    @lru_cache(maxsize=256)
    def get_ohlcv(self, symbol: str, period_days: int | None = None) -> pd.DataFrame:
        """Daily OHLCV for any Yahoo-listed symbol.

        Returns DataFrame indexed by date, columns open/high/low/close/volume.
        Empty on error.
        """
        days = period_days or self.lookback_days
        period2 = int(time.time())
        period1 = period2 - days * 86400
        url = _YF_URL.format(sym=symbol)
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
        }
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            logger.debug("YF fetch failed for %s", symbol, exc_info=True)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[cols]

    def latest_close(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, period_days=10)
        if df.empty or "close" not in df.columns:
            return float("nan")
        return float(df["close"].iloc[-1])

    def cross_asset_snapshot(self) -> dict:
        """Latest level + 1-day return for the cross-asset pack."""
        out = {}
        for name, sym in CROSS_ASSET_PACK.items():
            df = self.get_ohlcv(sym, period_days=10)
            if df.empty or "close" not in df.columns or len(df) < 2:
                out[name] = {"price": float("nan"), "ret_1d": float("nan")}
                continue
            last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
            out[name] = {
                "price": last,
                "ret_1d": (last - prev) / prev if prev else float("nan"),
            }
        return out

    def sector_returns(self, period_days: int = 30) -> dict:
        """Per-sector cumulative return over the window.

        Feeds portfolio concentration / sector rotation dashboards.
        """
        out = {}
        for sector, sym in SECTOR_ETFS.items():
            df = self.get_ohlcv(sym, period_days=period_days + 5)
            if df.empty or "close" not in df.columns or len(df) < 2:
                out[sector] = float("nan")
                continue
            last, first = float(df["close"].iloc[-1]), float(df["close"].iloc[0])
            out[sector] = (last - first) / first if first else float("nan")
        return out
