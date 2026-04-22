"""
FRED (Federal Reserve Economic Data) adapter.

FRED publishes a free-to-use CSV endpoint for every series. No API key
is strictly required for CSV downloads via ``fred.stlouisfed.org``.
For higher volume / programmatic use, set ``FRED_API_KEY`` as an env
var and the adapter will use the JSON API instead.

Series we care about for the wheel engine
-----------------------------------------
- DFF            Federal Funds Effective Rate (daily)
- DGS3MO         3-Month Treasury Constant Maturity Rate
- DGS2            2-Year Treasury Constant Maturity Rate
- DGS10           10-Year Treasury Constant Maturity Rate
- T10Y2Y         10Y - 2Y Treasury spread (yield curve proxy)
- BAMLH0A0HYM2   HY Corporate OAS (credit risk proxy)
- BAMLC0A0CM     IG Corporate OAS
- CPIAUCSL       CPI, All Urban Consumers (monthly)
- PCE            Personal Consumption Expenditures (monthly)
- UNRATE         Unemployment rate
- PAYEMS         Non-farm Payrolls
- NAPM           ISM Manufacturing Index
- SOFR            Secured Overnight Financing Rate

These feed:
- regime_detector (rate regime + credit-stress gates)
- event_gate (CPI/NFP/FOMC dates)
- risk_manager (HY/IG spread widening = de-risk)
"""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Default series pack consumed by the engine
SERIES_PACK = {
    "fed_funds": "DFF",
    "treasury_3m": "DGS3MO",
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
    "curve_2y10y": "T10Y2Y",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "cpi": "CPIAUCSL",
    "pce": "PCE",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "sofr": "SOFR",
}


class FREDAdapter:
    """Thin wrapper for FRED series downloads with caching."""

    def __init__(self, api_key: str | None = None, timeout: int = 15) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.timeout = timeout
        self._session = requests.Session()

    @lru_cache(maxsize=64)
    def get_series(self, series_id: str) -> pd.Series:
        """Return a FRED series as pd.Series indexed by date.

        Missing values / server errors → empty Series.
        """
        try:
            if self.api_key:
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                }
                resp = self._session.get(_FRED_API_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
                df = pd.DataFrame(obs)
                if df.empty or "date" not in df or "value" not in df:
                    return pd.Series(dtype=float)
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                return df.set_index("date")["value"].dropna().sort_index()

            # CSV fallback (no API key needed)
            params = {"id": series_id}
            resp = self._session.get(_FRED_CSV_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if df.empty or df.shape[1] < 2:
                return pd.Series(dtype=float)
            date_col = df.columns[0]
            val_col = df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            return df.set_index(date_col)[val_col].dropna().sort_index()
        except Exception:
            logger.debug("FRED fetch failed for %s", series_id, exc_info=True)
            return pd.Series(dtype=float)

    def latest(self, series_id: str) -> float:
        """Return the latest observation, or NaN if unavailable."""
        s = self.get_series(series_id)
        return float(s.iloc[-1]) if not s.empty else float("nan")

    def snapshot(self, pack: dict[str, str] | None = None) -> dict[str, float]:
        """Return {name: latest value} for every series in the pack."""
        pack = pack or SERIES_PACK
        return {name: self.latest(sid) for name, sid in pack.items()}

    def history(self, pack: dict[str, str] | None = None) -> pd.DataFrame:
        """Return a wide DataFrame with one column per series in the pack."""
        pack = pack or SERIES_PACK
        frames = []
        for name, sid in pack.items():
            s = self.get_series(sid)
            if not s.empty:
                frames.append(s.rename(name))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    # ------------------------------------------------------------------
    # Derived regime signals
    # ------------------------------------------------------------------

    def credit_regime(self) -> dict:
        """HY/IG OAS levels + percentile rank over 5 years.

        Returns {'hy_oas', 'ig_oas', 'hy_pct_5y', 'ig_pct_5y', 'regime'}
        where regime is 'benign' | 'stressed' | 'crisis' based on
        HY OAS percentile (>80th = stressed, >95th = crisis).
        """
        hy = self.get_series("BAMLH0A0HYM2")
        ig = self.get_series("BAMLC0A0CM")
        out = {
            "hy_oas": float(hy.iloc[-1]) if not hy.empty else float("nan"),
            "ig_oas": float(ig.iloc[-1]) if not ig.empty else float("nan"),
            "hy_pct_5y": float("nan"),
            "ig_pct_5y": float("nan"),
            "regime": "unknown",
        }
        if not hy.empty:
            hy5 = hy.tail(5 * 252)
            out["hy_pct_5y"] = float((hy5 < hy.iloc[-1]).mean())
            if out["hy_pct_5y"] > 0.95:
                out["regime"] = "crisis"
            elif out["hy_pct_5y"] > 0.80:
                out["regime"] = "stressed"
            else:
                out["regime"] = "benign"
        if not ig.empty:
            ig5 = ig.tail(5 * 252)
            out["ig_pct_5y"] = float((ig5 < ig.iloc[-1]).mean())
        return out

    def yield_curve_signal(self) -> dict:
        """Yield-curve shape + inversion flag.

        Inverted curve (2s10s < 0) has preceded every US recession
        for the last 60 years — a hard risk-off signal for the wheel.
        """
        s = self.get_series("T10Y2Y")
        if s.empty:
            return {"spread_2y10y": float("nan"), "inverted": False}
        last = float(s.iloc[-1])
        return {
            "spread_2y10y": last,
            "inverted": bool(last < 0),
            "spread_1m_change": (
                float(last - s.iloc[-22]) if len(s) >= 22 else float("nan")
            ),
        }
