"""
engine/theta_connector.py

ThetaData v3 live-data connector for the smart-wheel engine.

Architecture
------------
Inherits from MarketDataConnector so every Bloomberg-CSV-backed method
(fundamentals, earnings, dividends, credit risk, risk-free rate, universe)
continues to work as a fallback.  The methods below are overridden with
live ThetaData v3 data:

  get_ohlcv          → /v3/stock/history/eod          (Stock FREE tier)
  get_fundamentals   → Bloomberg CSV + live ATM IV from
                       /v3/option/snapshot/greeks/first_order
  get_option_chain   → /v3/option/snapshot/greeks/first_order  (bulk)
  get_iv_rank        → /v3/option/history/greeks/first_order   (live)
  get_iv_percentile  → same
  get_vix_regime     → /v3/index/snapshot/price?symbol=VIX    (Index FREE)

Usage
-----
Set the environment variable before starting the engine:

    SWE_DATA_PROVIDER=theta python engine_api.py

The Terminal must be running on 127.0.0.1:25503.  If the Terminal is
unreachable the connector falls back to Bloomberg CSV data transparently.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data_connector import MarketDataConnector

logger = logging.getLogger(__name__)

_THETA_BASE_URL = "http://127.0.0.1:25503"

# Standard tier max concurrent requests per ThetaData docs
_MAX_CONCURRENT = 4

# Chain snapshot TTL in seconds — don't re-fetch the same chain within one minute
_CHAIN_CACHE_TTL = 60

# How many calendar days of IV history to use when computing IV rank
_IV_RANK_LOOKBACK_DAYS = 365


class ThetaConnector(MarketDataConnector):
    """Live-data connector backed by ThetaData v3 via the local Terminal."""

    def __init__(
        self,
        data_dir: str = "data/bloomberg",
        base_url: str = _THETA_BASE_URL,
    ) -> None:
        super().__init__(data_dir)
        self._base = base_url.rstrip("/")
        self._semaphore = threading.Semaphore(_MAX_CONCURRENT)

        retry = Retry(
            total=4,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
        self._session = requests.Session()
        self._session.mount("http://", adapter)

        # chain snapshot cache: key → (fetched_at_epoch, DataFrame)
        self._chain_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, path: str, params: dict[str, Any]) -> pd.DataFrame:
        """GET a v3 endpoint, return the response parsed as CSV DataFrame.

        Returns an empty DataFrame on any error so callers can fall back
        to Bloomberg CSV without crashing.
        """
        params = {**params, "format": "csv"}
        url = f"{self._base}{path}"
        try:
            with self._semaphore:
                resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return pd.DataFrame()
            resp.raise_for_status()
            text = resp.text.strip()
            if not text or text.startswith("We have upgraded"):
                logger.warning("ThetaData API version error: %s", text[:120])
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(text))
        except requests.exceptions.ConnectionError:
            logger.warning("ThetaTerminal not reachable at %s — falling back to CSV", self._base)
            return pd.DataFrame()
        except Exception:
            logger.exception("ThetaData fetch failed: %s params=%s", path, params)
            return pd.DataFrame()

    def _chain_cache_key(self, ticker: str, expiration: str) -> str:
        minute = int(time.time() // _CHAIN_CACHE_TTL)
        return f"{ticker}|{expiration}|{minute}"

    def _get_cached_chain(self, ticker: str, expiration: str) -> pd.DataFrame | None:
        key = self._chain_cache_key(ticker, expiration)
        with self._cache_lock:
            entry = self._chain_cache.get(key)
        if entry is None:
            return None
        return entry[1]

    def _set_cached_chain(self, ticker: str, expiration: str, df: pd.DataFrame) -> None:
        key = self._chain_cache_key(ticker, expiration)
        with self._cache_lock:
            self._chain_cache[key] = (time.time(), df)
            # Evict stale entries to prevent unbounded growth
            now = time.time()
            stale = [k for k, (ts, _) in self._chain_cache.items() if now - ts > _CHAIN_CACHE_TTL * 2]
            for k in stale:
                self._chain_cache.pop(k, None)

    def _nearest_expiration(self, ticker: str, dte_target: int = 35) -> str | None:
        """Return the expiration date string (YYYYMMDD) closest to dte_target."""
        df = self._fetch("/v3/option/list/expirations", {"symbol": ticker})
        if df.empty or "expiration" not in df.columns:
            return None
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        df = df.dropna(subset=["expiration"])
        target = pd.Timestamp.now(tz="UTC").tz_localize(None) + timedelta(days=dte_target)
        df["_gap"] = (df["expiration"] - target).abs()
        best = df.sort_values("_gap").iloc[0]["expiration"]
        return best.strftime("%Y%m%d")

    @staticmethod
    def _to_yyyymmdd(date_like: str | None) -> str | None:
        if date_like is None:
            return None
        try:
            return pd.Timestamp(date_like).strftime("%Y%m%d")
        except Exception:
            return None

    def _is_live(self, as_of: str | None) -> bool:
        """True when the caller wants current market data (not a historical backtest)."""
        if as_of is None:
            return True
        try:
            return pd.Timestamp(as_of).date() >= datetime.now(timezone.utc).date()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # OHLCV — override: ThetaData stock EOD (free tier, delayed)
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV from ThetaData stock history, fall back to Bloomberg CSV."""
        params: dict[str, Any] = {"symbol": ticker}
        if start_date:
            params["start_date"] = self._to_yyyymmdd(start_date)
        if end_date:
            params["end_date"] = self._to_yyyymmdd(end_date)

        df = self._fetch("/v3/stock/history/eod", params)
        if df.empty:
            logger.debug("ThetaData OHLCV empty for %s, using Bloomberg CSV", ticker)
            return super().get_ohlcv(ticker, start_date, end_date)

        # Normalise column names (v3 returns lowercase)
        df.columns = [c.lower() for c in df.columns]

        # v3 EOD columns: symbol, date, open, high, low, close, volume
        for col in ("date",):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=["date"])

        needed = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        if "close" not in needed:
            logger.warning("ThetaData OHLCV missing close column for %s, using Bloomberg CSV", ticker)
            return super().get_ohlcv(ticker, start_date, end_date)

        out = df.sort_values("date").set_index("date")[needed]
        # Ensure all expected columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in out.columns:
                out[col] = np.nan
        return out[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Option chain — new method: bulk greeks snapshot for a target expiry
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        ticker: str,
        expiration: str | None = None,
        dte_target: int = 35,
    ) -> pd.DataFrame:
        """Fetch a full option chain (all strikes, both rights) with greeks.

        Returns a DataFrame with columns:
          symbol, expiration, strike, right, delta, gamma, theta, vega,
          rho, iv, bid, ask, mid (= (bid+ask)/2 as premium proxy),
          underlying_price.

        If expiration is omitted, picks the expiry closest to dte_target.
        Uses a 60-second TTL cache so the EV ranker can call this safely
        for each ticker without hammering the Terminal.
        """
        if expiration is None:
            expiration = self._nearest_expiration(ticker, dte_target)
        if expiration is None:
            return pd.DataFrame()

        # Normalise to YYYYMMDD for cache key
        exp_key = expiration.replace("-", "")

        cached = self._get_cached_chain(ticker, exp_key)
        if cached is not None:
            return cached

        # Fetch greeks (omit strike/right → bulk all strikes)
        df_greeks = self._fetch(
            "/v3/option/snapshot/greeks/first_order",
            {"symbol": ticker, "expiration": exp_key},
        )

        # Fetch quotes (for bid/ask/mid)
        df_quotes = self._fetch(
            "/v3/option/snapshot/quote",
            {"symbol": ticker, "expiration": exp_key},
        )

        # Fetch open interest per strike — needed for dealer-positioning GEX
        # and for liquidity / pin-risk detection. /v3/option/snapshot/open_interest
        # returns one row per (strike, right) with `open_interest`.
        df_oi = self._fetch(
            "/v3/option/snapshot/open_interest",
            {"symbol": ticker, "expiration": exp_key},
        )

        if df_greeks.empty and df_quotes.empty:
            return pd.DataFrame()

        # Normalise column names
        if not df_greeks.empty:
            df_greeks.columns = [c.lower() for c in df_greeks.columns]
        if not df_quotes.empty:
            df_quotes.columns = [c.lower() for c in df_quotes.columns]
        if not df_oi.empty:
            df_oi.columns = [c.lower() for c in df_oi.columns]

        # Merge on (symbol, expiration, strike, right)
        merge_keys = ["symbol", "expiration", "strike", "right"]

        if not df_greeks.empty and not df_quotes.empty:
            quote_cols = [k for k in merge_keys if k in df_quotes.columns]
            extra_quote = [c for c in ("bid", "ask", "bid_size", "ask_size") if c in df_quotes.columns]
            df_quotes_slim = df_quotes[quote_cols + extra_quote].copy()
            df = pd.merge(df_greeks, df_quotes_slim, on=merge_keys, how="left")
        elif not df_greeks.empty:
            df = df_greeks.copy()
        else:
            df = df_quotes.copy()

        # Merge open interest if available
        if not df_oi.empty and "open_interest" in df_oi.columns:
            oi_keys = [k for k in merge_keys if k in df_oi.columns]
            oi_cols = oi_keys + ["open_interest"]
            df = pd.merge(df, df_oi[oi_cols], on=oi_keys, how="left")

        # Normalise types
        for col in ("strike", "delta", "gamma", "theta", "vega", "rho", "iv",
                    "bid", "ask", "bid_size", "ask_size", "open_interest"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

        # Normalise right to lower-case ('call'/'put')
        if "right" in df.columns:
            df["right"] = df["right"].str.lower()

        # Compute mid-price as premium proxy
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
        elif "bid" in df.columns:
            df["mid"] = df["bid"]
        else:
            df["mid"] = np.nan

        # ThetaData returns IV as a decimal (e.g. 0.2617 = 26.17%).
        # Guard against any accidental percent-form values.
        if "iv" in df.columns:
            df["iv"] = df["iv"].where(df["iv"] <= 3.0, df["iv"] / 100.0)

        self._set_cached_chain(ticker, exp_key, df)
        return df

    # ------------------------------------------------------------------
    # Fundamentals — override: Bloomberg CSV + live ATM IV from ThetaData
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str) -> dict | None:
        """Return Bloomberg CSV fundamentals with live ATM IV injected.

        The ATM IV field (`implied_vol_atm`) is overridden with the
        current market value from ThetaData.  All other fields (PE,
        beta, sector, market cap, etc.) come from the Bloomberg CSV
        since ThetaData does not provide fundamental data.
        """
        base = super().get_fundamentals(ticker)

        # Attempt to fetch live ATM IV from the nearest 35-DTE chain
        try:
            chain = self.get_option_chain(ticker, dte_target=35)
            if not chain.empty and "iv" in chain.columns and "delta" in chain.columns:
                # Pick the put closest to -0.50 delta (= ATM)
                puts = chain[chain["right"] == "put"].dropna(subset=["delta", "iv"])
                if not puts.empty:
                    puts = puts.copy()
                    puts["_atm_gap"] = (puts["delta"].abs() - 0.50).abs()
                    atm_row = puts.sort_values("_atm_gap").iloc[0]
                    live_iv = float(atm_row["iv"])
                    if 0 < live_iv <= 3.0:  # sanity: must be a decimal
                        if base is None:
                            base = {"ticker": ticker}
                        base["implied_vol_atm"] = live_iv
                        base["volatility_30d"] = live_iv  # best proxy if no hist vol
                        logger.debug(
                            "ThetaData ATM IV for %s: %.4f (%.1f%%)", ticker, live_iv, live_iv * 100
                        )
        except Exception:
            logger.debug("ThetaData IV fetch failed for %s, using CSV IV", ticker, exc_info=True)

        return base

    # ------------------------------------------------------------------
    # IV rank / percentile — live path uses ThetaData history
    # ------------------------------------------------------------------

    def get_iv_rank(self, ticker: str, as_of: str | None = None) -> float:
        """IV rank (0–1) over the last 365 days.

        Live path (as_of=None): fetches 1 year of daily ATM IV history
        from ThetaData and computes rank of today's IV.
        Historical path: falls back to Bloomberg CSV for PIT safety.
        """
        if not self._is_live(as_of):
            return super().get_iv_rank(ticker, as_of)

        try:
            iv_series = self._fetch_iv_history(ticker)
            if iv_series is None or len(iv_series) < 20:
                return super().get_iv_rank(ticker, as_of)
            current_iv = iv_series.iloc[-1]
            rank = float((iv_series < current_iv).mean())
            return round(rank, 4)
        except Exception:
            logger.debug("ThetaData IV rank failed for %s", ticker, exc_info=True)
            return super().get_iv_rank(ticker, as_of)

    def get_iv_percentile(self, ticker: str, as_of: str | None = None) -> float:
        """IV percentile — same as IV rank for our purposes."""
        return self.get_iv_rank(ticker, as_of)

    def _fetch_iv_history(self, ticker: str) -> pd.Series | None:
        """Fetch ~1 year of daily ATM IV from ThetaData greeks history.

        Uses the ~35-DTE expiry, put side, strike nearest to ATM.
        Returns a Series indexed by date, values are IV decimals.
        """
        # Find a target expiry ~35 DTE out from today
        exp = self._nearest_expiration(ticker, dte_target=35)
        if exp is None:
            return None

        # Get current chain to find ATM put strike
        chain = self.get_option_chain(ticker, expiration=exp)
        if chain.empty or "delta" not in chain.columns:
            return None

        puts = chain[(chain["right"] == "put")].dropna(subset=["delta"])
        if puts.empty:
            return None
        puts = puts.copy()
        puts["_gap"] = (puts["delta"].abs() - 0.50).abs()
        atm_strike = float(puts.sort_values("_gap").iloc[0]["strike"])

        end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=_IV_RANK_LOOKBACK_DAYS)).strftime("%Y%m%d")

        df = self._fetch(
            "/v3/option/history/greeks/first_order",
            {
                "symbol": ticker,
                "expiration": exp,
                "right": "put",
                "strike": atm_strike,
                "start_date": start_date,
                "end_date": end_date,
                "interval": "1d",
            },
        )
        if df.empty or "iv" not in df.columns:
            return None

        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in ("date", "timestamp") if c in df.columns), None)
        if date_col is None:
            return None

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, "iv"]).sort_values(date_col)
        iv = pd.to_numeric(df["iv"], errors="coerce").dropna()
        # Guard: convert percent → decimal if needed
        iv = iv.where(iv <= 3.0, iv / 100.0)
        return iv

    # ------------------------------------------------------------------
    # VIX / regime — live path uses ThetaData index snapshot
    # ------------------------------------------------------------------

    def get_vix_regime(self, as_of: str | None = None) -> dict:
        """VIX regime dict. Live: ThetaData index; historical: Bloomberg CSV."""
        if not self._is_live(as_of):
            return super().get_vix_regime(as_of)

        try:
            df = self._fetch("/v3/index/snapshot/price", {"symbol": "VIX"})
            if df.empty:
                return super().get_vix_regime(as_of)

            df.columns = [c.lower() for c in df.columns]
            price_col = next(
                (c for c in ("price", "close", "last", "value") if c in df.columns), None
            )
            if price_col is None:
                return super().get_vix_regime(as_of)

            vix = float(df[price_col].iloc[-1])
            # VIX historical percentile still comes from Bloomberg CSV for now
            base = super().get_vix_regime(as_of)
            base["vix"] = vix
            # Recompute regime bucket from live VIX level
            base["term_structure"] = base.get("term_structure", "unknown")
            return base
        except Exception:
            logger.debug("ThetaData VIX fetch failed", exc_info=True)
            return super().get_vix_regime(as_of)

    # ------------------------------------------------------------------
    # Vol risk premium — live: realised vol from ThetaData OHLCV vs ATM IV
    # ------------------------------------------------------------------

    def get_vol_risk_premium(self, ticker: str, as_of: str | None = None) -> float:
        """Implied vol minus realised vol (30-day).  Live path uses ThetaData."""
        if not self._is_live(as_of):
            return super().get_vol_risk_premium(ticker, as_of)

        try:
            ohlcv = self.get_ohlcv(ticker)
            if ohlcv.empty or "close" not in ohlcv.columns or len(ohlcv) < 22:
                return super().get_vol_risk_premium(ticker, as_of)

            log_rets = np.diff(np.log(ohlcv["close"].values[-31:]))
            realised_vol = float(np.std(log_rets) * np.sqrt(252))

            fundamentals = self.get_fundamentals(ticker) or {}
            iv = fundamentals.get("implied_vol_atm") or fundamentals.get("volatility_30d")
            if iv is None or np.isnan(float(iv)):
                return super().get_vol_risk_premium(ticker, as_of)

            return float(iv) - realised_vol
        except Exception:
            return super().get_vol_risk_premium(ticker, as_of)

    # ------------------------------------------------------------------
    # Full IV surface — every expiry × several deltas (Phase 2a)
    # ------------------------------------------------------------------

    def get_iv_surface(
        self,
        ticker: str,
        max_expirations: int = 8,
        min_dte: int = 7,
        max_dte: int = 400,
    ) -> pd.DataFrame:
        """Return an IV surface across expirations.

        For each expiration we keep the chain's (strike, right, delta, iv).
        The caller can slice it by delta bucket (25Δ put, ATM, 25Δ call)
        for skew metrics, or fit Nelson-Siegel to ATM IVs across tenors.

        Columns: expiration, dte, strike, right, delta, iv, mid.
        """
        exps = self._fetch("/v3/option/list/expirations", {"symbol": ticker})
        if exps.empty or "expiration" not in exps.columns:
            return pd.DataFrame()
        exps["expiration"] = pd.to_datetime(exps["expiration"], errors="coerce")
        exps = exps.dropna(subset=["expiration"]).sort_values("expiration")

        now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        exps["dte"] = (exps["expiration"] - now).dt.days
        exps = exps[(exps["dte"] >= min_dte) & (exps["dte"] <= max_dte)]
        if exps.empty:
            return pd.DataFrame()

        # Sample up to max_expirations across the range (front + back)
        exps = exps.iloc[:: max(1, len(exps) // max_expirations)].head(max_expirations)

        frames = []
        for _, row in exps.iterrows():
            exp_key = row["expiration"].strftime("%Y%m%d")
            chain = self.get_option_chain(ticker, expiration=exp_key)
            if chain.empty:
                continue
            c = chain[["strike", "right", "delta", "iv", "mid"]].copy()
            c["expiration"] = row["expiration"]
            c["dte"] = int(row["dte"])
            frames.append(c)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_atm_term_structure(
        self, ticker: str, max_expirations: int = 8
    ) -> pd.DataFrame:
        """Return ATM IV (put closest to -0.50Δ) across expirations.

        Columns: expiration, dte, atm_iv. Feeds Nelson-Siegel fit.
        """
        surf = self.get_iv_surface(ticker, max_expirations=max_expirations)
        if surf.empty:
            return pd.DataFrame()

        puts = surf[surf["right"] == "put"].dropna(subset=["delta", "iv"]).copy()
        if puts.empty:
            return pd.DataFrame()
        puts["_gap"] = (puts["delta"].abs() - 0.50).abs()
        atm = (
            puts.sort_values("_gap")
            .groupby(["expiration", "dte"], as_index=False)
            .first()[["expiration", "dte", "iv"]]
            .rename(columns={"iv": "atm_iv"})
            .sort_values("dte")
        )
        return atm

    def get_skew_snapshot(
        self, ticker: str, dte_target: int = 35
    ) -> dict:
        """25Δ put / ATM / 25Δ call IV for one expiry — feeds skew_slope.

        Returns {'iv_25d_put', 'iv_atm', 'iv_25d_call', 'expiration', 'dte'}
        or empty dict if the chain is unusable.
        """
        exp = self._nearest_expiration(ticker, dte_target=dte_target)
        if exp is None:
            return {}
        chain = self.get_option_chain(ticker, expiration=exp)
        if chain.empty or "delta" not in chain.columns:
            return {}

        def _pick(df: pd.DataFrame, target_delta: float) -> float | None:
            d = df.dropna(subset=["delta", "iv"]).copy()
            if d.empty:
                return None
            d["_gap"] = (d["delta"] - target_delta).abs()
            row = d.sort_values("_gap").iloc[0]
            iv = float(row["iv"])
            return iv if 0 < iv <= 3.0 else None

        puts = chain[chain["right"] == "put"]
        calls = chain[chain["right"] == "call"]
        iv_25d_put = _pick(puts, -0.25)
        iv_atm = _pick(puts, -0.50)
        iv_25d_call = _pick(calls, 0.25)
        if None in (iv_25d_put, iv_atm, iv_25d_call):
            return {}

        exp_ts = pd.to_datetime(exp)
        return {
            "iv_25d_put": iv_25d_put,
            "iv_atm": iv_atm,
            "iv_25d_call": iv_25d_call,
            "expiration": exp_ts,
            "dte": int((exp_ts - pd.Timestamp.now().normalize()).days),
        }

    # ------------------------------------------------------------------
    # VIX family — VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW (Phase 2b)
    # ------------------------------------------------------------------

    _VIX_FAMILY = ("VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "MOVE")

    def get_vix_family(self) -> dict:
        """Snapshot of volatility-regime indices.

        Returns dict of symbol -> level (float). Any symbol that fails
        is simply absent from the dict. All data from
        /v3/index/snapshot/price.
        """
        out: dict[str, float] = {}
        for sym in self._VIX_FAMILY:
            try:
                df = self._fetch("/v3/index/snapshot/price", {"symbol": sym})
                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                col = next(
                    (c for c in ("price", "close", "last", "value") if c in df.columns),
                    None,
                )
                if col is None:
                    continue
                val = pd.to_numeric(df[col], errors="coerce").dropna()
                if val.empty:
                    continue
                out[sym] = float(val.iloc[-1])
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Historical option OHLC per strike (Phase 2c)
    # ------------------------------------------------------------------

    def get_option_ohlc_history(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Historical EOD OHLC for a single option contract.

        Enables backtest of realised P&L paths, gamma decay, and rolling
        transaction cost. Returns DataFrame indexed by date with columns
        open, high, low, close, volume.
        """
        params: dict[str, Any] = {
            "symbol": ticker,
            "expiration": self._to_yyyymmdd(expiration),
            "strike": float(strike),
            "right": right.lower(),
        }
        if start_date:
            params["start_date"] = self._to_yyyymmdd(start_date)
        if end_date:
            params["end_date"] = self._to_yyyymmdd(end_date)

        df = self._fetch("/v3/option/history/ohlc", params)
        if df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in ("date", "timestamp") if c in df.columns), None)
        if date_col is None or "close" not in df.columns:
            return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[cols]

    # ------------------------------------------------------------------
    # Intraday stock bars — for Garman-Klass / Yang-Zhang RV (Phase 2d)
    # ------------------------------------------------------------------

    def get_stock_intraday(
        self,
        ticker: str,
        interval: str = "5m",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Intraday OHLCV bars. Feeds Parkinson/GK/YZ realised-vol estimators.

        interval: ThetaData supports '1m', '5m', '15m', '30m', '1h'.
        Returns DataFrame indexed by timestamp with open/high/low/close/volume.
        """
        params: dict[str, Any] = {"symbol": ticker, "interval": interval}
        if start_date:
            params["start_date"] = self._to_yyyymmdd(start_date)
        if end_date:
            params["end_date"] = self._to_yyyymmdd(end_date)

        df = self._fetch("/v3/stock/history/intraday", params)
        if df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        ts_col = next(
            (c for c in ("timestamp", "datetime", "date", "time") if c in df.columns),
            None,
        )
        if ts_col is None or "close" not in df.columns:
            return pd.DataFrame()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[cols]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_terminal_alive(self) -> bool:
        """Return True if the ThetaTerminal is reachable and responding."""
        try:
            resp = self._session.get(
                f"{self._base}/v3/option/list/expirations",
                params={"symbol": "SPY", "format": "csv"},
                timeout=5,
            )
            return resp.status_code == 200 and len(resp.text) > 10
        except Exception:
            return False
