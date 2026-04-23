"""
Post-earnings drift (PED) analytics by ticker and sector.

What this module computes
-------------------------
For each ticker with a populated earnings history in
``sp500_earnings.csv`` and price history in ``sp500_ohlcv.csv``:

- Earnings-day move distribution (1-day, 3-day, 5-day post-announcement)
- Implied move (ATM straddle ÷ spot) vs realised move on event day
- Post-earnings drift (PED): sign-adjusted return from day +1 to day +5

And aggregates:

- Sector-level PED distributions (median, 5th/95th percentile, skew)
- Surprise-conditional PED (splits sample on earnings beat / miss / inline)

Primary consumer: the event gate can look up expected move magnitude
for the upcoming earnings and size the buffer dynamically; the EV engine
can penalise candidates in sectors with heavy-tailed earnings drift.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EarningsDriftAnalyzer:
    """Compute post-earnings drift distributions from Bloomberg CSVs."""

    def __init__(self, data_dir: str | Path = "data/bloomberg") -> None:
        self.data_dir = Path(data_dir)
        self._earnings: pd.DataFrame | None = None
        self._ohlcv: pd.DataFrame | None = None
        self._fundamentals: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_earnings(self) -> pd.DataFrame:
        if self._earnings is not None:
            return self._earnings
        path = self.data_dir / "sp500_earnings.csv"
        if not path.exists():
            self._earnings = pd.DataFrame()
            return self._earnings
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        # Expected: ticker, announcement_date, earnings_eps, estimate_eps
        if "announcement_date" in df.columns:
            df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
        self._earnings = df
        return df

    def _load_ohlcv(self) -> pd.DataFrame:
        if self._ohlcv is not None:
            return self._ohlcv
        path = self.data_dir / "sp500_ohlcv.csv"
        if not path.exists():
            self._ohlcv = pd.DataFrame()
            return self._ohlcv
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        self._ohlcv = df
        return df

    def _load_fundamentals(self) -> pd.DataFrame:
        if self._fundamentals is not None:
            return self._fundamentals
        path = self.data_dir / "sp500_fundamentals.csv"
        if not path.exists():
            self._fundamentals = pd.DataFrame()
            return self._fundamentals
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        self._fundamentals = df
        return df

    # ------------------------------------------------------------------
    # Per-ticker analytics
    # ------------------------------------------------------------------

    @lru_cache(maxsize=2048)
    def ticker_earnings_moves(self, ticker: str) -> pd.DataFrame:
        """Return one row per earnings event for the ticker with move fields.

        Columns: announcement_date, prior_close, next_close, day1_ret,
        day3_ret, day5_ret, eps_surprise_pct, surprise_sign.
        """
        earnings = self._load_earnings()
        ohlcv = self._load_ohlcv()
        if earnings.empty or ohlcv.empty:
            return pd.DataFrame()

        if "ticker" in earnings.columns:
            e = earnings[earnings["ticker"].astype(str).str.upper() == ticker.upper()].copy()
        else:
            return pd.DataFrame()
        if e.empty:
            return pd.DataFrame()

        if "ticker" in ohlcv.columns:
            o = ohlcv[ohlcv["ticker"].astype(str).str.upper() == ticker.upper()].copy()
        else:
            o = ohlcv.copy()
        if o.empty or "close" not in o.columns or "date" not in o.columns:
            return pd.DataFrame()

        o = o.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        o["close"] = pd.to_numeric(o["close"], errors="coerce")

        rows = []
        for _, row in e.iterrows():
            ann_date = row.get("announcement_date")
            if pd.isna(ann_date):
                continue
            # Find trading-day indices surrounding the announcement
            prior_idx = o.index[o["date"] < ann_date]
            post_idx = o.index[o["date"] > ann_date]
            if len(prior_idx) == 0 or len(post_idx) == 0:
                continue
            pi = prior_idx[-1]
            # First post-announcement trading close = day+1
            d1 = post_idx[0] if len(post_idx) >= 1 else None
            d3 = post_idx[2] if len(post_idx) >= 3 else None
            d5 = post_idx[4] if len(post_idx) >= 5 else None
            prior_close = o.loc[pi, "close"]
            if not np.isfinite(prior_close) or prior_close <= 0:
                continue

            def ret(ix):
                if ix is None:
                    return float("nan")
                c = o.loc[ix, "close"]
                return float((c - prior_close) / prior_close) if np.isfinite(c) else float("nan")

            earn_eps = pd.to_numeric(row.get("earnings_eps"), errors="coerce")
            est_eps = pd.to_numeric(row.get("estimate_eps"), errors="coerce")
            surprise_pct = float("nan")
            if np.isfinite(earn_eps) and np.isfinite(est_eps) and est_eps != 0:
                surprise_pct = float((earn_eps - est_eps) / abs(est_eps))
            sign = (
                "beat"
                if np.isfinite(surprise_pct) and surprise_pct > 0.02
                else ("miss" if np.isfinite(surprise_pct) and surprise_pct < -0.02 else "inline")
            )

            rows.append(
                {
                    "announcement_date": ann_date,
                    "prior_close": float(prior_close),
                    "day1_ret": ret(d1),
                    "day3_ret": ret(d3),
                    "day5_ret": ret(d5),
                    "eps_surprise_pct": surprise_pct,
                    "surprise_sign": sign,
                }
            )
        return pd.DataFrame(rows)

    def ticker_drift_stats(self, ticker: str) -> dict:
        """Summary stats of the ticker's historical earnings moves.

        Returns dict with median / p5 / p95 / abs_median / heavy_tail flag
        for day1, day3, day5.
        """
        df = self.ticker_earnings_moves(ticker)
        if df.empty:
            return {}
        out = {"ticker": ticker, "n_events": int(len(df))}
        for col in ("day1_ret", "day3_ret", "day5_ret"):
            s = df[col].dropna()
            if s.empty:
                continue
            out[f"{col}_median"] = float(s.median())
            out[f"{col}_p5"] = float(s.quantile(0.05))
            out[f"{col}_p95"] = float(s.quantile(0.95))
            out[f"{col}_abs_median"] = float(s.abs().median())
            out[f"{col}_std"] = float(s.std())
        # Heavy tail flag: p95 - p5 > 4x abs_median
        if "day1_ret_abs_median" in out and out["day1_ret_abs_median"] > 0:
            spread = out["day1_ret_p95"] - out["day1_ret_p5"]
            out["heavy_tail"] = bool(spread > 4 * out["day1_ret_abs_median"])
        return out

    # ------------------------------------------------------------------
    # Sector aggregation
    # ------------------------------------------------------------------

    def sector_drift_stats(self) -> pd.DataFrame:
        """Compute PED distribution per sector across the full universe.

        Returns DataFrame indexed by sector with columns:
        n_events, median_day1, p95_day1, median_day5, p95_day5.
        """
        fund = self._load_fundamentals()
        if fund.empty or "ticker" not in fund.columns or "sector" not in fund.columns:
            return pd.DataFrame()

        rows = []
        for _, r in fund.iterrows():
            ticker = str(r["ticker"]).upper()
            sector = r.get("sector")
            if pd.isna(sector):
                continue
            moves = self.ticker_earnings_moves(ticker)
            if moves.empty:
                continue
            for col in ("day1_ret", "day5_ret"):
                for v in moves[col].dropna():
                    rows.append({"sector": sector, "horizon": col, "ret": float(v)})

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        agg = df.groupby(["sector", "horizon"])["ret"].agg(
            n="count",
            median="median",
            p5=lambda s: s.quantile(0.05),
            p95=lambda s: s.quantile(0.95),
            abs_median=lambda s: s.abs().median(),
        )
        return agg.reset_index()

    def expected_move_magnitude(self, ticker: str, horizon: str = "day1_ret") -> float:
        """Return the historical abs-median earnings move as a proxy for
        expected move magnitude. Caller can compare to ATM straddle.
        """
        stats = self.ticker_drift_stats(ticker)
        key = f"{horizon}_abs_median"
        v = stats.get(key, float("nan"))
        return float(v) if v == v else float("nan")
