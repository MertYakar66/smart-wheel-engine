"""
News sentiment reader — a thin bridge between the existing
``news_pipeline`` / ``financial_news`` subsystems and the wheel EV ranker.

The news pipeline is orchestrated separately (scrapers, browser agents,
publishers). This module does not run any scraping itself — it just
looks for a parquet/json/sqlite store on disk and exposes a simple
``get_ticker_sentiment(ticker, lookback_hours=72)`` -> dict.

Expected store locations (checked in order)
-------------------------------------------
1. ``data_processed/news_sentiment.parquet``   (wide ticker×sentiment)
2. ``data/news/sentiment.parquet``
3. ``financial_news/storage/sentiment.sqlite``

Expected schema (wide, latest per ticker):
    ticker (str), as_of (datetime), sentiment (float in [-1, +1]),
    confidence (float in [0, 1]), n_articles (int)

If no store is found, every lookup returns ``{'sentiment': 0.0,
'confidence': 0.0, 'n_articles': 0}`` — neutral by default.

Primary consumers
-----------------
- wheel_runner: can optionally apply a sentiment multiplier to the
  regime scaling. Strong negative sentiment (< -0.3) with ≥ 5 articles
  triggers a soft de-rank (multiplier 0.90).
- dashboard: exposes sentiment as a column in the candidate table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_CANDIDATE_PATHS = (
    Path("data_processed/news_sentiment.parquet"),
    Path("data/news/sentiment.parquet"),
    Path("data_processed/news_sentiment.csv"),
    Path("data/news/sentiment.csv"),
)

_SQLITE_PATH = Path("financial_news/storage/sentiment.sqlite")


class NewsSentimentReader:
    """Read sentiment scores from the news pipeline's output store."""

    def __init__(self, base_dir: str | Path = ".") -> None:
        self.base_dir = Path(base_dir)
        self._cache: pd.DataFrame | None = None
        self._loaded_at: datetime | None = None
        self._cache_ttl = timedelta(minutes=5)

    def _load(self) -> pd.DataFrame:
        """Load sentiment store, caching the result for 5 minutes."""
        now = datetime.now(timezone.utc)
        if (
            self._cache is not None
            and self._loaded_at is not None
            and now - self._loaded_at < self._cache_ttl
        ):
            return self._cache

        df = pd.DataFrame()
        for rel in _CANDIDATE_PATHS:
            p = self.base_dir / rel
            if not p.exists():
                continue
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                break
            except Exception:
                logger.debug("Failed to read %s", p, exc_info=True)

        if df.empty:
            sqlite_path = self.base_dir / _SQLITE_PATH
            if sqlite_path.exists():
                try:
                    conn = sqlite3.connect(str(sqlite_path))
                    df = pd.read_sql_query("SELECT * FROM sentiment", conn)
                    conn.close()
                except Exception:
                    logger.debug("SQLite sentiment read failed", exc_info=True)

        if not df.empty:
            df.columns = [c.lower().strip() for c in df.columns]
            if "as_of" in df.columns:
                df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].astype(str).str.upper()

        self._cache = df
        self._loaded_at = now
        return df

    def get_ticker_sentiment(
        self,
        ticker: str,
        lookback_hours: int = 72,
    ) -> dict:
        """Return the latest sentiment for a ticker (neutral if none)."""
        df = self._load()
        if df.empty or "ticker" not in df.columns:
            return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}

        t = df[df["ticker"] == ticker.upper()]
        if t.empty:
            return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}

        if "as_of" in t.columns:
            cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(
                hours=lookback_hours
            )
            t = t[t["as_of"] >= cutoff]
            if t.empty:
                return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}
            row = t.sort_values("as_of").iloc[-1]
        else:
            row = t.iloc[-1]

        def _safe_float(v):
            try:
                f = float(v)
                return f if np.isfinite(f) else 0.0
            except Exception:
                return 0.0

        def _safe_int(v):
            try:
                return int(v)
            except Exception:
                return 0

        return {
            "sentiment": _safe_float(row.get("sentiment", 0.0)),
            "confidence": _safe_float(row.get("confidence", 0.0)),
            "n_articles": _safe_int(row.get("n_articles", 0)),
            "as_of": row.get("as_of"),
        }

    def sentiment_multiplier(self, ticker: str, lookback_hours: int = 72) -> float:
        """Map sentiment to an EV multiplier in [0.88, 1.05].

        - sentiment <= -0.3 with n_articles >= 5 -> 0.88 (soft derank)
        - sentiment in (-0.3, -0.1)              -> 0.95
        - neutral                                 -> 1.00
        - sentiment >= 0.3 with n_articles >= 5  -> 1.05
        """
        s = self.get_ticker_sentiment(ticker, lookback_hours)
        sent = s["sentiment"]
        n = s["n_articles"]
        if n < 5:
            return 1.0
        if sent <= -0.3:
            return 0.88
        if sent <= -0.1:
            return 0.95
        if sent >= 0.3:
            return 1.05
        return 1.0
