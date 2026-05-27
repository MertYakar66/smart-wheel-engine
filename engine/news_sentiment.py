"""
News sentiment reader — operator-facing transparency layer.

Status (DECISIONS.md D18, 2026-05-26): verbal news is **severed** from
the EV decision path. ``sentiment_multiplier`` is now a constant-1.0
stub. ``get_ticker_sentiment`` still reads the sentiment store so the
dashboard / row-dict / morning-brief can surface the underlying
sentiment + article count to the operator, but the score has zero
influence on the EV verdict.

Why severed
-----------
The previous EV-path scoring was VADER + a tiny finance lexicon over
news headlines (see ``scripts/pull_news_sentiment.py``). That is a poor
fit for the kind of qualitative input that actually moves wheel
candidates ("China blocks Nvidia chips", "FTC sues exec for fraud",
etc.) — exactly the cases where a 50-word lexicon can flip sign on
syntactic accident. The right place for verbal news is the operator
brief, not a multiplier on EV. See DECISIONS.md D3 (superseded for the
verbal-news clause) and D18 (the severance and its rationale).

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
- wheel_runner: row dict surfaces ``news_sentiment`` / ``news_n_articles``
  for transparency; ``news_multiplier`` is read from
  ``sentiment_multiplier`` which is now constant 1.0 — keeping the call
  site preserves the audit trail without re-introducing the override
  problem D1 forbids.
- dashboard: exposes sentiment as a column in the candidate table for
  the operator to read alongside the engine verdict. The engine itself
  ignores it.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
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
        now = datetime.now(UTC)
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
        as_of: str | pd.Timestamp | None = None,
    ) -> dict:
        """Return the latest sentiment for a ticker (neutral if none).

        ``as_of`` makes the lookup point-in-time: only news timestamped
        within ``lookback_hours`` before and up to ``as_of`` is eligible
        — no look-ahead. ``as_of=None`` uses wall-clock ``now()`` (live
        behaviour). A store with no per-row ``as_of`` column cannot be
        made point-in-time, so a historical ``as_of`` returns neutral
        rather than risk a leak.
        """
        df = self._load()
        if df.empty or "ticker" not in df.columns:
            return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}

        t = df[df["ticker"] == ticker.upper()]
        if t.empty:
            return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}

        if "as_of" in t.columns:
            # Point-in-time window. as_of=None -> wall-clock now (live);
            # a historical as_of -> only news at or before that instant
            # is eligible. lookback_hours bounds the lower edge.
            upper = (
                pd.Timestamp.now(tz="UTC").tz_localize(None)
                if as_of is None
                else pd.Timestamp(as_of)
            )
            lower = upper - pd.Timedelta(hours=lookback_hours)
            t = t[(t["as_of"] >= lower) & (t["as_of"] <= upper)]
            if t.empty:
                return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}
            row = t.sort_values("as_of").iloc[-1]
        elif as_of is not None:
            # No per-row timestamp -> point-in-time cannot be
            # established; refuse rather than return the latest row.
            return {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}
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

    def sentiment_multiplier(
        self,
        ticker: str,
        lookback_hours: int = 72,
        as_of: str | pd.Timestamp | None = None,
    ) -> float:
        """Severed EV-path stub. Always returns 1.0.

        DECISIONS.md D18 severed verbal news from the EV decision path.
        Previously this function mapped sentiment to a multiplier in
        [0.88, 1.05]; that channel is now closed. The signature is
        preserved so the existing call site in
        :mod:`engine.wheel_runner` keeps its audit-trail shape (the
        underlying ``sentiment`` and ``n_articles`` are still surfaced
        on the row dict via :meth:`get_ticker_sentiment`), but the
        multiplier itself has no EV influence.

        The arguments are accepted (and a stash-lookup is performed so
        the disk store is still validated and the 5-minute cache stays
        warm for ``get_ticker_sentiment`` callers) but ignored for the
        return value.
        """
        # Touch the store so call-site behaviour around cache warm-up
        # and PIT validation is unchanged from the operator-dashboard
        # perspective. The result is discarded.
        _ = self.get_ticker_sentiment(ticker, lookback_hours, as_of)
        return 1.0
