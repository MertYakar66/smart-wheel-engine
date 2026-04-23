#!/usr/bin/env python3
"""
News sentiment puller → writes the parquet the existing NewsSentimentReader expects.

Today ``engine/news_sentiment.py`` checks four paths and finds nothing, so
``sentiment_multiplier(ticker)`` returns 1.0 for every candidate. This puller
fills in the most-used path (``data_processed/news_sentiment.parquet``).

Providers
---------
``--provider polygon``  (default; free tier covers news)
``--provider finnhub``  (also free tier)
``--provider benzinga`` (paid)

Each provider reads its API key from an env var:
    POLYGON_API_KEY / FINNHUB_API_KEY / BENZINGA_API_KEY

Output schema (matches ``engine/news_sentiment.NewsSentimentReader``)
--------------------------------------------------------------------
    ticker     str            upper-case
    as_of      datetime[ns]   timestamp of the aggregation window end
    sentiment  float          weighted mean over lookback, in [-1, +1]
    confidence float          0..1 — low when n_articles is small
    n_articles int            count in the window

Sentiment-scoring fallback
--------------------------
If the provider doesn't return a sentiment score (Polygon free tier),
the script computes one locally with VADER if available, else a tiny
finance lexicon. Set ``--scorer vader`` / ``--scorer lexicon`` / ``--scorer provider``
(fail if provider doesn't supply).

Usage
-----
    # S&P 500, last 72 hours, Polygon free tier
    export POLYGON_API_KEY=xxx
    python scripts/pull_news_sentiment.py --universe sp500 --hours 72

    # Single-ticker smoke
    python scripts/pull_news_sentiment.py --tickers AAPL MSFT --hours 24
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import requests
except ImportError:
    print("requests is required: pip install requests")
    raise

logger = logging.getLogger(__name__)
OUT_PATH = _ROOT / "data_processed" / "news_sentiment.parquet"

# ----------------------------------------------------------------------
# Sentiment scorers
# ----------------------------------------------------------------------
_LEXICON_POS = {
    "beat", "beats", "surpass", "surge", "strong", "upgrade", "upgraded",
    "outperform", "record", "growth", "raised", "profit", "rally",
    "bullish", "breakthrough", "milestone", "accelerate", "advance",
    "top", "topped", "exceed", "boost", "positive",
}
_LEXICON_NEG = {
    "miss", "missed", "lawsuit", "sued", "plunge", "downgrade", "downgraded",
    "underperform", "warning", "warn", "weak", "disappoint", "cut", "slash",
    "bearish", "fraud", "investigation", "probe", "decline", "loss", "losses",
    "restate", "restatement", "delist", "halt", "halted", "recall", "bankrupt",
    "bankruptcy", "default", "breach", "negative",
}


def _lexicon_score(text: str) -> float:
    """Crude bag-of-words sentiment in [-1, 1]. Zero if no keywords hit."""
    if not text:
        return 0.0
    toks = [t.strip(".,!?()[]{}\"'").lower() for t in text.split()]
    pos = sum(1 for t in toks if t in _LEXICON_POS)
    neg = sum(1 for t in toks if t in _LEXICON_NEG)
    if pos + neg == 0:
        return 0.0
    return float((pos - neg) / (pos + neg))


_vader = None


def _vader_score(text: str) -> float:
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            _vader = False
    if _vader is False:
        return _lexicon_score(text)
    return float(_vader.polarity_scores(text or "")["compound"])


def _score_article(text: str, scorer: str) -> float:
    if scorer == "vader":
        return _vader_score(text)
    return _lexicon_score(text)


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------
class NewsProvider:
    name = "base"
    api_key_env = ""

    def __init__(self) -> None:
        self.api_key = os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"{self.name}: set {self.api_key_env} environment variable "
                f"(see .env.example for expected name)"
            )
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "smart-wheel-engine/1.0"

    def fetch(self, ticker: str, since: datetime) -> list[dict]:
        """Return a list of article dicts with at least these keys:
        {title, description, published_utc, provider_sentiment (or None)}"""
        raise NotImplementedError


class PolygonProvider(NewsProvider):
    name = "polygon"
    api_key_env = "POLYGON_API_KEY"
    URL = "https://api.polygon.io/v2/reference/news"

    def fetch(self, ticker: str, since: datetime) -> list[dict]:
        params = {
            "ticker": ticker,
            "published_utc.gte": since.strftime("%Y-%m-%d"),
            "limit": 100,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": self.api_key,
        }
        r = self.session.get(self.URL, params=params, timeout=15)
        if r.status_code == 401:
            raise RuntimeError("polygon: 401 unauthorized — check POLYGON_API_KEY")
        if r.status_code != 200:
            raise RuntimeError(f"polygon: HTTP {r.status_code}: {r.text[:200]}")
        payload = r.json()
        out: list[dict] = []
        for a in payload.get("results", []):
            # Paid tier has "insights" with per-ticker sentiment.
            provider_sent = None
            insights = a.get("insights") or []
            for ins in insights:
                if str(ins.get("ticker", "")).upper() == ticker.upper():
                    s = str(ins.get("sentiment", "")).lower()
                    provider_sent = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(s)
                    break
            out.append(
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "published_utc": a.get("published_utc"),
                    "provider_sentiment": provider_sent,
                }
            )
        return out


class FinnhubProvider(NewsProvider):
    name = "finnhub"
    api_key_env = "FINNHUB_API_KEY"
    URL = "https://finnhub.io/api/v1/company-news"

    def fetch(self, ticker: str, since: datetime) -> list[dict]:
        params = {
            "symbol": ticker,
            "from": since.strftime("%Y-%m-%d"),
            "to": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "token": self.api_key,
        }
        r = self.session.get(self.URL, params=params, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"finnhub: HTTP {r.status_code}: {r.text[:200]}")
        data = r.json() or []
        out: list[dict] = []
        for a in data:
            ts = a.get("datetime")
            iso = (
                datetime.fromtimestamp(ts, timezone.utc).isoformat()
                if isinstance(ts, (int, float))
                else None
            )
            out.append(
                {
                    "title": a.get("headline", ""),
                    "description": a.get("summary", ""),
                    "published_utc": iso,
                    "provider_sentiment": None,
                }
            )
        return out


class BenzingaProvider(NewsProvider):
    name = "benzinga"
    api_key_env = "BENZINGA_API_KEY"
    URL = "https://api.benzinga.com/api/v2/news"

    def fetch(self, ticker: str, since: datetime) -> list[dict]:
        params = {
            "tickers": ticker,
            "dateFrom": since.strftime("%Y-%m-%d"),
            "pageSize": 50,
            "token": self.api_key,
        }
        r = self.session.get(self.URL, params=params, timeout=15,
                             headers={"Accept": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"benzinga: HTTP {r.status_code}: {r.text[:200]}")
        data = r.json() or []
        out: list[dict] = []
        for a in data:
            out.append(
                {
                    "title": a.get("title", ""),
                    "description": a.get("teaser", ""),
                    "published_utc": a.get("created"),
                    "provider_sentiment": None,
                }
            )
        return out


PROVIDERS = {
    "polygon": PolygonProvider,
    "finnhub": FinnhubProvider,
    "benzinga": BenzingaProvider,
}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def _aggregate(articles: list[dict], scorer: str, as_of: datetime,
               lookback_hours: int) -> tuple[float, float, int]:
    """Weighted-mean sentiment with exponential recency decay."""
    if not articles:
        return 0.0, 0.0, 0
    scores: list[float] = []
    weights: list[float] = []
    cutoff = as_of - timedelta(hours=lookback_hours)
    for a in articles:
        ts_raw = a.get("published_utc")
        if ts_raw is None:
            continue
        try:
            ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
        except Exception:
            continue
        if ts < cutoff:
            continue
        text = f"{a.get('title','')}. {a.get('description','')}"
        if scorer == "provider" and a.get("provider_sentiment") is not None:
            s = float(a["provider_sentiment"])
        else:
            s = _score_article(text, scorer)
        # Exponential decay, half-life = 24h
        age_h = max(0.0, (as_of - ts).total_seconds() / 3600.0)
        w = 0.5 ** (age_h / 24.0)
        scores.append(s)
        weights.append(w)
    if not scores:
        return 0.0, 0.0, 0
    arr = np.array(scores)
    w = np.array(weights)
    mean = float(np.average(arr, weights=w))
    # Confidence grows with sqrt(n) up to n=20
    n = len(arr)
    confidence = min(1.0, np.sqrt(n / 20.0))
    return mean, float(confidence), n


def _process_ticker(provider: NewsProvider, ticker: str, lookback_hours: int,
                    scorer: str) -> tuple[str, float, float, int, str]:
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours + 6)
    try:
        articles = provider.fetch(ticker, since)
    except Exception as e:
        return ticker, 0.0, 0.0, 0, f"FAIL {type(e).__name__}: {e}"
    sentiment, conf, n = _aggregate(
        articles, scorer, datetime.now(timezone.utc), lookback_hours
    )
    detail = f"n={n} s={sentiment:+.3f} c={conf:.2f}"
    return ticker, sentiment, conf, n, detail


def load_universe(mode: str, pit_date: str | None = None) -> list[str]:
    if mode == "sp500":
        from data.consolidated_loader import get_bloomberg_loader
        L = get_bloomberg_loader()
        tickers = L.get_universe_as_of(pit_date)
        return sorted({t for t in tickers if all(c.isalpha() or c == "." for c in t)})
    raise ValueError(f"Unknown universe {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--universe", choices=["sp500"])
    ap.add_argument("--pit-date")
    ap.add_argument("--hours", type=int, default=72, help="Lookback window")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="polygon")
    ap.add_argument("--scorer", choices=["lexicon", "vader", "provider"], default="vader")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--dry-run", action="store_true", help="Fetch but don't write")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        provider = PROVIDERS[args.provider]()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe:
        tickers = load_universe(args.universe, args.pit_date)
    else:
        print("ERROR: --tickers or --universe required")
        return 2

    print(f"News-sentiment pull  provider={args.provider}  scorer={args.scorer}  "
          f"tickers={len(tickers)}  hours={args.hours}  workers={args.workers}")

    t0 = time.perf_counter()
    rows: list[dict] = []
    n_done = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_ticker, provider, t, args.hours, args.scorer): t
                for t in tickers}
        for fut in as_completed(futs):
            ticker, sent, conf, n, detail = fut.result()
            n_done += 1
            rows.append({
                "ticker": ticker,
                "as_of": datetime.now(timezone.utc),
                "sentiment": sent,
                "confidence": conf,
                "n_articles": n,
            })
            if detail.startswith("FAIL"):
                n_err += 1
                print(f"  [{n_done:>4}/{len(tickers)}] {ticker:<6}  {detail[:80]}", flush=True)
            elif n_done % 50 == 0 or n > 0:
                print(f"  [{n_done:>4}/{len(tickers)}] {ticker:<6}  {detail}", flush=True)

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)

    if args.dry_run:
        print(f"\nDry run: {len(df)} rows computed in {elapsed:.1f}s")
        print(df.head(10).to_string(index=False))
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows → {out}")
    print(f"Done in {elapsed:.1f}s  |  {n_err} provider errors")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
