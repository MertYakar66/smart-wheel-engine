"""Tests for engine/news_sentiment.py — sentiment reader + EV multiplier."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from engine.news_sentiment import NewsSentimentReader


def _write_sentiment_csv(base_dir: Path, rows: list[dict]) -> Path:
    target = base_dir / "data_processed" / "news_sentiment.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


class TestNoStore:
    def test_neutral_when_no_store_exists(self, tmp_path: Path):
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL")
        assert result == {"sentiment": 0.0, "confidence": 0.0, "n_articles": 0}

    def test_multiplier_neutral_when_no_store(self, tmp_path: Path):
        reader = NewsSentimentReader(base_dir=tmp_path)
        assert reader.sentiment_multiplier("AAPL") == 1.0


class TestCsvStore:
    def test_reads_latest_per_ticker(self, tmp_path: Path):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        _write_sentiment_csv(tmp_path, [
            {
                "ticker": "AAPL",
                "as_of": now - timedelta(hours=24),
                "sentiment": 0.1,
                "confidence": 0.5,
                "n_articles": 3,
            },
            {
                "ticker": "AAPL",
                "as_of": now - timedelta(hours=1),
                "sentiment": 0.4,
                "confidence": 0.8,
                "n_articles": 9,
            },
        ])
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL")
        assert result["sentiment"] == pytest.approx(0.4)
        assert result["confidence"] == pytest.approx(0.8)
        assert result["n_articles"] == 9

    def test_lookback_horizon_filters_old_rows(self, tmp_path: Path):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        _write_sentiment_csv(tmp_path, [
            {
                "ticker": "AAPL",
                "as_of": now - timedelta(hours=200),
                "sentiment": 0.4,
                "confidence": 0.8,
                "n_articles": 9,
            }
        ])
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL", lookback_hours=72)
        assert result["sentiment"] == 0.0
        assert result["n_articles"] == 0

    def test_unknown_ticker_returns_neutral(self, tmp_path: Path):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        _write_sentiment_csv(tmp_path, [
            {
                "ticker": "AAPL",
                "as_of": now - timedelta(hours=1),
                "sentiment": 0.5,
                "confidence": 0.9,
                "n_articles": 7,
            }
        ])
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("MSFT")
        assert result["sentiment"] == 0.0
        assert result["n_articles"] == 0

    def test_ticker_case_normalised(self, tmp_path: Path):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        _write_sentiment_csv(tmp_path, [
            {
                "ticker": "aapl",
                "as_of": now - timedelta(hours=1),
                "sentiment": 0.4,
                "confidence": 0.8,
                "n_articles": 9,
            }
        ])
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL")
        assert result["n_articles"] == 9

    def test_no_as_of_column_uses_last_row(self, tmp_path: Path):
        target = tmp_path / "data_processed" / "news_sentiment.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"ticker": "AAPL", "sentiment": 0.2, "confidence": 0.5, "n_articles": 4},
            {"ticker": "AAPL", "sentiment": 0.7, "confidence": 0.9, "n_articles": 8},
        ]).to_csv(target, index=False)
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL")
        assert result["sentiment"] == pytest.approx(0.7)

    def test_non_numeric_values_coerced_to_neutral(self, tmp_path: Path):
        target = tmp_path / "data_processed" / "news_sentiment.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "ticker,as_of,sentiment,confidence,n_articles\n"
            f"AAPL,{datetime.now(timezone.utc).replace(tzinfo=None).isoformat()},bad,bad,bad\n"
        )
        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL")
        assert result["sentiment"] == 0.0
        assert result["confidence"] == 0.0
        assert result["n_articles"] == 0


class TestSqliteFallback:
    def test_reads_from_sqlite_when_no_csv(self, tmp_path: Path):
        sqlite_dir = tmp_path / "financial_news" / "storage"
        sqlite_dir.mkdir(parents=True)
        path = sqlite_dir / "sentiment.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE sentiment (ticker TEXT, as_of TEXT, sentiment REAL, "
            "confidence REAL, n_articles INTEGER)"
        )
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        conn.execute(
            "INSERT INTO sentiment VALUES (?, ?, ?, ?, ?)",
            ("AAPL", now, 0.3, 0.7, 6),
        )
        conn.commit()
        conn.close()

        reader = NewsSentimentReader(base_dir=tmp_path)
        result = reader.get_ticker_sentiment("AAPL")
        assert result["sentiment"] == pytest.approx(0.3)
        assert result["n_articles"] == 6


class TestSentimentMultiplier:
    def _setup(self, tmp_path: Path, sentiment: float, n_articles: int) -> NewsSentimentReader:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        _write_sentiment_csv(tmp_path, [
            {
                "ticker": "AAPL",
                "as_of": now - timedelta(hours=1),
                "sentiment": sentiment,
                "confidence": 0.8,
                "n_articles": n_articles,
            }
        ])
        return NewsSentimentReader(base_dir=tmp_path)

    def test_few_articles_means_neutral(self, tmp_path: Path):
        reader = self._setup(tmp_path, sentiment=-0.9, n_articles=3)
        assert reader.sentiment_multiplier("AAPL") == 1.0

    def test_strong_negative_with_5plus_articles_derank(self, tmp_path: Path):
        reader = self._setup(tmp_path, sentiment=-0.5, n_articles=8)
        assert reader.sentiment_multiplier("AAPL") == 0.88

    def test_mild_negative_with_5plus_articles(self, tmp_path: Path):
        reader = self._setup(tmp_path, sentiment=-0.2, n_articles=8)
        assert reader.sentiment_multiplier("AAPL") == 0.95

    def test_neutral_band(self, tmp_path: Path):
        reader = self._setup(tmp_path, sentiment=0.05, n_articles=8)
        assert reader.sentiment_multiplier("AAPL") == 1.0

    def test_strong_positive_with_5plus_articles_boost(self, tmp_path: Path):
        reader = self._setup(tmp_path, sentiment=0.5, n_articles=8)
        assert reader.sentiment_multiplier("AAPL") == 1.05

    def test_multiplier_bounds_observed(self, tmp_path: Path):
        # Across all four bands the multiplier should stay in [0.88, 1.05]
        for sent, n in [(-0.9, 10), (-0.2, 10), (0.0, 10), (0.5, 10)]:
            reader = self._setup(tmp_path, sentiment=sent, n_articles=n)
            m = reader.sentiment_multiplier("AAPL")
            assert 0.88 <= m <= 1.05


class TestCacheBehavior:
    def test_cache_serves_within_ttl(self, tmp_path: Path):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        path = _write_sentiment_csv(tmp_path, [
            {
                "ticker": "AAPL",
                "as_of": now - timedelta(hours=1),
                "sentiment": 0.3,
                "confidence": 0.7,
                "n_articles": 6,
            }
        ])
        reader = NewsSentimentReader(base_dir=tmp_path)
        result1 = reader.get_ticker_sentiment("AAPL")
        # Mutate the file underneath
        path.write_text("ticker,as_of,sentiment,confidence,n_articles\n")
        # Cached result still served (TTL 5 min)
        result2 = reader.get_ticker_sentiment("AAPL")
        assert result1["sentiment"] == result2["sentiment"]
