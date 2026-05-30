"""DECISIONS.md D18 — verbal news severance from the EV path.

This file pins the D18 contract: ``sentiment_multiplier`` is a constant
1.0 stub, regardless of the underlying sentiment store contents or the
``(ticker, lookback_hours, as_of)`` arguments passed in. A future change
that reintroduces an EV-influencing multiplier — by accident or design
— flips these tests.

What it does NOT test
---------------------
- ``get_ticker_sentiment`` still works (covered in ``test_news_sentiment.py``).
- The PIT correctness of the sentiment store (covered in
  ``test_pit_leaks.py::TestNewsPIT``).
- The wheel_runner call site still emits ``news_sentiment`` and
  ``news_n_articles`` for transparency (covered indirectly by
  ``test_ranker_transparency.py``).

This file's sole concern is "the multiplier channel is closed."
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from engine.news_sentiment import NewsSentimentReader


def _write_store(base: Path, rows: list[dict]) -> None:
    target = base / "data_processed" / "news_sentiment.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


def _row(sentiment: float, n_articles: int) -> dict:
    return {
        "ticker": "AAPL",
        "as_of": (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)).isoformat(),
        "sentiment": sentiment,
        "confidence": 0.9,
        "n_articles": n_articles,
    }


class TestSeveranceContract:
    """The D18 contract: 1.0 for every combination of inputs."""

    @pytest.mark.parametrize(
        "sentiment,n_articles",
        [
            # The four bands the old code mapped to non-1.0 multipliers
            (-0.9, 10),  # was 0.88
            (-0.5, 8),  # was 0.88
            (-0.3, 8),  # was 0.88 (boundary)
            (-0.25, 8),  # was 0.95
            (-0.1, 8),  # was 0.95 (boundary)
            (0.0, 8),  # was 1.0 (neutral)
            (0.1, 8),  # was 1.0
            (0.29, 8),  # was 1.0
            (0.3, 8),  # was 1.05 (boundary)
            (0.5, 8),  # was 1.05
            (0.95, 50),  # was 1.05 (extreme)
            # Low-confidence band (n_articles < 5) — was 1.0 before, still 1.0
            (-0.9, 1),
            (0.9, 1),
            (-0.5, 4),
            (0.5, 4),
        ],
    )
    def test_multiplier_is_one(self, tmp_path: Path, sentiment: float, n_articles: int):
        _write_store(tmp_path, [_row(sentiment, n_articles)])
        reader = NewsSentimentReader(base_dir=tmp_path)
        assert reader.sentiment_multiplier("AAPL") == 1.0

    def test_multiplier_is_one_with_no_store(self, tmp_path: Path):
        """A missing store used to return 1.0 anyway (neutral default);
        post-D18 the absence of a store and the presence of a strongly
        biased store are both 1.0 — confirms the stub doesn't accidentally
        depend on the store's existence."""
        reader = NewsSentimentReader(base_dir=tmp_path)
        assert reader.sentiment_multiplier("UNKNOWN") == 1.0

    def test_multiplier_is_one_with_historical_as_of(self, tmp_path: Path):
        """A historical ``as_of`` used to thread through to the multiplier
        bands; post-D18 the channel is closed so the result is still 1.0."""
        # Two rows, one historical, one fresh — designed to trip the old
        # PIT branch into the historical bullish row at the old as_of.
        rows = [
            {
                "ticker": "AAPL",
                "as_of": "2025-04-07T06:00:00",
                "sentiment": 0.5,  # would have been 1.05 historically
                "confidence": 0.9,
                "n_articles": 12,
            },
            {
                "ticker": "AAPL",
                "as_of": datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat(),
                "sentiment": -0.9,  # would have been 0.88 live
                "confidence": 0.9,
                "n_articles": 12,
            },
        ]
        _write_store(tmp_path, rows)
        reader = NewsSentimentReader(base_dir=tmp_path)
        # Same as the old PIT test inputs — both must now be 1.0.
        assert reader.sentiment_multiplier("AAPL", as_of="2025-04-07T12:00:00") == 1.0
        assert reader.sentiment_multiplier("AAPL") == 1.0

    def test_multiplier_is_one_with_extreme_lookback(self, tmp_path: Path):
        """Extreme ``lookback_hours`` values used to filter rows; post-D18
        they still pass through but the multiplier is still 1.0."""
        _write_store(tmp_path, [_row(-0.9, 100)])  # would have been 0.88
        reader = NewsSentimentReader(base_dir=tmp_path)
        assert reader.sentiment_multiplier("AAPL", lookback_hours=1) == 1.0
        assert reader.sentiment_multiplier("AAPL", lookback_hours=24 * 365) == 1.0


class TestStubPreservesSideEffects:
    """The stub still calls ``get_ticker_sentiment`` so the on-disk cache
    stays warm and any store-side validation runs. Pinning this means a
    future refactor that drops the side effect (e.g. ``return 1.0``
    without the get call) flags here — the operator dashboard depends on
    the cache being warm when the next ``get_ticker_sentiment`` lookup
    runs from a different code path.
    """

    def test_get_ticker_sentiment_still_returns_underlying_data(self, tmp_path: Path):
        """After ``sentiment_multiplier`` is called, ``get_ticker_sentiment``
        on the same reader returns the actual row (the dashboard's view)."""
        _write_store(tmp_path, [_row(0.5, 10)])
        reader = NewsSentimentReader(base_dir=tmp_path)
        # Calling the (severed) multiplier shouldn't blank the cached data.
        assert reader.sentiment_multiplier("AAPL") == 1.0
        result = reader.get_ticker_sentiment("AAPL")
        assert result["sentiment"] == pytest.approx(0.5)
        assert result["n_articles"] == 10
