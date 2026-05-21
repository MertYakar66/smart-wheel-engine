"""Regression tests for the point-in-time (PIT) data leaks.

Usage tests S10 (news) and S11 (credit) found that the news and
credit-regime overlays silently read *today's* data when given a
historical ``as_of`` — look-ahead bias that quietly invalidates
backtests. These tests pin the fix:

- a historical ``as_of`` must never surface data timestamped after it;
- ``as_of=None`` must preserve live (wall-clock ``now()``) behaviour.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from engine.external_data.fred_adapter import FREDAdapter
from engine.news_sentiment import NewsSentimentReader

_HIST = "2025-04-07T12:00:00"  # a historical as_of, ~13 months before "now"


# ── news ───────────────────────────────────────────────────────────
def _news_reader(tmp_path, rows):
    """A NewsSentimentReader over a synthetic CSV store."""
    store = tmp_path / "data_processed"
    store.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(store / "news_sentiment.csv", index=False)
    return NewsSentimentReader(base_dir=tmp_path)


def _news_rows():
    # Row A: inside the PIT window for _HIST. Row B: dated now (after _HIST).
    # Both timestamps use identical second-precision ISO format so
    # pd.to_datetime parses them consistently.
    now_iso = datetime.now().replace(microsecond=0).isoformat()
    return [
        {
            "ticker": "TEST",
            "as_of": "2025-04-07T06:00:00",
            "sentiment": 0.40,
            "confidence": 0.9,
            "n_articles": 12,
        },
        {
            "ticker": "TEST",
            "as_of": now_iso,
            "sentiment": -0.90,
            "confidence": 0.9,
            "n_articles": 12,
        },
    ]


class TestNewsPIT:
    def test_historical_as_of_excludes_future_news(self, tmp_path):
        """A historical as_of returns the news at/before it — not the
        fresh post-as_of row."""
        r = _news_reader(tmp_path, _news_rows())
        s = r.get_ticker_sentiment("TEST", as_of=_HIST)
        assert s["sentiment"] == pytest.approx(0.40)  # Row A, not Row B
        assert pd.Timestamp(s["as_of"]) <= pd.Timestamp(_HIST)

    def test_multiplier_is_pit(self, tmp_path):
        """sentiment_multiplier honours as_of: bullish Row A (+0.40) at the
        historical as_of -> 1.05; bearish now-row (-0.90) live -> 0.88."""
        r = _news_reader(tmp_path, _news_rows())
        assert r.sentiment_multiplier("TEST", as_of=_HIST) == pytest.approx(1.05)
        assert r.sentiment_multiplier("TEST") == pytest.approx(0.88)

    def test_as_of_none_preserves_now_behaviour(self, tmp_path):
        """as_of=None keeps the live wall-clock window -> the fresh row."""
        r = _news_reader(tmp_path, _news_rows())
        s = r.get_ticker_sentiment("TEST")
        assert s["sentiment"] == pytest.approx(-0.90)  # Row B (now)

    def test_no_as_of_column_refuses_historical(self, tmp_path):
        """A store with no per-row as_of cannot support PIT — a historical
        as_of must decline (neutral), not silently return the latest row."""
        rows = [{"ticker": "TEST", "sentiment": -0.9, "confidence": 0.9, "n_articles": 12}]
        r = _news_reader(tmp_path, rows)
        s = r.get_ticker_sentiment("TEST", as_of=_HIST)
        assert s["n_articles"] == 0  # declined -> neutral, not the -0.9 row


# ── credit ─────────────────────────────────────────────────────────
def _synthetic_oas():
    """HY OAS: flat 3.0 baseline with an April-2025 spike to 9.0."""
    idx = pd.date_range("2021-01-01", "2026-05-01", freq="D")
    s = pd.Series(3.0, index=idx)
    s[(idx >= "2025-04-01") & (idx <= "2025-04-30")] = 9.0
    return s


class TestCreditPIT:
    def test_historical_as_of_sees_historical_regime(self, monkeypatch):
        """credit_regime(as_of=April-2025) sees the April-2025 OAS spike,
        not today's calm value."""
        oas = _synthetic_oas()
        monkeypatch.setattr(FREDAdapter, "get_series", lambda self, sid: oas)
        cr = FREDAdapter().credit_regime(as_of="2025-04-15")
        assert cr["hy_oas"] == pytest.approx(9.0)  # the spike
        assert cr["regime"] in ("stressed", "crisis")

    def test_as_of_none_uses_latest(self, monkeypatch):
        """as_of=None preserves live behaviour — the latest observation."""
        oas = _synthetic_oas()
        monkeypatch.setattr(FREDAdapter, "get_series", lambda self, sid: oas)
        cr = FREDAdapter().credit_regime()
        assert cr["hy_oas"] == pytest.approx(3.0)  # latest, calm
        assert cr["regime"] == "benign"
