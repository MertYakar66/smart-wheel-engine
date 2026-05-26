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


# ── ohlcv future-as_of silent substitution (S32 F3 closer) ──────────
class TestRankerAsOfBeyondData:
    """S32 F3: querying rank_candidates_by_ev with a future as_of
    well beyond the data cutoff used to silently substitute the
    latest available close as 'current spot' with no warning.
    This violates D11 'no silent substitution'. The fix gates such
    queries via max_as_of_staleness_days (default 30); over-the-
    threshold queries drop the ticker with reason='as_of_beyond_data'."""

    def _runner_with_data_through(self, last_date_iso: str):
        """Build a WheelRunner whose connector returns OHLCV ending at
        last_date_iso. Synthetic; no on-disk dependency."""
        import numpy as np
        import pandas as pd

        from engine.wheel_runner import WheelRunner

        end = pd.Timestamp(last_date_iso)
        idx = pd.date_range(end - pd.Timedelta(days=800), end, freq="B")
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
        oh = pd.DataFrame({"close": prices}, index=idx)

        class _Conn:
            def get_ohlcv(self, ticker):
                return oh

            def get_fundamentals(self, ticker):
                return {"implied_vol_atm": 0.28, "volatility_30d": 0.25, "dividend_yield": 0.01}

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["TEST"]

        r = WheelRunner()
        r._connector = _Conn()
        return r

    def test_far_future_as_of_drops_with_as_of_beyond_data_reason(self):
        """as_of well past data cutoff -> drop with explicit reason."""
        r = self._runner_with_data_through("2026-03-20")
        df = r.rank_candidates_by_ev(
            tickers=["TEST"],
            as_of="2030-01-01",
            top_n=10,
            min_ev_dollars=-1e9,
            use_dealer_positioning=False,
            use_news_sentiment=False,
            use_credit_regime=False,
            use_skew_dynamics=False,
        )
        assert df.empty, "engine must NOT silently substitute future as_of"
        drops = df.attrs.get("drops", [])
        assert len(drops) == 1
        d = drops[0]
        assert d["gate"] == "data"
        assert "beyond latest data" in d["reason"]
        assert "2030-01-01" in d["reason"]

    def test_within_threshold_as_of_proceeds_normally(self):
        """as_of within the default 30-day threshold proceeds normally."""
        r = self._runner_with_data_through("2026-03-20")
        df = r.rank_candidates_by_ev(
            tickers=["TEST"],
            as_of="2026-04-10",  # 21 days later, within 30
            top_n=10,
            min_ev_dollars=-1e9,
            use_dealer_positioning=False,
            use_news_sentiment=False,
            use_credit_regime=False,
            use_skew_dynamics=False,
        )
        # Drop is allowed for non-data reasons (insufficient history,
        # ev_threshold, etc.), but must NOT be an 'as_of_beyond_data'
        # rejection.
        drops = df.attrs.get("drops", [])
        for d in drops:
            assert "beyond latest data" not in d.get("reason", ""), (
                f"21-day gap should not trip the as_of_beyond_data gate: {d}"
            )

    def test_custom_threshold_overrides_default(self):
        """A caller can tighten the threshold (e.g. 0 days = strict)."""
        r = self._runner_with_data_through("2026-03-20")
        df = r.rank_candidates_by_ev(
            tickers=["TEST"],
            as_of="2026-04-05",  # 16 days later
            top_n=10,
            min_ev_dollars=-1e9,
            max_as_of_staleness_days=7,  # tight: 16 > 7 -> drop
            use_dealer_positioning=False,
            use_news_sentiment=False,
            use_credit_regime=False,
            use_skew_dynamics=False,
        )
        assert df.empty
        drops = df.attrs.get("drops", [])
        assert any("beyond latest data" in d["reason"] for d in drops)

    def test_historical_as_of_unaffected(self):
        """as_of in the PAST (well-covered by data) is unaffected by
        the gate."""
        r = self._runner_with_data_through("2026-03-20")
        df = r.rank_candidates_by_ev(
            tickers=["TEST"],
            as_of="2025-06-15",  # well within history
            top_n=10,
            min_ev_dollars=-1e9,
            use_dealer_positioning=False,
            use_news_sentiment=False,
            use_credit_regime=False,
            use_skew_dynamics=False,
        )
        drops = df.attrs.get("drops", [])
        for d in drops:
            assert "beyond latest data" not in d.get("reason", "")
