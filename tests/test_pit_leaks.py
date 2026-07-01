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
        """Post-D18: ``sentiment_multiplier`` is severed from the EV path
        and returns 1.0 unconditionally. PIT correctness is preserved
        trivially — a constant cannot leak future news.

        The real PIT regression check on the news store lives in
        ``test_historical_as_of_excludes_future_news`` above (which
        exercises ``get_ticker_sentiment``, the surface that still
        consumes ``as_of``).
        """
        r = _news_reader(tmp_path, _news_rows())
        assert r.sentiment_multiplier("TEST", as_of=_HIST) == 1.0
        assert r.sentiment_multiplier("TEST") == 1.0

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


def _runner_for_single_ranker(last_date_iso: str):
    """Shared synthetic-connector helper for the CC + strangle ranker
    PIT-gate tests. Returns a WheelRunner whose connector emits OHLCV
    ending at `last_date_iso`."""
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


# ── live (as_of=None) spot staleness (D-2 closer, 2026-06-15) ───────
class TestLiveSpotStaleness:
    """The as_of-provided staleness gate (TestRankerAsOfBeyondData) does NOT
    fire when as_of is None — so on the live path the engine silently priced
    off the latest close as 'today's' spot even when that close was months old
    (stale cache / missed refresh): the same D11 'no silent substitution'
    violation, but live. The fix surfaces spot_date on every row, warns once,
    and (opt-in) refuses. Uses a synthetic connector ending at a FIXED old date
    so the test is robust to data refreshes."""

    _STALE = "2020-01-01"  # always > 30 days before any realistic test clock

    def _kw(self, **extra):
        kw = {
            "tickers": ["TEST"],
            "as_of": None,
            "top_n": 10,
            "min_ev_dollars": -1e9,
            "use_dealer_positioning": False,
            "use_news_sentiment": False,
            "use_credit_regime": False,
            "use_skew_dynamics": False,
            "use_event_gate": False,
        }
        kw.update(extra)
        return kw

    def test_stale_live_spot_refused_when_opted_in(self):
        r = _runner_for_single_ranker(self._STALE)
        df = r.rank_candidates_by_ev(**self._kw(refuse_stale_live=True))
        assert df.empty, "stale live spot must be refused when refuse_stale_live=True"
        drops = df.attrs.get("drops", [])
        assert len(drops) == 1
        assert drops[0]["gate"] == "data"
        assert "stale" in drops[0]["reason"]
        assert self._STALE in drops[0]["reason"]  # surfaces the real spot date

    def test_stale_live_spot_warns_but_ranks_by_default(self, caplog):
        import logging

        r = _runner_for_single_ranker(self._STALE)
        with caplog.at_level(logging.WARNING, logger="engine.wheel_runner"):
            df = r.rank_candidates_by_ev(**self._kw())
        # Default path does NOT refuse — backtests run as_of=None against
        # committed (back-dated) data and must keep ranking.
        for d in df.attrs.get("drops", []):
            assert "refuse_stale_live" not in d["reason"]
        # ...but it is no longer SILENT: a warning fired and spot_date is exposed.
        assert any("live spot is" in rec.message for rec in caplog.records)
        if not df.empty:
            assert df["spot_date"].iloc[0] == self._STALE

    def test_fresh_live_spot_not_flagged(self):
        from datetime import date

        r = _runner_for_single_ranker(date.today().isoformat())
        df = r.rank_candidates_by_ev(**self._kw(refuse_stale_live=True))
        # Fresh data: the staleness gate must NOT fire (no over-refusal).
        for d in df.attrs.get("drops", []):
            assert "live spot is" not in d["reason"]

    def test_spot_date_provenance_on_as_of_path_unchanged(self):
        """An explicit (historical) as_of is unaffected by the live gate and
        still carries spot_date = the last bar <= as_of."""
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_candidates_by_ev(**self._kw(as_of="2025-06-15"))
        for d in df.attrs.get("drops", []):
            assert "live spot is" not in d["reason"]  # live gate dormant when as_of set
        if not df.empty:
            assert df["spot_date"].iloc[0] <= "2025-06-15"


class TestCoveredCallRankerAsOfBeyondData:
    """S33 F3 follow-up: rank_covered_calls_by_ev had the same
    silent-substitution surface as rank_candidates_by_ev. This
    pins the equivalent fix."""

    def test_far_future_as_of_drops_with_explicit_reason(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_covered_calls_by_ev(
            ticker="TEST",
            as_of="2030-01-01",
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
        )
        assert df.empty, "CC ranker must NOT silently substitute future as_of"
        drops = df.attrs.get("drops", [])
        assert any("beyond latest data" in d.get("reason", "") for d in drops)
        assert any("2030-01-01" in d.get("reason", "") for d in drops)

    def test_within_threshold_as_of_proceeds(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_covered_calls_by_ev(
            ticker="TEST",
            as_of="2026-04-10",  # 21 days, within 30
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
        )
        for d in df.attrs.get("drops", []):
            assert "beyond latest data" not in d.get("reason", "")

    def test_custom_threshold_overrides_default(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_covered_calls_by_ev(
            ticker="TEST",
            as_of="2026-04-05",  # 16 days
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
            max_as_of_staleness_days=7,  # tight
        )
        assert df.empty
        drops = df.attrs.get("drops", [])
        assert any("beyond latest data" in d.get("reason", "") for d in drops)

    def test_historical_as_of_unaffected(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_covered_calls_by_ev(
            ticker="TEST",
            as_of="2025-06-15",  # well within history
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
        )
        for d in df.attrs.get("drops", []):
            assert "beyond latest data" not in d.get("reason", "")


class TestStrangleRankerAsOfBeyondData:
    """S33 F3 follow-up: rank_strangles_by_ev had the same
    silent-substitution surface. This pins the equivalent fix."""

    def test_far_future_as_of_drops_with_explicit_reason(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_strangles_by_ev(
            ticker="TEST",
            as_of="2030-01-01",
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
            use_timing_gate=False,
        )
        assert df.empty, "strangle ranker must NOT silently substitute future as_of"
        drops = df.attrs.get("drops", [])
        assert any("beyond latest data" in d.get("reason", "") for d in drops)
        assert any("2030-01-01" in d.get("reason", "") for d in drops)

    def test_within_threshold_as_of_proceeds(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_strangles_by_ev(
            ticker="TEST",
            as_of="2026-04-10",  # 21 days, within 30
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
            use_timing_gate=False,
        )
        for d in df.attrs.get("drops", []):
            assert "beyond latest data" not in d.get("reason", "")

    def test_custom_threshold_overrides_default(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_strangles_by_ev(
            ticker="TEST",
            as_of="2026-04-05",  # 16 days
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
            use_timing_gate=False,
            max_as_of_staleness_days=7,
        )
        assert df.empty
        drops = df.attrs.get("drops", [])
        assert any("beyond latest data" in d.get("reason", "") for d in drops)

    def test_historical_as_of_unaffected(self):
        r = _runner_for_single_ranker("2026-03-20")
        df = r.rank_strangles_by_ev(
            ticker="TEST",
            as_of="2025-06-15",
            top_n=10,
            min_ev_dollars=-1e9,
            use_event_gate=False,
            use_timing_gate=False,
        )
        for d in df.attrs.get("drops", []):
            assert "beyond latest data" not in d.get("reason", "")


class TestAnalyzeTickerAsOfGate:
    """S33 audit holdover: WheelRunner.analyze_ticker used to ignore
    `as_of` entirely for the spot-price computation -- it called
    conn.get_ohlcv(ticker) without filtering and took the latest
    close via `.iloc[-1]`. A caller passing a historical or future
    as_of got today's spot, silently. This pins the fix:
    - PIT filter: spot_price respects as_of (uses latest close <= as_of).
    - Staleness gate: if as_of is more than max_as_of_staleness_days
      beyond the latest available row, spot_price stays at its 0.0
      default (the existing "no data available" signal). Caller
      sees a logged warning."""

    def _runner_with_data_through(self, last_date_iso: str):
        """Build a WheelRunner with synthetic OHLCV ending at
        last_date_iso. Mirrors the helper used by the ranker tests."""
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
                return {
                    "market_cap": 1e9,
                    "pe_ratio": 20.0,
                    "beta": 1.0,
                    "dividend_yield": 0.01,
                    "sector": "Test",
                    "implied_vol_atm": 0.28,
                    "volatility_30d": 0.25,
                }

            def get_credit_risk(self, ticker):
                return None

            def get_iv_rank(self, ticker, as_of=None):
                return 50.0

            def get_iv_percentile(self, ticker, as_of=None):
                return 50.0

            def get_vol_risk_premium(self, ticker, as_of=None):
                return 0.0

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_next_dividend(self, ticker, as_of=None):
                return None

            def get_risk_free_rate(self, as_of=None):
                return 0.04

        r = WheelRunner()
        r._connector = _Conn()
        return r

    def test_as_of_none_uses_latest_close(self):
        """as_of=None preserves live behavior: latest close from OHLCV."""
        r = self._runner_with_data_through("2026-03-20")
        ohlcv = r._connector.get_ohlcv("TEST")
        expected_spot = float(ohlcv["close"].iloc[-1])
        a = r.analyze_ticker("TEST")
        assert a.spot_price == pytest.approx(expected_spot)

    def test_historical_as_of_respects_pit(self):
        """A historical as_of returns the spot from THAT date, not today."""
        import pandas as pd

        r = self._runner_with_data_through("2026-03-20")
        ohlcv = r._connector.get_ohlcv("TEST")
        # Spot at the 2024-06-01 PIT cutoff
        target_date = pd.Timestamp("2024-06-01")
        pit_ohlcv = ohlcv.loc[ohlcv.index <= target_date]
        expected_spot = float(pit_ohlcv["close"].iloc[-1])
        a = r.analyze_ticker("TEST", as_of="2024-06-01")
        assert a.spot_price == pytest.approx(expected_spot)
        # And the live close is different (drift over ~21 months)
        live_spot = float(ohlcv["close"].iloc[-1])
        assert abs(a.spot_price - live_spot) > 0.01, (
            "PIT spot must differ from live spot for a meaningful test"
        )

    def test_far_future_as_of_leaves_spot_at_default(self):
        """as_of beyond data by > max_as_of_staleness_days: spot_price
        stays at the 0.0 default (no silent substitution)."""
        r = self._runner_with_data_through("2026-03-20")
        a = r.analyze_ticker("TEST", as_of="2030-01-01")
        assert a.spot_price == 0.0, (
            "spot_price must NOT silently substitute when as_of is years "
            "beyond data; should stay at default 0.0"
        )

    def test_within_threshold_as_of_proceeds(self):
        """A near-future as_of (within the 30-day default) proceeds
        normally, using the latest available close as 'current'."""
        r = self._runner_with_data_through("2026-03-20")
        ohlcv = r._connector.get_ohlcv("TEST")
        expected_spot = float(ohlcv["close"].iloc[-1])
        a = r.analyze_ticker("TEST", as_of="2026-04-10")  # 21 days, within 30
        assert a.spot_price == pytest.approx(expected_spot)

    def test_custom_threshold_tightens(self):
        """A tighter max_as_of_staleness_days catches shorter gaps."""
        r = self._runner_with_data_through("2026-03-20")
        a = r.analyze_ticker("TEST", as_of="2026-04-05", max_as_of_staleness_days=7)
        # 16-day gap > 7-day threshold -> spot stays at default 0.0
        assert a.spot_price == 0.0
