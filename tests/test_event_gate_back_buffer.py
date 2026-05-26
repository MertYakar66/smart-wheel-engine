"""Tests for the S23 F1 symmetric event-gate fix.

The :class:`engine.event_gate.EventGate` has always done symmetric
``_event_touches_window`` arithmetic (``window_start = trade_start -
timedelta(days=buf)``) and surfaces an explicit ``±{buf}d buffer``
reason string. But until this fix, the three rankers in
:mod:`engine.wheel_runner` only registered the **next future**
earnings on the gate — via
:meth:`engine.data_connector.MarketDataConnector.get_next_earnings`,
which strictly filters ``announcement_date > ref``. The result was
that a trade opened immediately *after* an earnings event was
silently allowed, even though the gate's docstring explicitly cites
"IV-crush dynamics" — a post-earnings phenomenon — as motivation.

The fix:

  * Adds :meth:`MarketDataConnector.get_recent_earnings(ticker, as_of,
    lookback_days)`, the symmetric complement of
    :meth:`get_next_earnings`, returning the most recent earnings in
    ``[as_of - lookback_days, as_of]``.
  * Updates the three rankers in :mod:`engine.wheel_runner` to also
    pull the recent past earnings and register it on the
    ``EventGate``. Defensive ``hasattr(conn, "get_recent_earnings")``
    so connectors / test stubs without the new method keep working.
  * Adds a symmetric soft-fallback in :meth:`rank_candidates_by_ev`
    for the ``use_event_gate=False`` path.

Pinned here:

  1. ``MarketDataConnector.get_recent_earnings`` — within lookback,
     past-only, most-recent row; complementary cutoff
     (``> ref`` vs ``<= ref``).
  2. End-to-end: a trade opened ``N`` days after a real earnings
     (``N < earnings_buffer_days``) is blocked by the symmetric gate
     and shows up in ``df.attrs["drops"]`` with the gate's
     ``±{buf}d buffer`` reason format.
  3. Backward compatibility: a connector without
     ``get_recent_earnings`` continues to work; the rest of the
     ranker path is unaffected.
  4. Soft fallback symmetry: with ``use_event_gate=False``, recent
     past earnings also produce a soft-skip drop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector
from engine.wheel_runner import WheelRunner

_OFFLINE = {
    "use_dealer_positioning": False,
    "use_news_sentiment": False,
    "use_credit_regime": False,
    "use_skew_dynamics": False,
}

_TICKERS = ["AAA", "BBB"]


# ----------------------------------------------------------------------
# 1. MarketDataConnector.get_recent_earnings — unit-level
# ----------------------------------------------------------------------
@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A minimal Bloomberg-shaped data dir with an earnings CSV."""
    rows = [
        # AAA: a past earnings at 2026-03-15 and a future one at 2026-06-15
        {"ticker": "AAA", "announcement_date": "2026-03-15", "year/period": "2026 Q1"},
        {"ticker": "AAA", "announcement_date": "2026-06-15", "year/period": "2026 Q2"},
        # BBB: a past one further back, plus a future
        {"ticker": "BBB", "announcement_date": "2026-02-01", "year/period": "2026 Q1"},
        {"ticker": "BBB", "announcement_date": "2026-05-01", "year/period": "2026 Q2"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "sp500_earnings.csv", index=False)
    return tmp_path


class TestGetRecentEarnings:
    def test_returns_most_recent_within_lookback(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        recent = conn.get_recent_earnings("AAA", as_of="2026-03-18", lookback_days=7)
        assert recent is not None
        assert recent["announcement_date"] == pd.Timestamp("2026-03-15")
        assert recent["year_period"] == "2026 Q1"

    def test_returns_none_when_no_past_in_window(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        # 2026-03-22 with a 5-day lookback can't reach back to 2026-03-15
        assert conn.get_recent_earnings("AAA", as_of="2026-03-22", lookback_days=5) is None

    def test_returns_none_when_all_future(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_recent_earnings("AAA", as_of="2024-01-01", lookback_days=30) is None

    def test_returns_none_when_no_file(self, tmp_path):
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_recent_earnings("AAA", as_of="2026-03-18") is None

    def test_returns_none_for_unknown_ticker(self, data_dir):
        conn = MarketDataConnector(data_dir=str(data_dir))
        assert conn.get_recent_earnings("NOSUCH", as_of="2026-03-18") is None

    def test_event_on_as_of_is_returned_by_recent_not_next(self, data_dir):
        """Cutoff convention: an event ON ``as_of`` is treated as
        past (returned by get_recent_earnings, not by get_next_earnings).
        The two methods are complementary and never overlap."""
        conn = MarketDataConnector(data_dir=str(data_dir))
        recent = conn.get_recent_earnings("AAA", as_of="2026-03-15", lookback_days=7)
        nxt = conn.get_next_earnings("AAA", as_of="2026-03-15")
        assert recent is not None
        assert recent["announcement_date"] == pd.Timestamp("2026-03-15")
        # get_next_earnings filters strictly > ref, so the 2026-03-15
        # event must not be returned by it — only the 2026-06-15 future is.
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-06-15")


# ----------------------------------------------------------------------
# 2. End-to-end through rank_candidates_by_ev — symmetric blocking
# ----------------------------------------------------------------------
class _SymEarningsConn:
    """Multi-ticker stub with both get_next_earnings and
    get_recent_earnings. Earnings dictated by ``recent_earnings`` /
    ``next_earnings`` dicts keyed by ticker → ISO date string."""

    def __init__(
        self,
        tickers,
        *,
        recent_earnings: dict[str, str] | None = None,
        next_earnings: dict[str, str] | None = None,
        default_days: int = 3000,
        snapshot_iv: float = 28.0,
    ) -> None:
        self._tickers = list(tickers)
        self._recent = dict(recent_earnings or {})
        self._next = dict(next_earnings or {})
        self._snapshot_iv = snapshot_iv
        self._ohlcv: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            idx = pd.date_range("2016-01-01", periods=default_days, freq="B")
            rng = np.random.default_rng(100 + i)
            base = 80.0 * (1.0 + 0.45 * i)
            close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.011, default_days)))
            self._ohlcv[t] = pd.DataFrame({"close": close}, index=idx)

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._ohlcv[ticker]

    def get_fundamentals(self, ticker: str) -> dict:
        return {
            "implied_vol_atm": self._snapshot_iv,
            "volatility_30d": self._snapshot_iv,
            "dividend_yield": 0.0,
        }

    def get_risk_free_rate(self, as_of=None) -> float:
        return 0.05

    def get_universe(self) -> list[str]:
        return list(self._tickers)

    def get_next_earnings(self, ticker: str, as_of=None) -> dict | None:
        d = self._next.get(ticker)
        return {"announcement_date": pd.Timestamp(d)} if d is not None else None

    def get_recent_earnings(self, ticker: str, as_of=None, lookback_days=7) -> dict | None:
        d = self._recent.get(ticker)
        if d is None:
            return None
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        ev = pd.Timestamp(d)
        lookback_start = ref - pd.Timedelta(days=int(lookback_days))
        if lookback_start <= ev <= ref:
            return {"announcement_date": ev}
        return None


def _runner(conn) -> WheelRunner:
    r = WheelRunner()
    r._connector = conn
    return r


def _rank(runner, **extra):
    kw = dict(tickers=_TICKERS, top_n=20, min_ev_dollars=-1e9, **_OFFLINE)
    kw.update(extra)
    return runner.rank_candidates_by_ev(**kw)


class TestSymmetricGateBlocksPostEarnings:
    def test_recent_earnings_within_buffer_blocks_trade(self):
        """AAA had earnings 2 days before as_of, buffer is 5 days.
        Symmetric gate must block; pre-fix this was silently allowed."""
        conn = _SymEarningsConn(
            _TICKERS,
            recent_earnings={"AAA": "2026-03-13"},  # 2 days before as_of
        )
        df = _rank(_runner(conn), as_of="2026-03-15", earnings_buffer_days=5)
        # AAA must be in drops with an event reason; BBB must survive.
        dropped_tickers = {d["ticker"] for d in df.attrs["drops"]}
        assert "AAA" in dropped_tickers, (
            f"AAA had earnings 2026-03-13 (2d before as_of=2026-03-15) and "
            f"a 5d buffer — must be blocked. drops={df.attrs['drops']}"
        )
        aaa_drop = [d for d in df.attrs["drops"] if d["ticker"] == "AAA"][0]
        assert aaa_drop["gate"] == "event"
        assert "earnings@2026-03-13" in aaa_drop["reason"]
        # The ± symbol is the explicit signature of the symmetric gate.
        assert "±5d" in aaa_drop["reason"]

    def test_recent_earnings_outside_buffer_does_not_block(self):
        """AAA had earnings 10 days before as_of, buffer is 5 days.
        Outside the back-buffer window → trade is allowed."""
        conn = _SymEarningsConn(
            _TICKERS,
            recent_earnings={"AAA": "2026-03-05"},  # 10 days before
        )
        df = _rank(_runner(conn), as_of="2026-03-15", earnings_buffer_days=5)
        # AAA must NOT be in event drops; lookback was 5d so the stub
        # returns None for the connector call, and the gate gets no
        # earnings to fire on.
        event_drops = [d for d in df.attrs["drops"] if d["gate"] == "event"]
        aaa_event_drops = [d for d in event_drops if d["ticker"] == "AAA"]
        assert not aaa_event_drops, (
            f"AAA earnings 10d ago is outside 5d back-buffer; must not be event-dropped. "
            f"got: {aaa_event_drops}"
        )

    def test_forward_earnings_still_blocks(self):
        """The legacy forward-only behavior must still work."""
        conn = _SymEarningsConn(
            _TICKERS,
            next_earnings={"AAA": "2026-03-18"},  # 3 days after as_of
        )
        df = _rank(_runner(conn), as_of="2026-03-15", earnings_buffer_days=5)
        dropped_tickers = {d["ticker"] for d in df.attrs["drops"]}
        assert "AAA" in dropped_tickers
        aaa_drop = [d for d in df.attrs["drops"] if d["ticker"] == "AAA"][0]
        assert aaa_drop["gate"] == "event"
        assert "earnings@2026-03-18" in aaa_drop["reason"]

    def test_both_directions_register(self):
        """If both a recent past and a near-future earnings exist
        within buffer, both should be on the gate. The gate picks the
        earlier (first-sorted) for the reason string."""
        conn = _SymEarningsConn(
            _TICKERS,
            recent_earnings={"AAA": "2026-03-13"},
            next_earnings={"AAA": "2026-03-17"},
        )
        df = _rank(_runner(conn), as_of="2026-03-15", earnings_buffer_days=5)
        dropped = [d for d in df.attrs["drops"] if d["ticker"] == "AAA"]
        assert dropped, f"AAA must be blocked; drops={df.attrs['drops']}"
        # EventGate.is_blocked sorts hits by date and uses the earliest
        # for the reason string. Past (2026-03-13) is earlier than
        # future (2026-03-17), so the reason cites the past event.
        assert "earnings@2026-03-13" in dropped[0]["reason"]


# ----------------------------------------------------------------------
# 3. Backward compatibility — connectors without get_recent_earnings
# ----------------------------------------------------------------------
class _LegacyForwardOnlyConn:
    """A connector with the legacy interface only — no
    get_recent_earnings. Mirrors ThetaConnector and many test stubs
    that pre-date this fix. The ranker must continue to work."""

    def __init__(self, tickers, *, next_earnings: dict[str, str] | None = None):
        self._tickers = list(tickers)
        self._next = dict(next_earnings or {})
        self._ohlcv: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            idx = pd.date_range("2016-01-01", periods=1400, freq="B")
            rng = np.random.default_rng(100 + i)
            base = 80.0 * (1.0 + 0.45 * i)
            close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.011, 1400)))
            self._ohlcv[t] = pd.DataFrame({"close": close}, index=idx)

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._ohlcv[ticker]

    def get_fundamentals(self, ticker: str) -> dict:
        return {"implied_vol_atm": 28.0, "volatility_30d": 25.0, "dividend_yield": 0.0}

    def get_risk_free_rate(self, as_of=None) -> float:
        return 0.05

    def get_universe(self) -> list[str]:
        return list(self._tickers)

    def get_next_earnings(self, ticker: str, as_of=None) -> dict | None:
        d = self._next.get(ticker)
        return {"announcement_date": pd.Timestamp(d)} if d is not None else None


class TestBackwardCompatibility:
    def test_legacy_connector_without_get_recent_earnings_works(self):
        """Connectors without the new method (~20 stubs + ThetaConnector)
        must continue to work — they fall through the hasattr() guard
        and get the legacy forward-only behavior."""
        conn = _LegacyForwardOnlyConn(_TICKERS, next_earnings={"AAA": "2026-04-15"})
        df = _rank(_runner(conn), as_of="2026-03-15", earnings_buffer_days=5)
        # AAA earnings 31 days out is past the gate's forward buffer
        # for a 35-DTE trade (35 + 5 = 40 day window from today, so
        # 2026-04-15 = 31 days out → inside the window). Should block.
        dropped = [d for d in df.attrs["drops"] if d["ticker"] == "AAA"]
        assert dropped, f"AAA's earnings within window must block; drops={df.attrs['drops']}"
        # Confirm no AttributeError / TypeError leaked from the missing method.
        for d in df.attrs["drops"]:
            assert d["gate"] in (
                "data",
                "history",
                "event",
                "strike",
                "premium",
                "chain_quality",
                "ev_threshold",
            )


# ----------------------------------------------------------------------
# 4. Soft-fallback symmetry (use_event_gate=False)
# ----------------------------------------------------------------------
class TestSoftFallbackSymmetry:
    def test_recent_earnings_blocks_when_event_gate_disabled(self):
        """The soft fallback (use_event_gate=False) historically only
        skipped on forward earnings. After the fix it also skips on
        recent past earnings within the buffer."""
        conn = _SymEarningsConn(
            _TICKERS,
            recent_earnings={"AAA": "2026-03-13"},
        )
        df = _rank(
            _runner(conn),
            as_of="2026-03-15",
            earnings_buffer_days=5,
            use_event_gate=False,
        )
        dropped = [d for d in df.attrs["drops"] if d["ticker"] == "AAA"]
        assert dropped, f"AAA must be soft-dropped; drops={df.attrs['drops']}"
        assert dropped[0]["gate"] == "event"
        assert "ago" in dropped[0]["reason"]
        assert "buffer 5d" in dropped[0]["reason"]

    def test_forward_earnings_still_soft_blocks(self):
        conn = _SymEarningsConn(
            _TICKERS,
            next_earnings={"AAA": "2026-03-18"},
        )
        df = _rank(
            _runner(conn),
            as_of="2026-03-15",
            earnings_buffer_days=5,
            use_event_gate=False,
        )
        dropped = [d for d in df.attrs["drops"] if d["ticker"] == "AAA"]
        assert dropped
        assert "in 3d" in dropped[0]["reason"]
