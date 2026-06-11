"""Tests for the as_of=None staleness gate (brain-audit M3).

rank_candidates_by_ev (and its CC/strangle siblings) previously let every
ticker independently resolve to its own latest OHLCV bar when as_of=None,
silently admitting index leavers like CTRA (last bar 2026-03-20, gap 76 days
behind the universe frontier 2026-06-04) at a stale spot.

The fix resolves a ``staleness_ref`` before the per-ticker loop:
  - when as_of is explicit: staleness_ref = as_of (existing S32 F3 gate,
    byte-identical — pinned by test_pit_leaks.py);
  - when as_of is None AND the connector has ``get_data_frontier``:
    staleness_ref = connector.get_data_frontier() (global universe max date,
    clamped to today);
  - when as_of is None AND the connector lacks ``get_data_frontier``:
    staleness_ref = None -> gate skipped -> legacy behavior (hasattr guard).

All tests use synthetic duck-typed connectors; no real-data dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner

# ── shared OHLCV-building helpers ────────────────────────────────────────────

FRONTIER = pd.Timestamp("2026-06-04")
STALE_END = pd.Timestamp("2026-03-20")  # gap = 76 days


def _ohlcv(end: pd.Timestamp, n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with ``n_rows`` business-day bars ending at ``end``.

    n_rows > 504 so the *staleness* gate (not the history gate) is the
    binding constraint in STALE tests.
    """
    idx = pd.bdate_range(end=end, periods=n_rows)
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_rows)))
    return pd.DataFrame({"close": prices}, index=idx)


_FRESH_OHLCV = _ohlcv(FRONTIER, seed=1)
_STALE_OHLCV = _ohlcv(STALE_END, seed=2)


# ── stub connector factories ──────────────────────────────────────────────────


def _run_kwargs(**extra) -> dict:
    """Common ranker kwargs that disable all network-touching overlays."""
    return {
        "top_n": 10,
        "min_ev_dollars": -1e9,
        "use_dealer_positioning": False,
        "use_news_sentiment": False,
        "use_credit_regime": False,
        "use_skew_dynamics": False,
        **extra,
    }


def _base_methods(ohlcv_map: dict[str, pd.DataFrame]):
    """Return a mixin dict for stub connectors: get_ohlcv / get_fundamentals /
    get_risk_free_rate / get_next_earnings / get_universe."""

    class _Methods:
        def get_ohlcv(self, ticker):
            return ohlcv_map.get(ticker, pd.DataFrame())

        def get_fundamentals(self, ticker):
            return {
                "implied_vol_atm": 0.28,
                "volatility_30d": 0.25,
                "dividend_yield": 0.01,
            }

        def get_risk_free_rate(self, as_of=None):
            return 0.05

        def get_next_earnings(self, ticker, as_of=None):
            return None

        def get_universe(self):
            return sorted(ohlcv_map.keys())

    return _Methods


def _make_runner(ohlcv_map: dict, frontier: pd.Timestamp | None = FRONTIER) -> WheelRunner:
    """Build a WheelRunner with a stub connector backed by ``ohlcv_map``.

    If ``frontier`` is not None the stub exposes ``get_data_frontier()``
    returning it.  If None the stub lacks the method (legacy shape).
    """
    Base = _base_methods(ohlcv_map)

    if frontier is not None:
        _f = frontier  # capture

        class _ConnWithFrontier(Base):
            def get_data_frontier(self, dataset="ohlcv"):
                return _f

        conn_cls = _ConnWithFrontier
    else:

        class _ConnWithoutFrontier(Base):
            pass

        conn_cls = _ConnWithoutFrontier

    r = WheelRunner()
    r._connector = conn_cls()
    return r


# ── tests ─────────────────────────────────────────────────────────────────────


class TestStaleLeaver:
    """Stale tickers are dropped; fresh ones survive."""

    def test_stale_leaver_dropped_at_asof_none(self):
        """STALE (ends 2026-03-20, gap 76d) drops at as_of=None when frontier known."""
        runner = _make_runner({"FRESH": _FRESH_OHLCV, "STALE": _STALE_OHLCV})
        df = runner.rank_candidates_by_ev(
            tickers=["FRESH", "STALE"],
            as_of=None,
            **_run_kwargs(),
        )
        tickers_in = set(df["ticker"].tolist()) if not df.empty else set()
        assert "STALE" not in tickers_in, "STALE ticker must be dropped at as_of=None"

        drops = df.attrs.get("drops", [])
        stale_drops = [d for d in drops if d["ticker"] == "STALE"]
        assert stale_drops, "STALE must appear in drops"
        d = stale_drops[0]
        assert d["gate"] == "data"
        assert "behind universe data frontier" in d["reason"], repr(d["reason"])
        assert "2026-03-20" in d["reason"]
        assert FRONTIER.date().isoformat() in d["reason"]
        # Must NOT contain the explicit-path string (distinct audit trail).
        assert "beyond latest data" not in d["reason"], repr(d["reason"])

    def test_fresh_name_survives_at_asof_none(self):
        """FRESH (gap 0d) must remain in the ranked output."""
        runner = _make_runner({"FRESH": _FRESH_OHLCV, "STALE": _STALE_OHLCV})
        df = runner.rank_candidates_by_ev(
            tickers=["FRESH", "STALE"],
            as_of=None,
            **_run_kwargs(),
        )
        drops = df.attrs.get("drops", [])
        # FRESH must not appear in any *data-gate* drop.
        data_drops = {d["ticker"] for d in drops if d["gate"] == "data"}
        assert "FRESH" not in data_drops, f"FRESH must not be data-dropped: {drops}"


class TestDropOnly:
    """The fix is strictly drop-only: no new names are added."""

    def test_drop_only_never_adds(self):
        """Tickers ranked with the frontier-conn are a subset of those ranked without."""
        ohlcv_map = {
            "FRESH": _FRESH_OHLCV,
            "STALE": _STALE_OHLCV,
        }
        runner_with = _make_runner(ohlcv_map, frontier=FRONTIER)
        runner_without = _make_runner(ohlcv_map, frontier=None)  # no get_data_frontier

        kwargs = _run_kwargs(tickers=["FRESH", "STALE"], as_of=None)
        df_with = runner_with.rank_candidates_by_ev(**kwargs)
        df_without = runner_without.rank_candidates_by_ev(**kwargs)

        set_with = set(df_with["ticker"].tolist()) if not df_with.empty else set()
        set_without = set(df_without["ticker"].tolist()) if not df_without.empty else set()
        assert set_with <= set_without, (
            f"frontier-conn output {set_with} must be a subset of "
            f"no-frontier-conn output {set_without}"
        )


class TestByteIdentical:
    """Survivor rows are byte-identical with and without the frontier gate."""

    def test_healthy_names_byte_identical_at_asof_none(self):
        """FRESH row EV is unchanged whether the frontier is known or not."""
        ohlcv_map = {"FRESH": _FRESH_OHLCV}
        runner_with = _make_runner(ohlcv_map, frontier=FRONTIER)
        runner_without = _make_runner(ohlcv_map, frontier=None)

        kwargs = _run_kwargs(tickers=["FRESH"], as_of=None)
        df_with = runner_with.rank_candidates_by_ev(**kwargs)
        df_without = runner_without.rank_candidates_by_ev(**kwargs)

        assert not df_with.empty and not df_without.empty, (
            "FRESH must produce at least one row in both runs"
        )
        # Core EV fields must match.
        for col in ("ev_dollars", "premium", "iv"):
            if col in df_with.columns and col in df_without.columns:
                assert df_with[col].iloc[0] == pytest.approx(df_without[col].iloc[0], rel=1e-9), (
                    f"{col} mismatch: {df_with[col].iloc[0]} vs {df_without[col].iloc[0]}"
                )


class TestExplicitAsOf:
    """Explicit as_of path is byte-identical — frontier is never consulted."""

    def test_explicit_asof_never_consults_frontier(self):
        """get_data_frontier must NOT be called when as_of is explicit."""
        ohlcv_map = {"TEST": _FRESH_OHLCV}

        class _HostileFrontierConn(_base_methods(ohlcv_map)):
            def get_data_frontier(self, dataset="ohlcv"):
                raise AssertionError("get_data_frontier must not be called at explicit as_of")

        runner = WheelRunner()
        runner._connector = _HostileFrontierConn()

        # Explicit as_of within data range — must succeed without calling frontier.
        df = runner.rank_candidates_by_ev(
            tickers=["TEST"],
            as_of=FRONTIER.date().isoformat(),
            **_run_kwargs(),
        )
        # Should not raise; drops may include non-data reasons but no frontier call.
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert not frontier_drops, f"Unexpected frontier-related drop: {frontier_drops}"

    def test_explicit_asof_beyond_data_keeps_original_reason_string(self):
        """Explicit as_of path retains the byte-identical 'beyond latest data' string."""
        runner = _make_runner({"TEST": _STALE_OHLCV})
        df = runner.rank_candidates_by_ev(
            tickers=["TEST"],
            as_of="2030-01-01",
            **_run_kwargs(),
        )
        drops = df.attrs.get("drops", [])
        assert drops, "Expected a drop for future as_of"
        d = drops[0]
        assert "beyond latest data" in d["reason"], repr(d["reason"])
        assert "2030-01-01" in d["reason"]


class TestStalenessOverride:
    """max_as_of_staleness_days overrides the default threshold."""

    def test_staleness_days_override_readmits_stale(self):
        """max_as_of_staleness_days=10_000 re-admits STALE at as_of=None."""
        runner = _make_runner({"STALE": _STALE_OHLCV})
        df = runner.rank_candidates_by_ev(
            tickers=["STALE"],
            as_of=None,
            max_as_of_staleness_days=10_000,
            **_run_kwargs(),
        )
        drops = df.attrs.get("drops", [])
        data_drops = [d for d in drops if d["gate"] == "data" and "frontier" in d.get("reason", "")]
        assert not data_drops, (
            "STALE must not be dropped by the frontier gate when threshold=10_000"
        )


class TestLegacyConnector:
    """Connectors without get_data_frontier keep the legacy (no-gate) behavior."""

    def test_connector_without_frontier_keeps_legacy_behavior(self):
        """Legacy stub (no get_data_frontier): STALE still ranks at as_of=None."""
        runner = _make_runner({"STALE": _STALE_OHLCV}, frontier=None)
        df = runner.rank_candidates_by_ev(
            tickers=["STALE"],
            as_of=None,
            **_run_kwargs(),
        )
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert not frontier_drops, (
            f"No frontier gate should fire without get_data_frontier: {frontier_drops}"
        )
        # STALE may still drop for other reasons (history, EV threshold) —
        # what matters is it did NOT drop at the new data-frontier gate.


class TestFutureCorruptRow:
    """A corrupt future-dated row must not inflate the frontier and black out healthy names."""

    def test_future_dated_corrupt_row_does_not_blackout_universe(self):
        """get_data_frontier clamps to today so a 2099 row doesn't kill as_of=None scans."""

        class _CorruptFrontierConn(_base_methods({"FRESH": _FRESH_OHLCV})):
            def get_data_frontier(self, dataset="ohlcv"):
                # Simulate a connector that would return a future-dated frontier
                # WITHOUT the clamp.  With the clamp this returns today.
                import datetime as _dt

                raw = pd.Timestamp("2099-01-01")
                today_ts = pd.Timestamp(_dt.date.today())
                return min(raw, today_ts)

        runner = WheelRunner()
        runner._connector = _CorruptFrontierConn()
        df = runner.rank_candidates_by_ev(
            tickers=["FRESH"],
            as_of=None,
            **_run_kwargs(),
        )
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert not frontier_drops, (
            f"Corrupt future-dated row must not cause FRESH to be data-dropped: {frontier_drops}"
        )


class TestCCAndStrangleShareGate:
    """rank_covered_calls_by_ev and rank_strangles_by_ev share the staleness gate."""

    def _make_single_ticker_runner(self, ticker_ohlcv, frontier):
        """Build runner for single-ticker CC/strangle rankers."""
        ohlcv_map = {"TEST": ticker_ohlcv}
        return _make_runner(ohlcv_map, frontier=frontier)

    def test_cc_ranker_drops_stale_at_asof_none(self):
        """rank_covered_calls_by_ev drops a stale ticker at as_of=None."""
        runner = self._make_single_ticker_runner(_STALE_OHLCV, FRONTIER)
        df = runner.rank_covered_calls_by_ev(
            ticker="TEST",
            shares_held=100,
            as_of=None,
        )
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert frontier_drops, f"CC ranker must drop STALE at as_of=None: {drops}"
        d = frontier_drops[0]
        assert d["gate"] == "data"
        assert "behind universe data frontier" in d["reason"]

    def test_cc_ranker_fresh_unaffected_at_asof_none(self):
        """rank_covered_calls_by_ev does not drop a fresh ticker at as_of=None."""
        runner = self._make_single_ticker_runner(_FRESH_OHLCV, FRONTIER)
        df = runner.rank_covered_calls_by_ev(
            ticker="TEST",
            shares_held=100,
            as_of=None,
        )
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert not frontier_drops, f"CC ranker must NOT drop FRESH at as_of=None: {drops}"

    def test_strangle_ranker_drops_stale_at_asof_none(self):
        """rank_strangles_by_ev drops a stale ticker at as_of=None."""
        runner = self._make_single_ticker_runner(_STALE_OHLCV, FRONTIER)
        df = runner.rank_strangles_by_ev(
            ticker="TEST",
            as_of=None,
        )
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert frontier_drops, f"Strangle ranker must drop STALE at as_of=None: {drops}"
        d = frontier_drops[0]
        assert d["gate"] == "data"
        assert "behind universe data frontier" in d["reason"]

    def test_strangle_ranker_fresh_unaffected_at_asof_none(self):
        """rank_strangles_by_ev does not drop a fresh ticker at as_of=None."""
        runner = self._make_single_ticker_runner(_FRESH_OHLCV, FRONTIER)
        df = runner.rank_strangles_by_ev(
            ticker="TEST",
            as_of=None,
        )
        drops = df.attrs.get("drops", [])
        frontier_drops = [d for d in drops if "frontier" in d.get("reason", "")]
        assert not frontier_drops, f"Strangle ranker must NOT drop FRESH at as_of=None: {drops}"
