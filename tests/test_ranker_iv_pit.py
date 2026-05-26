"""Tests for the IV-PIT fix in the three rankers.

S23 F3 logged that :meth:`WheelRunner.rank_candidates_by_ev` (and the
companion :meth:`rank_covered_calls_by_ev` /
:meth:`rank_strangles_by_ev`) used
``conn.get_fundamentals(ticker)["implied_vol_atm"]`` — a snapshot
column with no date axis — to feed the BSM strike-solve and the
synthetic premium. A ranker run with ``as_of="2026-03-05"`` therefore
priced strikes against today's IV instead of the IV that was actually
quoted on 2026-03-05. Concrete example from S23: AVGO snapshot IV was
42.96%, but the IV file's ``hist_put_imp_vol`` on 2026-03-05 was 49.5%
— a ~15% relative error in the IV input, material for any name near
earnings or in a vol regime change.

The fix introduces :func:`engine.wheel_runner._resolve_pit_atm_iv`,
which mirrors :meth:`engine.wheel_tracker.WheelTracker._connector_atm_iv`:

    PIT (get_iv_history.iloc[-1] at end_date=as_of)  →  fundamentals snapshot.

The fundamentals fallback stays as a hard-required compatibility path
for connectors / test stubs that don't expose ``get_iv_history`` (e.g.
``ThetaConnector`` per the docstring in ``_connector_atm_iv``; ~15
existing test stubs that only provide ``get_fundamentals``).

Pinned here:

* Helper :func:`_resolve_pit_atm_iv` — composite (put + call) / 2 from
  the most recent row, percent→decimal, defensive (no connector / no
  method / raise / empty / missing columns / degenerate → None).
* End-to-end on :meth:`rank_candidates_by_ev` — PIT-first when the
  connector has IV history, fundamentals fallback otherwise, and the
  ``iv`` column actually shifts with ``as_of``.
* Symmetric coverage on :meth:`rank_covered_calls_by_ev` and
  :meth:`rank_strangles_by_ev` — the same helper is wired in both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner, _resolve_pit_atm_iv

_OFFLINE = {
    "use_dealer_positioning": False,
    "use_news_sentiment": False,
    "use_credit_regime": False,
    "use_skew_dynamics": False,
}

_TICKERS = ["AAA", "BBB"]


# ----------------------------------------------------------------------
# Connector stubs
# ----------------------------------------------------------------------
class _PitIVConn:
    """Connector stub with both ``get_fundamentals`` and ``get_iv_history``.

    ``iv_by_date`` is a dict mapping ``date_iso → iv_pct`` (percent
    form, as Bloomberg stores it). Returning the same value on both
    ``hist_put_imp_vol`` and ``hist_call_imp_vol`` keeps the
    (put+call)/2 average equal to that value, so the test can assert
    against ``iv_by_date[as_of]`` directly.

    ``snapshot_iv`` is what ``get_fundamentals`` returns — distinct
    from any PIT value so the test can prove the PIT path is preferred.
    """

    def __init__(
        self,
        tickers,
        *,
        iv_by_date: dict[str, float] | None = None,
        snapshot_iv: float = 28.0,
        default_days: int = 3000,
    ) -> None:
        self._tickers = list(tickers)
        self._iv_by_date = dict(iv_by_date or {})
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

    def get_next_earnings(self, ticker: str, as_of=None):
        return None

    def get_universe(self) -> list[str]:
        return list(self._tickers)

    def get_iv_history(self, ticker, start_date=None, end_date=None) -> pd.DataFrame:
        # Replay the configured mapping: for every (date, iv_pct) row we
        # produce one DataFrame row with hist_put_imp_vol = hist_call_imp_vol
        # = iv_pct. Then trim to end_date so the ranker's lookup behaves
        # like a real PIT cut. Returns an empty frame when no rows match.
        if not self._iv_by_date:
            return pd.DataFrame()
        rows = sorted(self._iv_by_date.items())
        idx = pd.to_datetime([d for d, _ in rows])
        vals = [v for _, v in rows]
        df = pd.DataFrame(
            {"hist_put_imp_vol": vals, "hist_call_imp_vol": vals},
            index=idx,
        )
        if end_date is not None:
            df = df.loc[df.index <= pd.Timestamp(end_date)]
        return df


class _NoIVHistConn:
    """A connector that lacks ``get_iv_history`` entirely — mirrors
    ``ThetaConnector`` and the ~15 existing test stubs that only
    provide ``get_fundamentals``. Intentionally NOT inheriting from
    ``_PitIVConn`` so ``hasattr(conn, 'get_iv_history')`` is False at
    the type level (no fragile per-instance attribute juggling)."""

    def __init__(self, tickers, *, snapshot_iv: float = 28.0, default_days: int = 3000):
        self._tickers = list(tickers)
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

    def get_next_earnings(self, ticker: str, as_of=None):
        return None

    def get_universe(self) -> list[str]:
        return list(self._tickers)


class _RaisingIVHistConn(_PitIVConn):
    """A connector whose ``get_iv_history`` always raises."""

    def get_iv_history(self, ticker, start_date=None, end_date=None):
        raise RuntimeError("upstream iv history outage")


class _MissingColsConn(_PitIVConn):
    """A connector that returns IV history with non-IV columns only."""

    def get_iv_history(self, ticker, start_date=None, end_date=None):
        return pd.DataFrame({"volatility_30d": [25.0]}, index=pd.to_datetime(["2026-03-01"]))


def _runner_with(connector) -> WheelRunner:
    r = WheelRunner()
    r._connector = connector
    return r


def _rank(runner: WheelRunner, **extra) -> pd.DataFrame:
    kw = dict(tickers=_TICKERS, top_n=10, min_ev_dollars=-1e9, **_OFFLINE)
    kw.update(extra)
    return runner.rank_candidates_by_ev(**kw)


# ======================================================================
# 1. _resolve_pit_atm_iv — unit-level
# ======================================================================
class TestResolvePitAtmIv:
    def test_none_when_no_connector(self):
        assert _resolve_pit_atm_iv(None, "AAA", "2026-03-05") is None

    def test_none_when_connector_lacks_get_iv_history(self):
        conn = _NoIVHistConn(_TICKERS)
        assert not hasattr(conn, "get_iv_history")
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") is None

    def test_none_when_get_iv_history_raises(self):
        conn = _RaisingIVHistConn(_TICKERS, iv_by_date={"2026-03-05": 35.0})
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") is None

    def test_none_when_history_empty(self):
        conn = _PitIVConn(_TICKERS, iv_by_date={})
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") is None

    def test_none_when_iv_columns_absent(self):
        conn = _MissingColsConn(_TICKERS)
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") is None

    def test_percent_iv_normalised_to_decimal(self):
        conn = _PitIVConn(_TICKERS, iv_by_date={"2026-03-05": 40.0})
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") == pytest.approx(0.40)

    def test_already_decimal_passes_through(self):
        conn = _PitIVConn(_TICKERS, iv_by_date={"2026-03-05": 0.28})
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") == pytest.approx(0.28)

    def test_uses_most_recent_row_at_or_before_as_of(self):
        conn = _PitIVConn(
            _TICKERS,
            iv_by_date={
                "2026-02-01": 22.0,
                "2026-03-01": 28.0,
                "2026-03-15": 35.0,
                # 2026-04-01 is past as_of, must NOT be selected
                "2026-04-01": 99.0,
            },
        )
        # as_of=2026-03-20 → most recent at-or-before is 2026-03-15 (35.0%)
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-20") == pytest.approx(0.35)
        # as_of=2026-03-10 → most recent is 2026-03-01 (28.0%)
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-10") == pytest.approx(0.28)

    def test_put_and_call_iv_are_averaged(self):
        # Hand-build a frame with distinct put and call IVs so we can
        # observe the average. The _PitIVConn stub mirrors put == call,
        # so we need a small bespoke connector here.
        class _AsymmetricConn:
            def get_iv_history(self, ticker, start_date=None, end_date=None):
                return pd.DataFrame(
                    {"hist_put_imp_vol": [30.0], "hist_call_imp_vol": [40.0]},
                    index=pd.to_datetime(["2026-03-05"]),
                )

        iv = _resolve_pit_atm_iv(_AsymmetricConn(), "AAA", "2026-03-05")
        assert iv == pytest.approx(0.35)  # (30 + 40) / 2 = 35 → 0.35

    def test_returns_none_when_iv_above_sanity_band(self):
        # 600% → 6.0 decimal after normalisation, outside the (0, 5] band
        # → helper rejects so the caller falls back to fundamentals.
        conn = _PitIVConn(_TICKERS, iv_by_date={"2026-03-05": 600.0})
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") is None

    def test_returns_none_when_iv_non_positive(self):
        conn = _PitIVConn(_TICKERS, iv_by_date={"2026-03-05": 0.0})
        assert _resolve_pit_atm_iv(conn, "AAA", "2026-03-05") is None

    def test_skips_nan_legs_then_averages(self):
        class _NaNCallConn:
            def get_iv_history(self, ticker, start_date=None, end_date=None):
                return pd.DataFrame(
                    {
                        "hist_put_imp_vol": [30.0],
                        "hist_call_imp_vol": [float("nan")],
                    },
                    index=pd.to_datetime(["2026-03-05"]),
                )

        # Only the put leg is usable → result is just the put IV.
        assert _resolve_pit_atm_iv(_NaNCallConn(), "AAA", "2026-03-05") == pytest.approx(0.30)


# ======================================================================
# 2. rank_candidates_by_ev — PIT-first wiring + as_of sensitivity
# ======================================================================
class TestRankCandidatesPitWiring:
    def test_iv_column_equals_pit_iv_not_snapshot(self):
        """The single most important assertion: when both PIT and
        fundamentals are available with different IVs, the ranker's
        ``iv`` column matches the PIT value, not the snapshot."""
        conn = _PitIVConn(
            _TICKERS,
            snapshot_iv=20.0,  # fundamentals says 20%
            iv_by_date={"2026-03-05": 35.0, "2026-03-15": 35.0},  # PIT says 35%
        )
        df = _rank(_runner_with(conn), as_of="2026-03-15")
        assert not df.empty
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.35), (
                f"{row['ticker']}: ranker used iv={row['iv']!r} but PIT value at "
                f"as_of=2026-03-15 is 0.35 (fundamentals snapshot was 0.20)"
            )

    def test_iv_column_moves_with_as_of(self):
        """Same ticker, two different as_of dates, two different IVs in
        history. The ``iv`` column must move with ``as_of`` — this is
        the headline S23 F3 fix."""
        conn = _PitIVConn(
            _TICKERS,
            snapshot_iv=20.0,
            iv_by_date={
                "2026-02-15": 22.0,
                "2026-03-15": 38.0,
            },
        )
        df_early = _rank(_runner_with(conn), as_of="2026-02-20")
        df_late = _rank(_runner_with(conn), as_of="2026-03-20")
        assert not df_early.empty and not df_late.empty

        # Same survivor set (no event drops; geometry-only differences)
        common = set(df_early["ticker"]) & set(df_late["ticker"])
        assert common, "expected at least one common survivor across as_of"

        for ticker in common:
            iv_early = float(df_early.loc[df_early["ticker"] == ticker, "iv"].iloc[0])
            iv_late = float(df_late.loc[df_late["ticker"] == ticker, "iv"].iloc[0])
            assert iv_early == pytest.approx(0.22)
            assert iv_late == pytest.approx(0.38)
            assert iv_late > iv_early, (
                "PIT IV must move with as_of; early ≠ late was the S23 F3 bug"
            )

    def test_falls_back_to_fundamentals_when_no_get_iv_history(self):
        """Connectors without ``get_iv_history`` (e.g. ThetaConnector and
        most existing test stubs) must keep working — the fallback path
        gives them the legacy snapshot behavior."""
        conn = _NoIVHistConn(_TICKERS, snapshot_iv=27.0)
        df = _rank(_runner_with(conn), as_of="2026-03-05")
        assert not df.empty
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.27), (
                f"{row['ticker']}: ranker should fall back to fundamentals "
                f"(0.27) when get_iv_history is missing; got iv={row['iv']!r}"
            )

    def test_falls_back_to_fundamentals_when_get_iv_history_empty(self):
        """Connector has the method but the IV file has no rows for
        this ticker — must fall back to fundamentals."""
        conn = _PitIVConn(_TICKERS, snapshot_iv=27.0, iv_by_date={})
        df = _rank(_runner_with(conn), as_of="2026-03-05")
        assert not df.empty
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.27)

    def test_falls_back_to_fundamentals_when_get_iv_history_raises(self):
        conn = _RaisingIVHistConn(_TICKERS, snapshot_iv=27.0)
        df = _rank(_runner_with(conn), as_of="2026-03-05")
        assert not df.empty
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.27)

    def test_falls_back_when_get_iv_history_returns_only_non_iv_columns(self):
        conn = _MissingColsConn(_TICKERS, snapshot_iv=27.0)
        df = _rank(_runner_with(conn), as_of="2026-03-05")
        assert not df.empty
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.27)


# ======================================================================
# 3. Symmetric coverage — covered-call and strangle rankers
# ======================================================================
class TestPitWiringInOtherRankers:
    """Smoke test: the same _resolve_pit_atm_iv helper is wired into
    rank_covered_calls_by_ev and rank_strangles_by_ev. We use a
    connector where snapshot and PIT differ; if the helper were not
    wired in, the ``iv`` column would equal the snapshot."""

    def test_rank_covered_calls_uses_pit_iv(self):
        conn = _PitIVConn(
            ["AAA"],
            snapshot_iv=18.0,
            iv_by_date={"2026-03-15": 36.0},
        )
        df = _runner_with(conn).rank_covered_calls_by_ev(
            ticker="AAA",
            shares_held=100,
            as_of="2026-03-15",
            min_ev_dollars=-1e9,
            target_dtes=(35,),
            target_deltas=(0.25,),
        )
        # Either we got a row (then iv must be PIT) or every cell was
        # dropped (then a drop row exists; still validates the path
        # didn't crash). We expect a row in the no-event, no-history-gate
        # default config.
        assert not df.empty, (
            f"expected at least one row from rank_covered_calls_by_ev; "
            f"drops={df.attrs.get('drops', [])}"
        )
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.36), (
                f"covered-call ranker used iv={row['iv']!r}; PIT is 0.36, "
                f"snapshot is 0.18 — fix not wired into rank_covered_calls_by_ev"
            )

    def test_rank_strangles_uses_pit_iv(self):
        conn = _PitIVConn(
            ["AAA"],
            snapshot_iv=18.0,
            iv_by_date={"2026-03-15": 36.0},
        )
        df = _runner_with(conn).rank_strangles_by_ev(
            ticker="AAA",
            contracts=1,
            as_of="2026-03-15",
            min_ev_dollars=-1e9,
            target_dtes=(35,),
            target_deltas=(0.25,),
            use_timing_gate=False,
        )
        assert not df.empty, (
            f"expected at least one row from rank_strangles_by_ev; "
            f"drops={df.attrs.get('drops', [])}"
        )
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(0.36), (
                f"strangle ranker used iv={row['iv']!r}; PIT is 0.36, "
                f"snapshot is 0.18 — fix not wired into rank_strangles_by_ev"
            )


# ======================================================================
# 4. The realism gap, captured in code
# ======================================================================
class TestRealismGapVisible:
    """The S23 F3 finding in test form: when snapshot IV and PIT IV
    disagree materially, the realised difference in the ranker's iv
    column is large and explains downstream EV shifts. This is not a
    regression guard — it documents the behavior that justified the
    fix."""

    def test_snapshot_to_pit_delta_is_visible_in_iv_column(self):
        # Pre-fix: this connector would have produced iv=0.18 (snapshot).
        # Post-fix: it produces iv=0.36 (PIT) — a 2x relative shift.
        conn = _PitIVConn(
            _TICKERS,
            snapshot_iv=18.0,
            iv_by_date={"2026-03-15": 36.0},
        )
        df = _rank(_runner_with(conn), as_of="2026-03-15")
        assert not df.empty
        ivs = df["iv"].tolist()
        # Every survivor used the PIT IV (0.36), not the snapshot (0.18).
        for iv in ivs:
            assert iv == pytest.approx(0.36)
        # And NONE of them used the snapshot value.
        assert all(abs(iv - 0.18) > 0.05 for iv in ivs)
