"""Held finding F4-iv-fallback-lookahead — reproduction test.

FINDING (backtest lookahead). When ``_resolve_pit_atm_iv`` returns ``None``
at a *historical* ``as_of`` (e.g. the connector has no PIT IV history for the
name, or the IV file is empty at that date), the three EV rankers
unconditionally fall through to ``fundamentals.get("implied_vol_atm")``:

    engine/wheel_runner.py:1668-1672  (rank_candidates_by_ev / puts)
    engine/wheel_runner.py:3083-3085  (rank_covered_calls_by_ev)
    engine/wheel_runner.py:3716-3718  (rank_strangles_by_ev)

There is NO ``as_of is not None`` guard on that fall-through. And
``engine/data_connector.py:1565`` proves ``implied_vol_atm`` is read from the
CURRENT snapshot column ``30day_impvol_100.0%mny_df`` (no date axis) —

    "implied_vol_atm": self._clean_served_iv(r.get("30day_impvol_100.0%mny_df")),

So a *dated* backtest with missing PIT IV silently prices its BSM strike-solve
and synthetic premium off **today's (2026-snapshot) IV** — a look-ahead leak.

CORRECT (bug-free) BEHAVIOR asserted here: at a historical ``as_of`` where no
PIT IV is available, the ranker must NOT substitute the current-snapshot IV.
It should refuse the name (drop it) rather than book a survivor row priced off
snapshot IV. (Live ranking, ``as_of=None``, may legitimately use the snapshot;
this test only pins the *dated* case.)

This test drives the real ``WheelRunner.rank_candidates_by_ev`` through a
connector stub with NO ``get_iv_history`` (so ``_resolve_pit_atm_iv`` -> None,
per wheel_runner.py:188-189) but a distinctive current-snapshot
``implied_vol_atm``. On buggy code the ranker returns survivor rows whose
``iv`` column equals that snapshot IV; the assertion that no survivor carries
the snapshot IV therefore FAILS on current code (the proof), and will PASS
once an ``as_of is not None`` guard is added to the fall-through.

The stub / harness deliberately mirror ``tests/test_ranker_iv_pit.py`` so the
only behavioral difference under test is the missing PIT-at-historical-as_of
guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner, _resolve_pit_atm_iv

# Offline flags: keep the ranker off the network-y advisor stacks so the only
# input that matters here is the IV path (mirrors test_ranker_iv_pit.py).
_OFFLINE = {
    "use_dealer_positioning": False,
    "use_news_sentiment": False,
    "use_credit_regime": False,
    "use_skew_dynamics": False,
}

_TICKERS = ["AAA", "BBB"]

# A distinctive current-snapshot IV (percent form, as Bloomberg stores it).
# 88% is far from any plausible dated IV for these synthetic names, so if a
# survivor row carries iv == 0.88 we KNOW the snapshot leaked in.
_SNAPSHOT_IV_PCT = 88.0
_SNAPSHOT_IV_DEC = _SNAPSHOT_IV_PCT / 100.0

# A clearly historical backtest date — the leak is that the ranker prices this
# dated run off the (future, 2026) snapshot IV.
_HIST_AS_OF = "2020-06-15"


class _NoPitIVHistoricalConn:
    """Connector stub with a current-snapshot ``implied_vol_atm`` but NO
    point-in-time IV history — i.e. ``_resolve_pit_atm_iv`` returns ``None``
    (wheel_runner.py:188-189, ``not hasattr(conn, 'get_iv_history')``).

    Mirrors ``ThetaConnector`` and the ~15 existing snapshot-only stubs.
    OHLCV covers 2016->2021 so a 2020 ``as_of`` has ample history to clear the
    504-day gate; the snapshot ``implied_vol_atm`` is the ONLY IV source, and
    it is (by construction of a real Bloomberg pull) a 2026-snapshot value with
    no date axis.
    """

    def __init__(self, tickers, *, snapshot_iv_pct: float = _SNAPSHOT_IV_PCT):
        self._tickers = list(tickers)
        self._snapshot_iv = snapshot_iv_pct
        self._ohlcv: dict[str, pd.DataFrame] = {}
        # Business days from 2016-01-01 for ~1500 sessions -> reaches into 2021,
        # so _HIST_AS_OF (2020-06-15) has >504 prior sessions.
        for i, t in enumerate(tickers):
            idx = pd.date_range("2016-01-01", periods=1500, freq="B")
            rng = np.random.default_rng(100 + i)
            base = 80.0 * (1.0 + 0.45 * i)
            close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx))))
            self._ohlcv[t] = pd.DataFrame({"close": close}, index=idx)

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._ohlcv[ticker]

    def get_fundamentals(self, ticker: str, as_of=None) -> dict:
        # The real MarketDataConnector.get_fundamentals reads implied_vol_atm
        # from the CURRENT snapshot column (data_connector.py:1565) regardless
        # of as_of; only dividend_yield is PIT-overridden. We replicate that:
        # the snapshot IV is returned unchanged for every as_of.
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

    # NOTE: intentionally NO get_iv_history -> _resolve_pit_atm_iv returns None.


def _runner_with(connector) -> WheelRunner:
    r = WheelRunner()
    r._connector = connector
    return r


def _rank(runner: WheelRunner, **extra) -> pd.DataFrame:
    kw = dict(tickers=_TICKERS, top_n=10, min_ev_dollars=-1e9, **_OFFLINE)
    kw.update(extra)
    return runner.rank_candidates_by_ev(**kw)


@pytest.mark.xfail(
    reason="held finding F4-iv-fallback-lookahead; drop xfail when fixed",
    strict=True,
)
def test_ranker_does_not_use_snapshot_iv_at_historical_as_of():
    """At a historical ``as_of`` with no PIT IV, the puts ranker must NOT price
    off the current-snapshot ``implied_vol_atm``.

    Preconditions verified inline:
      * ``_resolve_pit_atm_iv`` returns None for this connector (no PIT axis),
        so the fall-through at wheel_runner.py:1671-1701 is the code under test.
      * The snapshot IV is a distinctive 0.88, distinguishable from any dated
        value.

    CORRECT behavior: at ``as_of=2020-06-15`` the ranker refuses the name (drops
    it) rather than booking a survivor whose ``iv`` == the 2026-snapshot 0.88.

    On CURRENT (buggy) code the ranker DOES substitute the snapshot IV: every
    survivor row carries iv == 0.88, so the assertion below FAILS. That failure
    IS the proof of the look-ahead leak (missing ``as_of is not None`` guard).
    """
    conn = _NoPitIVHistoricalConn(_TICKERS)

    # Precondition 1: no PIT IV path -> _resolve_pit_atm_iv returns None, so the
    # ranker is forced onto the snapshot fall-through under test.
    assert not hasattr(conn, "get_iv_history")
    assert _resolve_pit_atm_iv(conn, "AAA", _HIST_AS_OF) is None

    df = _rank(_runner_with(conn), as_of=_HIST_AS_OF)

    # Correct behavior: no survivor row may carry the current-snapshot IV as its
    # priced ``iv`` at a historical as_of. Either the name is dropped (empty
    # frame / drop rows) or it is priced off a genuine PIT IV — never the 2026
    # snapshot. Assert the survivor set contains NO snapshot-priced row.
    if not df.empty and "iv" in df.columns:
        leaked = [
            (row["ticker"], row["iv"])
            for _, row in df.iterrows()
            if abs(float(row["iv"]) - _SNAPSHOT_IV_DEC) < 1e-6
        ]
        assert not leaked, (
            "look-ahead leak: at historical as_of="
            f"{_HIST_AS_OF} the ranker priced survivor(s) {leaked} off the "
            f"CURRENT-snapshot IV {_SNAPSHOT_IV_DEC} (implied_vol_atm, "
            "data_connector.py:1565) because the fall-through at "
            "wheel_runner.py:1671-1701 lacks an `as_of is not None` guard"
        )
