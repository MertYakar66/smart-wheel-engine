"""Survivorship-aware backtest harness (R3 / docs/DATA_LAYER_DEEP_READ_DESIGN.md
Part B).

The regression drivers in ``backtests/regression/_common.py`` take a FIXED
``tickers`` list and never consult index membership, so they are
survivorship-biased by construction (only names that survived to today). This
harness fixes that:

* The universe at each rebalance is the **point-in-time** S&P 500 membership from
  ``data.consolidated_loader.ConsolidatedBloombergLoader.get_universe_as_of`` —
  which includes names that later delisted (Lehman, WaMu, …) and excludes names
  that had not yet joined. (``percentage_weight`` is the all-zeros sentinel, so
  selection is by membership presence only — ``min_weight`` is a no-op.)
* The connector is built ``deep_history=True`` (R2), so each PIT name — current
  or delisted — resolves its history from the assembled monolith ∪ deep ∪
  delisted panels.
* Terminal valuation is **delisting-aware**: when a position's underlying has no
  price on/after expiry because the name delisted, the harness values it at the
  last traded close on/before expiry (the delisting price), else 0.0 — so a name
  that crashed to zero realizes its loss instead of being silently dropped.

§2: every candidate is still ranked through
``WheelRunner.rank_candidates_by_ev`` — no engine change, no EV bypass. This
module lives in ``backtests/``; it is not on the decision-layer path.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from backtests.regression._common import (
    _compute_metrics,
    _forward_replay_realized_pnl,
    _next_business_day,
    _spot_on_or_after,
    assert_data_window_available,
    friction_adjusted_premium,
    friction_assignment_cost,
    friction_open_cost,
)

# Deep OHLCV slices (relative to the connector data_dir) used to extend the
# data-window floor assertion to the assembled span.
_DEEP_OHLCV_SLICES = (
    "deep/sp500_ohlcv__1994_2018.csv.gz",
    "deep/sp500_ohlcv__delisted.csv.gz",
)


def make_deep_connector(data_dir: str = "data/bloomberg"):
    """A ``MarketDataConnector`` with deep-history assembly ON."""
    from engine.data_connector import MarketDataConnector

    return MarketDataConnector(data_dir, deep_history=True)


def load_membership(data_dir: str = "data/bloomberg"):
    """A ``ConsolidatedBloombergLoader`` with PIT index membership loaded."""
    from data.consolidated_loader import ConsolidatedBloombergLoader

    loader = ConsolidatedBloombergLoader(data_dir=data_dir)
    loader.load_index_membership()
    return loader


def pit_universe(
    as_of: str,
    data_dir: str = "data/bloomberg",
    loader: Any | None = None,
) -> list[str]:
    """Point-in-time S&P 500 membership on ``as_of`` (normalized tickers).

    Selection is by membership presence only — the pulled ``percentage_weight``
    is the all-zeros sentinel, so ``min_weight`` is intentionally not used.
    """
    loader = loader or load_membership(data_dir)
    return loader.get_universe_as_of(as_of)


def terminal_spot(
    conn: Any,
    ticker: str,
    expiry: date,
    max_lookahead_days: int = 7,
) -> tuple[float | None, bool]:
    """Spot for terminal valuation at ``expiry`` — delisting-aware.

    Returns ``(spot, delisted)``:
    * the close on/after ``expiry`` within ``max_lookahead_days`` (normal case),
      ``delisted=False``;
    * else the last traded close on/before ``expiry`` (the delisting price),
      ``delisted=True``;
    * else ``(0.0, True)`` — truly no data, total loss.

    The key survivorship guarantee: this NEVER returns ``None`` for a name that
    has any history, so an assigned position on a delisted name realizes its loss
    instead of being silently dropped (which is what a plain on/after lookup
    does — it returns ``None`` past the delisting date).
    """
    spot = _spot_on_or_after(conn, ticker, expiry, max_lookahead_days)
    if spot is not None:
        return spot, False
    try:
        df = conn.get_ohlcv(ticker, end_date=expiry.isoformat())
    except Exception:
        df = None
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1]), True
    return 0.0, True


def run_survivorship_backtest(
    *,
    capital: float,
    start: str,
    end: str,
    data_dir: str = "data/bloomberg",
    friction_level: str = "none",
    rebalance_months: int = 3,
    tickers: Sequence[str] | None = None,
    max_universe: int | None = None,
    top_n: int = 10,
    max_new_per_day: int = 3,
    dte_target: int = 35,
    delta_target: float = 0.25,
    contracts: int = 1,
) -> dict:
    """Run a survivorship-aware wheel backtest over ``[start, end]``.

    Universe per rebalance period: ``tickers`` (curated, fixed) if given, else
    the PIT membership at the period start (recomputed every ``rebalance_months``,
    optionally capped to ``max_universe`` deterministically). Every candidate is
    ranked via ``WheelRunner.rank_candidates_by_ev`` against a ``deep_history``
    connector. Settlement + forward-replay use :func:`terminal_spot` so delisted
    names realize their loss.

    Returns ``{"metrics", "rank_log", "open_positions", "closed_positions"}``.
    """
    from engine.wheel_runner import WheelRunner
    from engine.wheel_tracker import PositionState, WheelTracker

    conn = make_deep_connector(data_dir)
    runner = WheelRunner(data_dir=data_dir)
    runner._connector = conn  # inject the deep connector (lazy property backer)

    assert_data_window_available(
        start,
        end,
        ohlcv_path=Path(data_dir) / "sp500_ohlcv.csv",
        extra_floor_paths=[Path(data_dir) / s for s in _DEEP_OHLCV_SLICES],
    )

    loader = None if tickers is not None else load_membership(data_dir)
    tracker = WheelTracker(initial_capital=capital, connector=conn)
    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    rank_log_rows: list[dict] = []

    universe_cache: dict[tuple[int, int], list[str]] = {}

    def _universe_for(day: date) -> list[str]:
        if tickers is not None:
            return list(tickers)
        period = (day.year, (day.month - 1) // rebalance_months)
        if period not in universe_cache:
            names = pit_universe(day.isoformat(), loader=loader)
            if max_universe is not None:
                names = names[:max_universe]
            universe_cache[period] = names
        return universe_cache[period]

    for today in trading_days:
        # 1) MTM open positions
        open_tickers = [
            t for t, p in tracker.positions.items() if p.state != PositionState.NO_POSITION
        ]
        prices_today: dict[str, float] = {}
        for t in open_tickers:
            spot = _spot_on_or_after(conn, t, today, max_lookahead_days=1)
            if spot is not None:
                prices_today[t] = spot
        if prices_today:
            tracker.mark_to_market(today, prices_today)

        # 2) Settle expirations — delisting-aware terminal spot so a name that
        #    delisted before expiry still settles (realizes the loss) rather than
        #    being skipped.
        for t in list(tracker.positions.keys()):
            pos = tracker.positions[t]
            if (
                pos.state == PositionState.SHORT_PUT
                and pos.put_expiration_date
                and pos.put_expiration_date <= today
            ):
                spot, _delisted = terminal_spot(conn, t, today)
                if spot is None:
                    continue
                was_assigned = spot < (pos.put_strike or 0.0)
                tracker.handle_put_expiration(t, today, spot)
                if was_assigned and friction_level == "full":
                    tracker.cash -= friction_assignment_cost(
                        pos.put_strike or 0.0, contracts, friction_level
                    )
            elif (
                pos.state == PositionState.COVERED_CALL
                and pos.call_expiration_date
                and pos.call_expiration_date <= today
            ):
                spot, _delisted = terminal_spot(conn, t, today)
                if spot is None:
                    continue
                was_called_away = spot > (pos.call_strike or float("inf"))
                tracker.handle_call_expiration(t, today, spot)
                if was_called_away and friction_level == "full":
                    tracker.cash -= friction_assignment_cost(
                        pos.call_strike or 0.0, contracts, friction_level
                    )

        # 3) Rank the PIT/curated universe and open EV>0 puts.
        expiration_default = _next_business_day(today + timedelta(days=dte_target))
        try:
            frame = runner.rank_candidates_by_ev(
                tickers=_universe_for(today),
                dte_target=dte_target,
                delta_target=delta_target,
                contracts=contracts,
                top_n=top_n,
                min_ev_dollars=-1e9,  # capture all ranked rows; gate on EV>0 at open
                as_of=today.isoformat(),
                include_diagnostic_fields=True,
            )
        except Exception:
            continue
        if frame is None or len(frame) == 0:
            continue

        for _, row in frame.iterrows():
            premium_raw = float(row.get("premium", 0.0))
            rank_log_rows.append(
                {
                    "date": today.isoformat(),
                    "ticker": str(row.get("ticker", "")),
                    "ev_dollars": float(row.get("ev_dollars", 0.0)),
                    "premium": friction_adjusted_premium(premium_raw, friction_level),
                    "premium_raw": premium_raw,
                    "strike": float(row.get("strike", 0.0)),
                    "iv": float(row.get("iv", 0.0)),
                    "prob_profit": float(row.get("prob_profit", float("nan"))),
                    "expiration_date": expiration_default.isoformat(),
                    "friction_level": friction_level,
                }
            )

        opens_today = 0
        for _, row in frame.iterrows():
            if opens_today >= max_new_per_day:
                break
            if float(row.get("ev_dollars", 0.0)) <= 0:
                continue
            t = str(row.get("ticker", ""))
            if t in tracker.positions and tracker.positions[t].state != PositionState.NO_POSITION:
                continue
            strike = float(row.get("strike", 0.0))
            premium = friction_adjusted_premium(float(row.get("premium", 0.0)), friction_level)
            if premium <= 0 or strike <= 0:
                continue
            if tracker.available_buying_power() < strike * 100 * contracts:
                continue
            opened = tracker.open_short_put(
                ticker=t,
                strike=strike,
                premium=premium,
                entry_date=today,
                expiration_date=expiration_default,
                iv=float(row.get("iv", 0.0)),
            )
            if opened:
                if friction_level == "full":
                    tracker.cash -= friction_open_cost(contracts, friction_level)
                opens_today += 1

    # Forward-replay realized P&L — delisting-aware so a crashed name's loss is
    # realized, never NaN-dropped.
    rank_log = pd.DataFrame(rank_log_rows)
    if not rank_log.empty:
        realized: list[float] = []
        spots: list[float] = []
        delisted_flags: list[bool] = []
        for r in rank_log.itertuples(index=False):
            exp_dt = date.fromisoformat(r.expiration_date)
            spot, delisted = terminal_spot(conn, r.ticker, exp_dt)
            spot = 0.0 if spot is None else spot
            realized.append(_forward_replay_realized_pnl(r.strike, r.premium, spot))
            spots.append(spot)
            delisted_flags.append(delisted)
        rank_log["spot_at_expiry"] = spots
        rank_log["realized_pnl"] = realized
        rank_log["delisted_at_expiry"] = delisted_flags

    metrics = _compute_metrics(rank_log, tracker)
    return {
        "metrics": metrics,
        "rank_log": rank_log,
        "open_positions": {
            t: p.state.value
            for t, p in tracker.positions.items()
            if p.state != PositionState.NO_POSITION
        },
        "closed_positions": list(tracker.closed_positions),
    }
