#!/usr/bin/env python3
"""
Pull intraday option trade + quote tape from Theta.

Heavy data — one ticker × one expiry × 1 day can be 50-500k rows. Use
sparingly. The dealer-positioning module benefits most from this, since
it lets us classify prints as buy-initiated (at ask) vs sell-initiated
(at bid) and back out real dealer inventory flow.

Output layout (partitioned parquet)::

    data_processed/theta/option_tape/
        ticker=<SYM>/
            date=<YYYY-MM-DD>/
                trades.parquet    # trade-by-trade prints
                quotes.parquet    # bid/ask/bid_size/ask_size bars (1m)

Per-file schema (trades)::

    ts, expiration, strike, right, price, size, exchange, condition,
    nbbo_bid, nbbo_ask, side_inferred (buy|sell|mid)

Per-file schema (quotes)::

    ts, expiration, strike, right, bid, ask, bid_size, ask_size, mid

Usage
-----
    # One ticker, next 35-DTE expiry, last 5 trading days
    python scripts/pull_theta_option_tape.py --tickers AAPL --days 5

    # Specific expiry (YYYYMMDD), longer window
    python scripts/pull_theta_option_tape.py --tickers SPY \\
        --expiration 20260516 --days 20

    # ATM-only (one strike closest to current spot) — keeps data tractable
    python scripts/pull_theta_option_tape.py --tickers AAPL --days 5 --atm-only

Budget warning
--------------
Full S&P universe × 1 year × full chain = hundreds of GB. For signal
research start with 10-20 high-liquidity tickers, ATM ± 10 strikes,
1-year window. Orchestrator ``pull_all.py`` does NOT run this by default.
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.theta_connector import ThetaConnector, _normalise_theta_symbol  # noqa: E402

logger = logging.getLogger(__name__)
OUT_ROOT = _ROOT / "data_processed" / "theta" / "option_tape"


def _theta_up() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 25503))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _business_days(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _nearest_expiration(conn: ThetaConnector, ticker: str, dte: int = 35) -> str | None:
    df = conn._fetch("/v3/option/list/expirations",
                     {"symbol": _normalise_theta_symbol(ticker)})
    if df is None or df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    col = next((c for c in ("expiration", "date") if c in df.columns), None)
    if col is None:
        return None
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    target = pd.Timestamp.now().normalize() + pd.Timedelta(days=dte)
    best = dates.iloc[(dates - target).abs().argsort().iloc[0]]
    return best.strftime("%Y%m%d")


def _atm_strike(conn: ThetaConnector, ticker: str, expiration: str) -> float | None:
    """Cheapest way to find the ATM strike: ask snapshot, find 50-delta."""
    df = conn._fetch("/v3/option/snapshot/greeks/first_order",
                     {"symbol": _normalise_theta_symbol(ticker),
                      "expiration": expiration})
    if df is None or df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    if "strike" not in df.columns or "delta" not in df.columns:
        return None
    puts = df[df.get("right", "").str.lower() == "put"].dropna(subset=["delta"])
    if puts.empty:
        return None
    puts = puts.copy()
    puts["_g"] = (puts["delta"].abs() - 0.50).abs()
    return float(puts.sort_values("_g").iloc[0]["strike"])


def _fetch_tape(
    conn: ThetaConnector, endpoint: str, ticker: str, expiration: str,
    day: date, strike: float | None, interval: str | None
) -> pd.DataFrame:
    params = {
        "symbol": _normalise_theta_symbol(ticker),
        "expiration": expiration,
        "start_date": day.strftime("%Y%m%d"),
        "end_date": day.strftime("%Y%m%d"),
    }
    if strike is not None:
        params["strike"] = strike
    if interval:
        params["interval"] = interval
    df = conn._fetch(endpoint, params)
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    return df


def _classify_side(row) -> str:
    """Infer buy-initiated vs sell-initiated from trade vs NBBO midpoint."""
    bid, ask, px = row.get("nbbo_bid"), row.get("nbbo_ask"), row.get("price")
    if pd.isna(bid) or pd.isna(ask) or pd.isna(px) or ask <= bid:
        return "mid"
    mid = (bid + ask) / 2
    if px >= ask - 1e-9:
        return "buy"
    if px <= bid + 1e-9:
        return "sell"
    return "buy" if px > mid else "sell"


def _write_day(ticker: str, day: date, trades: pd.DataFrame, quotes: pd.DataFrame) -> str:
    outdir = OUT_ROOT / f"ticker={ticker}" / f"date={day.isoformat()}"
    outdir.mkdir(parents=True, exist_ok=True)
    n_tr = n_q = 0
    if not trades.empty:
        trades.to_parquet(outdir / "trades.parquet", index=False)
        n_tr = len(trades)
    if not quotes.empty:
        quotes.to_parquet(outdir / "quotes.parquet", index=False)
        n_q = len(quotes)
    return f"trades={n_tr} quotes={n_q}"


def _one(ticker: str, expiration: str, day: date, strike: float | None) -> tuple[str, date, str]:
    try:
        conn = ThetaConnector()
    except Exception as e:
        return ticker, day, f"conn: {e}"

    trades = _fetch_tape(conn, "/v3/option/history/trade", ticker, expiration,
                        day, strike, interval=None)
    if not trades.empty:
        # Normalise / add side inference
        ts_col = next((c for c in ("ts", "timestamp", "trade_time", "created") if c in trades.columns), None)
        if ts_col:
            trades[ts_col] = pd.to_datetime(trades[ts_col], errors="coerce")
            trades = trades.dropna(subset=[ts_col]).rename(columns={ts_col: "ts"})
        if {"nbbo_bid", "nbbo_ask", "price"}.issubset(trades.columns):
            trades["side_inferred"] = trades.apply(_classify_side, axis=1)

    quotes = _fetch_tape(conn, "/v3/option/history/quote", ticker, expiration,
                        day, strike, interval="1m")
    if not quotes.empty:
        ts_col = next((c for c in ("ts", "timestamp", "bar_start", "created") if c in quotes.columns), None)
        if ts_col:
            quotes[ts_col] = pd.to_datetime(quotes[ts_col], errors="coerce")
            quotes = quotes.dropna(subset=[ts_col]).rename(columns={ts_col: "ts"})
        if {"bid", "ask"}.issubset(quotes.columns):
            quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2

    msg = _write_day(ticker, day, trades, quotes)
    return ticker, day, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--expiration", help="YYYYMMDD (default: nearest 35-DTE)")
    ap.add_argument("--dte", type=int, default=35, help="Target DTE when --expiration omitted")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--atm-only", action="store_true",
                    help="Restrict to the single ATM strike (cuts volume ~50x)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("Theta Terminal not reachable on 127.0.0.1:25503 — start it and retry.")
        return 2

    try:
        probe_conn = ThetaConnector()
    except Exception as e:
        print(f"Could not construct ThetaConnector: {e}")
        return 2

    # Resolve expiration + strike per ticker
    resolved: list[tuple[str, str, float | None]] = []
    for ticker in args.tickers:
        exp = args.expiration or _nearest_expiration(probe_conn, ticker, args.dte)
        if not exp:
            print(f"  {ticker}: could not resolve expiration, skipping")
            continue
        strike = _atm_strike(probe_conn, ticker, exp) if args.atm_only else None
        print(f"  {ticker}: expiration={exp} atm_strike={strike}")
        resolved.append((ticker, exp, strike))

    if not resolved:
        print("Nothing to fetch.")
        return 1

    end_d = date.today()
    start_d = end_d - timedelta(days=args.days * 2)
    days = list(_business_days(start_d, end_d))[-args.days:]

    print(f"\nPulling tape: {len(resolved)} contracts × {len(days)} days × {args.workers} workers")
    t0 = time.perf_counter()
    jobs = [(t, exp, d, strike) for t, exp, strike in resolved for d in days]
    n_done = n_empty = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, t, exp, d, strike): (t, d) for t, exp, d, strike in jobs}
        for fut in as_completed(futs):
            ticker, day, msg = fut.result()
            n_done += 1
            if "trades=0 quotes=0" in msg:
                n_empty += 1
            print(f"  [{n_done:>4}/{len(jobs)}] {ticker:<6} {day}  {msg}", flush=True)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Done in {elapsed:.1f}s  |  {n_empty} empty day-contracts")
    print(f"Output under {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
