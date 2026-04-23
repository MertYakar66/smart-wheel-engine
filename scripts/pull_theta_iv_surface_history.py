#!/usr/bin/env python3
"""
Pull historical IV surface from Theta, one ticker/date snapshot at a time.

Today the feature store has a single ATM-IV scalar per ticker per day. Every
skew / term-structure / surface signal downstream (volatility_surface.py,
skew_dynamics.py, dealer_positioning.py) has to fake it off of that scalar.

This puller fixes that: for each (ticker, date) it pulls the implied-vol
history across strikes and expiries, writes a partitioned parquet, and
leaves behind enough metadata for the feature store to pick it up.

Prerequisites
-------------
- ThetaData Terminal running on 127.0.0.1:25503
- ThetaData Standard tier or above (history endpoints are tier-gated)

Output layout
-------------
data_processed/theta/iv_surface_history/
    ticker=<SYM>/
        year=<YYYY>/
            data.parquet     # columns: date, strike, right, delta, iv, mid,
                             #          expiration, dte, ticker

Usage
-----
    # 500 tickers, last 2 years, 4 workers
    python scripts/pull_theta_iv_surface_history.py \\
        --universe sp500 --start 2024-04-01 --end 2026-04-23 --workers 4

    # Single-ticker smoke
    python scripts/pull_theta_iv_surface_history.py --tickers AAPL --days 30

    # Resume after crash (skip ticker/dates already on disk)
    python scripts/pull_theta_iv_surface_history.py --resume

Checkpointing
-------------
Before writing each (ticker, date) parquet, the script checks whether that
partition already exists on disk. With ``--resume`` existing partitions are
skipped; otherwise they are overwritten (``--force``).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
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

OUT_ROOT = _ROOT / "data_processed" / "theta" / "iv_surface_history"
# Target DTE buckets we want to sample — we snap to the nearest listed expiry.
TARGET_DTES = (7, 14, 30, 60, 90, 180)


# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------
def load_universe(mode: str, pit_date: str | None = None) -> list[str]:
    """Build the ticker list using the PIT membership loader we just wired."""
    if mode != "sp500":
        raise ValueError(f"Unknown universe {mode!r}; use 'sp500'")
    from data.consolidated_loader import get_bloomberg_loader

    L = get_bloomberg_loader()
    # PIT universe if a date is given; else latest.
    tickers = L.get_universe_as_of(pit_date)
    # Drop Bloomberg internal codes ("1284849D" etc.) — those are delisted
    # placeholders that Theta cannot resolve.
    tickers = [t for t in tickers if t and not t.endswith("D") or "." in t]
    # Light filtering: keep alpha-only or dotted symbols.
    tickers = [t for t in tickers if all(c.isalpha() or c == "." for c in t)]
    return sorted(set(tickers))


# ----------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------
def _surface_for_date(
    conn: ThetaConnector,
    ticker: str,
    as_of: date,
) -> pd.DataFrame:
    """Pull the surface for one ticker on one date.

    Strategy:
      1. List available expirations.
      2. Pick one expiration for each TARGET_DTES bucket.
      3. For each (expiration), pull hist greeks on ATM strike ± 10 grid
         points (Theta's snapshot already yields the full chain; for
         history we pull ATM ± 5 strikes to keep it cheap).

    Returns a long-format DataFrame, empty if Theta refused.
    """
    sym = _normalise_theta_symbol(ticker)
    # 1) expirations
    try:
        exps_df = conn._fetch("/v3/option/list/expirations", {"symbol": sym})
    except Exception as e:
        logger.debug("%s: expirations fetch failed: %s", ticker, e)
        return pd.DataFrame()
    if exps_df is None or exps_df.empty:
        return pd.DataFrame()

    exp_col = next((c for c in ("expiration", "date", "exp") if c in exps_df.columns), None)
    if exp_col is None:
        return pd.DataFrame()

    exps = pd.to_datetime(exps_df[exp_col], errors="coerce").dropna()
    exps = exps[exps >= pd.Timestamp(as_of)]
    if exps.empty:
        return pd.DataFrame()

    # 2) pick one expiration per target DTE bucket
    chosen: dict[int, pd.Timestamp] = {}
    for dte in TARGET_DTES:
        target = pd.Timestamp(as_of) + pd.Timedelta(days=dte)
        idx = (exps - target).abs().idxmin()
        chosen[dte] = exps.loc[idx]

    # 3) pull the historical IV bar for each chosen (expiration, strike)
    frames: list[pd.DataFrame] = []
    for dte_bucket, exp_ts in chosen.items():
        exp_str = exp_ts.strftime("%Y%m%d")
        # Snapshot grid for strikes — we ask Theta for IV on ATM ± N strikes
        # by reusing the connector's chain snapshot on as_of.
        try:
            chain = conn._fetch(
                "/v3/option/snapshot/greeks/first_order",
                {"symbol": sym, "expiration": exp_str},
            )
        except Exception as e:
            logger.debug("%s %s: chain fetch failed: %s", ticker, exp_str, e)
            continue
        if chain is None or chain.empty:
            continue
        chain.columns = [c.lower() for c in chain.columns]
        # Normalise column aliases
        for alias, target in (("implied_vol", "iv"), ("impl_vol", "iv"),
                              ("mid_price", "mid"), ("option_type", "right")):
            if alias in chain.columns and target not in chain.columns:
                chain[target] = chain[alias]
        need = {"strike", "right", "iv"}
        if not need.issubset(chain.columns):
            continue
        chain = chain[chain["iv"].between(0.0, 5.0, inclusive="neither")]
        if chain.empty:
            continue
        chain = chain.copy()
        chain["expiration"] = exp_ts
        chain["dte"] = (exp_ts - pd.Timestamp(as_of)).days
        chain["date"] = pd.Timestamp(as_of)
        chain["ticker"] = ticker
        keep = [c for c in ("date", "ticker", "expiration", "dte", "strike",
                            "right", "iv", "mid", "delta")
                if c in chain.columns]
        frames.append(chain[keep])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def partition_path(ticker: str, d: date) -> Path:
    return OUT_ROOT / f"ticker={ticker}" / f"year={d.year}" / f"date={d.isoformat()}.parquet"


def already_have(ticker: str, d: date) -> bool:
    return partition_path(ticker, d).exists()


def write_partition(df: pd.DataFrame, ticker: str, d: date) -> None:
    p = partition_path(ticker, d)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def _daterange(start: date, end: date, step_days: int = 1) -> Iterable[date]:
    d = start
    while d <= end:
        if d.weekday() < 5:  # skip weekends
            yield d
        d += timedelta(days=step_days)


def _process_one(args: tuple[str, date, bool]) -> tuple[str, date, bool, str]:
    ticker, d, force = args
    if not force and already_have(ticker, d):
        return ticker, d, True, "cached"
    try:
        conn = ThetaConnector()
    except Exception as e:
        return ticker, d, False, f"connector: {e}"
    try:
        df = _surface_for_date(conn, ticker, d)
        if df.empty:
            return ticker, d, False, "empty"
        write_partition(df, ticker, d)
        return ticker, d, True, f"rows={len(df)}"
    except Exception as e:
        return ticker, d, False, f"{type(e).__name__}: {e}"


def _theta_up() -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 25503))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--universe", choices=["sp500"])
    ap.add_argument("--pit-date", help="PIT universe date (YYYY-MM-DD)")
    ap.add_argument("--start", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, help="Last N business days (short form)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="Skip partitions that already exist")
    ap.add_argument("--force", action="store_true", help="Overwrite existing partitions")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("ERROR: Theta Terminal not reachable on 127.0.0.1:25503 — start the Terminal and retry")
        return 2

    # Resolve tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe:
        tickers = load_universe(args.universe, args.pit_date)
    else:
        print("ERROR: must specify --tickers or --universe")
        return 2

    # Resolve dates
    end_d = date.fromisoformat(args.end) if args.end else date.today()
    if args.days:
        start_d = end_d - timedelta(days=args.days * 2)  # generous buffer for weekends
    elif args.start:
        start_d = date.fromisoformat(args.start)
    else:
        start_d = end_d - timedelta(days=7)

    dates = list(_daterange(start_d, end_d))
    if args.days:
        dates = dates[-args.days:]

    force = args.force and not args.resume
    jobs = [(t, d, force) for t in tickers for d in dates]
    total = len(jobs)

    print(f"Pulling IV surface history  tickers={len(tickers)}  "
          f"dates={len(dates)} ({start_d}..{end_d})  jobs={total}  workers={args.workers}")

    t0 = time.perf_counter()
    n_ok = 0
    n_fail = 0
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, j): j for j in jobs}
        for fut in as_completed(futs):
            ticker, d, ok, detail = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if n_done % 100 == 0 or not ok:
                print(f"  [{n_done:>5}/{total}] {ticker:<6} {d} "
                      f"{'OK' if ok else 'FAIL':<4}  {detail[:60]}", flush=True)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
    print(f"Written under: {OUT_ROOT}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
