#!/usr/bin/env python3
"""
Pull daily OHLC history for the VIX family directly from Theta Terminal.

``pull_vol_indices.py`` already covers these via yfinance fallback, but
Theta is the authoritative CBOE feed — cleaner data, end-of-day timestamps
match exchange session close, no delisting gaps. Run this after
``probe_theta_capabilities.py`` confirms ``index/history_*`` returns 200.

Symbols tried (each can individually fail — blocked tier, new symbol, etc):

    VIX, VIX9D, VIX3M, VIX6M, SKEW, VVIX, VXN, MOVE, OVX, GVZ

Endpoints tried, in order, per symbol:
    /v3/index/history/eod
    /v3/index/history/ohlc
    /v3/index/history/price

First one that returns rows wins.

Output
------
Appends to ``data_processed/vol_indices.parquet`` with ``source="theta"``.
Downstream readers (regime detector, smoke test) already understand this
file — replacing yahoo rows with theta rows is transparent.

Usage
-----
    # Full 5y refresh from Theta
    python scripts/pull_theta_indices_history.py --years 5

    # Only what's missing since last Theta write
    python scripts/pull_theta_indices_history.py --incremental

    # A specific symbol
    python scripts/pull_theta_indices_history.py --symbols SKEW VVIX --years 2
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.theta_connector import ThetaConnector  # noqa: E402

logger = logging.getLogger(__name__)
OUT_LONG = _ROOT / "data_processed" / "vol_indices.parquet"
OUT_WIDE = _ROOT / "data_processed" / "vol_indices_wide.parquet"

DEFAULT_SYMBOLS = (
    "VIX", "VIX9D", "VIX3M", "VIX6M",
    "SKEW", "VVIX", "VXN", "MOVE", "OVX", "GVZ",
)

ENDPOINTS = (
    "/v3/index/history/eod",
    "/v3/index/history/ohlc",
    "/v3/index/history/price",
)


def _theta_up(host: str = "127.0.0.1", port: int = 25503) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _pull(conn: ThetaConnector, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Try each endpoint; return the first non-empty normalised frame."""
    params = {
        "symbol": symbol,
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
    }
    for ep in ENDPOINTS:
        try:
            df = conn._fetch(ep, {**params, "interval": "1d"} if "ohlc" in ep else params)
        except Exception as e:
            logger.debug("%s %s: %s", symbol, ep, e)
            continue
        if df is None or df.empty:
            continue

        df.columns = [c.lower() for c in df.columns]
        ts_col = next((c for c in ("date", "timestamp", "created") if c in df.columns), None)
        if ts_col is None:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "date"})

        # Harmonise any aliases
        for alias, target in (("last", "close"), ("price", "close"), ("value", "close"),
                              ("px_close", "close")):
            if alias in df.columns and target not in df.columns:
                df[target] = df[alias]
        if "close" not in df.columns:
            continue
        keep = [c for c in ("date", "open", "high", "low", "close") if c in df.columns]
        out = df[keep].copy()
        out["symbol"] = symbol
        out["source"] = "theta"
        return out

    return pd.DataFrame()


def _last_theta_date(symbol: str) -> date | None:
    if not OUT_LONG.exists():
        return None
    try:
        df = pd.read_parquet(OUT_LONG, columns=["date", "symbol", "source"])
    except Exception:
        return None
    rows = df[(df["symbol"] == symbol) & (df["source"] == "theta")]
    if rows.empty:
        return None
    return pd.to_datetime(rows["date"]).max().date()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--incremental", action="store_true",
                    help="Only fetch data newer than last Theta write per symbol")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("Theta Terminal not reachable on 127.0.0.1:25503 — start it and retry.")
        return 2

    conn = ThetaConnector()
    end_d = date.fromisoformat(args.end) if args.end else date.today()
    default_start = end_d - timedelta(days=int(args.years * 365))
    if args.start:
        default_start = date.fromisoformat(args.start)

    print(f"Theta indices pull  symbols={len(args.symbols)}  end={end_d}")
    t0 = time.perf_counter()
    frames: list[pd.DataFrame] = []
    n_ok = n_fail = 0
    for sym in args.symbols:
        s_start = default_start
        if args.incremental:
            last = _last_theta_date(sym)
            if last is not None:
                s_start = max(default_start, last + timedelta(days=1))
                if s_start > end_d:
                    print(f"  {sym:<7} up-to-date (last theta={last})")
                    continue
        df = _pull(conn, sym, s_start, end_d)
        if df.empty:
            n_fail += 1
            print(f"  {sym:<7} FAIL  no data from any endpoint  range={s_start}..{end_d}")
            continue
        n_ok += 1
        print(f"  {sym:<7} OK    rows={len(df):<5}  "
              f"{df['date'].min().date()} to {df['date'].max().date()}")
        frames.append(df)

    if not frames:
        print("Nothing fetched. Your tier may not include indices history — "
              "run probe_theta_capabilities.py to confirm.")
        return 1

    new = pd.concat(frames, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"]).dt.normalize()

    # Merge with existing: Theta rows win over yahoo on the same (date, symbol).
    if OUT_LONG.exists():
        old = pd.read_parquet(OUT_LONG)
        old["date"] = pd.to_datetime(old["date"]).dt.normalize()
        # Rank sources so Theta beats Yahoo during dedup
        src_rank = {"theta": 2, "yahoo": 1}
        combined = pd.concat([old, new], ignore_index=True)
        combined["_rank"] = combined["source"].map(src_rank).fillna(0)
        combined = (
            combined.sort_values(["symbol", "date", "_rank"])
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .drop(columns="_rank")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )
    else:
        combined = new

    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_LONG, index=False)

    # Rebuild wide view
    wide = combined.pivot_table(index="date", columns="symbol",
                                values="close", aggfunc="last")
    wide.columns = [f"{c.lower()}_close" for c in wide.columns]
    wide = wide.reset_index().sort_values("date")
    wide.to_parquet(OUT_WIDE, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {len(combined)} total rows → {OUT_LONG}")
    print(f"Wide view → {OUT_WIDE}")
    print(f"Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
