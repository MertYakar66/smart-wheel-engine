#!/usr/bin/env python3
"""
Pull VIX futures curve (VX front month through UX8) daily history from Theta.

VIX spot + the futures curve measures contango/backwardation. Front-month
VX minus spot VIX is a first-class regime signal — contango = short-vol
regime (good for premium sellers), backwardation = stress (pull back
sizing). Today the engine has no futures data at all.

Endpoints (tried in order per expiry):
    /v3/future/list/expirations   symbol=VX
    /v3/future/history/eod        symbol=VX,  expiration=<YYYYMMDD>
    /v3/future/history/ohlc       symbol=VX,  expiration=<YYYYMMDD>, interval=1d

Output
------
``data_processed/vix_futures.parquet``  columns::

    date, expiration, dte, month_index, open, high, low, close, volume

Where ``month_index`` is the 1-based position in the curve on that date
(1 = front month, 2 = second month, etc) — use this for UX1/UX2/... views.

A wide file ``data_processed/vix_futures_wide.parquet`` materialises the
curve in UX1..UX8 columns you can directly join to a price series.

Usage
-----
    python scripts/pull_theta_vix_futures.py                     # 5y back
    python scripts/pull_theta_vix_futures.py --years 10 --months 8
    python scripts/pull_theta_vix_futures.py --incremental
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
OUT_LONG = _ROOT / "data_processed" / "vix_futures.parquet"
OUT_WIDE = _ROOT / "data_processed" / "vix_futures_wide.parquet"


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


def _list_vx_expirations(conn: ThetaConnector, min_end: date) -> list[date]:
    """Return expirations whose settlement is ≥ ``min_end``."""
    df = conn._fetch("/v3/future/list/expirations", {"symbol": "VX"})
    if df is None or df.empty:
        logger.warning("VX list/expirations returned empty — your tier may not include futures")
        return []
    df.columns = [c.lower() for c in df.columns]
    col = next((c for c in ("expiration", "date", "exp") if c in df.columns), None)
    if col is None:
        return []
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    return sorted([d.date() for d in dates if d.date() >= min_end])


def _pull_future_history(
    conn: ThetaConnector, expiration: date, start: date, end: date
) -> pd.DataFrame:
    """Pull one VX futures contract's EOD history across the window."""
    params = {
        "symbol": "VX",
        "expiration": expiration.strftime("%Y%m%d"),
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
    }
    for ep, extras in (
        ("/v3/future/history/eod", {}),
        ("/v3/future/history/ohlc", {"interval": "1d"}),
    ):
        try:
            df = conn._fetch(ep, {**params, **extras})
        except Exception as e:
            logger.debug("%s @ %s: %s", ep, expiration, e)
            continue
        if df is None or df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        ts_col = next((c for c in ("date", "timestamp", "created") if c in df.columns), None)
        if ts_col is None:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "date"})
        # Aliases
        for alias, target in (("last", "close"), ("px_close", "close")):
            if alias in df.columns and target not in df.columns:
                df[target] = df[alias]
        if "close" not in df.columns:
            continue
        keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
        out = df[keep].copy()
        out["expiration"] = pd.Timestamp(expiration)
        out["dte"] = (out["expiration"] - pd.to_datetime(out["date"])).dt.days
        return out
    return pd.DataFrame()


def _last_date_in_store() -> date | None:
    if not OUT_LONG.exists():
        return None
    try:
        df = pd.read_parquet(OUT_LONG, columns=["date"])
    except Exception:
        return None
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--months", type=int, default=8,
                    help="How many future expirations forward from each date")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--incremental", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("Theta Terminal not reachable on 127.0.0.1:25503 — start it and retry.")
        return 2

    conn = ThetaConnector()
    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = (
        date.fromisoformat(args.start) if args.start
        else end_d - timedelta(days=int(args.years * 365))
    )
    if args.incremental:
        last = _last_date_in_store()
        if last is not None:
            start_d = max(start_d, last + timedelta(days=1))
            if start_d > end_d:
                print(f"Up-to-date (last={last}).")
                return 0

    print(f"Pulling VX futures  range={start_d}..{end_d}  months_forward={args.months}")
    t0 = time.perf_counter()

    # Enumerate all expirations whose settlement is after start_d — we need
    # each contract that was trading during the window, not just current.
    exps = _list_vx_expirations(conn, min_end=start_d)
    if not exps:
        print("No VX expirations returned — your subscription likely excludes futures.")
        print("Run probe_theta_capabilities.py to confirm.")
        return 1
    print(f"Found {len(exps)} VX expirations from {exps[0]} to {exps[-1]}")

    # Only pull contracts that expire within ~1y of window end (the "active"
    # curve). Capping prevents pulling 20+ contracts per date.
    active_exps = [e for e in exps if e <= end_d + timedelta(days=365)]

    frames: list[pd.DataFrame] = []
    for exp in active_exps:
        df = _pull_future_history(conn, exp, start_d, min(end_d, exp))
        if df.empty:
            continue
        frames.append(df)
        print(f"  {exp.isoformat()}  rows={len(df):<5}  "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    if not frames:
        print("No futures data returned — subscription may not include futures history.")
        return 1

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    long["expiration"] = pd.to_datetime(long["expiration"]).dt.normalize()
    long = long.sort_values(["date", "expiration"]).reset_index(drop=True)

    # Derive month_index per (date): front month is rank 1 by dte.
    long["month_index"] = long.groupby("date")["dte"].rank(method="first").astype(int)
    long = long[long["month_index"] <= args.months]

    # Merge with existing
    if OUT_LONG.exists():
        old = pd.read_parquet(OUT_LONG)
        combined = pd.concat([old, long], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "expiration"], keep="last")
    else:
        combined = long
    combined = combined.sort_values(["date", "expiration"]).reset_index(drop=True)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_LONG, index=False)

    # Wide view: UX1..UX{months} columns
    wide = combined.pivot_table(
        index="date", columns="month_index", values="close", aggfunc="last"
    )
    wide.columns = [f"ux{int(c)}" for c in wide.columns]
    wide = wide.reset_index().sort_values("date")
    wide.to_parquet(OUT_WIDE, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {len(combined)} long rows → {OUT_LONG}")
    print(f"Wide curve ({len(wide)} dates, {wide.shape[1]-1} columns) → {OUT_WIDE}")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
