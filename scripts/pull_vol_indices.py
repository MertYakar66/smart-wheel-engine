#!/usr/bin/env python3
"""
Pull daily history for the volatility-index family.

The engine's regime detector and dealer-positioning modules want the full
VIX-family surface (term structure + tail-risk index + vol-of-vol), but
the Bloomberg CSVs on disk only carry the spot VIX and a tiny 3M/6M slice.
This script fills the gap.

Sources (probed in order — first one that succeeds wins per symbol):
  1. Theta Terminal ``/v3/index/history/eod``  (Indices tier)
  2. Yahoo Finance via yfinance                (free, no key needed)

Output
------
Single tidy parquet: ``data_processed/vol_indices.parquet`` with columns::

    date        datetime64[ns]
    symbol      str           # VIX, VIX9D, VIX3M, VIX6M, SKEW, VVIX, ...
    open        float
    high        float
    low         float
    close       float
    source      str           # 'theta' or 'yahoo'

Also writes a *wide* view for easy term-structure queries::

    data_processed/vol_indices_wide.parquet
    columns: date, vix_close, vix9d_close, vix3m_close, vix6m_close,
             skew_close, vvix_close, vxn_close, move_close, ovx_close, gvz_close

The existing file at ``data_processed/theta/vix_family/vix_family.parquet``
is preserved; this writer creates a superset and is the new canonical
location the engine reads from.

Usage
-----
    # 5y of history
    python scripts/pull_vol_indices.py --years 5

    # Incremental (only since last saved date)
    python scripts/pull_vol_indices.py --incremental

    # Different symbol set
    python scripts/pull_vol_indices.py --symbols VIX VIX9D SKEW VVIX --years 2

    # Force yfinance only (skip Theta even if up)
    python scripts/pull_vol_indices.py --source yahoo --years 3
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

OUT_LONG = _ROOT / "data_processed" / "vol_indices.parquet"
OUT_WIDE = _ROOT / "data_processed" / "vol_indices_wide.parquet"

# Canonical symbols we track.
DEFAULT_SYMBOLS = (
    "VIX", "VIX9D", "VIX3M", "VIX6M",
    "SKEW", "VVIX", "VXN", "MOVE", "OVX", "GVZ",
)

# Yahoo uses "^" prefix for indices.
_YF_PREFIX = {
    "VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M",
    "SKEW": "^SKEW", "VVIX": "^VVIX", "VXN": "^VXN", "MOVE": "^MOVE",
    "OVX": "^OVX", "GVZ": "^GVZ", "RVX": "^RVX",
}


# ----------------------------------------------------------------------
# Theta
# ----------------------------------------------------------------------
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


def _pull_theta(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Attempt Theta historical EOD for a volatility index. Empty frame on failure."""
    from engine.theta_connector import ThetaConnector
    conn = ThetaConnector()
    for endpoint in (
        "/v3/index/history/eod",
        "/v3/index/history/ohlc",
    ):
        try:
            df = conn._fetch(
                endpoint,
                {
                    "symbol": symbol,
                    "start_date": start.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                },
            )
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        ts_col = next((c for c in ("date", "timestamp", "created") if c in df.columns), None)
        if ts_col is None:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "date"})
        # Harmonise column names
        for alias, target in (("last", "close"), ("px_close", "close")):
            if alias in df.columns and target not in df.columns:
                df[target] = df[alias]
        keep = [c for c in ("date", "open", "high", "low", "close") if c in df.columns]
        if "close" not in keep:
            continue
        out = df[keep].copy()
        out["symbol"] = symbol
        out["source"] = "theta"
        return out
    return pd.DataFrame()


# ----------------------------------------------------------------------
# Yahoo
# ----------------------------------------------------------------------
def _pull_yahoo(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    yf_sym = _YF_PREFIX.get(symbol, f"^{symbol}")
    t = yf.Ticker(yf_sym)
    try:
        h = t.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=False,
        )
    except Exception as e:
        logger.debug("%s yfinance error: %s", symbol, e)
        return pd.DataFrame()
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.reset_index().rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"}
    )
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
    h["symbol"] = symbol
    h["source"] = "yahoo"
    return h[["date", "open", "high", "low", "close", "symbol", "source"]]


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def _last_date_in_store(symbol: str) -> date | None:
    if not OUT_LONG.exists():
        return None
    try:
        df = pd.read_parquet(OUT_LONG, columns=["date", "symbol"])
    except Exception:
        return None
    rows = df[df["symbol"] == symbol]
    if rows.empty:
        return None
    return pd.to_datetime(rows["date"]).max().date()


def pull_one(symbol: str, start: date, end: date, source: str) -> pd.DataFrame:
    attempt_theta = source in ("auto", "theta") and _theta_up()
    if attempt_theta:
        df = _pull_theta(symbol, start, end)
        if not df.empty:
            return df
    if source in ("auto", "yahoo"):
        return _pull_yahoo(symbol, start, end)
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--years", type=float, default=5.0, help="Lookback")
    ap.add_argument("--start", help="YYYY-MM-DD (overrides --years)")
    ap.add_argument("--end", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--incremental", action="store_true",
                    help="Only fetch data since last saved date per symbol")
    ap.add_argument("--source", choices=["auto", "theta", "yahoo"], default="auto")
    ap.add_argument("--out", default=str(OUT_LONG))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = (
        date.fromisoformat(args.start)
        if args.start
        else end_d - timedelta(days=int(args.years * 365))
    )

    theta_note = "UP" if _theta_up() else "DOWN"
    print(f"Pulling vol indices  source={args.source}  (theta={theta_note})  "
          f"symbols={len(args.symbols)}  range={start_d}..{end_d}")

    t0 = time.perf_counter()
    frames: list[pd.DataFrame] = []
    n_ok = n_fail = 0
    for sym in args.symbols:
        s_start = start_d
        if args.incremental:
            last = _last_date_in_store(sym)
            if last is not None:
                s_start = max(start_d, last + timedelta(days=1))
                if s_start > end_d:
                    print(f"  {sym:<7} up-to-date (last={last})")
                    continue
        df = pull_one(sym, s_start, end_d, args.source)
        if df.empty:
            n_fail += 1
            print(f"  {sym:<7} FAIL  no data from any source  range={s_start}..{end_d}")
            continue
        n_ok += 1
        src = df["source"].iloc[0]
        print(f"  {sym:<7} OK    rows={len(df):<5}  {df['date'].min().date()} to {df['date'].max().date()}  ({src})")
        frames.append(df)

    if not frames:
        print("No data fetched. Exiting.")
        return 1

    new = pd.concat(frames, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Upsert: merge with existing, drop exact (date,symbol) duplicates favouring new rows.
    if out.exists():
        old = pd.read_parquet(out)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = (
        combined.sort_values(["symbol", "date", "source"])
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    combined.to_parquet(out, index=False)

    # Wide view
    wide = combined.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    )
    wide.columns = [f"{c.lower()}_close" for c in wide.columns]
    wide = wide.reset_index().sort_values("date")
    wide.to_parquet(OUT_WIDE, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {len(combined)} rows → {out}")
    print(f"Wide view (term-structure ready) → {OUT_WIDE}")
    print(f"Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
