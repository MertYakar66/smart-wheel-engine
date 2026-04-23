#!/usr/bin/env python3
"""
Pull daily options-flow aggregates per ticker from Theta.

Today the EV engine uses open_interest only for slippage estimation. The
put/call volume ratio, OI changes, and unusual-volume flags that are first-
class wheel signals aren't pulled at all.

This puller queries Theta for daily aggregated volume + OI on puts and
calls and emits one row per (ticker, date) with:

    date, ticker,
    call_volume, put_volume, total_volume,
    call_oi, put_oi, total_oi,
    call_oi_change, put_oi_change,
    put_call_volume_ratio, put_call_oi_ratio,
    unusual_volume_flag           # volume > 2x trailing-20 mean

Output
------
data_processed/theta/options_flow/<TICKER>.parquet  (upserted, full history)

Prerequisites
-------------
- Theta Terminal on 127.0.0.1:25503
- Tier that exposes `/v3/option/history/volume` and `.../open_interest`
  (Standard tier covers these per the connector docstring).

Usage
-----
    # Full S&P history
    python scripts/pull_theta_options_flow.py --universe sp500 \\
        --start 2022-01-01 --end 2026-04-23 --workers 6

    # Smoke
    python scripts/pull_theta_options_flow.py --tickers AAPL MSFT --days 90
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.theta_connector import ThetaConnector, _normalise_theta_symbol  # noqa: E402

logger = logging.getLogger(__name__)
OUT_DIR = _ROOT / "data_processed" / "theta" / "options_flow"


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


def _chunk_daterange(start: date, end: date, chunk_days: int = 28):
    """Theta history endpoints cap multi-day queries — yield monthly windows."""
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def _fetch_flow(conn: ThetaConnector, ticker: str, start: date, end: date) -> pd.DataFrame:
    """Pull daily per-ticker aggregates from Theta.

    Works on the Options Standard tier as of 2026-04-23: Theta v3 does NOT
    expose bulk aggregates or ``/v3/option/history/volume``. Instead we
    iterate expirations and use ``/v3/option/history/eod`` (per-contract
    daily OHLC + volume) + ``/v3/option/history/open_interest`` (per-contract
    daily OI), then aggregate locally into ticker-level totals.

    This is more expensive per ticker (~N_expirations API calls) but it
    works without a bulk-aggregate subscription.
    """
    sym = _normalise_theta_symbol(ticker)

    # Enumerate expirations that overlap the window. Theta returns all
    # listed expirations; we keep those settling at or after ``start``.
    exp_df = conn._fetch("/v3/option/list/expirations", {"symbol": sym})
    if exp_df is None or exp_df.empty:
        return pd.DataFrame()
    exp_df.columns = [c.lower() for c in exp_df.columns]
    exp_col = next((c for c in ("expiration", "date") if c in exp_df.columns), None)
    if exp_col is None:
        return pd.DataFrame()
    exps = pd.to_datetime(exp_df[exp_col], errors="coerce").dropna()
    # For per-ticker daily flow, only the ACTIVE-DTE expirations matter
    # (front weeklies + nearest monthlies). Iterating every listed
    # expiration — which can be 50+ on SPY — multiplies API calls
    # without adding signal. Keep expirations from -7d to +120d of end.
    window_start = pd.Timestamp(start)
    window_end = pd.Timestamp(end) + pd.Timedelta(days=120)
    exps = exps[(exps >= window_start - pd.Timedelta(days=7)) & (exps <= window_end)]
    if exps.empty:
        return pd.DataFrame()
    # Cap at the 12 nearest expirations to ``end`` so very chain-rich names
    # (SPY, QQQ) don't blow up the budget.
    if len(exps) > 12:
        exps = sorted(exps, key=lambda d: abs((d - pd.Timestamp(end)).days))[:12]
        exps = pd.Series(sorted(exps))

    frames: list[pd.DataFrame] = []
    for exp in exps:
        exp_str = exp.strftime("%Y%m%d")
        for s, e in _chunk_daterange(start, end):
            params = {
                "symbol": sym,
                "expiration": exp_str,
                "start_date": s.strftime("%Y%m%d"),
                "end_date": e.strftime("%Y%m%d"),
            }
            # Daily per-contract OHLC + volume
            df_eod = conn._fetch("/v3/option/history/eod", params)
            if df_eod is not None and not df_eod.empty:
                df_eod.columns = [c.lower() for c in df_eod.columns]
                # Optional: merge in OI
                df_oi = conn._fetch("/v3/option/history/open_interest", params)
                if df_oi is not None and not df_oi.empty:
                    df_oi.columns = [c.lower() for c in df_oi.columns]
                    # Align on (date, strike, right). OI endpoint usually has
                    # a single row per contract per day; join on date keys.
                    join_keys = [c for c in ("created", "last_trade", "date",
                                             "strike", "right")
                                 if c in df_eod.columns and c in df_oi.columns]
                    if join_keys:
                        df_eod = df_eod.merge(
                            df_oi[join_keys + [c for c in ("open_interest",)
                                               if c in df_oi.columns]],
                            on=join_keys, how="left",
                        )
                frames.append(df_eod)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reduce raw per-contract bars to daily ticker-level aggregates."""
    if raw.empty:
        return raw

    # Field aliases — Theta schemas vary across endpoints.
    # Option EOD returns ``last_trade`` (session-close timestamp) and
    # ``created`` (report generation ts). Prefer last_trade because two
    # contracts written on the same ``created`` can still be from different
    # sessions if Theta backfills.
    date_col = next(
        (c for c in ("last_trade", "date", "trade_date", "bar_date", "created")
         if c in raw.columns), None
    )
    vol_col = next((c for c in ("volume", "trade_volume", "daily_volume") if c in raw.columns), None)
    oi_col = next((c for c in ("open_interest", "oi") if c in raw.columns), None)
    right_col = next((c for c in ("right", "option_type", "cp") if c in raw.columns), None)
    if date_col is None or right_col is None:
        return pd.DataFrame()

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])
    # Collapse to daily — timestamps are per-second; we want day-level buckets.
    raw[date_col] = raw[date_col].dt.normalize()
    raw[right_col] = raw[right_col].astype(str).str.lower()

    agg_map: dict = {}
    if vol_col:
        agg_map[vol_col] = "sum"
    if oi_col:
        agg_map[oi_col] = "sum"

    grp = raw.groupby([date_col, right_col]).agg(agg_map).reset_index()

    # Pivot call/put to wide columns
    out = grp.pivot(index=date_col, columns=right_col).reset_index()
    out.columns = [
        f"{a}_{b}" if b else a
        for a, b in (c if isinstance(c, tuple) else (c, "") for c in out.columns)
    ]
    out = out.rename(columns={date_col: "date"})

    # Derived fields (guard against missing call/put columns)
    def col(metric: str, side: str) -> pd.Series:
        nm = f"{metric}_{side}"
        return out[nm] if nm in out.columns else pd.Series(0, index=out.index)

    if vol_col:
        out["call_volume"] = col(vol_col, "call")
        out["put_volume"] = col(vol_col, "put")
        out["total_volume"] = out["call_volume"] + out["put_volume"]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["put_call_volume_ratio"] = out["put_volume"] / out["call_volume"]
        avg_20 = out["total_volume"].rolling(20, min_periods=5).mean()
        out["unusual_volume_flag"] = (out["total_volume"] > 2 * avg_20).astype(int)

    if oi_col:
        out["call_oi"] = col(oi_col, "call")
        out["put_oi"] = col(oi_col, "put")
        out["total_oi"] = out["call_oi"] + out["put_oi"]
        out["call_oi_change"] = out["call_oi"].diff()
        out["put_oi_change"] = out["put_oi"].diff()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["put_call_oi_ratio"] = out["put_oi"] / out["call_oi"]

    out["ticker"] = ticker
    keep = [
        c for c in ("date", "ticker", "call_volume", "put_volume", "total_volume",
                    "call_oi", "put_oi", "total_oi", "call_oi_change", "put_oi_change",
                    "put_call_volume_ratio", "put_call_oi_ratio", "unusual_volume_flag")
        if c in out.columns
    ]
    return out[keep].sort_values("date").reset_index(drop=True)


def load_universe(mode: str, pit_date: str | None = None) -> list[str]:
    if mode != "sp500":
        raise ValueError(f"Unknown universe {mode!r}")
    from data.consolidated_loader import get_bloomberg_loader

    L = get_bloomberg_loader()
    tickers = L.get_universe_as_of(pit_date)
    return sorted({t for t in tickers if all(c.isalpha() or c == "." for c in t)})


def _process_one(args: tuple[str, date, date, bool]) -> tuple[str, bool, str]:
    ticker, start, end, force = args
    out_path = OUT_DIR / f"{ticker}.parquet"
    if not force and out_path.exists():
        return ticker, True, "cached"
    try:
        conn = ThetaConnector()
        raw = _fetch_flow(conn, ticker, start, end)
        df = _summarise(raw, ticker)
    except Exception as e:
        return ticker, False, f"{type(e).__name__}: {e}"
    if df.empty:
        return ticker, False, "no data"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return ticker, True, f"rows={len(df)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--universe", choices=["sp500"])
    ap.add_argument("--pit-date")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("ERROR: Theta Terminal not reachable on 127.0.0.1:25503")
        return 2

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe:
        tickers = load_universe(args.universe, args.pit_date)
    else:
        print("ERROR: --tickers or --universe required")
        return 2

    end_d = date.fromisoformat(args.end) if args.end else date.today()
    if args.days > 0:
        start_d = end_d - timedelta(days=args.days)
    elif args.start:
        start_d = date.fromisoformat(args.start)
    else:
        start_d = end_d - timedelta(days=365)

    print(f"Options-flow pull  tickers={len(tickers)}  range={start_d}..{end_d}  workers={args.workers}")

    t0 = time.perf_counter()
    n_ok = n_fail = n_done = 0
    jobs = [(t, start_d, end_d, args.force) for t in tickers]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, j): j for j in jobs}
        for fut in as_completed(futs):
            ticker, ok, detail = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if n_done % 25 == 0 or not ok:
                print(f"  [{n_done:>4}/{len(tickers)}] {ticker:<6} "
                      f"{'OK' if ok else 'FAIL':<4}  {detail[:70]}", flush=True)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
    print(f"Written under: {OUT_DIR}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
