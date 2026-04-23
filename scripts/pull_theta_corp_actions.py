#!/usr/bin/env python3
"""
Pull stock splits + dividends from Theta — fills the empty
``data/bloomberg/sp500_corporate_actions.csv`` (it's literally 2 bytes on
disk today, so the event gate is blind to splits and specials).

Theta endpoints tried (per ticker):
    /v3/stock/history/split      symbol=<X>, start_date, end_date
    /v3/stock/history/dividend   symbol=<X>, start_date, end_date

Outputs
-------
``data_processed/corporate_actions/splits.parquet``
    ticker, ex_date, ratio, numerator, denominator

``data_processed/corporate_actions/dividends.parquet``
    ticker, declared_date, ex_date, record_date, payable_date,
    dividend_amount, dividend_frequency, dividend_type

Both write fresh copies (not append) so running twice does not duplicate
rows. The engine's event gate currently reads ``sp500_dividends.csv`` —
this script also writes a compatible view at
``data/bloomberg/sp500_dividends_theta.csv`` so the loader can consume it
alongside the Bloomberg file.

Usage
-----
    # Full history on all S&P names
    python scripts/pull_theta_corp_actions.py --universe sp500

    # One ticker, 10y back
    python scripts/pull_theta_corp_actions.py --tickers AAPL --years 10

    # Only splits (skip dividends)
    python scripts/pull_theta_corp_actions.py --tickers AAPL --skip-dividends
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.theta_connector import ThetaConnector, _normalise_theta_symbol  # noqa: E402

logger = logging.getLogger(__name__)
OUT_DIR = _ROOT / "data_processed" / "corporate_actions"
OUT_SPLITS = OUT_DIR / "splits.parquet"
OUT_DIV = OUT_DIR / "dividends.parquet"
COMPAT_DIV_CSV = _ROOT / "data" / "bloomberg" / "sp500_dividends_theta.csv"


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


def load_universe(pit_date: str | None = None) -> list[str]:
    from data.consolidated_loader import get_bloomberg_loader
    L = get_bloomberg_loader()
    u = L.get_universe_as_of(pit_date)
    return sorted({t for t in u if all(c.isalpha() or c == "." for c in t)})


def _fetch_splits(conn: ThetaConnector, ticker: str, start: date, end: date) -> pd.DataFrame:
    params = {
        "symbol": _normalise_theta_symbol(ticker),
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
    }
    for ep in ("/v3/stock/history/split",
               "/v3/stock/history/splits"):
        df = conn._fetch(ep, params)
        if df is None or df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        ex_col = next((c for c in ("ex_date", "date", "effective_date") if c in df.columns), None)
        if ex_col is None:
            continue
        df[ex_col] = pd.to_datetime(df[ex_col], errors="coerce")
        df = df.dropna(subset=[ex_col]).rename(columns={ex_col: "ex_date"})
        df["ticker"] = ticker
        keep = [c for c in ("ticker", "ex_date", "ratio", "numerator", "denominator")
                if c in df.columns]
        return df[keep]
    return pd.DataFrame()


def _fetch_dividends(conn: ThetaConnector, ticker: str, start: date, end: date) -> pd.DataFrame:
    params = {
        "symbol": _normalise_theta_symbol(ticker),
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
    }
    for ep in ("/v3/stock/history/dividend",
               "/v3/stock/history/dividends"):
        df = conn._fetch(ep, params)
        if df is None or df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        ex_col = next((c for c in ("ex_date", "ex_dividend_date", "date") if c in df.columns), None)
        if ex_col is None:
            continue
        df[ex_col] = pd.to_datetime(df[ex_col], errors="coerce")
        df = df.dropna(subset=[ex_col]).rename(columns={ex_col: "ex_date"})
        # Alias normalisation to the loader's Bloomberg schema
        aliases = {
            "amount": "dividend_amount", "cash_amount": "dividend_amount",
            "frequency": "dividend_frequency", "freq": "dividend_frequency",
            "type": "dividend_type", "div_type": "dividend_type",
            "declaration_date": "declared_date",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
        df["ticker"] = ticker
        keep = [c for c in (
            "ticker", "declared_date", "ex_date", "record_date", "payable_date",
            "dividend_amount", "dividend_frequency", "dividend_type"
        ) if c in df.columns]
        return df[keep]
    return pd.DataFrame()


def _one_ticker(ticker: str, start: date, end: date,
                do_splits: bool, do_divs: bool) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    try:
        conn = ThetaConnector()
    except Exception as e:
        return ticker, pd.DataFrame(), pd.DataFrame(), f"conn: {e}"
    sp = _fetch_splits(conn, ticker, start, end) if do_splits else pd.DataFrame()
    dv = _fetch_dividends(conn, ticker, start, end) if do_divs else pd.DataFrame()
    msg = f"splits={len(sp)} divs={len(dv)}"
    return ticker, sp, dv, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--universe", choices=["sp500"], default=None)
    ap.add_argument("--pit-date")
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-splits", action="store_true")
    ap.add_argument("--skip-dividends", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("Theta Terminal not reachable on 127.0.0.1:25503 — start it and retry.")
        return 2

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe:
        tickers = load_universe(args.pit_date)
    else:
        print("ERROR: specify --tickers or --universe sp500")
        return 2

    end_d = date.today()
    start_d = end_d - timedelta(days=int(args.years * 365))
    do_splits = not args.skip_splits
    do_divs = not args.skip_dividends
    print(f"Corp actions pull  tickers={len(tickers)}  range={start_d}..{end_d}  "
          f"splits={do_splits}  divs={do_divs}  workers={args.workers}")

    t0 = time.perf_counter()
    splits_frames: list[pd.DataFrame] = []
    div_frames: list[pd.DataFrame] = []
    n_done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one_ticker, t, start_d, end_d, do_splits, do_divs): t
                for t in tickers}
        for fut in as_completed(futs):
            ticker, sp, dv, msg = fut.result()
            n_done += 1
            if msg.startswith("conn:"):
                n_err += 1
            if not sp.empty:
                splits_frames.append(sp)
            if not dv.empty:
                div_frames.append(dv)
            if n_done % 50 == 0 or n_done == len(tickers):
                print(f"  [{n_done:>4}/{len(tickers)}] {ticker:<6}  {msg}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_sp = n_dv = 0

    if splits_frames:
        sp = pd.concat(splits_frames, ignore_index=True)
        sp["ex_date"] = pd.to_datetime(sp["ex_date"]).dt.normalize()
        sp = sp.sort_values(["ticker", "ex_date"]).drop_duplicates(
            subset=["ticker", "ex_date"], keep="last"
        )
        sp.to_parquet(OUT_SPLITS, index=False)
        n_sp = len(sp)

    if div_frames:
        dv = pd.concat(div_frames, ignore_index=True)
        dv["ex_date"] = pd.to_datetime(dv["ex_date"]).dt.normalize()
        dv = dv.sort_values(["ticker", "ex_date"]).drop_duplicates(
            subset=["ticker", "ex_date"], keep="last"
        )
        dv.to_parquet(OUT_DIV, index=False)
        n_dv = len(dv)

        # Loader-compatible CSV view.  The loader normalises by ticker column
        # so we write in Bloomberg "AAPL US Equity" format.
        compat = dv.copy()
        compat["ticker"] = compat["ticker"].astype(str).str.upper() + " US Equity"
        for col in ("declared_date", "record_date", "payable_date"):
            if col not in compat.columns:
                compat[col] = pd.NaT
        cols = ["declared_date", "ex_date", "record_date", "payable_date",
                "dividend_amount", "dividend_frequency", "dividend_type", "ticker"]
        compat[cols].to_csv(COMPAT_DIV_CSV, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {n_sp} split rows → {OUT_SPLITS}")
    print(f"Wrote {n_dv} dividend rows → {OUT_DIV}")
    if n_dv:
        print(f"Loader-compatible dividends → {COMPAT_DIV_CSV}")
    print(f"Done in {elapsed:.1f}s  |  {n_err} connector errors")
    return 0 if (n_sp + n_dv) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
