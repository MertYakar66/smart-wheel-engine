"""
ThetaData bulk backfill — pulls and persists all datasets the engine needs.

This script runs on the machine where the ThetaTerminal is running
(127.0.0.1:25503). All data is written to ``data_processed/theta/`` as
parquet (preferred) or CSV (fallback when pyarrow isn't installed).

Subcommands
-----------
    python -m scripts.theta_backfill stocks-eod
        Daily OHLCV for every SP500 constituent.

    python -m scripts.theta_backfill vix-family
        Daily history for VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW, MOVE.

    python -m scripts.theta_backfill chains
        Current option-chain snapshots (greeks + quote + OI) for the
        SP500 universe (or --tickers list), one parquet per expiry per
        ticker.

    python -m scripts.theta_backfill iv-surface
        Full IV surface across expiries — stored as one long-format
        parquet per ticker.

    python -m scripts.theta_backfill iv-history
        One-year daily ATM-IV series per ticker (35-DTE ATM put).

    python -m scripts.theta_backfill intraday
        1-minute intraday bars for the watchlist (last 30 calendar days).

    python -m scripts.theta_backfill option-ohlc
        Historical per-contract OHLC for ATM + 25Δ wings, 90 days back.

    python -m scripts.theta_backfill all
        Run every backfill in order.

Flags
-----
    --tickers    Comma-separated override (defaults to SP500)
    --limit      Cap on tickers processed (default: no cap)
    --out-dir    Override output directory (default data_processed/theta)
    --overwrite  Re-fetch even if the target file already exists
    --start      Start date (YYYY-MM-DD) for history pulls (default: 365 days ago)
    --end        End date (YYYY-MM-DD) for history pulls (default: today)
    --workers    Parallel workers (respects Terminal's 4-concurrent cap; default 3)
    --quiet      Minimal logging

Idempotency
-----------
Every subcommand skips tickers whose output file already exists, unless
``--overwrite`` is passed. Running nightly is safe.

Rate limiting
-------------
The ThetaConnector enforces a 4-concurrent semaphore. We use 3 parallel
workers by default to leave headroom for other callers (engine, tests).

Written to
----------
    data_processed/theta/
    ├── stocks_eod/                 per-ticker daily OHLCV parquet
    ├── vix_family/vix_family.parquet
    ├── chains/{ticker}_{YYYYMMDD}.parquet
    ├── iv_surface/{ticker}.parquet
    ├── iv_history/{ticker}.parquet
    ├── intraday/{ticker}.parquet
    ├── option_ohlc/{ticker}_{expiry}_{strike}_{right}.parquet
    └── _manifest.json              (index of what has been pulled)

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure we can import the engine package when run as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.theta_connector import ThetaConnector  # noqa: E402

logger = logging.getLogger("theta_backfill")

# Tickers that are worth intraday + full-surface pulls (can override via --tickers)
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "JPM", "BAC", "GS", "MS", "WFC",
    "XOM", "CVX", "COP",
    "JNJ", "PFE", "UNH", "LLY",
]

_DEFAULT_OUT_DIR = Path("data_processed/theta")
_CONSTITUENTS_CSV = Path("data_raw/sp500_constituents_current.csv")
_PARQUET_OK: bool | None = None


def _can_parquet() -> bool:
    global _PARQUET_OK
    if _PARQUET_OK is None:
        try:
            import pyarrow  # noqa: F401

            _PARQUET_OK = True
        except Exception:
            try:
                import fastparquet  # noqa: F401

                _PARQUET_OK = True
            except Exception:
                _PARQUET_OK = False
    return _PARQUET_OK


def _write_df(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame as parquet if possible, else CSV. Returns final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _can_parquet():
        p = path.with_suffix(".parquet")
        df.to_parquet(p, index=True)
        return p
    p = path.with_suffix(".csv")
    df.to_csv(p, index=True)
    return p


def _write_exists(path: Path) -> bool:
    for ext in (".parquet", ".csv"):
        if path.with_suffix(ext).exists():
            return True
    return False


def _load_universe() -> list[str]:
    """Load the SP500 ticker universe from the CSV."""
    if not _CONSTITUENTS_CSV.exists():
        logger.warning("No constituents CSV at %s — using watchlist only", _CONSTITUENTS_CSV)
        return list(DEFAULT_WATCHLIST)
    df = pd.read_csv(_CONSTITUENTS_CSV)
    tickers = df["ticker"].astype(str).str.upper().tolist()
    return tickers


def _parse_tickers(arg: str | None, default: list[str]) -> list[str]:
    if not arg:
        return default
    return [t.strip().upper() for t in arg.split(",") if t.strip()]


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
class Manifest:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "_manifest.json"
        self.data: dict = {"runs": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                pass

    def record(self, subcommand: str, n_ok: int, n_skipped: int, n_failed: int, details: dict | None = None) -> None:
        self.data.setdefault("runs", []).append(
            {
                "subcommand": subcommand,
                "ran_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "ok": n_ok,
                "skipped": n_skipped,
                "failed": n_failed,
                "details": details or {},
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


# ----------------------------------------------------------------------
# Worker helpers
# ----------------------------------------------------------------------
def _parallel_run(tickers: list[str], worker, workers: int) -> tuple[int, int, int]:
    """Run ``worker(ticker)`` in parallel, returning (ok, skipped, failed) counts.

    Each worker should return one of 'ok', 'skip', 'fail'.
    """
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                status = fut.result()
            except Exception:
                logger.exception("worker raised on %s", t)
                n_fail += 1
                continue
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
            if i % 25 == 0 or i == len(tickers):
                logger.info("progress: %d/%d  ok=%d skip=%d fail=%d",
                            i, len(tickers), n_ok, n_skip, n_fail)
    return n_ok, n_skip, n_fail


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------
def cmd_stocks_eod(conn: ThetaConnector, args) -> tuple[int, int, int]:
    tickers = _parse_tickers(args.tickers, _load_universe())
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "stocks_eod"
    start = args.start or (datetime.utcnow() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    def worker(t: str) -> str:
        dst = out_dir / t
        if not args.overwrite and _write_exists(dst):
            return "skip"
        df = conn.get_ohlcv(t, start_date=start, end_date=end)
        if df is None or df.empty:
            return "fail"
        df["ticker"] = t
        _write_df(df, dst)
        return "ok"

    return _parallel_run(tickers, worker, args.workers)


def cmd_vix_family(conn: ThetaConnector, args) -> tuple[int, int, int]:
    """Daily OHLC history for each VIX-family index.

    We use get_ohlcv with symbol type 'index' via the underlying fetch.
    ThetaData v3 exposes historical EOD for indices at the same path as
    stocks — the Terminal treats VIX/VIX9D/VIX3M as index symbols.
    """
    symbols = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "MOVE"]
    start = args.start or (datetime.utcnow() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    out_dir = args.out_dir / "vix_family"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "vix_family"

    if not args.overwrite and _write_exists(merged_path):
        logger.info("vix_family already present — skip (use --overwrite to force)")
        return 0, 1, 0

    frames = []
    for sym in symbols:
        try:
            df = conn._fetch(  # index history uses v3/index/history/eod
                "/v3/index/history/eod",
                {
                    "symbol": sym,
                    "start_date": sym and conn._to_yyyymmdd(start),
                    "end_date": conn._to_yyyymmdd(end),
                },
            )
            if df.empty:
                logger.warning("vix-family: %s returned empty", sym)
                continue
            df.columns = [c.lower() for c in df.columns]
            date_col = next((c for c in ("date", "timestamp") if c in df.columns), None)
            if date_col is None or "close" not in df.columns:
                continue
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
            keep = [c for c in ("open", "high", "low", "close") if c in df.columns]
            df = df[keep].copy()
            df["symbol"] = sym
            frames.append(df)
        except Exception:
            logger.exception("vix-family fetch failed for %s", sym)

    if not frames:
        return 0, 0, 1
    out = pd.concat(frames)
    _write_df(out, merged_path)
    return 1, 0, 0


def cmd_chains(conn: ThetaConnector, args) -> tuple[int, int, int]:
    tickers = _parse_tickers(args.tickers, _load_universe())
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "chains"
    today = datetime.utcnow().strftime("%Y%m%d")

    def worker(t: str) -> str:
        dst = out_dir / f"{t}_{today}"
        if not args.overwrite and _write_exists(dst):
            return "skip"
        df = conn.get_option_chain(t, dte_target=35)
        if df is None or df.empty:
            return "fail"
        df = df.copy()
        df["ticker"] = t
        df["snapshot_date"] = today
        _write_df(df, dst)
        return "ok"

    return _parallel_run(tickers, worker, args.workers)


def cmd_iv_surface(conn: ThetaConnector, args) -> tuple[int, int, int]:
    tickers = _parse_tickers(args.tickers, DEFAULT_WATCHLIST)
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "iv_surface"
    today = datetime.utcnow().strftime("%Y%m%d")

    def worker(t: str) -> str:
        dst = out_dir / f"{t}_{today}"
        if not args.overwrite and _write_exists(dst):
            return "skip"
        df = conn.get_iv_surface(t, max_expirations=8)
        if df is None or df.empty:
            return "fail"
        df = df.copy()
        df["ticker"] = t
        df["snapshot_date"] = today
        _write_df(df, dst)
        return "ok"

    return _parallel_run(tickers, worker, args.workers)


def cmd_iv_history(conn: ThetaConnector, args) -> tuple[int, int, int]:
    tickers = _parse_tickers(args.tickers, _load_universe())
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "iv_history"

    def worker(t: str) -> str:
        dst = out_dir / t
        if not args.overwrite and _write_exists(dst):
            return "skip"
        iv = conn._fetch_iv_history(t)
        if iv is None or iv.empty:
            return "fail"
        df = iv.to_frame("iv_atm")
        df["ticker"] = t
        _write_df(df, dst)
        return "ok"

    return _parallel_run(tickers, worker, args.workers)


def cmd_intraday(conn: ThetaConnector, args) -> tuple[int, int, int]:
    tickers = _parse_tickers(args.tickers, DEFAULT_WATCHLIST)
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "intraday"
    start = args.start or (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")
    interval = args.interval or "1m"

    def worker(t: str) -> str:
        dst = out_dir / f"{t}_{interval}"
        if not args.overwrite and _write_exists(dst):
            return "skip"
        df = conn.get_stock_intraday(t, interval=interval, start_date=start, end_date=end)
        if df is None or df.empty:
            return "fail"
        df = df.copy()
        df["ticker"] = t
        df["interval"] = interval
        _write_df(df, dst)
        return "ok"

    return _parallel_run(tickers, worker, args.workers)


def cmd_option_ohlc(conn: ThetaConnector, args) -> tuple[int, int, int]:
    """Pull per-contract EOD OHLC for ATM put + 25Δ put + 25Δ call on
    each watchlist ticker's nearest 35-DTE expiry.
    """
    tickers = _parse_tickers(args.tickers, DEFAULT_WATCHLIST)
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "option_ohlc"
    start = args.start or (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    def pick(df: pd.DataFrame, target_delta: float, right: str) -> dict | None:
        d = df[df["right"] == right].dropna(subset=["delta", "strike"]).copy()
        if d.empty:
            return None
        d["_gap"] = (d["delta"] - target_delta).abs()
        row = d.sort_values("_gap").iloc[0]
        return {
            "strike": float(row["strike"]),
            "right": right,
            "delta": float(row["delta"]),
        }

    def worker(t: str) -> str:
        exp = conn._nearest_expiration(t, dte_target=35)
        if exp is None:
            return "fail"
        chain = conn.get_option_chain(t, expiration=exp)
        if chain.empty:
            return "fail"
        targets = [
            pick(chain, -0.25, "put"),
            pick(chain, -0.50, "put"),  # ATM
            pick(chain, 0.25, "call"),
        ]
        n_ok = 0
        for tg in targets:
            if tg is None:
                continue
            tag = f"{t}_{exp}_{int(tg['strike'])}_{tg['right']}"
            dst = out_dir / tag
            if not args.overwrite and _write_exists(dst):
                continue
            try:
                hist = conn.get_option_ohlc_history(
                    t, exp, tg["strike"], tg["right"], start, end
                )
                if hist is None or hist.empty:
                    continue
                hist = hist.copy()
                hist["ticker"] = t
                hist["expiration"] = exp
                hist["strike"] = tg["strike"]
                hist["right"] = tg["right"]
                _write_df(hist, dst)
                n_ok += 1
            except Exception:
                logger.exception("option-ohlc fail: %s strike=%s", t, tg["strike"])
        return "ok" if n_ok else "fail"

    return _parallel_run(tickers, worker, args.workers)


def cmd_all(conn: ThetaConnector, args) -> tuple[int, int, int]:
    total_ok = total_skip = total_fail = 0
    manifest = Manifest(args.out_dir)
    steps = [
        ("vix-family", cmd_vix_family),
        ("iv-history", cmd_iv_history),
        ("stocks-eod", cmd_stocks_eod),
        ("chains", cmd_chains),
        ("iv-surface", cmd_iv_surface),
        ("intraday", cmd_intraday),
        ("option-ohlc", cmd_option_ohlc),
    ]
    for name, fn in steps:
        logger.info("=== running %s ===", name)
        t0 = time.time()
        ok, skip, fail = fn(conn, args)
        manifest.record(name, ok, skip, fail, {"elapsed_s": round(time.time() - t0, 1)})
        total_ok += ok
        total_skip += skip
        total_fail += fail
    return total_ok, total_skip, total_fail


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
_SUBCOMMANDS = {
    "stocks-eod": cmd_stocks_eod,
    "vix-family": cmd_vix_family,
    "chains": cmd_chains,
    "iv-surface": cmd_iv_surface,
    "iv-history": cmd_iv_history,
    "intraday": cmd_intraday,
    "option-ohlc": cmd_option_ohlc,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ThetaData bulk backfill")
    parser.add_argument("subcommand", choices=list(_SUBCOMMANDS))
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker override")
    parser.add_argument("--limit", type=int, default=None, help="Cap on tickers")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Force re-fetch")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (<=4)")
    parser.add_argument("--interval", default="1m", help="Intraday interval (1m/5m/15m)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = ThetaConnector()
    if not conn.is_terminal_alive():
        print(
            "ERROR: ThetaTerminal not reachable at 127.0.0.1:25503.\n"
            "Start it first with:\n"
            "    java -jar ThetaTerminalv3.jar <email> <password>\n",
            file=sys.stderr,
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(args.out_dir)

    t0 = time.time()
    ok, skip, fail = _SUBCOMMANDS[args.subcommand](conn, args)
    elapsed = time.time() - t0
    manifest.record(args.subcommand, ok, skip, fail, {"elapsed_s": round(elapsed, 1)})

    logger.info(
        "done: ok=%d skipped=%d failed=%d  elapsed=%.1fs  out=%s",
        ok, skip, fail, elapsed, args.out_dir,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
