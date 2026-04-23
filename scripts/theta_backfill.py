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
from datetime import datetime, timedelta, timezone
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
                "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
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
    """Per-ticker daily OHLCV.

    Probes Theta with AAPL — if Theta returns empty (Stocks tier missing)
    we route every ticker directly to the Bloomberg CSV in ``get_ohlcv``'s
    parent class. This saves ~100s of 400s across the full universe.
    """
    tickers = _parse_tickers(args.tickers, _load_universe())
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "stocks_eod"
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Probe Theta endpoint once — AAPL is guaranteed to be a constituent
    probe = conn._fetch(
        "/v3/stock/history/eod",
        {
            "symbol": "AAPL",
            "start_date": conn._to_yyyymmdd(start),
            "end_date": conn._to_yyyymmdd(end),
        },
    )
    theta_enabled = not probe.empty
    if not theta_enabled:
        logger.info(
            "stocks-eod: Theta probe empty — bypassing Theta and "
            "routing every ticker straight to Bloomberg CSV"
        )
        from engine.data_connector import MarketDataConnector

        bloomberg_conn = MarketDataConnector(str(conn._data_dir))

    def worker(t: str) -> str:
        dst = out_dir / t
        if not args.overwrite and _write_exists(dst):
            return "skip"
        if theta_enabled:
            df = conn.get_ohlcv(t, start_date=start, end_date=end)
        else:
            df = bloomberg_conn.get_ohlcv(t, start_date=start, end_date=end)
        if df is None or df.empty:
            return "fail"
        df = df.copy()
        df["ticker"] = t
        df["source"] = "theta" if theta_enabled else "bloomberg"
        _write_df(df, dst)
        return "ok"

    return _parallel_run(tickers, worker, args.workers)


def cmd_vix_family(conn: ThetaConnector, args) -> tuple[int, int, int]:
    """Daily OHLC history for each VIX-family index.

    Tries ThetaData ``/v3/index/history/eod`` first (requires Indices tier),
    then falls back to ``CBOEAdapter`` public CSVs for each symbol that
    Theta didn't return. CBOE is free and has full EOD history for all
    VIX-family indices.
    """
    symbols = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "MOVE"]
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_dir = args.out_dir / "vix_family"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "vix_family"

    if not args.overwrite and _write_exists(merged_path):
        logger.info("vix_family already present — skip (use --overwrite to force)")
        return 0, 1, 0

    frames = []
    theta_failed: list[str] = []
    for sym in symbols:
        try:
            df = conn._fetch(
                "/v3/index/history/eod",
                {
                    "symbol": sym,
                    "start_date": conn._to_yyyymmdd(start),
                    "end_date": conn._to_yyyymmdd(end),
                },
            )
            if df.empty:
                theta_failed.append(sym)
                continue
            df.columns = [c.lower() for c in df.columns]
            date_col = next((c for c in ("date", "timestamp") if c in df.columns), None)
            if date_col is None or "close" not in df.columns:
                theta_failed.append(sym)
                continue
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
            keep = [c for c in ("open", "high", "low", "close") if c in df.columns]
            df = df[keep].copy()
            df["symbol"] = sym
            df["source"] = "theta"
            frames.append(df)
        except Exception:
            theta_failed.append(sym)

    # CBOE fallback for anything Theta didn't serve
    cboe_failed: list[str] = []
    if theta_failed:
        logger.info(
            "vix-family: falling back to CBOE for %s (no Indices tier required)",
            theta_failed,
        )
        try:
            from engine.external_data.cboe_adapter import CBOEAdapter

            cboe = CBOEAdapter()
            for sym in theta_failed:
                df = cboe.get_series(sym)
                if df.empty:
                    cboe_failed.append(sym)
                    continue
                # Clip to requested date range
                df = df.loc[pd.Timestamp(start): pd.Timestamp(end)]
                if df.empty:
                    cboe_failed.append(sym)
                    continue
                df = df.copy()
                df["symbol"] = sym
                df["source"] = "cboe"
                frames.append(df)
        except Exception:
            cboe_failed.extend(theta_failed)
            logger.exception("CBOE fallback failed")

    # Yahoo fallback for anything CBOE didn't serve (VVIX, SKEW, MOVE)
    if cboe_failed:
        logger.info(
            "vix-family: falling back to Yahoo for %s (CBOE doesn't publish these)",
            cboe_failed,
        )
        try:
            from engine.external_data.yfinance_adapter import YFinanceAdapter

            yf = YFinanceAdapter()
            yf_map = {
                "VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M",
                "VIX6M": "^VIX6M", "VVIX": "^VVIX", "SKEW": "^SKEW",
                "MOVE": "^MOVE",
            }
            # Compute period_days from start/end
            period_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            for sym in cboe_failed:
                yf_sym = yf_map.get(sym, f"^{sym}")
                df = yf.get_ohlcv(yf_sym, period_days=period_days)
                if df.empty:
                    logger.warning("vix-family: Yahoo fallback empty for %s", sym)
                    continue
                df = df.copy()
                # Keep only OHLC columns
                cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
                df = df[cols]
                df["symbol"] = sym
                df["source"] = "yahoo"
                frames.append(df)
        except Exception:
            logger.debug("Yahoo fallback failed", exc_info=True)

    if not frames:
        return 0, 0, 1
    out = pd.concat(frames)
    _write_df(out, merged_path)
    got = sorted(out["symbol"].unique().tolist())
    logger.info("vix-family: wrote %d symbols (%s), %d rows", len(got), got, len(out))
    return 1, 0, 0


def cmd_chains(conn: ThetaConnector, args) -> tuple[int, int, int]:
    tickers = _parse_tickers(args.tickers, _load_universe())
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "chains"
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

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
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

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
    """Per-ticker daily IV history.

    ThetaData ``/v3/option/history/greeks/first_order`` returns 500s on
    Options Standard tier. Fall back to slicing the Bloomberg
    ``sp500_vol_iv_full.csv`` per ticker — that file contains
    volatility_30d / volatility_60d / volatility_90d / volatility_260d
    daily for every S&P 500 constituent from 2015.

    Strategy per ticker:
      1. Try Theta once (5-ticker circuit breaker — stop after 5 fails).
      2. On fail, write Bloomberg slice.
      3. If neither source returns data, record fail.
    """
    tickers = _parse_tickers(args.tickers, _load_universe())
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "iv_history"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load the Bloomberg IV CSV once (1.36M rows, slice per ticker)
    bb_iv = pd.DataFrame()
    try:
        from engine.data_connector import normalize_ticker

        bb_path = Path(conn._data_dir) / "sp500_vol_iv_full.csv"
        if bb_path.exists():
            bb_iv = pd.read_csv(bb_path, low_memory=False)
            bb_iv.columns = [c.lower() for c in bb_iv.columns]
            if "ticker" in bb_iv.columns:
                bb_iv["ticker"] = bb_iv["ticker"].apply(normalize_ticker).str.upper()
            if "date" in bb_iv.columns:
                bb_iv["date"] = pd.to_datetime(bb_iv["date"], errors="coerce")
            # Group once for fast per-ticker access
            bb_iv = bb_iv.set_index("ticker", drop=False).sort_index()
            logger.info("iv-history: Bloomberg IV CSV loaded (%d rows)", len(bb_iv))
    except Exception:
        logger.exception("iv-history: Bloomberg CSV load failed")

    # Circuit breaker for the Theta path — if the first 5 tickers all
    # fail on Theta, skip the rest (saves 10+ minutes of futile 500s).
    theta_failed_early = {"count": 0, "disabled": False}
    theta_lock = __import__("threading").Lock()

    def worker(t: str) -> str:
        dst = out_dir / t
        if not args.overwrite and _write_exists(dst):
            return "skip"

        iv = None
        if not theta_failed_early["disabled"]:
            try:
                iv = conn._fetch_iv_history(t)
            except Exception:
                iv = None
            if iv is None or iv.empty:
                with theta_lock:
                    theta_failed_early["count"] += 1
                    if theta_failed_early["count"] >= 5 and not theta_failed_early["disabled"]:
                        theta_failed_early["disabled"] = True
                        logger.warning(
                            "iv-history: Theta endpoint disabled after 5 fails — "
                            "using Bloomberg CSV only for remaining tickers"
                        )

        if iv is not None and not iv.empty:
            df = iv.to_frame("iv_atm")
            df["ticker"] = t
            df["source"] = "theta"
            _write_df(df, dst)
            return "ok"

        # Bloomberg fallback
        if not bb_iv.empty and t in bb_iv.index:
            try:
                rows = bb_iv.loc[[t]]
                if isinstance(rows, pd.Series):
                    rows = rows.to_frame().T
                cols = [c for c in ("date", "volatility_30d") if c in rows.columns]
                if "date" in cols and "volatility_30d" in cols:
                    df = rows[["date", "volatility_30d"]].dropna().copy()
                    df = df.set_index("date").sort_index()
                    df = df.rename(columns={"volatility_30d": "iv_atm"})
                    # Normalise from percent (Bloomberg) to decimal
                    df["iv_atm"] = df["iv_atm"].astype(float) / 100.0
                    df["ticker"] = t
                    df["source"] = "bloomberg"
                    _write_df(df, dst)
                    return "ok"
            except Exception:
                logger.debug("bloomberg iv slice failed for %s", t, exc_info=True)

        return "fail"

    return _parallel_run(tickers, worker, args.workers)


def cmd_intraday(conn: ThetaConnector, args) -> tuple[int, int, int]:
    """Intraday bars. Requires Stocks tier — skip if probe fails."""
    tickers = _parse_tickers(args.tickers, DEFAULT_WATCHLIST)
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "intraday"
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    interval = args.interval or "1m"

    # Probe with the first ticker — if it fails, skip the whole step
    # rather than loop through the watchlist hitting 400s.
    probe = tickers[0] if tickers else "SPY"
    probe_df = conn.get_stock_intraday(probe, interval=interval, start_date=start, end_date=end)
    if probe_df is None or probe_df.empty:
        logger.info(
            "intraday: probe %s returned empty — likely needs Stocks tier. "
            "Skipping intraday backfill.", probe,
        )
        return 0, len(tickers), 0

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

    Probes SPY's nearest expiry first. If ``/v3/option/history/ohlc``
    isn't in the subscription, the probe returns empty and we skip the
    whole step.
    """
    tickers = _parse_tickers(args.tickers, DEFAULT_WATCHLIST)
    if args.limit:
        tickers = tickers[: args.limit]
    out_dir = args.out_dir / "option_ohlc"
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Probe the endpoint once using SPY's ATM put
    try:
        probe_exp = conn._nearest_expiration("SPY", dte_target=35)
        probe_chain = conn.get_option_chain("SPY", expiration=probe_exp) if probe_exp else None
        if probe_chain is not None and not probe_chain.empty and "delta" in probe_chain.columns:
            puts = probe_chain[probe_chain["right"] == "put"].dropna(subset=["delta", "strike"]).copy()
            if not puts.empty:
                puts["_gap"] = (puts["delta"] - (-0.50)).abs()
                atm = puts.sort_values("_gap").iloc[0]
                probe_hist = conn.get_option_ohlc_history(
                    "SPY", probe_exp, float(atm["strike"]), "put", start, end
                )
                if probe_hist is None or probe_hist.empty:
                    logger.info(
                        "option-ohlc: probe returned empty — endpoint likely needs Pro tier. "
                        "Skipping option-ohlc backfill."
                    )
                    return 0, len(tickers), 0
    except Exception:
        logger.info("option-ohlc: probe raised — skipping the whole step")
        return 0, len(tickers), 0

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


def cmd_index_options(conn: ThetaConnector, args) -> tuple[int, int, int]:
    """Pull option chains for the major cash-settled index options.

    Options Standard includes SPX/SPXW/VIX/NDX/RUT/XSP index options —
    the deepest-liquidity market in the world for SPX. Writes one
    parquet per index per snapshot, plus a full IV surface.

    SPX is the dealer-positioning gold standard: GEX on SPX is the
    market's true gamma position. VIX options tell you the distribution
    of future VIX (vol of vol).
    """
    # Index options supported in Options Standard (verify against tier docs)
    INDEX_SYMBOLS = ["SPX", "SPXW", "VIX", "NDX", "RUT", "XSP", "DJX"]
    tickers = _parse_tickers(args.tickers, INDEX_SYMBOLS)
    if args.limit:
        tickers = tickers[: args.limit]

    chain_out = args.out_dir / "index_options_chains"
    surf_out = args.out_dir / "index_options_surfaces"
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

    ok = skip = fail = 0
    for sym in tickers:
        # Current chain snapshot
        dst_chain = chain_out / f"{sym}_{today}"
        if args.overwrite or not _write_exists(dst_chain):
            try:
                chain = conn.get_option_chain(sym, dte_target=35)
                if chain is not None and not chain.empty:
                    chain = chain.copy()
                    chain["ticker"] = sym
                    chain["snapshot_date"] = today
                    _write_df(chain, dst_chain)
                    ok += 1
                else:
                    logger.warning("index-options: chain empty for %s", sym)
                    fail += 1
                    continue
            except Exception:
                logger.exception("index-options chain fail: %s", sym)
                fail += 1
                continue
        else:
            skip += 1

        # Full IV surface across expirations
        dst_surf = surf_out / f"{sym}_{today}"
        if args.overwrite or not _write_exists(dst_surf):
            try:
                surf = conn.get_iv_surface(sym, max_expirations=10)
                if surf is not None and not surf.empty:
                    surf = surf.copy()
                    surf["ticker"] = sym
                    surf["snapshot_date"] = today
                    _write_df(surf, dst_surf)
                else:
                    logger.debug("index-options: surface empty for %s", sym)
            except Exception:
                logger.exception("index-options surface fail: %s", sym)

    logger.info(
        "index-options: wrote %d chains (%d skipped, %d failed)",
        ok, skip, fail,
    )
    return ok, skip, fail


def cmd_all(conn: ThetaConnector, args) -> tuple[int, int, int]:
    total_ok = total_skip = total_fail = 0
    manifest = Manifest(args.out_dir)
    steps = [
        ("vix-family", cmd_vix_family),
        ("iv-history", cmd_iv_history),
        ("stocks-eod", cmd_stocks_eod),
        ("chains", cmd_chains),
        ("iv-surface", cmd_iv_surface),
        ("index-options", cmd_index_options),
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
    "index-options": cmd_index_options,
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
