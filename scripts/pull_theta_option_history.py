#!/usr/bin/env python3
"""
Pull per-(ticker, expiration) historical option EOD OHLC + OI from Theta.

Two strike-selection modes:

**``--all-strikes`` (primary for the larder).** One BULK call per expiration —
``/v3/option/history/eod`` with (symbol, expiration) but NO strike/right — returns
the ENTIRE chain (all strikes, both rights) in a single request (~100x fewer calls
than per-contract; ~88-235 strikes verified). Needs no spot, so it also covers
ranges with no spot source. This is what makes a full-universe / full-depth pull
feasible at the 4-connection cap.

**Banded (default).** Per-(strike, right) calls within ±``strike_band_pct`` of
spot at expiration, with spot looked up from Bloomberg OHLCV (``sp500_ohlcv.csv``,
2018+) — falls back to Theta stock EOD if the connector tier allows it. Used when
you want a small near-the-money grid and have a spot source.

Both paths fetch the contract's EOD over a ``--lookback-days`` window before expiry
(default 210 = full life; 90 = the 0-90 DTE cross-section a 30-45 DTE wheel + skew
fit read), clamped to the 2016-01-01 STANDARD-tier history floor (a start_date
below the floor makes the EOD endpoint return EMPTY for the whole range — see
``_THETA_HISTORY_FLOOR``). Partitions are written atomically (tmp + rename) so a
crash mid-write can't leave a partial file that ``--resume`` skips as "done".

Output
------
``--out-dir`` (default ``data_processed/theta/option_history/``)
    ticker=<SYM>/expiration=<YYYYMMDD>/data.parquet
        columns: ticker, expiration, strike, right, created/last_trade, open,
        high, low, close, volume, count, bid, ask, [open_interest if --include-oi]

Cadence: ``monthly`` (3rd Friday), ``weekly`` (all Friday expirations =
weeklies+monthlies, drops Mon-Thu 0DTE), or ``all`` (every listed expiration).

Quick start
-----------
    # All-strikes larder: top liquid names, 2018+, 90d window, with OI, resumable
    python scripts/pull_theta_option_history.py --tickers AAPL,MSFT \
        --start 2018-01-01 --cadence all --all-strikes --include-oi \
        --lookback-days 90 --workers 4 --resume

Tier / timeout note
-------------------
``/v3/option/history/eod`` and ``/v3/option/history/open_interest`` work on
OPTION.STANDARD both per-contract AND in bulk (no strike → whole chain). Bulk
calls return large responses (45-110s for liquid names, past the connector's 30s
default), so this puller raises the connector read timeout via ``--read-timeout``
(default 180) — without it, mega-cap bulk calls time out and silently drop.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import socket
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace", write_through=True)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.data_connector import normalize_ticker  # noqa: E402
from engine.theta_connector import ThetaConnector, _normalise_theta_symbol  # noqa: E402

logger = logging.getLogger(__name__)
OUT_ROOT = _ROOT / "data_processed" / "theta" / "option_history"
_THETA_HISTORY_FLOOR = pd.Timestamp("2016-01-01")  # STANDARD tier option-history start
_CONSTITUENTS_CSV = _ROOT / "data_raw" / "sp500_constituents_current.csv"
_BLOOMBERG_OHLCV = _ROOT / "data" / "bloomberg" / "sp500_ohlcv.csv"


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


def _load_universe() -> list[str]:
    if not _CONSTITUENTS_CSV.exists():
        raise SystemExit(f"S&P constituents not found at {_CONSTITUENTS_CSV}")
    df = pd.read_csv(_CONSTITUENTS_CSV)
    return df["ticker"].astype(str).str.upper().tolist()


def _load_bloomberg_spot() -> pd.DataFrame:
    """Per-ticker daily close from Bloomberg OHLCV. Returns long-format
    DataFrame keyed by (ticker, date) with column 'close'.

    Bloomberg ticker format is "AAPL UW Equity" / "A UN Equity" — strips
    the exchange suffix via engine.data_connector.normalize_ticker.
    """
    if not _BLOOMBERG_OHLCV.exists():
        raise SystemExit(f"Bloomberg OHLCV not found at {_BLOOMBERG_OHLCV}")
    df = pd.read_csv(_BLOOMBERG_OHLCV, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    if "ticker" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        raise SystemExit("Bloomberg OHLCV missing ticker/date/close columns")
    df["ticker"] = df["ticker"].astype(str).apply(normalize_ticker).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values(["ticker", "date"])
    return df[["ticker", "date", "close"]]


def _theta_stock_close(conn: ThetaConnector, ticker: str, target: pd.Timestamp) -> float | None:
    """Fallback spot source: Theta /v3/stock/history/eod for the week up to
    target. Returns the close on (or just before) target. Used when
    Bloomberg lookup misses (recent expirations or pre-Bloomberg range)."""
    try:
        start = (target - pd.Timedelta(days=10)).strftime("%Y%m%d")
        end = target.strftime("%Y%m%d")
        df = conn._fetch(
            "/v3/stock/history/eod",
            {"symbol": _normalise_theta_symbol(ticker), "start_date": start, "end_date": end},
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return None
    return float(df["close"].iloc[-1])


def _is_third_friday(d: pd.Timestamp) -> bool:
    """Standard monthly equity-option expiration: 3rd Friday of the month."""
    return d.weekday() == 4 and 15 <= d.day <= 21


def _list_expirations(conn: ThetaConnector, ticker: str) -> pd.DatetimeIndex:
    df = conn._fetch("/v3/option/list/expirations", {"symbol": _normalise_theta_symbol(ticker)})
    if df is None or df.empty:
        return pd.DatetimeIndex([])
    df.columns = [c.lower() for c in df.columns]
    col = next((c for c in ("expiration", "date") if c in df.columns), None)
    if col is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(
        pd.to_datetime(df[col], errors="coerce").dropna().unique()
    ).sort_values()


def _list_strikes(conn: ThetaConnector, ticker: str, expiration: str) -> list[float]:
    df = conn._fetch(
        "/v3/option/list/strikes",
        {"symbol": _normalise_theta_symbol(ticker), "expiration": expiration},
    )
    if df is None or df.empty:
        return []
    df.columns = [c.lower() for c in df.columns]
    if "strike" not in df.columns:
        return []
    return sorted({float(s) for s in df["strike"].dropna().tolist()})


def _pick_strike_band(
    strikes: list[float], spot: float, band_pct: float, max_strikes: int
) -> list[float]:
    """Keep strikes within ±band_pct of spot, capped at max_strikes total
    (evenly distributed around spot)."""
    if not strikes or spot <= 0 or pd.isna(spot):
        return []
    lo, hi = spot * (1 - band_pct / 100.0), spot * (1 + band_pct / 100.0)
    band = [s for s in strikes if lo <= s <= hi]
    if len(band) <= max_strikes:
        return band
    # Sort by distance from spot, keep nearest max_strikes
    band_sorted = sorted(band, key=lambda s: abs(s - spot))
    return sorted(band_sorted[:max_strikes])


def _fetch_contract_history(
    conn: ThetaConnector,
    ticker: str,
    expiration: str,
    strike: float,
    right: str,
    include_oi: bool,
    lookback_days: int = 210,
) -> pd.DataFrame:
    """Return per-contract daily OHLC (+ optional OI) for the full contract
    lifetime. Empty DataFrame if Theta returns no data.
    """
    base_params: dict = {
        "symbol": _normalise_theta_symbol(ticker),
        "expiration": expiration,
        "strike": strike,
        "right": right.lower(),
    }
    # EOD endpoint requires explicit start/end. Use the expiration date as
    # the end, and 6 months prior as the start (covers the full life of
    # standard monthly contracts; weeklies and LEAPs both fit).
    exp_dt = pd.Timestamp(expiration)
    start_dt = exp_dt - pd.Timedelta(days=lookback_days)
    # Clamp to the Theta STANDARD option-history floor. A start_date before the
    # subscription's data window makes the EOD endpoint return EMPTY for the WHOLE
    # range (not a partial), which silently zeroed every early-2016 contract.
    if start_dt < _THETA_HISTORY_FLOOR:
        start_dt = _THETA_HISTORY_FLOOR
    start = start_dt.strftime("%Y%m%d")
    end = (exp_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")
    eod_params = {**base_params, "start_date": start, "end_date": end}
    df_eod = conn._fetch("/v3/option/history/eod", eod_params)
    if df_eod is None or df_eod.empty:
        return pd.DataFrame()
    df_eod.columns = [c.lower() for c in df_eod.columns]
    if include_oi:
        df_oi = conn._fetch("/v3/option/history/open_interest", eod_params)
        if df_oi is not None and not df_oi.empty:
            df_oi.columns = [c.lower() for c in df_oi.columns]
            # OI rows have a 'timestamp' or 'created' column — derive a
            # date column for joining with EOD.
            ts_col_oi = next(
                (c for c in ("timestamp", "created", "date") if c in df_oi.columns), None
            )
            ts_col_eod = next(
                (c for c in ("created", "last_trade", "date", "timestamp") if c in df_eod.columns),
                None,
            )
            if ts_col_oi and ts_col_eod:
                df_oi["_d"] = pd.to_datetime(df_oi[ts_col_oi], errors="coerce").dt.normalize()
                df_eod["_d"] = pd.to_datetime(df_eod[ts_col_eod], errors="coerce").dt.normalize()
                if "open_interest" in df_oi.columns:
                    df_eod = df_eod.merge(
                        df_oi[["_d", "open_interest"]].drop_duplicates("_d"),
                        on="_d",
                        how="left",
                    )
                df_eod = df_eod.drop(columns=["_d"], errors="ignore")
    return df_eod


def _fetch_expiration_bulk(
    conn: ThetaConnector,
    ticker: str,
    expiration: str,
    include_oi: bool,
    lookback_days: int = 210,
) -> pd.DataFrame:
    """ALL strikes & rights for one expiration in a SINGLE call (no strike
    filter) — ~100x fewer requests than per-contract. Used in --all-strikes
    mode, which is what makes a full-universe / full-depth pull feasible at the
    4-connection cap. The bulk EOD response already carries strike+right
    columns; OI is merged on (strike, right, date)."""
    exp_dt = pd.Timestamp(expiration)
    start_dt = exp_dt - pd.Timedelta(days=lookback_days)
    if start_dt < _THETA_HISTORY_FLOOR:
        start_dt = _THETA_HISTORY_FLOOR
    params = {
        "symbol": _normalise_theta_symbol(ticker),
        "expiration": expiration,
        "start_date": start_dt.strftime("%Y%m%d"),
        "end_date": (exp_dt + pd.Timedelta(days=1)).strftime("%Y%m%d"),
    }
    df_eod = conn._fetch("/v3/option/history/eod", params)
    if df_eod is None or df_eod.empty:
        return pd.DataFrame()
    df_eod.columns = [c.lower() for c in df_eod.columns]
    if include_oi and {"strike", "right"}.issubset(df_eod.columns):
        df_oi = conn._fetch("/v3/option/history/open_interest", params)
        if df_oi is not None and not df_oi.empty:
            df_oi.columns = [c.lower() for c in df_oi.columns]
            ts_oi = next((c for c in ("timestamp", "created", "date") if c in df_oi.columns), None)
            ts_eod = next(
                (c for c in ("created", "last_trade", "date", "timestamp") if c in df_eod.columns),
                None,
            )
            if (
                ts_oi
                and ts_eod
                and "open_interest" in df_oi.columns
                and {"strike", "right"}.issubset(df_oi.columns)
            ):
                df_oi["_d"] = pd.to_datetime(df_oi[ts_oi], errors="coerce").dt.normalize()
                df_eod["_d"] = pd.to_datetime(df_eod[ts_eod], errors="coerce").dt.normalize()
                df_eod = df_eod.merge(
                    df_oi[["strike", "right", "_d", "open_interest"]].drop_duplicates(
                        ["strike", "right", "_d"]
                    ),
                    on=["strike", "right", "_d"],
                    how="left",
                )
                df_eod = df_eod.drop(columns=["_d"], errors="ignore")
    return df_eod


def _partition_exists(ticker: str, expiration: str) -> bool:
    p = OUT_ROOT / f"ticker={ticker}" / f"expiration={expiration}" / "data.parquet"
    return p.exists() and p.stat().st_size > 0


def _tmp_partition_path(outdir: Path) -> Path:
    """Per-writer-unique tmp path for the atomic partition write.

    A FIXED ``data.parquet.tmp`` name races when two writers target the same
    partition dir at once — e.g. two pulls sharing one larder, or (defensively)
    a retried unit re-entering before the first finished. They'd clobber each
    other's in-flight tmp and one ``replace`` could rename a half-written file
    into place. PID + uuid4 makes every writer's tmp collision-free.
    """
    return outdir / f"data.parquet.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _write_partition(ticker: str, expiration: str, frames: list[pd.DataFrame]) -> tuple[int, int]:
    """Concatenate per-contract frames and write one parquet for the
    (ticker, expiration) partition. Returns (rows, contracts)."""
    if not frames:
        return 0, 0
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = ticker
    df["expiration"] = expiration
    outdir = OUT_ROOT / f"ticker={ticker}" / f"expiration={expiration}"
    outdir.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash mid-write must NOT leave a partial data.parquet that
    # the resume check (file-exists + size>0) would skip as "done". Write to a
    # per-writer-unique tmp, then atomically rename into place; the finally clears
    # the tmp if the rename never happened so unique names can't accumulate.
    tmp = _tmp_partition_path(outdir)
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(outdir / "data.parquet")
    finally:
        tmp.unlink(missing_ok=True)
    contracts = df.groupby(["strike", "right"]).ngroups if not df.empty else 0
    return len(df), contracts


def _one_ticker(
    ticker: str,
    conn: ThetaConnector,
    spot_lookup: pd.DataFrame,
    start_date: date,
    end_date: date,
    band_pct: float,
    max_strikes: int,
    include_oi: bool,
    resume: bool,
    cadence: str,
    all_strikes: bool = False,
    lookback_days: int = 210,
) -> dict:
    """Process every qualifying expiration for one ticker. Returns stats dict."""
    stats = {
        "ticker": ticker,
        "expirations_total": 0,
        "expirations_done": 0,
        "expirations_skip": 0,
        "expirations_fail": 0,
        "contracts": 0,
        "rows": 0,
    }
    try:
        exps = _list_expirations(conn, ticker)
    except Exception as e:
        logger.warning("%s: list_expirations failed: %s", ticker, e)
        return stats
    if exps.empty:
        return stats
    mask = (exps.date >= start_date) & (exps.date <= end_date)
    exps = exps[mask]
    if cadence == "monthly":
        exps = exps[[_is_third_friday(d) for d in exps]]
    elif cadence == "weekly":
        # Friday expirations = standard weeklies + monthlies; drops Mon–Thu
        # dailies / 0DTE (the SPY/QQQ bloat a 30–45 DTE engine never reads).
        exps = exps[[d.weekday() == 4 for d in exps]]
    stats["expirations_total"] = len(exps)
    if len(exps) == 0:
        return stats

    spot_t = spot_lookup[spot_lookup["ticker"] == ticker].set_index("date")["close"]

    for exp in exps:
        exp_str = exp.strftime("%Y%m%d")
        if resume and _partition_exists(ticker, exp_str):
            stats["expirations_skip"] += 1
            continue
        if all_strikes:
            # Bulk: ONE call returns every strike & right for this expiration —
            # ~100x fewer requests than per-contract, which is what makes the
            # full-universe / full-depth "all we can" pull feasible at the
            # 4-connection cap. No spot or strike list needed.
            bulk = _fetch_expiration_bulk(conn, ticker, exp_str, include_oi, lookback_days)
            frames: list[pd.DataFrame] = [bulk] if not bulk.empty else []
        else:
            try:
                strikes = _list_strikes(conn, ticker, exp_str)
            except Exception:
                stats["expirations_fail"] += 1
                continue
            spot = None
            try:
                # exact match in Bloomberg lookup
                if exp in spot_t.index:
                    spot = float(spot_t.loc[exp])
                else:
                    # fall back to nearest prior trading day in Bloomberg
                    prior = spot_t.index[spot_t.index <= exp]
                    if len(prior):
                        spot = float(spot_t.loc[prior[-1]])
            except Exception:
                pass
            # If Bloomberg misses (recent expiration after Bloomberg's last
            # date, or ticker not in CSV), fall back to Theta stock EOD.
            if spot is None or spot <= 0:
                spot = _theta_stock_close(conn, ticker, exp)
            if spot is None or spot <= 0:
                stats["expirations_fail"] += 1
                continue
            keep = _pick_strike_band(strikes, spot, band_pct, max_strikes)
            if not keep:
                stats["expirations_fail"] += 1
                continue
            frames = []
            for K in keep:
                for r in ("call", "put"):
                    try:
                        df = _fetch_contract_history(
                            conn, ticker, exp_str, K, r, include_oi, lookback_days
                        )
                    except Exception as e:
                        logger.debug("%s %s %s %s: %s", ticker, exp_str, K, r, e)
                        continue
                    if not df.empty:
                        df["strike"] = K
                        df["right"] = r
                        frames.append(df)
        rows, n_contracts = _write_partition(ticker, exp_str, frames)
        stats["rows"] += rows
        stats["contracts"] += n_contracts
        if rows > 0:
            stats["expirations_done"] += 1
        else:
            stats["expirations_fail"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    ap.add_argument(
        "--tickers",
        help="Comma-separated tickers (default: full S&P 500 from constituents file)",
    )
    ap.add_argument("--start", help="Start expiration date YYYY-MM-DD (default: 10 years ago)")
    ap.add_argument("--end", help="End expiration date YYYY-MM-DD (default: today)")
    ap.add_argument(
        "--cadence",
        choices=["monthly", "weekly", "all"],
        default="monthly",
        help="monthly (3rd Friday only), weekly (all Friday expirations = "
        "weeklies+monthlies, drops Mon-Thu dailies/0DTE), or all (every listed)",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=210,
        help="Per-contract EOD history window before expiry (default 210 = full "
        "contract life; 90 covers the 0-90 DTE band a 30-45 DTE wheel + skew fit read)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override output root (e.g. a reference dir for index ETFs kept out "
        "of the wheel universe). Defaults to data_processed/theta/option_history.",
    )
    ap.add_argument(
        "--strike-band-pct",
        type=float,
        default=15.0,
        help="Strike band as ± percent of spot at expiration (default 15)",
    )
    ap.add_argument(
        "--max-strikes",
        type=int,
        default=10,
        help="Maximum strikes per expiration, evenly around spot (default 10)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-oi", action="store_true", help="Also fetch open_interest")
    ap.add_argument(
        "--all-strikes",
        action="store_true",
        help="Pull EVERY available strike (no spot band). Required for ranges with no "
        "spot source (e.g. pre-2018, where Bloomberg OHLCV doesn't reach). Bloomberg "
        "is not loaded in this mode.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip (ticker, expiration) partitions that already exist",
    )
    ap.add_argument("--limit-tickers", type=int, help="Cap number of tickers (testing)")
    ap.add_argument(
        "--read-timeout",
        type=int,
        default=180,
        help="Connector read timeout (s). Bulk all-strikes calls run 45-110s for "
        "liquid names, past the connector's 30s default; 180 prevents mega-cap "
        "calls timing out and silently dropping. Set lower only for banded pulls.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    if args.out_dir:
        global OUT_ROOT
        OUT_ROOT = Path(args.out_dir)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not _theta_up():
        print("Theta Terminal not reachable on 127.0.0.1:25503 — start it and retry.")
        return 2

    end_date = datetime.fromisoformat(args.end).date() if args.end else date.today()
    if args.start:
        start_date = datetime.fromisoformat(args.start).date()
    else:
        start_date = date(end_date.year - 10, end_date.month, end_date.day)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _load_universe()
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]

    logger.info(
        "option-history pull  tickers=%d  range=%s..%s  cadence=%s  band=±%.1f%%  "
        "max_strikes=%d  include_oi=%s  workers=%d  resume=%s",
        len(tickers),
        start_date.isoformat(),
        end_date.isoformat(),
        args.cadence,
        args.strike_band_pct,
        args.max_strikes,
        args.include_oi,
        args.workers,
        args.resume,
    )

    if args.all_strikes:
        spot_lookup = pd.DataFrame(columns=["ticker", "date", "close"])
        logger.info("--all-strikes mode: every strike, no spot band (Bloomberg not loaded)")
    else:
        spot_lookup = _load_bloomberg_spot()
        logger.info("Loaded spot lookup: %d rows", len(spot_lookup))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    conn = ThetaConnector()  # one shared connector; its semaphore caps at 4
    # All-strikes bulk EOD calls return whole chains over the lookback window and
    # can take 45-110s for liquid names — well past the connector's default 30s
    # read timeout (which would fail mega-caps outright). Raise it for this pull.
    conn._read_timeout = args.read_timeout

    all_stats: list[dict] = []
    t0 = pd.Timestamp.utcnow()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _one_ticker,
                t,
                conn,
                spot_lookup,
                start_date,
                end_date,
                args.strike_band_pct,
                args.max_strikes,
                args.include_oi,
                args.resume,
                args.cadence,
                args.all_strikes,
                args.lookback_days,
            ): t
            for t in tickers
        }
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                stats = fut.result()
            except Exception as e:
                logger.exception("%s failed: %s", t, e)
                stats = {"ticker": t, "error": str(e)}
            all_stats.append(stats)
            # Watchdog: bail with rc=2 if the shared connector entered
            # Terminal-down mode. A bash supervisor wrapping this script
            # uses rc=2 as the "restart Theta + retry" signal.
            if getattr(conn, "_terminal_down", False):
                logger.error(
                    "Connector entered Terminal-down mode after %d/%d tickers "
                    "— bailing so supervisor can restart Theta and relaunch with --resume",
                    i,
                    len(tickers),
                )
                return 2
            if i % 10 == 0 or i == len(tickers):
                done = sum(s.get("expirations_done", 0) for s in all_stats)
                skipped = sum(s.get("expirations_skip", 0) for s in all_stats)
                failed = sum(s.get("expirations_fail", 0) for s in all_stats)
                rows = sum(s.get("rows", 0) for s in all_stats)
                logger.info(
                    "progress: %d/%d tickers  expirations: %d done / %d skip / %d fail  rows=%d",
                    i,
                    len(tickers),
                    done,
                    skipped,
                    failed,
                    rows,
                )

    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
    total_done = sum(s.get("expirations_done", 0) for s in all_stats)
    total_skip = sum(s.get("expirations_skip", 0) for s in all_stats)
    total_fail = sum(s.get("expirations_fail", 0) for s in all_stats)
    total_rows = sum(s.get("rows", 0) for s in all_stats)
    total_contracts = sum(s.get("contracts", 0) for s in all_stats)
    logger.info(
        "done: tickers=%d  expirations: %d done / %d skip / %d fail  contracts=%d  rows=%d  elapsed=%.1fs",
        len(tickers),
        total_done,
        total_skip,
        total_fail,
        total_contracts,
        total_rows,
        elapsed,
    )

    # Manifest sidecar
    manifest = {
        "ran_at": datetime.now(UTC).isoformat(),
        "tickers": len(tickers),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "cadence": args.cadence,
        "strike_band_pct": args.strike_band_pct,
        "max_strikes": args.max_strikes,
        "include_oi": args.include_oi,
        "stats": {
            "expirations_done": total_done,
            "expirations_skip": total_skip,
            "expirations_fail": total_fail,
            "contracts": total_contracts,
            "rows": total_rows,
            "elapsed_s": elapsed,
        },
    }
    manifest_path = OUT_ROOT / "_manifest.json"
    history: list = []
    if manifest_path.exists():
        try:
            history = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            history = []
    history.append(manifest)
    manifest_path.write_text(json.dumps({"runs": history}, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
