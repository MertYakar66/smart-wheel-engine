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
import json
import logging
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.theta_connector import ThetaConnector, _normalise_theta_symbol  # noqa: E402

logger = logging.getLogger(__name__)

OUT_ROOT = _ROOT / "data_processed" / "theta" / "iv_surface_history"
# Target DTE buckets we want to sample — we snap to the nearest listed expiry.
TARGET_DTES = (7, 14, 30, 60, 90, 180)


# ----------------------------------------------------------------------
# Per-ticker expirations cache — shared across worker threads.
# ``option/list/expirations`` returns a static list per ticker (it doesn't
# vary by date), so fetching it for every (ticker, date) job wastes ~365×
# the API calls per ticker. Cache by normalised symbol with a lock so
# concurrent first-touch fetches don't race.
# ----------------------------------------------------------------------
def _get_cached_expirations(
    conn: ThetaConnector,
    sym: str,
    cache: dict[str, pd.Series] | None,
    cache_lock: threading.Lock | None,
) -> pd.Series | None:
    """Return a Series of expiration Timestamps for ``sym`` (cached).

    Returns ``None`` if Theta refuses or returns malformed data.
    """
    if cache is not None and sym in cache:
        return cache[sym]
    if cache_lock is not None:
        cache_lock.acquire()
    try:
        if cache is not None and sym in cache:
            return cache[sym]
        try:
            exps_df = conn._fetch("/v3/option/list/expirations", {"symbol": sym})
        except Exception as e:
            logger.debug("%s: expirations fetch failed: %s", sym, e)
            return None
        if exps_df is None or exps_df.empty:
            return None
        exp_col = next((c for c in ("expiration", "date", "exp") if c in exps_df.columns), None)
        if exp_col is None:
            return None
        exps = pd.to_datetime(exps_df[exp_col], errors="coerce").dropna()
        if cache is not None:
            cache[sym] = exps
        return exps
    finally:
        if cache_lock is not None:
            cache_lock.release()


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
    # Keep only real equity symbols: alpha-only or dotted (e.g. BRK.B). This also
    # drops Bloomberg internal placeholder codes like "1284849D" (they contain
    # digits). A prior `not t.endswith("D")` filter had an operator-precedence
    # bug -- ((t and not t.endswith("D")) or "." in t) -- that silently dropped
    # legitimate all-alpha tickers ending in D (AMD, GOLD, HOOD, ...). The
    # alpha-only filter below already excludes the numeric codes, so it was removed.
    tickers = [t for t in tickers if t and all(c.isalpha() or c == "." for c in t)]
    return sorted(set(tickers))


# ----------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------
_HIST_GREEKS_ENDPOINTS = (
    "/v3/option/history/greeks/implied_volatility",
    "/v3/option/history/greeks/first_order",
)


_FALLBACK_K_DEFAULT = 10


def _try_fetch_history_greeks(
    conn: ThetaConnector,
    sym: str,
    expiration: pd.Timestamp,
    as_of_str: str,
) -> pd.DataFrame:
    """Fetch the historical Greeks bars for one ``(symbol, expiration, day)``.

    Tries ``implied_volatility`` first (exposes the IV column directly),
    then falls back to ``first_order`` if the IV variant returns nothing.
    Returns the first non-empty response, or an empty DataFrame if both
    endpoints come back empty / 472 NO_DATA / raise.
    """
    exp_str = expiration.strftime("%Y%m%d")
    for ep in _HIST_GREEKS_ENDPOINTS:
        params = {
            "symbol": sym,
            "expiration": exp_str,
            "start_date": as_of_str,
            "end_date": as_of_str,
            "interval": "1h",
        }
        try:
            chain = conn._fetch(ep, params)
        except Exception as e:
            logger.debug(
                "%s %s %s: history greeks fetch failed (%s): %s",
                sym,
                as_of_str,
                exp_str,
                ep,
                e,
            )
            continue
        if chain is not None and not chain.empty:
            return chain
    return pd.DataFrame()


def _normalise_chain_to_surface(
    chain: pd.DataFrame,
    *,
    ticker: str,
    as_of: date,
    expiration: pd.Timestamp,
) -> pd.DataFrame:
    """Convert a raw history-greeks response into the long-format columns
    we write to disk. Returns empty if the response can't be normalised
    (missing required columns, all IV out of bounds, no usable rows after
    the hourly→latest collapse).
    """
    chain = chain.copy()
    chain.columns = [c.lower() for c in chain.columns]
    rename_map = {
        "implied_vol": "iv",
        "implied_volatility": "iv",
        "mid_iv": "iv",
        "impl_vol": "iv",
        "midpoint": "mid",
        "mid_price": "mid",
        "option_type": "right",
    }
    for src, dst in rename_map.items():
        if src in chain.columns and dst not in chain.columns:
            chain = chain.rename(columns={src: dst})
    need = {"strike", "right", "iv"}
    if not need.issubset(chain.columns):
        return pd.DataFrame()
    chain = chain[chain["iv"].between(0.0, 5.0, inclusive="neither")]
    if chain.empty:
        return pd.DataFrame()
    # Hourly bars → one row per (strike, right) for ``as_of``. Keep the
    # last bar of the session; if there's only one bar the tail(1) keeps it.
    ts_col = next((c for c in ("timestamp", "date") if c in chain.columns), None)
    if ts_col is not None:
        chain[ts_col] = pd.to_datetime(chain[ts_col], errors="coerce")
        chain = (
            chain.dropna(subset=[ts_col])
            .sort_values(ts_col)
            .groupby(["strike", "right"], as_index=False)
            .tail(1)
        )
    if chain.empty:
        return pd.DataFrame()
    chain["expiration"] = expiration
    chain["dte"] = (expiration - pd.Timestamp(as_of)).days
    chain["date"] = pd.Timestamp(as_of)
    chain["ticker"] = ticker
    keep = [
        c
        for c in ("date", "ticker", "expiration", "dte", "strike", "right", "iv", "mid", "delta")
        if c in chain.columns
    ]
    return chain[keep]


def _surface_for_date(
    conn: ThetaConnector,
    ticker: str,
    as_of: date,
    *,
    expirations_cache: dict[str, pd.Series] | None = None,
    cache_lock: threading.Lock | None = None,
    strict: bool = True,
    fallback_k: int = _FALLBACK_K_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Pull the historical IV surface for one ticker on one date.

    Strategy:
      1. List available expirations (cached per ticker — see
         :func:`_get_cached_expirations`).
      2. For each :data:`TARGET_DTES` bucket, find the **nearest
         expiration that Theta has IV history for**. Try up to
         ``fallback_k`` nearest candidates per bucket; first non-empty
         response wins. Used expirations are claimed and not offered
         to subsequent buckets.

         *Why this matters:* Theta's ``option/list/expirations`` returns
         every listed weekly + monthly + LEAPS expiration, but its IV
         history coverage is sparser — many weekly expirations on liquid
         names like AAPL return ``"No data found"`` (HTTP 472). The old
         "snap to nearest, take it or leave it" approach had ~13%
         full-coverage rate on AAPL during Profile D. Iterating to the
         next-nearest with data raises that to ~100%.

      3. ``interval='1h'`` is the finest the Greeks history endpoint
         supports; collapse to one row per ``(strike, right)`` keeping
         the latest hourly bar of the session.

    Returns a tuple ``(df, status)``.

      ``df``: long-format DataFrame, empty if no usable surface was
        produced for ``as_of`` (weekend / holiday / Theta refused / strict
        mode rejected a partial surface).

      ``status``: dict with these keys (always present):

        ``target_dtes``      number of TARGET_DTES buckets attempted (6)
        ``succeeded_buckets`` buckets that landed on an expiration with data
        ``failed_buckets``   list[int] — DTE values whose ``fallback_k``
                             candidates all returned empty
        ``chosen_expirations`` list[date] — actual expirations picked,
                              ordered by bucket
        ``partial``          ``True`` iff ``0 < succeeded < target_dtes``
        ``rejected_partial`` ``True`` iff ``strict`` was set and the
                             surface was discarded because ``partial``
                             was True. The partition is NOT written.

    The default ``strict=True`` is the safe choice: silently writing
    partial surfaces (3-of-6 expirations) was the bug that polluted
    Profile D — a partial surface looks identical to a complete one
    downstream, so the SVI calibrator can't tell it's missing data.
    """
    sym = _normalise_theta_symbol(ticker)
    target_n = len(TARGET_DTES)

    def _empty_status(extra: dict | None = None) -> dict:
        s = {
            "target_dtes": target_n,
            "succeeded_buckets": 0,
            "failed_buckets": [],
            "chosen_expirations": [],
            "partial": False,
            "rejected_partial": False,
        }
        if extra:
            s.update(extra)
        return s

    # 1) expirations (cached per ticker, optional)
    if expirations_cache is not None:
        exps_all = _get_cached_expirations(conn, sym, expirations_cache, cache_lock)
    else:
        exps_all = _get_cached_expirations(conn, sym, None, None)
    if exps_all is None or exps_all.empty:
        return pd.DataFrame(), _empty_status({"failed_buckets": list(TARGET_DTES)})

    exps_future = exps_all[exps_all >= pd.Timestamp(as_of)]
    if exps_future.empty:
        return pd.DataFrame(), _empty_status({"failed_buckets": list(TARGET_DTES)})

    # 2 + 3) For each TARGET_DTES bucket, iterate up to fallback_k nearest
    # expirations; first one that returns data wins. Used expirations are
    # excluded from later buckets so each bucket gets a distinct expiry.
    as_of_str = pd.Timestamp(as_of).strftime("%Y%m%d")
    used_exps: set[pd.Timestamp] = set()
    frames: list[pd.DataFrame] = []
    chosen: list[pd.Timestamp] = []
    failed_buckets: list[int] = []

    for dte in TARGET_DTES:
        target = pd.Timestamp(as_of) + pd.Timedelta(days=dte)
        # Filter out expirations that earlier buckets already claimed
        available = exps_future[~exps_future.isin(used_exps)]
        if available.empty:
            failed_buckets.append(dte)
            continue
        # Order by distance to target, take top ``fallback_k`` nearest
        distances = (available - target).abs()
        candidates = available.loc[distances.sort_values().index].head(fallback_k)

        bucket_done = False
        for cand in candidates:
            chain = _try_fetch_history_greeks(conn, sym, cand, as_of_str)
            if chain.empty:
                continue
            frame = _normalise_chain_to_surface(
                chain,
                ticker=ticker,
                as_of=as_of,
                expiration=cand,
            )
            if frame.empty:
                continue
            frames.append(frame)
            chosen.append(cand)
            used_exps.add(cand)
            bucket_done = True
            break

        if not bucket_done:
            failed_buckets.append(dte)

    succeeded = len(frames)
    partial = 0 < succeeded < target_n
    status = {
        "target_dtes": target_n,
        "succeeded_buckets": succeeded,
        "failed_buckets": failed_buckets,
        "chosen_expirations": [(c.date() if hasattr(c, "date") else c) for c in chosen],
        "partial": partial,
        "rejected_partial": False,
    }

    if not frames:
        return pd.DataFrame(), status

    if strict and partial:
        # Loudly drop the partial surface: we have data for some
        # buckets but not all. Writing this would silently degrade
        # downstream consumers (SVI calibrator, term-structure features).
        logger.warning(
            "%s %s: rejecting partial surface — succeeded=%d/%d, failed_buckets=%s, chosen=%s",
            ticker,
            as_of,
            succeeded,
            target_n,
            failed_buckets,
            [c.isoformat() for c in status["chosen_expirations"]],
        )
        status["rejected_partial"] = True
        return pd.DataFrame(), status

    return pd.concat(frames, ignore_index=True), status


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


def _process_one(
    conn: ThetaConnector,
    expirations_cache: dict[str, pd.Series],
    cache_lock: threading.Lock,
    strict: bool,
    args: tuple[str, date, bool],
) -> tuple[str, date, bool, str]:
    """Worker callable. ``conn`` / ``expirations_cache`` / ``cache_lock`` /
    ``strict`` are bound by the caller (typically via ``functools.partial``)
    so every job shares one connector + one cache, capping aggregate
    concurrency at the connector's tier-correct semaphore."""
    ticker, d, force = args
    if not force and already_have(ticker, d):
        return ticker, d, True, "cached"
    try:
        df, status = _surface_for_date(
            conn,
            ticker,
            d,
            expirations_cache=expirations_cache,
            cache_lock=cache_lock,
            strict=strict,
        )
    except Exception as e:
        return ticker, d, False, f"{type(e).__name__}: {e}"

    succeeded = status.get("succeeded_buckets", 0)
    target_n = status.get("target_dtes", len(TARGET_DTES))

    if df.empty:
        if status.get("rejected_partial"):
            failed_dtes = status.get("failed_buckets") or []
            return (
                ticker,
                d,
                False,
                (f"partial-strict {succeeded}/{target_n} failed_dtes={failed_dtes}"),
            )
        if status.get("failed_buckets") == list(TARGET_DTES):
            return ticker, d, False, "all-buckets-empty"
        return ticker, d, False, "empty"

    write_partition(df, ticker, d)
    if status.get("partial"):
        return ticker, d, True, (f"rows={len(df)} PARTIAL {succeeded}/{target_n}")
    return ticker, d, True, f"rows={len(df)} {succeeded}/{target_n}"


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
    # CLI utf-8 fix (Windows console / redirected output). reconfigure()
    # changes the encoding in place — it does not replace stdout/stderr,
    # so it stays safe under pytest's capture machinery.
    for _stream in (sys.stdout, sys.stderr):
        if isinstance(_stream, io.TextIOWrapper):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--universe", choices=["sp500"])
    ap.add_argument("--pit-date", help="PIT universe date (YYYY-MM-DD)")
    ap.add_argument("--start", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, help="Last N business days (short form)")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker threads. All workers share a single "
        "ThetaConnector whose internal semaphore caps "
        "aggregate concurrency at the tier limit, so "
        "extra workers queue rather than oversubscribe.",
    )
    ap.add_argument("--resume", action="store_true", help="Skip partitions that already exist")
    ap.add_argument("--force", action="store_true", help="Overwrite existing partitions")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write surfaces with fewer than 6 chosen expirations "
        "(default: strict — partial surfaces are dropped). "
        "Only use when you've decided silently degraded "
        "coverage is acceptable for your downstream consumer.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print(
            "ERROR: Theta Terminal not reachable on 127.0.0.1:25503 — start the Terminal and retry"
        )
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
    # Theta's history endpoints reject same-day requests without explicit
    # start_time / end_time bounds ("Current day requests must have a start
    # time less than current time"). The historical puller deliberately
    # does not pass time bounds, so we drop today (and any future date)
    # here. Today's surface can be backfilled tomorrow.
    today = date.today()
    skipped_today = [d for d in dates if d >= today]
    dates = [d for d in dates if d < today]
    if skipped_today:
        print(
            f"Skipping {len(skipped_today)} same-day/future date(s) "
            f"({skipped_today[0]}..{skipped_today[-1]}) — "
            f"Theta history endpoints require explicit time bounds for "
            f"current-day requests"
        )
    if args.days:
        dates = dates[-args.days :]

    force = args.force and not args.resume
    strict = not args.allow_partial
    jobs = [(t, d, force) for t in tickers for d in dates]
    total = len(jobs)

    # ONE connector, ONE expirations cache, ONE lock — shared across workers.
    # Per-thread connectors caused the Profile-D blow-up: each thread had its
    # own _MAX_CONCURRENT=4 semaphore, so 4 workers × 4 = 16 concurrent
    # requests against a STANDARD-tier 4-concurrent ceiling. Theta returned
    # 472 NO_DATA under contention and the puller silently dropped partial
    # surfaces. Sharing the connector caps aggregate concurrency.
    conn = ThetaConnector()
    try:
        expirations_cache: dict[str, pd.Series] = {}
        cache_lock = threading.Lock()
        process = partial(_process_one, conn, expirations_cache, cache_lock, strict)

        print(
            f"Pulling IV surface history  tickers={len(tickers)}  "
            f"dates={len(dates)} ({start_d}..{end_d})  jobs={total}  "
            f"workers={args.workers}  strict={strict}"
        )

        t0 = time.perf_counter()
        n_ok = 0
        n_partial = 0
        n_fail = 0
        n_partial_rejected = 0
        n_done = 0

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, j): j for j in jobs}
            for fut in as_completed(futs):
                ticker, d, ok, detail = fut.result()
                n_done += 1
                if ok:
                    n_ok += 1
                    if "PARTIAL" in detail:
                        n_partial += 1
                else:
                    n_fail += 1
                    if "partial-strict" in detail:
                        n_partial_rejected += 1
                if n_done % 100 == 0 or not ok:
                    print(
                        f"  [{n_done:>5}/{total}] {ticker:<6} {d} "
                        f"{'OK' if ok else 'FAIL':<4}  {detail[:80]}",
                        flush=True,
                    )

        elapsed = time.perf_counter() - t0
        print()
        print(f"Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
        if n_partial:
            print(f"  ↳ of OK: {n_partial} written with PARTIAL coverage (--allow-partial was set)")
        if n_partial_rejected:
            print(
                f"  ↳ of FAIL: {n_partial_rejected} dropped by strict mode "
                "(use --allow-partial to keep)"
            )
        print(f"Written under: {OUT_ROOT}")
        return 0 if n_fail == 0 else 1
    finally:
        # Per-endpoint failure manifest sidecar (issue #71, DECISIONS.md D11).
        # Drains conn.get_failures() and writes a JSON sidecar so a downstream
        # observer can distinguish "this ticker had no data" from "this ticker
        # hit a per-endpoint timeout while the Terminal was healthy". Runs in
        # finally so half-run pulls (KeyboardInterrupt etc.) still emit the
        # manifest — half-run is exactly when this signal matters most.
        failures = conn.get_failures()
        if failures:
            manifest_dir = Path("data_processed/theta")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            step_name = Path(__file__).stem
            manifest_path = manifest_dir / f"_manifest_failures_{step_name}_{ts}.json"
            payload = [asdict(r) for r in failures]
            manifest_path.write_text(json.dumps(payload, indent=2))
            print(
                f"[{step_name}] wrote {len(failures)} per-endpoint failure(s) → "
                f"{manifest_path.relative_to(Path.cwd())}",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
