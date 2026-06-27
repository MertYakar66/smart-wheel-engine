#!/usr/bin/env python3
"""Produce a compact, point-in-time option-premium table from the Theta larder.

Reads   ``data_processed/theta/option_history/ticker=<T>/expiration=<YYYYMMDD>/data.parquet``
writes  ``data_processed/option_premium/<T>.parquet``  (+ ``_manifest.json``)

Why this exists
---------------
The wheel's candidate premium is *synthetic* today: ``wheel_runner`` sets
``ShortOptionTrade.premium`` to ``black_scholes_price(sigma=iv)`` — the **same**
BSM call the EV engine uses for the risk-neutral fair value — so
``edge_vs_fair = premium - fair == 0`` and skew / variance-risk-premium are
EV-inert (see ``docs/PHASE2_SKEW_EXECUTION_SPEC.md``). Letting the *real* market
premium reach the engine is the prerequisite for skew to move EV.

This producer distils the real EOD option **mid** ``((bid + ask) / 2)`` from the
Theta option-history larder into the per-ticker parquet the connector accessor
``MarketDataConnector.get_option_premium*`` serves. It is the data half of the
"real-premium producer"; the (separate, EV-moving, §2-panel) ranker wiring that
swaps the synthetic premium for this mid is intentionally NOT part of this rail.

Snapshot-safety
---------------
Output lives under ``data_processed/`` which is ``.gitignore``-d, so it never
enters ``connector_data_sha256`` (the regression fingerprint) — this is a pure
additive rail with **zero re-baseline**. Wherever the produced files are absent
(CI, a fresh clone), the accessor returns empty and callers fall back to the
synthetic-BSM premium.

Usage
-----
    python scripts/produce_option_premiums.py --tickers AAPL,MSFT,NVDA
    python scripts/produce_option_premiums.py --tickers all --workers 4
    python scripts/produce_option_premiums.py --tickers all --dte-max 75
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# --- repo-root bootstrap so ``python scripts/...`` can import ``engine`` -------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.data_connector import OPTION_PREMIUM_COLUMNS  # noqa: E402

logger = logging.getLogger("produce_option_premiums")

# Default DTE belt: wide enough for the wheel's short-put (~30-45 DTE),
# covered-call and timing-gated strangle belts plus buffer, while keeping the
# distilled table a small fraction of the full larder.
DTE_MIN_DEFAULT = 0
DTE_MAX_DEFAULT = 75

_RAW_NEEDED = ("created", "expiration", "strike", "right", "bid", "ask")
_RIGHT_MAP = {"PUT": "put", "P": "put", "CALL": "call", "C": "call"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OPTION_PREMIUM_COLUMNS))


def distill_expiration_frame(
    df: pd.DataFrame,
    *,
    dte_min: int = DTE_MIN_DEFAULT,
    dte_max: int = DTE_MAX_DEFAULT,
) -> pd.DataFrame:
    """Pure transform: one raw Theta expiration frame → normalized premium rows.

    Computes the EOD ``mid``, normalizes ``right``, parses the ``created`` date
    axis and ``expiration``, filters to a two-sided market and the DTE belt, and
    collapses to one row per ``(date, strike, right)`` (latest snapshot wins).
    Returns a frame with :data:`OPTION_PREMIUM_COLUMNS` (possibly empty). No I/O.
    """
    if df is None or df.empty or not set(_RAW_NEEDED).issubset(df.columns):
        return _empty()

    work = pd.DataFrame()
    work["date"] = pd.to_datetime(df["created"], errors="coerce").dt.normalize()
    work["expiration"] = pd.to_datetime(
        df["expiration"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce"
    )
    work["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    work["right"] = df["right"].astype(str).str.strip().str.upper().map(_RIGHT_MAP)
    work["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    work["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    work["close"] = pd.to_numeric(df["close"], errors="coerce") if "close" in df.columns else np.nan
    work["volume"] = (
        pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else np.nan
    )
    work["open_interest"] = (
        pd.to_numeric(df["open_interest"], errors="coerce")
        if "open_interest" in df.columns
        else np.nan
    )

    work = work.dropna(subset=["date", "expiration", "strike", "right", "bid", "ask"])
    if work.empty:
        return _empty()

    # A real, uncrossed, two-sided market. A legitimate far-OTM 0-bid is allowed
    # (mirrors the EV engine's truthiness-free spread test) as long as the ask is
    # positive, but mid must be strictly positive to be a tradeable premium.
    valid = (
        (work["bid"] >= 0)
        & (work["ask"] >= 0)
        & (work["ask"] >= work["bid"])
        & ((work["bid"] > 0) | (work["ask"] > 0))
    )
    work = work[valid]
    if work.empty:
        return _empty()

    work["mid"] = (work["bid"] + work["ask"]) / 2.0
    work["dte"] = (work["expiration"] - work["date"]).dt.days
    work = work[
        (work["mid"] > 0)
        & (work["strike"] > 0)
        & (work["dte"] >= dte_min)
        & (work["dte"] <= dte_max)
    ]
    if work.empty:
        return _empty()

    work = (
        work.sort_values(["date", "strike", "right"])
        .drop_duplicates(["date", "strike", "right"], keep="last")
        .reset_index(drop=True)
    )
    return work[list(OPTION_PREMIUM_COLUMNS)]


def _expiration_parquets(ticker_dir: Path) -> list[Path]:
    return sorted(ticker_dir.glob("expiration=*/*.parquet"))


def produce_ticker(
    ticker: str,
    larder_dir: Path,
    out_dir: Path,
    *,
    dte_min: int = DTE_MIN_DEFAULT,
    dte_max: int = DTE_MAX_DEFAULT,
) -> dict:
    """Distil one ticker's option-history partitions → ``<out_dir>/<T>.parquet``.

    Returns a stats dict (``ticker, rows, n_expirations, date_min, date_max``).
    Writes atomically (tmp + rename). A ticker with no usable rows writes nothing
    and reports ``rows=0``.
    """
    ticker_dir = larder_dir / f"ticker={ticker}"
    parts = _expiration_parquets(ticker_dir)
    frames: list[pd.DataFrame] = []
    read_cols = list(_RAW_NEEDED) + ["close", "volume", "open_interest"]
    for p in parts:
        try:
            raw = pd.read_parquet(p)
        except Exception:
            logger.warning("skip unreadable parquet %s", p, exc_info=True)
            continue
        # Read only what we need when the column exists (keeps memory bounded on
        # mega-cap chains); fall back to the full frame if selection fails.
        present = [c for c in read_cols if c in raw.columns]
        if present:
            raw = raw[present]
        d = distill_expiration_frame(raw, dte_min=dte_min, dte_max=dte_max)
        if not d.empty:
            frames.append(d)

    if not frames:
        return {
            "ticker": ticker,
            "rows": 0,
            "n_expirations": 0,
            "date_min": None,
            "date_max": None,
        }

    full = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "expiration", "strike", "right"])
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.parquet"
    tmp_path = out_dir / f".{ticker}.parquet.tmp"
    full.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    return {
        "ticker": ticker,
        "rows": int(len(full)),
        "n_expirations": int(full["expiration"].nunique()),
        "date_min": full["date"].min().date().isoformat(),
        "date_max": full["date"].max().date().isoformat(),
    }


def discover_tickers(larder_dir: Path) -> list[str]:
    return sorted(
        p.name.split("=", 1)[1] for p in larder_dir.glob("ticker=*") if p.is_dir() and "=" in p.name
    )


def _worker(args: tuple) -> dict:
    ticker, larder_dir, out_dir, dte_min, dte_max = args
    return produce_ticker(ticker, Path(larder_dir), Path(out_dir), dte_min=dte_min, dte_max=dte_max)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tickers",
        default="all",
        help="comma-separated symbols, or 'all' to scan the larder",
    )
    ap.add_argument(
        "--larder-dir",
        default=str(_REPO / "data_processed" / "theta" / "option_history"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(_REPO / "data_processed" / "option_premium"),
    )
    ap.add_argument("--dte-min", type=int, default=DTE_MIN_DEFAULT)
    ap.add_argument("--dte-max", type=int, default=DTE_MAX_DEFAULT)
    ap.add_argument("--workers", type=int, default=1, help="parallel ticker workers")
    ap.add_argument("--limit", type=int, default=None, help="cap number of tickers")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    larder_dir = Path(args.larder_dir)
    out_dir = Path(args.out_dir)
    if not larder_dir.exists():
        logger.error("larder dir not found: %s", larder_dir)
        return 2

    if args.tickers.strip().lower() == "all":
        tickers = discover_tickers(larder_dir)
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.limit:
        tickers = tickers[: args.limit]
    if not tickers:
        logger.error("no tickers to process")
        return 2

    logger.info(
        "producing option premiums for %d tickers (dte %d-%d) -> %s",
        len(tickers),
        args.dte_min,
        args.dte_max,
        out_dir,
    )

    stats: list[dict] = []
    if args.workers > 1:
        payload = [(t, str(larder_dir), str(out_dir), args.dte_min, args.dte_max) for t in tickers]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, p): p[0] for p in payload}
            for fut in as_completed(futs):
                s = fut.result()
                stats.append(s)
                logger.info(
                    "  %-8s rows=%-8d exp=%-5d %s..%s",
                    s["ticker"],
                    s["rows"],
                    s["n_expirations"],
                    s["date_min"],
                    s["date_max"],
                )
    else:
        for t in tickers:
            s = produce_ticker(t, larder_dir, out_dir, dte_min=args.dte_min, dte_max=args.dte_max)
            stats.append(s)
            logger.info(
                "  %-8s rows=%-8d exp=%-5d %s..%s",
                s["ticker"],
                s["rows"],
                s["n_expirations"],
                s["date_min"],
                s["date_max"],
            )

    stats.sort(key=lambda s: s["ticker"])
    produced = [s for s in stats if s["rows"] > 0]
    manifest = {
        "params": {"dte_min": args.dte_min, "dte_max": args.dte_max},
        "n_tickers_requested": len(tickers),
        "n_tickers_produced": len(produced),
        "total_rows": int(sum(s["rows"] for s in stats)),
        "tickers": stats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "done: %d/%d tickers produced, %d total rows -> %s",
        len(produced),
        len(tickers),
        manifest["total_rows"],
        out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
