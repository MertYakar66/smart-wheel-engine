#!/usr/bin/env python3
"""EDGAR earnings-release history puller.

Pulls Form 8-K Item 2.02 ("Results of Operations and Financial Condition")
filings for a ticker universe and writes them to
``data_processed/edgar/earnings_history.parquet`` — the canonical
PIT-correct earnings-date store that the news-architecture redesign
campaign feeds into the existing ``EventGate``.

Why EDGAR for earnings dates?
-----------------------------
The current ``MarketDataConnector.get_next_earnings`` reads from
``data/bloomberg/sp500_earnings_yf.csv`` — a yfinance snapshot of the
*current* next-earnings date. For live use it's fine; for historical
backtests it leaks lookahead because yfinance always returns the most
recent known schedule, not what was known on the as-of date.

EDGAR 8-K filings are immutable historical records: the ``filing_date``
of an Item-2.02 filing IS the date the earnings release was made
public. Wiring EDGAR into the event-lockout gate replaces a forward-
looking calendar with a backward-looking record + a projection
heuristic (see ``EDGARAdapter.project_next_earnings``).

PR3/9 of the news-architecture redesign campaign — see
``docs/NEWS_REDESIGN_CAMPAIGN.md``. This PR ships the puller + storage
contract. A follow-up PR will wire ``EDGARAdapter.project_next_earnings``
into ``MarketDataConnector.get_next_earnings`` (or alongside it as a
PIT-correct alternative source).

Usage
-----
::

    # Single-ticker smoke test
    python scripts/pull_edgar_earnings.py --tickers AAPL MSFT

    # Full S&P 500 universe
    python scripts/pull_edgar_earnings.py --universe sp500

    # Constrain history depth (default: no lower bound, returns the
    # full ``recent`` block ≈ last 1000 filings per ticker, which is
    # >>10 years of 8-K history for any active company)
    python scripts/pull_edgar_earnings.py --universe sp500 --since 2018-01-01

    # Refresh: re-pull tickers already in the parquet (default is append
    # only — skips tickers with existing rows).
    python scripts/pull_edgar_earnings.py --universe sp500 --refresh

The SEC rate limit is 10 req/sec; the adapter defaults to a 120 ms
inter-call sleep so a 500-ticker pull takes ~60 s.

Output schema
-------------
``data_processed/edgar/earnings_history.parquet``:

    ticker            str          upper-case
    filing_date       datetime[ns] the SEC-recorded filing date
    accession         str          unique accession number
    items             str          comma-separated 8-K item codes
                                   (will always contain ``2.02``)
    primary_document  str          relative URL within the filing
    pulled_at         datetime[ns] UTC timestamp of this puller run
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# UTF-8-safe stdout/stderr (Windows console / redirected output).
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.external_data.edgar_adapter import EDGARAdapter  # noqa: E402

logger = logging.getLogger(__name__)
OUT_PATH = _ROOT / "data_processed" / "edgar" / "earnings_history.parquet"


def load_universe(mode: str, pit_date: str | None = None) -> list[str]:
    if mode == "sp500":
        from data.consolidated_loader import get_bloomberg_loader

        loader = get_bloomberg_loader()
        tickers = loader.get_universe_as_of(pit_date)
        return sorted({t for t in tickers if all(c.isalpha() or c == "." for c in t)})
    raise ValueError(f"Unknown universe {mode!r}")


def _existing_tickers(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    try:
        df = pd.read_parquet(out_path, columns=["ticker"])
        return set(df["ticker"].astype(str).str.upper().unique())
    except Exception:
        return set()


def pull_ticker(
    adapter: EDGARAdapter,
    ticker: str,
    since: str | None,
) -> tuple[pd.DataFrame, str]:
    """Pull a single ticker's earnings history. Returns (df, detail_string)."""
    try:
        df = adapter.earnings_history(ticker, since=since)
    except Exception as exc:
        return pd.DataFrame(), f"FAIL {type(exc).__name__}: {exc}"
    if df.empty:
        return df, "0 filings"
    df = df.copy()
    df["ticker"] = ticker.upper()
    df["pulled_at"] = datetime.now(UTC).replace(tzinfo=None)
    cols = ["ticker", "filing_date", "accession", "items", "primary_document", "pulled_at"]
    df = df[cols]
    return (
        df,
        f"{len(df)} filings, {df['filing_date'].min().date()} → {df['filing_date'].max().date()}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="+", help="Explicit ticker list")
    ap.add_argument("--universe", choices=["sp500"], help="Pull a named universe")
    ap.add_argument("--pit-date", help="PIT date for universe survivorship (YYYY-MM-DD)")
    ap.add_argument("--since", help="Lower-bound filing date (YYYY-MM-DD); default: no bound")
    ap.add_argument("--out", default=str(OUT_PATH), help=f"Output parquet (default: {OUT_PATH})")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Re-pull tickers already in the parquet (default: append-only, skip them)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Pull but don't write")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe:
        tickers = load_universe(args.universe, args.pit_date)
    else:
        print("ERROR: --tickers or --universe required")
        return 2

    out_path = Path(args.out)
    existing = set() if args.refresh else _existing_tickers(out_path)
    todo = [t for t in tickers if t not in existing]
    skipped = len(tickers) - len(todo)

    print(
        f"EDGAR earnings pull  universe={len(tickers)}  todo={len(todo)}  "
        f"skip_existing={skipped}  refresh={args.refresh}  since={args.since or 'none'}"
    )

    adapter = EDGARAdapter()
    t0 = time.perf_counter()
    new_rows: list[pd.DataFrame] = []
    n_done = n_err = n_empty = 0

    for ticker in todo:
        df, detail = pull_ticker(adapter, ticker, args.since)
        n_done += 1
        if detail.startswith("FAIL"):
            n_err += 1
        elif df.empty:
            n_empty += 1
        else:
            new_rows.append(df)
        if n_done % 25 == 0 or detail.startswith("FAIL") or not df.empty:
            print(f"  [{n_done:>4}/{len(todo)}] {ticker:<6}  {detail[:80]}", flush=True)

    elapsed = time.perf_counter() - t0

    if not new_rows:
        print(f"\nNo new rows. Done in {elapsed:.1f}s  |  {n_err} errors, {n_empty} empty.")
        return 0 if n_err == 0 else 1

    new_df = pd.concat(new_rows, ignore_index=True)

    if args.dry_run:
        print(f"\nDry run: {len(new_df)} new rows computed in {elapsed:.1f}s")
        print(new_df.head(10).to_string(index=False))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.refresh:
        # Append to existing parquet
        old_df = pd.read_parquet(out_path)
        combined = (
            pd.concat([old_df, new_df], ignore_index=True)
            .drop_duplicates(subset=["ticker", "accession"], keep="last")
            .sort_values(["ticker", "filing_date"])
            .reset_index(drop=True)
        )
        combined.to_parquet(out_path, index=False)
        print(
            f"\nAppended {len(new_df)} rows ({len(combined)} total) → {out_path}  "
            f"in {elapsed:.1f}s  |  {n_err} errors, {n_empty} empty."
        )
    else:
        new_df = new_df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
        new_df.to_parquet(out_path, index=False)
        print(
            f"\nWrote {len(new_df)} rows → {out_path}  in {elapsed:.1f}s  "
            f"|  {n_err} errors, {n_empty} empty."
        )

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
