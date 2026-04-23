#!/usr/bin/env python3
"""
Pull a fundamentals snapshot for every ticker in the universe from yfinance.

The existing ``data/bloomberg/sp500_fundamentals.csv`` can go stale between
Bloomberg refreshes. This puller refreshes a parallel file in the **same
schema** so the consolidated loader can pick it up seamlessly (it writes
to ``sp500_fundamentals_yf.csv`` — the wheel runner merges this on top of
the Bloomberg snapshot, ticker-by-ticker).

Fields (mapped to the Bloomberg column names the loader already understands)::

    ticker                            # "AAPL UW Equity" style for compatibility
    cur_mkt_cap                       # market cap, dollars
    pe_ratio, best_pe_ratio           # trailing / forward P/E
    beta_raw_overridable              # 5y beta
    eqy_dvd_yld_12m                   # trailing-12m dividend yield, %
    return_com_eqy                    # ROE, %
    tot_debt_to_tot_eqy               # debt/equity ratio
    free_cash_flow_yield              # FCF yield, %
    gics_sector_name                  # Yahoo sector (naming convention differs)
    gics_industry_group_name          # Yahoo industry
    volatility_30d                    # realised vol, 30d, computed from price history
    30day_impvol_100.0%mny_df         # left NaN — yfinance does not publish IV

Run
---
    python scripts/pull_fundamentals_yf.py                    # all S&P names
    python scripts/pull_fundamentals_yf.py --tickers AAPL MSFT
    python scripts/pull_fundamentals_yf.py --workers 8

yfinance imposes a soft rate limit; default 4 workers is safe. With 8
workers Yahoo will sometimes drop a ticker (script logs and continues).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

logger = logging.getLogger(__name__)
OUT_CSV = _ROOT / "data" / "bloomberg" / "sp500_fundamentals_yf.csv"


def load_universe(pit_date: str | None = None) -> list[str]:
    from data.consolidated_loader import get_bloomberg_loader
    L = get_bloomberg_loader()
    u = L.get_universe_as_of(pit_date)
    return sorted({t for t in u if all(c.isalpha() or c == "." for c in t)})


def _pull_one(ticker: str) -> dict:
    """Return one row of fundamentals. Missing fields = None."""
    out = {
        "ticker": f"{ticker} US Equity",  # matches loader's Bloomberg format
        "cur_mkt_cap": None,
        "pe_ratio": None,
        "best_pe_ratio": None,
        "beta_raw_overridable": None,
        "eqy_dvd_yld_12m": None,
        "return_com_eqy": None,
        "tot_debt_to_tot_eqy": None,
        "free_cash_flow_yield": None,
        "gics_sector_name": None,
        "gics_industry_group_name": None,
        "volatility_30d": None,
        "30day_impvol_100.0%mny_df": None,
    }
    try:
        t = yf.Ticker(ticker)
        info = t.get_info() or {}
    except Exception as e:
        return {**out, "_error": f"{type(e).__name__}: {e}"[:100]}

    def f(key):
        v = info.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    out["cur_mkt_cap"] = f("marketCap")
    out["pe_ratio"] = f("trailingPE")
    out["best_pe_ratio"] = f("forwardPE") or f("trailingPE")
    out["beta_raw_overridable"] = f("beta") or f("beta3Year")
    # yfinance dividendYield is already in percent form like Bloomberg's eqy_dvd_yld_12m
    out["eqy_dvd_yld_12m"] = f("dividendYield")
    # ROE comes as fraction (0..1); Bloomberg stores as %  — scale up
    roe = f("returnOnEquity")
    out["return_com_eqy"] = roe * 100.0 if roe is not None else None
    out["tot_debt_to_tot_eqy"] = f("debtToEquity")
    # FCF yield: Bloomberg stores as %, yfinance doesn't expose it directly. Approximate:
    #   fcf_yield = freeCashflow / marketCap * 100
    fcf = f("freeCashflow")
    mc = out["cur_mkt_cap"]
    if fcf and mc and mc > 0:
        out["free_cash_flow_yield"] = (fcf / mc) * 100.0
    out["gics_sector_name"] = info.get("sector")
    out["gics_industry_group_name"] = info.get("industry")

    # Realised 30d vol from recent history
    try:
        h = t.history(period="60d", auto_adjust=False, actions=False)
        if not h.empty and "Close" in h.columns:
            log_rets = np.diff(np.log(h["Close"].dropna().values))
            if len(log_rets) >= 20:
                out["volatility_30d"] = float(np.std(log_rets[-30:], ddof=1) * np.sqrt(252) * 100.0)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--pit-date")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = (
        [t.upper() for t in args.tickers]
        if args.tickers else
        load_universe(args.pit_date)
    )
    print(f"Fundamentals pull  tickers={len(tickers)}  workers={args.workers}")

    t0 = time.perf_counter()
    rows: list[dict] = []
    n_done = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_pull_one, t): t for t in tickers}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            n_done += 1
            if "_error" in row:
                n_err += 1
            if n_done % 50 == 0 or "_error" in row:
                lead = "FAIL" if "_error" in row else "OK  "
                detail = row.get("_error", f"pe={row.get('pe_ratio')}")
                tk = row["ticker"].split(" ")[0]
                print(f"  [{n_done:>4}/{len(tickers)}] {tk:<6} {lead}  {detail}", flush=True)

    df = pd.DataFrame(rows).drop(columns=["_error"], errors="ignore")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {len(df)} rows → {out}")
    print(f"Done in {elapsed:.1f}s  |  {n_err} errors  |  "
          f"non-null P/E: {df['pe_ratio'].notna().sum()}  |  "
          f"non-null beta: {df['beta_raw_overridable'].notna().sum()}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
