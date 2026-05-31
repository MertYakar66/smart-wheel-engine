"""Grounding probe for the 2026-05-31 heavy-verification campaign.

Read-only. Establishes the data facts every later driver depends on:
  1. Theta option-history parquet schema (real bid/ask/close columns).
  2. The connector's OHLCV access method + the rotated-column rename invariant.
  3. Fundamentals columns (market cap, sector) for the cap-weighted index proxy.
  4. Theta coverage shape (how many tickers/expirations, date span).

Run:  python docs/verification_artifacts/campaign_2026-05-31/_grounding.py

Portable: ROOT is derived from __file__ so this works from any clone/worktree
(the old pit_realism_driver hardcoded an absolute WORKTREE path -- fixed here).
The sys.path.insert is load-bearing on Windows: a user-site .pth can otherwise
shadow `import engine.*` with an older primary clone (see memory: sys-path-worktree-shadow).
"""
from __future__ import annotations

import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

print(f"ROOT = {ROOT}")
print(f"engine import will resolve from: {ROOT}\\engine")
print("=" * 78)


def section(t: str) -> None:
    print(f"\n### {t}\n" + "-" * 60)


# ---------------------------------------------------------------------------
section("1. Theta option-history parquet schema (AAPL sample)")
# data_processed/ is gitignored -> the Theta history lives only in the PRIMARY
# clone, not in this worktree. Search candidate roots (worktree first, then the
# known primary clone). Later drivers reuse this resolution.
THETA_CANDIDATES = [
    os.path.join(ROOT, "data_processed", "theta", "option_history"),
    r"C:\Users\merty\Desktop\smart-wheel-engine\data_processed\theta\option_history",
]
theta_root = next((p for p in THETA_CANDIDATES if os.path.isdir(p)), THETA_CANDIDATES[0])
print(f"THETA_ROOT resolved -> {theta_root}  (exists={os.path.isdir(theta_root)})")
aapl_glob = os.path.join(theta_root, "ticker=AAPL", "expiration=*", "data.parquet")
samples = sorted(glob.glob(aapl_glob))
print(f"AAPL expiration parquets found: {len(samples)}")
if samples:
    df = pd.read_parquet(samples[len(samples) // 2])  # a mid-history one
    print(f"sample file: ...{samples[len(samples)//2][len(theta_root):]}")
    print(f"columns: {df.columns.tolist()}")
    print(f"dtypes:\n{df.dtypes}")
    print(f"rows: {len(df)}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(4).to_string())
    # Do bid/ask exist and are they real (non-degenerate)?
    for col in ("bid", "ask"):
        if col in df.columns:
            print(f"{col}: min={df[col].min()} max={df[col].max()} nnull={df[col].isna().sum()}")
    if {"bid", "ask"}.issubset(df.columns):
        sp = (df["ask"] - df["bid"])
        print(f"spread (ask-bid): min={sp.min():.4f} mean={sp.mean():.4f} max={sp.max():.4f} neg={int((sp<0).sum())}")

# ---------------------------------------------------------------------------
section("2. Connector OHLCV access + rotated-column rename invariant")
from engine.data_connector import MarketDataConnector  # noqa: E402

methods = [m for m in dir(MarketDataConnector) if not m.startswith("__")]
ohlcv_like = [m for m in methods if any(k in m.lower() for k in ("ohlcv", "price", "hist", "univ", "ticker", "spot"))]
print(f"connector methods (ohlcv/price/universe-like): {ohlcv_like}")

# Replicate the documented rename and CHECK the invariant on the real file.
raw = pd.read_csv(os.path.join(ROOT, "data", "bloomberg", "sp500_ohlcv.csv"))
print(f"raw OHLCV rows={len(raw)} cols={raw.columns.tolist()}")
# Documented rotation: raw open=HIGH, raw high=CLOSE, raw close=OPEN, raw low=LOW
ren = raw.rename(columns={"open": "t_high", "high": "t_close", "close": "t_open", "low": "t_low"})
ohlc = ren.dropna(subset=["t_high", "t_close", "t_open", "t_low"])
bad_high = int((ohlc["t_high"] < ohlc[["t_open", "t_close", "t_low"]].max(axis=1)).sum())
bad_low = int((ohlc["t_low"] > ohlc[["t_open", "t_close", "t_high"]].min(axis=1)).sum())
print(f"AFTER documented rename: bad_high(viol high>=max)={bad_high}  bad_low(viol low<=min)={bad_low}  of {len(ohlc)}")
print("  -> 0/0 confirms the rename quirk: TRUE CLOSE lives in the raw `high` column.")
# date span + ticker count
raw["date"] = pd.to_datetime(raw["date"])
print(f"date span: {raw['date'].min().date()} -> {raw['date'].max().date()}")
print(f"distinct raw tickers: {raw['ticker'].nunique()}  (e.g. {raw['ticker'].iloc[0]!r})")

# ---------------------------------------------------------------------------
section("3. Fundamentals columns (market cap / sector for index proxy)")
for fn in ("sp500_fundamentals_yf.csv", "sp500_fundamentals.csv"):
    fp = os.path.join(ROOT, "data", "bloomberg", fn)
    if os.path.exists(fp):
        fdf = pd.read_csv(fp, nrows=3)
        print(f"{fn}: {fdf.columns.tolist()}")

# ---------------------------------------------------------------------------
section("4. Theta coverage shape")
if not os.path.isdir(theta_root):
    print(f"THETA_ROOT not found: {theta_root} -- skipping coverage")
    print("\nGROUNDING COMPLETE.")
    sys.exit(0)
tickers = [d for d in os.listdir(theta_root) if d.startswith("ticker=")]
print(f"theta tickers: {len(tickers)}")
mani = os.path.join(theta_root, "_manifest.json")
if os.path.exists(mani):
    m = json.load(open(mani))
    for run in m.get("runs", []):
        s = run.get("stats", {})
        print(f"  run {run.get('start')}..{run.get('end')} cadence={run.get('cadence')} "
              f"strikes<= {run.get('max_strikes')} band={run.get('strike_band_pct')}% "
              f"-> exp_done={s.get('expirations_done')} contracts={s.get('contracts')} rows={s.get('rows')}")

print("\nGROUNDING COMPLETE.")
