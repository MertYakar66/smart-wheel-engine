"""
Cheap context-index backfill (LONG schema: date,ticker,close).

Backs vix_futures_curve / vol_indices / spx_correlation / rates_fx_vol. Each is
only a handful of index series, so we pull each series floor->END in ONE bdh
call (per-ticker, so one bad ticker can't null the rest), then merge onto the
existing file (new wins on overlap), dedupe (date,ticker), sort (ticker,date).
This simultaneously makes-current AND backfills to inception.

Usage:  python scripts/pull_context_index.py <dataset>
        dataset in {vix_futures_curve, vol_indices, spx_correlation, rates_fx_vol}

Bloomberg ticker = "<label> Index". Series that began after the file floor simply
return fewer rows (Bloomberg starts at each series' inception).

Env: SWE_PULL_END, SWE_PULL_NO_WRITE.
"""

import io
import os
import sys

import pandas as pd
from xbbg import blp

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8", errors="replace")

END = os.environ.get("SWE_PULL_END") or "2026-06-04"
NO_WRITE = bool(os.environ.get("SWE_PULL_NO_WRITE"))

CONFIGS = {
    # floor = earliest inception among the series in the file
    "vix_futures_curve": {"labels": ["UX1", "UX2", "UX3", "UX4", "UX5", "UX6", "UX7"], "floor": "2004-01-01"},
    "vol_indices": {"labels": ["SKEW", "VXN", "RVX", "VVIX", "OVX", "GVZ", "VXEEM", "VIX9D"], "floor": "1990-01-01"},
    "spx_correlation": {"labels": ["COR1M", "COR3M", "COR6M"], "floor": "2006-01-01"},
    "rates_fx_vol": {"labels": ["MOVE", "CVIX", "JPMVXYG7"], "floor": "1988-01-01"},
}

if len(sys.argv) < 2 or sys.argv[1] not in CONFIGS:
    print(f"usage: pull_context_index.py <{'|'.join(CONFIGS)}>")
    raise SystemExit(2)

dataset = sys.argv[1]
cfg = CONFIGS[dataset]
floor = os.environ.get("SWE_PULL_FLOOR") or cfg["floor"]
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg", f"{dataset}.csv")
OUT_COLS = ["date", "ticker", "close"]


def to_native(obj):
    return obj.to_native() if hasattr(obj, "to_native") else obj


print(f"{dataset}: floor={floor} end={END}  series={cfg['labels']}")
rows = []
for label in cfg["labels"]:
    tkr = f"{label} Index"
    print(f"Pulling {tkr} PX_LAST {floor} -> {END} ...", flush=True)
    try:
        raw = to_native(blp.bdh(tickers=tkr, flds="PX_LAST", start_date=floor, end_date=END))
        if raw is None or len(raw) == 0:
            print(f"  (no data for {tkr})")
            continue
        if "value" in raw.columns:  # long form
            df = raw[["date", "value"]].rename(columns={"value": "close"})
        else:  # wide single-col fallback
            df = raw.copy()
            df.columns = ["date", "close"][: len(df.columns)]
        df["ticker"] = label
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        rows.append(df[OUT_COLS])
        print(f"  got {len(df)} rows ({df['date'].min()} -> {df['date'].max()})")
    except Exception as e:
        print(f"  ERROR {tkr}: {e}")

if not rows:
    print("Nothing pulled.")
    raise SystemExit(1)

delta = pd.concat(rows, ignore_index=True)
print(f"\nPulled {len(delta):,} rows across {delta['ticker'].nunique()} series.")

if NO_WRITE:
    print("NO_WRITE -> per-series range:")
    print(delta.groupby("ticker")["date"].agg(["min", "max", "count"]).to_string())
    raise SystemExit(0)

if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
    existing = pd.read_csv(out_path, dtype={"date": str})
    combined = pd.concat([existing[OUT_COLS], delta], ignore_index=True)
else:
    combined = delta
combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
combined.to_csv(out_path, index=False)
print(f"WROTE {dataset}.csv: {len(combined):,} rows, {combined['ticker'].nunique()} series, "
      f"{combined['date'].min()} -> {combined['date'].max()}")
