"""
VIX term structure (spot / 3M / 6M) -> data/bloomberg/vix_term_structure.csv

Cheap single-index pull: BDH PX_LAST for VIX / VIX3M / VIX6M from a 1990 floor
to END_DATE. Each index is pulled SEPARATELY so one bad ticker cannot null the
whole batch. Merged onto the existing file (new wins) and sorted.

Schema (preserved EXACTLY, WIDE): date,vix,vix_3m,vix_6m
  vix     <- VIX Index   (inception ~1990)
  vix_3m  <- VIX3M Index (inception ~2007; NaN before)
  vix_6m  <- VIX6M Index (inception ~2008; NaN before)

Env: SWE_PULL_END, SWE_PULL_FLOOR, SWE_PULL_NO_WRITE.
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
FLOOR = os.environ.get("SWE_PULL_FLOOR") or "1990-01-01"
NO_WRITE = bool(os.environ.get("SWE_PULL_NO_WRITE"))
SERIES = [("VIX Index", "vix"), ("VIX3M Index", "vix_3m"), ("VIX6M Index", "vix_6m")]
OUT_COLS = ["date", "vix", "vix_3m", "vix_6m"]
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg", "vix_term_structure.csv")


def to_native(obj):
    return obj.to_native() if hasattr(obj, "to_native") else obj


frames = []
for tkr, label in SERIES:
    print(f"Pulling {tkr} PX_LAST {FLOOR} -> {END} ...", flush=True)
    try:
        raw = to_native(blp.bdh(tickers=tkr, flds="PX_LAST", start_date=FLOOR, end_date=END))
        if raw is None or len(raw) == 0:
            print(f"  (no data for {tkr})")
            continue
        df = raw.rename(columns={"value": label})[["date", label]] if "value" in raw.columns else None
        if df is None:
            # already-wide fallback
            df = raw.copy()
            df.columns = ["date", label] if len(raw.columns) == 2 else raw.columns
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        frames.append(df[["date", label]])
        print(f"  got {len(df)} rows ({df['date'].min()} -> {df['date'].max()})")
    except Exception as e:
        print(f"  ERROR {tkr}: {e}")

if not frames:
    print("Nothing pulled.")
    raise SystemExit(1)

wide = frames[0]
for f in frames[1:]:
    wide = wide.merge(f, on="date", how="outer")
for c in OUT_COLS:
    if c not in wide.columns:
        wide[c] = pd.NA
wide = wide[OUT_COLS].sort_values("date").reset_index(drop=True)
print(f"\nPulled wide: {len(wide)} rows ({wide['date'].min()} -> {wide['date'].max()})")

if NO_WRITE:
    print("NO_WRITE -> sample:")
    print(wide.head(4).to_string(index=False))
    print(wide.tail(4).to_string(index=False))
    raise SystemExit(0)

if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
    existing = pd.read_csv(out_path, dtype={"date": str})
    combined = pd.concat([existing[OUT_COLS], wide], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
else:
    combined = wide
combined = combined.sort_values("date").reset_index(drop=True)
combined.to_csv(out_path, index=False)
print(f"WROTE vix_term_structure.csv: {len(combined)} rows ({combined['date'].min()} -> {combined['date'].max()})")
