"""
#14 - Liquidity Metrics (critical for execution)
Pull S&P 500 liquidity metrics from Bloomberg -> data/bloomberg/sp500_liquidity.csv

Refresh-aware (frugal with the metered Desktop API): if the output CSV already
exists, only the missing window (last date in file + 1 -> END_DATE) is pulled
and merged/deduped onto the existing rows. On a fresh machine it does the full
historical pull from START_DATE_FULL.

Columns (committed schema, preserved EXACTLY): date,avg_vol_30d,turnover,shares_out,ticker
  - shares_out comes from EQY_SH_OUT and MUST NOT be dropped.
  - column order matches the committed file.
Ticker format: "AAPL UW" (exchange code, no " Equity" suffix).

xbbg >= 1.2 returns long-format narwhals frames (ticker,date,field,value); we
convert to native pandas via .to_native() and pivot to wide.

Env knobs (testing/ops only):
  SWE_PULL_LIMIT     pull only the first N members (smoke test). 0/unset = all.
  SWE_PULL_NO_WRITE  if set, pull + reshape + print summary but skip merge/write.
"""

import os

import pandas as pd
from xbbg import blp

START_DATE_FULL = "2015-01-01"
END_DATE = "2026-06-02"
FIELDS = ["VOLUME_AVG_30D", "TURNOVER", "EQY_SH_OUT"]
FIELD_MAP = {"VOLUME_AVG_30D": "avg_vol_30d", "TURNOVER": "turnover", "EQY_SH_OUT": "shares_out"}
OUT_COLS = ["date", "avg_vol_30d", "turnover", "shares_out", "ticker"]
CHUNK_SIZE = 30

LIMIT = int(os.environ.get("SWE_PULL_LIMIT", "0") or "0")
NO_WRITE = bool(os.environ.get("SWE_PULL_NO_WRITE"))

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg", "sp500_liquidity.csv")


def to_native(obj):
    return obj.to_native() if hasattr(obj, "to_native") else obj


existing = None
start_date = START_DATE_FULL
if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
    existing = pd.read_csv(out_path, dtype={"date": str})
    last = existing["date"].max()
    start_date = (pd.to_datetime(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Existing CSV: {len(existing):,} rows, last date {last}. Delta pull {start_date} -> {END_DATE}")
else:
    print(f"No existing CSV. Full pull {start_date} -> {END_DATE}")

if pd.to_datetime(start_date) > pd.to_datetime(END_DATE):
    print("Already up to date; nothing to pull.")
    raise SystemExit(0)

print("Getting S&P 500 members...")
members = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
cand = [c for c in members.columns if "member" in c.lower() and "ticker" in c.lower()]
mcol = cand[0] if cand else "Member Ticker and Exchange Code"
tickers = [t + " Equity" for t in members[mcol].tolist()]
if LIMIT:
    tickers = tickers[:LIMIT]
    print(f"SWE_PULL_LIMIT active -> {len(tickers)} tickers")
print(f"Pulling {len(tickers)} tickers")

all_chunks = []
for i in range(0, len(tickers), CHUNK_SIZE):
    chunk = tickers[i : i + CHUNK_SIZE]
    print(f"Pulling {i + 1}-{min(i + CHUNK_SIZE, len(tickers))} of {len(tickers)}...")
    try:
        raw = to_native(
            blp.bdh(tickers=chunk, flds=FIELDS, start_date=start_date, end_date=END_DATE, Fill="P")
        )
        if raw is None or len(raw) == 0:
            print("  (no data in window)")
            continue
        wide = raw.pivot_table(
            index=["date", "ticker"], columns="field", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(columns=FIELD_MAP)
        for c in ("avg_vol_30d", "turnover", "shares_out"):
            if c not in wide.columns:
                wide[c] = pd.NA
        wide["ticker"] = wide["ticker"].str.replace(" Equity", "", regex=False)
        wide["date"] = pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")
        all_chunks.append(wide[OUT_COLS])
        print(f"  got {len(wide)} rows")
    except Exception as e:
        print(f"  Error on chunk {i}: {e}")

if not all_chunks:
    print("No new rows pulled.")
    raise SystemExit(0)

delta = pd.concat(all_chunks, ignore_index=True)
print(f"Delta rows: {len(delta):,}  ({delta['date'].min()} -> {delta['date'].max()}, {delta['ticker'].nunique()} tickers)")

if NO_WRITE:
    print("SWE_PULL_NO_WRITE set -> skipping merge/write. Sample:")
    print(delta.head(8).to_string())
    raise SystemExit(0)

combined = pd.concat([existing[OUT_COLS], delta], ignore_index=True) if existing is not None else delta
combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
combined.to_csv(out_path, index=False)
print(
    f"\nDone! {len(combined):,} rows. Tickers: {combined['ticker'].nunique()}. "
    f"Range {combined['date'].min()} -> {combined['date'].max()}"
)
