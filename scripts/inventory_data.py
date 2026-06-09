"""One-off verified data-inventory pass over Bloomberg + Theta corpora.

Reads ACTUAL files (CSV date columns, parquet footers/filenames) to report
type + verified date range + row/file/ticker counts. No values are taken from
docs. Run with the repo venv:  .venv/Scripts/python.exe scripts/inventory_data.py
Writes a JSON blob to data_processed/_inventory_scan.json for the markdown doc.
"""

import csv
import glob
import gzip
import json
import os
import re

import pyarrow.parquet as pq

ROOT = "C:/Users/merty/Desktop/smart-wheel-engine"
BBG = f"{ROOT}/data/bloomberg"
DEEP = f"{ROOT}/data/bloomberg/deep"
THETA = f"{ROOT}/data_processed/theta"
OUT = {}


def csv_date_range(
    path,
    date_field_candidates=("date", "Date", "DATE", "as_of", "asof"),
    ticker_field=("ticker", "Ticker", "symbol", "Symbol"),
    opener=open,
):
    """Stream a (optionally gz) CSV; return rows, min/max of first matching date
    column, distinct ticker count. Streaming keeps memory flat on 80MB files."""
    rows = 0
    dmin = dmax = None
    tickers = set()
    dcol = tcol = None
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return {"rows": 0, "date_min": None, "date_max": None, "tickers": 0, "cols": []}
        lower = [h.strip().lower() for h in header]
        for cand in date_field_candidates:
            if cand.lower() in lower:
                dcol = lower.index(cand.lower())
                break
        for cand in ticker_field:
            if cand.lower() in lower:
                tcol = lower.index(cand.lower())
                break
        for row in r:
            rows += 1
            if dcol is not None and dcol < len(row):
                v = row[dcol].strip()
                if v:
                    if dmin is None or v < dmin:
                        dmin = v
                    if dmax is None or v > dmax:
                        dmax = v
            if tcol is not None and tcol < len(row):
                tickers.add(row[tcol].strip())
    return {
        "rows": rows,
        "date_min": dmin,
        "date_max": dmax,
        "tickers": len(tickers),
        "date_col": header[dcol] if dcol is not None else None,
        "cols": header,
    }


def pq_rows(path):
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0


def pq_col_minmax(path, col):
    """Pull min/max for a column from parquet row-group stats (no data decode)."""
    try:
        md = pq.ParquetFile(path).metadata
        ci = md.schema.names.index(col)
        lo = hi = None
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ci).statistics
            if st is None or not st.has_min_max:
                return None, None
            if lo is None or st.min < lo:
                lo = st.min
            if hi is None or st.max > hi:
                hi = st.max
        return lo, hi
    except Exception:
        return None, None


def pq_date_col_range(path, candidates=("date", "snapshot_date", "created", "last_trade")):
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    for c in candidates:
        if c in names:
            t = pf.read(columns=[c]).column(0).to_pylist()
            t = [str(x)[:10] for x in t if x is not None]
            if t:
                return c, min(t), max(t)
    return None, None, None


# ---------- Bloomberg monolith CSVs ----------
print("== Bloomberg monoliths ==")
bbg = {}
for path in sorted(glob.glob(f"{BBG}/*.csv")):
    name = os.path.basename(path)
    info = csv_date_range(path)
    info["bytes"] = os.path.getsize(path)
    bbg[name] = info
    print(
        f"  {name}: rows={info['rows']} {info['date_min']}->{info['date_max']} "
        f"tickers={info['tickers']} (datecol={info.get('date_col')})"
    )
OUT["bloomberg_monoliths"] = bbg

# ---------- Bloomberg deep-history gz slices ----------
print("== Bloomberg deep gz ==")
deep = {}
for path in sorted(glob.glob(f"{DEEP}/*.csv.gz")) + sorted(glob.glob(f"{DEEP}/*.csv")):
    name = os.path.basename(path)
    opener = gzip.open if path.endswith(".gz") else open
    info = csv_date_range(path, opener=opener)
    info["bytes"] = os.path.getsize(path)
    deep[name] = info
    print(
        f"  {name}: rows={info['rows']} {info['date_min']}->{info['date_max']} tickers={info['tickers']}"
    )
OUT["bloomberg_deep"] = deep

# ---------- Theta ----------
print("== Theta ==")
theta = {}


def filename_date_set(paths, rx=re.compile(r"_(\d{8})")):
    dates, syms = set(), set()
    for p in paths:
        b = os.path.basename(p)
        m = rx.search(b)
        if m:
            dates.add(m.group(1))
        syms.add(b.split("_")[0])
    return dates, syms


# snapshot-style datasets: date encoded in filename, total rows from footers
for ds, pattern in [
    ("chains", f"{THETA}/chains/*.parquet"),
    ("iv_surface", f"{THETA}/iv_surface/*.parquet"),
    ("index_options_chains", f"{THETA}/index_options_chains/*.parquet"),
    ("index_options_surfaces", f"{THETA}/index_options_surfaces/*.parquet"),
    ("option_ohlc", f"{THETA}/option_ohlc/*.parquet"),
]:
    paths = glob.glob(pattern)
    dates, syms = filename_date_set(paths)
    total = sum(pq_rows(p) for p in paths)
    d = {
        "files": len(paths),
        "snapshot_dates": sorted(dates),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "symbols": len(syms),
        "rows": total,
    }
    theta[ds] = d
    print(
        f"  {ds}: files={d['files']} snaps {d['date_min']}->{d['date_max']} syms={d['symbols']} rows={total}"
    )

# per-ticker time series: read date column
for ds in ["iv_history", "stocks_eod"]:
    paths = glob.glob(f"{THETA}/{ds}/*.parquet")
    dmin = dmax = None
    total = 0
    for p in paths:
        total += pq_rows(p)
        c, lo, hi = pq_date_col_range(p, candidates=("date",))
        if lo and (dmin is None or lo < dmin):
            dmin = lo
        if hi and (dmax is None or hi > dmax):
            dmax = hi
    theta[ds] = {
        "files": len(paths),
        "tickers": len(paths),
        "date_min": dmin,
        "date_max": dmax,
        "rows": total,
    }
    print(f"  {ds}: tickers={len(paths)} {dmin}->{dmax} rows={total}")

# vix_family single file
p = f"{THETA}/vix_family/vix_family.parquet"
if os.path.exists(p):
    c, lo, hi = pq_date_col_range(p, candidates=("date",))
    syms = sorted(set(pq.ParquetFile(p).read(columns=["symbol"]).column(0).to_pylist()))
    theta["vix_family"] = {
        "files": 1,
        "symbols": syms,
        "date_min": lo,
        "date_max": hi,
        "rows": pq_rows(p),
    }
    print(f"  vix_family: {lo}->{hi} rows={pq_rows(p)} symbols={syms}")

# iv_surface_history: date from partition dir name; ticker from partition
ivsh = glob.glob(f"{THETA}/iv_surface_history/ticker=*/year=*/date=*.parquet")
dates = set()
tk = set()
for p in ivsh:
    m = re.search(r"date=(\d{4}-\d{2}-\d{2})", p)
    if m:
        dates.add(m.group(1))
    m2 = re.search(r"ticker=([^/\\]+)", p)
    if m2:
        tk.add(m2.group(1))
theta["iv_surface_history"] = {
    "files": len(ivsh),
    "tickers": sorted(tk),
    "date_min": min(dates) if dates else None,
    "date_max": max(dates) if dates else None,
    "rows": sum(pq_rows(p) for p in ivsh),
}
print(
    f"  iv_surface_history: files={len(ivsh)} tickers={len(tk)} "
    f"{theta['iv_surface_history']['date_min']}->{theta['iv_surface_history']['date_max']}"
)

# option_history (+ banded backup): expiration span from partitions, observation
# span from 'created' column stats, totals from footers.
for ds in ["option_history", "option_history_banded_backup_2026-06-01"]:
    files = glob.glob(f"{THETA}/{ds}/ticker=*/expiration=*/*.parquet")
    tk = set()
    exps = set()
    for p in files:
        m = re.search(r"ticker=([^/\\]+)", p)
        if m:
            tk.add(m.group(1))
        m2 = re.search(r"expiration=(\d{8})", p)
        if m2:
            exps.add(m2.group(1))
    total = sum(pq_rows(p) for p in files)
    # observation window: sample 'created'/'last_trade' stats from up to 200 files
    obs_lo = obs_hi = None
    for p in files[:: max(1, len(files) // 200)]:
        for col in ("created", "last_trade"):
            lo, hi = pq_col_minmax(p, col)
            if lo is not None:
                lo, hi = str(lo)[:10], str(hi)[:10]
                if obs_lo is None or lo < obs_lo:
                    obs_lo = lo
                if obs_hi is None or hi > obs_hi:
                    obs_hi = hi
                break
    theta[ds] = {
        "files": len(files),
        "tickers": len(tk),
        "expiration_min": min(exps) if exps else None,
        "expiration_max": max(exps) if exps else None,
        "obs_min_sampled": obs_lo,
        "obs_max_sampled": obs_hi,
        "rows": total,
    }
    print(
        f"  {ds}: files={len(files)} tickers={len(tk)} "
        f"exp {theta[ds]['expiration_min']}->{theta[ds]['expiration_max']} "
        f"obs~{obs_lo}->{obs_hi} rows={total}"
    )

OUT["theta"] = theta

with open(f"{ROOT}/data_processed/_inventory_scan.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nWROTE data_processed/_inventory_scan.json")
