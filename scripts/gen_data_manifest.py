#!/usr/bin/env python3
"""Generate data/data_manifest.json — the single machine-readable map of every
engine data file: on-disk path, size, sha256, engine role, and its Google Drive
home. Run from repo root. sha256 is computed for files present in the working
tree; deep-archive slices (gitignored, Drive-only) are listed from the known
Drive inventory with sha256=null."""

import hashlib
import json
import os
import subprocess

ROOT = os.getcwd()
BBG = "data/bloomberg"

# Drive folder anchors (stable folder IDs; per-file IDs are resolved by title at
# fetch time so a re-upload never breaks the manifest).
DRIVE_ROOT = "1xpRvaQglsmcUuTKgVKHR39_3H-vbdIFh"  # Drive: .../data/bloomberg
FOLDERS = {
    "data/bloomberg": DRIVE_ROOT,
    "data/bloomberg/broad_pull/per_name": "1GF0HCnt9ch2Lj8-vjc1ftELnd-nbDDhG",
    "data/bloomberg/broad_pull/iv_surface": "1JdolIi5nswEl5TdV-z2fkRtOCVFIUn3r",
    "data/bloomberg/broad_pull/short_interest": "1aH6b-yq9wfRKKaoxr5KmNM6_7KDXPlXF",
    "data/bloomberg/broad_pull/dividend_pit": "17144fuEZ3pgf5X5i6SyjBZqMWnGhsa2K",
    "data/bloomberg/broad_pull/macro_vol": "1uAYXlAoIvQDzvQWnZY8LZirqSvJkaUdA",
    "data/bloomberg/broad_pull/macro_rates": "1mq2wY26_DqzTn6HrBZX5F193iw5brlnE",
    "data/bloomberg/broad_pull/macro_calendar": "1X0zhgUyujmo7Z1Swl4vpc-liZUaxyGuk",
    "data/bloomberg/deep": "1m_9LQNtbHzQo7MG5t3OxAINCXiwkhkna",
}

# _FILES served contract (engine/data_connector.py) — basename -> engine key.
SERVED = {
    "sp500_ohlcv.csv": "ohlcv",
    "sp500_vol_iv_full.csv": "vol_iv",
    "sp500_dividends.csv": "dividends",
    "sp500_earnings.csv": "earnings",
    "treasury_yields.csv": "treasury",
    "vix_term_structure.csv": "vix",
    "sp500_fundamentals.csv": "fundamentals",
    "sp500_credit_risk.csv": "credit_risk",
    "sp500_liquidity.csv": "liquidity",
    "sp500_corporate_actions.csv": "corporate_actions",
}
# in-repo producer (edit end_date / SWE_PULL_END), where one exists
PRODUCER = {
    "sp500_ohlcv.csv": "scripts/pull_ohlcv.py",
    "sp500_vol_iv_full.csv": "scripts/pull_vol_iv.py",
    "sp500_liquidity.csv": "scripts/pull_liquidity.py",
    "treasury_yields.csv": "scripts/pull_treasury_yields.py",
    "vix_term_structure.csv": "scripts/pull_vix_term_structure.py",
    "sp500_dividends.csv": "scripts/pull_dividends.py",
    "sp500_corporate_actions.csv": "scripts/pull_corporate_actions.py",
    "sp500_fundamentals.csv": "scripts/pull_snapshots.py",
    "sp500_credit_risk.csv": "scripts/pull_snapshots.py",
    "sp500_snapshot_bdp.csv": "scripts/pull_snapshot_bdp.py",
    "sp500_dividend_yield_pit.csv": "scripts/pull_dividend_yield_pit.py",
    "sp500_short_interest.csv": "scripts/pull_short_interest.py",
    "sp500_historical_fundamentals.csv": "scripts/pull_historical_fundamentals.py",
    "sp500_index_membership.csv": "scripts/pull_index_membership.py",
    "sp500_sector_etfs.csv": "scripts/pull_sector_etfs.py",
    "sp500_macro.csv": "scripts/pull_macro.py",
    "sp500_vix_full.csv": "scripts/pull_vix_full.py",
}
# deep-archive slices (gitignored, Drive-only) — title -> size (from Drive)
DEEP = {
    "sp500_ohlcv__1994_2018.csv.gz": 31705057,
    "sp500_ohlcv__delisted.csv.gz": 34075979,
    "sp500_vol_iv_full__1994_2012.csv.gz": 28297508,
    "sp500_vol_iv_full__2012_2018.csv.gz": 10596777,
    "sp500_vol_iv__delisted.csv.gz": 40025527,
    "sp500_liquidity__1994_2015.csv.gz": 20919884,
    "sp500_liquidity__delisted.csv.gz": 24169257,
    "sp500_iv_surface__2005_2011.csv.gz": 51586372,
    "sp500_iv_surface__2012_2018.csv.gz": 59392342,
    "sp500_iv_surface__2019_2026.csv.gz": 72500413,
    "delisted_status.csv": 71569,
    "ohlcv_dropped_ticks.csv": 8962,
    "ohlcv_dropped_ticks__delisted.csv": 4957,
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# git-tracked files under data/bloomberg on origin/main
tracked = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", "origin/main", "--", BBG], text=True
).splitlines()

files = []
for rel in sorted(tracked):
    if not rel.endswith((".csv", ".gz")):
        continue  # skip EXTRACTION_GUIDE.md etc.
    base = os.path.basename(rel)
    d = os.path.dirname(rel)
    ap = os.path.join(ROOT, rel)
    present = os.path.exists(ap)
    files.append(
        {
            "path": rel,
            "basename": base,
            "size": os.path.getsize(ap) if present else None,
            "sha256": sha256(ap) if present else None,
            "drive_folder": FOLDERS.get(d, "UNKNOWN-" + d),
            "served": base in SERVED,
            "engine_key": SERVED.get(base),
            "producer": PRODUCER.get(base),
            "tier": "git-tracked",
        }
    )

# add deep-archive (Drive-only, opt-in via deep_history)
for title, size in DEEP.items():
    files.append(
        {
            "path": f"{BBG}/deep/{title}",
            "basename": title,
            "size": size,
            "sha256": None,
            "drive_folder": FOLDERS["data/bloomberg/deep"],
            "served": False,
            "engine_key": None,
            "producer": None,
            "tier": "deep-archive (opt-in, gitignored)",
        }
    )

manifest = {
    "schema": 1,
    "note": "Single source of truth for engine data files and their Google Drive home. "
    "The engine reads these from data/bloomberg/ on disk; they are NOT in git "
    "(migrated to Drive). Fetch with scripts/fetch_data.py before running the engine.",
    "drive_root_folder_id": DRIVE_ROOT,
    "drive_folders": FOLDERS,
    "counts": {
        "git_tracked": sum(1 for f in files if f["tier"] == "git-tracked"),
        "deep_archive": len(DEEP),
    },
    "files": files,
}
os.makedirs("data", exist_ok=True)
with open("data/data_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(
    f"wrote data/data_manifest.json: {len(files)} files "
    f"({manifest['counts']['git_tracked']} git-tracked + {manifest['counts']['deep_archive']} deep)"
)
served = [f["basename"] for f in files if f["served"]]
print("served (_FILES):", len(served), sorted(served))
