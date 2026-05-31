"""I3-A supplement: scan every monthly snapshot for the WORST-CASE default-path
concentration, to find a date where the dormant 10%/25% caps WOULD have bound.

Two book constructions per snapshot (account=$1M, cash-secured, one contract/name):
  (1) top-N-by-ev_dollars (the naive disciplined book) for N in {10,20,40}
  (2) take-all-positive-EV (most aggressive a gate-less path admits)

Reports the max single-name % and max sector % each reaches, and flags
snapshots where a cap would have fired. READ-ONLY.
"""
from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402
import pandas as pd  # noqa: E402

SNAP = os.path.join(HERE, "snapshots")
ACCOUNT = 1_000_000.0


def sector_for(r) -> str:
    s = r.get("sector")
    if isinstance(s, str) and s not in ("", "Unknown", "nan"):
        return s
    return L.sector_of(r["ticker"])


def book_conc(df: pd.DataFrame, top_n: int | None) -> dict:
    pos = df[df["ev_dollars"] > 0].copy()
    if pos.empty:
        return {"n": 0}
    pos = pos.sort_values("ev_dollars", ascending=False).drop_duplicates("ticker")
    pos["sector_resolved"] = pos.apply(sector_for, axis=1)
    book, cash = [], ACCOUNT
    for _, r in pos.iterrows():
        coll = float(r["strike"]) * 100.0
        if coll > cash:
            continue
        book.append(r)
        cash -= coll
        if top_n is not None and len(book) >= top_n:
            break
    if not book:
        return {"n": 0}
    b = pd.DataFrame(book)
    b["coll"] = b["strike"] * 100.0
    name_pct = (b["coll"] / ACCOUNT)
    sec_pct = b.groupby("sector_resolved")["coll"].sum() / ACCOUNT
    return {
        "n": len(b),
        "max_name_pct": float(name_pct.max()),
        "max_name": b.loc[name_pct.idxmax(), "ticker"],
        "max_sector_pct": float(sec_pct.max()),
        "max_sector": sec_pct.idxmax(),
        "n_name_over10": int((name_pct > 0.10).sum()),
        "n_sec_over25": int((sec_pct > 0.25).sum()),
    }


def main() -> None:
    files = sorted(glob.glob(os.path.join(SNAP, "put_*.parquet")))
    print(f"Scanning {len(files)} snapshots (account=${ACCOUNT:,.0f}, cash-secured)")
    print("=" * 100)
    worst_name = (0.0, None, None)
    worst_sec = (0.0, None, None)
    breaches = []
    for f in files:
        nm = os.path.basename(f)
        df = pd.read_parquet(f)
        if "ev_dollars" not in df.columns or len(df) == 0:
            print(f"  (skip {nm}: no ev_dollars / empty, cols={list(df.columns)[:6]})")
            continue
        for tag, tn in (("top10", 10), ("top20", 20), ("top40", 40), ("ALLpos", None)):
            c = book_conc(df, tn)
            if c["n"] == 0:
                continue
            if c["max_name_pct"] > worst_name[0]:
                worst_name = (c["max_name_pct"], f"{nm}/{tag}", c["max_name"])
            if c["max_sector_pct"] > worst_sec[0]:
                worst_sec = (c["max_sector_pct"], f"{nm}/{tag}", c["max_sector"])
            if c["n_name_over10"] > 0 or c["n_sec_over25"] > 0:
                breaches.append((nm, tag, c))

    print(f"\nWORST single-name %: {worst_name[0]:.1%} at {worst_name[1]} ({worst_name[2]})")
    print(f"WORST sector %:      {worst_sec[0]:.1%} at {worst_sec[1]} ({worst_sec[2]})")
    print(f"\n# snapshot/book combos where a dormant cap WOULD have fired: {len(breaches)}")
    print("\nAll breaching combos (R10 single-name>10% or R9 sector>25%):")
    for nm, tag, c in breaches:
        flags = []
        if c["n_name_over10"]:
            flags.append(f"R10x{c['n_name_over10']} (max {c['max_name']} {c['max_name_pct']:.1%})")
        if c["n_sec_over25"]:
            flags.append(f"R9x{c['n_sec_over25']} (max {c['max_sector']} {c['max_sector_pct']:.1%})")
        print(f"  {nm} [{tag}] n={c['n']}: {'; '.join(flags)}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
