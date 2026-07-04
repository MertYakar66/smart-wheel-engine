"""Broad pull — macro/vol/rates index series (BDH PX_LAST). Bloomberg lab 2026-06-17.

All FLDS-verified via a recent slice (each returns sane data; NFCI errors → omitted).
Deep history (2004→today; each series returns from its own inception). Staging only —
new files, monoliths untouched. xbbg 1.3.0 returns a narwhals tidy frame → pivot to wide.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2004-01-01", "2026-06-17"
OUT = os.path.dirname(__file__)

# target_file -> {column_name: bloomberg_ticker}
SPECS = {
    "sp500_vol_indices.csv": {
        "vix": "VIX Index", "vvix": "VVIX Index", "skew": "SKEW Index", "vxn": "VXN Index",
        "rvx": "RVX Index", "ovx": "OVX Index", "gvz": "GVZ Index", "move": "MOVE Index",
        "vxeem": "VXEEM Index", "cvix": "CVIX Index",
    },
    "spx_correlation.csv": {"cor1m": "COR1M Index", "cor3m": "COR3M Index", "cor6m": "COR6M Index"},
    "credit_spreads.csv": {"ig_oas": "LUACOAS Index", "hy_oas": "LF98OAS Index"},
}


def pull_wide(ticker_map):
    cols = {}
    for col, tkr in ticker_map.items():
        nw = blp.bdh(tkr, ["PX_LAST"], START, END)
        d = nw.to_native() if hasattr(nw, "to_native") else nw
        if not {"date", "value"}.issubset(d.columns) or not len(d):
            print(f"    WARN {tkr} empty — skipped")
            continue
        s = d.set_index(pd.to_datetime(d["date"]))["value"].astype(float)
        cols[col] = s
    wide = pd.DataFrame(cols).sort_index()
    wide.index.name = "date"
    return wide.reset_index()


for fname, tmap in SPECS.items():
    w = pull_wide(tmap)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w.to_csv(os.path.join(OUT, fname), index=False)
    # validation: coverage + sane bands per column
    rep = []
    for c in [c for c in w.columns if c != "date"]:
        v = pd.to_numeric(w[c], errors="coerce")
        nn = v.notna()
        rep.append(f"{c}={int(nn.sum())}rows[{v[nn].min():.2f},{v[nn].max():.2f}]")
    print(f"{fname}: {len(w)} rows {w['date'].min()}..{w['date'].max()} | " + " ".join(rep))
