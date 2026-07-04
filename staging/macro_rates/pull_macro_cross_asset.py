"""Bucket D — macro / rates / FX / commodities / global-vol index series (BDH PX_LAST).
Bloomberg lab 2026-06-18. All tickers FLDS-verified entitled (recent slice non-null).
Blocked substitutes recorded: SOFR Index->SOFRRATE, BESIUSD->CESIUSD/CESIG10, ZQ futures->FF1.
xbbg 1.3.0 narwhals tidy -> pivot to wide. Deep history (each series returns from inception). Staging only.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2000-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)

SPECS = {
    "ois_sofr_curve.csv": {
        "ois_1m": "USSOA Curncy", "ois_3m": "USSOC Curncy", "ois_6m": "USSOF Curncy",
        "ois_1y": "USSO1 Curncy", "ois_2y": "USSO2 Curncy", "ois_5y": "USSO5 Curncy",
        "ois_10y": "USSO10 Curncy", "ois_30y": "USSO30 Curncy",
        "sofr_on": "SOFRRATE Index", "sofr_1y": "USOSFR1 Curncy", "sofr_2y": "USOSFR2 Curncy",
        "sofr_5y": "USOSFR5 Curncy", "sofr_10y": "USOSFR10 Curncy",
    },
    "real_yields.csv": {
        "tips_2y": "USGGT02Y Index", "tips_5y": "USGGT05Y Index", "tips_10y": "USGGT10Y Index",
        "tips_30y": "USGGT30Y Index", "infl_swap_2y": "USSWIT2 Curncy",
        "infl_swap_5y": "USSWIT5 Curncy", "infl_swap_10y": "USSWIT10 Curncy",
    },
    "fed_funds.csv": {"fed_target": "FDTR Index", "ff_fut_front": "FF1 Comdty"},
    "macro_surprise.csv": {"citi_surprise_usd": "CESIUSD Index", "citi_surprise_g10": "CESIG10 Index"},
    "fx.csv": {"dxy": "DXY Curncy", "eurusd": "EURUSD Curncy", "usdjpy": "USDJPY Curncy", "gbpusd": "GBPUSD Curncy"},
    "commodities.csv": {"wti": "CL1 Comdty", "gold": "GC1 Comdty", "copper": "HG1 Comdty", "natgas": "NG1 Comdty"},
    "global_vol.csv": {
        "vstoxx": "V2X Index", "vhsi": "VHSI Index", "vnky": "VNKY Index", "vkospi": "VKOSPI Index",
        "cdx_ig_5y": "IBOXUMAE Index", "cdx_hy_5y": "IBOXHYSE Index",
    },
}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


def pull_wide(ticker_map):
    cols = {}
    for col, tkr in ticker_map.items():
        d = native(blp.bdh(tkr, ["PX_LAST"], START, END))
        if not {"date", "value"}.issubset(d.columns) or not len(d):
            print(f"    WARN {tkr} empty — skipped")
            continue
        s = d[d["value"].notna()].set_index(pd.to_datetime(d[d["value"].notna()]["date"]))["value"].astype(float)
        cols[col] = s
    wide = pd.DataFrame(cols).sort_index()
    wide.index.name = "date"
    return wide.reset_index()


for fname, tmap in SPECS.items():
    w = pull_wide(tmap)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w.to_csv(os.path.join(OUT, fname), index=False)
    rep = []
    for c in [c for c in w.columns if c != "date"]:
        v = pd.to_numeric(w[c], errors="coerce"); nn = v.notna()
        rep.append(f"{c}={int(nn.sum())}[{v[nn].min():.3g},{v[nn].max():.3g}]")
    print(f"{fname}: {len(w)} rows {w['date'].min()}..{w['date'].max()}\n    " + " ".join(rep))
