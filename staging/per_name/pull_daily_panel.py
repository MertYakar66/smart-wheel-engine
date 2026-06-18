"""Bucket C — per-name DAILY panels (resumable, append, chunked). Bloomberg lab 2026-06-18.

Usage: python pull_daily_panel.py <dataset> <start_idx> <end_idx>
  dataset in {returns_micro, vol_term_rv}; indices slice the sorted ohlcv universe (recent-first
  not needed for full-history daily — resume by ticker-range). Appends to <dataset>.csv (header on
  first batch). Round to keep gzip < GitHub 100MB. All fields FLDS-verified entitled (non-null).
xbbg 1.3.0 narwhals tidy -> pivot.
"""
import os
import sys
import pandas as pd
from xbbg import blp

START, END = "2010-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
CHUNK = 30

DATASETS = {
    "returns_micro": {
        "fields": ["TOT_RETURN_INDEX_NET_DVDS", "PX_BID", "PX_ASK"],
        "map": {"TOT_RETURN_INDEX_NET_DVDS": "tot_return", "PX_BID": "px_bid", "PX_ASK": "px_ask"},
        "round": {"tot_return": 4, "px_bid": 4, "px_ask": 4},
    },
    "vol_term_rv": {
        "fields": [f"{t}_IMPVOL_100.0%MNY_DF" for t in ["30DAY", "60DAY", "3MTH", "6MTH", "12MTH", "24MTH"]]
                  + [f"VOLATILITY_{n}D" for n in [10, 20, 30, 60, 90, 120, 180, 260]],
        "map": {"30DAY_IMPVOL_100.0%MNY_DF": "atm_iv_30d", "60DAY_IMPVOL_100.0%MNY_DF": "atm_iv_60d",
                "3MTH_IMPVOL_100.0%MNY_DF": "atm_iv_90d", "6MTH_IMPVOL_100.0%MNY_DF": "atm_iv_180d",
                "12MTH_IMPVOL_100.0%MNY_DF": "atm_iv_365d", "24MTH_IMPVOL_100.0%MNY_DF": "atm_iv_730d",
                **{f"VOLATILITY_{n}D": f"rv_{n}d" for n in [10, 20, 30, 60, 90, 120, 180, 260]}},
        "round": None,  # 2dp for all vol cols, applied below
    },
    "options_sentiment": {
        "fields": ["PUT_CALL_OPEN_INTEREST_RATIO", "PUT_CALL_VOLUME_RATIO_CUR_DAY",
                   "OPEN_INT_TOTAL_CALL", "OPEN_INT_TOTAL_PUT", "NEWS_SENTIMENT_DAILY_AVG"],
        "map": {"PUT_CALL_OPEN_INTEREST_RATIO": "pc_oi_ratio", "PUT_CALL_VOLUME_RATIO_CUR_DAY": "pc_vol_ratio",
                "OPEN_INT_TOTAL_CALL": "oi_call", "OPEN_INT_TOTAL_PUT": "oi_put",
                "NEWS_SENTIMENT_DAILY_AVG": "news_sent"},
        "round": {"pc_oi_ratio": 4, "pc_vol_ratio": 4, "oi_call": 0, "oi_put": 0, "news_sent": 4},
    },
}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


def main():
    ds, s, e = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = DATASETS[ds]
    outfile = os.path.join(OUT, f"{ds}.csv")
    uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
    batch = uni[s:e]
    cols = ["date", "ticker"] + list(cfg["map"].values())
    first = not os.path.exists(outfile)
    print(f"{ds}: universe {len(uni)}; batch [{s}:{e}] = {len(batch)} names; file_exists={not first}")
    total = 0
    for i in range(0, len(batch), CHUNK):
        ch = batch[i:i + CHUNK]
        d = native(blp.bdh(ch, cfg["fields"], START, END))
        if not {"ticker", "date", "field", "value"}.issubset(d.columns):
            print(f"  [{s+i}] empty shape"); continue
        d = d[d["value"].notna()].copy()
        d["col"] = d["field"].map(cfg["map"])
        d = d[d["col"].notna()]
        d["ticker"] = d["ticker"].str.replace(" Equity", "", regex=False)
        w = d.pivot_table(index=["date", "ticker"], columns="col", values="value", aggfunc="first").reset_index()
        for c in cols:
            if c not in w.columns:
                w[c] = pd.NA
        w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
        valcols = [c for c in cols if c not in ("date", "ticker")]
        if cfg["round"]:
            for c, nd in cfg["round"].items():
                if c in w.columns:
                    w[c] = pd.to_numeric(w[c], errors="coerce").round(nd)
        else:
            w[valcols] = w[valcols].apply(pd.to_numeric, errors="coerce").round(2)
        w = w[cols].sort_values(["ticker", "date"])
        w.to_csv(outfile, mode="a", header=first, index=False)
        first = False
        total += len(w)
        cov = w[valcols].notna().mean().mean() if len(w) else 0
        print(f"  [{s+i}:{s+i+len(ch)}] {ch[0]}..{ch[-1]}: +{len(w)} rows cov {cov:.0%} (total {total})")
    print(f"DONE {ds} [{s}:{e}] total {total} rows")


if __name__ == "__main__":
    main()
