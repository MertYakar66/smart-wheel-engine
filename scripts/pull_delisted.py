"""
Phase 2: delisted/removed SPX constituents' price-vol backfill (survivorship completion).

Driver: data/bloomberg/_delisted_universe.csv (PIT Bloomberg IDs + first/last in-index).
For each name, pulls ONLY its in-index window [first_in_index - 1Q, last_in_index + 1Q]
(cost discipline -- ~5x smaller than 1990-2026) for three panels, matching each
current-universe monolith's exact field map / ticker format:

  OHLCV    rotated map PX_HIGH->open, PX_LAST->high, PX_LOW->low, PX_OPEN->close,
           PX_VOLUME->volume; ticker "X UN Equity"; no fill. Rotation gate per name
           (stored open==row-max AND low==row-min); bad ticks DROPPED+audited under a
           0.5% per-name valve (name skipped if it exceeds the valve).
  vol_iv   HIST_PUT/CALL_IMP_VOL + VOLATILITY_30/60/90/260D; ticker "X UN"; Fill=P.
           (IV floor ~1994; names gone before then yield no vol_iv -- expected.)
  liquidity VOLUME_AVG_30D/TURNOVER/EQY_SH_OUT; ticker "X UN"; Fill=P.

Outputs (uncompressed working CSVs in SWE_DEL_OUT; gz+push handled separately):
  sp500_ohlcv__delisted.csv, sp500_vol_iv__delisted.csv, sp500_liquidity__delisted.csv,
  ohlcv_dropped_ticks__delisted.csv, delisted_status.csv
Errors per name are logged+skipped (never crash the run).

Env knobs:
  SWE_DEL_OFFSET   start index into the (sorted) universe (default 0).
  SWE_DEL_LIMIT    process N names (default 0 = all).
  SWE_DEL_TICKERS  comma list of tickers to process instead of a slice (demo/repair).
  SWE_DEL_OUT      output dir (default C:/Users/mertmert/deep_scratch/delisted).
  SWE_DEL_APPEND   if set, merge into existing output files (restart-safe); else fresh.
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
from xbbg import blp

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE = os.path.join(REPO, "data", "bloomberg", "_delisted_universe.csv")

OHLCV_FLDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
OHLCV_MAP = {"PX_HIGH": "open", "PX_LAST": "high", "PX_LOW": "low", "PX_OPEN": "close", "PX_VOLUME": "volume"}
OHLCV_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
VOLIV_FLDS = ["HIST_PUT_IMP_VOL", "HIST_CALL_IMP_VOL", "VOLATILITY_30D", "VOLATILITY_60D", "VOLATILITY_90D", "VOLATILITY_260D"]
VOLIV_MAP = {f: f.lower() for f in VOLIV_FLDS}
VOLIV_COLS = ["date", "hist_put_imp_vol", "hist_call_imp_vol", "volatility_30d", "volatility_60d", "volatility_90d", "volatility_260d", "ticker"]
LIQ_FLDS = ["VOLUME_AVG_30D", "TURNOVER", "EQY_SH_OUT"]
LIQ_MAP = {"VOLUME_AVG_30D": "avg_vol_30d", "TURNOVER": "turnover", "EQY_SH_OUT": "shares_out"}
LIQ_COLS = ["date", "avg_vol_30d", "turnover", "shares_out", "ticker"]
VALVE = 0.005


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def _reshape(raw, fmap, suffix: str):
    raw = to_native(raw)
    if raw is None or len(raw) == 0:
        return None
    raw.columns = [c.lower() for c in raw.columns]
    w = raw.pivot_table(index="date", columns="field", values="value", aggfunc="first").reset_index()
    w.columns.name = None
    w = w.rename(columns={c: fmap[c.upper()] for c in w.columns if c.upper() in fmap})
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    return w


def pull_ohlcv(tk, ws, we):
    w = _reshape(blp.bdh(tickers=tk + " Equity", flds=OHLCV_FLDS, start_date=ws, end_date=we), OHLCV_MAP, "")
    if w is None:
        return None
    for c in ("open", "high", "low", "close", "volume"):
        if c not in w.columns:
            w[c] = pd.NA
    w["ticker"] = tk + " Equity"
    return w[OHLCV_COLS]


def pull_voliv(tk, ws, we):
    w = _reshape(blp.bdh(tickers=tk + " Equity", flds=VOLIV_FLDS, start_date=ws, end_date=we, Fill="P"), VOLIV_MAP, "")
    if w is None:
        return None
    for c in VOLIV_MAP.values():
        if c not in w.columns:
            w[c] = pd.NA
    w["ticker"] = tk
    if w[list(VOLIV_MAP.values())].notna().sum().sum() == 0:
        return None
    return w[VOLIV_COLS]


def pull_liq(tk, ws, we):
    w = _reshape(blp.bdh(tickers=tk + " Equity", flds=LIQ_FLDS, start_date=ws, end_date=we, Fill="P"), LIQ_MAP, "")
    if w is None:
        return None
    for c in LIQ_MAP.values():
        if c not in w.columns:
            w[c] = pd.NA
    w["ticker"] = tk
    if w[list(LIQ_MAP.values())].notna().sum().sum() == 0:
        return None
    return w[LIQ_COLS]


def clean_rotation(df):
    """Return (cleaned_df, bad_rows_df, fail_rate) for the rotated OHLCV gate."""
    sub = df.dropna(subset=["open", "high", "low", "close"])
    if len(sub) == 0:
        return df, df.iloc[0:0], 0.0
    ohlc = sub[["open", "high", "low", "close"]]
    badmask = (sub["open"] != ohlc.max(axis=1)) | (sub["low"] != ohlc.min(axis=1))
    bad = sub[badmask]
    rate = len(bad) / len(sub)
    cleaned = df.drop(index=bad.index)
    return cleaned, bad, rate


def gate_violations(df):
    sub = df.dropna(subset=["open", "high", "low", "close"])
    if len(sub) == 0:
        return 0
    ohlc = sub[["open", "high", "low", "close"]]
    return int(((sub["open"] != ohlc.max(axis=1)) | (sub["low"] != ohlc.min(axis=1))).sum())


FILES = {
    "ohlcv": ("sp500_ohlcv__delisted.csv", OHLCV_COLS),
    "voliv": ("sp500_vol_iv__delisted.csv", VOLIV_COLS),
    "liq": ("sp500_liquidity__delisted.csv", LIQ_COLS),
    "dropped": ("ohlcv_dropped_ticks__delisted.csv", OHLCV_COLS + ["which", "window"]),
    "status": ("delisted_status.csv", None),
}
CHECKPOINT = 50


def main():
    out_dir = os.environ.get("SWE_DEL_OUT", r"C:/Users/mertmert/deep_scratch/delisted")
    os.makedirs(out_dir, exist_ok=True)
    offset = int(os.environ.get("SWE_DEL_OFFSET", "0") or "0")
    limit = int(os.environ.get("SWE_DEL_LIMIT", "0") or "0")
    tlist = os.environ.get("SWE_DEL_TICKERS")
    fresh = bool(os.environ.get("SWE_DEL_FRESH"))

    uni = pd.read_csv(UNIVERSE, dtype=str)
    if tlist:
        want = [t.strip() for t in tlist.split(",")]
        uni = uni[uni["ticker"].isin(want)].reset_index(drop=True)
    else:
        uni = uni.iloc[offset: offset + limit] if limit else uni.iloc[offset:]

    # --- RESUME: load existing checkpoint + skip already-done tickers ---
    oh_all, vi_all, lq_all, dropped, status = [], [], [], [], []
    done = set()
    spath = os.path.join(out_dir, FILES["status"][0])
    if not fresh and os.path.exists(spath):
        prev = pd.read_csv(spath, dtype=str)
        done = set(prev["ticker"])
        status = prev.to_dict("records")
        for key in ("ohlcv", "voliv", "liq", "dropped"):
            p = os.path.join(out_dir, FILES[key][0])
            if os.path.exists(p):
                d = pd.read_csv(p, dtype=str)
                if len(d):
                    (oh_all if key == "ohlcv" else vi_all if key == "voliv"
                     else lq_all if key == "liq" else dropped).append(d)
        uni = uni[~uni["ticker"].isin(done)].reset_index(drop=True)
        print(f"RESUME: {len(done)} names already on disk; {len(uni)} remaining", flush=True)

    def flush():
        for key, acc in (("ohlcv", oh_all), ("voliv", vi_all), ("liq", lq_all), ("dropped", dropped)):
            cols = FILES[key][1]
            parts = [p for p in acc if p is not None and len(p)]
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
            if key in ("ohlcv", "voliv", "liq") and len(df):
                df = df.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["ticker", "date"])
            df.to_csv(os.path.join(out_dir, FILES[key][0]), index=False)
        pd.DataFrame(status).to_csv(spath, index=False)

    print(f"Processing {len(uni)} delisted names (offset={offset} limit={limit or 'all'})", flush=True)
    for i, row in enumerate(uni.itertuples(index=False), 1):
        tk = row.ticker
        ws = (pd.Timestamp(row.first_in_index) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        we = (pd.Timestamp(row.last_in_index) + pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        st = {"ticker": tk, "name": row.name, "window": f"{ws}..{we}",
              "ohlcv_rows": 0, "voliv_rows": 0, "liq_rows": 0, "dropped": 0, "status": "ok"}
        try:
            oh = pull_ohlcv(tk, ws, we)
            if oh is not None and len(oh):
                cleaned, bad, rate = clean_rotation(oh)
                if rate > VALVE:
                    st["status"] = f"SKIP_ohlcv_valve({rate:.3%})"
                else:
                    if len(bad):
                        b = bad.copy(); b["which"] = "rotation"; b["window"] = f"{ws}..{we}"
                        dropped.append(b)
                    oh_all.append(cleaned); st["ohlcv_rows"] = len(cleaned); st["dropped"] = len(bad)
        except Exception as e:
            st["status"] = f"ERR_ohlcv:{str(e)[:50]}"
        try:
            vi = pull_voliv(tk, ws, we)
            if vi is not None and len(vi):
                vi_all.append(vi); st["voliv_rows"] = len(vi)
        except Exception as e:
            st["status"] += f"|ERR_voliv:{str(e)[:40]}"
        try:
            lq = pull_liq(tk, ws, we)
            if lq is not None and len(lq):
                lq_all.append(lq); st["liq_rows"] = len(lq)
        except Exception as e:
            st["status"] += f"|ERR_liq:{str(e)[:40]}"
        if st["ohlcv_rows"] == 0 and st["voliv_rows"] == 0 and st["liq_rows"] == 0 and st["status"] == "ok":
            st["status"] = "no_data"
        status.append(st)
        if i % 25 == 0:
            print(f"  {i}/{len(uni)} ({tk})", flush=True)
        if i % CHECKPOINT == 0:
            flush()
            print(f"  [checkpoint flushed at {i}/{len(uni)}]", flush=True)

    flush()
    oh_df = pd.read_csv(os.path.join(out_dir, FILES["ohlcv"][0]))
    vi_df = pd.read_csv(os.path.join(out_dir, FILES["voliv"][0]))
    lq_df = pd.read_csv(os.path.join(out_dir, FILES["liq"][0]))
    drp = pd.read_csv(os.path.join(out_dir, FILES["dropped"][0]))
    sdf = pd.DataFrame(status)
    for c in ("ohlcv_rows", "voliv_rows", "liq_rows", "dropped"):
        if c in sdf.columns:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce").fillna(0).astype(int)

    # ---- report ----
    viol = gate_violations(oh_df)
    print("\n================= BATCH REPORT =================")
    print(f"names processed: {len(uni)}")
    print(f"  OHLCV resolved : {(sdf['ohlcv_rows'] > 0).sum()} names, {len(oh_df):,} rows  "
          f"{oh_df['date'].min() if len(oh_df) else '-'} -> {oh_df['date'].max() if len(oh_df) else '-'}")
    print(f"  vol_iv resolved: {(sdf['voliv_rows'] > 0).sum()} names, {len(vi_df):,} rows")
    print(f"  liq resolved   : {(sdf['liq_rows'] > 0).sum()} names, {len(lq_df):,} rows")
    print(f"  ROTATION GATE on retained OHLCV: {viol} violations  (bad ticks dropped: {len(drp)})")
    print(f"  no_data names  : {(sdf['status'] == 'no_data').sum()}")
    skips = sdf[sdf['status'].str.startswith(('SKIP', 'ERR'))]
    print(f"  skipped/errored: {len(skips)}")
    if len(skips):
        print(skips[["ticker", "name", "status"]].head(20).to_string(index=False))
    print(f"\nwrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
