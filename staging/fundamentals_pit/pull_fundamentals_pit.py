"""#354 — point-in-time (dated) fundamentals panel for the EV-consumed fields.

Closes the lookahead where the engine reads the single DATELESS sp500_fundamentals.csv
for historical backtests. Pulls a MONTHLY dated panel (2015-01-01..2026-06-04) of the
EV-path fundamental fields for the 503 fundamentals-universe names.

Staging only (staging/fundamentals_pit/) — NOT integrated into the monoliths and NOT
wired into the connector (that #354 accessor + as_of threading is a trio change for a
later §2-reviewed session). This persists the Bloomberg-gated DATA now.

Validation: the latest monthly row per name should ~match the current dateless snapshot
(sp500_fundamentals.csv) for the same field. Writes incrementally (append per chunk) so a
mid-session interruption leaves partial data on disk.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2015-01-01", "2026-06-04"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
OUTFILE = os.path.join(OUT, "sp500_fundamentals_pit.csv")
CHUNK = 30

# snapshot col (lowercase) -> Bloomberg field (uppercase). EV-consumed fields only;
# IV/realized-vol excluded (covered by the vol_iv PIT panel); gics_* excluded (categorical/static).
FIELD_MAP = {
    "eqy_dvd_yld_12m": "EQY_DVD_YLD_12M",
    "beta_raw_overridable": "BETA_RAW_OVERRIDABLE",
    "pe_ratio": "PE_RATIO",
    "best_pe_ratio": "BEST_PE_RATIO",
    "cur_mkt_cap": "CUR_MKT_CAP",
    "return_com_eqy": "RETURN_COM_EQY",
    "tot_debt_to_tot_eqy": "TOT_DEBT_TO_TOT_EQY",
    "free_cash_flow_yield": "FREE_CASH_FLOW_YIELD",
}
COLS_OUT = ["date", "ticker"] + list(FIELD_MAP.keys())
BBG_FIELDS = list(FIELD_MAP.values())
INV_MAP = {v: k for k, v in FIELD_MAP.items()}


def main():
    snap = pd.read_csv(os.path.join(MONO, "sp500_fundamentals.csv"))
    tickers = snap["ticker"].dropna().unique().tolist()
    print(f"universe: {len(tickers)} tickers; fields: {BBG_FIELDS}")
    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)
    wrote_header = False
    total = 0
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            nw = blp.bdh(chunk, BBG_FIELDS, START, END, Per="M")
            df = nw.to_native() if hasattr(nw, "to_native") else nw
            if not {"ticker", "date", "field", "value"}.issubset(set(df.columns)):
                print(f"  chunk {i}: unexpected shape {list(df.columns)}; skipped")
                continue
            w = df.pivot_table(index=["date", "ticker"], columns="field", values="value", aggfunc="first").reset_index()
            w = w.rename(columns=INV_MAP)
            for c in FIELD_MAP:           # ensure all field cols present even if a chunk lacked one
                if c not in w.columns:
                    w[c] = pd.NA
            w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
            w = w[COLS_OUT].sort_values(["ticker", "date"])
            w.to_csv(OUTFILE, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            total += len(w)
            print(f"  chunk {i}-{min(i+CHUNK,len(tickers))}: +{len(w)} rows (total {total})")
        except Exception as e:
            print(f"  chunk {i} ERROR: {type(e).__name__}: {e}")

    # validation vs current snapshot (latest monthly row per name)
    out = pd.read_csv(OUTFILE)
    out["date"] = pd.to_datetime(out["date"])
    latest = out.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    snap2 = snap.set_index("ticker")
    print(f"\n=== {OUTFILE}: {len(out)} rows, {out['ticker'].nunique()} names, "
          f"{out['date'].min().date()}..{out['date'].max().date()} ===")
    print("validation: latest PIT row vs dateless snapshot (eqy_dvd_yld_12m sample)")
    for tk in ["AAPL UW Equity", "XOM UN Equity", "CAG UN Equity"]:
        if tk in latest.index and tk in snap2.index:
            print(f"  {tk}: pit={latest.loc[tk,'eqy_dvd_yld_12m']:.4f}  snapshot={snap2.loc[tk,'eqy_dvd_yld_12m']:.4f}")
    # coverage of the carry-critical field
    cov = out["eqy_dvd_yld_12m"].notna().mean()
    print(f"eqy_dvd_yld_12m non-null coverage: {cov:.1%}")


if __name__ == "__main__":
    main()
