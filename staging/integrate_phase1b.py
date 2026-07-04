"""Phase 1B integration — fold Phase-1A fragments into the monoliths + BK<->BNY
collapse + dividends epsilon-clamp. Pure git / no Bloomberg. git is the undo.

dtype=str / keep_default_na=False so UNCHANGED rows write back byte-identical.

Design decisions:
- Batch-1 ohlcv fragments carry the corrected '<NAME> UN Equity' source ticker, but
  are RELABELED to the existing '<NAME> UW Equity' on integration. Rationale: on main
  these NYSE names are mislabeled `UW` *consistently across all 7 files* (ohlcv/vol_iv/
  liquidity/fundamentals/credit/earnings/dividends) and the connector normalizes to bare
  `WMT`, so the #355 goal is the missing HISTORY, not the exchange code. Storing the full
  history under the established `UW` label fixes the data with zero cross-file inconsistency
  (test_fundamentals_credit_subset_of_ohlcv stays green). The cosmetic `UW`->`UN` relabel
  across all files is a separable follow-up.
- VRT's 29 pre-2020 (pre-SPAC-merger) NaN-OHLCV rows (26 NaN-price + 3 NaN-volume) are dropped
  from the FRAGMENT before integration, so the monolith keeps exactly the known 4 NaN-price rows
  (BIIB/TPL) and stays volume-non-negative.
"""
import glob

import pandas as pd

B = "data/bloomberg/"
rep = []
BATCH1_RELABEL = {f"{n} UN Equity": f"{n} UW Equity" for n in ("WMT", "KMB", "CPB", "DPZ", "PLTR")}


def integrate_file(fname, frag_paths, reticker=None, relabel=None, drop_nan_ohlc=False,
                   sort_keys=("ticker", "date")):
    mono = pd.read_csv(B + fname, dtype=str, keep_default_na=False)
    cols = mono.columns.tolist()
    before = len(mono)
    lines = []

    if frag_paths:
        frag = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in frag_paths],
                         ignore_index=True)[cols]
        if relabel:
            frag["ticker"] = frag["ticker"].replace(relabel)
        if drop_nan_ohlc:
            before_f = len(frag)
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            keep = ~(frag[ohlcv_cols] == "").any(axis=1)
            frag = frag[keep]
            if before_f - len(frag):
                lines.append(f"    dropped {before_f - len(frag)} NaN-OHLCV fragment rows (VRT pre-2020)")
        names = sorted(frag["ticker"].unique())
        b4 = {n: int((mono["ticker"] == n).sum()) for n in names}
        mono = mono[~mono["ticker"].isin(set(names))]
        mono = pd.concat([mono, frag], ignore_index=True)
        af = {n: int((mono["ticker"] == n).sum()) for n in names}
        for n in names:
            lines.append(f"    {n:18} {b4[n]:>5} -> {af[n]:>5}")

    if reticker:
        src, dst = reticker
        nshift = int((mono["ticker"] == src).sum())
        mono.loc[mono["ticker"] == src, "ticker"] = dst
        lines.append(f"    reticker {src!r} -> {dst!r}: {nshift} rows")

    mono = mono.sort_values(list(sort_keys), kind="mergesort").reset_index(drop=True)[cols]
    mono.to_csv(B + fname, index=False)
    rep.append(f"{fname}: {before} -> {len(mono)} rows, {mono['ticker'].nunique()} tickers")
    rep.extend(lines)


CASY, BC = "staging/casy/", "staging/blue_chips/"

integrate_file("sp500_ohlcv.csv",
               [CASY + "casy_ohlcv.csv"] + sorted(glob.glob(BC + "*_ohlcv.csv")),
               reticker=("BNY UN Equity", "BK UN Equity"), relabel=BATCH1_RELABEL, drop_nan_ohlc=True)

integrate_file("sp500_vol_iv_full.csv",
               [CASY + "casy_vol_iv.csv"] + sorted(glob.glob(BC + "*_vol_iv.csv")),
               reticker=("BNY UN", "BK UN"))

integrate_file("sp500_liquidity.csv", [CASY + "casy_liquidity.csv"], reticker=("BNY UN", "BK UN"))

integrate_file("sp500_fundamentals.csv", [], reticker=("BNY UN Equity", "BK UN Equity"),
               sort_keys=("ticker",))

# credit_risk: BNY->BK collapse (connector-read via get_credit_risk; subset test spine)
integrate_file("sp500_credit_risk.csv", [], reticker=("BNY UN", "BK UN"), sort_keys=("ticker",))

# earnings: append CASY (file not ticker-sorted)
em = pd.read_csv(B + "sp500_earnings.csv", dtype=str, keep_default_na=False)
ef = pd.read_csv(CASY + "casy_earnings.csv", dtype=str, keep_default_na=False)[em.columns.tolist()]
b4e = len(em)
em = pd.concat([em[em["ticker"] != "CASY UW"], ef], ignore_index=True)
em.to_csv(B + "sp500_earnings.csv", index=False)
rep.append(f"sp500_earnings.csv: {b4e} -> {len(em)} rows (CASY +{len(ef)}; appended, partial EPS/time)")

# dividends: epsilon-clamp only (union pre-satisfied on main)
dv = pd.read_csv(B + "sp500_dividends.csv", dtype=str, keep_default_na=False)
mask = pd.to_numeric(dv["dividend_amount"], errors="coerce") < 0
nclamp = int(mask.sum())
dv.loc[mask, "dividend_amount"] = "0.0"
dv.to_csv(B + "sp500_dividends.csv", index=False)
rep.append(f"sp500_dividends.csv: {len(dv)} rows; epsilon-clamped {nclamp} negative dividend_amount -> 0.0 "
           f"(union pre-satisfied)")

print("=== Phase 1B integration report (option C: batch-1 stored under existing UW label) ===")
for line in rep:
    print(line)
