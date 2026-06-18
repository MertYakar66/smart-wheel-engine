"""Bucket E — treasury yields tail refresh 06-06 -> 06-18 (staged fragment; monolith byte-untouched).
Overlap-validated at 06-05 vs committed treasury_yields.csv. xbbg 1.3.0. All 8 tenors FLDS-verified.
"""
import os
import pandas as pd
from xbbg import blp

OVERLAP = "2026-06-05"
START, END = "2026-06-05", "2026-06-18"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg", "treasury_yields.csv")
TMAP = {"rate_1m": "USGG1M Index", "rate_3m": "USGG3M Index", "rate_6m": "USGG6M Index",
        "rate_2y": "USGG2YR Index", "rate_5y": "USGG5YR Index", "rate_10y": "USGG10YR Index",
        "rate_30y": "USGG30YR Index", "sofr": "SOFRRATE Index"}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


d = native(blp.bdh(list(TMAP.values()), ["PX_LAST"], START, END))
p = d[d["value"].notna()].pivot_table(index="date", columns="ticker", values="value", aggfunc="first").reset_index()
p["date"] = pd.to_datetime(p["date"])
out = pd.DataFrame({"date": p["date"]})
for col, tk in TMAP.items():
    out[col] = p[tk] if tk in p.columns else pd.NA

# overlap-to-cent vs monolith at 06-05
mono = pd.read_csv(MONO); mono["date"] = pd.to_datetime(mono["date"])
m0 = mono[mono["date"] == pd.Timestamp(OVERLAP)]
f0 = out[out["date"] == pd.Timestamp(OVERLAP)]
print("overlap 06-05 check:")
for c in [c for c in TMAP if c in m0.columns and c in f0.columns]:
    a = pd.to_numeric(m0[c], errors="coerce").iloc[0] if len(m0) else float("nan")
    b = pd.to_numeric(f0[c], errors="coerce").iloc[0] if len(f0) else float("nan")
    if pd.notna(a) and pd.notna(b):
        print(f"  {c:8s} mono={a:.4f} frag={b:.4f} d={abs(a-b):.4g}")

frag = out[out["date"] > pd.Timestamp(OVERLAP)].copy()
frag["date"] = frag["date"].dt.strftime("%Y-%m-%d")
frag.to_csv(os.path.join(OUT, "treasury_yields__2026-06-06_2026-06-18.csv"), index=False)
print(f"\ntreasury tail: {len(frag)} rows {frag['date'].min()}..{frag['date'].max()}")
print(frag.to_string(index=False))
