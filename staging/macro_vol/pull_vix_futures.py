"""Bucket B — VIX futures term structure UX1..UX7 (BDH PX_LAST). Bloomberg lab 2026-06-18.

Generic CBOE VIX futures front-7. FLDS-verified (recent slice: all 7 populated, [16.11, 22.6]).
Deep history from inception (~2006). xbbg 1.3.0 narwhals tidy -> pivot to wide. Staging only.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2006-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
TMAP = {f"UX{i} Index": f"ux{i}" for i in range(1, 8)}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


long = native(blp.bdh(list(TMAP), ["PX_LAST"], START, END))
p = long[long["value"].notna()].pivot_table(index="date", columns="ticker", values="value", aggfunc="first").reset_index()
p["date"] = pd.to_datetime(p["date"])
out = pd.DataFrame({"date": p["date"].dt.strftime("%Y-%m-%d")})
for tkr, col in TMAP.items():
    out[col] = p[tkr] if tkr in p.columns else pd.NA
out = out.sort_values("date")
out.to_csv(os.path.join(OUT, "vix_futures_curve.csv"), index=False)

rep = []
for c in [c for c in out.columns if c != "date"]:
    v = pd.to_numeric(out[c], errors="coerce"); nn = v.notna()
    rep.append(f"{c}={int(nn.sum())}[{v[nn].min():.2f},{v[nn].max():.2f}]")
print(f"vix_futures_curve.csv: {len(out)} rows {out['date'].min()}..{out['date'].max()}")
print("  " + " ".join(rep))
# contango sanity: median ux1<ux7 in calm, and front spikes in stress
both = out.dropna(subset=["ux1", "ux7"])
print(f"  ux1<ux7 (contango) frac = {(pd.to_numeric(both['ux1'])<pd.to_numeric(both['ux7'])).mean():.2%}")
print(out.tail(3).to_string(index=False))
