"""T0-5 — Macro-event calendar (shared: wheel event-lockout + day-bot intraday). Lab 2026-06-17.

Two complementary outputs:
  sp500_macro_calendar.csv  — the release SCHEDULE with intraday timestamps:
      bds ECO_FUTURE_RELEASE_DATE_LIST -> "Release Dates & Times" (recent + forward ~2y).
      Columns: event, ticker, name, country, release_datetime, release_date, release_time.
  sp500_macro_releases.csv  — the actual-value HISTORY (deeper): bdh PX_LAST 2015->now.
      Columns: event, ticker, date, actual.

FLDS-verified: ECO_FUTURE_RELEASE_DATE_LIST (bds) returns datetimes w/ times; ECO_RELEASE_TIME
(bdp) gives the scheduled time; PX_LAST (bdh) the released figure. xbbg 1.3.0 narwhals -> to_native.
Tickers that don't resolve are skipped (graceful). Staging only; held.
"""
import os
import pandas as pd
from xbbg import blp

OUT = os.path.dirname(__file__)
START = "2015-01-01"
# (bloomberg ticker, short event label)
EVENTS = [
    ("FDTR Index", "fed_funds_target"), ("CPI YOY Index", "cpi_yoy"), ("CPI XYOY Index", "core_cpi_yoy"),
    ("PCE CYOY Index", "pce_yoy"), ("NFP TCH Index", "nonfarm_payrolls"), ("USURTOT Index", "unemployment_rate"),
    ("INJCJC Index", "initial_jobless_claims"), ("GDP CQOQ Index", "gdp_qoq"),
    ("NAPMPMI Index", "ism_manufacturing"), ("NAPMNMI Index", "ism_services"), ("RSTAMOM Index", "retail_sales_mom"),
]


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


sched, vals = [], []
for tkr, label in EVENTS:
    try:
        meta = native(blp.bdp([tkr], ["NAME", "COUNTRY"]))
        m = {r["field"]: r["value"] for _, r in meta.iterrows()} if len(meta) else {}
        name, country = m.get("NAME", ""), m.get("COUNTRY", "")
    except Exception:
        name, country = "", ""
    # schedule (date + time)
    try:
        d = native(blp.bds(tkr, "ECO_FUTURE_RELEASE_DATE_LIST"))
        col = next((c for c in d.columns if "Release" in c or "Date" in c), None)
        n_sched = 0
        if col:
            for dt in pd.to_datetime(d[col], errors="coerce").dropna():
                sched.append({"event": label, "ticker": tkr, "name": name, "country": country,
                              "release_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                              "release_date": dt.strftime("%Y-%m-%d"), "release_time": dt.strftime("%H:%M:%S")})
                n_sched += 1
    except Exception as e:
        n_sched = f"ERR {type(e).__name__}"
    # actual-value history
    try:
        h = native(blp.bdh(tkr, ["PX_LAST"], START, "2026-06-17"))
        n_val = 0
        if {"date", "value"}.issubset(h.columns):
            for _, r in h.dropna(subset=["value"]).iterrows():
                vals.append({"event": label, "ticker": tkr, "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                             "actual": round(float(r["value"]), 4)})
                n_val += 1
    except Exception as e:
        n_val = f"ERR {type(e).__name__}"
    print(f"  {label:22} {tkr:16} schedule={n_sched} values={n_val}  ({name})")

sc = pd.DataFrame(sched).sort_values(["release_datetime", "event"])
sc.to_csv(os.path.join(OUT, "sp500_macro_calendar.csv"), index=False)
vh = pd.DataFrame(vals).sort_values(["event", "date"])
vh.to_csv(os.path.join(OUT, "sp500_macro_releases.csv"), index=False)
print(f"\nsp500_macro_calendar.csv : {len(sc)} scheduled releases, {sc['event'].nunique()} events, "
      f"{sc['release_date'].min()}..{sc['release_date'].max()}")
print(f"sp500_macro_releases.csv : {len(vh)} actual prints, {vh['date'].min()}..{vh['date'].max()}")
print("distinct release times:", sorted(sc['release_time'].unique()))
