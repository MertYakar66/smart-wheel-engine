import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__
import pandas as pd
from engine.wheel_runner import WheelRunner

wr = WheelRunner()
# CTRA/LW/PAYC: last OHLCV bar 2026-03-20 (left index at the 2026-03-23 seam, still listed).
# as_of=None ("today" scan): does the staleness gate fire, or does it rank on an 80-day-old spot?
df = wr.rank_candidates_by_ev(
    tickers=["CTRA", "LW", "BK", "AAPL"],
    as_of=None,
    top_n=10, min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
print("rows returned:", len(df))
if len(df):
    for _, r in df.iterrows():
        print(r.get("ticker"), "spot=", r.get("spot"), "ev_dollars=", r.get("ev_dollars"),
              "premium=", r.get("premium"), "iv=", r.get("iv"))
# also check the drops diagnostics if exposed
try:
    df2, drops = wr.rank_candidates_by_ev(
        tickers=["CTRA"], as_of=None, top_n=5, min_ev_dollars=-1e9, return_drops=True
    )
    print("drops:", drops)
except TypeError as e:
    print("no return_drops kw:", e)
# last real bar for CTRA
print("CTRA last bar:", wr.connector.get_ohlcv("CTRA").index.max())
# and the explicit-as_of case for contrast: as_of=today
df3 = wr.rank_candidates_by_ev(tickers=["CTRA"], as_of="2026-06-11", top_n=5, min_ev_dollars=-1e9)
print("explicit as_of=2026-06-11 rows:", len(df3))
