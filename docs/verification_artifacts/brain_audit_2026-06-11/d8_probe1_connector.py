import os, sys, warnings, inspect
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__
import pandas as pd, numpy as np
from engine.data_connector import MarketDataConnector

conn = MarketDataConnector()
print("connector module:", MarketDataConnector.__module__, "->", engine.__file__)
print("deep_history default:", conn._deep_history)

# ---- (a) IV sentinel gate: raw vs served ----
raw = pd.read_csv('data/bloomberg/sp500_vol_iv_full.csv', low_memory=False)
served = conn._load('vol_iv')
for col in ('hist_put_imp_vol', 'hist_call_imp_vol'):
    rn = pd.to_numeric(raw[col], errors='coerce')
    sn = pd.to_numeric(served[col], errors='coerce')
    print(f"{col}: raw_n={int(rn.notna().sum())} raw_sub3={int(((rn>0)&(rn<=3.0)).sum())} "
          f"raw_gt10000={int((rn>10000).sum())} raw_min={rn.min():.4f} raw_max={rn.max():.2f} | "
          f"served_n={int(sn.notna().sum())} served_min={sn.min():.4f} served_max={sn.max():.2f} "
          f"nulled={int(rn.notna().sum()-sn.notna().sum())}")

# what reaches the engine: PIT IV read used by the ranker
iv = conn.get_iv_history('AAPL', end_date='2026-06-04')
print("AAPL get_iv_history last date:", iv.index.max(), "put_iv:", float(iv['hist_put_imp_vol'].iloc[-1]))

# ---- (c) dateless fundamentals / credit ----
print("get_fundamentals sig:", inspect.signature(conn.get_fundamentals))
print("get_credit_risk sig:", inspect.signature(conn.get_credit_risk))
for t in ('XOM', 'AAPL', 'T'):
    f = conn.get_fundamentals(t) or {}
    print(f"{t}: snapshot dividend_yield(pct)={f.get('dividend_yield')} sector={f.get('sector')}")

# ---- (e) treasury / D20 ----
for d in ('2012-06-01', '2020-06-01', '2021-06-01', '2026-06-04', None):
    print("rf rate_3m as_of", d, "->", conn.get_risk_free_rate(d))
from engine.data_integration import get_current_risk_free_rate
print("data_integration rf pre-coverage 1990-01-01:", get_current_risk_free_rate('1990-01-01'))
print("connector rf pre-coverage 1990-01-01:", conn.get_risk_free_rate('1990-01-01'))
tr = pd.read_csv('data/bloomberg/treasury_yields.csv')
tr['date'] = pd.to_datetime(tr['date'])
print("treasury range:", tr['date'].min().date(), "->", tr['date'].max().date(),
      "rate_3m nulls:", int(pd.to_numeric(tr['rate_3m'], errors='coerce').isna().sum()))

# ---- (b)/(d) frontier + universe ----
oh = pd.read_csv('data/bloomberg/sp500_ohlcv.csv', usecols=['date'])
mx = pd.to_datetime(oh['date']).max()
print("OHLCV rows:", len(oh), "frontier:", mx.date(), "days stale vs 2026-06-11:", (pd.Timestamp('2026-06-11')-mx).days)
uni = conn.get_universe()
print("get_universe size:", len(uni))
# as_of=None spot resolution: what does the ranker see as 'today' spot?
ohlcv = conn.get_ohlcv('AAPL')
print("AAPL last bar (as_of=None):", ohlcv.index.max().date(), "close:", float(ohlcv['close'].iloc[-1]))
