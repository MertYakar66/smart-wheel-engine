import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__
import pandas as pd, numpy as np
from engine.wheel_runner import WheelRunner

wr = WheelRunner()
print("provider:", type(wr.connector).__name__)

# Historical as_of ranking: which dividend yield does the BSM/delta solve use?
df = wr.rank_candidates_by_ev(
    tickers=["XOM", "T", "AAPL"],
    as_of="2020-06-01",
    top_n=10, min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
cols = [c for c in df.columns if any(k in c.lower() for k in ("div", "yield", "iv", "spot", "strike", "premium", "ev_"))]
print("columns w/ div/iv/ev:", cols)
for _, r in df.iterrows():
    print({k: r.get(k) for k in ["ticker", "spot", "strike", "premium", "iv", "ev_dollars"] if k in df.columns},
          {k: r.get(k) for k in cols if "div" in k.lower() or "yield" in k.lower()})

# XOM mid-2020 reality check: dividend $3.48/yr, spot ~ probe spot
conn = wr.connector
spot_2020 = float(conn.get_ohlcv("XOM", end_date="2020-06-01")["close"].iloc[-1])
true_q_2020 = 3.48 / spot_2020
snap_q = (conn.get_fundamentals("XOM") or {}).get("dividend_yield") / 100.0
print(f"\nXOM 2020-06-01 spot={spot_2020:.2f} TRUE trailing yield ~{true_q_2020:.4f} "
      f"vs snapshot(2026) q used by ranker={snap_q:.4f} -> error {abs(true_q_2020-snap_q)*100:.2f}pp")

# Quantify BSM impact: 35 DTE put at the same strike, sigma from PIT IV
from engine.option_pricer import black_scholes_price
iv_hist = conn.get_iv_history("XOM", end_date="2020-06-01")
sigma = float((iv_hist["hist_put_imp_vol"].iloc[-1] + iv_hist["hist_call_imp_vol"].iloc[-1]) / 2.0) / 100.0
T = 35 / 365
K = round(spot_2020 * 0.93)  # ~ -7% OTM short put
r = float(conn.get_risk_free_rate("2020-06-01"))
p_snap = black_scholes_price(spot_2020, K, T, r, sigma, option_type="put", q=snap_q)
p_true = black_scholes_price(spot_2020, K, T, r, sigma, option_type="put", q=true_q_2020)
print(f"BSM put K={K} T=35d sigma={sigma:.3f} r={r:.4f}: premium q_snap={p_snap:.4f} "
      f"q_true={p_true:.4f} diff={100*(p_true-p_snap):.2f} $/contract ({(p_true/p_snap-1)*100:.1f}%)")
