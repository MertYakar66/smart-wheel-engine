import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

import numpy as np
import pandas as pd
from engine.wheel_runner import WheelRunner, _resolve_pit_atm_iv
from engine.option_pricer import black_scholes_price, black_scholes_delta

AS_OF = "2026-03-20"
TICKERS = ["AAPL", "MSFT", "JPM"]

wr = WheelRunner()
conn = wr.connector
print("provider:", type(conn).__name__)

print("\n=== B1. Raw served IV is PERCENT (> 3.0 connector gate) ===")
for t in TICKERS:
    hist = conn.get_iv_history(t, end_date=AS_OF)
    last = hist.iloc[-1]
    raw_put, raw_call = float(last["hist_put_imp_vol"]), float(last["hist_call_imp_vol"])
    dec = _resolve_pit_atm_iv(conn, t, AS_OF)
    comp = (raw_put + raw_call) / 2.0
    print(f"{t}: raw put_iv={raw_put:.4f} call_iv={raw_call:.4f} (date {hist.index[-1] if hist.index.name else last.get('date','?')})"
          f" -> _resolve_pit_atm_iv={dec:.6f}  ratio raw_comp/dec={comp/dec:.2f}")
    # gate check on full served history
    full = conn.get_iv_history(t)
    mn = float(pd.to_numeric(full["hist_put_imp_vol"], errors="coerce").min())
    mx = float(pd.to_numeric(full["hist_put_imp_vol"], errors="coerce").max())
    print(f"     served hist_put_imp_vol band: min={mn:.4f} max={mx:.4f}  (gate: must be >3.0 and <=10000)")

print("\n=== B2. Ranker premium is priced FROM DECIMAL vol (reconstruct row premium) ===")
df = wr.rank_candidates_by_ev(tickers=TICKERS, top_n=10, min_ev_dollars=-1e9,
                              as_of=AS_OF, include_diagnostic_fields=True)
print("columns:", list(df.columns))
print(df[[c for c in ("ticker","spot","strike","premium","dte","iv","delta_target","ev_dollars","prob_profit") if c in df.columns]].to_string())

# resolve rf + q the same way the ranker does
rf_raw = conn.get_risk_free_rate(AS_OF)
rf = float(rf_raw)
if rf > 1.0: rf /= 100.0
if not (0.0 <= rf <= 0.25): rf = 0.05
print(f"\nrisk_free_rate used: {rf:.6f}")

for _, row in df.iterrows():
    t = row["ticker"]
    fund = conn.get_fundamentals(t) or {}
    qy = float(fund.get("dividend_yield", 0.0) or 0.0)
    qy = qy / 100.0 if np.isfinite(qy) and qy >= 0 else 0.0
    if qy > 0.30: qy = 0.0
    iv = float(row["iv"])
    T = float(row["dte"]) / 365.0
    recon = black_scholes_price(S=float(row["spot"]), K=float(row["strike"]), T=T,
                                r=rf, sigma=iv, option_type="put", q=qy)
    recon_pct = black_scholes_price(S=float(row["spot"]), K=float(row["strike"]), T=T,
                                    r=rf, sigma=iv*100, option_type="put", q=qy)
    d = black_scholes_delta(float(row["spot"]), float(row["strike"]), T, rf, iv, "put", qy)
    print(f"{t}: row premium={row['premium']:.4f} | recon(decimal iv={iv:.4f})={recon:.4f} "
          f"diff={row['premium']-recon:+.2e} | recon(if percent vol)={recon_pct:.2f} | bsm delta@strike={d:+.4f}")

print("\n=== B3. Realism: ~30-delta ~30d CSP premium yield vs market plausibility ===")
for _, row in df.iterrows():
    spot, strike, prem, dte = float(row["spot"]), float(row["strike"]), float(row["premium"]), float(row["dte"])
    y_spot = prem / spot * 100
    y_strike = prem / strike * 100
    y_month = y_strike * (30.0 / dte)
    print(f"{row['ticker']}: spot={spot:.2f} strike={strike:.2f} ({strike/spot*100:.1f}% of spot) dte={dte:.0f} "
          f"premium={prem:.2f} | yield-on-strike={y_strike:.2f}% per {dte:.0f}d (~{y_month:.2f}%/30d) iv={row['iv']:.3f}")
