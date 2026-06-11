import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

import numpy as np
from engine.transaction_costs import (
    DEFAULT_COMMISSION_PER_CONTRACT, DEFAULT_ASSIGNMENT_FEE, DEFAULT_SLIPPAGE_PCT,
    calculate_commission, calculate_slippage, calculate_total_entry_cost,
    estimate_round_trip_cost,
)
from engine.ev_engine import EVEngine, ShortOptionTrade

print("=== A. Cost parameters (imported live) ===")
print(f"commission/contract = ${DEFAULT_COMMISSION_PER_CONTRACT}")
print(f"assignment fee      = ${DEFAULT_ASSIGNMENT_FEE}")
print(f"slippage base       = {DEFAULT_SLIPPAGE_PCT*100:.0f}% of spread")

print("\n=== B. Cost as % of premium — synthetic ranker path (spread=10% of mid, OI=1000) ===")
for prem in (0.50, 1.00, 1.50, 2.00, 5.00):
    spread = prem * 0.10  # ranker proxy: bid=0.95*mid, ask=1.05*mid
    slip = calculate_slippage(mid_price=prem, bid_ask_spread=spread,
                              trade_direction="sell", open_interest=1000)
    entry = 0.65 + slip * 100
    rt = 2 * 0.65 + 2 * slip * 100  # exit_slippage = entry_slippage in ev_engine
    gross = prem * 100
    print(f"premium ${prem:5.2f}: entry_slip/sh=${slip:.4f}  "
          f"entry cost=${entry:.2f} ({entry/gross*100:5.2f}% of prem)  "
          f"round-trip(total_transaction_cost)=${rt:.2f} ({rt/gross*100:5.2f}%)")

print("\n=== C. Illiquid path (OI=40, spread=40% of mid) ===")
for prem in (1.00,):
    spread = prem * 0.40
    slip = calculate_slippage(mid_price=prem, bid_ask_spread=spread,
                              trade_direction="sell", open_interest=40)
    entry = 0.65 + slip * 100
    print(f"premium ${prem:.2f}: slip/sh=${slip:.4f} entry=${entry:.2f} "
          f"({entry/(prem*100)*100:.1f}% of prem)  "
          f"[base 0.15 x2.5(OI<50) x1.5(spread>30%) capped at 0.50]")

print("\n=== D. EVEngine.evaluate — verify what EV actually subtracts (D19 check) ===")
# Deterministic degenerate scenario: terminal price fixed far OTM -> put expires worthless.
# pnl should be exactly net_premium_in = gross - entry_comm - entry_slip (no exit cost netted).
trade = ShortOptionTrade(
    option_type="put", underlying="TEST", spot=100.0, strike=90.0,
    premium=1.50, dte=35, iv=0.25, risk_free_rate=0.04, dividend_yield=0.0,
    contracts=1, bid=1.50*0.95, ask=1.50*1.05, open_interest=1000,
    regime_multiplier=1.0,
)
eng = EVEngine()
scen = np.full(5000, 150.0)  # always far OTM, never ITM, no assignment fee path
res = eng.evaluate(trade, price_scenarios=scen)
gross = 1.50 * 100
spread = 1.50*1.05 - 1.50*0.95
slip_sh = calculate_slippage(1.50, spread, "sell", open_interest=1000)
entry_cost = 0.65 + slip_sh*100
expected_net = gross - entry_cost
print(f"gross premium            = ${gross:.2f}")
print(f"entry cost (comm+slip)   = ${entry_cost:.4f}")
print(f"expected net_premium_in  = ${expected_net:.4f}")
print(f"EVResult.ev_dollars      = ${res.ev_dollars:.4f}")
print(f"EVResult.total_transaction_cost = ${res.total_transaction_cost:.4f}")
print(f"-> EV nets ONLY entry leg: {abs(res.ev_dollars - expected_net) < 1e-6}")
print(f"-> exit leg (${res.total_transaction_cost - entry_cost:.2f}) reported but NOT in EV (D19): "
      f"{abs(res.total_transaction_cost - 2*entry_cost) < 1e-6}")

print("\n=== E. Assignment-fee application (always-ITM scenario) ===")
scen_itm = np.full(5000, 50.0)  # always deep ITM at expiry
res_itm = eng.evaluate(trade, price_scenarios=scen_itm)
# pnl = net_premium_in - intrinsic*100 - assignment_fee
intrinsic = (90.0 - 50.0) * 100
expected_itm = expected_net - intrinsic - 5.00
print(f"expected (net_prem - intrinsic - $5 assign fee) = ${expected_itm:.4f}")
print(f"ev_dollars = ${res_itm.ev_dollars:.4f}  match={abs(res_itm.ev_dollars-expected_itm)<1e-6}")

print("\n=== F. estimate_round_trip_cost helper for $1 / $2 premium ===")
for prem in (1.00, 2.00):
    rt = estimate_round_trip_cost(entry_premium=prem, expected_exit_premium=prem*0.5,
                                  open_interest=1000)
    print(f"premium ${prem:.2f}: total=${rt['total_costs']:.2f} "
          f"cost_as_pct_of_premium={rt['cost_as_pct_of_premium']*100:.2f}%")

print("\n=== G. S32 harness friction vs engine internal, $1.50 premium ===")
prem = 1.50
s32_half_spread = max(0.05, 0.08*prem) * 100   # per contract
eng_slip = calculate_slippage(prem, prem*0.10, "sell", open_interest=1000) * 100
print(f"S32 backtest half-spread haircut: ${s32_half_spread:.2f}/contract ({s32_half_spread/(prem*100)*100:.1f}% of premium)")
print(f"Engine internal entry slippage  : ${eng_slip:.2f}/contract ({eng_slip/(prem*100)*100:.1f}% of premium)")
print(f"Ratio S32/engine = {s32_half_spread/eng_slip:.2f}x")
