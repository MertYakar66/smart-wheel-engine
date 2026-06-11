"""Probe 1: replicate EVEngine.evaluate by hand for a synthetic short put.

Fixed forward log-returns, known premium/strike. Compare engine vs hand math
to the cent for: ev_dollars, prob_profit, Wilson CI, cvar_5, omega, prob_touch,
prob_assignment, mean/std/skew, breakeven, net_premium_in, total cost.
"""

import os
import sys
import warnings

WT = r"C:\Users\merty\Desktop\swe-main"
os.chdir(WT)
sys.path.insert(0, WT)
warnings.filterwarnings("ignore")
import engine  # noqa: E402

assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

import numpy as np  # noqa: E402

from engine.ev_engine import EVEngine, ShortOptionTrade, _wilson_score_interval  # noqa: E402
from engine.option_pricer import black_scholes_price  # noqa: E402

# --- synthetic candidate -------------------------------------------------
trade = ShortOptionTrade(
    option_type="put",
    underlying="SYNTH",
    spot=100.0,
    strike=95.0,
    premium=2.50,
    bid=2.40,
    ask=2.60,
    dte=30,
    iv=0.25,
    risk_free_rate=0.04,
    dividend_yield=0.0,
    contracts=1,
    open_interest=1000,
)

# 12 fixed forward log-returns incl. a deep-tail crash (-30%) and a -10% loss
fwd = np.array(
    [-0.30, -0.10, -0.06, -0.03, -0.01, 0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]
)

eng = EVEngine()
res = eng.evaluate(trade, forward_log_returns=fwd)

# --- hand replication ----------------------------------------------------
mult = 100  # 1 contract
gross_premium = 2.50 * mult  # 250.00
entry_commission = 0.65
spread = 2.60 - 2.40  # 0.20 (both sides present, not crossed)
# calculate_slippage hand: base=0.15, OI=1000 -> no liq adj; spread_pct=0.08 -> no
# wide-spread adj; cap 0.50 not binding; no adv -> no sqrt impact
entry_slip_per_share = 0.20 * 0.15  # 0.03
entry_slippage = entry_slip_per_share * mult  # 3.00
exit_commission = 0.65
exit_slippage = entry_slippage
total_cost_hand = entry_commission + entry_slippage + exit_commission + exit_slippage
net_premium_in = gross_premium - entry_commission - entry_slippage  # 246.35

terminal = 100.0 * np.exp(fwd)
intrinsic = np.maximum(95.0 - terminal, 0.0)
pnl = net_premium_in - intrinsic * mult
itm = terminal < 95.0
pnl = pnl - itm.astype(float) * 5.0  # assignment fee $5 x 1 contract

ev_raw_hand = float(np.mean(pnl))
prob_profit_hand = float(np.mean(pnl > 0))
prob_itm_hand = float(np.mean(itm))
n = len(pnl)
k = int(np.count_nonzero(pnl > 0))
ci_hand = _wilson_score_interval(k, n)
# independent Wilson check (formula re-derived, no engine code)
import math  # noqa: E402

z = 1.96
p = k / n
den = 1 + z * z / n
ctr = (p + z * z / (2 * n)) / den
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
ci_indep = (max(0.0, ctr - half), min(1.0, ctr + half))

cvar5_hand = float(np.min(pnl))  # n=12 < 20 -> min path
gains = pnl[pnl > 0].sum()
losses = -pnl[pnl < 0].sum()
omega_hand = min(float(gains / losses), 1000.0) if losses > 1e-9 else (1000.0 if gains > 0 else 0.0)
prob_touch_hand = min(1.0, 2.0 * prob_itm_hand)  # OTM at entry
std_hand = float(np.std(pnl, ddof=1))
skew_hand = float(np.mean((pnl - ev_raw_hand) ** 3) / std_hand**3)
p25h, p50h, p75h = (float(x) for x in np.percentile(pnl, [25, 50, 75]))
breakeven_price = 95.0 - (2.50 - total_cost_hand / mult)
breakeven_hand = (breakeven_price - 100.0) / 100.0
fair_hand = black_scholes_price(S=100.0, K=95.0, T=30 / 365.0, r=0.04, sigma=0.25, option_type="put", q=0.0)
edge_hand = (2.50 - fair_hand) * mult
# expected_days_held hand
stop_thr = -2.0 * net_premium_in
prob_stop = float(np.mean(pnl <= stop_thr))
prob_hold = max(0.0, 1.0 - prob_profit_hand - prob_stop)
edh_hand = max(1.0, prob_profit_hand * 15.0 + prob_stop * 10.0 + prob_hold * 30.0)

rows = [
    ("ev_dollars (== ev_raw, regime=1)", res.ev_dollars, ev_raw_hand),
    ("mean_pnl (ev_raw)", res.mean_pnl, ev_raw_hand),
    ("prob_profit", res.prob_profit, prob_profit_hand),
    ("prob_assignment", res.prob_assignment, prob_itm_hand),
    ("prob_touch", res.prob_touch, prob_touch_hand),
    ("cvar_5 (n<20 -> min)", res.cvar_5, cvar5_hand),
    ("omega", res.omega_ratio, omega_hand),
    ("std_pnl", res.std_pnl, std_hand),
    ("skew_pnl", res.skew_pnl, skew_hand),
    ("pnl_p25", res.pnl_p25, p25h),
    ("pnl_p50", res.pnl_p50, p50h),
    ("pnl_p75", res.pnl_p75, p75h),
    ("breakeven_move_pct", res.breakeven_move_pct, breakeven_hand),
    ("fair_value", res.fair_value, fair_hand),
    ("edge_vs_fair", res.edge_vs_fair, edge_hand),
    ("total_transaction_cost", res.total_transaction_cost, total_cost_hand),
    ("net_premium_in (metadata)", res.metadata["net_premium_in"], net_premium_in),
    ("expected_days_held", res.expected_days_held, edh_hand),
    ("ev_per_day", res.ev_per_day, ev_raw_hand / edh_hand),
    ("ci_low", res.prob_profit_ci_low, ci_indep[0]),
    ("ci_high", res.prob_profit_ci_high, ci_indep[1]),
]
print(f"n_scenarios={res.n_scenarios} (expect 12), dist_src={res.metadata['distribution_source']}")
ok = True
for name, got, want in rows:
    match = abs(got - want) < 5e-9
    ok &= match
    print(f"{'OK ' if match else 'FAIL'} {name:35s} engine={got:.6f} hand={want:.6f}")
print(f"wilson engine vs internal helper: {ci_hand} vs {ci_indep}")

# Show the EV is NOT premium-if-all-goes-well:
print(f"\npremium-if-all-goes-well = net_premium_in = {net_premium_in:.2f}")
print(f"ev_dollars (tail-weighted) = {res.ev_dollars:.2f}  <-- includes -30% crash path")
crash_pnl = net_premium_in - max(95 - 100 * math.exp(-0.30), 0) * 100 - 5.0
print(f"crash-path pnl (hand) = {crash_pnl:.2f}; min(pnl) engine cvar_5 = {res.cvar_5:.2f}")

# contracts=2 unit scaling check
trade2 = ShortOptionTrade(
    option_type="put", underlying="SYNTH", spot=100.0, strike=95.0, premium=2.50,
    bid=2.40, ask=2.60, dte=30, iv=0.25, risk_free_rate=0.04, contracts=2,
    open_interest=1000,
)
res2 = EVEngine().evaluate(trade2, forward_log_returns=fwd)
# hand for 2 contracts: mult=200, commission=1.30, slip=6.0, fee=10.0
np2 = 500.0 - 1.30 - 6.00
pnl2 = np2 - intrinsic * 200 - itm.astype(float) * 10.0
print(f"\ncontracts=2: engine ev={res2.ev_dollars:.4f} hand={np.mean(pnl2):.4f} "
      f"match={abs(res2.ev_dollars - np.mean(pnl2)) < 5e-9}")
print("ALL_MATCH" if ok else "MISMATCH_FOUND")
