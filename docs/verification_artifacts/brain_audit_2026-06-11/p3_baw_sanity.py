import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

import itertools
import numpy as np
from engine.option_pricer import american_option_price, black_scholes_price
from engine.binomial_tree import binomial_american_price

print("=== C1. BAW American put: EEP >= 0, bounded, vs CRR-800 binomial ===")
S = 100.0
grid = list(itertools.product(
    [70, 85, 95, 100, 105, 120],          # K
    [7/365, 30/365, 90/365, 0.5, 1.0],    # T
    [0.01, 0.037, 0.08],                  # r
    [0.15, 0.30, 0.60],                   # sigma
    [0.0, 0.02],                          # q
))
neg_eep = []; bound_viol = []; rel_errs = []
worst = (0, None)
for K, T, r, sig, q in grid:
    euro = black_scholes_price(S, K, T, r, sig, "put", q)
    amer = american_option_price(S, K, T, r, sig, "put", q)
    eep = amer - euro
    if eep < -1e-8: neg_eep.append((K, T, r, sig, q, eep))
    intrinsic = max(0.0, K - S)
    if amer < intrinsic - 1e-8 or amer > K + 1e-8:
        bound_viol.append((K, T, r, sig, q, amer))
    crr = binomial_american_price(S, K, T, r, sig, "put", q, 800)
    re = abs(amer - crr) / max(crr, 0.01)
    rel_errs.append(re)
    if re > worst[0]: worst = (re, (K, T, r, sig, q, amer, crr))
print(f"grid size: {len(grid)}")
print(f"negative early-exercise premia (< -1e-8): {len(neg_eep)}")
for v in neg_eep[:5]: print("   ", v)
print(f"bound violations (amer < intrinsic or > K): {len(bound_viol)}")
print(f"BAW vs CRR-800: median rel err={np.median(rel_errs):.5f}  p95={np.percentile(rel_errs,95):.5f}  max={worst[0]:.5f}")
print(f"   worst case (K,T,r,sig,q,baw,crr): {worst[1]}")

print("\n=== C2. r<=0 short-circuit (put + call) ===")
for r in (0.0, -0.005):
    e = black_scholes_price(100, 100, 0.25, r, 0.3, "put", 0.0)
    a = american_option_price(100, 100, 0.25, r, 0.3, "put", 0.0)
    print(f"put  r={r}: euro={e:.6f} amer={a:.6f} equal={abs(a-e)<1e-12}")
    ec = black_scholes_price(100, 100, 0.25, r, 0.3, "call", 0.03)
    ac = american_option_price(100, 100, 0.25, r, 0.3, "call", 0.03)
    print(f"call r={r} q=0.03: euro={ec:.6f} amer={ac:.6f} equal={abs(ac-ec)<1e-12}")

print("\n=== C3. EEP magnitude at wheel-typical short-put node (30-delta-ish, 35d) ===")
for (Spot, K, sig, q) in [(247.64, 234.0, 0.3079, 0.004), (381.60, 358.5, 0.3383, 0.007)]:
    T = 35/365; r = 0.037
    e = black_scholes_price(Spot, K, T, r, sig, "put", q)
    a = american_option_price(Spot, K, T, r, sig, "put", q)
    print(f"S={Spot} K={K}: euro={e:.4f} amer={a:.4f} EEP={a-e:.4f} ({(a-e)/e*100:.2f}% of premium)")

print("\n=== C4. Deep-ITM short put (assignment zone) ===")
for K in (110, 130, 150):
    T = 35/365; r = 0.037; sig = 0.30
    e = black_scholes_price(100, K, T, r, sig, "put", 0.0)
    a = american_option_price(100, K, T, r, sig, "put", 0.0)
    crr = binomial_american_price(100, K, T, r, sig, "put", 0.0, 800)
    print(f"K={K}: intrinsic={K-100} euro={e:.4f} baw={a:.4f} crr={crr:.4f} (baw-crr={a-crr:+.4f})")
