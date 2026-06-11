import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

from engine.option_pricer import black_scholes_price, black_scholes_delta

print("=== D1. Zero-skew bias: 25-delta put priced at ATM vol vs typical smile vol ===")
# AAPL node from the real as-of-2026-03-20 ranking: S=247.64 K=234.0 T=35d iv_atm=0.3079
S, K, T, r, q = 247.64, 234.0, 35/365, 0.037, 0.004
atm = 0.3079
for skew_pts in (2, 4, 6):
    smile = atm + skew_pts/100
    p_atm = black_scholes_price(S, K, T, r, atm, "put", q)
    p_smile = black_scholes_price(S, K, T, r, smile, "put", q)
    d_atm = black_scholes_delta(S, K, T, r, atm, "put", q)
    d_smile = black_scholes_delta(S, K, T, r, smile, "put", q)
    print(f"smile +{skew_pts} vol pts: premium {p_atm:.3f} -> {p_smile:.3f} "
          f"({(p_smile/p_atm-1)*100:+.1f}%); true delta at engine strike {d_atm:+.4f} -> {d_smile:+.4f}")

print("\n=== D2. Covered-call side: OTM call at ATM vol vs slightly-lower call-wing vol ===")
Kc = 262.0  # ~25-delta call strike region
for wing in (0, -1, -2):
    v = atm + wing/100
    pc = black_scholes_price(S, Kc, T, r, v, "call", q)
    print(f"call wing {wing:+d} pts (iv={v:.4f}): premium={pc:.3f}")
