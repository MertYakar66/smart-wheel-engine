import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

import numpy as np
from engine.option_pricer import (
    black_scholes_price, black_scholes_delta, black_scholes_gamma,
    black_scholes_theta, black_scholes_vega, black_scholes_rho,
    black_scholes_all_greeks, implied_volatility,
)

print("=== A1. Put-call parity: C - P == S e^{-qT} - K e^{-rT} ===")
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(500):
    S = float(rng.uniform(5, 500)); K = float(rng.uniform(0.5, 1.5)) * S
    T = float(rng.uniform(0.01, 2.0)); r = float(rng.uniform(-0.01, 0.08))
    q = float(rng.uniform(0.0, 0.05)); sig = float(rng.uniform(0.05, 1.5))
    C = black_scholes_price(S, K, T, r, sig, "call", q)
    P = black_scholes_price(S, K, T, r, sig, "put", q)
    lhs = C - P; rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    worst = max(worst, abs(lhs - rhs))
print(f"500 random (S,K,T,r,q,sigma): max |parity violation| = {worst:.3e}")

print("\n=== A2. Hand-checked closed form (Hull): S=42 K=40 T=0.5 r=0.10 sigma=0.20 q=0 ===")
C = black_scholes_price(42, 40, 0.5, 0.10, 0.20, "call")
P = black_scholes_price(42, 40, 0.5, 0.10, 0.20, "put")
print(f"call = {C:.6f}  (Hull: 4.759422)   diff={C-4.759422:+.2e}")
print(f"put  = {P:.6f}  (Hull: 0.808599)   diff={P-0.808599:+.2e}")

print("\n=== A2b. Merton dividend case (independent manual recompute) ===")
from scipy.stats import norm
S,K,T,r,q,sig = 100.0, 95.0, 30/365, 0.0433, 0.012, 0.28
d1 = (np.log(S/K) + (r - q + 0.5*sig**2)*T) / (sig*np.sqrt(T))
d2 = d1 - sig*np.sqrt(T)
manual_put = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
eng_put = black_scholes_price(S,K,T,r,sig,"put",q)
print(f"manual={manual_put:.8f} engine={eng_put:.8f} diff={eng_put-manual_put:.2e}")

print("\n=== A3. Greek signs + finite-difference unit checks (GREEKS_UNIT_CONTRACT) ===")
S,K,T,r,q,sig = 100.0, 95.0, 30/365, 0.0433, 0.012, 0.28
for ot in ("call","put"):
    g = black_scholes_all_greeks(S,K,T,r,sig,ot,q)
    h = 1e-5
    fd_delta = (black_scholes_price(S+h,K,T,r,sig,ot,q)-black_scholes_price(S-h,K,T,r,sig,ot,q))/(2*h)
    fd_gamma = (black_scholes_price(S+1,K,T,r,sig,ot,q)-2*g['price']+black_scholes_price(S-1,K,T,r,sig,ot,q))/1.0
    # vega contract: vega ~= P(sigma+0.01) - P(sigma)  (per 1 vol point)
    fd_vega_1pt = black_scholes_price(S,K,T,r,sig+0.01,ot,q)-g['price']
    # rho contract: rho ~= P(r+0.01) - P(r)  (per 100bps)
    fd_rho_1pct = black_scholes_price(S,K,T,r+0.01,sig,ot,q)-g['price']
    # theta contract: annual; daily = theta/365 ~= P(T - 1d) - P(T)
    fd_theta_1d = black_scholes_price(S,K,T-1/365,r,sig,ot,q)-g['price']
    print(f"[{ot}] delta={g['delta']:+.6f} (fd {fd_delta:+.6f})  gamma={g['gamma']:.6f} (fd~{fd_gamma:.6f})")
    print(f"      vega={g['vega']:.6f} vs 1-volpt bump {fd_vega_1pt:.6f}  | rho={g['rho']:+.6f} vs 100bp bump {fd_rho_1pct:+.6f}")
    print(f"      theta/365={g['theta']/365:+.6f} vs 1d decay {fd_theta_1d:+.6f}")
    # sign assertions
    if ot == "call":
        assert 0 <= g['delta'] <= 1 and g['rho'] > 0
    else:
        assert -1 <= g['delta'] <= 0 and g['rho'] < 0
    assert g['gamma'] > 0 and g['vega'] > 0 and g['theta'] < 0
print("sign assertions passed: call delta [0,1] rho>0; put delta [-1,0] rho<0; gamma,vega>0; theta<0")

print("\n=== A4. Second/third-order Greeks vs finite differences ===")
g = black_scholes_all_greeks(S,K,T,r,sig,"put",q)
h = 1e-4
# vanna = dDelta/dsigma
fd_vanna = (black_scholes_delta(S,K,T,r,sig+h,"put",q)-black_scholes_delta(S,K,T,r,sig-h,"put",q))/(2*h)
# volga = d2P/dsigma2 (per unit sigma) -- engine volga = vega(scaled)*d1*d2/sigma; vega scaled /100 => volga is /100 of raw
fd_volga_raw = (black_scholes_price(S,K,T,r,sig+h,"put",q)-2*g['price']+black_scholes_price(S,K,T,r,sig-h,"put",q))/h**2
# charm = dDelta/dt (calendar) = -dDelta/dT
fd_charm = -(black_scholes_delta(S,K,T+h,r,sig,"put",q)-black_scholes_delta(S,K,T-h,r,sig,"put",q))/(2*h)
# speed = dGamma/dS
fd_speed = (black_scholes_gamma(S+h,K,T,r,sig,q)-black_scholes_gamma(S-h,K,T,r,sig,q))/(2*h)
print(f"vanna={g['vanna']:+.6f} fd={fd_vanna:+.6f}")
print(f"volga(engine)={g['volga']:+.6f}  raw fd d2P/dsig2={fd_volga_raw:+.6f}  fd/100={fd_volga_raw/100:+.6f}")
print(f"charm={g['charm']:+.6f} fd={fd_charm:+.6f}")
print(f"speed={g['speed']:+.6e} fd={fd_speed:+.6e}")

print("\n=== A5. IV solver round-trip ===")
for sig_true in (0.12, 0.28, 0.65, 1.4):
    px = black_scholes_price(S,K,T,r,sig_true,"put",q)
    iv = implied_volatility(px, S,K,T,r,"put",q)
    print(f"sigma_true={sig_true:.3f} -> price={px:.4f} -> iv_recovered={iv:.6f}")
