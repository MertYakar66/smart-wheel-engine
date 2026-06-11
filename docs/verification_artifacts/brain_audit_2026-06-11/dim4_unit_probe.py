"""Dimension 4 probe B: synthetic unit probes for GPD / HMM / dealer / rv-widening guards."""

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
import pandas as pd  # noqa: E402

from engine.dealer_positioning import MarketStructure, DealerAssumption, dealer_regime_multiplier  # noqa: E402
from engine.ev_engine import EVEngine, ShortOptionTrade  # noqa: E402
from engine.forward_distribution import realized_vol_ratio, realized_vol_widening_factor  # noqa: E402
from engine.regime_hmm import GaussianHMM  # noqa: E402
from engine.tail_risk import fit_gpd_tail, gpd_var_cvar, pot_gpd_cvar  # noqa: E402

rng = np.random.default_rng(7)

# --- 1. GPD on a genuinely heavy-tailed sample (t df=2.5) ---
t_losses = np.abs(rng.standard_t(2.5, size=5000)) * 0.02
fit = fit_gpd_tail(t_losses)
var99, cvar99 = gpd_var_cvar(fit, confidence=0.99)
print("[GPD-t2.5] converged=%s xi=%.3f beta=%.5f n_exc=%d var99=%.4f cvar99=%.4f cvar>=var=%s"
      % (fit.converged, fit.shape_xi, fit.scale_beta, fit.n_exceedances, var99, cvar99, cvar99 >= var99))
# monotonic in confidence
prev = -1
mono = True
for c in (0.95, 0.97, 0.99, 0.995, 0.999):
    v, cv = gpd_var_cvar(fit, confidence=c)
    if v < prev:
        mono = False
    prev = v
print("[GPD] VaR monotone in confidence:", mono)
# thin-tail control
g_losses = np.abs(rng.normal(0, 0.02, size=5000))
d = pot_gpd_cvar(g_losses)
print("[GPD-normal] xi=%.3f heavy_tail=%s quality=%s" % (d["xi"], d["heavy_tail"], d["fit_quality"]))
# empty / tiny inputs
print("[GPD-empty]", fit_gpd_tail(np.array([])).converged, gpd_var_cvar(fit_gpd_tail(np.array([]))))
print("[GPD-NaNs]", fit_gpd_tail(np.array([np.nan] * 50)).converged)

# --- 2. EVEngine heavy-tail penalty integration (>=200 scenarios) ---
trade = ShortOptionTrade(
    option_type="put", underlying="TEST", spot=100.0, strike=95.0, premium=2.0,
    dte=30, iv=0.30, risk_free_rate=0.04, dividend_yield=0.0, contracts=1,
    bid=1.95, ask=2.05, open_interest=1000, regime_multiplier=1.0,
)
eng = EVEngine()
# Fat-tailed terminal prices: lognormal body + crash cluster
prices = 100.0 * np.exp(rng.standard_t(2.05, size=2000) * 0.06)
res = eng.evaluate(trade, price_scenarios=prices)
print("[EV-heavytail] heavy_tail=%s tail_xi=%.3f cvar_99_evt=%.2f regime_mult=%.3f mean_pnl=%.2f ev=%.2f ratio=%.3f"
      % (res.heavy_tail, res.tail_xi, res.cvar_99_evt, res.regime_multiplier,
         res.mean_pnl, res.ev_dollars, res.ev_dollars / res.mean_pnl if res.mean_pnl else float("nan")))
# small-sample path: EVT must NOT fire
res_small = eng.evaluate(trade, price_scenarios=prices[:50])
print("[EV-small-n] heavy_tail=%s tail_xi=%s (expect False/nan at n<200)" % (res_small.heavy_tail, res_small.tail_xi))
# sign preservation under max boost regime
trade_neg = ShortOptionTrade(
    option_type="put", underlying="TEST", spot=100.0, strike=120.0, premium=0.10,
    dte=30, iv=0.30, risk_free_rate=0.04, dividend_yield=0.0, contracts=1,
    bid=0.05, ask=0.15, open_interest=1000, regime_multiplier=1.25,
)
ms_boost = MarketStructure(ticker="X", as_of=pd.Timestamp("2026-03-20"), spot=100.0,
                           expiry=pd.Timestamp("2026-04-20").date(),
                           assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
                           regime="long_gamma_dampening", confidence=1.0)
res_neg = eng.evaluate(trade_neg, price_scenarios=prices, market_structure=ms_boost)
print("[EV-sign] ev_raw(mean_pnl)=%.2f ev_dollars=%.2f dealer_mult=%.3f sign preserved=%s"
      % (res_neg.mean_pnl, res_neg.ev_dollars, res_neg.dealer_multiplier,
         (res_neg.mean_pnl < 0) == (res_neg.ev_dollars < 0)))

# --- 3. HMM guards (#386) + envelope ---
try:
    GaussianHMM(n_states=4, random_state=42).fit(np.array([0.01, np.nan] + [0.001] * 300))
    print("[HMM-nonfinite] FAILED TO RAISE")
except ValueError as e:
    print("[HMM-nonfinite] ValueError:", str(e)[:60])
try:
    GaussianHMM(n_states=4, random_state=42).fit(np.zeros(300))
    print("[HMM-constant] FAILED TO RAISE")
except ValueError as e:
    print("[HMM-constant] ValueError:", str(e)[:60])
# envelope: multiplier is convex combo of {0.2,0.5,1.0,1.25}
h = GaussianHMM(n_states=4, n_iter=20, random_state=42)
rets = rng.normal(0.0004, 0.012, 504)
h.fit(rets)
probs = h.predict_proba(rets)
mults = [h.position_multiplier(p) for p in probs]
print("[HMM-envelope] min=%.4f max=%.4f within [0.2,1.25]=%s"
      % (min(mults), max(mults), all(0.2 - 1e-9 <= m <= 1.25 + 1e-9 for m in mults)))
# extreme corner posteriors
for corner in (np.array([1.0, 0, 0, 0]), np.array([0, 0, 0, 1.0])):
    print("  corner", corner.tolist(), "->", h.position_multiplier(corner))

# --- 4. rv-ratio non-finite guard (#401) ---
dates = pd.date_range("2024-01-01", periods=400, freq="B")
closes = 100 + np.cumsum(rng.normal(0, 1, 400))
closes = np.abs(closes) + 50
ohlcv = pd.DataFrame({"close": closes}, index=dates)
print("[RV-clean] ratio=%.3f factor=%.3f" % (realized_vol_ratio(ohlcv), realized_vol_widening_factor(ohlcv)))
bad = ohlcv.copy()
bad.iloc[-10, 0] = -5.0  # poison inside the 30d window
print("[RV-poisoned] ratio=%.3f factor=%.3f (expect 1.0/1.0 no-fire)"
      % (realized_vol_ratio(bad), realized_vol_widening_factor(bad)))
# spike scenario: factor caps at 1.15
spike = ohlcv.copy()
spike.iloc[-30:, 0] = spike.iloc[-30:, 0].values * np.exp(np.cumsum(rng.normal(0, 0.08, 30)))
f = realized_vol_widening_factor(spike)
print("[RV-spike] factor=%.4f <=1.15: %s" % (f, f <= 1.15 + 1e-12))

# --- 5. dealer clamp corner cases ---
for regime in ("long_gamma_dampening", "short_gamma_amplifying", "near_flip", "neutral"):
    for conf in (-3.0, 0.5, 1.0, 7.0, float("nan"), float("inf")):
        ms = MarketStructure(ticker="X", as_of=pd.Timestamp("2026-03-20"), spot=100.0,
                             expiry=pd.Timestamp("2026-04-20").date(),
                             assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
                             regime=regime, confidence=conf)
        m = dealer_regime_multiplier(ms)
        assert np.isfinite(m) and 0.70 <= m <= 1.05, (regime, conf, m)
print("[DEALER] clamp [0.70,1.05] held for all regimes x confidences incl nan/inf")
print("[DEALER] None ->", dealer_regime_multiplier(None))
