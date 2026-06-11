import os, sys, warnings
WT = r'C:\Users\merty\Desktop\swe-main'
os.chdir(WT); sys.path.insert(0, WT); warnings.filterwarnings('ignore')
import engine; assert os.path.abspath(engine.__file__).startswith(WT), engine.__file__

from datetime import date
import numpy as np

print("=== PROBE 1: cross-ticker token consumption ===")
from engine.wheel_tracker import WheelTracker
t = WheelTracker(initial_capital=1_000_000.0, require_ev_authority=True)
row = {"ticker": "AAPL", "strike": 180.0, "premium": 2.50, "dte": 32,
       "ev_dollars": 25.0, "prob_profit": 0.85, "distribution_source": "empirical"}
tok = t.issue_ev_authority_token(row)
# Now try to open a DIFFERENT ticker/strike with the AAPL token
ok = t.open_short_put(
    ticker="MSFT", strike=400.0, premium=0.50,
    entry_date=date(2026, 3, 20), expiration_date=date(2026, 4, 20),
    iv=0.25, ev_authority_token=tok, current_ev_dollars=5.0,
)
print(f"open MSFT with AAPL-issued token -> {ok}")
print(f"positions: {list(t.positions.keys())}")
print(f"audit tail: {t._ev_authority_log[-1]}")

print()
print("=== PROBE 2: dealer_regime_multiplier bounds sweep ===")
from engine.dealer_positioning import dealer_regime_multiplier
class FakeMS:
    def __init__(self, regime, confidence):
        self.regime = regime; self.confidence = confidence
worst = []
for regime in ["long_gamma_dampening", "short_gamma_amplifying", "near_flip",
               "neutral", "FORGED_REGIME", ""]:
    for conf in [-1e9, -1.0, 0.0, 0.5, 1.0, 2.0, 1e9, float("inf"), float("-inf"), float("nan")]:
        try:
            m = dealer_regime_multiplier(FakeMS(regime, conf))
        except Exception as e:
            print(f"  EXC regime={regime!r} conf={conf}: {type(e).__name__}: {e}")
            continue
        if not (0.70 - 1e-12 <= m <= 1.05 + 1e-12):
            worst.append((regime, conf, m))
print(f"out-of-[0.70,1.05] cases: {worst if worst else 'NONE'}")
print(f"None -> {dealer_regime_multiplier(None)}")
# NaN confidence special case
m_nan = dealer_regime_multiplier(FakeMS("short_gamma_amplifying", float("nan")))
print(f"short_gamma conf=nan -> {m_nan} (finite? {np.isfinite(m_nan)})")

print()
print("=== PROBE 3: ev_raw untouched by dealer multiplier; sign preservation ===")
from engine.ev_engine import EVEngine, ShortOptionTrade
trade = ShortOptionTrade(
    option_type="put", underlying="TEST", spot=100.0, strike=95.0,
    premium=0.10, dte=30, iv=0.25, risk_free_rate=0.04, dividend_yield=0.0,
    contracts=1, bid=0.095, ask=0.105, open_interest=1000, regime_multiplier=1.0,
)
rng = np.random.default_rng(7)
# bearish forward distribution -> negative EV trade
fwd = rng.normal(-0.10, 0.05, 4000)
e = EVEngine()
base = e.evaluate(trade, forward_log_returns=fwd)
print(f"base: ev_dollars={base.ev_dollars:.2f} mean_pnl(ev_raw)={base.mean_pnl:.2f} dealer_mult={base.dealer_multiplier}")
ms_long = FakeMS("long_gamma_dampening", 1.0)
ms_long.gex_total = 1e9; ms_long.flip_distance_pct = 0.1
ms_long.nearest_put_wall = None; ms_long.nearest_call_wall = None; ms_long.pinning_zones = []
boosted = e.evaluate(trade, forward_log_returns=fwd, market_structure=ms_long)
print(f"long-gamma(conf=1): ev_dollars={boosted.ev_dollars:.2f} mean_pnl={boosted.mean_pnl:.2f} dealer_mult={boosted.dealer_multiplier}")
assert boosted.mean_pnl == base.mean_pnl, "ev_raw CHANGED by dealer overlay!"
assert boosted.dealer_multiplier <= 1.05 + 1e-12
assert (base.ev_dollars < 0) == (boosted.ev_dollars < 0), "SIGN FLIP!"
print("ev_raw identical across overlay; sign preserved; mult <= 1.05  OK")

print()
print("=== PROBE 4: reviewer downgrade-only — negative EV + best-case extras ===")
from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
rev = EnginePhaseReviewer()
d = CandidateDossier(ticker="AAPL", ev_row={"ev_dollars": -50.0, "spot": 100.0})
v, r, _ = rev.review(d)
print(f"negative EV, no chart -> {v} / {r}")
assert v == "blocked"
for bad in [float("inf"), float("-inf"), float("nan")]:
    d2 = CandidateDossier(ticker="AAPL", ev_row={"ev_dollars": bad})
    v2, r2, _ = rev.review(d2)
    assert v2 == "blocked" and r2 == "ev_non_finite", (bad, v2, r2)
print("inf/-inf/nan all -> blocked/ev_non_finite  OK")
# R11 fires on top-bin + high vix
d3 = CandidateDossier(ticker="AAPL",
                      ev_row={"ev_dollars": 500.0, "spot": 100.0, "prob_profit": 0.95, "cvar_5": -1200.0},
                      vix_level=30.0)
v3, r3, _ = rev.review(d3)
print(f"top-bin + VIX 30 (no chart) -> {v3} / {r3}")  # chart missing fires first (R2)
print("done")
