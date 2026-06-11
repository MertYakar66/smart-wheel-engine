"""Probe 2: multiplier order + clamps, regime anomaly tags, dealer clamp,
event lockout short-circuit, degenerate inputs (dte<=0, empty/NaN forward),
negative-EV shrink asymmetry, lognormal fallback determinism.
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

from engine.dealer_positioning import dealer_regime_multiplier  # noqa: E402
from engine.ev_engine import EVEngine, ShortOptionTrade, rank_candidates  # noqa: E402


def mk(regime=1.0, dte=30, premium=2.50, strike=95.0, **kw):
    return ShortOptionTrade(
        option_type="put", underlying="SYNTH", spot=100.0, strike=strike,
        premium=premium, bid=premium - 0.10, ask=premium + 0.10, dte=dte,
        iv=0.25, risk_free_rate=0.04, open_interest=1000,
        regime_multiplier=regime, **kw,
    )


fwd = np.array([-0.30, -0.10, -0.06, -0.03, -0.01, 0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12])
eng = EVEngine()

# --- (1) regime multiplier ordering: ev_dollars = ev_raw * regime_mult ----
base = eng.evaluate(mk(1.0), forward_log_returns=fwd)
r08 = eng.evaluate(mk(0.8), forward_log_returns=fwd)
print("[1] regime ordering")
print(f"  ev_raw base={base.mean_pnl:.6f}  r0.8: mean_pnl={r08.mean_pnl:.6f} (must equal raw)")
print(f"  ev_dollars r0.8={r08.ev_dollars:.6f} expect {base.mean_pnl * 0.8:.6f} "
      f"match={abs(r08.ev_dollars - base.mean_pnl * 0.8) < 1e-9}")
print(f"  percentiles unscaled: p50 base={base.pnl_p50:.2f} r0.8={r08.pnl_p50:.2f}")

# --- (2) regime clamp + anomaly tags --------------------------------------
for val, expect_mult, tag in [
    (float("nan"), 1.0, "regime_mult_nonfinite"),
    (float("inf"), 1.0, "regime_mult_nonfinite"),
    (-0.5, 0.0, "regime_mult_negative"),
    (2.0, 1.25, "regime_mult_over_cap"),
]:
    r = eng.evaluate(mk(val), forward_log_returns=fwd)
    ok = abs(r.regime_multiplier - expect_mult) < 1e-12 and tag in r.metadata["regime_anomaly"]
    print(f"[2] regime={val}: mult={r.regime_multiplier} tag={r.metadata['regime_anomaly']!r} "
          f"ev={r.ev_dollars:.4f} {'OK' if ok else 'FAIL'}")

# --- (3) dealer multiplier clamp sweep ------------------------------------
class FakeMS:
    def __init__(self, regime, confidence):
        self.regime = regime
        self.confidence = confidence
        self.gex_total = 1.0e9
        self.flip_distance_pct = 0.05
        self.nearest_put_wall = None
        self.nearest_call_wall = None
        self.pinning_zones = []


print("[3] dealer clamp sweep")
vals = []
for reg in ["long_gamma_dampening", "short_gamma_amplifying", "near_flip", "neutral", "garbage"]:
    for conf in [-5.0, 0.0, 0.5, 1.0, 99.0, float("nan")]:
        try:
            m = dealer_regime_multiplier(FakeMS(reg, conf))
            vals.append(m)
            if not (0.70 - 1e-12 <= m <= 1.05 + 1e-12):
                print(f"  CLAMP VIOLATION: {reg} conf={conf} -> {m}")
        except Exception as e:  # noqa: BLE001
            print(f"  RAISE: {reg} conf={conf} -> {type(e).__name__}: {e}")
print(f"  swept {len(vals)} combos, range=[{min(vals):.4f}, {max(vals):.4f}]")

# dealer mult applied inside evaluate: ev = raw * regime * dealer
ms = FakeMS("short_gamma_amplifying", 1.0)
rd = eng.evaluate(mk(0.8), forward_log_returns=fwd, market_structure=ms)
exp = base.mean_pnl * 0.8 * 0.70
print(f"  evaluate w/ dealer 0.70 x regime 0.8: ev={rd.ev_dollars:.6f} expect {exp:.6f} "
      f"match={abs(rd.ev_dollars - exp) < 1e-9}; dealer_multiplier field={rd.dealer_multiplier}; "
      f"regime_multiplier field={rd.regime_multiplier} (composite); mean_pnl untouched={rd.mean_pnl == base.mean_pnl}")

# --- (4) negative-EV shrink asymmetry -------------------------------------
print("[4] negative-EV trade under regime/dealer shrink (sign must be preserved)")
neg_fwd = np.array([-0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05])
nbase = eng.evaluate(mk(1.0), forward_log_returns=neg_fwd)
nshrunk = eng.evaluate(mk(0.5), forward_log_returns=neg_fwd)
nzero = eng.evaluate(mk(0.0), forward_log_returns=neg_fwd)
nd = eng.evaluate(mk(1.0), forward_log_returns=neg_fwd, market_structure=ms)
print(f"  ev_raw={nbase.ev_dollars:.2f}; x0.5 -> {nshrunk.ev_dollars:.2f} (LESS negative); "
      f"x0.0 -> {nzero.ev_dollars} (zeroed! is < 0: {nzero.ev_dollars < 0}); "
      f"dealer 0.70 -> {nd.ev_dollars:.2f}")
print(f"  sign flip possible? raw<0 and shrunk>0: {nbase.ev_dollars < 0 and nshrunk.ev_dollars > 0}")

# --- (5) event lockout ------------------------------------------------------
class FakeGate:
    def is_blocked(self, ticker, start, end):
        return True, "earnings 2026-06-15 inside holding window"


from datetime import date  # noqa: E402

geng = EVEngine(event_gate=FakeGate())
lr = geng.evaluate(mk(1.0), forward_log_returns=fwd,
                   trade_start=date(2026, 6, 10), trade_end=date(2026, 7, 10))
print("[5] event lockout")
print(f"  ev={lr.ev_dollars} reason={lr.event_lockout_reason!r} n_scen={lr.n_scenarios} "
      f"prob_profit={lr.prob_profit} ci=({lr.prob_profit_ci_low},{lr.prob_profit_ci_high}) "
      f"blocked_meta={lr.metadata.get('blocked')}")
# gate not consulted without dates:
lr2 = geng.evaluate(mk(1.0), forward_log_returns=fwd)
print(f"  WITHOUT dates: lockout fired? {bool(lr2.event_lockout_reason)} (gate needs both dates)")
# convenience rank_candidates: does the ev=0 lockout row pass min_ev=0?
ranked = rank_candidates([mk(1.0)], engine=geng, min_ev=0.0)
print(f"  rank_candidates(min_ev=0) on locked candidate (no dates passed through): n={len(ranked)}")

# --- (6) degenerates ----------------------------------------------------------
print("[6] degenerates")
d0 = eng.evaluate(mk(1.0, dte=0), forward_log_returns=fwd)
print(f"  dte=0: ev={d0.ev_dollars:.4f} edh={d0.expected_days_held} ev_per_day={d0.ev_per_day:.4f} fair={d0.fair_value:.4f}")
dneg = eng.evaluate(mk(1.0, dte=-5), forward_log_returns=fwd)
print(f"  dte=-5: ev={dneg.ev_dollars:.4f} edh={dneg.expected_days_held} fair={dneg.fair_value:.4f}")
e1 = eng.evaluate(mk(1.0), forward_log_returns=np.array([]))
print(f"  empty fwd -> src={e1.metadata['distribution_source']} n={e1.n_scenarios} ev={e1.ev_dollars:.4f}")
e2 = eng.evaluate(mk(1.0), forward_log_returns=np.array([np.nan, np.inf, -np.inf]))
print(f"  all-NaN fwd -> src={e2.metadata['distribution_source']} n={e2.n_scenarios}")
e3 = eng.evaluate(mk(1.0), forward_log_returns=np.array([np.nan, 0.05, np.inf]))
print(f"  partial-NaN fwd -> src={e3.metadata['distribution_source']} n={e3.n_scenarios} (NaN dropped, 1 kept)")
e4 = eng.evaluate(mk(1.0), price_scenarios=np.array([90.0, 100.0, np.nan]))
print(f"  price_scenarios w/ NaN -> src={e4.metadata['distribution_source']} n={e4.n_scenarios}")
# fallback determinism
f1 = eng.evaluate(mk(1.0))
f2 = EVEngine().evaluate(mk(1.0))
print(f"  lognormal fallback deterministic across instances: {f1.ev_dollars == f2.ev_dollars} "
      f"(ev={f1.ev_dollars:.4f}, n={f1.n_scenarios})")
# zero-premium / crossed market
zc = eng.evaluate(mk(1.0, premium=0.0), forward_log_returns=fwd)
print(f"  premium=0: ev={zc.ev_dollars:.4f} net_prem={zc.metadata['net_premium_in']:.4f}")
cross = ShortOptionTrade(option_type="put", underlying="X", spot=100.0, strike=95.0,
                         premium=2.50, bid=2.70, ask=2.40, dte=30, iv=0.25)
cr = eng.evaluate(cross, forward_log_returns=fwd)
print(f"  crossed market (bid>ask): falls to 10% proxy, total_cost={cr.total_transaction_cost:.4f}")

# --- (7) heavy-tail penalty path (n>=200) ----------------------------------
rng = np.random.default_rng(7)
ht = np.concatenate([rng.normal(0.0, 0.04, 260), rng.normal(-0.45, 0.10, 40)])
h = eng.evaluate(mk(1.0), forward_log_returns=ht)
print(f"[7] n=300 heavy sample: heavy_tail={h.heavy_tail} xi={h.tail_xi:.3f} "
      f"regime_mult={h.regime_multiplier} cvar99_evt={h.cvar_99_evt:.2f} "
      f"ev={h.ev_dollars:.2f} raw={h.mean_pnl:.2f}")
if h.heavy_tail:
    print(f"  penalty applied: ev/raw = {h.ev_dollars / h.mean_pnl:.3f} (expect 0.5)")
