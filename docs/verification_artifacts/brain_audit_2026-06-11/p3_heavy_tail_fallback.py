"""Probe 3: heavy-tail penalty path (n>=200, GPD xi>0.3) + penalty asymmetry on
negative ev_raw + fallback EV vs edge_vs_fair relationship (production premium
== BSM fair => fallback EV ~ -costs, conservative)."""

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

from engine.ev_engine import EVEngine, ShortOptionTrade  # noqa: E402
from engine.option_pricer import black_scholes_price  # noqa: E402


def mk(premium, regime=1.0):
    return ShortOptionTrade(
        option_type="put", underlying="SYNTH", spot=100.0, strike=95.0,
        premium=premium, bid=premium * 0.96, ask=premium * 1.04, dte=30,
        iv=0.25, risk_free_rate=0.04, open_interest=1000, regime_multiplier=regime,
    )


eng = EVEngine()

# --- heavy-tail sample: Pareto(alpha=1.8) losses => xi ~ 0.55 ---------------
rng = np.random.default_rng(42)
body = rng.normal(0.01, 0.03, 400)
tail = -(0.05 + 0.08 * rng.pareto(1.8, 100))  # power-law crash returns
fwd = np.concatenate([body, tail])
r = eng.evaluate(mk(2.50), forward_log_returns=fwd)
print(f"heavy sample n={r.n_scenarios}: heavy_tail={r.heavy_tail} xi={r.tail_xi:.3f} "
      f"regime_mult_field={r.regime_multiplier} ev={r.ev_dollars:.2f} raw={r.mean_pnl:.2f}")
if r.heavy_tail:
    print(f"  ratio ev/raw = {r.ev_dollars / r.mean_pnl:.3f} (0.5 penalty confirmed: "
          f"{abs(r.ev_dollars - 0.5 * r.mean_pnl) < 1e-9})")
    print(f"  NOTE raw {'NEGATIVE' if r.mean_pnl < 0 else 'positive'} -> penalty makes ev "
          f"{'LESS negative (optimistic display)' if r.mean_pnl < 0 else 'smaller (conservative)'}")

# positive-raw heavy-tail case: rich premium so raw > 0
r2 = eng.evaluate(mk(9.0), forward_log_returns=fwd)
print(f"rich premium: heavy={r2.heavy_tail} raw={r2.mean_pnl:.2f} ev={r2.ev_dollars:.2f} "
      f"ratio={r2.ev_dollars / r2.mean_pnl if r2.mean_pnl else float('nan'):.3f}")

# --- fallback with premium == BSM fair (production-style synthetic chain) ---
fair = black_scholes_price(S=100.0, K=95.0, T=30 / 365.0, r=0.04, sigma=0.25,
                           option_type="put", q=0.0)
rf = eng.evaluate(mk(float(fair)))  # no forward dist -> internal lognormal fallback
print(f"\nfallback w/ premium==fair({fair:.4f}): src={rf.metadata['distribution_source']} "
      f"ev={rf.ev_dollars:.2f} edge_vs_fair={rf.edge_vs_fair:.2f} "
      f"total_cost={rf.total_transaction_cost:.2f}")
print("  (EV ~ small/negative when premium==fair => fallback conservative on production "
      "synthetic-chain premiums; my p2 +152 came from a synthetically rich premium)")

# fallback drift check for CALLS: risk-neutral drift r=4% vs physical
rc = eng.evaluate(ShortOptionTrade(option_type="call", underlying="S", spot=100.0,
                                   strike=105.0, premium=float(black_scholes_price(
                                       S=100.0, K=105.0, T=30 / 365.0, r=0.04,
                                       sigma=0.25, option_type="call", q=0.0)),
                                   dte=30, iv=0.25, risk_free_rate=0.04,
                                   open_interest=1000))
print(f"call fallback premium==fair: ev={rc.ev_dollars:.2f} (drift=r, not physical mu)")
