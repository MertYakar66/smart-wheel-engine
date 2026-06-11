"""Probe 4: real production ranker path — verify EV-core identities on real data:
ev_dollars == ev_raw x regime_multiplier (2dp rounding), distribution_source tier,
n_scenarios, Wilson CI suppression on non-IID tiers, dealer neutrality in sandbox.
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

from engine.wheel_runner import WheelRunner  # noqa: E402

wr = WheelRunner()
print(f"provider={type(wr.connector).__name__}")
df = wr.rank_candidates_by_ev(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
    top_n=10,
    min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
cols = [c for c in [
    "ticker", "strike", "premium", "iv", "ev_raw", "ev_dollars", "regime_multiplier",
    "dealer_multiplier", "prob_profit", "n_scenarios", "prob_profit_ci_low",
    "prob_profit_ci_high", "distribution_source", "heavy_tail", "tail_xi",
] if c in df.columns]
print(df[cols].to_string())
print("\ncolumns:", sorted(df.columns.tolist()))
if "ev_raw" in df.columns and "regime_multiplier" in df.columns:
    chk = (df["ev_raw"] * df["regime_multiplier"]).round(2)
    print("\nev_dollars == round(ev_raw*regime_mult,2):",
          (abs(chk - df["ev_dollars"]) <= 0.011).all())
