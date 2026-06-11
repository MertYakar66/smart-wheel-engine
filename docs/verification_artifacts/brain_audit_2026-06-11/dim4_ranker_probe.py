"""Dimension 4 probe A: multiplier envelopes across ~20 real tickers at the data frontier."""

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

from engine.wheel_runner import WheelRunner  # noqa: E402

TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "UNH", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "COST", "WMT", "PG", "JNJ", "V", "HD", "KO", "PEP", "MRK", "BAC",
]

wr = WheelRunner()
print("connector:", type(wr.connector).__name__)
df = wr.rank_candidates_by_ev(
    tickers=TICKERS,
    top_n=50,
    min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
print("rows:", len(df))
cols = [
    "ticker", "ev_dollars", "hmm_multiplier", "hmm_regime",
    "hmm_realized_vol_252d_ann", "hmm_realized_return_252d_ann",
    "tail_widening_factor", "dealer_multiplier", "dealer_regime",
    "regime_multiplier", "skew_multiplier", "news_multiplier",
    "credit_multiplier", "heavy_tail", "tail_xi", "cvar_99_evt", "ev_raw",
]
have = [c for c in cols if c in df.columns]
print(df[have].to_string())

print("\n--- envelope checks ---")
hm = df["hmm_multiplier"].astype(float)
print("hmm_multiplier: min=%.4f max=%.4f all finite=%s in [0.2,1.25]=%s"
      % (hm.min(), hm.max(), np.isfinite(hm).all(), bool(((hm >= 0.2 - 1e-9) & (hm <= 1.25 + 1e-9)).all())))
tw = df["tail_widening_factor"].astype(float)
print("tail_widening_factor: min=%.4f max=%.4f all finite=%s in [1.0,1.15]=%s"
      % (tw.min(), tw.max(), np.isfinite(tw).all(), bool(((tw >= 1.0 - 1e-9) & (tw <= 1.15 + 1e-9)).all())))
dm = df["dealer_multiplier"].astype(float)
print("dealer_multiplier: unique=%s (sandbox neutral expected 1.0)" % sorted(dm.unique()))
rm = df["regime_multiplier"].astype(float)
print("regime_multiplier(final): min=%.4f max=%.4f all finite=%s"
      % (rm.min(), rm.max(), np.isfinite(rm).all()))
print("label distribution:", df["hmm_regime"].value_counts().to_dict())
print("heavy_tail any:", bool(df["heavy_tail"].any()), "| tail_xi non-null:", int(df["tail_xi"].notna().sum()))
# ev_dollars / ev_raw should equal regime_multiplier (where ev_raw != 0)
ratio = df["ev_dollars"].astype(float) / df["ev_raw"].replace(0, np.nan).astype(float)
mism = (ratio - rm).abs()
print("max |ev_dollars/ev_raw - regime_multiplier| =", float(mism.max()))
