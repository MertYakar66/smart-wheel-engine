"""Probe 2: PIT leak differential tests, demotion cascade, NOS phase jitter,
HAR-RV / bootstrap tier honesty."""

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

from engine.forward_distribution import (  # noqa: E402
    best_available_forward_distribution,
    block_bootstrap_log_returns,
    har_rv_conditional_distribution,
    realized_vol_ratio,
)
from engine.wheel_runner import WheelRunner  # noqa: E402

AS_OF = "2026-03-20"
H = 35

conn = WheelRunner().connector
ohlcv = conn.get_ohlcv("AAPL")
cut = pd.Timestamp(AS_OF)

print("--- 1. PIT differential: full frame + as_of  vs  pre-trimmed frame ---")
# Simulate "future rows exist beyond as_of" by using an EARLIER as_of so the
# frame genuinely contains post-cutoff rows the function must ignore.
early = "2025-06-30"
full_r, m1 = best_available_forward_distribution(ohlcv, horizon_days=H, as_of=early)
trimmed = ohlcv.loc[ohlcv.index <= pd.Timestamp(early)]
trim_r, m2 = best_available_forward_distribution(trimmed, horizon_days=H, as_of=None)
print(f"as_of={early}: tiers {m1}/{m2} identical={np.array_equal(full_r, trim_r)}")
r1 = realized_vol_ratio(ohlcv, as_of=early)
r2 = realized_vol_ratio(trimmed, as_of=None)
print(f"rv ratio with-future={r1:.6f} pre-trimmed={r2:.6f} equal={r1 == r2}")
# Same for bootstrap + HAR tiers (force them directly)
b1 = block_bootstrap_log_returns(ohlcv, H, n_scenarios=500, as_of=early)
b2 = block_bootstrap_log_returns(trimmed, H, n_scenarios=500, as_of=None)
h1 = har_rv_conditional_distribution(ohlcv, H, n_scenarios=500, as_of=early)
h2 = har_rv_conditional_distribution(trimmed, H, n_scenarios=500, as_of=None)
print(f"bootstrap identical={np.array_equal(b1, b2)}  har identical={np.array_equal(h1, h2)}")

print("\n--- 2. Demotion cascade on synthetic short histories ---")
rng = np.random.default_rng(7)


def synth(nbars):
    idx = pd.bdate_range("2024-01-02", periods=nbars)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, nbars)))
    return pd.DataFrame({"close": px}, index=idx)


for nbars in [1300, 300, 130, 95, 70, 40]:
    df = synth(nbars)
    r, m = best_available_forward_distribution(df, horizon_days=H)
    print(f"bars={nbars:5d} -> tier={m:26s} n={len(r)}")

print("\n--- 3. Day-over-day NOS phase jitter (AAPL, 8 consecutive as_of) ---")
dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-03-11", "2026-03-20")]
prev = None
for d in dates:
    r, m = best_available_forward_distribution(ohlcv, horizon_days=H, as_of=d)
    sd = np.std(r, ddof=1)
    mu = np.mean(r)
    overlap = ""
    if prev is not None and len(prev) == len(r):
        overlap = f" |d_std|={abs(sd - np.std(prev, ddof=1)):.4f} |d_mean|={abs(mu - np.mean(prev)):.4f}"
    print(f"as_of={d} n={len(r):3d} mean={mu:+.4f} std={sd:.4f}{overlap}")
    prev = r

print("\n--- 4. HAR-RV tier honesty (AAPL, forced) ---")
h = har_rv_conditional_distribution(ohlcv, H, n_scenarios=20000, as_of=AS_OF)
print(f"n={len(h)} mean={np.mean(h):+.5f} std={np.std(h):.4f} "
      f"ann={np.std(h)/np.sqrt(H/252):.4f}")
# determinism
h_b = har_rv_conditional_distribution(ohlcv, H, n_scenarios=20000, as_of=AS_OF)
print(f"deterministic (seed=42): {np.array_equal(h, h_b)}")

print("\n--- 5. Block bootstrap honesty (AAPL, forced) ---")
b = block_bootstrap_log_returns(ohlcv, H, n_scenarios=5000, as_of=AS_OF)
lr = np.diff(np.log(ohlcv.loc[ohlcv.index <= cut, "close"].dropna().values))
lr5 = lr[-(1255):]
print(f"n={len(b)} mean={np.mean(b):+.4f} std={np.std(b):.4f} "
      f"daily-scaled expect std~{np.std(lr5)*np.sqrt(H):.4f} "
      f"mean~{np.mean(lr5)*H:+.4f}")
b_b = block_bootstrap_log_returns(ohlcv, H, n_scenarios=5000, as_of=AS_OF)
print(f"deterministic (seed=42): {np.array_equal(b, b_b)}")

print("\n--- 6. as_of beyond data frontier (stale-tree guard at module level?) ---")
r, m = best_available_forward_distribution(ohlcv, horizon_days=H, as_of="2027-01-01")
print(f"as_of=2027-01-01 -> tier={m} n={len(r)} (module itself has no staleness gate; "
      "ranker gates at max_as_of_staleness_days=30)")
