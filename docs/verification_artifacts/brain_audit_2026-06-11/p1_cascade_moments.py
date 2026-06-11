"""Probe 1: cascade tier + moments for AAPL/BIIB/XOM at as_of=2026-03-20.

Compares the forward distribution's implied vol vs prior realized vol vs
the vol_iv ATM IV, plus NOS sample staleness and widening factor.
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
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from engine.forward_distribution import (  # noqa: E402
    best_available_forward_distribution,
    empirical_forward_log_returns,
    realized_vol_ratio,
    realized_vol_widening_factor,
    realized_vol_widened_log_returns,
)
from engine.wheel_runner import WheelRunner, _resolve_pit_atm_iv  # noqa: E402

AS_OF = "2026-03-20"
H = 35  # ranker default dte_target (calendar DTE passed straight in, D21)

runner = WheelRunner()
conn = runner.connector
print("provider:", type(conn).__name__)

for tkr in ["AAPL", "BIIB", "XOM"]:
    ohlcv = conn.get_ohlcv(tkr)
    cut = pd.Timestamp(AS_OF)
    pit = ohlcv.loc[ohlcv.index <= cut]
    print(f"\n=== {tkr} ===  rows total={len(ohlcv)} pit={len(pit)} last_pit={pit.index.max().date()}")

    rets, method = best_available_forward_distribution(ohlcv, horizon_days=H, as_of=AS_OF)
    n = len(rets)
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sk = float(stats.skew(rets))
    ku = float(stats.kurtosis(rets))
    print(f"tier={method} n={n} mean={mu:+.4f} std={sd:.4f} skew={sk:+.2f} exkurt={ku:+.2f}")
    print(f"min={np.min(rets):+.4f} max={np.max(rets):+.4f} p05={np.percentile(rets,5):+.4f}")

    # Annualize: the sampler treats horizon_days as TRADING BARS (D21)
    sig_ann_bars = sd / np.sqrt(H / 252.0)  # what the dist actually spans (35 bars)
    sig_ann_cal = sd / np.sqrt(H / 365.0)  # if you believed it was 35 calendar days
    print(f"implied ann sigma (35 trading bars, true span): {sig_ann_bars:.4f}")
    print(f"implied ann sigma (if 35 cal days, engine's implicit claim): {sig_ann_cal:.4f}")

    # Prior realized vols at as_of
    lr = np.diff(np.log(pit["close"].dropna().values))
    rv21 = float(np.std(lr[-21:], ddof=1)) * np.sqrt(252)
    rv30 = float(np.std(lr[-30:], ddof=1)) * np.sqrt(252)
    rv252 = float(np.std(lr[-252:], ddof=1)) * np.sqrt(252)
    print(f"prior realized ann sigma: 21d={rv21:.4f} 30d={rv30:.4f} 252d={rv252:.4f}")

    ratio = realized_vol_ratio(ohlcv, as_of=AS_OF)
    wf = realized_vol_widening_factor(ohlcv, as_of=AS_OF)
    print(f"rv30/rv252 ratio={ratio:.4f} widening_factor={wf:.4f}")
    wrets = realized_vol_widened_log_returns(rets, ohlcv, as_of=AS_OF)
    print(f"widened std={np.std(wrets, ddof=1):.4f} (vs {sd:.4f}); mean preserved: "
          f"{abs(np.mean(wrets) - mu) < 1e-12}")

    # ATM IV from vol_iv file (PIT)
    iv = _resolve_pit_atm_iv(conn, tkr, AS_OF)
    print(f"vol_iv ATM IV (decimal) at as_of: {iv}")

    # NOS phase staleness: which window does the LAST NOS sample cover?
    df5 = pit.loc[pit.index >= pit.index.max() - pd.Timedelta(days=int(5 * 365))]
    prices = df5["close"].dropna()
    nbar = len(prices)
    nfwd = nbar - H
    last_start = ((nfwd - 1) // H) * H
    last_end = last_start + H
    stale_bars = (nbar - 1) - last_end
    print(f"NOS window: bars={nbar} samples={(nfwd - 1)//H + 1} "
          f"last sample ends {prices.index[last_end].date()} "
          f"({stale_bars} bars before as_of)")

    # Cross-check NOS count against the function
    nos = empirical_forward_log_returns(ohlcv, horizon_days=H, as_of=AS_OF)
    print(f"empirical NOS n={len(nos)} (matches tier n: {len(nos) == n})")
