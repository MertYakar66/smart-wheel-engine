"""I1 -- full-universe point-in-time prob_profit calibration.

Consumes the monthly ranked snapshots (rank_snapshots.py), realizes each
short-put forecast against the actual underlying path, and measures calibration:

  * Reliability table: forecast prob_profit bin -> realized rate, with Wilson 95% CI.
  * Two attributions, reported side by side to resolve the ~12pp dispute:
      - engine_exact  (close > strike - premium): the engine's OWN prob_profit
        definition P(pnl>0). This is the apples-to-apples calibration target.
      - otm_expire    (close >= strike): "did the put expire worthless" -- the
        assignment-avoidance question the prior studies used.
  * Brier score + Expected Calibration Error (ECE) for both attributions
    (neither prior study computed these).
  * Top-bin over-confidence with binomial CI (HT-B's top bin was n=8; full
    universe lifts n by orders of magnitude).
  * EV calibration: forecast ev_raw bin -> realized mean P&L per contract.
  * Stratification by regime / distribution_source / year.

Run:  python i1_calibration.py            # uses whatever snapshots exist
"""
from __future__ import annotations

import os
import sys
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Windows console defaults to cp1252; force UTF-8 so Greek/math chars don't crash.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

SNAP_DIR = os.path.join(HERE, "snapshots")
OUT_DIR = os.path.join(HERE, "raw_output")
os.makedirs(OUT_DIR, exist_ok=True)

BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]
BIN_LABELS = ["(0,.5]", "(.5,.6]", "(.6,.7]", "(.7,.8]", "(.8,.85]",
              "(.85,.9]", "(.9,.95]", "(.95,1]"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def verdict(delta_pp: float) -> str:
    a = abs(delta_pp)
    return "OK" if a <= 5 else ("WARN" if a <= 10 else "MISCAL")


# --------------------------------------------------------------------------- #
def load_realized() -> pd.DataFrame:
    files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("put_") and f.endswith(".parquet")])
    if not files:
        raise SystemExit("no snapshots yet")
    frames = [pd.read_parquet(os.path.join(SNAP_DIR, f)) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(files)} snapshots, {len(df)} candidate rows")

    # Vectorized-ish realization via cached per-ticker close series.
    rc, otm, exact, pnl = [], [], [], []
    for tk, k, ao, dte, prem in zip(df["ticker"], df["strike"], df["as_of"],
                                    df["dte"], df["premium"]):
        r = L.realize_short_put(tk, float(k), ao, int(dte), float(prem))
        if r is None:
            rc.append(np.nan); otm.append(np.nan); exact.append(np.nan); pnl.append(np.nan)
        else:
            rc.append(r["realized_close"]); otm.append(r["otm_expire"])
            exact.append(r["engine_exact"]); pnl.append(r["realized_pnl_synth"])
    df["realized_close"] = rc
    df["otm_expire"] = otm
    df["engine_exact"] = exact
    df["realized_pnl_synth"] = pnl
    n0 = len(df)
    df = df.dropna(subset=["realized_close"]).copy()
    print(f"realized {len(df)}/{n0} rows ({n0 - len(df)} unrealizable -- data ends before expiry)")
    df["bin"] = pd.cut(df["prob_profit"], bins=BINS, labels=BIN_LABELS, right=True, include_lowest=True)
    return df


def reliability_table(df: pd.DataFrame, outcome_col: str, title: str) -> pd.DataFrame:
    rows = []
    for lbl in BIN_LABELS:
        sub = df[df["bin"] == lbl]
        n = len(sub)
        if n == 0:
            continue
        fc = float(sub["prob_profit"].mean())
        k = int(sub[outcome_col].sum())
        realized = k / n
        lo, hi = wilson(k, n)
        delta = (realized - fc) * 100
        rows.append({
            "bin": lbl, "n": n, "fc_mean": round(fc, 4), "realized": round(realized, 4),
            "ci95": f"[{lo:.3f},{hi:.3f}]", "delta_pp": round(delta, 2),
            "verdict": verdict(delta),
        })
    rt = pd.DataFrame(rows)
    # Brier + ECE
    o = df[outcome_col].astype(float).values
    p = df["prob_profit"].astype(float).values
    brier = float(np.mean((p - o) ** 2))
    ece = float((rt["n"] * (rt["fc_mean"] - rt["realized"]).abs()).sum() / rt["n"].sum())
    wmad = float((rt["n"] * rt["delta_pp"].abs()).sum() / rt["n"].sum())
    print(f"\n### {title}   (Brier={brier:.4f}  ECE={ece*100:.2f}pp  weightedMAD={wmad:.2f}pp  N={len(df)})")
    print(rt.to_string(index=False))
    return rt, {"brier": brier, "ece_pp": ece * 100, "wmad_pp": wmad}


def _boot_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean (fat-tailed data -> mean is unstable)."""
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def ev_calibration(df: pd.DataFrame) -> None:
    print("\n### EV calibration: forecast ev_raw bin -> realized P&L / contract")
    print("    (mean is fat-tail-dominated for short puts; median + win-rate + bootstrap CI added)")
    ev_bins = [-1e9, -200, -50, 0, 50, 100, 200, 500, 1e9]
    df["ev_bin"] = pd.cut(df["ev_raw"], bins=ev_bins)
    rows = []
    for b, sub in df.groupby("ev_bin", observed=True):
        x = sub["realized_pnl_synth"].values.astype(float)
        lo, hi = _boot_mean_ci(x)
        rows.append({
            "ev_bin": str(b), "n": len(sub), "ev_raw_mean": round(sub["ev_raw"].mean(), 1),
            "real_mean": round(float(x.mean()), 1), "mean_ci95": f"[{lo:.0f},{hi:.0f}]",
            "real_median": round(float(np.median(x)), 1),
            "win_rate": round(float((x > 0).mean()), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ROC space removes the notional-size confound ($500-stock swings dwarf $50-stock).
    df["ev_roc"] = df["ev_raw"] / df["collateral"].replace(0, np.nan)
    df["real_roc"] = df["realized_pnl_synth"] / df["collateral"].replace(0, np.nan)
    sp_d = df[["ev_raw", "realized_pnl_synth"]].corr(method="spearman").iloc[0, 1]
    sp_roc = df[["ev_roc", "real_roc"]].dropna().corr(method="spearman").iloc[0, 1]
    sp_pp = df[["prob_profit", "realized_pnl_synth"]].corr(method="spearman").iloc[0, 1]
    sp_dollars = df[["ev_dollars", "realized_pnl_synth"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman(ev_raw,   realized_pnl)   = {sp_d:+.4f}   (dollar rank-discrimination)")
    print(f"Spearman(ev_roc,   realized_roc)   = {sp_roc:+.4f}   (ROC-normalized, no size confound)")
    print(f"Spearman(ev_dollars,realized_pnl)  = {sp_dollars:+.4f}   (haircut ranking score)")
    print(f"Spearman(prob_profit,realized_pnl) = {sp_pp:+.4f}")

    # Tail decomposition: do a few catastrophes drive the mean inversion?
    worst = df.nsmallest(10, "realized_pnl_synth")[["ticker", "as_of", "strike",
            "premium", "realized_close", "ev_raw", "realized_pnl_synth", "hmm_regime"]]
    print("\n### 10 worst realized trades (tail that dominates the mean):")
    print(worst.to_string(index=False))
    x_all = df["realized_pnl_synth"].values.astype(float)
    cut = np.percentile(x_all, 1)
    trimmed = x_all[x_all >= cut]
    print(f"\nmean realized P&L: all={x_all.mean():.1f}  trimmed-bottom-1%={trimmed.mean():.1f} "
          f"(n_trimmed={len(x_all)-len(trimmed)})")

    # Crash-excluded sanity: does the inversion survive removing the 2020 crash window?
    nc = df[~((df["as_of"] >= "2020-01-01") & (df["as_of"] <= "2020-06-30"))]
    if len(nc) > 100:
        sp_nc = nc[["ev_raw", "realized_pnl_synth"]].corr(method="spearman").iloc[0, 1]
        print(f"\nSpearman(ev_raw, realized_pnl) EXCLUDING 2020 crash = {sp_nc:+.4f}  (n={len(nc)})")


def stratify(df: pd.DataFrame) -> None:
    for col in ("hmm_regime", "distribution_source"):
        if col not in df.columns:
            continue
        print(f"\n### top-bin (.9,1] engine_exact realized rate by {col}")
        top = df[df["prob_profit"] > 0.9]
        for val, sub in top.groupby(col, observed=True):
            n = len(sub)
            if n < 10:
                continue
            fc = sub["prob_profit"].mean()
            re_ = sub["engine_exact"].mean()
            ro = sub["otm_expire"].mean()
            print(f"  {str(val):28s} n={n:5d} fc={fc:.3f} exact={re_:.3f} (Δ{(re_-fc)*100:+.1f}) otm={ro:.3f} (Δ{(ro-fc)*100:+.1f})")
    # by year
    print("\n### weighted-MAD (engine_exact) by entry year")
    df["year"] = df["as_of"].str[:4]
    for yr, sub in df.groupby("year"):
        rows = []
        for lbl in BIN_LABELS:
            s2 = sub[sub["bin"] == lbl]
            if len(s2):
                rows.append((len(s2), abs((s2["engine_exact"].mean() - s2["prob_profit"].mean()) * 100)))
        if rows:
            wmad = sum(n * d for n, d in rows) / sum(n for n, _ in rows)
            print(f"  {yr}: n={len(sub):5d} weightedMAD={wmad:.2f}pp")


def main() -> int:
    df = load_realized()
    summary = {"n_realized": len(df), "n_dates": df["as_of"].nunique()}
    rt_exact, m_exact = reliability_table(df, "engine_exact",
                                          "RELIABILITY (engine_exact = close > strike-premium = engine's prob_profit definition)")
    rt_otm, m_otm = reliability_table(df, "otm_expire",
                                      "RELIABILITY (otm_expire = close >= strike = prior-study convention)")
    summary["engine_exact"] = m_exact
    summary["otm_expire"] = m_otm

    # Top-bin deep dive
    print("\n### TOP-BIN OVER-CONFIDENCE (.95,1] and (.9,.95]")
    for lbl in ["(.9,.95]", "(.95,1]"]:
        sub = df[df["bin"] == lbl]
        if not len(sub):
            continue
        for oc in ("engine_exact", "otm_expire"):
            k, n = int(sub[oc].sum()), len(sub)
            lo, hi = wilson(k, n)
            fc = sub["prob_profit"].mean()
            print(f"  {lbl} [{oc}] n={n} fc={fc:.4f} realized={k/n:.4f} CI95=[{lo:.3f},{hi:.3f}] Δ={(k/n-fc)*100:+.2f}pp")

    ev_calibration(df)
    stratify(df)

    with open(os.path.join(OUT_DIR, "i1_calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    out = df.copy()
    for c in ("bin", "ev_bin"):  # Interval/categorical -> str so parquet can serialize
        if c in out.columns:
            out[c] = out[c].astype(str)
    out.to_parquet(os.path.join(OUT_DIR, "i1_realized_rows.parquet"), index=False)
    print(f"\nsaved summary + {len(df)} realized rows to raw_output/")
    print("I1 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
