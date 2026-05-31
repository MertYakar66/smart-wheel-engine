"""I6 (Wave 2) -- deepening the highest-value campaign findings.

All analysis on the I1 realized rows (run i1_calibration.py first; it writes
raw_output/i1_realized_rows.parquet). No new heavy compute.

  W2-A  Does the HMM regime overlay EARN ITS KEEP? On Bloomberg the regime
        multiplier == HMM multiplier (dealer/skew/news/credit are 1.0). Test
        whether the HMM label actually predicts realized outcome and whether the
        multiplier de-rates the regimes that realize worse.
  W2-B  Which engine signal best SELECTS profitable trades? Monthly top-K by
        ev_dollars (haircut) vs ev_raw vs prob_profit vs ev_roc vs random.
  W2-C  Out-of-sample RECALIBRATION demo: train a per-bin (and per-bin x regime)
        map on 2020-2023, apply to 2024-2026, measure ECE/Brier before/after.
        Quantifies how much a regime-conditional haircut would FIX I1's top-bin
        over-confidence (observe-only -- demonstrates fix value, changes no engine code).

Run:  python i6_deepening.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402,F401  (sets sys.path; used indirectly)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

ROWS = os.path.join(HERE, "raw_output", "i1_realized_rows.parquet")
BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def ece(p, o, bins=BINS):
    """Expected calibration error (pp) given forecasts p and outcomes o."""
    df = pd.DataFrame({"p": p, "o": o})
    df["b"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    g = df.groupby("b", observed=True)
    tot = len(df)
    return float(sum(len(s) * abs(s["p"].mean() - s["o"].mean()) for _, s in g) / tot) * 100


def main() -> int:
    if not os.path.exists(ROWS):
        raise SystemExit("run i1_calibration.py first (writes i1_realized_rows.parquet)")
    df = pd.read_parquet(ROWS)
    df["year"] = df["as_of"].str[:4]
    print(f"loaded {len(df)} realized rows, {df['as_of'].nunique()} dates")

    # ---------------------------------------------------------------- W2-A
    hr("W2-A  Does the HMM regime overlay earn its keep?")
    print("Per-regime realized outcome + the multiplier the engine applied:")
    g = df.groupby("hmm_regime", observed=True).agg(
        n=("realized_pnl_synth", "size"),
        mean_regime_mult=("regime_multiplier", "mean"),
        mean_prob_profit=("prob_profit", "mean"),
        realized_win=("engine_exact", "mean"),
        realized_mean_pnl=("realized_pnl_synth", "mean"),
        realized_median_pnl=("realized_pnl_synth", "median"),
    ).round(3).sort_values("mean_regime_mult")
    print(g.to_string())
    print("\nReading: if the regimes the engine de-rates most (lowest mean_regime_mult)")
    print("also realize the WORST (lowest win / median), the haircut is directionally")
    print("correct. Rank-corr(mean_regime_mult, realized_median across regimes):")
    print(f"  Spearman = {g['mean_regime_mult'].corr(g['realized_median_pnl'], method='spearman'):+.3f}")
    # Does the multiplier improve EV ranking vs raw? (sign-discipline check)
    neg_ev = df[df["ev_dollars"] <= 0]
    pos_ev = df[df["ev_dollars"] > 0]
    print(f"\nsign discipline: ev_dollars<=0 rows realize median "
          f"{neg_ev['realized_pnl_synth'].median():.0f} (n={len(neg_ev)}); "
          f"ev_dollars>0 realize median {pos_ev['realized_pnl_synth'].median():.0f} (n={len(pos_ev)})")

    # ---------------------------------------------------------------- W2-B
    hr("W2-B  Which engine signal best SELECTS profitable trades? (monthly top-K)")
    df["ev_roc"] = df["ev_raw"] / df["collateral"].replace(0, np.nan)
    K = 10
    scorers = ["ev_dollars", "ev_raw", "prob_profit", "ev_roc"]
    rng = np.random.default_rng(13)
    rows = []
    for sc in scorers + ["random", "all"]:
        picks = []
        for ao, sub in df.groupby("as_of"):
            sub = sub.dropna(subset=["realized_pnl_synth"])
            if not len(sub):
                continue
            if sc == "all":
                picks.append(sub)
            elif sc == "random":
                picks.append(sub.sample(min(K, len(sub)), random_state=int(ao.replace("-", "")[:8]) % (2**31)))
            else:
                picks.append(sub.nlargest(K, sc))
        p = pd.concat(picks)
        x = p["realized_pnl_synth"].values.astype(float)
        # ROC-space realized for a scale-free Sharpe-like stat
        roc = (p["realized_pnl_synth"] / p["collateral"].replace(0, np.nan)).dropna().values
        sharpe = float(roc.mean() / roc.std()) if roc.std() > 0 else float("nan")
        rows.append({
            "selector": sc, "n": len(p), "mean_pnl": round(float(x.mean()), 1),
            "median_pnl": round(float(np.median(x)), 1), "win": round(float((x > 0).mean()), 3),
            "realized_roc_mean_%": round(float(np.nanmean(roc) * 100), 3),
            "roc_sharpe": round(sharpe, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nReading: if ev_dollars/ev_roc top-K beats random and 'all', the engine's")
    print("ranking adds selection value; compare which signal ranks best by realized ROC.")

    # ---------------------------------------------------------------- W2-C
    hr("W2-C  Out-of-sample recalibration demo (does a regime-conditional haircut fix it?)")
    train = df[df["year"] <= "2023"].copy()
    test = df[df["year"] >= "2024"].copy()
    print(f"train (<=2023) n={len(train)}  test (>=2024) n={len(test)}")

    def bin_of(p):
        return pd.cut([p], bins=BINS, include_lowest=True)[0]

    # (1) per-bin recalibration map
    train["b"] = pd.cut(train["prob_profit"], bins=BINS, include_lowest=True)
    binmap = train.groupby("b", observed=True)["engine_exact"].mean()
    test["b"] = pd.cut(test["prob_profit"], bins=BINS, include_lowest=True)
    test["recal_bin"] = test["b"].map(binmap).astype(float)

    # (2) per-bin x regime recalibration map (the regime-conditional fix)
    train["br"] = list(zip(train["b"], train["hmm_regime"]))
    brmap = train.groupby(["b", "hmm_regime"], observed=True)["engine_exact"].mean()
    def lookup_br(row):
        key = (row["b"], row["hmm_regime"])
        if key in brmap.index:
            return float(brmap.loc[key])
        return float(binmap.get(row["b"], row["prob_profit"]))  # fallback to bin-only
    test["recal_binregime"] = test.apply(lookup_br, axis=1)

    o = test["engine_exact"].astype(float).values
    raw = test["prob_profit"].astype(float).values
    rb = test["recal_bin"].fillna(test["prob_profit"]).astype(float).values
    rbr = test["recal_binregime"].fillna(test["prob_profit"]).astype(float).values

    def brier(p, o):
        return float(np.mean((p - o) ** 2))

    print(f"\nOUT-OF-SAMPLE (test >=2024):")
    print(f"  raw prob_profit         : ECE={ece(raw,o):.2f}pp  Brier={brier(raw,o):.4f}")
    print(f"  +per-bin recalibration  : ECE={ece(rb,o):.2f}pp  Brier={brier(rb,o):.4f}")
    print(f"  +bin x regime recalib   : ECE={ece(rbr,o):.2f}pp  Brier={brier(rbr,o):.4f}")
    # top-bin specifically
    for lbl, lo, hi in [(">0.90", 0.90, 1.01), (">0.95", 0.95, 1.01)]:
        m = (raw > lo) & (raw <= hi)
        if m.sum() > 5:
            print(f"  test {lbl} bin (n={int(m.sum())}): raw fc={raw[m].mean():.3f} realized={o[m].mean():.3f} "
                  f"| per-bin-recal fc={rb[m].mean():.3f} | binxregime fc={rbr[m].mean():.3f}")
    print("\nReading: if recalibrated ECE << raw ECE out-of-sample, a histogram/regime-")
    print("conditional haircut on prob_profit generalizes -- i.e. the over-confidence is")
    print("learnable and fixable (the candidate fix flagged in I1/I3-E). Observe-only.")
    print("\nI6 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
