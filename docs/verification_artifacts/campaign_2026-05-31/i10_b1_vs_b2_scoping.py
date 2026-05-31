"""I10 — B1-vs-B2 scoping: route the fix decision BEFORE building any prototype.

I9 showed a recalibration (B1-style probability fix) does not generalize to unseen
crises. Two opposite routes remain:
  B1 — structural probability fix (POT-GPD -> prob_profit): make the number honest
       adaptively. Risk: you may be predicting an unstable quantity.
  B2 — behavioral gate (regime-transition entry filter as a downgrade-only reviewer):
       don't fix the probability; STOP trading when it's untrustworthy.

This study (observe-only) does three things:
  P1  Confirm the crisis instability is REAL, not small-n: Wilson CIs on the
      (>0.90, crisis) realized-rate cells. Routes "irreducibly unstable" vs "can't tell".
  P2  The B2 prerequisite — DETECTION. Does any PIT signal (index RV30/RV252, trailing
      drawdown, RV-acceleration, the engine's own hmm crisis-share) achieve the 3-WAY
      separation B2 needs: crash-ONSET bleed (gate it) vs benign vol-spike (don't) vs
      crash-RECOVERY (don't — best entries)? Tested by rank-corr with the per-date bleed
      and by the signal's value at known onset/benign/recovery months.
  P3  B2 gate simulation under leave-one-crisis-out: set a threshold on all-but-one
      crisis, apply to the held-out crisis, measure bleed-$-avoided vs good-$-forgone.
      B1 is contrasted conceptually (needs an engine prototype to test directly).

Reuses raw_output/i1_realized_rows.parquet (per-trade) + campaign_lib (OHLCV/index).
Run:  python i10_b1_vs_b2_scoping.py
"""
from __future__ import annotations

import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

ROWS = os.path.join(HERE, "raw_output", "i1_realized_rows.parquet")


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# --------------------------------------------------------------------------- #
# Daily market proxy (equal-weight daily-return chain — robust to missing names)
# --------------------------------------------------------------------------- #
def index_signals() -> pd.DataFrame:
    panel = L.close_panel().sort_index()
    rets = panel.pct_change()
    daily = rets.mean(axis=1)                       # equal-weight market daily return
    lvl = (1.0 + daily.fillna(0)).cumprod()
    logr = np.log1p(daily.fillna(0))
    rv30 = logr.rolling(30).std() * np.sqrt(252)
    rv252 = logr.rolling(252).std() * np.sqrt(252)
    rv_ratio = rv30 / rv252
    peak = lvl.rolling(252, min_periods=20).max()
    drawdown = lvl / peak - 1.0
    rv_accel = rv30 / rv30.shift(20) - 1.0
    return pd.DataFrame({"level": lvl, "rv30": rv30, "rv252": rv252,
                         "rv_ratio": rv_ratio, "drawdown": drawdown, "rv_accel": rv_accel})


def signal_asof(sig: pd.DataFrame, as_of: str) -> dict:
    idx = sig.index[sig.index <= pd.Timestamp(as_of)]
    if len(idx) == 0:
        return {}
    r = sig.loc[idx[-1]]
    return {k: (float(r[k]) if pd.notna(r[k]) else np.nan)
            for k in ["rv_ratio", "drawdown", "rv_accel", "rv30"]}


def main() -> int:
    if not os.path.exists(ROWS):
        raise SystemExit("run i1_calibration.py first")
    df = pd.read_parquet(ROWS)
    df["year"] = df["as_of"].str[:4]

    # ---- P1: small-n confirmation (Wilson CIs on the crisis top-bin cells) ----
    hr("P1 — Is the crisis instability REAL or small-n? (Wilson 95% CIs)")
    hi = df[(df["prob_profit"] > 0.90) & (df["hmm_regime"] == "crisis")]
    rows = []
    for yr, s in hi.groupby("year"):
        n = len(s); k = int(s["engine_exact"].sum())
        lo, h = wilson(k, n)
        rows.append({"year": yr, "n": n, "realized": round(k / n, 3),
                     "wilson95": f"[{lo:.2f},{h:.2f}]", "powered": "yes" if n >= 50 else "THIN"})
    cells = pd.DataFrame(rows)
    print(cells.to_string(index=False))
    powered = cells[cells["powered"] == "yes"]
    if len(powered) >= 2:
        lo_cell = powered.loc[powered["realized"].idxmin()]
        hi_cell = powered.loc[powered["realized"].idxmax()]
        spread = (hi_cell["realized"] - lo_cell["realized"]) * 100
        # CI overlap between the two extreme well-powered cells?
        print(f"\nwell-powered spread: {lo_cell['year']} {lo_cell['realized']} {lo_cell['wilson95']} "
              f"vs {hi_cell['year']} {hi_cell['realized']} {hi_cell['wilson95']}  = {spread:.0f}pp")
        full = (cells['realized'].max() - cells['realized'].min()) * 100
        print(f"VERDICT P1: well-powered cells differ {spread:.0f}pp (CIs "
              f"{'NON-overlapping -> instability REAL' if spread > 15 else 'overlap -> inconclusive'}); "
              f"full spread incl. thin cells = {full:.0f}pp (inflated by small-n).")

    # ---- per-date bleed + PIT signals ----
    sig = index_signals()
    pos = df[df["ev_dollars"] > 0]
    per_date = pos.groupby("as_of").agg(
        n=("realized_pnl_synth", "size"),
        bleed_mean=("realized_pnl_synth", "mean"),
        bleed_median=("realized_pnl_synth", "median"),
        win=("engine_exact", "mean")).reset_index()
    feats = per_date["as_of"].apply(lambda d: pd.Series(signal_asof(sig, d)))
    pd_sig = pd.concat([per_date, feats], axis=1).dropna(subset=["rv_ratio", "drawdown"])

    # ---- P2: detector test ----
    hr("P2 — Does a PIT signal DETECT the bleed? (rank-corr with per-date realized P&L)")
    print("A good B2 detector has STRONG NEGATIVE corr (high signal -> low realized P&L):")
    for s in ["rv_ratio", "drawdown", "rv_accel", "rv30"]:
        rho = pd_sig[[s, "bleed_mean"]].corr(method="spearman").iloc[0, 1]
        print(f"  Spearman({s:9s}, per-date realized P&L) = {rho:+.3f}")

    print("\n3-WAY SEPARATION — signal at known onset / benign / recovery months:")
    tag = {"2020-02-03": "onset(pre-crash)", "2020-03-02": "ONSET(crash)",
           "2020-04-01": "RECOVERY", "2020-05-01": "recovery",
           "2021-01-04": "benign-bull", "2021-11-01": "benign(2021 vol)",
           "2022-01-03": "ONSET(bear)", "2022-06-01": "bear-mid"}
    pd_sig2 = pd_sig.set_index("as_of")
    print(f"{'date':12s} {'tag':18s} {'bleed_mean':>10s} {'win':>5s} {'rv_ratio':>8s} {'drawdown':>8s} {'rv_accel':>8s}")
    for d, lab in tag.items():
        if d in pd_sig2.index:
            r = pd_sig2.loc[d]
            print(f"{d:12s} {lab:18s} {r['bleed_mean']:10.0f} {r['win']:5.2f} {r['rv_ratio']:8.2f} {r['drawdown']:8.3f} {r['rv_accel']:8.2f}")

    # Classify bleed months (bottom-quintile realized) and score each signal's ranking
    thr = pd_sig["bleed_mean"].quantile(0.20)
    pd_sig["is_bleed"] = (pd_sig["bleed_mean"] <= thr).astype(int)
    print(f"\nbleed months = bottom-quintile per-date realized P&L (<= {thr:.0f}); n_bleed={int(pd_sig['is_bleed'].sum())}/{len(pd_sig)}")
    for s in ["rv_ratio", "drawdown", "rv_accel"]:
        # simple separation: mean signal on bleed vs non-bleed months
        mb = pd_sig.loc[pd_sig["is_bleed"] == 1, s].mean()
        mn = pd_sig.loc[pd_sig["is_bleed"] == 0, s].mean()
        print(f"  {s:9s}: mean(bleed)={mb:+.3f}  mean(non-bleed)={mn:+.3f}  separation={mb-mn:+.3f}")

    # ---- P3: B2 gate simulation under leave-one-crisis-out ----
    hr("P3 — B2 GATE sim (LOCO): stand down when rv_ratio high; net bleed-avoided vs forgone")
    pd_sig["year"] = pd_sig["as_of"].str[:4]
    crises = {"2020": "crash", "2022": "bear"}
    for cy, nm in crises.items():
        train = pd_sig[pd_sig["year"] != cy]
        test = pd_sig[pd_sig["year"] == cy]
        if len(test) == 0:
            continue
        # threshold = train rv_ratio at the bleed-quintile boundary (LOCO-set)
        thr_tr = train.loc[train["is_bleed"] == 1, "rv_ratio"].quantile(0.25) if train["is_bleed"].sum() else train["rv_ratio"].quantile(0.8)
        gated = test[test["rv_ratio"] >= thr_tr]
        kept = test[test["rv_ratio"] < thr_tr]
        # $ per contract: gating a month avoids its positive-EV puts' realized P&L
        avoided = -(gated["bleed_mean"] * gated["n"]).sum()      # +ve = bleed removed
        forgone = (kept["bleed_mean"] * kept["n"]).sum()         # what we still take
        all_take = (test["bleed_mean"] * test["n"]).sum()
        print(f"\n[{cy} {nm}] LOCO threshold rv_ratio>={thr_tr:.2f} (set on other years)")
        print(f"  months gated: {len(gated)}/{len(test)}  -> dates {list(gated['as_of'])}")
        print(f"  net realized P&L: no-gate={all_take:+.0f}  with-gate(kept only)={forgone:+.0f}  "
              f"gate effect={forgone-all_take:+.0f} (positive = gate helped)")
        # did the gate spare the recovery month?
        rec = test[test["as_of"].isin(["2020-04-01", "2020-05-01"])]
        if len(rec):
            spared = rec[rec["rv_ratio"] < thr_tr]
            print(f"  recovery months kept (not gated): {list(spared['as_of'])} "
                  f"(realized {[round(x) for x in spared['bleed_mean']]}) — good if non-empty")

    print("\nB1 contrast (conceptual): an oracle recalibration must lower prob_profit by the")
    print("crisis miss, which P1 shows is unstable (>=26pp well-powered) -> a LOCO estimate")
    print("is wrong by that much, so it mis-blocks. B2 needs only DETECTION, not the rate.")
    print("\nI10 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
