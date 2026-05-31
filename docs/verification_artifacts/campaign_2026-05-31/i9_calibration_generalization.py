"""I9 (B-verification gate) — does the calibration fix GENERALIZE to an unseen crisis?

The case for "the top-bin over-confidence is fixable" rests on I6-C: a recalibration
map cut out-of-sample ECE 3.17->1.29pp. BUT I6-C's test window (2024-2026) was benign.
A fix that only works on calm test data is worthless — the over-confidence bites in
CRISIS. This study is the gate before committing to a re-baseline: does a recalibration
learned WITHOUT seeing a given crisis still fix that crisis's over-confidence?

SCOPE: this tests the RECALIBRATION-LAYER fix (a histogram map on prob_profit, the one
I6-C demonstrated). It does NOT test wiring POT-GPD into prob_profit — that needs an
engine prototype and is out of observe-only scope. Conclusions apply to recalibration.

Three tests, all on raw_output/i1_realized_rows.parquet (run i1_calibration.py first):
  T1 Walk-forward by year: train on years < Y, test on Y. Does a map fit on history
     fix the NEXT period (incl. the 2022 bear)?
  T2 Leave-one-crisis-out: train EXCLUDING a crisis year, test on it. The hard test:
     does a map fit on non-crash data fix 2020 / 2022?
  T3 Regime-holdout: train on hmm_regime != 'crisis', test on 'crisis'. Does a fix
     learned without crisis data fix crisis over-confidence?
  + STABILITY: is the (top-bin, crisis) realized rate consistent ACROSS crises? (If yes,
    a regime-conditional map generalizes; if not, even regime-conditioning fails.)

Run:  python i9_calibration_generalization.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

ROWS = os.path.join(HERE, "raw_output", "i1_realized_rows.parquet")
BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]
OUTCOME = "engine_exact"   # the engine's own prob_profit definition


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def binlab(p):
    return pd.cut(p, bins=BINS, include_lowest=True)


def ece_brier(p, o):
    p = np.asarray(p, float); o = np.asarray(o, float)
    if len(p) == 0:
        return float("nan"), float("nan")
    d = pd.DataFrame({"p": p, "o": o, "b": binlab(p)})
    n = len(d)
    e = float(sum(len(s) * abs(s["p"].mean() - s["o"].mean()) for _, s in d.groupby("b", observed=True)) / n) * 100
    return e, float(np.mean((p - o) ** 2))


def fit_map(train, by_regime):
    train = train.copy()
    train["b"] = binlab(train["prob_profit"])
    if by_regime:
        m = train.groupby(["b", "hmm_regime"], observed=True)[OUTCOME].mean()
    else:
        m = train.groupby("b", observed=True)[OUTCOME].mean()
    binonly = train.groupby("b", observed=True)[OUTCOME].mean()
    return m, binonly


def apply_map(test, m, binonly, by_regime):
    test = test.copy()
    test["b"] = binlab(test["prob_profit"])
    out = []
    for _, r in test.iterrows():
        if by_regime:
            key = (r["b"], r["hmm_regime"])
            v = m.loc[key] if key in m.index else (binonly.loc[r["b"]] if r["b"] in binonly.index else r["prob_profit"])
        else:
            v = m.loc[r["b"]] if r["b"] in m.index else r["prob_profit"]
        out.append(float(v))
    return np.asarray(out, float)


def topbin(p_raw, p_recal, o, lo=0.90):
    """Report the >lo region: raw forecast vs realized vs recalibrated forecast."""
    m = np.asarray(p_raw, float) > lo
    n = int(m.sum())
    if n == 0:
        return f"(>{lo}: n=0)"
    return (f"(>{lo}: n={n}) raw_fc={np.asarray(p_raw)[m].mean():.3f} "
            f"realized={np.asarray(o)[m].mean():.3f} "
            f"recal_fc={np.asarray(p_recal)[m].mean():.3f}")


def evaluate(train, test, label):
    o = test[OUTCOME].astype(float).values
    raw = test["prob_profit"].astype(float).values
    e_raw, b_raw = ece_brier(raw, o)
    res = {"label": label, "n_train": len(train), "n_test": len(test),
           "ece_raw": e_raw, "brier_raw": b_raw}
    for by_reg, tag in [(False, "bin"), (True, "binxregime")]:
        m, binonly = fit_map(train, by_reg)
        rec = apply_map(test, m, binonly, by_reg)
        e, b = ece_brier(rec, o)
        res[f"ece_{tag}"] = e
        res[f"brier_{tag}"] = b
        res[f"top_{tag}"] = topbin(raw, rec, o)
    return res


def show(res):
    print(f"\n[{res['label']}]  n_train={res['n_train']} n_test={res['n_test']}")
    print(f"  ECE   raw={res['ece_raw']:.2f}pp  +bin={res['ece_bin']:.2f}pp  +binxregime={res['ece_binxregime']:.2f}pp")
    print(f"  Brier raw={res['brier_raw']:.4f}  +bin={res['brier_bin']:.4f}  +binxregime={res['brier_binxregime']:.4f}")
    print(f"  top  bin-only   : {res['top_bin']}")
    print(f"  top  binxregime : {res['top_binxregime']}")


def main() -> int:
    if not os.path.exists(ROWS):
        raise SystemExit("run i1_calibration.py first (writes i1_realized_rows.parquet)")
    df = pd.read_parquet(ROWS)
    df["year"] = df["as_of"].str[:4]
    print(f"loaded {len(df)} realized rows; years {sorted(df['year'].unique())}")
    print(f"outcome = {OUTCOME} (engine's own prob_profit definition)")

    # ---- T1: walk-forward by year ----
    hr("T1 — WALK-FORWARD: train on years < Y, test on Y (does history fix the next period?)")
    for Y in ["2021", "2022", "2023", "2024", "2025"]:
        tr = df[df["year"] < Y]; te = df[df["year"] == Y]
        if len(tr) < 200 or len(te) < 50:
            continue
        show(evaluate(tr, te, f"train<{Y} -> test {Y}" + ("  [2022 = BEAR]" if Y == "2022" else "")))

    # ---- T2: leave-one-crisis-out ----
    hr("T2 — LEAVE-ONE-CRISIS-OUT: train EXCLUDING the crisis, test on it (the hard test)")
    for cy in ["2020", "2022"]:
        tr = df[df["year"] != cy]; te = df[df["year"] == cy]
        tag = "CRASH" if cy == "2020" else "BEAR"
        show(evaluate(tr, te, f"train(all but {cy}) -> test {cy}  [{tag}]"))

    # ---- T3: regime-holdout ----
    hr("T3 — REGIME-HOLDOUT: train on hmm_regime != 'crisis', test on 'crisis'")
    tr = df[df["hmm_regime"] != "crisis"]; te = df[df["hmm_regime"] == "crisis"]
    show(evaluate(tr, te, "train(non-crisis) -> test(crisis regime)"))

    # ---- Stability: is the (top-bin, crisis) realized rate consistent across crises? ----
    hr("STABILITY — realized engine_exact rate in the >0.90 bin, per (year, hmm_regime)")
    hi = df[df["prob_profit"] > 0.90]
    g = hi.groupby(["year", "hmm_regime"], observed=True).agg(
        n=("prob_profit", "size"), fc=("prob_profit", "mean"), realized=(OUTCOME, "mean")).round(3)
    g = g[g["n"] >= 15]
    print(g.to_string())
    cr = hi[hi["hmm_regime"] == "crisis"].groupby("year", observed=True)[OUTCOME].agg(["size", "mean"])
    cr = cr[cr["size"] >= 15]
    print("\n>0.90 CRISIS-regime realized rate by year (the cells a regime-conditional fix relies on):")
    print(cr.round(3).to_string())
    if len(cr) >= 2:
        spread = cr["mean"].max() - cr["mean"].min()
        print(f"\ncross-crisis spread in crisis-bin realized rate: {spread*100:.1f}pp "
              f"-> {'STABLE (regime-conditional map should generalize)' if spread < 0.12 else 'UNSTABLE (even regime-conditioning may not generalize; favors a structural/POT-GPD fix)'}")

    print("\nI9 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
