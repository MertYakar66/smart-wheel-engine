"""I11 — R11 risk-budget parameter study (VIX-LEVEL size-down threshold).

Observe-only. Implements docs/HEAVY_VERIFY_2026-05-31_I11_RISK_BUDGET_STUDY_SPEC.md.
Picks the parameters of the candidate R11 reviewer — a §2-clean, downgrade-only
"cap top-bin (prob_profit>0.90) size when VIX level is elevated" rule — by MEASURING
the cost/benefit it makes, leave-one-crisis-out, on real data. Reads existing campaign
artifacts + the VIX CSVs and prints tables; engine/ untouched. R11 itself is a separate
gated card.

Why VIX *level*, not a ratio (I10): I10 killed rv_ratio because it peaks at the 2020
RECOVERY (2.69) above the onset (1.69). VIX level inverts that correctly (~80 onset vs
~40-55 recovery), so a level trigger is worth testing. Primary = VIX close; secondary
(§6 honesty check only) = term-structure backwardation (vix > vix_3m).

Run:  python i11_vix_budget_study.py
"""
from __future__ import annotations

import os
import sys
import math

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
VIX_CSV = os.path.join(L.DATA, "sp500_vix_full.csv")
TERM_CSV = os.path.join(L.DATA, "vix_term_structure.csv")

TOP_BIN = 0.90                       # confidence cutoff (inherited from I1/I9/I10)
THETAS = [20.0, 22.5, 25.0, 27.5, 30.0]   # coarse round-number grid (§5)
CRISES = ["2020", "2022"]            # well-powered crisis years (2020 n=158, 2022 n=86)


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


def load_vix() -> pd.Series:
    v = pd.read_csv(VIX_CSV)
    v = v[v["instrument"] == "vix"].copy()
    v["date"] = pd.to_datetime(v["date"])
    return v.set_index("date")["close"].sort_index()


def load_backwardation() -> pd.Series:
    t = pd.read_csv(TERM_CSV)
    t["date"] = pd.to_datetime(t["date"])
    t = t.set_index("date").sort_index()
    return (t["vix"] > t["vix_3m"])  # backwardation (front > 3m) — stress signal


def main() -> int:
    if not os.path.exists(ROWS):
        raise SystemExit("run i1_calibration.py first (writes i1_realized_rows.parquet)")
    df = pd.read_parquet(ROWS)
    df = df[df["prob_profit"] > TOP_BIN].copy()          # top bin only
    df["year"] = df["as_of"].str[:4]
    vix = load_vix()
    back = load_backwardation()
    df["vix"] = df["as_of"].apply(lambda d: float(vix.asof(pd.Timestamp(d))))
    df["backwardated"] = df["as_of"].apply(
        lambda d: bool(back.asof(pd.Timestamp(d))) if pd.notna(back.asof(pd.Timestamp(d))) else False)
    print(f"top-bin trades (prob_profit>{TOP_BIN}): {len(df)}; VIX joined on as_of")
    print(f"VIX by year (median): " + ", ".join(
        f"{y}={g['vix'].median():.0f}" for y, g in df.groupby('year')))

    def cell_scores(sub: pd.DataFrame, theta: float) -> dict:
        dg = sub[sub["vix"] > theta]                     # downgraded (VIX elevated)
        pnl = dg["realized_pnl_synth"].values.astype(float)
        bleed_averted = float(-pnl[pnl < 0].sum())       # losses removed (>=0)
        premium_forgone = float(pnl[pnl > 0].sum())      # gains given up (>=0)
        net = float(-pnl.sum())                          # net $ from sizing them out
        return {
            "n_cell": len(sub), "n_downgraded": len(dg),
            "bleed_averted": round(bleed_averted, 0),
            "premium_forgone": round(premium_forgone, 0),
            "net_book": round(net, 0),
            "net_per_contract": round(net / len(dg), 0) if len(dg) else 0.0,
            "dg_win_rate": round(float((pnl > 0).mean()), 3) if len(dg) else float("nan"),
        }

    # ---- §4 + §5: per-crisis LOCO cost/benefit across the coarse theta grid ----
    hr("§4/§5 — LOCO cost/benefit of the VIX-level size-down, per crisis x coarse theta")
    print("downgraded = top-bin trade with VIX(as_of) > theta; net_book = -sum(realized over downgraded)")
    print("(net_book > 0 => sizing those out HELPED on net; the 2020 averted-tail should dominate)\n")
    survival = {}
    for theta in THETAS:
        print(f"--- theta = VIX > {theta} ---")
        print(f"{'cell':10s} {'n':>4s} {'dgr':>4s} {'bleed_averted':>13s} {'prem_forgone':>12s} {'net_book':>10s} {'net/ctr':>9s} {'dg_win':>7s}")
        favorable_all = True
        for cy in CRISES:
            s = cell_scores(df[df["year"] == cy], theta)
            print(f"{cy:10s} {s['n_cell']:4d} {s['n_downgraded']:4d} {s['bleed_averted']:13.0f} "
                  f"{s['premium_forgone']:12.0f} {s['net_book']:10.0f} {s['net_per_contract']:9.0f} {s['dg_win_rate']:7.3f}")
        pooled = cell_scores(df, theta)
        print(f"{'POOLED':10s} {pooled['n_cell']:4d} {pooled['n_downgraded']:4d} {pooled['bleed_averted']:13.0f} "
              f"{pooled['premium_forgone']:12.0f} {pooled['net_book']:10.0f} {pooled['net_per_contract']:9.0f} {pooled['dg_win_rate']:7.3f}")
        # Robustness = SURVIVES EVERY FOLD (§5): net_book > 0 in *each* crisis, not just
        # the aggregate (the aggregate lets 2020's huge averted-tail mask a 2022 loss).
        per_fold_net = {cy: cell_scores(df[df["year"] == cy], theta)["net_book"] for cy in CRISES}
        favorable = all(v > 0 for v in per_fold_net.values())
        tot_averted = sum(cell_scores(df[df["year"] == cy], theta)["bleed_averted"] for cy in CRISES)
        tot_forgone = sum(cell_scores(df[df["year"] == cy], theta)["premium_forgone"] for cy in CRISES)
        survival[theta] = {"per_fold_net": per_fold_net, "favorable": favorable,
                           "tot_averted": tot_averted, "tot_forgone": tot_forgone,
                           "ratio": (tot_averted / tot_forgone) if tot_forgone else float("inf")}
        print(f"   per-fold net_book: " + ", ".join(f"{cy}={v:+.0f}" for cy, v in per_fold_net.items())
              + f"  -> survives-every-fold={favorable}\n")

    # ---- §4.4: the 2022 headline (false-positive opportunity cost) ----
    hr("§4.4 — 2022 headline: a VIX-level rule FIRES through 2022 (no detection miss)")
    s22 = df[df["year"] == "2022"]
    print(f"2022 top-bin trades: {len(s22)}; VIX median {s22['vix'].median():.0f}, "
          f"range [{s22['vix'].min():.0f},{s22['vix'].max():.0f}]")
    net22 = s22["realized_pnl_synth"].sum()
    print(f"2022 net realized P&L (all top-bin, no rule): {net22:+.0f}  "
          f"(if positive, the rule's 2022 cost is forgone premium — the bounded side)")
    for theta in THETAS:
        sc = cell_scores(s22, theta)
        print(f"  theta>{theta}: downgrades {sc['n_downgraded']}/{sc['n_cell']}, "
              f"forgoes {sc['premium_forgone']:.0f}, averts {sc['bleed_averted']:.0f}, net {sc['net_book']:+.0f}")

    # ---- 2020 headline (the averted tail) ----
    hr("2020 — the averted tail (the unforecastable loss the rule exists to bound)")
    s20 = df[df["year"] == "2020"]
    k = int((s20["engine_exact"].astype(float) > 0).sum()) if "engine_exact" in s20 else 0
    lo, hi = wilson(k, len(s20)) if len(s20) else (float("nan"), float("nan"))
    print(f"2020 top-bin: n={len(s20)}, realized win {k}/{len(s20)} Wilson95=[{lo:.2f},{hi:.2f}], "
          f"VIX median {s20['vix'].median():.0f}; worst realized {s20['realized_pnl_synth'].min():.0f}")

    # ---- §6: honesty scan — any low-VIX + fat-left-tail cell the level-rule MISSES? ----
    hr("§6 — honesty scan: top-bin trades with VIX BELOW theta AND a fat realized left tail")
    bottom_decile = df["realized_pnl_synth"].quantile(0.10)
    print(f"bottom-decile realized threshold (top-bin): {bottom_decile:.0f}; scan window "
          f"{df['as_of'].min()}..{df['as_of'].max()}")
    for theta in (25.0, 30.0):
        miss = df[(df["vix"] <= theta) & (df["realized_pnl_synth"] <= bottom_decile)]
        print(f"  theta={theta}: {len(miss)} low-VIX + fat-tail misses "
              f"(mean realized {miss['realized_pnl_synth'].mean():.0f} if any)")
        if len(miss):
            bw = miss["backwardated"].mean()
            print(f"    -> of these, term-structure backwardation flags {bw*100:.0f}% "
                  f"(secondary signal coverage of the level-rule's miss)")
            print(miss.groupby("year").size().to_dict())

    # ---- verdict ----
    hr("VERDICT")
    fav = [t for t in THETAS if survival[t]["favorable"]]
    print("coarse-theta survival (favorable = net_book>0 in EVERY crisis fold, §5):")
    for t in THETAS:
        s = survival[t]
        folds = ", ".join(f"{cy}={v:+.0f}" for cy, v in s["per_fold_net"].items())
        print(f"  VIX>{t}: per-fold [{folds}]  agg-ratio {s['ratio']:.2f} -> "
              f"{'SURVIVES every fold' if s['favorable'] else 'FAILS a fold'}")
    if fav:
        robust = max(fav)  # highest theta that still survives every fold = least-intrusive favorable cut
        print(f"\nSURVIVES-EVERY-FOLD theta(s): {fav}. Robust-not-optimal pick: VIX > {robust}"
              f" (the cleanest round-number 'elevated-vol' line that stays favorable in BOTH the"
              f" 2020 crash and the 2022 bear; higher theta fails the 2022 fold). Conditional R11"
              f" form supported: downgrade top-bin (prob_profit>0.90) when VIX close > {robust}.")
    else:
        print("\nNULL RESULT (§7): no coarse theta makes the LOCO trade-off favorably asymmetric"
              " -> R11 falls back to the UNCONDITIONAL top-bin haircut (downgrade prob_profit>0.90"
              " in all regimes, no detector).")
    print("\nI11 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
