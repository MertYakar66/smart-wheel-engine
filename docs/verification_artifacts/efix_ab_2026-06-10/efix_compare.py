"""A/B comparator for the (E)-fix heavy-test validation. Pure stdlib.

(1) STRICT determinism diff: baseline (origin/main) vs integration (1985547 +
    #382/#384/#386) raw payloads, exact + NaN-aware on every metric leaf. The
    #386 fix is on the EVEngine.evaluate path; determinism-null <=> zero diffs.
(2) DRIFT check: each arm vs the committed snapshot, using the regression
    test's per-metric tolerance bands (answers "is main still green?").

Run: py -3.12 C:/Users/merty/Desktop/swe-main/_efix_compare.py
"""

import json
import math
import os

BASE_DIR = r"C:/Users/merty/Desktop/swe-main/.efix-validation"
INTEG_DIR = r"C:/Users/merty/Desktop/swe-efixes/.efix-validation"
SNAP_DIR = r"C:/Users/merty/Desktop/swe-main/backtests/regression/snapshots"

IDS = ["s27_ivpit_24t_100k", "s32_friction_24t_1m", "s34_universe_100t_1m", "s35_oos_24t_100k"]
METRIC_SECTIONS = ("aggregate", "per_year", "per_quartile", "per_friction_level")

# Mirror of tests/test_backtest_regression._TOLERANCES
TOL = {
    "row_count": ("exact", 0), "executed_trades": ("exact", 0), "put_assignments": ("exact", 0),
    "open_at_end": ("exact", 0), "n": ("exact", 0),
    "spearman_rho": ("abs", 0.005), "rho": ("abs", 0.005), "hit_rate": ("abs", 0.005), "hit": ("abs", 0.005),
    "spearman_p": ("rel", 0.01), "p": ("rel", 0.01),
    "final_nav": ("rel", 1e-5), "final_cash": ("rel", 1e-5),
    "mean_realized": ("rel", 1e-4), "ev_mean": ("rel", 1e-4), "pnl_mean": ("rel", 1e-4),
    "iv_mean": ("abs", 1e-4),
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def leaves(d, path=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(leaves(v, f"{path}/{k}" if path else k))
    else:
        out[path] = d
    return out


def metric_leaves(payload):
    out = {}
    for sec in METRIC_SECTIONS:
        if payload and sec in payload:
            out.update(leaves(payload[sec], sec))
    return out


def eq_exact(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def within_tol(exp, act, key):
    spec = TOL.get(key)
    if spec is None:
        return True, "skip"
    if isinstance(exp, float) and math.isnan(exp):
        return (isinstance(act, float) and math.isnan(act)), "nan"
    kind, tol = spec
    try:
        if kind == "exact":
            return exp == act, f"exp={exp} act={act}"
        if kind == "abs":
            return abs(act - exp) <= tol + 1e-12, f"|d|={abs(act - exp):.4g} tol={tol}"
        if kind == "rel":
            r = abs(act - exp) / abs(exp) if exp else (0.0 if act == exp else float("inf"))
            return r <= tol + 1e-12, f"rel={r:.4g} tol={tol}"
    except TypeError:
        return False, f"type exp={exp!r} act={act!r}"
    return False, "?"


print("=" * 78)
print(" (E)-FIX A/B VALIDATION  —  baseline(origin/main 1985547)  vs  integration(+382/384/386)")
print("=" * 78)

ab_overall_ok = True
for sid in IDS:
    main_p = load(os.path.join(BASE_DIR, f"main_{sid}.json"))
    integ_p = load(os.path.join(INTEG_DIR, f"integ_{sid}.json"))
    snap = load(os.path.join(SNAP_DIR, f"{sid}.json"))
    print(f"\n#### {sid}")
    if main_p is None or integ_p is None:
        print(f"   [SKIP A/B] missing payload  main={main_p is not None} integ={integ_p is not None}")
    else:
        ml, il = metric_leaves(main_p), metric_leaves(integ_p)
        keys = sorted(set(ml) | set(il))
        diffs = []
        for k in keys:
            if k not in ml or k not in il:
                diffs.append(f"   KEYSET {k}: main={ml.get(k,'<missing>')} integ={il.get(k,'<missing>')}")
            elif not eq_exact(ml[k], il[k]):
                diffs.append(f"   DIFF {k}: main={ml[k]!r} integ={il[k]!r}")
        # fingerprint data sha cross-check
        msha = (main_p.get("fingerprint") or {}).get("data_csv_sha256")
        isha = (integ_p.get("fingerprint") or {}).get("data_csv_sha256")
        if diffs:
            ab_overall_ok = False
            print(f"   [A/B  FAIL] {len(diffs)} differing leaf(s) of {len(keys)} — #386 is NOT output-identical:")
            for d in diffs[:40]:
                print(d)
        else:
            print(f"   [A/B  PASS] all {len(keys)} metric leaves byte-identical (determinism-null confirmed)")
        print(f"   data_csv_sha256 match: {msha == isha}  ({str(msha)[:12]}… vs {str(isha)[:12]}…)")

    # Drift check each arm vs committed snapshot
    if snap is not None:
        sl = metric_leaves(snap)
        for label, arm in (("main", main_p), ("integ", integ_p)):
            if arm is None:
                continue
            al = metric_leaves(arm)
            fails = []
            for k, exp in sl.items():
                key = k.split("/")[-1]
                if key not in TOL:
                    continue
                if k not in al:
                    fails.append(f"      {k}: MISSING in {label}")
                    continue
                ok, detail = within_tol(exp, al[k], key)
                if not ok:
                    fails.append(f"      {k}: {detail}")
            verdict = "GREEN" if not fails else f"DRIFT ({len(fails)})"
            print(f"   [snapshot vs {label}] {verdict}")
            for fl in fails[:20]:
                print(fl)
    else:
        print("   [snapshot] none committed")

print("\n" + "=" * 78)
print(f" A/B DETERMINISM OVERALL: {'PASS — all backtests output-identical' if ab_overall_ok else 'FAIL — see diffs above'}")
print("=" * 78)
