"""Real-premium rail data-quality + wiring audit (Windows main terminal, 2026-06-27).

Complements the Mac terminal's rail-INDEPENDENT campaign (#436): this audits the
gitignored real-premium rail (data_processed/option_premium, PR #435) that only
this machine has. Five checks, all light (single rank calls + parquet reads):

  C1 COVERAGE   — across historical as_of dates x UNIVERSE_100, what fraction of
                  ranked candidates get premium_source=="market_mid" vs fall back
                  to synthetic_bsm? (How much does the rail actually engage?)
  C2 SPLIT-ADJ  — for each in-window Stock Split (from get_corporate_actions) on
                  known split names, rank ~3 weeks pre-split (so the ~35-DTE option
                  crosses the split) and assert the served strike scale matches the
                  engine's split-ADJUSTED spot (no 10x/20x raw-strike mismatch).
  C3 UNITS      — confirm edge_vs_fair == (premium - fair_value) * 100 on market_mid
                  rows (the per-contract-vs-per-share finding from the R11b work).
  C4 EDGE/VIX   — edge_vs_fair (per-contract) distribution by VIX regime; confirms
                  the skew premium R11b gates on is real and fattest in crisis.
  C5 QUOTE SANE — scan a sample of rail parquets for crossed/garbage quotes leaking
                  past the producer's validity gate (bid<=ask, mid>0, mid==(b+a)/2,
                  dte in belt, right in {put,call}).

REQUIRES the rail: set SWE_OPTION_PREMIUM_DIR to the produced parquet dir. Run from
the worktree. Outputs summary.json + prints a markdown summary.

Usage:
    SWE_OPTION_PREMIUM_DIR=<rail> python .../audit_rail.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtests.regression.universes import UNIVERSE_100  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

# Historical as_of dates spanning regimes (within the 504-day-gate feasible range).
COVERAGE_DATES = ["2020-03-16", "2021-06-15", "2022-10-14", "2023-06-15", "2024-06-14"]
# Names with notable splits in 2020-2024; the audit pulls the actual ratios/dates
# from get_corporate_actions rather than trusting hardcoded values.
SPLIT_NAMES = ["AAPL", "TSLA", "NVDA", "AMZN", "GOOGL"]


def _vix_bucket(v):
    if v is None or not np.isfinite(v):
        return "unknown"
    if v <= 15:
        return "vix<=15"
    if v <= 25:
        return "15-25"
    if v <= 35:
        return "25-35"
    return "vix>35"


def c1_coverage(runner, conn) -> dict:
    out = {}
    for d in COVERAGE_DATES:
        try:
            f = runner.rank_candidates_by_ev(
                tickers=UNIVERSE_100, as_of=d, top_n=300, min_ev_dollars=-1e9,
                include_diagnostic_fields=True,
            )
        except Exception as e:  # noqa: BLE001
            out[d] = {"error": str(e)}
            continue
        if f is None or len(f) == 0:
            out[d] = {"n": 0}
            continue
        vc = f["premium_source"].value_counts().to_dict() if "premium_source" in f else {}
        n = len(f)
        mm = int(vc.get("market_mid", 0))
        out[d] = {
            "n_candidates": n,
            "market_mid": mm,
            "synthetic_bsm": int(vc.get("synthetic_bsm", 0)),
            "market_mid_pct": round(100 * mm / n, 1) if n else 0.0,
            "n_tickers": int(f["ticker"].nunique()) if "ticker" in f else None,
        }
    return out


def c2_split_adjust(runner, conn) -> dict:
    out = {}
    for tkr in SPLIT_NAMES:
        try:
            ca = conn.get_corporate_actions(tkr)
        except Exception as e:  # noqa: BLE001
            out[tkr] = {"error": f"corp_actions: {e}"}
            continue
        if ca is None or len(ca) == 0:
            out[tkr] = {"splits_in_window": 0}
            continue
        # find split rows with an effective date in 2020-2024
        rows = []
        for _, r in ca.iterrows():
            txt = " ".join(str(v) for v in r.to_dict().values()).lower()
            if "split" not in txt:
                continue
            eff = None
            for v in r.to_dict().values():
                s = str(v)
                if s[:4].isdigit() and "2020" <= s[:4] <= "2024" and "-" in s:
                    eff = s[:10]
                    break
            if eff is None:
                continue
            ratio = None
            for k, v in r.to_dict().items():
                if "ratio" in str(k).lower():
                    try:
                        ratio = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append((eff, ratio))
        checks = []
        for eff, ratio in rows:
            eff_d = pd.Timestamp(eff)
            # ~60d before the split so a ~35-DTE option does NOT span it (else the
            # engine's correct corp-action event lockout fires and there is no
            # candidate to scale-check). Earnings lockouts may still apply.
            asof = (eff_d - pd.Timedelta(days=60)).date().isoformat()
            try:
                f = runner.rank_candidates_by_ev(
                    tickers=[tkr], as_of=asof, top_n=5, min_ev_dollars=-1e9,
                    include_diagnostic_fields=True,
                )
            except Exception as e:  # noqa: BLE001
                checks.append({"eff": eff, "ratio": ratio, "asof": asof, "error": str(e)})
                continue
            if f is None or len(f) == 0:
                drops = f.attrs.get("drops") if f is not None else None
                checks.append({"eff": eff, "ratio": ratio, "asof": asof, "n": 0, "drops": drops})
                continue
            row = f.iloc[0]
            spot = float(row.get("spot", 0.0))
            strike = float(row.get("strike", 0.0))
            prem = float(row.get("premium", 0.0))
            ratio_strike_spot = round(strike / spot, 3) if spot else None
            # A ~25-delta put strike should be ~0.7-1.0x spot; a split mismatch
            # makes it ~ratio x off. premium should be << spot (single/double digit).
            sane = (
                spot > 0 and strike > 0 and 0.4 <= (strike / spot) <= 1.2 and prem < spot
            )
            checks.append({
                "eff": eff, "ratio": ratio, "asof": asof,
                "spot": round(spot, 2), "strike": round(strike, 2),
                "premium": round(prem, 2), "premium_source": str(row.get("premium_source", "")),
                "strike_over_spot": ratio_strike_spot, "scale_sane": bool(sane),
            })
        out[tkr] = {"splits_in_window": len(rows), "checks": checks}
    return out


def c3_units(runner, conn) -> dict:
    f = runner.rank_candidates_by_ev(
        tickers=UNIVERSE_100, as_of="2020-03-16", top_n=300, min_ev_dollars=-1e9,
        include_diagnostic_fields=True,
    )
    if f is None or len(f) == 0 or "edge_vs_fair" not in f or "fair_value" not in f:
        return {"error": "no diagnostic frame"}
    mm = f[f.get("premium_source", "") == "market_mid"].copy()
    if mm.empty:
        return {"market_mid_rows": 0}
    implied = (mm["premium"] - mm["fair_value"]) * 100.0
    diff = (mm["edge_vs_fair"] - implied).abs()
    return {
        "market_mid_rows": int(len(mm)),
        "max_abs_diff_per_contract_vs_edge": round(float(diff.max()), 3),
        "median_abs_diff": round(float(diff.median()), 3),
        "interpretation": (
            "edge_vs_fair == (premium - fair_value)*100 (per-contract) confirmed"
            if float(diff.max()) < 1.0
            else "edge_vs_fair does NOT equal (premium-fair_value)*100 — investigate"
        ),
    }


def c4_edge_by_vix(runner, conn) -> dict:
    rows = []
    for d in COVERAGE_DATES:
        try:
            f = runner.rank_candidates_by_ev(
                tickers=UNIVERSE_100, as_of=d, top_n=300, min_ev_dollars=-1e9,
                include_diagnostic_fields=True,
            )
        except Exception:  # noqa: BLE001
            continue
        if f is None or len(f) == 0:
            continue
        try:
            v = conn.get_vix_regime(d).get("vix")
            v = float(v) if v is not None else None
        except Exception:  # noqa: BLE001
            v = None
        mm = f[f.get("premium_source", "") == "market_mid"]
        for e in mm.get("edge_vs_fair", pd.Series(dtype=float)).tolist():
            rows.append({"vix_bucket": _vix_bucket(v), "edge": float(e)})
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    out = {}
    for b, g in df.groupby("vix_bucket"):
        out[str(b)] = {
            "n": int(len(g)),
            "edge_mean_per_contract": round(float(g["edge"].mean()), 2),
            "edge_median": round(float(g["edge"].median()), 2),
            "pct_positive": round(100 * float((g["edge"] > 0).mean()), 1),
        }
    return out


def c5_quote_sanity(rail_dir: Path) -> dict:
    files = sorted(rail_dir.glob("*.parquet"))[:8]
    out = {"sampled_files": len(files), "per_file": {}}
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:  # noqa: BLE001
            out["per_file"][fp.stem] = {"error": str(e)}
            continue
        n = len(df)
        crossed = int((df["bid"] > df["ask"]).sum()) if {"bid", "ask"} <= set(df.columns) else None
        nonpos_mid = int((df["mid"] <= 0).sum()) if "mid" in df else None
        mid_off = (
            int((((df["bid"] + df["ask"]) / 2 - df["mid"]).abs() > 0.01).sum())
            if {"bid", "ask", "mid"} <= set(df.columns) else None
        )
        dte_oob = int((~df["dte"].between(0, 75)).sum()) if "dte" in df else None
        bad_right = (
            int((~df["right"].astype(str).str.lower().str[0].isin(["p", "c"])).sum())
            if "right" in df else None
        )
        out["per_file"][fp.stem] = {
            "rows": n, "crossed_bid_gt_ask": crossed, "nonpositive_mid": nonpos_mid,
            "mid_not_midpoint": mid_off, "dte_out_of_belt": dte_oob, "bad_right": bad_right,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rail = os.environ.get("SWE_OPTION_PREMIUM_DIR")
    if not rail or not list(Path(rail).glob("*.parquet")):
        raise SystemExit("REFUSING: SWE_OPTION_PREMIUM_DIR not set / no parquets — need the rail.")
    rail_dir = Path(rail)
    print(f"[rail_audit] rail: {rail} ({len(list(rail_dir.glob('*.parquet')))} parquets)", flush=True)

    runner = WheelRunner()
    conn = runner.connector
    print(f"[rail_audit] connector: {type(conn).__name__}", flush=True)

    res = {"rail_dir": rail}
    print("[rail_audit] C1 coverage...", flush=True)
    res["c1_coverage"] = c1_coverage(runner, conn)
    print("[rail_audit] C2 split-adjust...", flush=True)
    res["c2_split_adjust"] = c2_split_adjust(runner, conn)
    print("[rail_audit] C3 units...", flush=True)
    res["c3_units"] = c3_units(runner, conn)
    print("[rail_audit] C4 edge-by-vix...", flush=True)
    res["c4_edge_by_vix"] = c4_edge_by_vix(runner, conn)
    print("[rail_audit] C5 quote-sanity...", flush=True)
    res["c5_quote_sanity"] = c5_quote_sanity(rail_dir)

    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    # Markdown summary
    print("\n" + "=" * 70)
    print("REAL-PREMIUM RAIL AUDIT")
    print("=" * 70)
    print("\n## C1 Coverage (market_mid % of ranked candidates)")
    for d, m in res["c1_coverage"].items():
        if "error" in m:
            print(f"- {d}: ERROR {m['error']}")
        else:
            print(f"- {d}: {m.get('market_mid_pct')}% market_mid "
                  f"({m.get('market_mid')}/{m.get('n_candidates')}, {m.get('n_tickers')} tickers)")
    print("\n## C2 Split-adjust scale sanity (pre-split rank, strike/spot must be ~0.4-1.2)")
    for t, m in res["c2_split_adjust"].items():
        if "error" in m:
            print(f"- {t}: ERROR {m['error']}")
            continue
        for c in m.get("checks", []):
            if "error" in c or c.get("n") == 0:
                print(f"- {t} {c.get('eff')} (ratio {c.get('ratio')}): no candidate/{c.get('error','n=0')}")
            else:
                flag = "OK" if c["scale_sane"] else "*** MISMATCH ***"
                print(f"- {t} {c['eff']} (ratio {c['ratio']}): spot {c['spot']} strike {c['strike']} "
                      f"(s/spot {c['strike_over_spot']}) prem {c['premium']} [{c['premium_source']}] {flag}")
    print("\n## C3 Units")
    print(f"- {res['c3_units']}")
    print("\n## C4 edge_vs_fair (per-contract) by VIX bucket")
    for b, m in sorted(res["c4_edge_by_vix"].items()):
        print(f"- {b}: n={m['n']} mean=${m['edge_mean_per_contract']} median=${m['edge_median']} "
              f"pct_positive={m['pct_positive']}%")
    print("\n## C5 Quote sanity (sample parquets; all counts should be 0)")
    for t, m in res["c5_quote_sanity"]["per_file"].items():
        print(f"- {t}: {m}")
    print(f"\nwritten: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
