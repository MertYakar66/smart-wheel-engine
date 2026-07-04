"""Skew-edge as a SELECTION signal — does a richer real skew-edge predict better
realized returns? (Windows main, 2026-06-28; forward research on the validated
#435 real-premium rail.)

The rail audit (C4) showed edge_vs_fair is positive across every VIX regime — the
VRP/skew premium is real. The R11b A/B showed gating high-edge crisis trades was
net-COSTLY (they were winners). This asks the natural next question: is the
skew-edge a usable SELECTION/sizing lever, i.e. do HIGHER-edge market_mid
candidates realize BETTER put-leg returns, or is the edge just risk-compensation
(flat/inverse)?

Design (cross-sectional predictive, NOT a tracker sim → lighter):
  - Weekly (every 5th business day) over 2020-2024, rank 100t (rail ON).
  - Keep market_mid candidates; record edge_vs_fair, premium, fair_value, iv,
    strike, spot, dte, prob_profit, vix.
  - Forward-replay the short-put realized P&L held to expiry.
  - Sort by (a) raw edge_vs_fair and (b) edge RICHNESS = edge/(premium*100) =
    (premium-fair)/premium, into quintiles; report mean realized return
    (normalized by collateral = strike*100) per quintile + Spearman(edge, ret).
  - Stratify by VIX regime. Verdict: monotone-up = lever; flat = neutral;
    down = trap.

REQUIRES the rail (SWE_OPTION_PREMIUM_DIR). Read-only; no engine/data edits.
Usage: SWE_OPTION_PREMIUM_DIR=<rail> python .../skew_edge_selection.py --out-dir <d>
       python .../skew_edge_selection.py --analyze --out-dir <d>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
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

from backtests.regression._common import (  # noqa: E402
    _forward_replay_realized_pnl,
    _spot_on_or_after,
    assert_data_window_available,
)
from backtests.regression.universes import UNIVERSE_100  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402


def vix_regime(v):
    if v is None or not np.isfinite(v):
        return "unknown"
    if v <= 15:
        return "calm"
    if v <= 25:
        return "elevated"
    return "crisis"


def collect(out_dir: Path, start: str, end: str, dte_target: int, step_days: int) -> pd.DataFrame:
    assert_data_window_available(start, end)
    runner = WheelRunner()
    conn = runner.connector
    print(f"[skew_sel] connector={type(conn).__name__} window={start}->{end} step={step_days}bd", flush=True)
    all_days = [d.date() for d in pd.bdate_range(start, end)]
    days = all_days[::step_days]
    rows = []
    spot_cache: dict[tuple, float | None] = {}
    vix_cache: dict[str, float | None] = {}
    t0 = time.time()
    for i, d in enumerate(days):
        if i and i % 20 == 0:
            el = time.time() - t0
            print(f"[skew_sel] {i}/{len(days)} ({100*i/len(days):.0f}%) {el/60:.1f}min rows={len(rows)}", flush=True)
        asof = d.isoformat()
        if asof not in vix_cache:
            try:
                v = conn.get_vix_regime(asof).get("vix")
                vix_cache[asof] = float(v) if v is not None and np.isfinite(v) else None
            except Exception:
                vix_cache[asof] = None
        vix = vix_cache[asof]
        try:
            f = runner.rank_candidates_by_ev(
                tickers=UNIVERSE_100, as_of=asof, top_n=400, min_ev_dollars=-1e9,
                include_diagnostic_fields=True,
            )
        except Exception:
            continue
        if f is None or len(f) == 0 or "premium_source" not in f:
            continue
        mm = f[f["premium_source"] == "market_mid"]
        for _, r in mm.iterrows():
            strike = float(r.get("strike", 0) or 0)
            prem = float(r.get("premium", 0) or 0)
            if strike <= 0 or prem <= 0:
                continue
            exp = d + timedelta(days=dte_target)
            key = (str(r.get("ticker", "")), exp.isoformat())
            if key not in spot_cache:
                spot_cache[key] = _spot_on_or_after(conn, str(r.get("ticker", "")), exp)
            sx = spot_cache[key]
            if sx is None:
                continue
            realized = _forward_replay_realized_pnl(strike, prem, sx)  # per-contract $
            rows.append({
                "date": asof, "ticker": str(r.get("ticker", "")), "vix": vix,
                "regime": vix_regime(vix), "strike": strike, "premium": prem,
                "fair_value": float(r.get("fair_value", 0) or 0),
                "edge_vs_fair": float(r.get("edge_vs_fair", 0) or 0),
                "iv": float(r.get("iv", 0) or 0), "prob_profit": float(r.get("prob_profit", 0) or 0),
                "realized_pnl": realized, "collateral": strike * 100,
                "ret_on_collateral": realized / (strike * 100),
            })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "candidates.csv", index=False)
    print(f"[skew_sel] collected {len(df)} market_mid candidates -> candidates.csv", flush=True)
    return df


def _quintile_table(df: pd.DataFrame, by: str) -> list[dict]:
    d = df.dropna(subset=[by, "ret_on_collateral"]).copy()
    if len(d) < 25:
        return []
    try:
        d["q"] = pd.qcut(d[by], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except ValueError:
        return []
    out = []
    for q, g in d.groupby("q", observed=True):
        out.append({
            "quintile": int(q), "n": int(len(g)),
            f"{by}_mean": round(float(g[by].mean()), 4),
            "ret_on_collateral_mean_pct": round(100 * float(g["ret_on_collateral"].mean()), 3),
            "realized_pnl_mean": round(float(g["realized_pnl"].mean()), 2),
            "win_rate": round(float((g["realized_pnl"] > 0).mean()), 3),
        })
    return out


def _spearman(df: pd.DataFrame, a: str, b: str) -> float:
    d = df.dropna(subset=[a, b])
    if len(d) < 25:
        return float("nan")
    return float(d[a].rank().corr(d[b].rank()))


def analyze(out_dir: Path) -> dict:
    df = pd.read_csv(out_dir / "candidates.csv")
    df["edge_richness"] = df["edge_vs_fair"] / (df["premium"] * 100).replace(0, np.nan)
    res = {"n_total": int(len(df)), "windows": "weekly 2020-2024 100t market_mid"}
    res["overall"] = {
        "ret_on_collateral_mean_pct": round(100 * float(df["ret_on_collateral"].mean()), 3),
        "win_rate": round(float((df["realized_pnl"] > 0).mean()), 3),
        "spearman_edge_vs_ret": round(_spearman(df, "edge_vs_fair", "ret_on_collateral"), 4),
        "spearman_richness_vs_ret": round(_spearman(df, "edge_richness", "ret_on_collateral"), 4),
    }
    res["by_raw_edge_quintile"] = _quintile_table(df, "edge_vs_fair")
    res["by_richness_quintile"] = _quintile_table(df, "edge_richness")
    res["by_regime"] = {}
    for reg, g in df.groupby("regime"):
        res["by_regime"][str(reg)] = {
            "n": int(len(g)),
            "ret_mean_pct": round(100 * float(g["ret_on_collateral"].mean()), 3),
            "spearman_richness_vs_ret": round(_spearman(g, "edge_richness", "ret_on_collateral"), 4),
            "richness_quintiles": _quintile_table(g, "edge_richness"),
        }
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SKEW-EDGE AS A SELECTION SIGNAL")
    print("=" * 72)
    o = res["overall"]
    print(f"\nn={res['n_total']} market_mid candidates (weekly 2020-2024, 100t)")
    print(f"overall: ret/collateral {o['ret_on_collateral_mean_pct']}%/trade, win {o['win_rate']}, "
          f"Spearman(edge,ret)={o['spearman_edge_vs_ret']}, Spearman(richness,ret)={o['spearman_richness_vs_ret']}")
    print("\n## By edge RICHNESS quintile ((premium-fair)/premium) — Q1 lean .. Q5 rich")
    print("| Q | n | richness | ret/collat % | realized$ | win |")
    print("|---|---|---|---|---|---|")
    for r in res["by_richness_quintile"]:
        print(f"| {r['quintile']} | {r['n']} | {r['edge_richness_mean']} | "
              f"{r['ret_on_collateral_mean_pct']} | {r['realized_pnl_mean']} | {r['win_rate']} |")
    print("\n## By raw edge_vs_fair quintile")
    print("| Q | n | edge$ | ret/collat % | realized$ | win |")
    print("|---|---|---|---|---|---|")
    for r in res["by_raw_edge_quintile"]:
        print(f"| {r['quintile']} | {r['n']} | {r['edge_vs_fair_mean']} | "
              f"{r['ret_on_collateral_mean_pct']} | {r['realized_pnl_mean']} | {r['win_rate']} |")
    print("\n## By VIX regime (richness Q1->Q5 ret/collat %)")
    for reg, m in res["by_regime"].items():
        qs = " -> ".join(str(q["ret_on_collateral_mean_pct"]) for q in m["richness_quintiles"])
        print(f"  {reg}: n={m['n']} ret {m['ret_mean_pct']}% rho={m['spearman_richness_vs_ret']} | Q: {qs}")
    print("\nVERDICT GUIDE: richness quintile ret monotone-UP => skew-edge is a usable selection lever; "
          "flat => neutral (risk-comp); DOWN => trap. Spearman sign/size = strength.")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--dte-target", type=int, default=35)
    ap.add_argument("--step-days", type=int, default=5)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.analyze:
        analyze(out_dir)
        return 0
    import os
    if not os.environ.get("SWE_OPTION_PREMIUM_DIR"):
        raise SystemExit("REFUSING: SWE_OPTION_PREMIUM_DIR unset — need the real-premium rail.")
    collect(out_dir, args.start, args.end, args.dte_target, args.step_days)
    analyze(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
