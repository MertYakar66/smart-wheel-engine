"""Post-processor for the MAG7 reliability campaign.

Reads the per-window artifacts written by
``backtests.regression.mag7_reliability`` (``%TEMP%/mag7_backtest/<wid>/{none,
bid_ask,full}/{metrics.json,tracker_state.json,rank_log.csv}``) and produces a
verified analysis: returns, drawdown, funnel, EV / prob_profit calibration,
single-name concentration (the load-bearing Mag7 honesty metric, with a
self-consistency check against the tracker's own ``num_positions``), and an
equal-weight Mag7 buy-and-hold benchmark per window.

Every number here is derived from the on-disk artifacts or the connector —
nothing is asserted that the data doesn't support. Run::

    py -3.12 scripts/mag7_analysis.py --all
    py -3.12 scripts/mag7_analysis.py --window w4_2022_bear
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.wheel_runner import WheelRunner  # noqa: E402

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
CAP = 200_000.0
ROOT = Path(os.environ.get("TEMP", "/tmp")) / "mag7_backtest"
WINDOW_IDS = [
    "w1_2020_crash_entry",
    "w2_2020_recovery",
    "w3_2021_calm_bull",
    "w4_2022_bear",
    "w5_2022_bottom_entry",
    "w6_2023_chop",
    "w7_2024_late_cycle",
    "w8_2025_recent",
]


def _max_drawdown(equity_curve: list[dict]) -> float:
    pv = [float(e["portfolio_value"]) for e in equity_curve]
    peak, mdd = -1e18, 0.0
    for v in pv:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def _prob_profit_reliability(rank: pd.DataFrame) -> list[dict]:
    """Engine prob_profit vs realized win-rate (win = hypothetical realized_pnl>0),
    over ALL ranked candidate-rows (the engine's probability calibration)."""
    df = rank.dropna(subset=["prob_profit", "realized_pnl"]).copy()
    if df.empty:
        return []
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
    labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95+"]
    df["bin"] = pd.cut(df["prob_profit"], bins=edges, labels=labels, right=False)
    out = []
    for lab, g in df.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        out.append(
            {
                "bin": str(lab),
                "n": int(len(g)),
                "pred": round(float(g["prob_profit"].mean()), 4),
                "realized": round(float((g["realized_pnl"] > 0).mean()), 4),
                "gap_pp": round(
                    100 * (float((g["realized_pnl"] > 0).mean()) - float(g["prob_profit"].mean())),
                    1,
                ),
            }
        )
    return out


def _ev_quintile(rank: pd.DataFrame) -> list[dict]:
    df = rank.dropna(subset=["ev_dollars", "realized_pnl"]).copy()
    if len(df) < 10:
        return []
    df["q"] = pd.qcut(df["ev_dollars"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    out = []
    for q, g in df.groupby("q", observed=True):
        out.append(
            {
                "quintile": int(q),
                "n": int(len(g)),
                "ev_pred_mean": round(float(g["ev_dollars"].mean()), 2),
                "realized_mean": round(float(g["realized_pnl"].mean()), 2),
            }
        )
    return out


def _concentration(tracker: dict) -> dict:
    """Reconstruct the daily book (collateral per ticker) from open + closed
    positions and report peak single-name exposure as % of NAV. Validates the
    reconstruction against the tracker's own per-day ``num_positions``."""
    ec = tracker["equity_curve"]
    if not ec:
        return {}
    days = pd.to_datetime([e["date"] for e in ec])
    navs = {pd.Timestamp(e["date"]): float(e["portfolio_value"]) for e in ec}
    tracker_npos = {pd.Timestamp(e["date"]): int(e.get("num_positions", 0)) for e in ec}

    # Build (ticker, strike, entry, exit) for every position the book ever held.
    legs: list[tuple[str, float, pd.Timestamp, pd.Timestamp]] = []

    def _strike_of(p: dict) -> float | None:
        if p.get("put_strike") is not None:
            return float(p["put_strike"])
        # closed-position fallback: strike lives in notes as e.g. "Sold 35d 6.0P for ..."
        notes = p.get("notes")
        text = " ".join(notes) if isinstance(notes, list) else (notes or "")
        import re

        m = re.search(r"(\d+(?:\.\d+)?)\s*P\b", str(text))  # "<strike>P"
        if m:
            return float(m.group(1))
        m = re.search(r"(?:put[_ ]?strike|strike)[^0-9]*([0-9]+(?:\.[0-9]+)?)", str(text), re.I)
        return float(m.group(1)) if m else None

    def _date(p: dict, *keys) -> pd.Timestamp | None:
        for k in keys:
            v = p.get(k)
            if v:
                return pd.Timestamp(str(v)[:10])
        return None

    open_pos = tracker.get("positions", {})
    open_iter = open_pos.values() if isinstance(open_pos, dict) else open_pos
    last_day = days.max()
    for p in open_iter:
        sk = _strike_of(p)
        ent = _date(p, "put_entry_date", "entry_date")
        if sk and ent:
            legs.append((p.get("ticker", "?"), sk, ent, last_day))
    closed = tracker.get("closed_positions", [])
    parsed_closed = 0
    for p in closed:
        sk = _strike_of(p)
        ent = _date(p, "put_entry_date", "entry_date")
        ext = (
            _date(p, "exit_date", "close_date", "put_expiration_date", "expiration_date")
            or last_day
        )
        if sk and ent:
            legs.append((p.get("ticker", "?"), sk, ent, ext))
            parsed_closed += 1

    # Daily exposure reconstruction.
    peak_single = 0.0
    peak_single_name = None
    peak_single_date = None
    recon_count_match = 0
    recon_count_total = 0
    for d in days:
        held = [(tk, sk) for (tk, sk, e, x) in legs if e <= d <= x]
        nav = navs.get(d, CAP) or CAP
        by_name: dict[str, float] = {}
        for tk, sk in held:
            by_name[tk] = by_name.get(tk, 0.0) + sk * 100.0
        if by_name:
            mx_name = max(by_name, key=by_name.get)
            mx_pct = by_name[mx_name] / nav
            if mx_pct > peak_single:
                peak_single, peak_single_name, peak_single_date = mx_pct, mx_name, str(d)[:10]
        # consistency vs tracker's num_positions
        recon_count_total += 1
        if len(held) == tracker_npos.get(d, -1):
            recon_count_match += 1

    return {
        "peak_single_name_pct": round(100 * peak_single, 1),
        "peak_single_name": peak_single_name,
        "peak_single_date": peak_single_date,
        "peak_num_positions": max(tracker_npos.values()) if tracker_npos else 0,
        "mean_num_positions": round(float(np.mean(list(tracker_npos.values()))), 2)
        if tracker_npos
        else 0,
        "closed_parsed": parsed_closed,
        "closed_total": len(closed),
        "recon_vs_tracker_count_match_pct": round(
            100 * recon_count_match / max(recon_count_total, 1), 1
        ),
    }


def _ew_benchmark(conn, start: str, end: str) -> float | None:
    rets = []
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    for t in MAG7:
        d = conn.get_ohlcv(t)
        cl = d["close"].astype(float)
        cl = cl[(cl.index >= s0) & (cl.index <= s1)]
        if len(cl) > 1:
            rets.append(cl.iloc[-1] / cl.iloc[0] - 1.0)
    return round(100 * float(np.mean(rets)), 2) if rets else None


def analyze_window(wid: str, conn) -> dict:
    d = ROOT / wid
    if not (d / "full" / "metrics.json").exists():
        return {"wid": wid, "status": "MISSING"}
    res: dict = {"wid": wid, "status": "ok"}
    summ = (
        json.load(open(d / "summary.json", encoding="utf-8"))
        if (d / "summary.json").exists()
        else {}
    )
    res["start"], res["end"], res["note"] = summ.get("start"), summ.get("end"), summ.get("note")
    per_fr = {}
    for fr in ("none", "bid_ask", "full"):
        mp = d / fr / "metrics.json"
        if mp.exists():
            agg = json.load(open(mp, encoding="utf-8"))["aggregate"]
            per_fr[fr] = {
                "ret_pct": round(100 * (agg["final_nav"] / CAP - 1), 2),
                "final_nav": round(agg["final_nav"], 0),
                "executed_trades": agg.get("executed_trades"),
                "put_assignments": agg.get("put_assignments"),
                "hit_rate": round(agg.get("hit_rate"), 3)
                if agg.get("hit_rate") is not None
                else None,
                "spearman_rho": round(agg.get("spearman_rho", float("nan")), 3)
                if agg.get("spearman_rho") is not None
                else None,
                "spearman_p": agg.get("spearman_p"),
                "open_at_end": agg.get("open_at_end"),
            }
    res["friction"] = per_fr
    tracker = json.load(open(d / "full" / "tracker_state.json", encoding="utf-8"))
    res["max_drawdown_pct"] = round(100 * _max_drawdown(tracker["equity_curve"]), 2)
    res["concentration"] = _concentration(tracker)
    rank = pd.read_csv(d / "full" / "rank_log.csv")
    res["prob_profit_reliability"] = _prob_profit_reliability(rank)
    res["ev_quintile"] = _ev_quintile(rank)
    if res.get("start") and res.get("end"):
        res["ew_mag7_bh_pct"] = _ew_benchmark(conn, res["start"], res["end"])
        full = per_fr.get("full", {})
        if full.get("ret_pct") is not None and res["ew_mag7_bh_pct"] is not None:
            res["alpha_vs_ew_pp"] = round(full["ret_pct"] - res["ew_mag7_bh_pct"], 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "analysis.json"))
    args = ap.parse_args()
    conn = WheelRunner().connector
    wids = [args.window] if args.window else WINDOW_IDS
    results = [analyze_window(w, conn) for w in wids]
    done = [r for r in results if r.get("status") == "ok"]
    # campaign roll-up (full friction)
    rets = [r["friction"]["full"]["ret_pct"] for r in done if "full" in r["friction"]]
    rollup = {}
    if rets:
        rollup = {
            "n_windows": len(rets),
            "mean_ret_pct": round(float(np.mean(rets)), 2),
            "median_ret_pct": round(float(np.median(rets)), 2),
            "std_ret_pct": round(float(np.std(rets, ddof=1)), 2) if len(rets) > 1 else None,
            "n_positive": int(sum(1 for x in rets if x > 0)),
            "worst": round(min(rets), 2),
            "best": round(max(rets), 2),
        }
    payload = {"rollup": rollup, "windows": results}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # human summary
    print(f"\n{'=' * 100}\nMAG7 CAMPAIGN — full-friction summary  (root={ROOT})\n{'=' * 100}")
    print(
        f"{'window':22s}{'ret%':>8}{'EW B&H%':>9}{'alpha':>8}{'maxDD%':>8}{'trades':>8}{'assign':>7}{'hit':>6}{'rho':>7}{'pk1name%':>10}{'recon✓':>8}"
    )
    for r in done:
        f = r["friction"].get("full", {})
        c = r.get("concentration", {})
        print(
            f"{r['wid']:22s}{f.get('ret_pct', 0):>8}{str(r.get('ew_mag7_bh_pct', '-')):>9}{str(r.get('alpha_vs_ew_pp', '-')):>8}"
            f"{r.get('max_drawdown_pct', '-'):>8}{str(f.get('executed_trades', '-')):>8}{str(f.get('put_assignments', '-')):>7}"
            f"{str(f.get('hit_rate', '-')):>6}{str(f.get('spearman_rho', '-')):>7}{str(c.get('peak_single_name_pct', '-')):>10}{str(c.get('recon_vs_tracker_count_match_pct', '-')):>8}"
        )
    if rollup:
        print(
            f"\nROLLUP: n={rollup['n_windows']} mean={rollup['mean_ret_pct']}% median={rollup['median_ret_pct']}% "
            f"std={rollup['std_ret_pct']} positive={rollup['n_positive']}/{rollup['n_windows']} "
            f"worst={rollup['worst']}% best={rollup['best']}%"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
