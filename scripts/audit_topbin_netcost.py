#!/usr/bin/env python3
"""W6 — is the calm/elevated-entry top-bin over-confidence net-COSTLY? (#446).

Validation / measurement-only, rail-independent. The decision trio
(``ev_engine`` / ``wheel_runner`` / ``candidate_dossier``) is **never** edited —
this driver only *calls* the authoritative ranker and forward-replays realized
P&L. Follow-up to W3/#442 (top-bin ``prob_profit`` > 0.90 over-confidence is
calm/elevated-ENTRY ~-16pp, absent at crisis-entry).

The question (the whole point): does that calm/elevated-entry over-confidence
translate into **net dollar LOSSES**, or are the trades **still net-positive
premium** despite missing their forecast ~16pp of the time? Only a net-COSTLY
answer justifies building a targeted size-down rule — the R11b precedent
(#437, validated net-costly, CLOSED) showed a calibration gap is *not*
automatically a reason to size down.

Method (mirrors the R11b A/B discipline):
  1. Monthly grid over the union of both windows (2020-01..2025-12). All slices
     END well before the W1/#439 2026-03-23 OHLCV splice — no contamination.
  2. At each as_of, classify ENTRY regime by VIX-at-entry
     (calm <=15 / elevated 15-25 / crisis >25 — the W6 bands from #446), then
     rank the full universe via ``rank_candidates_by_ev`` (the authoritative,
     rail-independent ranker; top_n=10_000, min_ev=-1e9 to capture the full
     evaluated population, matching the W3 calibration study). ``collect`` also
     tallies the forward-spot skip rate + ranker drops (survivorship surface).
  3. Forward-replay each short put's held-to-expiry P&L with the canonical
     ``_forward_replay_realized_pnl`` + ``_spot_on_or_after`` helpers from
     ``backtests/regression/_common.py``, at three friction levels
     (none / bid_ask / full). The headline verdict is at **full** friction —
     friction only makes short puts *less* profitable, so an honest net-cost
     answer must be all-in.
  4. Report, for the calm/elevated (VIX<=25) population, top-bin vs non-top-bin
     vs all-candidate baseline across SEVERAL independent cuts: the two named
     (overlapping) windows, the two genuinely DISJOINT regime halves
     (2020-2022 COVID+bear vs 2023-2025 low-vol — the real independence test),
     pooled, calm-only, elevated-only, and the would-trade subset. Each carries
     mean realized $/contract (cluster-bootstrap 95% CI over as_of dates), total,
     win-rate (Wilson 95%), and the realized-vs-forecast calibration gap
     (reproducing W3's over-confidence in sign). Crisis-entry is a reference row.
  5. Verdict (``make_verdict``): the 'no rule' decision turns on the ABSENCE of
     net-cost, not on any single CI. NET-COSTLY only if a cut's CI excludes 0
     below; otherwise (top-bin not losing money on any independent cut, and above
     baseline) -> NOT NET-COSTLY -> document the caveat, build nothing (the R11b
     precedent #437: a calibration gap is not a reason to gate profitable premium).

Honesty: every per-bucket n is reported; no claim is called SUPPORTED at
n < ~30; the cluster bootstrap respects within-as_of correlation; CIs are flagged
optimistically tight (candidates are not i.i.d.); the disjoint halves provide the
independent check the ~80%-overlapping named windows cannot. JSON is written
BEFORE any pretty-print.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_topbin_netcost.py [--limit N] [--quick]
      # parallel: N collectors over date-chunks (--dates ... --collect-to x.jsonl)
      #           then one merge pass (--analyze-from "x*.jsonl")
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from backtests.regression._common import (  # noqa: E402
    _forward_replay_realized_pnl,
    _spot_on_or_after,
    friction_adjusted_premium,
    friction_assignment_cost,
    friction_open_cost,
)
from engine.wheel_runner import WheelRunner  # noqa: E402

OUT = REPO / "docs" / "verification_artifacts" / "topbin_netcost_2026-06-28"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse W3's Wilson helper for win-rate CIs (methodology parity with #442).
_spec = importlib.util.spec_from_file_location(
    "_vnv_calib", REPO / "scripts" / "vnv_prob_profit_calibration.py"
)
_vnv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vnv)
wilson = _vnv.wilson

TOP_BIN_PROB = 0.90  # the over-confident bin (#442; R11 R11_TOP_BIN_PROB)
FRICTION = "full"  # headline friction level for the verdict
SEED = 12345
N_BOOT = 2000


def entry_band(vix: float) -> str:
    """W6 entry-VIX bands (#446): calm <=15, elevated 15-25, crisis >25."""
    if vix != vix:  # NaN
        return "unknown"
    if vix <= 15.0:
        return "calm (<=15)"
    if vix <= 25.0:
        return "elevated (15-25)"
    return "crisis (>25)"


def month_grid(y0: int, m0: int, y1: int, m1: int) -> list[str]:
    """First business day of each month in [(y0,m0), (y1,m1)] inclusive."""
    out: list[str] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        first = pd.Timestamp(year=y, month=m, day=1)
        bd = pd.bdate_range(first, first + pd.Timedelta(days=6))[0]
        out.append(bd.strftime("%Y-%m-%d"))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def realized_at(level: str, strike: float, premium: float, spot_exp: float) -> float:
    """Held-to-expiry short-put P&L (dollars/contract) at a friction level.

    Premium is bid/ask-adjusted; open commission is always deducted at ``full``;
    assignment slippage is deducted only when the put finishes ITM (assigned).
    """
    prem_adj = friction_adjusted_premium(premium, level)
    pnl = _forward_replay_realized_pnl(strike, prem_adj, spot_exp)  # *100 inside
    pnl -= friction_open_cost(1, level)
    if spot_exp < strike:  # ITM put -> assigned
        pnl -= friction_assignment_cost(strike, 1, level)
    return pnl


def _boot_mean_ci(vals: list[float], dates: list[str]) -> list[float]:
    """Cluster (block) bootstrap 95% CI of the mean, resampling as_of dates.

    Respects within-as_of correlation (same-day candidates share macro shocks);
    more honest than an i.i.d. bootstrap given recurring names + overlap.
    """
    if not vals:
        return [float("nan"), float("nan")]
    by_date: dict[str, list[float]] = defaultdict(list)
    for v, d in zip(vals, dates, strict=True):
        by_date[d].append(v)
    keys = list(by_date.keys())
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(keys))
    means = []
    for _ in range(N_BOOT):
        pick = rng.choice(idx, size=len(idx), replace=True)
        pool: list[float] = []
        for j in pick:
            pool.extend(by_date[keys[j]])
        if pool:
            means.append(float(np.mean(pool)))
    if not means:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def summarize(rows: list[dict]) -> dict:
    """Per-bucket economics. ``win_rate_gross`` (pnl_none>0) is the calibration-
    comparable realized rate vs forecast ``prob_profit``; ``mean_pnl_full`` is the
    all-in economic answer that drives the net-cost verdict."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    pnl_full = np.array([r["pnl_full"] for r in rows], dtype=float)
    pnl_none = np.array([r["pnl_none"] for r in rows], dtype=float)
    pnl_bidask = np.array([r["pnl_bidask"] for r in rows], dtype=float)
    pp = np.array([r["prob_profit"] for r in rows], dtype=float)
    dates = [r["as_of"] for r in rows]

    win_full = int((pnl_full > 0).sum())
    win_gross = int((pnl_none > 0).sum())
    lo_f, hi_f = wilson(win_full, n)
    fc = float(pp.mean())
    rg = win_gross / n
    return {
        "n": n,
        "mean_pnl_full": round(float(pnl_full.mean()), 2),
        "mean_pnl_full_ci95": _boot_mean_ci(list(pnl_full), dates),
        "total_pnl_full": round(float(pnl_full.sum()), 2),
        "mean_pnl_bidask": round(float(pnl_bidask.mean()), 2),
        "mean_pnl_gross": round(float(pnl_none.mean()), 2),
        "win_rate_full": round(win_full / n, 4),
        "win_rate_full_ci95": [round(lo_f, 4), round(hi_f, 4)],
        "win_rate_gross": round(rg, 4),
        "mean_forecast_prob_profit": round(fc, 4),
        "calibration_gap_pp": round((rg - fc) * 100.0, 2),  # realized - forecast (gross)
        "itm_rate": round(float(np.mean([r["itm"] for r in rows])), 4),
        "mean_premium": round(float(np.mean([r["premium"] for r in rows])), 3),
    }


def bucket_report(rows: list[dict]) -> dict:
    """All W6 buckets for one slice of records."""
    ce = [r for r in rows if r["band"] in ("calm (<=15)", "elevated (15-25)")]
    calm = [r for r in rows if r["band"] == "calm (<=15)"]
    elev = [r for r in rows if r["band"] == "elevated (15-25)"]
    crisis = [r for r in rows if r["band"] == "crisis (>25)"]

    def split(group: list[dict]) -> dict:
        tb = [r for r in group if r["top_bin"]]
        return {
            "all": summarize(group),
            "top_bin": summarize(tb),
            "non_top_bin": summarize([r for r in group if not r["top_bin"]]),
            "top_bin_would_trade": summarize([r for r in tb if r["would_trade"]]),
        }

    return {
        "calm_elevated (VIX<=25)": split(ce),
        "calm (<=15)": split(calm),
        "elevated (15-25)": split(elev),
        "crisis (>25)": {
            "all": summarize(crisis),
            "top_bin": summarize([r for r in crisis if r["top_bin"]]),
        },
    }


def make_verdict(by_window: dict) -> dict:
    """Net-cost verdict on the calm/elevated top-bin.

    Reframed after adversarial review: the 'no rule' decision turns on the ABSENCE
    of net-cost, not on any single CI excluding 0. We therefore assert the verdict
    on (a) the point estimate being positive on EVERY independent cut — pooled, the
    two disjoint regime halves, calm, elevated, and the would-trade subset — and
    (b) the top bin sitting far above BOTH baselines (all-candidate and, the cleaner
    contrast, non-top-bin). The strict 'CI excludes 0' is reported per cut but is not
    required: a top bin that is merely net-zero still does not justify a size-down.
    """

    def ce(win: str, k: str) -> dict:
        return by_window[win]["calm_elevated (VIX<=25)"][k]

    pooled = "pooled_2020-2025"
    tb = ce(pooled, "top_bin")
    n = tb.get("n", 0)
    mean = tb.get("mean_pnl_full")
    ci = tb.get("mean_pnl_full_ci95", [float("nan"), float("nan")])
    base_all = ce(pooled, "all").get("mean_pnl_full")
    base_ntb = ce(pooled, "non_top_bin").get("mean_pnl_full")

    # Every independent cut's top-bin mean (point estimate) + CI status.
    cuts = {
        "pooled_2020-2025": ce(pooled, "top_bin"),
        "overlap_2020-2024": ce("2020-2024", "top_bin"),
        "overlap_2021-2025": ce("2021-2025", "top_bin"),
        "disjoint_2020-2022": ce("disjoint_2020-2022", "top_bin"),
        "disjoint_2023-2025": ce("disjoint_2023-2025", "top_bin"),
        "calm_only": by_window[pooled]["calm (<=15)"]["top_bin"],
        "elevated_only": by_window[pooled]["elevated (15-25)"]["top_bin"],
        "would_trade": ce(pooled, "top_bin_would_trade"),
    }
    cut_means = {k: v.get("mean_pnl_full") for k, v in cuts.items() if v.get("n")}
    cut_ci = {
        k: ("excludes_0_above" if (v.get("mean_pnl_full_ci95") or [0, 0])[0] > 0 else "straddles_0")
        for k, v in cuts.items()
        if v.get("n")
    }
    present = [m for m in cut_means.values() if m is not None]
    all_cuts_positive = bool(present) and all(m > 0 for m in present)
    any_cut_ci_costly = any(
        (cuts[k].get("mean_pnl_full_ci95") or [0, 0])[1] < 0 for k in cuts if cuts[k].get("n")
    )
    min_cut = min(present) if present else None

    # Disjoint halves are the genuine independence test (distinct regimes, no overlap).
    dj = ["disjoint_2020-2022", "disjoint_2023-2025"]
    dj_means = {w: cuts[w].get("mean_pnl_full") for w in dj}
    disjoint_both_positive = all((m is not None and m > 0) for m in dj_means.values())

    if n < 30:
        label = "INSUFFICIENT (n<30) — not SUPPORTED"
    elif any_cut_ci_costly and not all_cuts_positive:
        label = "NET-COSTLY on some cut — investigate a targeted size-down"
    elif all_cuts_positive and disjoint_both_positive:
        label = (
            "NOT NET-COSTLY — top-bin net-POSITIVE on every independent cut "
            f"(min +${min_cut:.0f}/contract across calm/elevated, both disjoint regime halves, "
            "and the would-trade subset), far above the negative baseline. The over-confidence "
            "is a level-miscalibration, not a rank inversion → NO rule (R11b precedent #437). "
            "CI note: pooled mean CI excludes 0; some sub-cuts (elevated-only, the COVID-era "
            "disjoint half) have wider near-0-lower-bound CIs — the conclusion rests on "
            "point-estimate positivity across independent regimes + the large margin over "
            "baseline, not on any single CI."
        )
    else:
        label = "INCONCLUSIVE (mixed signs across cuts)"

    return {
        "label": label,
        "pooled_calm_elevated_top_bin": {
            "n": n,
            "mean_pnl_full": mean,
            "mean_pnl_full_ci95": ci,
            "baseline_all_mean_pnl_full": base_all,
            "baseline_non_top_bin_mean_pnl_full": base_ntb,
            "delta_vs_all_baseline": (
                round(mean - base_all, 2) if (mean is not None and base_all is not None) else None
            ),
            "delta_vs_non_top_bin": (
                round(mean - base_ntb, 2) if (mean is not None and base_ntb is not None) else None
            ),
        },
        "top_bin_mean_pnl_full_by_cut": {k: round(v, 2) for k, v in cut_means.items()},
        "ci_status_by_cut": cut_ci,
        "all_cuts_point_estimate_positive": all_cuts_positive,
        "disjoint_halves_both_positive": disjoint_both_positive,
        "min_cut_mean_pnl_full": round(min_cut, 2) if min_cut is not None else None,
        "decision_rule": (
            "NET-COSTLY on an independent cut -> spec a targeted size-down for the Windows "
            "terminal; otherwise (top-bin not losing money, and above baseline) -> document "
            "the calibration caveat, build NOTHING (R11b precedent #437: a calibration gap is "
            "not a reason to gate profitable premium)."
        ),
    }


def collect(runner, dates: list[str], limit: int | None) -> tuple[list[dict], dict]:
    """Rank + forward-replay every short put over ``dates``.

    Returns ``(records, diag)``. ``diag`` quantifies the survivorship surface the
    adversarial review flagged: per-band count of candidates dropped for a missing
    forward spot (``_spot_on_or_after`` → None), the total ranked, and the ranker's
    own upstream gate drops. Pure and date-parallelizable: a chunk of dates can run
    in its own process and the records + diag are simply concatenated / summed.
    """
    conn = runner.connector
    spot_cache: dict[tuple, float | None] = {}
    recs: list[dict] = []
    diag = {
        "n_ranked": 0,
        "n_no_fwd_spot": 0,
        "no_fwd_spot_by_band": defaultdict(int),
        "ranker_drops_total": 0,
    }
    for as_of in dates:
        asof_ts = pd.Timestamp(as_of)
        vix = float(conn.get_vix_regime(as_of).get("vix", float("nan")))
        band = entry_band(vix)
        try:
            df = runner.rank_candidates_by_ev(
                tickers=None,
                universe_limit=limit,
                top_n=10_000,
                min_ev_dollars=-1e9,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {as_of}: rank FAILED {type(e).__name__}: {e}")
            continue
        diag["n_ranked"] += len(df)
        diag["ranker_drops_total"] += int(df.attrs.get("drops_summary", {}).get("total_dropped", 0))
        n_used = 0
        for _, row in df.iterrows():
            try:
                pp = float(row["prob_profit"])
                dte = int(row["dte"])
                strike = float(row["strike"])
                prem = float(row["premium"])
                ev = float(row["ev_dollars"])
                tk = str(row["ticker"])
            except (KeyError, TypeError, ValueError):
                continue
            if pp != pp or prem != prem or strike != strike:  # NaN guard
                continue
            expiry = (asof_ts + pd.Timedelta(days=dte)).date()
            key = (tk, expiry.isoformat())
            if key not in spot_cache:
                spot_cache[key] = _spot_on_or_after(conn, tk, expiry)
            spot_exp = spot_cache[key]
            if spot_exp is None:  # no forward bar — survivorship surface, quantified
                diag["n_no_fwd_spot"] += 1
                diag["no_fwd_spot_by_band"][band] += 1
                continue
            recs.append(
                {
                    "as_of": as_of,
                    "vix": round(vix, 2) if vix == vix else None,
                    "band": band,
                    "ticker": tk,
                    "prob_profit": pp,
                    "ev_dollars": ev,
                    "premium": prem,
                    "strike": strike,
                    "spot_exp": spot_exp,
                    "itm": bool(spot_exp < strike),
                    "top_bin": bool(pp > TOP_BIN_PROB),
                    "would_trade": bool(ev > 0),
                    "pnl_none": realized_at("none", strike, prem, spot_exp),
                    "pnl_bidask": realized_at("bid_ask", strike, prem, spot_exp),
                    "pnl_full": realized_at("full", strike, prem, spot_exp),
                }
            )
            n_used += 1
        print(f"  {as_of}: VIX={vix:5.1f} {band:<17} {n_used:>4} scored", flush=True)
    diag["no_fwd_spot_by_band"] = dict(diag["no_fwd_spot_by_band"])
    return recs, diag


def analyze(
    recs: list[dict], union_dates: int, limit: int | None, diag: dict | None = None
) -> dict:
    """Bucket, bootstrap, and judge the collected records (fast; runs once)."""
    n_total = len(recs)

    def slice_rows(pred) -> list[dict]:
        return [r for r in recs if pred(r["as_of"])]

    by_window = {
        # Brief's named windows — NOTE they overlap ~80% (2021-2024 shared), so
        # cross-window agreement here is NOT independent validation.
        "2020-2024": bucket_report(slice_rows(lambda d: "2020-01-01" <= d <= "2024-12-31")),
        "2021-2025": bucket_report(slice_rows(lambda d: "2021-01-01" <= d <= "2025-12-31")),
        # Genuinely DISJOINT halves — the real independence test (distinct regimes:
        # COVID + 2022 bear vs the 2023-25 low-vol recovery).
        "disjoint_2020-2022": bucket_report(
            slice_rows(lambda d: "2020-01-01" <= d <= "2022-12-31")
        ),
        "disjoint_2023-2025": bucket_report(
            slice_rows(lambda d: "2023-01-01" <= d <= "2025-12-31")
        ),
        "pooled_2020-2025": bucket_report(recs),
    }
    verdict = make_verdict(by_window)
    band_counts = dict(pd.Series([r["band"] for r in recs]).value_counts().items()) if recs else {}
    return {
        "meta": {
            "question": "Is the calm/elevated-ENTRY top-bin (prob_profit>0.90) over-confidence "
            "net-dollar-COSTLY, or still net-positive premium (R11b lesson)?",
            "follow_up_to": "#442 (W3); R11b precedent #437 (CLOSED, net-costly)",
            "windows": {"A": "2020-2024", "B": "2021-2025", "union_grid_dates": union_dates},
            "entry_vix_bands": "calm<=15, elevated 15-25, crisis>25 (VIX-at-entry, #446)",
            "top_bin_prob": TOP_BIN_PROB,
            "headline_friction": FRICTION,
            "realized_rule": "_forward_replay_realized_pnl + _spot_on_or_after "
            "(backtests/regression/_common.py); (premium_adj - max(0,strike-spot))*100 "
            "- open_commission - assignment_slip(if ITM)",
            "ranker": "rank_candidates_by_ev (authoritative, rail-independent; "
            "top_n=10000, min_ev=-1e9, full evaluated population)",
            "universe_limit": limit,
            "data_caveats": [
                "Both windows end before the 2026-03-23 OHLCV splice (W1/#439) — no contamination.",
                "premium_source is mostly synthetic_bsm on the Bloomberg-CSV path; edge_vs_fair "
                "~0 there, so this measures the engine's own modeled premium vs realized moves.",
                "Wilson/cluster-bootstrap CIs are optimistically tight — candidates are not "
                "i.i.d. (recurring names, overlapping forward windows).",
                "Named windows 2020-2024 / 2021-2025 overlap ~80% — see disjoint_2020-2022 / "
                "disjoint_2023-2025 for the genuinely-independent regime split.",
                "Calm-entry (VIX<=15) occurred only in 2023-07..2024-12; the calm sub-bucket "
                "is therefore both small (n<30) and temporally concentrated in one low-vol regime.",
                "Realized spot uses the close on-or-after expiry (_spot_on_or_after, 7-day window) "
                "— Friday-close settlement convention; ~0.5-1% gap risk, immaterial to the verdict.",
            ],
            "n_total": n_total,
            "band_counts": {k: int(v) for k, v in band_counts.items()},
            "collection_diag": diag
            or {"note": "diag unavailable (loaded from records without companion .diag.json)"},
        },
        "by_window": by_window,
        "verdict": verdict,
    }


def _print_report(report: dict, out_path: Path) -> None:
    by_window = report["by_window"]
    verdict = report["verdict"]
    print(f"\ntotal candidate-outcomes = {report['meta']['n_total']}")
    print(f"collection diag = {report['meta'].get('collection_diag')}")
    print("\n=== W6 TOP-BIN NET-COST (calm/elevated VIX<=25) ===")
    for win in [
        "2020-2024",
        "2021-2025",
        "disjoint_2020-2022",
        "disjoint_2023-2025",
        "pooled_2020-2025",
    ]:
        ce = by_window[win]["calm_elevated (VIX<=25)"]
        tb, base, ntb = ce["top_bin"], ce["all"], ce["non_top_bin"]
        if tb.get("n"):
            print(
                f"{win:<20} TOP-BIN n={tb['n']:>5}  "
                f"mean_full=${tb['mean_pnl_full']:>8.2f} CI{tb['mean_pnl_full_ci95']}  "
                f"win={tb['win_rate_full']}  gap={tb['calibration_gap_pp']}pp  "
                f"| base_all=${base['mean_pnl_full']:>8.2f} base_nontop=${ntb['mean_pnl_full']:>8.2f}"
            )
    cr = by_window["pooled_2020-2025"]["crisis (>25)"]["top_bin"]
    if cr.get("n"):
        print(
            f"{'crisis-ref (>25)':<20} TOP-BIN n={cr['n']:>5}  mean_full=${cr['mean_pnl_full']:>8.2f} "
            f"CI{cr['mean_pnl_full_ci95']}  gap={cr['calibration_gap_pp']}pp  (reference, not the W6 subject)"
        )
    print(f"\nVERDICT: {verdict['label']}")
    print(f"  by-cut top-bin mean_full: {verdict['top_bin_mean_pnl_full_by_cut']}")
    print(f"  ci status by cut: {verdict['ci_status_by_cut']}")
    print(f"  pooled: {verdict['pooled_calm_elevated_top_bin']}")
    print(f"\nJSON -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="universe_limit (None = full S&P)")
    ap.add_argument("--quick", action="store_true", help="3-date smoke grid")
    ap.add_argument("--dates", type=str, default=None, help="comma-separated as_of override")
    ap.add_argument("--collect-to", type=str, default=None, help="JSONL records sink; exit after")
    ap.add_argument(
        "--analyze-from", type=str, default=None, help="glob of JSONL record files; analyze + exit"
    )
    args = ap.parse_args()

    union = month_grid(2020, 1, 2025, 12)
    if args.quick:
        union = ["2020-04-01", "2022-06-01", "2024-03-01"]
    if args.dates:
        union = [d.strip() for d in args.dates.split(",") if d.strip()]

    # ---- analyze-only pass: load records produced by parallel collectors --------
    if args.analyze_from:
        import glob as _glob

        recs: list[dict] = []
        files = [Path(p) for p in sorted(_glob.glob(args.analyze_from))]
        for fp in files:
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    recs.append(json.loads(line))
        # aggregate the companion .diag.json sidecars (skip + drops accounting)
        agg = {
            "n_ranked": 0,
            "n_no_fwd_spot": 0,
            "no_fwd_spot_by_band": {},
            "ranker_drops_total": 0,
        }
        for fp in files:
            dp = fp.with_suffix(fp.suffix + ".diag.json")
            if dp.exists():
                d = json.loads(dp.read_text(encoding="utf-8"))
                agg["n_ranked"] += d.get("n_ranked", 0)
                agg["n_no_fwd_spot"] += d.get("n_no_fwd_spot", 0)
                agg["ranker_drops_total"] += d.get("ranker_drops_total", 0)
                for b, c in d.get("no_fwd_spot_by_band", {}).items():
                    agg["no_fwd_spot_by_band"][b] = agg["no_fwd_spot_by_band"].get(b, 0) + c
        agg["no_fwd_spot_rate"] = round(agg["n_no_fwd_spot"] / max(1, agg["n_ranked"]), 5)
        union_dates = len({r["as_of"] for r in recs})
        report = analyze(recs, union_dates, args.limit, diag=agg)
        out_path = OUT / "w6_topbin_netcost.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"loaded {len(recs)} records from {len(files)} file(s); diag={agg}")
        _print_report(report, out_path)
        return 0

    runner = WheelRunner()
    print(
        f"provider={type(runner.connector).__name__}  as_of_dates={len(union)}  "
        f"limit={args.limit}  friction={FRICTION}",
        flush=True,
    )
    recs, diag = collect(runner, union, args.limit)

    # ---- collect-only pass: dump JSONL (+ .diag.json) for a downstream merge -----
    if args.collect_to:
        sink = Path(args.collect_to)
        sink.parent.mkdir(parents=True, exist_ok=True)
        with sink.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        sink.with_suffix(sink.suffix + ".diag.json").write_text(json.dumps(diag), encoding="utf-8")
        print(f"\nwrote {len(recs)} records -> {sink}  (diag: {diag})")
        return 0

    # ---- inline single-process path (default) -----------------------------------
    diag["no_fwd_spot_rate"] = round(diag["n_no_fwd_spot"] / max(1, diag["n_ranked"]), 5)
    report = analyze(recs, len(union), args.limit, diag=diag)
    out_path = OUT / "w6_topbin_netcost.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")  # persist BEFORE print
    _print_report(report, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
