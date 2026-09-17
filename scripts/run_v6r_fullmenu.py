"""V6-r1 — un-censoring H1 (plan doc §10.1). Every parameter FROZEN here.

    export SWE_DATA_PROVIDER=bloomberg
    python scripts/run_v6r_fullmenu.py run

SECOND read of the already-spent 2007-01-03 -> 2009-06-30 slice
(validation data per plan §2.2 — counted in the ledger, p-values
discounted; 1998/LTCM stays locked). Identical window and parameters to
the §9 lockbox spec EXCEPT top_n=100: the FULL ~100-name menu is logged
per date, so the §9.3 top-of-book censoring on H1 is directly resolved.
Frozen verdicts (§10.1): CAVEAT_RETIRED if the full-menu grind/baseline
EV-positive ratio > 0.5 (H1's FAIL is unconditional; F-V6-1 stands);
CENSORING_LOAD_BEARING if <= 0.5 (real refusal below the top-15 —
F-V6-1 must be rewritten; good news, reported as such). Synthetic BSM
premiums; NAV is not evidence (§2.4). Deterministic; a crash may be
restarted; never re-parameterized. Raw artifacts are written before
verdict computation and each verdict is exception-captured (the V6
attempt-1/2 lesson).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "v6r_fullmenu"
)

#: THE frozen spec (plan doc §10.1). Identical to §9.0 except top_n.
SPEC = {
    "start": "2007-01-03",
    "end": "2009-06-30",
    "capital": 1_000_000.0,
    "friction_level": "full",
    "top_n": 100,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
    "max_universe": 100,
    "rebalance_months": 3,
    "baseline_months": [f"2007-{m:02d}" for m in range(1, 13)],
    "grind_months": ["2008-10", "2008-11", "2008-12"],
    "ratio_cut": 0.5,
    "saturation_guard": 15,
    "opens_appetite": 3,
}


def fm1_full_menu_rate(rank_log: pd.DataFrame) -> dict:
    """The un-censored H1: full-menu monthly EV-positive rate, grind/baseline."""
    df = rank_log.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    df["ev_pos"] = pd.to_numeric(df["ev_dollars"], errors="coerce") > 0
    monthly = df.groupby("month")["ev_pos"].mean()
    base = monthly[monthly.index.isin(SPEC["baseline_months"])]
    grind = monthly[monthly.index.isin(SPEC["grind_months"])]
    if base.empty or grind.empty:
        return {"verdict": "INSUFFICIENT", "n_base_months": len(base), "n_grind_months": len(grind)}
    ratio = float(grind.mean() / base.mean()) if base.mean() > 0 else float("inf")
    return {
        "baseline_rate": float(base.mean()),
        "grind_rate": float(grind.mean()),
        "ratio": ratio,
        "monthly_rates": {k: float(v) for k, v in monthly.items()},
        "verdict": "CAVEAT_RETIRED" if ratio > SPEC["ratio_cut"] else "CENSORING_LOAD_BEARING",
    }


def fm2_depth_profile(rank_log: pd.DataFrame) -> dict:
    """Report-only: per-date EV-positive count vs the two appetite lines."""
    df = rank_log.copy()
    df["ev_pos"] = pd.to_numeric(df["ev_dollars"], errors="coerce") > 0
    per_date = df.groupby("date")["ev_pos"].sum()
    if per_date.empty:
        return {"verdict": "INSUFFICIENT"}
    month = pd.to_datetime(per_date.index).strftime("%Y-%m")
    monthly = per_date.groupby(month).agg(["min", "median"])
    return {
        "verdict": "REPORT_ONLY",
        "n_dates": int(len(per_date)),
        "n_dates_below_saturation_guard": int((per_date < SPEC["saturation_guard"]).sum()),
        "n_dates_below_opens_appetite": int((per_date < SPEC["opens_appetite"]).sum()),
        "min_ev_pos_per_date": int(per_date.min()),
        "monthly_min_median": {
            k: {"min": int(r["min"]), "median": float(r["median"])} for k, r in monthly.iterrows()
        },
    }


def _safe(fn, *fn_args) -> dict:
    try:
        return fn(*fn_args)
    except Exception:  # noqa: BLE001
        return {"verdict": "ERROR", "traceback": traceback.format_exc()}


def cmd_run(args: argparse.Namespace) -> int:
    from backtests.survivorship import run_survivorship_backtest

    deep = _REPO_ROOT / "data" / "bloomberg" / "deep"
    if not deep.exists():
        print(
            f"[v6r] PRECONDITION FAILED: {deep} absent — deep panels live on operator machines only",
            file=sys.stderr,
        )
        return 2
    print(
        f"[v6r] SECOND read of the spent 2007-2009 slice (plan §10.1, counted): "
        f"{json.dumps({k: v for k, v in SPEC.items() if not isinstance(v, list)})}",
        flush=True,
    )
    t0 = time.monotonic()
    result = run_survivorship_backtest(
        capital=SPEC["capital"],
        start=SPEC["start"],
        end=SPEC["end"],
        friction_level=SPEC["friction_level"],
        rebalance_months=SPEC["rebalance_months"],
        max_universe=SPEC["max_universe"],
        top_n=SPEC["top_n"],
        max_new_per_day=SPEC["max_new_per_day"],
        dte_target=SPEC["dte_target"],
        delta_target=SPEC["delta_target"],
        contracts=SPEC["contracts"],
    )
    elapsed = time.monotonic() - t0
    rank_log = result["rank_log"]
    if not isinstance(rank_log, pd.DataFrame):
        rank_log = pd.DataFrame(rank_log)
    print(
        f"[v6r] engine pass done: {len(rank_log)} ranked rows in {elapsed / 60:.1f} min",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_log.to_csv(out_dir / "v6r_rank_log.csv.gz", index=False, compression="gzip")
    print(f"[v6r] raw capture written to {out_dir}", flush=True)

    report = {
        "meta": {
            "spec": SPEC,
            "generated_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "rows_ranked": int(len(rank_log)),
        "diagnostic_columns_present": sorted(
            c for c in ("cvar_5", "pnl_p25", "pnl_p50", "pnl_p75", "n_scenarios") if c in rank_log
        ),
        "FM1_full_menu_rate": _safe(fm1_full_menu_rate, rank_log),
        "FM2_depth_profile": _safe(fm2_depth_profile, rank_log),
    }
    out = out_dir / "v6r_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[v6r] wrote {out}", flush=True)
    print("\n=== V6-r1 full-menu — verdicts ===")
    for key in ("FM1_full_menu_rate", "FM2_depth_profile"):
        block = report[key]
        keys = {k: v for k, v in block.items() if not isinstance(v, dict)}
        print(f"  {key}: {keys}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["run"])
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
