#!/usr/bin/env python3
"""#442 — R11 band-split net-cost: does R11 (VIX>25, top-bin) over-fire into a
well-calibrated VIX>=30 band, and does it miss a net-costly band it should hit?

Reuses the W6 collection machinery VERBATIM (scripts/audit_topbin_netcost.collect:
authoritative ranker -> forward-replay realized P&L at full friction). This driver
only (a) parallelizes the collection over a DENSE >=25 grid (28 distinct VIX>=30
dates + 40 VIX 25-30 dates — vs W3's single >=30 date) and (b) re-bands the
per-candidate records FINELY around the R11 threshold to split the >25 zone into
25-30 vs >=30. Read-only on the engine; no trio edit.

The decisive cell: top-bin (prob_profit>0.90) mean realized $/contract at VIX>=30
across MANY dates. If strongly net-POSITIVE + multi-date robust -> R11 over-fires
there (a band-aware carve-out would reclaim premium). If negative/near-zero or
single-regime -> keep R11 as-is.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # repo root (docs/verification_artifacts/<dir>/<file>)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

# Outputs land alongside this script when re-run from the committed location.
SCRATCH = Path(__file__).resolve().parent

# Dense >=25 weekly grid from probe_vix.py (real VIX-at-entry dates).
GE25 = [
    "2020-02-26",
    "2020-03-04",
    "2020-03-11",
    "2020-03-18",
    "2020-03-25",
    "2020-04-01",
    "2020-04-08",
    "2020-04-15",
    "2020-04-22",
    "2020-04-29",
    "2020-05-06",
    "2020-05-13",
    "2020-05-20",
    "2020-05-27",
    "2020-06-03",
    "2020-06-10",
    "2020-06-17",
    "2020-06-24",
    "2020-07-01",
    "2020-07-08",
    "2020-07-15",
    "2020-09-02",
    "2020-09-09",
    "2020-09-16",
    "2020-09-23",
    "2020-09-30",
    "2020-10-07",
    "2020-10-14",
    "2020-10-21",
    "2020-10-28",
    "2020-11-04",
    "2021-01-06",
    "2021-01-27",
    "2021-03-03",
    "2021-05-12",
    "2021-12-01",
    "2022-01-26",
    "2022-02-23",
    "2022-03-02",
    "2022-03-09",
    "2022-03-16",
    "2022-04-27",
    "2022-05-04",
    "2022-05-11",
    "2022-05-18",
    "2022-05-25",
    "2022-06-01",
    "2022-06-15",
    "2022-06-22",
    "2022-06-29",
    "2022-07-06",
    "2022-07-13",
    "2022-08-31",
    "2022-09-14",
    "2022-09-21",
    "2022-09-28",
    "2022-10-05",
    "2022-10-12",
    "2022-10-19",
    "2022-10-26",
    "2022-11-02",
    "2022-11-09",
    "2023-03-15",
    "2024-08-07",
    "2024-12-18",
    "2025-04-09",
    "2025-04-16",
    "2025-04-23",
]

N_WORKERS = 6


def _collect_chunk(dates: list[str]) -> tuple[list[dict], dict]:
    """Worker: build a runner and run the Mac's validated collect() over a chunk."""
    import warnings as _w

    _w.filterwarnings("ignore")
    sys.path.insert(0, str(REPO))
    from engine.wheel_runner import WheelRunner

    spec = importlib.util.spec_from_file_location(
        "_atn", REPO / "scripts" / "audit_topbin_netcost.py"
    )
    atn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atn)
    runner = WheelRunner()
    recs, diag = atn.collect(runner, dates, None)
    return recs, diag


def fine_band(vix: float) -> str:
    if vix != vix:
        return "unknown"
    if vix < 20.0:
        return "calm (<20)"
    if vix < 25.0:
        return "lowelev (20-25)"
    if vix < 30.0:
        return "highelev (25-30)"
    return "crisis (>=30)"


def main() -> int:
    # Load the Mac's summarize/_boot_mean_ci/wilson for methodology parity.
    spec = importlib.util.spec_from_file_location(
        "_atn", REPO / "scripts" / "audit_topbin_netcost.py"
    )
    atn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atn)
    summarize = atn.summarize

    chunks = [GE25[i::N_WORKERS] for i in range(N_WORKERS)]
    print(f"launching {N_WORKERS} collectors over {len(GE25)} dates...", flush=True)
    recs: list[dict] = []
    diag_tot = {"n_ranked": 0, "n_no_fwd_spot": 0, "ranker_drops_total": 0}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for rs, dg in ex.map(_collect_chunk, chunks):
            recs.extend(rs)
            diag_tot["n_ranked"] += dg.get("n_ranked", 0)
            diag_tot["n_no_fwd_spot"] += dg.get("n_no_fwd_spot", 0)
            diag_tot["ranker_drops_total"] += dg.get("ranker_drops_total", 0)
    print(f"collected {len(recs)} records; diag={diag_tot}", flush=True)

    # persist raw records FIRST (never lose compute)
    (SCRATCH / "r11_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8"
    )

    for r in recs:
        r["fband"] = fine_band(r["vix"])

    def cell(rows: list[dict]) -> dict:
        s = summarize(rows)
        s["n_dates"] = len({r["as_of"] for r in rows})
        return s

    bands = ["calm (<20)", "lowelev (20-25)", "highelev (25-30)", "crisis (>=30)"]
    report: dict = {"by_fine_band": {}, "policy_ab": {}, "meta": {}}
    for b in bands:
        grp = [r for r in recs if r["fband"] == b]
        tb = [r for r in grp if r["top_bin"]]
        report["by_fine_band"][b] = {
            "all": cell(grp),
            "top_bin": cell(tb),
            "top_bin_would_trade": cell([r for r in tb if r["would_trade"]]),
        }

    # ---- R11 policy A/B: net $/contract of the FLAGGED top-bin set per policy ----
    tb_all = [r for r in recs if r["top_bin"]]

    def flagged(pred) -> dict:
        f = [r for r in tb_all if pred(r["vix"])]
        return cell(f)

    report["policy_ab"] = {
        "current_R11_gt25": flagged(lambda v: v > 25.0),  # what R11 sizes down today
        "bandaware_25to30_only": flagged(lambda v: 25.0 < v < 30.0),  # spare >=30
        "w3_elevated_20to30": flagged(lambda v: 20.0 <= v < 30.0),
        "ge30_carveout_test": flagged(lambda v: v >= 30.0),  # DECISIVE: over-fire?
    }
    report["meta"] = {
        "question": "Does R11 (VIX>25, prob_profit>0.90) over-fire into a "
        "well-calibrated VIX>=30 band, or miss a net-costly band? (#442)",
        "grid": "dense weekly VIX>=25 (28 dates >=30, 40 dates 25-30); <25 bands "
        "are EMPTY by construction here — cite W6 (calm +$266 / elevated +$125, "
        "both net-positive) for the bands R11 does not fire on.",
        "n_total": len(recs),
        "collection_diag": diag_tot,
        "machinery": "scripts/audit_topbin_netcost.collect (authoritative ranker + "
        "_forward_replay_realized_pnl, full friction); re-banded by raw VIX-at-entry.",
        "top_bin_prob": 0.90,
        "R11_live_threshold": "VIX>25.0 AND prob_profit>0.90 (candidate_dossier.py)",
    }

    out = SCRATCH / "r11_band_split.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- console table -----------------------------------------------------------
    print("\n=== TOP-BIN (prob_profit>0.90) net $/contract by fine VIX-at-entry band ===")
    for b in bands:
        tb = report["by_fine_band"][b]["top_bin"]
        if tb.get("n"):
            print(
                f"{b:<18} n={tb['n']:>4} dates={tb['n_dates']:>2}  "
                f"mean_full=${tb['mean_pnl_full']:>9.2f} CI{tb['mean_pnl_full_ci95']}  "
                f"win={tb['win_rate_gross']}  fc={tb['mean_forecast_prob_profit']}  "
                f"gap={tb['calibration_gap_pp']}pp"
            )
    print("\n=== R11 POLICY A/B (flagged top-bin set net $/contract) ===")
    for k, v in report["policy_ab"].items():
        if v.get("n"):
            print(
                f"{k:<24} n={v['n']:>4} dates={v['n_dates']:>2}  "
                f"mean_full=${v['mean_pnl_full']:>9.2f} CI{v['mean_pnl_full_ci95']}  "
                f"win={v['win_rate_gross']} gap={v['calibration_gap_pp']}pp"
            )
    print(f"\nJSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
