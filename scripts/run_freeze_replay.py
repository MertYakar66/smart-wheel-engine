"""Driver for the V2 parameter freeze-replay validation harness (C1).

Phases (see ``docs/VALIDATION_PHASE_PLAN.md`` section 5 for the
pre-registered design):

    # V2-a — amnesia test: A/A determinism control + full-vs-truncated A/B
    python scripts/run_freeze_replay.py amnesia

    # V2-b — build / verify the C1 freeze snapshot fixture (committed)
    python scripts/run_freeze_replay.py snapshot
    python scripts/run_freeze_replay.py verify

    # V2-c — frozen-knowledge replay of the holdout grid + comparison
    python scripts/run_freeze_replay.py frozen-build     # expensive engine pass
    python scripts/run_freeze_replay.py compare          # fast, pure numpy

Measurement-only (CLAUDE.md section 2): read-only ranker calls, rail pinned
off, the freeze patch scoped inside the harness; nothing feeds back.  Run
artifacts land under ``$SWE_VALIDATION_DIR`` or the gitignored
``data_processed/validation/freeze_replay/``; the ONLY committed output is
the C1 fixture ``tests/fixtures/freeze_replay/freeze_snapshot_24t.json``
(derived deterministically from committed CSVs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests import freeze_replay as fz  # noqa: E402
from backtests.parameter_oos import sample_business_days  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "freeze_replay"
)
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "bloomberg"
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "freeze_replay" / "freeze_snapshot_24t.json"

#: Amnesia grid — five regime-spanning business days (calm / elevated /
#: crisis-adjacent / post-spike / near-frontier).
AMNESIA_DATES = ("2022-06-15", "2023-11-15", "2024-08-06", "2025-04-15", "2026-05-01")

#: V2-c configs — each reuses ITS V1 capture's grid phase (same start /
#: cadence as the corresponding run_tail_exceedance config) restricted to
#: the parameter_oos-canonical holdout, so the frozen table joins the
#: production V1 capture date-for-date.  ``block_len`` scales the
#: moving-block clustered CI to the ~25-trading-day option-horizon overlap
#: at each cadence (every-5-bday -> 7 dates; every-2-bday -> 13).
HOLDOUT_START = "2023-08-20"
DTE_TARGET = 35
DELTA_TARGET = 0.25

CONFIGS = {
    "24t": {
        "universe": "UNIVERSE_24",
        "grid_start": "2022-01-03",
        "every_n_bdays": 5,
        "block_len": 7,
    },
    "100t": {
        "universe": "UNIVERSE_100",
        "grid_start": "2020-02-03",
        "every_n_bdays": 2,
        "block_len": 13,
    },
}


def _universe(name: str) -> list[str]:
    from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100

    return list(UNIVERSE_24 if name == "UNIVERSE_24" else UNIVERSE_100)


def _universe_24() -> list[str]:
    return _universe("UNIVERSE_24")


def _resolve_sample_end(dte_target: int) -> str:
    """Frontier - (dte + 7d), the run_tail_exceedance convention."""
    from engine.data_connector import MarketDataConnector

    frontier = MarketDataConnector().get_data_frontier()
    if frontier is None:
        return (date.today() - timedelta(days=dte_target + 7)).isoformat()
    return (pd.Timestamp(frontier).date() - timedelta(days=dte_target + 7)).isoformat()


def _holdout_grid(cfg: dict) -> list[date]:
    end = _resolve_sample_end(DTE_TARGET)
    dates = sample_business_days(cfg["grid_start"], end, cfg["every_n_bdays"])
    return [d for d in dates if d.isoformat() >= HOLDOUT_START]


def _default_production_table(config: str) -> str:
    return str(
        Path(
            os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation"))
        )
        / "tail_exceedance"
        / f"tail_table_{config}.csv"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[freeze_replay] wrote {path}", flush=True)


def cmd_amnesia(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    report = fz.amnesia_report(
        data_dir=args.data_dir,
        tickers=_universe_24(),
        dates=list(args.dates),
        work_dir=out_dir,
    )
    report["generated_at"] = datetime.now(UTC).isoformat()
    _write_json(out_dir / "amnesia_report.json", report)
    print("\n=== V2-a amnesia test ===")
    for d in report["per_date"]:
        print(
            f"  {d['as_of']}: {d['verdict']} "
            f"({d['n_ranked_rows']} rows, {d['n_drops']} drops; "
            f"A/A identical={d['aa_control']['identical']}, "
            f"A/B identical={d['ab_truncated']['identical']})"
        )
        if d["verdict"] != "PASS":
            for ex in d["ab_truncated"].get("diff_examples", [])[:5]:
                print(f"      diff: {ex}")
    print(f"OVERALL: {report['overall_verdict']}")
    return 0 if report["overall_verdict"] == "PASS" else 1


def cmd_snapshot(args: argparse.Namespace) -> int:
    snap = fz.build_freeze_snapshot(args.data_dir, _universe_24(), cutoff=args.cutoff)
    snap["generated_at"] = datetime.now(UTC).isoformat()
    _write_json(FIXTURE_PATH, snap)
    n_hmm = sum(1 for v in snap["tickers"].values() if "hmm" in v)
    n_gpd = sum(1 for v in snap["tickers"].values() if "gpd" in v)
    print(f"\n=== V2-b snapshot @ {args.cutoff} ===")
    print(f"  {len(snap['tickers'])} tickers: {n_hmm} HMM fits, {n_gpd} GPD fits")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if not FIXTURE_PATH.exists():
        print(f"[freeze_replay] no fixture at {FIXTURE_PATH} — run snapshot first", file=sys.stderr)
        return 2
    snap = json.loads(FIXTURE_PATH.read_text())
    diff = fz.verify_freeze_snapshot(snap, args.data_dir)
    print(f"\n=== V2-b reproducibility verify (rtol={diff['rtol']}) ===")
    if diff["identical"]:
        print(f"  REPRODUCED: all {diff['n_tickers']} tickers match the committed fixture")
        return 0
    print(f"  DRIFT in {len(diff['mismatched'])}/{diff['n_tickers']} tickers:")
    for t, fields in list(diff["mismatched"].items())[:10]:
        print(f"    {t}: {fields[:6]}")
    return 1


def cmd_frozen_build(args: argparse.Namespace) -> int:
    cfg = CONFIGS[args.config]
    tickers = _universe(cfg["universe"])
    grid = _holdout_grid(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"frozen_tail_table_{args.config}.csv"
    if out.exists():
        # Resume path: the engine pass is deterministic given the pinned
        # config, so an existing table is reused rather than re-spent (~46
        # min at 100t). Delete the CSV to force a rebuild.
        table = pd.read_csv(out)
        print(f"[freeze_replay] reusing existing {out} ({len(table)} rows)", flush=True)
    else:
        print(
            f"[freeze_replay] frozen build: config={args.config} cutoff={args.cutoff} "
            f"grid={grid[0]}..{grid[-1]} ({len(grid)} dates, every {cfg['every_n_bdays']} bdays)",
            flush=True,
        )
        table = fz.build_frozen_tail_table(
            tickers=tickers,
            sample_dates=grid,
            cutoff=args.cutoff,
            dte_target=DTE_TARGET,
            delta_target=DELTA_TARGET,
            top_n=100,
        )
        table.to_csv(out, index=False)
        print(f"[freeze_replay] wrote {len(table)} rows -> {out}", flush=True)

    dates = sorted({str(d) for d in table["date"]})
    mults = fz.frozen_hmm_multipliers(args.data_dir, tickers, dates, cutoff=args.cutoff)
    mout = out_dir / f"frozen_hmm_multipliers_{args.config}.csv"
    mults.to_csv(mout, index=False)
    print(f"[freeze_replay] wrote {len(mults)} multiplier rows -> {mout}", flush=True)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = CONFIGS[args.config]
    out_dir = Path(args.out_dir)
    frozen_path = out_dir / f"frozen_tail_table_{args.config}.csv"
    mults_path = out_dir / f"frozen_hmm_multipliers_{args.config}.csv"
    prod_path = Path(args.production_table or _default_production_table(args.config))
    for p in (frozen_path, mults_path, prod_path):
        if not p.exists():
            print(
                f"[freeze_replay] missing {p} — run frozen-build (and V1 build) first",
                file=sys.stderr,
            )
            return 2
    frozen = pd.read_csv(frozen_path)
    mults = pd.read_csv(mults_path)
    production = pd.read_csv(prod_path)
    production = production[production["date"] >= HOLDOUT_START]

    report = fz.compare_frozen_vs_production(
        production, frozen, mults, cutoff=args.cutoff, block_len=cfg["block_len"]
    )
    report["meta"] = {
        "config": args.config,
        "cutoff": args.cutoff,
        "block_len": cfg["block_len"],
        "production_table": str(prod_path),
        "frozen_table": str(frozen_path),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(out_dir / f"freeze_replay_report_{args.config}.json", report)

    print(f"\n=== V2-c frozen replay vs production (cutoff {args.cutoff}) ===")
    print(
        f"rows: production {report['rows_production']}, frozen {report['rows_frozen']}, "
        f"joined {report['rows_joined']} over {report['n_common_dates']} dates"
    )
    print(f"  rank (per-date cross-sectional rho, mean [block-{cfg['block_len']} CI95]):")
    for name, entry in report["rank"].items():
        for tier in ("all", "top15", "top5"):
            x, c = entry[tier]["xsec"], entry[tier]["ci_block"]
            print(
                f"    {name:>28} {tier:>5}: {x['mean_rho']:+.3f} "
                f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] (n_dates={x['n_dates']})"
            )
    for variant in ("production", "frozen"):
        cv = report["risk"][variant]["cvar_5"]
        p25 = report["risk"][variant]["quantiles"]["p25"]
        print(
            f"  risk[{variant}]: cvar_5 breach {cv.get('rate', float('nan')):.4f} "
            f"({cv.get('verdict', '?')}), p25 viol "
            f"{p25.get('kupiec', {}).get('rate', float('nan')):.3f} ({p25.get('verdict', '?')})"
        )
    if "paired" in report:
        pr = report["paired"]
        print(
            f"  paired: median d_cvar5 {pr['median_d_cvar5_dollars']:+.1f}$, "
            f"median d_prob {pr['median_d_prob_profit']:+.4f}, "
            f"ev_raw ordering agreement rho {pr['ev_raw_ordering_agreement_mean_rho']:.3f}"
        )
    print("  drift (cvar_5 breach rate, production -> frozen):")
    for label, b in report["drift_by_months_since_cutoff"].items():
        print(
            f"    {label:>7}: {b['production']['cvar5_breach_rate']:.4f} -> "
            f"{b['frozen']['cvar5_breach_rate']:.4f} "
            f"(n {b['production']['cvar5_n']} / {b['frozen']['cvar5_n']})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["amnesia", "snapshot", "verify", "frozen-build", "compare"])
    p.add_argument("--config", choices=sorted(CONFIGS), default="24t")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--cutoff", default=fz.FREEZE_CUTOFF_DEFAULT)
    p.add_argument("--dates", nargs="+", default=list(AMNESIA_DATES), help="amnesia as_of dates")
    p.add_argument(
        "--production-table",
        default=None,
        help="V1 production capture used as the compare baseline "
        "(default: tail_table_<config>.csv in the V1 out dir)",
    )
    args = p.parse_args(argv)
    return {
        "amnesia": cmd_amnesia,
        "snapshot": cmd_snapshot,
        "verify": cmd_verify,
        "frozen-build": cmd_frozen_build,
        "compare": cmd_compare,
    }[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
