"""Driver for the V3 parameter-plateau sweep (plan doc section 6).

Phases:

    # activation diagnostic — decides the section-6.0 NOT-POWERED gates
    python scripts/run_param_plateau.py activation

    # V3-a — R11 cutoff sweep, offline on a V1 tail table (seconds)
    python scripts/run_param_plateau.py r11 [--table .../tail_table_100t.csv]

    # V3-b — F4 engine sweeps, one axis at a time (~13 min per value)
    python scripts/run_param_plateau.py f4-build --axis threshold
    python scripts/run_param_plateau.py f4-build --axis cap
    python scripts/run_param_plateau.py f4-analyze --axis threshold
    python scripts/run_param_plateau.py f4-analyze --axis cap

Measurement-only (CLAUDE.md section 2): read-only ranker calls, rail pinned
off, patches scoped inside the harness, no production default changes.
Outputs land under ``$SWE_VALIDATION_DIR`` or the gitignored
``data_processed/validation/param_plateau/``.  Existing sweep tables are
reused (delete a CSV to force a rebuild) so interrupted sweeps resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests import param_plateau as pp  # noqa: E402
from backtests.parameter_oos import sample_business_days  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "param_plateau"
)
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "bloomberg"
V1_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "tail_exceedance"
)

#: Sweep grid — breach-rich and onset-inclusive (COVID entry), every 10
#: bdays on UNIVERSE_24; block_len 3 scales the moving-block CI to the ~25
#: trading-day horizon at this cadence.
GRID_START = "2020-02-03"
GRID_EVERY_N = 10
BLOCK_LEN = 3
DTE_TARGET = 35
DELTA_TARGET = 0.25

#: Activation diagnostic — a light unpatched pass over a spread of dates.
ACTIVATION_DATES = (
    "2020-04-15",
    "2021-06-15",
    "2022-06-15",
    "2023-03-15",
    "2024-01-16",
    "2024-08-06",
    "2025-04-15",
    "2026-02-17",
)


#: Universe per config. Grid/cadence/block are shared (the 6.2 sweep grid).
CONFIGS = {
    "24t": {"universe": "UNIVERSE_24"},
    "100t": {"universe": "UNIVERSE_100"},
}


def _universe(config: str) -> list[str]:
    from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100

    name = CONFIGS[config]["universe"]
    return list(UNIVERSE_24 if name == "UNIVERSE_24" else UNIVERSE_100)


def _resolve_sample_end(dte_target: int) -> str:
    from engine.data_connector import MarketDataConnector

    frontier = MarketDataConnector().get_data_frontier()
    if frontier is None:
        return (date.today() - timedelta(days=dte_target + 7)).isoformat()
    return (pd.Timestamp(frontier).date() - timedelta(days=dte_target + 7)).isoformat()


def _sweep_grid() -> list[date]:
    return sample_business_days(GRID_START, _resolve_sample_end(DTE_TARGET), GRID_EVERY_N)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[param_plateau] wrote {path}", flush=True)


def _table_path(out_dir: Path, axis: str, value: float, config: str) -> Path:
    # 24t keeps the original un-prefixed names (artifacts from the first
    # sweep run predate the config dimension); other configs are prefixed.
    tag = "" if config == "24t" else f"{config}_"
    _param, _grid, shipped = pp.F4_AXES[axis]
    if value == shipped:
        return out_dir / f"f4_{tag}shipped.csv"  # one baseline shared by both axes
    return out_dir / f"f4_{tag}{axis}_{value:g}.csv"


def _suffix(config: str) -> str:
    return "" if config == "24t" else f"_{config}"


def cmd_activation(args: argparse.Namespace) -> int:
    from backtests.regression._common import _option_premium_rail_pinned_off
    from engine.wheel_runner import WheelRunner

    with _option_premium_rail_pinned_off():
        runner = WheelRunner(data_dir=str(args.data_dir))
        _ = runner.connector
    frames = []
    for as_of in ACTIVATION_DATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = runner.rank_candidates_by_ev(
                tickers=_universe(args.config),
                dte_target=DTE_TARGET,
                delta_target=DELTA_TARGET,
                top_n=100,
                min_ev_dollars=-1e9,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
        frames.append(frame)
        print(f"[param_plateau] activation {as_of}: {0 if frame is None else len(frame)} rows")
    report = pp.activation_from_frames(frames)
    report["dates"] = list(ACTIVATION_DATES)
    report["config"] = args.config
    report["generated_at"] = datetime.now(UTC).isoformat()
    _write_json(Path(args.out_dir) / f"activation_report{_suffix(args.config)}.json", report)
    print(f"\n=== V3 activation diagnostic ({args.config}) ===")
    print(
        f"  rows={report['n_rows']}  gpd_fit_rate={report['gpd_fit_rate']:.4f}  "
        f"heavy_tail_rate={report['heavy_tail_rate']:.4f}  "
        f"n>=200 rate={report['n_scenarios_ge_200_rate']:.4f}"
    )
    print(f"  distribution mix: {report['distribution_source_mix']}")
    print(f"  VERDICT (floor {report['activation_floor']:.0%}): {report['verdict']}")
    return 0


def cmd_r11(args: argparse.Namespace) -> int:
    src = Path(args.table)
    if not src.exists():
        print(f"[param_plateau] no table at {src} — run the V1 build first", file=sys.stderr)
        return 2
    table = pd.read_csv(src)
    report = pp.r11_sweep(table)
    report["table"] = str(src)
    report["generated_at"] = datetime.now(UTC).isoformat()
    tag = src.stem.replace("tail_table_", "")
    _write_json(Path(args.out_dir) / f"r11_sweep_{tag}.json", report)
    print(f"\n=== V3-a R11 cutoff sweep ({tag}; n={report['n_rows']}) ===")
    print(
        "  vix_thr prob_thr  n_flag  flag_breach  unflag_breach   lift"
        "   flag_ocgap  unflag_ocgap  d_mean_realized"
    )
    for c in report["cells"]:
        mark = " *" if c["shipped"] else "  "
        fb = c.get("flagged_breach_rate", float("nan"))
        ub = c.get("unflagged_breach_rate", float("nan"))
        fg = c.get("flagged_overconfidence_gap", float("nan"))
        ug = c.get("unflagged_overconfidence_gap", float("nan"))
        dm = c.get("flagged_mean_realized", float("nan")) - c.get(
            "unflagged_mean_realized", float("nan")
        )
        print(
            f"{mark}{c['vix_threshold']:6.1f}  {c['prob_threshold']:5.2f}  "
            f"{c['n_flagged']:6d}  {fb:10.4f}  {ub:12.4f}  {c['lift']:6.2f}  "
            f"{fg:+10.4f}  {ug:+11.4f}  {dm:12.0f}$"
        )
    return 0


def cmd_f4_build(args: argparse.Namespace) -> int:
    axis = args.axis
    _param, grid, _shipped = pp.F4_AXES[axis]
    grid_dates = _sweep_grid()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[param_plateau] f4-build config={args.config} axis={axis} values={list(grid)} "
        f"grid={grid_dates[0]}..{grid_dates[-1]} ({len(grid_dates)} dates)",
        flush=True,
    )
    for value in grid:
        out = _table_path(out_dir, axis, value, args.config)
        if out.exists():
            print(f"[param_plateau] {axis}={value:g}: reusing {out.name}", flush=True)
            continue
        print(f"[param_plateau] {axis}={value:g}: engine pass...", flush=True)
        table = pp.build_f4_sweep_table(
            axis=axis,
            value=value,
            tickers=_universe(args.config),
            sample_dates=grid_dates,
            dte_target=DTE_TARGET,
            delta_target=DELTA_TARGET,
        )
        table.to_csv(out, index=False)
        print(f"[param_plateau] {axis}={value:g}: {len(table)} rows -> {out.name}", flush=True)
    return 0


def cmd_f4_analyze(args: argparse.Namespace) -> int:
    axis = args.axis
    _param, grid, _shipped = pp.F4_AXES[axis]
    out_dir = Path(args.out_dir)
    tables: dict[float, pd.DataFrame] = {}
    for value in grid:
        p = _table_path(out_dir, axis, value, args.config)
        if not p.exists():
            print(f"[param_plateau] missing {p} — run f4-build first", file=sys.stderr)
            return 2
        tables[value] = pd.read_csv(p)
    report = pp.f4_axis_report(axis, tables, block_len=BLOCK_LEN)
    report["config"] = args.config
    report["generated_at"] = datetime.now(UTC).isoformat()
    _write_json(out_dir / f"f4_{axis}_report{_suffix(args.config)}.json", report)
    print(f"\n=== V3-b F4 {axis} sweep ({args.config}; shipped {report['shipped']:g}) ===")
    print("   value  fire_rate  elev+crisis breach [n]      pooled   guard rho (top15)  ok")
    for val in report["verdict"]["values"]:
        e = report["per_value"][str(val)]
        pr, po, gr = e["primary_elev_crisis"], e["pooled"], e["guard_rho"]
        mark = " *" if val == report["shipped"] else "  "
        ok = "Y" if gr["xsec"]["mean_rho"] >= report["guard_shipped_ci_low"] else "N"
        print(
            f"{mark}{val:6g}  {e['fire_rate']:9.4f}  {pr['rate']:10.4f} [{pr['n']:5d}]  "
            f"{po['rate']:10.4f}  {gr['xsec']['mean_rho']:+15.3f}  {ok:>3}"
        )
    print(f"VERDICT: {report['verdict']['verdict']}")
    if report["verdict"]["dominated_by"]:
        print(f"  dominated by: {report['verdict']['dominated_by']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["activation", "r11", "f4-build", "f4-analyze"])
    p.add_argument("--config", choices=sorted(CONFIGS), default="24t")
    p.add_argument("--axis", choices=sorted(pp.F4_AXES), default="threshold")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--table",
        default=str(V1_OUT_DIR / "tail_table_24t.csv"),
        help="V1 tail table for the r11 phase",
    )
    args = p.parse_args(argv)
    return {
        "activation": cmd_activation,
        "r11": cmd_r11,
        "f4-build": cmd_f4_build,
        "f4-analyze": cmd_f4_analyze,
    }[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
