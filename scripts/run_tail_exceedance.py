"""Driver for the tail-risk exceedance validation harness.

Two phases, split so an analysis bug never forces a costly engine re-run
(the ``run_parameter_oos.py`` pattern):

    # 1) EXPENSIVE one-time engine pass -> tail table CSV
    python scripts/run_tail_exceedance.py build

    # 2) FAST pure-numpy analysis -> report JSON + console summary
    python scripts/run_tail_exceedance.py analyze

Measurement-only (CLAUDE.md section 2): the ranker is called read-only with
the option-premium rail pinned off; nothing here trades, gates, or mutates a
production default.  See ``backtests.tail_exceedance`` for the statistics and
their rationale.

Outputs land under ``$SWE_VALIDATION_DIR`` or the gitignored
``data_processed/validation/tail_exceedance/`` (files: ``tail_table_<id>.csv``,
``report_<id>.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# scripts/ direct invocation lacks the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests import tail_exceedance as tex  # noqa: E402
from backtests.parameter_oos import sample_business_days  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "tail_exceedance"
)

CONFIGS = {
    # In-sandbox / laptop-fast: 24-name universe, every 5th business day.
    "24t": {
        "universe": "UNIVERSE_24",
        "sample_start": "2022-01-03",
        "every_n_bdays": 5,
        "dte_target": 35,
        "delta_target": 0.25,
        "top_n": 100,
    },
    # Terminal full-scale: 100-name universe, every 2nd business day.
    "100t": {
        "universe": "UNIVERSE_100",
        "sample_start": "2020-06-01",
        "every_n_bdays": 2,
        "dte_target": 35,
        "delta_target": 0.25,
        "top_n": 100,
    },
}


def _universe(name: str) -> list[str]:
    from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100

    return list(UNIVERSE_24 if name == "UNIVERSE_24" else UNIVERSE_100)


def _resolve_sample_end(dte_target: int) -> str:
    """Latest entry date whose expiry resolves inside the data frontier.

    Frontier - (dte + 5 business-day settlement pad) so every captured row's
    held-to-expiry outcome exists; a capped grid beats NaN-realized rows.
    """
    from engine.data_connector import MarketDataConnector

    frontier = MarketDataConnector().get_data_frontier()
    if frontier is None:
        return (date.today() - timedelta(days=dte_target + 7)).isoformat()
    return (pd.Timestamp(frontier).date() - timedelta(days=dte_target + 7)).isoformat()


def cmd_build(args: argparse.Namespace) -> int:
    cfg = CONFIGS[args.config]
    sample_end = args.sample_end or _resolve_sample_end(cfg["dte_target"])
    dates = sample_business_days(cfg["sample_start"], sample_end, cfg["every_n_bdays"])
    print(
        f"[tail_exceedance] build config={args.config} universe={cfg['universe']} "
        f"grid={cfg['sample_start']}..{sample_end} every {cfg['every_n_bdays']} bdays "
        f"-> {len(dates)} dates",
        flush=True,
    )
    table = tex.build_tail_table(
        tickers=_universe(cfg["universe"]),
        sample_dates=dates,
        dte_target=cfg["dte_target"],
        delta_target=cfg["delta_target"],
        top_n=cfg["top_n"],
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tail_table_{args.config}.csv"
    table.to_csv(out, index=False)
    print(f"[tail_exceedance] wrote {len(table)} rows -> {out}", flush=True)
    return 0


def _fmt_quantile(name: str, q: dict) -> str:
    if q.get("n", 0) == 0:
        return f"  {name}: INSUFFICIENT (0 rows)"
    k, c = q["kupiec"], q["clustered"]
    cl = q["clustering"]
    return (
        f"  {name}: viol {k['rate']:.3f} vs nominal {q['nominal']:.2f} "
        f"(n={q['n']}, kupiec p={k['p_value']:.2e}, "
        f"clustered CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}], "
        f"clustering ac1={cl['autocorr']:.2f} p={cl['p_value']:.3f}) "
        f"-> {q['verdict']}"
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    src = out_dir / f"tail_table_{args.config}.csv"
    if not src.exists():
        print(f"[tail_exceedance] no table at {src} — run build first", file=sys.stderr)
        return 2
    table = pd.read_csv(src)
    meta = {
        "config": args.config,
        **CONFIGS[args.config],
        "table": str(src),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    report = tex.full_report(table, meta=meta)
    out = out_dir / f"report_{args.config}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== Tail-risk exceedance report ({args.config}) ===")
    print(f"rows: {report['rows_total']} captured, {report['rows_resolved']} resolved")
    for name in ("p25", "p50", "p75"):
        print(_fmt_quantile(name, report["quantiles"][name]))
    cv = report["cvar_5"]
    if cv.get("n", 0):
        c = cv["clustered"]
        print(
            f"  cvar_5 breach: {cv['rate']:.4f} vs bound {cv['bound']:.2f} "
            f"(n={cv['n']}, breaches={cv.get('breaches', 0)}, "
            f"binom p={cv['binom_p_one_sided']:.2e}, "
            f"clustered CI [{c['ci_low']:.4f}, {c['ci_high']:.4f}]) -> {cv['verdict']}"
        )
        if "severity" in cv:
            s = cv["severity"]
            print(
                f"    severity | mean excess ${s['mean_excess_dollars']:,.0f}, "
                f"median realized/cvar {s['median_realized_over_cvar']:.2f}x"
            )
        for k, v in cv.get("strata", {}).items():
            print(f"    stratum {k}: n={v['n']} rate={v['rate']:.4f}")
    pp = report["prob_profit_pooled"]
    print(
        f"  prob_profit pooled: observed {pp['observed_wins']:.0f} vs expected "
        f"{pp['expected_wins']:.0f} wins (n={pp['n']:.0f}, z={pp['z']:.2f}, "
        f"p={pp['p_value']:.2e})"
    )
    print(f"OVERALL: {report['overall_verdict']}")
    print(f"report -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["build", "analyze", "full"])
    p.add_argument("--config", choices=sorted(CONFIGS), default="24t")
    p.add_argument("--sample-end", default=None, help="override the auto frontier-capped end date")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)
    if args.phase in ("build", "full"):
        rc = cmd_build(args)
        if rc:
            return rc
    if args.phase in ("analyze", "full"):
        return cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
