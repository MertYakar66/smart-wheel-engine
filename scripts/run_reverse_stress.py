"""Driver for the V5 reverse-stress harness (plan doc section 8).

    # V5-a — worst admissible book, both variants (minutes, offline)
    python scripts/run_reverse_stress.py search --table .../tail_table_100t.csv

    # V5-b — assignment wave + levered counterfactual (minutes)
    python scripts/run_reverse_stress.py margin --table .../tail_table_100t.csv

    # V5-b-full (plan §10.2) — fresh full-menu rank at each crisis eve,
    # intrinsic A/A control + entry-IV BSM TV marking, IV bracket {1,1.5,2}
    python scripts/run_reverse_stress.py margin-full --table .../tail_table_100t.csv

Measurement-only (CLAUDE.md section 2). Outputs land under
``$SWE_VALIDATION_DIR`` or the gitignored
``data_processed/validation/reverse_stress/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests import reverse_stress as rs  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "reverse_stress"
)
DEFAULT_TABLE = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "tail_exceedance"
    / "tail_table_100t.csv"
)


def _conn():
    from engine.data_connector import MarketDataConnector

    return MarketDataConnector()


def _write(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[reverse_stress] wrote {p}", flush=True)


def cmd_search(args: argparse.Namespace) -> int:
    table = pd.read_csv(args.table)
    conn = _conn()
    sectors = rs.load_sector_map(conn, sorted(table["ticker"].unique()))
    print(f"[reverse_stress] search: {len(table)} rows, {table['date'].nunique()} dates")
    report: dict = {"generated_at": datetime.now(UTC).isoformat(), "table": str(args.table)}
    for top_bin in (False, True):
        r = rs.reverse_stress_search(table, sectors, top_bin_only=top_bin)
        report[r["variant"]] = r
        print(f"\n=== V5-a worst admissible book [{r['variant']}] ===")
        print(
            f"  dates={r['n_dates']}  ruin(>= {rs.RUIN_PCT:.0%} NAV)={r['n_ruin_dates']}  "
            f"loss%NAV q50/q90/q99 = "
            + " / ".join(f"{r['loss_pct_quantiles'][q]:.3f}" for q in ("0.5", "0.9", "0.99"))
        )
        for b in r["worst"][:5]:
            print(
                f"    {b['date']}  loss {b['loss_pct_nav']:6.1%}  "
                f"modeled-cvar x{b['realized_over_modeled_cvar']:5.2f}  "
                f"names {b['n_names']:3d}  vix {b['vix_entry']:5.1f}"
            )
        print(f"  by_vix_band: {r['by_vix_band']}")
    _write(Path(args.out_dir), "reverse_stress_search_100t.json", report)
    return 0


def cmd_margin(args: argparse.Namespace) -> int:
    table = pd.read_csv(args.table)
    conn = _conn()
    report: dict = {"generated_at": datetime.now(UTC).isoformat(), "windows": {}}
    print("\n=== V5-b assignment wave + levered counterfactual ===")
    dates = sorted(table["date"].unique())
    for eve in rs.CRISIS_EVES:
        usable = [d for d in dates if d <= eve]
        if not usable:
            print(f"  {eve}: no capture rows at/before eve — skipped")
            continue
        entry = usable[-1]
        book = rs.build_saturated_book(table[table["date"] == entry])
        out = rs.replay_assignment_wave(book, conn, eve=entry)
        report["windows"][eve] = {"entry_date_used": entry, **out}
        if "error" in out:
            print(f"  {eve}: {out['error']}")
            continue
        lev = ", ".join(
            f"x{m}: call@{v['first_call_bday']}" for m, v in out["levered_counterfactual"].items()
        )
        print(
            f"  {eve} (entry {entry}): n={out['n_positions']} "
            f"trough {out['trough_liquidation_pct_nav']:+.1%} NAV @bday {out['trough_bday']}, "
            f"terminal {out['terminal_pct_nav']:+.1%}, assigned {out['assignment_fraction']:.0%}, "
            f"levered [{lev}]"
        )
    _write(Path(args.out_dir), "reverse_stress_margin_100t.json", report)
    return 0


#: §10.2 frozen constants — the IV bracket and the trough clause cut.
IV_MULTS = (1.0, 1.5, 2.0)
CLAUSE_PCT = 0.20
COVID_EVE = "2020-02-19"


def cmd_margin_full(args: argparse.Namespace) -> int:
    """V5-b-full (plan §10.2): full-ranking book at each crisis eve, intrinsic
    A/A control + entry-IV BSM time-value marking across the IV bracket."""
    from engine.wheel_runner import WheelRunner

    table = pd.read_csv(args.table)
    tickers = sorted(table["ticker"].astype(str).unique())
    runner = WheelRunner()
    conn = runner.connector
    print(
        f"[reverse_stress] margin-full: {len(tickers)} tickers, "
        f"connector={type(conn).__name__}, iv_mults={IV_MULTS}"
    )
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "spec": {
            "iv_mults": list(IV_MULTS),
            "clause_pct": CLAUSE_PCT,
            "top_n": 100,
            "min_ev_dollars": -1e9,
            "dte_target": 35,
            "delta_target": 0.25,
        },
        "windows": {},
    }
    sanity_violations: list[str] = []
    print("\n=== V5-b-full: full-menu assignment wave, TV-marked (plan §10.2) ===")
    for eve in rs.CRISIS_EVES:
        frame = runner.rank_candidates_by_ev(
            tickers=tickers,
            dte_target=35,
            delta_target=0.25,
            contracts=1,
            top_n=100,
            min_ev_dollars=-1e9,
            as_of=eve,
            include_diagnostic_fields=True,
        )
        if frame is None or len(frame) == 0:
            report["windows"][eve] = {"error": "empty rank frame"}
            print(f"  {eve}: empty rank frame — skipped")
            continue
        book = rs.build_saturated_book(frame)
        collateral = float(sum(p["collateral"] for p in book))
        legs: dict = {"intrinsic": rs.replay_assignment_wave(book, conn, eve=eve)}
        for m in IV_MULTS:
            legs[f"tv_x{m:g}"] = rs.replay_assignment_wave(
                book, conn, eve=eve, marking="time_value", iv_mult=m
            )
        # Expectation-1 sanity: TV trough damage >= intrinsic, every leg.
        base = legs["intrinsic"].get("trough_liquidation_pct_nav")
        for m in IV_MULTS:
            tv = legs[f"tv_x{m:g}"].get("trough_liquidation_pct_nav")
            if base is not None and tv is not None and tv < base - 1e-12:
                sanity_violations.append(f"{eve} tv_x{m:g} {tv:.4f} < intrinsic {base:.4f}")
        report["windows"][eve] = {
            "n_ranked": int(len(frame)),
            "n_book": len(book),
            "collateral_used": collateral,
            "budget_saturation": collateral / rs.NAV,
            "legs": legs,
        }
        troughs = " ".join(
            f"{k}={v.get('trough_liquidation_pct_nav', float('nan')):.1%}"
            for k, v in legs.items()
            if "error" not in v
        )
        print(
            f"  {eve}: ranked {len(frame)}, book {len(book)} names "
            f"({collateral / rs.NAV:.0%} budget) | troughs: {troughs} | "
            f"assigned {legs['intrinsic'].get('assignment_fraction', float('nan')):.0%}"
        )

    # Frozen §10.2 expectation-3 disposition on the COVID window.
    disposition = "INSUFFICIENT"
    covid = report["windows"].get(COVID_EVE, {})
    legs = covid.get("legs", {})
    if legs and all("error" not in legs.get(k, {"error": 1}) for k in ("tv_x1", "tv_x2")):
        t10 = legs["tv_x1"]["trough_liquidation_pct_nav"]
        t20 = legs["tv_x2"]["trough_liquidation_pct_nav"]
        if t10 >= CLAUSE_PCT:
            disposition = "ESTABLISHED"
        elif t20 < CLAUSE_PCT:
            disposition = "RETIRED_PRACTICAL"
        else:
            disposition = "OPEN_BRACKET_STRADDLES"
    report["clause_disposition"] = {
        "rule": "x1.0 >= 20% -> ESTABLISHED; x2.0 < 20% -> RETIRED_PRACTICAL; else OPEN",
        "verdict": disposition,
    }
    report["sanity_violations"] = sanity_violations
    print(f"\n  trough-clause disposition (frozen rule): {disposition}")
    if sanity_violations:
        print(f"  !! SANITY VIOLATIONS (harness bug, do not record): {sanity_violations}")
    _write(Path(args.out_dir), "reverse_stress_margin_full_100t.json", report)
    return 0 if not sanity_violations else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["search", "margin", "margin-full"])
    p.add_argument("--table", default=str(DEFAULT_TABLE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)
    return {"search": cmd_search, "margin": cmd_margin, "margin-full": cmd_margin_full}[args.phase](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
