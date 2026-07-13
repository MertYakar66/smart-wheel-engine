"""Driver for the V5 reverse-stress harness (plan doc section 8).

    # V5-a — worst admissible book, both variants (minutes, offline)
    python scripts/run_reverse_stress.py search --table .../tail_table_100t.csv

    # V5-b — assignment wave + levered counterfactual (minutes)
    python scripts/run_reverse_stress.py margin --table .../tail_table_100t.csv

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["search", "margin"])
    p.add_argument("--table", default=str(DEFAULT_TABLE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)
    return {"search": cmd_search, "margin": cmd_margin}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
