"""Driver for the parameter-OOS validation gate (E5).

Two phases, split so an analysis bug never forces a costly engine re-run:

    # 1) EXPENSIVE one-time engine pass -> committed rank-table fixture CSV
    python scripts/run_parameter_oos.py build

    # 2) FAST pure-numpy analysis -> committed snapshot JSON
    python scripts/run_parameter_oos.py analyze

``build`` runs the production ranker over a sampled date grid and forward-
replays each ranked row (see ``backtests.parameter_oos.build_rank_table``).
``analyze`` reads the fixture, cuts the leakage-certified train/holdout split,
and writes the walk-forward + parameter-holdout snapshot.

The engine's production defaults are NEVER mutated: every alternative parameter
value is applied offline to the captured ``ev_raw`` column (CLAUDE.md §2 /
task invariant 3).  Nothing here trades or ranks.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# scripts/ direct invocation lacks the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests import parameter_oos as poos  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical configuration — pinned so build and analyze agree, and so the
# snapshot fingerprint records exactly what produced it.
# ---------------------------------------------------------------------------

SNAPSHOT_ID = "param_oos_regime_24t"
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "param_oos" / "rank_table_24t.csv"
SNAPSHOT_PATH = _REPO_ROOT / "backtests" / "regression" / "snapshots" / f"{SNAPSHOT_ID}.json"

CONFIG = {
    "universe": "UNIVERSE_24",
    "sample_start": "2020-06-01",
    "sample_end": "2025-06-30",
    "every_n_bdays": 8,
    "dte_target": 35,
    "delta_target": 0.25,
    "top_n": 100,
    "seed": 42,
    # Leakage-certified split (embargo > max DTE horizon; per-row proof in the
    # snapshot's leakage_certificate).
    "train_end": "2023-06-30",
    "holdout_start": "2023-08-20",
    "n_walk_forward_folds": 5,
}


def _universe() -> list[str]:
    from backtests.regression.universes import UNIVERSE_24

    return list(UNIVERSE_24)


def build() -> None:
    """Run the engine pass and persist the raw rank-table fixture."""
    dates = poos.sample_business_days(
        CONFIG["sample_start"], CONFIG["sample_end"], CONFIG["every_n_bdays"]
    )
    print(
        f"[build] {CONFIG['universe']} × {len(dates)} sampled days "
        f"({CONFIG['sample_start']}..{CONFIG['sample_end']} every "
        f"{CONFIG['every_n_bdays']} bdays)",
        flush=True,
    )
    table = poos.build_rank_table(
        tickers=_universe(),
        sample_dates=dates,
        dte_target=CONFIG["dte_target"],
        delta_target=CONFIG["delta_target"],
        top_n=CONFIG["top_n"],
    )
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIXTURE_PATH, index=False)
    resolved = int(table["realized_pnl"].notna().sum())
    print(
        f"[build] wrote {FIXTURE_PATH} — {len(table)} rows, {resolved} resolved "
        f"(realized_pnl not null)",
        flush=True,
    )


def _fingerprint() -> dict:
    from backtests.regression._common import (
        connector_data_sha256,
        ohlcv_sha256,
        treasury_sha256,
        vol_iv_sha256,
    )

    return {
        **CONFIG,
        "data_csv_sha256": ohlcv_sha256(),
        "vol_iv_sha256": vol_iv_sha256(),
        "treasury_sha256": treasury_sha256(),
        "connector_data_sha256": connector_data_sha256(),
        "option_premium_rail": "pinned_off",
        "generated_at": datetime.now(UTC).isoformat(),
    }


def analyze() -> dict:
    """Read the fixture, cut the leakage-certified split, write the snapshot."""
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"{FIXTURE_PATH} missing — run `build` first.")
    table = pd.read_csv(FIXTURE_PATH)

    # Phase 1 — walk-forward out-of-window scorecard (fixed shipped params).
    folds = poos.rolling_folds(table, n_folds=CONFIG["n_walk_forward_folds"])

    # Phase 2 — parameter hold-out (regime overlay re-fit on train only).
    part = poos.make_partition(
        table, train_end=CONFIG["train_end"], holdout_start=CONFIG["holdout_start"]
    )
    holdout_report = poos.parameter_holdout_report(part)

    # Overall fixed-parameter scorecard (all resolved rows) for context.
    overall = poos.scorecard(table, signal_col="ev_dollars")
    overall_ev_raw = poos.scorecard(table, signal_col="ev_raw")

    payload = {
        "snapshot_id": SNAPSHOT_ID,
        "fingerprint": _fingerprint(),
        "overall": {"ev_dollars": overall, "ev_raw": overall_ev_raw},
        "walk_forward_folds": folds,
        "parameter_holdout": holdout_report,
    }

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")
    print(f"[analyze] wrote {SNAPSHOT_PATH}", flush=True)
    return payload


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "analyze"
    if mode == "build":
        build()
    elif mode == "analyze":
        payload = analyze()
        # Human-readable summary.
        pr = payload["parameter_holdout"]
        print("\n=== Parameter hold-out (regime overlay) ===", flush=True)
        print(f"  leakage: {pr['leakage_certificate']}", flush=True)
        print(f"  train_n={pr['train_n']}  holdout_n={pr['holdout_n']}", flush=True)
        for name, v in pr["variants"].items():
            print(
                f"  {name:24s} train_rho={v['train_rho']:+.4f}  "
                f"holdout_rho={v['holdout_rho']:+.4f}",
                flush=True,
            )
        print(
            f"  optimism_gap(scalars)={pr['optimism_gap_regime_scalars']:+.4f}  "
            f"optimism_gap(gamma)={pr['optimism_gap_tilt_exponent']:+.4f}",
            flush=True,
        )
        print(
            f"  refit_beats_shipped_on_holdout={pr['refit_beats_shipped_on_holdout']}  "
            f"overlay_adds_oos_value={pr['overlay_adds_oos_value']}",
            flush=True,
        )
    elif mode == "both":
        build()
        analyze()
    else:
        print(f"unknown mode {mode!r}; use build | analyze | both", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
