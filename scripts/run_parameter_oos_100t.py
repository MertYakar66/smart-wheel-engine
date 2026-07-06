"""Driver for the 100-name parameter-OOS replication (extends PR #484).

Reuses the shared analysis library ``backtests.parameter_oos`` verbatim — the
24-name (#484) and 100-name runs share the SAME leakage/re-fit/scorecard code so
the two are directly comparable; only the universe, cadence, and significance
treatment differ. The one methodological upgrade for daily sampling lives in the
library too (``per_date_cross_sectional_rho`` + ``cluster_bootstrap_ci``): daily
as_of dates create heavily OVERLAPPING forward windows and recur the same names,
so raw row count massively overstates independent trials — every significance
figure is a date-CLUSTERED bootstrap over as_of dates, never a naive z on the
inflated pooled N.

    # 1) EXPENSIVE daily engine pass -> committed rank-table fixture (checkpointed
    #    per 6-month batch so a partial capture is salvageable + honestly labelled)
    python scripts/run_parameter_oos_100t.py build

    # 2) FAST pure-numpy analysis -> committed snapshot JSON
    python scripts/run_parameter_oos_100t.py analyze

Engine defaults are NEVER mutated: alternative parameter values are applied
offline to the captured ``ev_raw`` column (CLAUDE.md §2 / task invariant 3).
Nothing here trades or ranks.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests import parameter_oos as poos  # noqa: E402

SNAPSHOT_ID = "param_oos_regime_100t"
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "param_oos" / "rank_table_100t.csv"
SNAPSHOT_PATH = _REPO_ROOT / "backtests" / "regression" / "snapshots" / f"{SNAPSHOT_ID}.json"

CONFIG = {
    "universe": "UNIVERSE_100",
    "sample_start": "2020-06-01",
    "sample_end": "2025-06-30",
    "every_n_bdays": 1,  # daily — the finest cadence; ~935 as_of dates
    "dte_target": 35,
    "delta_target": 0.25,
    "top_n": 200,  # > 100 so no ranked row is truncated
    "seed": 42,
    # Same leakage-certified split as #484 (24t) for direct comparability.
    "train_end": "2023-06-30",
    "holdout_start": "2023-08-20",
    "n_walk_forward_folds": 5,
    "robustness_splits": [
        ["2022-12-30", "2023-02-20"],
        ["2023-06-30", "2023-08-20"],
        ["2023-12-29", "2024-02-20"],
        ["2024-06-28", "2024-08-19"],
    ],
    # S34 in-sample reference window + rho (docs/BACKTEST_REGRESSION_CAMPAIGN.md).
    "s34_window": ["2022-01-03", "2024-12-31"],
    "s34_in_sample_rho": 0.3130,
    # E3 dominant single-name (docs/BACKTEST_REGRESSION_CAMPAIGN.md E3).
    "e3_dominant_name": "BKNG",
    "bootstrap_n": 2000,
    "bootstrap_seed": 12345,
    # Moving-block bootstrap block length (chronological dates). The 35-DTE
    # option horizon is ~25 trading days, so daily rank dates within ~25 days
    # share most of their forward path — the block breaks that serial dependence
    # and widens the CI to what the data actually support (vs the tighter,
    # optimistic block_len=1). ~25 trading days ≈ round(35 × 252/365).
    "bootstrap_block_len": 25,
}


def _universe() -> list[str]:
    from backtests.regression.universes import UNIVERSE_100

    return list(UNIVERSE_100)


def _six_month_batches(dates: list[date]) -> list[list[date]]:
    """Split an ordered date list into contiguous ~6-month batches (checkpoint
    granularity — a killed daily run keeps every completed batch)."""
    if not dates:
        return []
    batches: list[list[date]] = []
    cur: list[date] = [dates[0]]
    anchor = dates[0]
    for d in dates[1:]:
        if (d.year - anchor.year) * 12 + (d.month - anchor.month) >= 6:
            batches.append(cur)
            cur = []
            anchor = d
        cur.append(d)
    if cur:
        batches.append(cur)
    return batches


def build() -> None:
    """Daily engine pass, checkpointed per 6-month batch (append to the CSV)."""
    dates = poos.sample_business_days(
        CONFIG["sample_start"], CONFIG["sample_end"], CONFIG["every_n_bdays"]
    )
    batches = _six_month_batches(dates)
    print(
        f"[build100] {CONFIG['universe']} × {len(dates)} DAILY as_of dates "
        f"({CONFIG['sample_start']}..{CONFIG['sample_end']}) in {len(batches)} batches",
        flush=True,
    )
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Resume: skip batches already fully covered by an existing partial CSV.
    done_dates: set[str] = set()
    if FIXTURE_PATH.exists():
        try:
            done_dates = set(pd.read_csv(FIXTURE_PATH, usecols=["date"])["date"].astype(str))
            print(f"[build100] resume: {len(done_dates)} as_of dates already captured", flush=True)
        except Exception:
            done_dates = set()
    wrote_header = FIXTURE_PATH.exists() and bool(done_dates)
    for bi, batch in enumerate(batches):
        batch = [d for d in batch if d.isoformat() not in done_dates]
        if not batch:
            print(f"[build100] batch {bi + 1}/{len(batches)} already done — skip", flush=True)
            continue
        print(
            f"[build100] batch {bi + 1}/{len(batches)}: {batch[0]}..{batch[-1]} "
            f"({len(batch)} days)",
            flush=True,
        )
        tbl = poos.build_rank_table(
            tickers=_universe(),
            sample_dates=batch,
            dte_target=CONFIG["dte_target"],
            delta_target=CONFIG["delta_target"],
            top_n=CONFIG["top_n"],
        )
        tbl.to_csv(FIXTURE_PATH, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        print(
            f"[build100] batch {bi + 1} appended {len(tbl)} rows "
            f"({int(tbl['realized_pnl'].notna().sum())} resolved)",
            flush=True,
        )
    total = pd.read_csv(FIXTURE_PATH)
    print(
        f"[build100] DONE — {FIXTURE_PATH} : {len(total)} rows, "
        f"{int(total['realized_pnl'].notna().sum())} resolved, "
        f"{total['date'].nunique()} distinct as_of dates",
        flush=True,
    )


def _fingerprint(table: pd.DataFrame) -> dict:
    from backtests.regression._common import (
        connector_data_sha256,
        ohlcv_sha256,
        treasury_sha256,
        vol_iv_sha256,
    )

    return {
        **CONFIG,
        "actual_sample_dates": int(table["date"].nunique()),
        "actual_first_date": str(table["date"].min()),
        "actual_last_date": str(table["date"].max()),
        "data_csv_sha256": ohlcv_sha256(),
        "vol_iv_sha256": vol_iv_sha256(),
        "treasury_sha256": treasury_sha256(),
        "connector_data_sha256": connector_data_sha256(),
        "option_premium_rail": "pinned_off",
        "generated_at": datetime.now(UTC).isoformat(),
    }


def analyze() -> dict:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"{FIXTURE_PATH} missing — run `build` first.")
    table = pd.read_csv(FIXTURE_PATH)

    folds = poos.rolling_folds(table, n_folds=CONFIG["n_walk_forward_folds"])
    part = poos.make_partition(
        table, train_end=CONFIG["train_end"], holdout_start=CONFIG["holdout_start"]
    )
    holdout_report = poos.parameter_holdout_report(part)
    robustness = poos.split_robustness_report(
        table, [tuple(s) for s in CONFIG["robustness_splits"]]
    )

    overall = poos.scorecard(table, signal_col="ev_dollars")
    overall_ev_raw = poos.scorecard(table, signal_col="ev_raw")

    # --- independence-corrected significance (the daily-sampling upgrade) ---
    # Per-date cross-sectional rank-rho: "does ev_dollars order TODAY's menu?"
    # aggregated over genuinely independent date-level draws.
    bl = CONFIG["bootstrap_block_len"]
    xsec = poos.per_date_cross_sectional_rho(table, signal_col="ev_dollars")
    xsec_ci = poos.cluster_bootstrap_ci(
        table, stat="cross_sectional", signal_col="ev_dollars",
        n_boot=CONFIG["bootstrap_n"], seed=CONFIG["bootstrap_seed"], block_len=bl,
    )
    pooled_ci = poos.cluster_bootstrap_ci(
        table, stat="pooled", signal_col="ev_dollars",
        n_boot=CONFIG["bootstrap_n"], seed=CONFIG["bootstrap_seed"], block_len=bl,
    )
    # Holdout pooled + cross-sectional with moving-BLOCK CI (the decisive OOS #),
    # plus a naive block_len=1 contrast to show how much the block widens it.
    holdout_tbl = part.holdout
    holdout_pooled_ci = poos.cluster_bootstrap_ci(
        holdout_tbl, stat="pooled", signal_col="ev_dollars",
        n_boot=CONFIG["bootstrap_n"], seed=CONFIG["bootstrap_seed"], block_len=bl,
    )
    holdout_pooled_ci_naive = poos.cluster_bootstrap_ci(
        holdout_tbl, stat="pooled", signal_col="ev_dollars",
        n_boot=CONFIG["bootstrap_n"], seed=CONFIG["bootstrap_seed"], block_len=1,
    )
    holdout_xsec_ci = poos.cluster_bootstrap_ci(
        holdout_tbl, stat="cross_sectional", signal_col="ev_dollars",
        n_boot=CONFIG["bootstrap_n"], seed=CONFIG["bootstrap_seed"], block_len=bl,
    )

    # --- S34 reconciliation: rho on the in-sample S34 window (2022-2024) ---
    s0, s1 = CONFIG["s34_window"]
    d = pd.to_datetime(table["date"]).dt.date
    s34_mask = (d >= date.fromisoformat(s0)) & (d <= date.fromisoformat(s1))
    s34_sub = table[s34_mask]
    s34_recon = {
        "window": CONFIG["s34_window"],
        "s34_in_sample_rho": CONFIG["s34_in_sample_rho"],
        "our_pooled_rho": poos.scorecard(s34_sub, signal_col="ev_dollars")["rho"],
        "our_cross_sectional": poos.per_date_cross_sectional_rho(s34_sub, "ev_dollars"),
        "our_pooled_ci": poos.cluster_bootstrap_ci(
            s34_sub, stat="pooled", signal_col="ev_dollars",
            n_boot=CONFIG["bootstrap_n"], seed=CONFIG["bootstrap_seed"], block_len=bl,
        ),
        "n": int(s34_sub["realized_pnl"].notna().sum()),
    }

    # --- E3 robustness: drop dominant name + leave-one-name-out ---
    e3 = poos.dominant_name_robustness(
        table,
        holdout_tbl,
        dominant=CONFIG["e3_dominant_name"],
        signal_col="ev_dollars",
    )

    payload = {
        "snapshot_id": SNAPSHOT_ID,
        "fingerprint": _fingerprint(table),
        "overall": {"ev_dollars": overall, "ev_raw": overall_ev_raw},
        "independence_corrected": {
            "pooled_rho": overall["rho"],
            "pooled_rho_ci": pooled_ci,
            "cross_sectional_mean_rho": xsec["mean_rho"],
            "cross_sectional_median_rho": xsec["median_rho"],
            "cross_sectional_n_dates": xsec["n_dates"],
            "cross_sectional_rho_ci": xsec_ci,
            "holdout_pooled_rho": holdout_report["variants"]["shipped"]["holdout_rho"],
            "holdout_pooled_ci": holdout_pooled_ci,
            "holdout_pooled_ci_naive_block1": holdout_pooled_ci_naive,
            "holdout_cross_sectional_ci": holdout_xsec_ci,
            "block_len": bl,
        },
        "walk_forward_folds": folds,
        "parameter_holdout": holdout_report,
        "split_robustness": robustness,
        "s34_reconciliation": s34_recon,
        "e3_robustness": e3,
    }

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")
    print(f"[analyze100] wrote {SNAPSHOT_PATH}", flush=True)
    return payload


def _summary(payload: dict) -> None:
    ic = payload["independence_corrected"]
    pr = payload["parameter_holdout"]
    s34 = payload["s34_reconciliation"]
    e3 = payload["e3_robustness"]
    print("\n=== 100-name parameter-OOS ===", flush=True)
    print(f"  sampling: {payload['fingerprint']['actual_sample_dates']} daily as_of dates "
          f"{payload['fingerprint']['actual_first_date']}..{payload['fingerprint']['actual_last_date']}", flush=True)
    print(f"  pooled rho={ic['pooled_rho']:+.4f}  CI95={ic['pooled_rho_ci']['ci95']}", flush=True)
    print(f"  cross-sectional mean rho={ic['cross_sectional_mean_rho']:+.4f} "
          f"(n_dates={ic['cross_sectional_n_dates']})  CI95={ic['cross_sectional_rho_ci']['ci95']}", flush=True)
    print(f"  HOLDOUT pooled rho={ic['holdout_pooled_rho']:+.4f}  CI95={ic['holdout_pooled_ci']['ci95']}", flush=True)
    print(f"  HOLDOUT x-sec CI95={ic['holdout_cross_sectional_ci']['ci95']}", flush=True)
    print(f"  optimism_gap(scalars)={pr['optimism_gap_regime_scalars']:+.4f}", flush=True)
    print(f"  S34 recon: in-sample {s34['s34_in_sample_rho']:+.3f} vs our 2022-2024 pooled "
          f"{s34['our_pooled_rho']:+.4f} (CI95 {s34['our_pooled_ci']['ci95']})", flush=True)
    print(f"  E3: full rho={e3['full_rho']:+.4f}  drop-{e3['dominant']} rho={e3['drop_dominant_rho']:+.4f}  "
          f"LOO rho range=[{e3['loo_min_rho']:+.4f},{e3['loo_max_rho']:+.4f}]", flush=True)


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "analyze"
    if mode == "build":
        build()
    elif mode == "analyze":
        _summary(analyze())
    elif mode == "both":
        build()
        _summary(analyze())
    else:
        print(f"unknown mode {mode!r}; use build | analyze | both", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
