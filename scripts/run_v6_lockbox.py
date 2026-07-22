"""V6 — the lockbox spend. Every parameter is FROZEN here (plan doc §9).

    export SWE_DATA_PROVIDER=bloomberg
    python scripts/run_v6_lockbox.py run

One pre-registered deep-history read: 2007-01-03 -> 2009-06-30, PIT
universe incl. delisted names (max_universe=100), through the committed
survivorship harness. Validates SELECTION / REFUSAL / ASSIGNMENT behavior
only — premiums are synthetic BSM from the deep IV panels; NAV and dollar
P&L are NOT evidence (§2.4). The four H-verdicts are computed in code so
the run is reported whatever it says. Deterministic; a crash may be
restarted; the run is never re-parameterized.

Reporting hardened after the 2026-07-13 attempt-1/2 post-processing crash
(read NOT spent — no verdict computed, no report written): raw artifacts
are dumped before verdict computation and each H-verdict is
exception-captured, so the engine pass can no longer be lost to a
reporting bug. SPEC and the H-verdict functions are byte-identical to the
pinned pre-spend commit (1473857).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "v6_lockbox"
)

#: THE frozen spec (plan doc §9.0-§9.1). Do not add CLI overrides.
SPEC = {
    "start": "2007-01-03",
    "end": "2009-06-30",
    "capital": 1_000_000.0,
    "friction_level": "full",
    "top_n": 15,
    "max_new_per_day": 3,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
    "max_universe": 100,
    "rebalance_months": 3,
    "baseline_months": [f"2007-{m:02d}" for m in range(1, 13)],
    "grind_months": ["2008-10", "2008-11", "2008-12"],
    "cliff_end": "2008-09-12",  # Lehman eve
    "cliff_lookback_tdays": 10,
    "august": "2008-08",
    "wave_months": ["2008-09", "2008-10", "2008-11", "2008-12"],
}


def _month(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y-%m")


def h1_refusal_grind(rank_log: pd.DataFrame) -> dict:
    """Oct-Dec 2008 EV-positive rate <= 0.5x the 2007 monthly average."""
    df = rank_log.copy()
    df["month"] = _month(df["date"])
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
        "verdict": "PASS" if ratio <= 0.5 else "FAIL",
    }


def h2_cliff_lag(opens: pd.Series) -> dict:
    """Opens/day in the 10 tdays ending at the Lehman eve >= 0.7x August's.

    CONFIRMED = onset blindness generalizes; the falsifier is GOOD news.
    """
    d = pd.to_datetime(opens)
    aug = d[d.dt.strftime("%Y-%m") == SPEC["august"]]
    aug_days = pd.bdate_range("2008-08-01", "2008-08-29")
    aug_rate = len(aug) / len(aug_days)
    cliff_days = pd.bdate_range(end=SPEC["cliff_end"], periods=SPEC["cliff_lookback_tdays"])
    cliff = d[(d >= cliff_days[0]) & (d <= cliff_days[-1])]
    cliff_rate = len(cliff) / len(cliff_days)
    ratio = float(cliff_rate / aug_rate) if aug_rate > 0 else float("nan")
    verdict = (
        "INSUFFICIENT"
        if not np.isfinite(ratio)
        else ("CONFIRMED_BLIND" if ratio >= 0.7 else "FALSIFIED_GOOD_NEWS")
    )
    return {
        "august_opens_per_day": float(aug_rate),
        "cliff_window_opens_per_day": float(cliff_rate),
        "ratio": ratio,
        "verdict": verdict,
    }


def h3_assignment_wave(rank_log: pd.DataFrame) -> dict:
    """(a) >= 50% ITM-at-expiry among EV-positive ranked rows expiring in some
    Sep-Dec 2008 month. Schema-safe substrate: under the locked forward-replay
    convention, realized = (premium - intrinsic) x 100, so ITM (assignment)
    <=> realized_pnl < premium x 100 strictly. (b) is reported separately by
    the delisted-participant census in cmd_run."""
    df = rank_log.copy()
    df = df[pd.to_numeric(df["ev_dollars"], errors="coerce") > 0]
    df = df[np.isfinite(pd.to_numeric(df["realized_pnl"], errors="coerce"))]
    if df.empty:
        return {"verdict": "INSUFFICIENT", "n_rows": 0}
    df["month"] = df["expiration_date"].astype(str).str[:7]
    df["itm"] = pd.to_numeric(df["realized_pnl"], errors="coerce") < (
        pd.to_numeric(df["premium"], errors="coerce") * 100.0 - 1e-9
    )
    wave = df[df["month"].isin(SPEC["wave_months"])]
    if wave.empty:
        return {"verdict": "INSUFFICIENT", "n_rows": 0}
    monthly = wave.groupby("month")["itm"].agg(["mean", "count"])
    peak = float(monthly["mean"].max())
    return {
        "monthly_itm": {
            k: {"rate": float(r["mean"]), "n": int(r["count"])} for k, r in monthly.iterrows()
        },
        "peak_monthly_itm_rate": peak,
        "verdict": "PASS" if peak >= 0.5 else "FAIL",
    }


def h4_rank_rho(rank_log: pd.DataFrame) -> dict:
    """Report-only: per-date rho with block CI (synthetic-premium caveat)."""
    from backtests.parameter_oos import cluster_bootstrap_ci, per_date_cross_sectional_rho

    df = rank_log.dropna(subset=["realized_pnl"])
    if df.empty:
        return {"verdict": "REPORT_ONLY", "n": 0}
    return {
        "verdict": "REPORT_ONLY",
        "caveat": "synthetic BSM premiums — selection sanity only, never dollar evidence",
        "xsec": per_date_cross_sectional_rho(df, "ev_dollars"),
        "ci_block7": cluster_bootstrap_ci(
            df, stat="cross_sectional", signal_col="ev_dollars", n_boot=400, block_len=7
        ),
    }


def collect_entry_dates(*sources) -> pd.Series:
    """Entry dates across closed and still-open position records.

    Defensive on shape — the 2026-07-13 attempt-1/2 crash was the vehicle's
    legacy ``open_positions`` ``{ticker: state_string}`` map reaching a
    ``.get("entry_date")`` path. Accepts lists of dicts, DataFrames, or the
    legacy map (whose string values carry no dates and are skipped), and
    ignores anything else rather than dying after the engine pass.
    """
    dates: list[str] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, pd.DataFrame):
            source = source.to_dict("records")
        elif isinstance(source, dict):
            source = list(source.values())
        for p in source:
            if isinstance(p, dict) and p.get("entry_date"):
                dates.append(str(p.get("entry_date")))
    return pd.Series(dates, dtype="object")


def _safe(fn, *fn_args) -> dict:
    """One H-verdict, crash-proof: a bug in a verdict function is recorded in
    the report as ``verdict=ERROR`` with the traceback — the spend's report is
    written whatever happens."""
    try:
        return fn(*fn_args)
    except Exception:  # noqa: BLE001
        return {"verdict": "ERROR", "traceback": traceback.format_exc()}


def build_report(result: dict, delisted_participants: list, elapsed_seconds: float) -> dict:
    """Pure post-processing of the vehicle's return value -> the report dict.

    Kept separate from cmd_run (and exception-captured per H) so the verdict
    path is testable against the vehicle's exact return shape without a
    deep-history read.
    """
    rank_log = result["rank_log"]
    if not isinstance(rank_log, pd.DataFrame):
        rank_log = pd.DataFrame(rank_log)
    # Opens = entry dates across closed AND still-open positions (assigned
    # puts wheel onward, they do not close at expiry). Still-open entry dates
    # come from open_position_records; the legacy open_positions state map
    # carries no dates.
    opens = collect_entry_dates(result.get("closed_positions"), result.get("open_position_records"))
    return {
        "meta": {
            "spec": SPEC,
            "generated_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1),
        },
        "rows_ranked": int(len(rank_log)),
        "n_opens_counted": int(len(opens)),
        "H1_refusal_grind": _safe(h1_refusal_grind, rank_log),
        "H2_cliff_lag": (_safe(h2_cliff_lag, opens) if len(opens) else {"verdict": "INSUFFICIENT"}),
        "H3_assignment_wave": _safe(h3_assignment_wave, rank_log),
        "H3b_delisted_participants": {
            "n": len(delisted_participants),
            "tickers": delisted_participants[:20],
            "note": "PIT-only names whose history ends in-window; 0 is a valid outcome",
        },
        "H4_rank_rho": _safe(h4_rank_rho, rank_log),
        "metrics": result.get("metrics", {}),
    }


def cmd_run(args: argparse.Namespace) -> int:
    from backtests.survivorship import run_survivorship_backtest

    deep = _REPO_ROOT / "data" / "bloomberg" / "deep"
    if not deep.exists():
        print(
            f"[v6] PRECONDITION FAILED: {deep} absent — deep panels live on operator machines only",
            file=sys.stderr,
        )
        return 2
    print(
        f"[v6] LOCKBOX SPEND — spec frozen at plan §9: {json.dumps({k: v for k, v in SPEC.items() if not isinstance(v, list)})}",
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
        result["rank_log"] = rank_log
    print(
        f"[v6] engine pass done: {len(rank_log)} ranked rows in {elapsed / 60:.1f} min", flush=True
    )

    # Raw spend artifacts FIRST — the engine pass must never again be lost to
    # a post-processing crash; the H-verdicts are recomputable offline from
    # these (gitignored, operator machine only).
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_log.to_csv(out_dir / "v6_rank_log.csv.gz", index=False, compression="gzip")
    (out_dir / "v6_positions.json").write_text(
        json.dumps(
            {
                "closed_positions": result.get("closed_positions", []),
                "open_position_records": result.get("open_position_records", []),
                "open_positions": result.get("open_positions", {}),
                "metrics": result.get("metrics", {}),
            },
            indent=2,
            default=str,
        )
    )
    print(f"[v6] raw artifacts written to {out_dir}", flush=True)

    # Delisted-participant census (H3b, report-only): ranked tickers whose
    # deep-history OHLCV ends inside the window are PIT-only participants.
    delisted_participants: list[dict] = []
    try:
        from backtests.survivorship import make_deep_connector

        conn = make_deep_connector()
        for t in sorted(set(rank_log["ticker"].astype(str))):
            try:
                df_t = conn.get_ohlcv(t)
                if df_t is not None and not df_t.empty and str(df_t.index.max())[:10] < SPEC["end"]:
                    delisted_participants.append(
                        {"ticker": t, "last_bar": str(df_t.index.max())[:10]}
                    )
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        delisted_participants = [{"census_error": traceback.format_exc()}]

    report = build_report(result, delisted_participants, elapsed)
    out = out_dir / "v6_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[v6] wrote {out}", flush=True)
    print("\n=== V6 lockbox spend — H-verdicts ===")
    for h in ("H1_refusal_grind", "H2_cliff_lag", "H3_assignment_wave", "H4_rank_rho"):
        block = report[h]
        keys = {k: v for k, v in block.items() if not isinstance(v, dict)}
        print(f"  {h}: {keys}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["run"])
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
