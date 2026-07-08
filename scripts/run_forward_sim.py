"""Drive the distributional forward simulated-portfolio track.

Reads the persisted artifacts of a WheelTracker-driven forward book (produced
by ``backtests.regression._common.run_backtest`` / ``run_backtest_multi_friction``
with ``output_dir``), derives a strategy-return series from the tracker's
equity curve, projects the full outcome distribution with
:mod:`engine.sim_portfolio` (Monte Carlo equity fan + terminal / drawdown
distributions + correlation-to-1 copula tail), reconciles the MC median with
the deterministic backtest NAV, and persists the report to the SIM namespace.

This is a **reporting** driver — it never constructs or mutates an EV verdict
and touches none of the decision-layer trio (CLAUDE.md §2). It only READS the
backtest's persisted state and the connector's OHLCV, then writes simulated
artifacts to a namespace that is separate from real IBKR data
(``data_processed/ibkr/`` is owned by the Dashboard terminal and never
touched here).

SIM namespace (invariant 4): output goes to ``$SWE_SIM_DATA_DIR`` if set, else
``data_processed/sim/`` — which is gitignored, so simulated output is never
committed.

Usage
-----
Analyse an already-generated backtest dir::

    python scripts/run_forward_sim.py \
        --output-dir data_processed/sim/primary_2024 \
        --start 2024-01-02 --end 2024-12-31 \
        --report-name primary_2024

Run a fresh backtest and then analyse it (``--run`` + a universe)::

    python scripts/run_forward_sim.py --run \
        --tickers AAPL MSFT NVDA JPM GS XOM UNH JNJ HD PG KO CAT \
        --start 2024-01-02 --end 2024-12-31 --capital 200000 \
        --report-name primary_2024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ direct invocation lacks the repo root on sys.path (tests->scripts
# sys.path trap) — add it before importing engine.*.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.sim_portfolio import build_sim_report  # noqa: E402


def _sim_dir() -> Path:
    """Resolve the SIM output namespace (env override or the gitignored default)."""
    env = os.environ.get("SWE_SIM_DATA_DIR")
    base = Path(env) if env else _REPO_ROOT / "data_processed" / "sim"
    return base


def _load_backtest_artifacts(output_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Load ``tracker_state.json`` + ``rank_log.csv`` from a backtest dir."""
    state_path = output_dir / "tracker_state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"tracker_state.json not found in {output_dir}. Run the backtest with "
            f"output_dir set (run_backtest(..., output_dir=...)) or pass --run."
        )
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    rank_log = pd.DataFrame()
    rl_path = output_dir / "rank_log.csv"
    if rl_path.exists():
        rank_log = pd.read_csv(rl_path)
    return state, rank_log


def _traded_book(state: dict) -> dict[str, int]:
    """Names actually traded (short put opened) → position count.

    Counts closed positions with a real put premium plus still-open positions
    that carry a put. Closed records lack ``put_strike`` (schema), so notional
    is reconstructed from rank_log strikes in :func:`_book_weights`.
    """
    counts: dict[str, int] = {}
    for rec in state.get("closed_positions", []):
        if (rec.get("put_premium") or 0) > 0:
            t = rec.get("ticker", "")
            if t:
                counts[t] = counts.get(t, 0) + 1
    for t, pos in state.get("positions", {}).items():
        if (pos.get("put_premium") or 0) and pos.get("state") != "no_position":
            counts[t] = counts.get(t, 0) + 1
    return counts


def _book_weights(state: dict, rank_log: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Per-name short-put notional weights for the copula overlay.

    ``notional[name] = median(strike from rank_log) * 100 * position_count``.
    Short-put books are long the underlying, so weights are positive.

    A name with no rank_log strike falls back to the **book-median strike**
    (× 100 × count), NOT a raw position count — mixing a count-scale weight
    (~1-3) with notional-scale weights (~tens of thousands) would silently
    near-zero the fallback name after the copula's sum-of-abs normalization.
    Only when NO name has any rank_log strike do all names use position count
    (a consistent scale). Note the median strike is over every day the name was
    ranked, a benign approximation for a reporting weight — the correlation
    matrix (not the weights) drives the tail headline, and
    ``tail_amplification`` is scale-free.
    """
    counts = _traded_book(state)
    if not counts:
        return {}, "none_traded"
    med_strike: dict[str, float] = {}
    if not rank_log.empty and {"ticker", "strike"}.issubset(rank_log.columns):
        for t, grp in rank_log.groupby("ticker"):
            s = pd.to_numeric(grp["strike"], errors="coerce")
            s = s[s > 0]
            if len(s):
                med_strike[str(t)] = float(s.median())

    if not med_strike:
        # No strike data anywhere → consistent count-scale weights.
        return {t: float(cnt) for t, cnt in counts.items()}, "position_count"

    # Book-median strike keeps fallback names on the same notional scale.
    book_median_strike = float(np.median(list(med_strike.values())))
    weights: dict[str, float] = {}
    used_fallback = False
    for t, cnt in counts.items():
        strike = med_strike.get(t)
        if strike is None:
            strike = book_median_strike
            used_fallback = True
        weights[t] = strike * 100.0 * cnt
    basis = "short_put_notional_strike_x100_x_count"
    if used_fallback:
        basis += "_with_book_median_fallback"
    return weights, basis


def _per_name_returns(
    tickers: list[str], start: str, end: str
) -> tuple[dict[str, np.ndarray], str]:
    """Daily close-to-close returns per name over [start, end] from the connector.

    Returns date-aligned arrays (inner-joined on trading dates, NaNs dropped)
    plus the provider class actually selected (logged — silent provider
    selection is a recurring bug).
    """
    os.environ.setdefault("SWE_DATA_PROVIDER", "bloomberg")
    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()
    conn = runner.connector
    provider = type(conn).__name__

    series: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            df = conn.get_ohlcv(t, start_date=start, end_date=end)
        except Exception:
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        d = df.copy()
        # Index on date if present so names align on the same trading calendar.
        if "date" in d.columns:
            d = d.set_index(pd.to_datetime(d["date"]))
        series[t] = pd.to_numeric(d["close"], errors="coerce")

    if not series:
        return {}, provider

    # Jointly align on the shared trading calendar BEFORE differencing so the
    # copula sees date-matched returns (per-column dropna would give unequal
    # lengths that the copula then tail-truncates → misaligned dates →
    # spuriously low cross-name correlation).
    frame = pd.DataFrame(series).sort_index()
    rets = frame.pct_change().dropna(how="any")  # rows where every name has a return
    out: dict[str, np.ndarray] = {}
    for t in rets.columns:
        col = rets[t].to_numpy(dtype=float)
        if len(col) >= 2:
            out[t] = col
    return out, provider


def _run_backtest(args) -> Path:
    """Optionally generate the backtest artifacts before analysis."""
    from backtests.regression._common import run_backtest

    out = _sim_dir() / args.report_name
    print(f"[run_forward_sim] running backtest -> {out}", flush=True)
    run_backtest(
        capital=args.capital,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        friction_level=args.friction,
        top_n=len(args.tickers),
        max_new_per_day=args.max_new_per_day,
        dte_target=args.dte_target,
        delta_target=args.delta_target,
        output_dir=out,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=str, default=None, help="Existing backtest artifacts dir")
    p.add_argument("--run", action="store_true", help="Run a fresh backtest first")
    p.add_argument("--tickers", nargs="+", default=None, help="Universe (with --run)")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--capital", type=float, default=200_000.0)
    p.add_argument("--friction", type=str, default="full", choices=["none", "bid_ask", "full"])
    p.add_argument("--max-new-per-day", type=int, default=3, dest="max_new_per_day")
    p.add_argument("--dte-target", type=int, default=35, dest="dte_target")
    p.add_argument("--delta-target", type=float, default=0.25, dest="delta_target")
    p.add_argument("--n-simulations", type=int, default=10_000, dest="n_simulations")
    p.add_argument("--block-size", type=int, default=21, dest="block_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-name", type=str, default="forward_sim", dest="report_name")
    args = p.parse_args(argv)

    if args.run:
        if not args.tickers:
            p.error("--run requires --tickers")
        output_dir = _run_backtest(args)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        p.error("pass --output-dir <dir> or --run --tickers ...")

    state, rank_log = _load_backtest_artifacts(output_dir)

    equity_curve = state.get("equity_curve", [])
    initial_capital = float(state.get("initial_capital", args.capital))
    realized_final_nav = (
        float(equity_curve[-1]["portfolio_value"])
        if equity_curve
        else float(state.get("cash", initial_capital))
    )

    # Copula inputs: traded book weights + per-name returns over the window.
    weights, weight_basis = _book_weights(state, rank_log)
    per_name_returns, provider = _per_name_returns(list(weights.keys()), args.start, args.end)
    print(
        f"[run_forward_sim] provider={provider} traded_names={len(weights)} "
        f"names_with_returns={len(per_name_returns)} weight_basis={weight_basis}",
        flush=True,
    )

    try:
        report = build_sim_report(
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            realized_final_nav=realized_final_nav,
            per_name_returns=per_name_returns or None,
            weights=weights or None,
            n_simulations=args.n_simulations,
            block_size=args.block_size,
            seed=args.seed,
            label=args.report_name,
        )
    except ValueError as exc:
        # Too few equity marks for the block bootstrap (empty/short book).
        # Emit an insufficient-history report instead of crashing so the SIM
        # namespace still records what happened.
        n_marks = len(equity_curve)
        print(
            f"[run_forward_sim] insufficient history: {n_marks} equity marks, "
            f"block_size={args.block_size} -> {exc}",
            flush=True,
        )
        sim_out = _sim_dir() / args.report_name
        sim_out.mkdir(parents=True, exist_ok=True)
        with open(sim_out / "mc_forward_sim.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": args.report_name,
                    "status": "insufficient_history",
                    "reason": str(exc),
                    "n_equity_marks": n_marks,
                    "block_size": args.block_size,
                    "engine_measured": {
                        "kind": "engine-measured",
                        "initial_capital": initial_capital,
                        "realized_final_nav": realized_final_nav,
                    },
                },
                f,
                indent=2,
                default=str,
            )
        return 0

    # Provenance — pull the backtest fingerprint if present.
    fingerprint = {}
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            fingerprint = json.load(f).get("fingerprint", {})
    report["provenance"] = {
        "source_output_dir": str(output_dir),
        "window": {"start": args.start, "end": args.end},
        "provider": provider,
        "weight_basis": weight_basis,
        "backtest_fingerprint": fingerprint,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    # Persist to the SIM namespace.
    sim_out = _sim_dir() / args.report_name
    sim_out.mkdir(parents=True, exist_ok=True)

    report_path = sim_out / "mc_forward_sim.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # portfolio_history-shaped artifact (SIMULATED — mirrors a dashboard
    # portfolio_history payload but from the deterministic backtest path).
    history_path = sim_out / "sim_portfolio_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "simulated",
                "label": args.report_name,
                "as_of": args.end,
                "initial_capital": initial_capital,
                "final_nav": realized_final_nav,
                "history": equity_curve,
            },
            f,
            indent=2,
            default=str,
        )

    # Bands-only file for a future chart panel.
    bands_path = sim_out / "mc_equity_bands.json"
    with open(bands_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "model",
                "band_days": report["model"]["band_days"],
                "equity_bands": report["model"]["equity_bands"],
                "quantiles": report["model"]["quantiles"],
            },
            f,
            indent=2,
            default=str,
        )

    # Console summary.
    rec = report["reconciliation"]
    model = report["model"]
    print("\n=== Forward simulated-portfolio report ===", flush=True)
    print(f"  wrote: {report_path}", flush=True)
    print(f"  wrote: {history_path}", flush=True)
    print(f"  wrote: {bands_path}", flush=True)
    print(
        f"  engine-measured realized final NAV: ${realized_final_nav:,.0f} "
        f"({report['engine_measured']['realized_total_return']:+.2%})",
        flush=True,
    )
    print(
        f"  model MC median terminal return: {model['median_return']:+.2%} "
        f"(p5 {model['terminal_return_quantiles']['p5']:+.2%} / "
        f"p95 {model['terminal_return_quantiles']['p95']:+.2%})",
        flush=True,
    )
    print(
        f"  RECONCILIATION: in_band={rec['in_band']} median_close={rec['median_close']} "
        f"gap={rec['median_abs_pct_gap']:.2f}% -> reconciled={rec['reconciled']}",
        flush=True,
    )
    if "correlation_tail" in report:
        ct = report["correlation_tail"]
        emp = ct.get("empirical", {})
        sve = ct.get("stress_vs_empirical", {})
        if not emp.get("skipped"):
            print(
                f"  copula EMPIRICAL: t_cvar={emp['t_cvar']:.4f} "
                f"tail_amplification(t/gauss)={emp['tail_amplification']:.3f} "
                f"verdict={emp['verdict']}",
                flush=True,
            )
            print(
                f"  copula CORR->1 STRESS: t_cvar worsens x{sve['t_cvar_multiple']:.2f} "
                f"vs realized correlation (mean|corr|={ct['mean_abs_correlation']:.2f})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
