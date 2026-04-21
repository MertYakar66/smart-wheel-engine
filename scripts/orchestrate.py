#!/usr/bin/env python3
"""
Smart Wheel Engine — Unified Daily Orchestrator.

One command to run the full daily workflow:

    python scripts/orchestrate.py morning    # Pre-market: data + news + EV ranking
    python scripts/orchestrate.py intraday   # Market hours: refresh quotes + re-rank
    python scripts/orchestrate.py evening    # After close: journal + calibration check
    python scripts/orchestrate.py full       # All three in sequence

Designed to be invoked by:
  - A human in the terminal
  - Claude Code (claude) in a chat session
  - OpenClaw / any MCP-connected agent
  - A cron job / systemd timer

Each stage produces a structured JSON output on stdout so an agent
can parse the results programmatically. Human-readable summaries go
to stderr. Exit code 0 = success, 1 = partial failure, 2 = critical.

Architecture:
  orchestrate.py
    ├── refresh_daily_data()    → yfinance + FRED (free sources)
    ├── run_news_pipeline()     → morning_run.py subprocess
    ├── run_ev_ranking()        → WheelRunner.rank_candidates_by_ev
    ├── run_regime_check()      → regime_detector + regime_hmm
    ├── run_calibration_check() → DriftDetector.check_calibration
    └── output_daily_brief()    → combined JSON summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _log(msg: str) -> None:
    """Human-readable log to stderr (agents read stdout JSON)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def _result(stage: str, status: str, data: dict | None = None, error: str = "") -> dict:
    return {
        "stage": stage,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data or {},
        "error": error,
    }


# ======================================================================
# Stage 1: Refresh daily market data from free sources
# ======================================================================
def refresh_daily_data(tickers: list[str] | None = None) -> dict:
    """Pull OHLCV from yfinance, VIX + rates from FRED."""
    _log("Stage 1: Refreshing daily market data...")
    results = {"ohlcv": {}, "vix": None, "rates": None}

    try:
        import yfinance as yf

        if tickers is None:
            # Load universe from the engine if available
            try:
                from engine.data_connector import MarketDataConnector

                conn = MarketDataConnector()
                tickers = conn.get_universe()[:50]
            except Exception:
                tickers = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                    "JPM", "BAC", "WFC", "GS", "UNH", "JNJ", "LLY", "ABBV",
                    "XOM", "CVX", "PG", "KO", "HD", "MCD", "CAT", "HON", "GE",
                ]

        # Pull OHLCV for each ticker (last 5 days to catch gaps)
        _log(f"  Pulling OHLCV for {len(tickers)} tickers...")
        for ticker in tickers:
            try:
                df = yf.download(ticker, period="5d", progress=False)
                if df is not None and len(df) > 0:
                    out_dir = PROJECT_ROOT / "data_raw" / "ohlcv_daily"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out_dir / f"{ticker}.csv")
                    results["ohlcv"][ticker] = len(df)
            except Exception:
                pass

        _log(f"  Got OHLCV for {len(results['ohlcv'])} tickers")

        # Pull VIX
        try:
            vix = yf.download("^VIX", period="5d", progress=False)
            if vix is not None and len(vix) > 0:
                results["vix"] = round(float(vix["Close"].iloc[-1]), 2)
                _log(f"  VIX: {results['vix']}")
        except Exception:
            _log("  VIX pull failed")

    except ImportError:
        _log("  yfinance not installed — skipping OHLCV/VIX refresh")

    # Pull treasury rates from FRED (if fredapi available)
    try:
        import pandas_datareader.data as pdr

        rates = pdr.get_data_fred(
            ["DGS3MO", "DGS2", "DGS10"],
            start=date.today().replace(day=1),
        )
        if rates is not None and len(rates) > 0:
            latest = rates.dropna().iloc[-1]
            results["rates"] = {
                "3m": round(float(latest.get("DGS3MO", 5.0)), 2),
                "2y": round(float(latest.get("DGS2", 4.5)), 2),
                "10y": round(float(latest.get("DGS10", 4.0)), 2),
            }
            _log(f"  Rates: {results['rates']}")
    except Exception:
        _log("  FRED rates pull skipped (pandas_datareader not available)")

    return _result("refresh_data", "ok", results)


# ======================================================================
# Stage 2: Run news pipeline
# ======================================================================
def run_news_pipeline(
    visible: bool = False,
    scrape_only: bool = False,
    output_json: bool = True,
) -> dict:
    """Run the news pipeline via morning_run.py subprocess."""
    _log("Stage 2: Running news pipeline...")

    morning_run = PROJECT_ROOT / "morning_run.py"
    if not morning_run.exists():
        return _result("news", "skip", error="morning_run.py not found")

    cmd = [sys.executable, str(morning_run)]
    if scrape_only:
        cmd.append("--scrape-only")
    if visible:
        cmd.append("--visible")
    if output_json:
        cmd.append("--json")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode == 0:
            try:
                stories = json.loads(proc.stdout) if proc.stdout.strip() else []
            except json.JSONDecodeError:
                stories = []
            _log(f"  Got {len(stories) if isinstance(stories, list) else '?'} stories")
            return _result("news", "ok", {"story_count": len(stories) if isinstance(stories, list) else 0, "stories": stories})
        else:
            _log(f"  News pipeline returned {proc.returncode}")
            return _result("news", "partial", error=proc.stderr[:500])
    except subprocess.TimeoutExpired:
        return _result("news", "timeout", error="News pipeline timed out (300s)")
    except Exception as e:
        return _result("news", "error", error=str(e)[:200])


# ======================================================================
# Stage 3: Run EV ranking
# ======================================================================
def run_ev_ranking(
    top_n: int = 15,
    dte_target: int = 35,
    delta_target: float = 0.25,
    use_dealer: bool = False,
) -> dict:
    """Run the EV ranker and return the top candidates."""
    _log("Stage 3: Running EV ranking...")

    try:
        from engine.wheel_runner import WheelRunner

        runner = WheelRunner()
        df = runner.rank_candidates_by_ev(
            dte_target=dte_target,
            delta_target=delta_target,
            top_n=top_n,
            min_ev_dollars=0.0,
            include_diagnostic_fields=True,
            use_dealer_positioning=use_dealer,
        )

        if df is None or df.empty:
            _log("  No candidates passed the EV filter")
            return _result("ev_ranking", "ok", {"count": 0, "candidates": []})

        candidates = df.to_dict(orient="records")
        _log(f"  {len(candidates)} candidates ranked by ev_per_day")

        # Log top 5 to stderr for human readability
        for i, c in enumerate(candidates[:5]):
            _log(
                f"  #{i+1} {c.get('ticker'):>5s}  "
                f"EV/day=${c.get('ev_per_day', 0):+.2f}  "
                f"prob={c.get('prob_profit', 0):.0%}  "
                f"strike={c.get('strike', 0):.0f}  "
                f"source={c.get('distribution_source', '?')}"
            )

        return _result("ev_ranking", "ok", {"count": len(candidates), "candidates": candidates})

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return _result("ev_ranking", "error", error=str(e)[:300])


# ======================================================================
# Stage 4: Regime check
# ======================================================================
def run_regime_check() -> dict:
    """Check current market regime via rule-based + HMM."""
    _log("Stage 4: Regime check...")

    regime_info = {}
    try:
        from engine.data_connector import MarketDataConnector

        conn = MarketDataConnector()

        # VIX regime
        vix = conn.get_vix_regime()
        if vix:
            regime_info["vix"] = vix.get("vix", 0)
            regime_info["vix_regime"] = vix.get("regime", "unknown")
            _log(f"  VIX: {regime_info['vix']:.1f} ({regime_info['vix_regime']})")

        # HMM regime (if OHLCV for SPY available)
        try:
            import numpy as np
            from engine.regime_hmm import GaussianHMM

            spy = conn.get_ohlcv("SPY")
            if spy is not None and len(spy) > 200:
                log_rets = np.diff(np.log(spy["close"].values))
                hmm = GaussianHMM(n_states=4, n_iter=30, random_state=42)
                fit = hmm.fit(log_rets[-504:])
                probs = hmm.predict_proba(log_rets[-504:])
                last_probs = probs[-1]
                state_idx = int(np.argmax(last_probs))
                regime_info["hmm_state"] = fit.state_labels[state_idx]
                regime_info["hmm_confidence"] = round(float(last_probs[state_idx]), 3)
                regime_info["hmm_multiplier"] = round(hmm.position_multiplier(last_probs), 3)
                _log(
                    f"  HMM: {regime_info['hmm_state']} "
                    f"(conf={regime_info['hmm_confidence']:.0%}, "
                    f"mult={regime_info['hmm_multiplier']:.2f})"
                )
        except Exception as e:
            regime_info["hmm_error"] = str(e)[:100]
            _log(f"  HMM regime check failed: {e}")

    except Exception as e:
        return _result("regime", "error", error=str(e)[:200])

    return _result("regime", "ok", regime_info)


# ======================================================================
# Stage 5: Calibration check (evening)
# ======================================================================
def run_calibration_check() -> dict:
    """Run model calibration and drift checks."""
    _log("Stage 5: Calibration check...")

    try:
        from ml.model_governance import DriftDetector

        # Check if there's a recent predictions log to validate
        pred_log = PROJECT_ROOT / "data" / "predictions_log.csv"
        if not pred_log.exists():
            _log("  No predictions log found — skip calibration")
            return _result("calibration", "skip", error="No predictions_log.csv")

        import pandas as pd

        df = pd.read_csv(pred_log)
        if len(df) < 30:
            _log(f"  Only {len(df)} predictions logged — need 30+ for calibration")
            return _result("calibration", "skip", error=f"Only {len(df)} rows")

        preds = df["predicted_prob"].tolist()
        actuals = df["actual_outcome"].astype(int).tolist()
        result = DriftDetector.check_calibration(preds, actuals)

        _log(
            f"  Brier={result['brier_score']:.4f}  "
            f"ECE={result['ece']:.4f}  "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
        return _result("calibration", "ok" if result["passed"] else "fail", result)

    except Exception as e:
        return _result("calibration", "error", error=str(e)[:200])


# ======================================================================
# Workflow runners
# ======================================================================
def run_morning(args: argparse.Namespace) -> list[dict]:
    """Pre-market workflow: data + news + EV ranking + regime."""
    results = []
    results.append(refresh_daily_data())
    results.append(run_news_pipeline(
        visible=args.visible,
        scrape_only=args.scrape_only,
    ))
    results.append(run_ev_ranking(
        top_n=args.top_n,
        dte_target=args.dte,
        delta_target=args.delta,
        use_dealer=args.dealer,
    ))
    results.append(run_regime_check())
    return results


def run_intraday(args: argparse.Namespace) -> list[dict]:
    """Market-hours: quick data refresh + re-rank."""
    results = []
    results.append(refresh_daily_data())
    results.append(run_ev_ranking(
        top_n=args.top_n,
        dte_target=args.dte,
        delta_target=args.delta,
        use_dealer=args.dealer,
    ))
    return results


def run_evening(_args: argparse.Namespace) -> list[dict]:
    """After-close: calibration + journal review."""
    results = []
    results.append(run_calibration_check())
    return results


def run_full(args: argparse.Namespace) -> list[dict]:
    """Complete daily workflow — all stages."""
    results = []
    results.extend(run_morning(args))
    results.append(run_calibration_check())
    return results


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Smart Wheel Engine — Unified Daily Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/orchestrate.py morning          # Pre-market: data + news + EV ranking
  python scripts/orchestrate.py morning --top-n 20 --dealer  # With dealer positioning
  python scripts/orchestrate.py intraday         # Quick refresh + re-rank
  python scripts/orchestrate.py evening          # Calibration check
  python scripts/orchestrate.py full             # Everything
  python scripts/orchestrate.py morning --json   # Machine-readable output for agents
        """,
    )

    parser.add_argument(
        "stage",
        choices=["morning", "intraday", "evening", "full"],
        help="Which workflow stage to run",
    )
    parser.add_argument("--top-n", type=int, default=15, dest="top_n")
    parser.add_argument("--dte", type=int, default=35)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--dealer", action="store_true", help="Enable dealer positioning")
    parser.add_argument("--visible", action="store_true", help="Show browser windows")
    parser.add_argument("--scrape-only", action="store_true", dest="scrape_only")
    parser.add_argument("--json", action="store_true", help="JSON output on stdout")

    args = parser.parse_args()

    _log(f"=== Smart Wheel Engine — {args.stage.upper()} ===")
    _log(f"Date: {date.today().isoformat()}")
    start = time.time()

    dispatch = {
        "morning": run_morning,
        "intraday": run_intraday,
        "evening": run_evening,
        "full": run_full,
    }

    results = dispatch[args.stage](args)

    elapsed = time.time() - start
    _log(f"=== Done in {elapsed:.1f}s ===")

    # Summary
    statuses = {r["stage"]: r["status"] for r in results}
    any_error = any(s in ("error", "fail") for s in statuses.values())

    summary = {
        "workflow": args.stage,
        "date": date.today().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "statuses": statuses,
        "overall": "partial" if any_error else "ok",
        "stages": results,
    }

    if args.json:
        # Machine-readable: full JSON on stdout
        print(json.dumps(summary, indent=2, default=str))
    else:
        # Human-readable: just the summary
        _log(f"Stages: {statuses}")
        if any_error:
            _log("⚠  Some stages had issues — check details above")
        else:
            _log("✓  All stages completed successfully")

        # Print top candidates for quick glance
        ev_stage = next((r for r in results if r["stage"] == "ev_ranking"), None)
        if ev_stage and ev_stage["data"].get("candidates"):
            _log("")
            _log("Top candidates:")
            for i, c in enumerate(ev_stage["data"]["candidates"][:5]):
                _log(
                    f"  #{i+1} {c.get('ticker', '?'):>5s}  "
                    f"EV/day=${c.get('ev_per_day', 0):+.2f}  "
                    f"prob={c.get('prob_profit', 0):.0%}  "
                    f"K={c.get('strike', 0):.0f}"
                )

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
