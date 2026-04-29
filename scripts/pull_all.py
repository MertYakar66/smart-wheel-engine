#!/usr/bin/env python3
"""
One-command data refresh for smart-wheel-engine.

Runs every puller in dependency order, skips ones whose upstream is
unreachable, prints a clear per-step status line. Designed so you can run
it daily/weekly without remembering the individual script flags.

Order (why it matters):
  1. Vol-index family         (yfinance; regime detector needs it)
  2. Treasury yields          (yfinance; option pricer risk-free rate)
  3. Fundamentals snapshot    (yfinance; P/E, beta, sector)
  4. Earnings calendar        (yfinance; activates event gate)
  5. Theta IV-surface history (requires Theta Terminal UP — skipped if down)
  6. Theta options flow       (requires Theta Terminal UP — skipped if down)
  7. News sentiment           (needs POLYGON_API_KEY or FINNHUB_API_KEY — skipped if unset)
  8. Feature pipeline backfill (uses all of the above)

Quick start
-----------
    python scripts/pull_all.py               # full daily refresh, default settings
    python scripts/pull_all.py --dry-run     # print the plan without executing
    python scripts/pull_all.py --skip theta news  # skip specific steps
    python scripts/pull_all.py --only vol treasury  # run just these
    python scripts/pull_all.py --years 2 --workers 8

Exit code: 0 if every step that *should* have run either passed or was
explicitly skipped; 1 if any required step failed.
"""

from __future__ import annotations

import argparse
import io
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]


def _theta_up(host: str = "127.0.0.1", port: int = 25503) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


@dataclass
class Step:
    name: str
    script: str
    description: str
    needs_theta: bool = False
    needs_news_key: bool = False
    is_feature_backfill: bool = False

    def should_skip(self) -> tuple[bool, str]:
        if self.needs_theta and not _theta_up():
            return True, "Theta Terminal not reachable on 127.0.0.1:25503"
        if self.needs_news_key:
            has = any(os.environ.get(k) for k in
                      ("POLYGON_API_KEY", "FINNHUB_API_KEY", "BENZINGA_API_KEY"))
            if not has:
                return True, "no POLYGON/FINNHUB/BENZINGA API key in env"
        return False, ""


STEPS: list[Step] = [
    Step("vol_indices",
         "pull_vol_indices.py",
         "10 vol indices (VIX family + SKEW + VVIX + MOVE + OVX + GVZ)"),
    Step("treasury",
         "pull_treasury_yields_yf.py",
         "Treasury yield curve (3m/6m/2y/10y) → data/bloomberg/treasury_yields.csv"),
    Step("fundamentals",
         "pull_fundamentals_yf.py",
         "Per-ticker snapshot (P/E, beta, mkt cap, sector, RV30)"),
    Step("earnings",
         "pull_earnings_yf.py",
         "Past + upcoming earnings calendar (activates event gate)"),
    Step("theta_indices",
         "pull_theta_indices_history.py",
         "Authoritative VIX-family history from Theta (supersedes yfinance rows)",
         needs_theta=True),
    Step("theta_vix_futures",
         "pull_theta_vix_futures.py",
         "VIX futures UX1–UX8 curve history — needs Theta futures tier",
         needs_theta=True),
    Step("theta_corp_actions",
         "pull_theta_corp_actions.py",
         "Stock splits + dividends per ticker (fills empty corp actions file)",
         needs_theta=True),
    Step("theta_iv_surface",
         "pull_theta_iv_surface_history.py",
         "IV surface (strike × expiry × date) — requires Theta",
         needs_theta=True),
    Step("theta_flow",
         "pull_theta_options_flow.py",
         "Daily PCR/OI/unusual volume — requires Theta",
         needs_theta=True),
    Step("news",
         "pull_news_sentiment.py",
         "Per-ticker news sentiment — requires API key",
         needs_news_key=True),
    Step("features",
         "backfill_features.py",
         "Recompute feature store for all tickers (uses everything above)",
         is_feature_backfill=True),
]


def _match(step: Step, patterns: list[str]) -> bool:
    return any(p.lower() in step.name.lower() for p in patterns)


def run_step(step: Step, extra_args: list[str], dry_run: bool) -> tuple[str, str, float]:
    """Return (status, detail, elapsed_seconds)."""
    skip, reason = step.should_skip()
    if skip:
        return "SKIP", reason, 0.0
    cmd = [sys.executable, str(_ROOT / "scripts" / step.script), *extra_args]
    if dry_run:
        return "DRY", " ".join(cmd[2:]), 0.0
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    tail = (proc.stdout or "").strip().splitlines()[-1:]
    tail_line = tail[0] if tail else ""
    if proc.returncode == 0:
        return "OK", tail_line[:120], elapsed
    err_line = (proc.stderr or "").strip().splitlines()[-1:]
    return "FAIL", f"rc={proc.returncode} {err_line[0][:120] if err_line else tail_line[:120]}", elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every smart-wheel-engine puller in order")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan without executing")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only run steps whose name matches any of these substrings")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="Skip steps whose name matches any of these substrings")
    ap.add_argument("--years", type=float, default=5.0,
                    help="History years for pullers that accept --years")
    ap.add_argument("--workers", type=int, default=4,
                    help="Worker count for parallel pullers")
    ap.add_argument("--backfill-force", action="store_true",
                    help="Pass --force to the feature backfill")
    args = ap.parse_args()

    # Build the plan
    plan: list[Step] = []
    for step in STEPS:
        if args.only and not _match(step, args.only):
            continue
        if args.skip and _match(step, args.skip):
            continue
        plan.append(step)

    theta_note = "UP" if _theta_up() else "DOWN"
    news_note = "set" if any(
        os.environ.get(k) for k in ("POLYGON_API_KEY", "FINNHUB_API_KEY", "BENZINGA_API_KEY")
    ) else "unset"
    mode = "DRY-RUN" if args.dry_run else "LIVE"

    print()
    print("=" * 80)
    print(f" Smart Wheel Engine data refresh  ({mode})")
    print(f" Theta={theta_note}   news-key={news_note}   years={args.years}   workers={args.workers}")
    print("=" * 80)

    results: list[tuple[Step, str, str, float]] = []
    for i, step in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {step.name:<18}  {step.description}")
        # Decide extra args per step
        extra: list[str] = []
        if step.name in ("vol_indices", "treasury"):
            extra = ["--years", str(args.years)]
        elif step.name in ("fundamentals", "earnings"):
            extra = ["--workers", str(args.workers)]
        elif step.name == "theta_indices":
            extra = ["--years", str(args.years), "--incremental"]
        elif step.name == "theta_vix_futures":
            extra = ["--years", str(args.years), "--months", "8", "--incremental"]
        elif step.name == "theta_corp_actions":
            extra = ["--universe", "sp500", "--years", str(args.years), "--workers", str(args.workers)]
        elif step.name == "theta_iv_surface":
            extra = ["--universe", "sp500", "--days", "7", "--workers", str(args.workers)]
        elif step.name == "theta_flow":
            extra = ["--universe", "sp500", "--days", "30", "--workers", str(args.workers)]
        elif step.name == "news":
            extra = ["--universe", "sp500", "--hours", "72", "--workers", str(args.workers)]
        elif step.name == "features":
            extra = ["--workers", str(args.workers)]
            if args.backfill_force:
                extra.append("--force")

        status, detail, sec = run_step(step, extra, args.dry_run)
        if status == "DRY":
            print(f"       DRY: {detail}")
        else:
            tag = {"OK": "✓", "FAIL": "✗", "SKIP": "-"}[status]
            print(f"       {tag} {status} ({sec:>5.1f}s)  {detail}")
        results.append((step, status, detail, sec))

    # Summary
    n_ok = sum(1 for _, s, _, _ in results if s == "OK")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _, _ in results if s == "SKIP")
    total_t = sum(sec for _, _, _, sec in results)

    print()
    print("=" * 80)
    print(f" {n_ok} OK   {n_fail} FAIL   {n_skip} SKIP   |  total {total_t:.1f}s")
    print("=" * 80)
    if n_skip:
        print("\nSkipped steps (and why):")
        for step, status, detail, _ in results:
            if status == "SKIP":
                print(f"  - {step.name}: {detail}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
