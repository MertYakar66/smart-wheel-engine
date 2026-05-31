"""S45 R10 firing probe (2026-05-29).

Empirical R10 (single-name 10%-NAV cap) firing analysis against the
S34 backtest's rank_log. Operates on
``%TEMP%/s45_s34_rebaseline/s34_run/{none,bid_ask,full}/rank_log.csv``
that S45's snapshot regeneration writes — the friction-level rank
order and strikes are identical across the three subdirectories
(``_common.py`` shares one ``rank_candidates_by_ev`` call per day);
only premium / realized_pnl differ. The probe defaults to the ``full``
subdirectory's rank_log because that is what S34's headline result
references.

Background. R10 (``engine.candidate_dossier`` rule R10, PR #262)
downgrades ``proceed → review`` when a candidate's
``proposed_notional > max_single_name_pct × NAV`` (default
10% NAV = $100k at $1M). It also hard-blocks at
``WheelTracker.open_short_put`` time when ``require_ev_authority=True``.
**S34's backtest runs with** ``require_ev_authority=False`` **AND
without a** ``PortfolioContext`` **attached** — so R10 does NOT fire
in the live S34 run. This probe answers the COUNTERFACTUAL:
*how often would R10 have fired if it had been wired in?*

Two analyses:

1. ``static_r10_per_ticker`` — purely from the rank_log, the
   per-ticker max single-contract notional. Tickers whose
   first-contract notional already exceeds the $100k cap are
   refused by R10 on every entry attempt with zero held positions
   in the same name needed. The strongest, most defensible
   headline claim.

2. ``replay_r10_counterfactual`` — replays the harness's open logic
   (top_n=15, max_new_per_day=3, EV>0, one open per ticker at a
   time, settle on expiration_date <= today) and calls
   ``check_single_name_cap`` at each open attempt. The same
   per-ticker rule the live tracker would apply if R10 were wired
   in. Counts R10 fires per ticker, actual-opens-post-R10
   per ticker, and a side-by-side with the (uninstrumented)
   actual harness's executed trades.

Read-only client of the engine. NOT a §2 surface change.
Re-runnable: ``py -3.12 docs/verification_artifacts/s45_r10_firing_driver.py``.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import pandas as pd  # noqa: E402

from engine.portfolio_risk_gates import (  # noqa: E402
    _DEFAULT_MAX_SINGLE_NAME_PCT,
    check_single_name_cap,
)

NAV = 1_000_000.0  # S34 / S45 starting capital
CAP_DOLLARS = NAV * _DEFAULT_MAX_SINGLE_NAME_PCT  # $100,000
DEFAULT_RUN_DIR = Path(os.environ.get("TEMP", "/tmp")) / "s45_s34_rebaseline" / "s34_run"


# --------------------------------------------------------------------- #
# Analysis 1: static (per-ticker) — needs only the rank_log
# --------------------------------------------------------------------- #
def static_r10_per_ticker(rank_log: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker static R10 verdict — purely from strike + EV columns.

    For each ticker:
    - ``max_fcn`` = max(strike) × 100 = first-contract dollar notional
      at the most-expensive strike the engine ranked. Single contract,
      zero held positions in the same name.
    - ``r10_blocks_1st`` = max_fcn > $100k. The engine cannot open ANY
      position in this name even on a clean slate.
    - ``r10_blocks_2nd_at_max`` = max_fcn > $50k AND not r10_blocks_1st.
      The engine can open one contract but a second is refused.
    - ``r10_blocks_3rd_at_max`` = max_fcn > $33k AND not above.
    """
    by_t = rank_log.groupby("ticker").agg(
        n_rows=("strike", "size"),
        n_ev_positive=("ev_dollars", lambda s: int((s > 0).sum())),
        max_strike=("strike", "max"),
        min_strike=("strike", "min"),
        median_strike=("strike", "median"),
        max_ev=("ev_dollars", "max"),
        mean_ev=("ev_dollars", "mean"),
        sum_realized=("realized_pnl", "sum"),
    )
    by_t["max_fcn"] = (by_t["max_strike"] * 100).round(0)
    by_t["r10_verdict"] = by_t["max_fcn"].apply(
        lambda fcn: "blocks_1st"
        if fcn > CAP_DOLLARS
        else "blocks_2nd"
        if fcn > CAP_DOLLARS / 2
        else "blocks_3rd"
        if fcn > CAP_DOLLARS / 3
        else "non_binding"
    )
    by_t["max_fcn_vs_cap_x"] = (by_t["max_fcn"] / CAP_DOLLARS).round(2)
    return by_t.sort_values("max_fcn", ascending=False)


# --------------------------------------------------------------------- #
# Analysis 2: counterfactual replay — what would R10 have refused?
# --------------------------------------------------------------------- #
def replay_r10_counterfactual(
    rank_log: pd.DataFrame,
    *,
    top_n: int = 15,
    max_new_per_day: int = 3,
) -> dict:
    """Replay the backtest's open logic, with R10 wired in.

    Mirrors ``_common.py``'s ``_tracker_try_opens`` plus the expiration
    settlement loop: per-day expirations release held notional once
    ``expiration_date <= today``; opens require EV > 0, no existing
    position in the same name, and ``opens_today < max_new_per_day``;
    R10 is checked via ``check_single_name_cap`` at each open attempt
    against the synthetic held-options snapshot.

    Returns per-ticker counts:
    - ``attempted`` — how many EV>0 rows the harness considered for an
      open (after the no-existing-position filter and the daily cap)
    - ``r10_refused`` — how many R10 would have refused
    - ``opens_post_r10`` — how many would have actually opened with R10
      wired in (attempted - r10_refused)
    """
    df = rank_log.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    held: dict[str, list[tuple[pd.Timestamp, float, pd.Timestamp]]] = defaultdict(list)
    attempted: dict[str, int] = defaultdict(int)
    r10_refused: dict[str, int] = defaultdict(int)
    opens_post: dict[str, int] = defaultdict(int)
    r10_refused_notional_total = 0.0
    r10_refused_per_year: dict[int, int] = defaultdict(int)

    for today, day_frame in df.groupby("date"):
        # Settle expirations
        for tkr in list(held.keys()):
            held[tkr] = [(od, st, ed) for od, st, ed in held[tkr] if ed > today]
            if not held[tkr]:
                del held[tkr]

        opens_today = 0
        # Iterate in rank order — the rank_log is appended in rank-order
        # per day in ``_common.py:493`` and ``:830``.
        day_frame = day_frame.head(top_n)
        for _, row in day_frame.iterrows():
            if opens_today >= max_new_per_day:
                break
            if row["ev_dollars"] <= 0:
                continue
            tkr = str(row["ticker"])
            if tkr in held:
                continue  # existing-position rule
            strike = float(row["strike"])
            if strike <= 0:
                continue
            attempted[tkr] += 1
            proposed_notional = strike * 100.0  # 1 contract

            held_options = []
            for tkr2, lst in held.items():
                for _od, st, ed in lst:
                    held_options.append(
                        {
                            "symbol": tkr2,
                            "option_type": "put",
                            "strike": float(st),
                            "dte": max(0, (ed.date() - today.date()).days),
                            "iv": 0.3,
                            "contracts": 1,
                            "is_short": True,
                        }
                    )

            result = check_single_name_cap(
                symbol=tkr,
                proposed_notional=proposed_notional,
                held_option_positions=held_options,
                nav=NAV,
            )
            if not result.passed:
                r10_refused[tkr] += 1
                r10_refused_notional_total += proposed_notional
                r10_refused_per_year[today.year] += 1
                # Critical: opens_today is NOT incremented; max_new_per_day
                # is a "successful opens" cap.
                continue

            # Would open. Track + advance state.
            held[tkr].append((today, strike, row["expiration_date"]))
            opens_post[tkr] += 1
            opens_today += 1

    summary = pd.DataFrame(
        {
            "attempted": pd.Series(attempted),
            "r10_refused": pd.Series(r10_refused),
            "opens_post_r10": pd.Series(opens_post),
        }
    ).fillna(0).astype(int)
    summary["refused_rate"] = summary["r10_refused"] / summary["attempted"].clip(lower=1)
    summary = summary.sort_values("r10_refused", ascending=False)

    return {
        "per_ticker": summary,
        "total_r10_refused": int(sum(r10_refused.values())),
        "total_refused_notional": float(r10_refused_notional_total),
        "refused_per_year": dict(r10_refused_per_year),
    }


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main() -> None:
    rank_log_path = (
        Path(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_RUN_DIR / "full" / "rank_log.csv"
    )
    if not rank_log_path.exists():
        sys.exit(f"rank_log not found at {rank_log_path}")

    print(f"# S45 R10 firing probe — {rank_log_path}")
    print(f"# NAV={NAV:,.0f}  CAP={CAP_DOLLARS:,.0f} (max_single_name_pct={_DEFAULT_MAX_SINGLE_NAME_PCT:.2%})")
    print()

    rank_log = pd.read_csv(rank_log_path)
    print(f"rank_log: {len(rank_log):,} rows; "
          f"tickers={rank_log['ticker'].nunique()}; "
          f"dates={rank_log['date'].nunique()}")
    print()

    # ---- Analysis 1
    print("## Static per-ticker R10 verdict (max_fcn = max-strike × 100)")
    static = static_r10_per_ticker(rank_log)
    by_verdict = static["r10_verdict"].value_counts()
    print(by_verdict.to_string())
    print()
    print("Top 25 tickers by max_fcn (descending):")
    cols = ["n_rows", "n_ev_positive", "max_strike", "max_fcn",
            "max_fcn_vs_cap_x", "r10_verdict", "sum_realized"]
    print(static[cols].head(25).to_string())
    print()
    headline_names = ["BKNG", "NVR", "AZO", "ORLY", "GS", "COST", "MTD", "FICO"]
    present = [t for t in headline_names if t in static.index]
    if present:
        print("Headline-name verdicts:")
        print(static.loc[present, cols].to_string())
        print()

    # ---- Analysis 2
    print("## R10 counterfactual replay (top_n=15, max_new_per_day=3)")
    repl = replay_r10_counterfactual(rank_log)
    print(f"Total R10 fires: {repl['total_r10_refused']:,}")
    print(f"Total refused notional: ${repl['total_refused_notional']:,.0f}")
    print(f"R10 fires per year: {dict(sorted(repl['refused_per_year'].items()))}")
    print()
    print("Per-ticker (sorted by r10_refused desc):")
    pt = repl["per_ticker"]
    nonzero = pt[pt["r10_refused"] > 0]
    print(f"Tickers with >=1 R10 fire: {len(nonzero)}")
    print(nonzero.head(25).to_string())


if __name__ == "__main__":
    main()
