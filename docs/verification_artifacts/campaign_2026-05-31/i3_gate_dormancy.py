"""I3-A: RISK-GATE BINDING & DEFAULT-PATH DORMANCY.

Question: do the D17 portfolio hard-blocks (R9 sector 25% NAV, R10 single-name
10% NAV, portfolio-delta, Kelly) bind on the DEFAULT path a $1M account uses,
or are they dormant because nothing sets require_ev_authority=True / attaches
a PortfolioContext with returns_data?

We (1) re-state the code facts with file:line, (2) QUANTIFY the consequence by
building the book the DEFAULT path admits from a real snapshot and measuring
concentration vs the 10%/25% caps, and (3) trace R7/R8 silent-skip.

READ-ONLY. Imports engine but never mutates it.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = L.ROOT
SNAP = os.path.join(HERE, "snapshots")


def load_snapshot(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(SNAP, name))


def sector_for(df_row) -> str:
    """Prefer the snapshot's sector; fall back to the fundamentals lookup."""
    s = df_row.get("sector")
    if isinstance(s, str) and s not in ("", "Unknown", "nan"):
        return s
    return L.sector_of(df_row["ticker"])


def build_default_book(df: pd.DataFrame, account: float, top_n: int) -> pd.DataFrame:
    """The DISCIPLINED-LOOKING default book a $1M account would take:
    positive-ev_dollars rows, one contract per name, ranked by ev_dollars
    desc, cash-secured (reserve strike*100 collateral), filled until either
    top_n names or the cash budget is exhausted. This is exactly what a user
    of rank_candidates_by_ev + 'take the top names' produces -- NO D17 gate
    (require_ev_authority defaults False) and NO select_book cap.
    """
    pos = df[df["ev_dollars"] > 0].copy()
    pos = pos.sort_values("ev_dollars", ascending=False)
    pos = pos.drop_duplicates(subset=["ticker"], keep="first")
    book = []
    cash = account
    for _, r in pos.iterrows():
        coll = float(r["strike"]) * 100.0
        if coll > cash:
            continue
        book.append(r)
        cash -= coll
        if len(book) >= top_n:
            break
    b = pd.DataFrame(book)
    if not b.empty:
        b["collateral_used"] = b["strike"] * 100.0
        b["sector_resolved"] = b.apply(sector_for, axis=1)
    return b


def concentration_report(b: pd.DataFrame, account: float) -> dict:
    if b.empty:
        return {"n": 0}
    b = b.copy()
    # Single-name % = each name's collateral / account (one contract/name here).
    b["name_pct"] = b["collateral_used"] / account
    by_sector = b.groupby("sector_resolved")["collateral_used"].sum() / account
    return {
        "n_positions": int(len(b)),
        "total_collateral": float(b["collateral_used"].sum()),
        "capital_deployed_pct": float(b["collateral_used"].sum() / account),
        "max_single_name_pct": float(b["name_pct"].max()),
        "max_single_name_ticker": b.loc[b["name_pct"].idxmax(), "ticker"],
        "n_names_over_10pct": int((b["name_pct"] > 0.10).sum()),
        "names_over_10pct": list(
            b.loc[b["name_pct"] > 0.10, ["ticker", "name_pct"]].itertuples(index=False, name=None)
        ),
        "max_sector_pct": float(by_sector.max()),
        "max_sector": by_sector.idxmax(),
        "n_sectors_over_25pct": int((by_sector > 0.25).sum()),
        "sectors_over_25pct": {k: float(v) for k, v in by_sector[by_sector > 0.25].items()},
        "sector_breakdown": {k: float(v) for k, v in by_sector.sort_values(ascending=False).items()},
    }


def main() -> None:
    print("=" * 78)
    print("I3-A  RISK-GATE BINDING & DEFAULT-PATH DORMANCY")
    print("=" * 78)

    # ---- Part 1: code facts (cite file:line) -----------------------------
    print("\n--- PART 1: CODE FACTS (read-only static confirmation) ---")
    wt = os.path.join(ROOT, "engine", "wheel_tracker.py")
    with open(wt, encoding="utf-8") as fh:
        wt_lines = fh.readlines()

    def show(path_lines, ln, label):
        print(f"  {label}: line {ln}: {path_lines[ln - 1].rstrip()}")

    show(wt_lines, 512, "open_short_put token-guard")
    show(wt_lines, 532, "open_short_put D17 hard-block guard")
    show(wt_lines, 552, "  -> _evaluate_d17_hard_blocks call")
    # default of require_ev_authority in __init__
    show(wt_lines, 278, "WheelTracker.__init__ default")

    # confirm RiskManager NOT imported by wheel_runner top-level ranker path
    wr = os.path.join(ROOT, "engine", "wheel_runner.py")
    with open(wr, encoding="utf-8") as fh:
        wr_src = fh.read()
    print(f"\n  'RiskManager' appears in wheel_runner.py? {('RiskManager' in wr_src)}")
    print(f"  'portfolio_risk_gates' imported in wheel_runner.py? "
          f"{('portfolio_risk_gates' in wr_src)}")
    # rank_candidates_by_ev body: does it reference require_ev_authority or D17 gates?
    rcbe_start = wr_src.index("def rank_candidates_by_ev")
    rcbe_end = wr_src.index("def select_book")
    rcbe_body = wr_src[rcbe_start:rcbe_end]
    print(f"  rank_candidates_by_ev references 'require_ev_authority'? "
          f"{('require_ev_authority' in rcbe_body)}")
    print(f"  rank_candidates_by_ev references any D17 gate "
          f"(check_sector_cap/check_single_name_cap/check_portfolio_delta)? "
          f"{any(g in rcbe_body for g in ['check_sector_cap', 'check_single_name_cap', 'check_portfolio_delta'])}")

    # ---- Part 2: quantify the consequence on a real snapshot -------------
    print("\n--- PART 2: WHAT THE DEFAULT PATH ADMITS (real snapshot) ---")
    # Pick a snapshot with many positive-ev rows. Scan a few; report counts.
    candidates = [
        "put_2021-06-01.parquet", "put_2021-11-01.parquet", "put_2024-07-01.parquet",
        "put_2021-04-01.parquet", "put_2023-12-01.parquet",
    ]
    best = None
    for nm in candidates:
        try:
            d = load_snapshot(nm)
        except FileNotFoundError:
            continue
        npos = int((d["ev_dollars"] > 0).sum())
        print(f"  {nm}: rows={len(d)} positive_ev={npos}")
        if best is None or npos > best[1]:
            best = (nm, npos, d)
    snap_name, npos, df = best
    print(f"\n  USING {snap_name} ({npos} positive-EV rows)")

    account = 1_000_000.0
    for top_n in (10, 20, 40):
        b = build_default_book(df, account, top_n)
        rep = concentration_report(b, account)
        print(f"\n  >>> DEFAULT BOOK top_n={top_n}, account=${account:,.0f}")
        for k, v in rep.items():
            if k in ("sector_breakdown", "names_over_10pct", "sectors_over_25pct"):
                continue
            print(f"      {k}: {v}")
        print(f"      sector_breakdown: {rep.get('sector_breakdown')}")
        if rep.get("names_over_10pct"):
            print(f"      NAMES > 10% NAV (R10 would block): {rep['names_over_10pct']}")
        if rep.get("sectors_over_25pct"):
            print(f"      SECTORS > 25% NAV (R9 would block): {rep['sectors_over_25pct']}")

    # ---- Part 2b: an ADVERSARIAL but realistic admitted book -------------
    # A senior trader chasing the highest single-name ev_dollars on this
    # snapshot, one contract per name, would the default path stop them
    # piling into one sector? Build a 'concentration-seeking' book: take the
    # single highest-EV name and replicate the discipline a naive user has
    # (none). Show the most concentrated sector a top-N-by-EV book reaches.
    print("\n--- PART 2c: most-concentrated admitted book (adversarial) ---")
    # All positive-EV names in the single richest sector, cash-secured.
    pos = df[df["ev_dollars"] > 0].copy()
    pos["sector_resolved"] = pos.apply(sector_for, axis=1)
    pos = pos.drop_duplicates("ticker")
    sec_counts = pos.groupby("sector_resolved").agg(
        n=("ticker", "count"), coll=("strike", lambda s: (s * 100).sum())
    )
    print("  positive-EV names per sector (collateral if all taken):")
    for sec, row in sec_counts.sort_values("coll", ascending=False).iterrows():
        print(f"      {sec}: n={int(row['n'])} collateral=${row['coll']:,.0f} "
              f"({row['coll'] / account:.1%} of $1M)")

    # ---- Part 3: R7/R8 silent-skip trace ---------------------------------
    print("\n--- PART 3: R7/R8 SILENT-SKIP TRACE (default reviewer path) ---")
    from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
    from engine.chart_context import ChartContext
    from datetime import datetime
    from pathlib import Path

    # Build a positive-EV dossier with a valid chart but NO portfolio_context
    # (exactly what build_dossiers produces when no caller assembles one).
    row = df[df["ev_dollars"] > 0].sort_values("ev_dollars", ascending=False).iloc[0].to_dict()
    # Fake a passing chart (screenshot present, price matches spot, no error).
    chart = ChartContext(
        ticker=row["ticker"], timeframe="1D", captured_at=datetime(2021, 6, 1),
        screenshot_path=Path("fake.png"), visible_price=float(row["spot"]),
        source="probe",
    )
    doss = CandidateDossier(ticker=row["ticker"], ev_row=row, chart_context=chart,
                            portfolio_context=None)
    verdict, reason, notes = EnginePhaseReviewer().review(doss)
    print(f"  Default dossier (portfolio_context=None): verdict={verdict} reason={reason}")
    r7_fired = any("R7" in n for n in notes)
    r8_fired = any("R8" in n for n in notes)
    print(f"  R7 note present in audit trail? {r7_fired}")
    print(f"  R8 note present in audit trail? {r8_fired}")
    print(f"  notes: {notes}")

    # Now attach a PortfolioContext WITHOUT returns_data -> R7 must skip with
    # missing_data, never fire.
    from engine.portfolio_risk_gates import PortfolioContext
    ctx_no_returns = PortfolioContext(nav=account, returns_data=None, correlation_matrix=None)
    doss2 = CandidateDossier(ticker=row["ticker"], ev_row=row, chart_context=chart,
                             portfolio_context=ctx_no_returns)
    v2, r2, notes2 = EnginePhaseReviewer().review(doss2)
    print(f"\n  With PortfolioContext but returns_data=None: verdict={v2} reason={r2}")
    print(f"  R7 VaR notes: {[n for n in notes2 if 'R7' in n]}")
    print(f"  (expect 'VaR check skipped (no_correlation_matrix_or_returns_data)')")

    print("\nDONE.")


if __name__ == "__main__":
    main()
