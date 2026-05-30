"""Persona walkthrough driver — HT-A (heavy-verify, 2026-05-30).

Scripted session of a professional quant trader using the Smart Wheel Engine
end-to-end. Read-only against the engine: scans SP500, ranks, builds dossiers,
issues + consumes EV-authority tokens, opens positions in WheelTracker.

This driver is deliberately a single file with flat-printed output so the
captured `_raw_output.txt` is grep-able and the findings doc in
`docs/HEAVY_PERSONA_WALKTHROUGH.md` can cite specific numbers/lines.

Four operator asks, in order:
    Ask 1 — "rank me 20 names" — full SP500 scan at as_of=2026-03-20.
    Ask 2 — "why was X filtered" — pick a dropped name, surface gate + reason.
    Ask 3 — "size this within a $250k book" — knapsack fit + tracker with
            require_ev_authority=True + PortfolioContext + R7-R10 soft-warns.
    Ask 4 — "what's the downside if I get assigned" — for one survivor:
            forward distribution percentiles, CVaR_5, CVaR_99 EVT, tail_xi,
            dealer regime, walls, breakeven, ROC.

End-of-run trace of §2 invariants:
    * issuance refuses non-positive EV (D16 leg 1).
    * consume rejects stale EV (D16 leg 2).
    * dossier R1 blocks negative-EV candidate.
    * R7-R10 only downgrade.

This file lives under `docs/verification_artifacts/` per
`docs/verification_artifacts/README.md`. Per that README the driver writes
to stdout; the operator captures with:

    "/c/Users/.../python.exe" docs/verification_artifacts/persona_walkthrough_driver.py \
        > docs/verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt

The driver does NOT modify any engine module. Bugs surfaced here go in
`docs/HEAVY_PERSONA_WALKTHROUGH.md` as findings for the Major Session to
triage into fix-cards.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

WORKTREE = Path(r"C:\Users\merty\Desktop\swe-terminal-a").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import pandas as pd  # noqa: E402  (sys.path bootstrap above)

from engine.candidate_dossier import (  # noqa: E402
    EnginePhaseReviewer,
    build_dossiers,
)
from engine.chart_context import ChartContext, ChartContextProvider  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402
from engine.wheel_tracker import EVAuthorityRefused, WheelTracker  # noqa: E402

AS_OF = "2026-03-20"  # freshest Bloomberg date per SessionStart hook
HEADLINE_COLS = [
    "ticker",
    "spot",
    "strike",
    "premium",
    "dte",
    "iv",
    "ev_dollars",
    "ev_per_day",
    "pnl_p25",
    "pnl_p50",
    "pnl_p75",
    "cvar_5",
    "prob_profit",
    "prob_assignment",
    "collateral",
    "roc",
    "sector",
    "distribution_source",
]
DIAG_COLS = [
    "ticker",
    "ev_raw",
    "ev_dollars",
    "regime_multiplier",
    "hmm_regime",
    "hmm_multiplier",
    "news_multiplier",
    "credit_multiplier",
    "dealer_multiplier",
    "dealer_regime",
    "tail_widening_factor",
    "heavy_tail",
    "tail_xi",
    "cvar_99_evt",
    "gex_total",
    "gamma_flip_distance_pct",
    "nearest_put_wall_strike",
    "nearest_call_wall_strike",
    "skew_multiplier",
    "skew_source",
    "oi_source",
    "premium_source",
]


def hline(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


class _NullChartProvider(ChartContextProvider):
    """No-op chart provider so dossiers can be built without TradingView.

    Returns ``ChartContext`` with ``error`` populated — the reviewer's R2
    will set verdict='review' / verdict_reason='chart_context_missing'.
    This is the realistic state for an operator running off the chart-less
    Bloomberg path (matches `docs/PROJECT_STATE.md` §3 TradingView MCP
    integration: stage 3 lands behind SWE_USE_MCP_CHART, default off).
    """

    def fetch(self, ticker, timeframe, as_of=None):  # type: ignore[override]
        return ChartContext(
            ticker=ticker,
            timeframe=timeframe,
            captured_at=datetime.utcnow(),  # required dataclass field
            screenshot_path=None,
            source="null",
            visible_price=None,
            visible_indicators={},
            error="no_chart_provider_attached",
        )


# ---------------------------------------------------------------------------
# Persona setup
# ---------------------------------------------------------------------------
hline("HT-A — Persona walkthrough (heavy-verify, 2026-05-30)")
print(
    f"Persona: professional quant trader.\n"
    f"As-of date: {AS_OF} (freshest Bloomberg data per SessionStart hook).\n"
    f"Mode: read-only — no engine/ edits. Observe + document only."
)

runner = WheelRunner()
provider_name = type(runner.connector).__name__
print(f"\nProvider: {provider_name}")
print("  (default per CLAUDE.md §4: MarketDataConnector = bloomberg CSV path)")

# Sanity: how big is the universe the connector actually sees?
try:
    universe_size = len(runner.connector.get_universe())
except Exception as e:  # pragma: no cover — runtime probe only
    universe_size = -1
    print(f"  ! connector.get_universe() raised: {e}")
print(f"Connector universe size: {universe_size}")


# ---------------------------------------------------------------------------
# Ask 1 — "rank me 20 names"
# ---------------------------------------------------------------------------
hline("Ask 1 — 'rank me 20 names'")
print(
    "Trader request: 'Scan the universe and show me the 20 best short-put\n"
    "candidates right now, with EV and downside.'\n"
)

# Default rank: 35 DTE / 25-delta / 1 contract / full universe.
# min_ev_dollars=0 keeps it operator-realistic (positive-EV only, since
# the dossier reviewer would block anything negative anyway).
ranking = runner.rank_candidates_by_ev(
    tickers=None,
    dte_target=35,
    delta_target=0.25,
    contracts=1,
    top_n=20,
    min_ev_dollars=0.0,
    as_of=AS_OF,
    include_diagnostic_fields=True,
)
print(f"Rows returned: {len(ranking)}  (requested top_n=20)")

if not ranking.empty:
    headline = ranking[HEADLINE_COLS].copy()
    print("\nHeadline columns (sorted by ev_per_day desc):")
    print(headline.to_string(index=False))
    print("\nDiagnostic columns (same order):")
    print(ranking[DIAG_COLS].to_string(index=False))

# `.attrs` exposes what the operator would otherwise miss.
drops = ranking.attrs.get("drops", [])
drops_summary = ranking.attrs.get("drops_summary", {})
print(f"\n.attrs['drops_summary'] = {drops_summary}")
print(f".attrs['drops'] length = {len(drops)}")

# Per-gate breakdown so the persona doc can cite exact counts.
sub("Per-gate drop counts (silent-filtering census)")
by_gate = drops_summary.get("by_gate", {}) if drops_summary else {}
if not by_gate and drops:
    # Defensive: rebuild if summary attribute is missing.
    by_gate = dict(Counter(d.get("gate", "?") for d in drops))
for gate, count in sorted(by_gate.items(), key=lambda kv: -kv[1]):
    print(f"  {gate:>14s} : {count}")

sub("Sample of dropped tickers per gate (up to 5 each)")
seen_per_gate: dict[str, int] = {}
for drop in drops:
    g = drop.get("gate", "?")
    if seen_per_gate.get(g, 0) >= 5:
        continue
    seen_per_gate[g] = seen_per_gate.get(g, 0) + 1
    print(f"  [{g}] {drop.get('ticker', '?'):>6s} : {drop.get('reason', '')}")

# Also expose the universe coverage so the persona doc can cite it.
print(
    f"\nCoverage: {len(ranking)} survivors + {len(drops)} drops = "
    f"{len(ranking) + len(drops)} of {universe_size} universe tickers"
)

# Cache a positive-EV survivor for downstream asks.
top_survivor = ranking.iloc[0].to_dict() if not ranking.empty else None
if top_survivor is not None:
    print(
        f"\nTop survivor cached for Asks 3/4: "
        f"{top_survivor['ticker']} {top_survivor['strike']}P  "
        f"ev_dollars=${top_survivor['ev_dollars']}  "
        f"ev_per_day=${top_survivor['ev_per_day']}"
    )


# ---------------------------------------------------------------------------
# Ask 2 — "why was X filtered"
# ---------------------------------------------------------------------------
hline("Ask 2 — 'why was X filtered'")
print(
    "Trader request: 'I expected to see PRGO and a few others on the screen.\n"
    " Why didn't they make it?'\n"
)

# Pick (a) a ticker that was definitely dropped, plus a small batch of
# names from each gate bucket so the persona doc can cite the operator
# experience verbatim.
probe_tickers: list[str] = []
seen_gates: set[str] = set()
for drop in drops:
    g = drop.get("gate", "?")
    if g not in seen_gates:
        probe_tickers.append(drop["ticker"])
        seen_gates.add(g)
    if len(probe_tickers) >= 6:
        break

# Bound the probe so we don't accidentally re-rank a huge universe.
if probe_tickers:
    print(f"Probing dropped tickers (one per gate where present): {probe_tickers}")
    # Pull the matching drop reason from the original list so we
    # show exactly what the ranker logged.
    for t in probe_tickers:
        matching = [d for d in drops if d.get("ticker") == t]
        if matching:
            d = matching[0]
            print(f"  {t:>6s}  gate={d.get('gate', '?'):>10s}  reason={d.get('reason', '')}")

    # Re-run the ranker for just these names so the persona doc can show
    # whether they'd survive in isolation — sometimes a per-ticker rank
    # surfaces a more useful reason than the universe-wide drops list.
    sub("Single-ticker re-rank (each in isolation, same kwargs)")
    for t in probe_tickers:
        try:
            sub_df = runner.rank_candidates_by_ev(
                tickers=[t],
                dte_target=35,
                delta_target=0.25,
                contracts=1,
                top_n=1,
                min_ev_dollars=-1e9,  # include even negative-EV survivors
                as_of=AS_OF,
                include_diagnostic_fields=True,
            )
            if sub_df.empty:
                sub_drops = sub_df.attrs.get("drops", [])
                if sub_drops:
                    d = sub_drops[0]
                    print(
                        f"  {t:>6s}  dropped  gate={d.get('gate', '?')}  reason={d.get('reason', '')}"
                    )
                else:
                    print(f"  {t:>6s}  dropped  (no drop record returned)")
            else:
                r = sub_df.iloc[0].to_dict()
                print(
                    f"  {t:>6s}  survived in isolation: "
                    f"ev=${r['ev_dollars']}  prob_profit={r['prob_profit']}  "
                    f"premium=${r['premium']}  iv={r['iv']}"
                )
        except Exception as e:
            print(f"  {t:>6s}  per-ticker rank raised: {e!r}")
else:
    print("(no drops to probe — universe-wide rank returned zero filtered names)")


# Also test what happens when the trader asks about a name that doesn't
# exist in the connector universe at all. This is the operator-typo case.
sub("Operator typo case: rank for an unknown ticker")
typo_df = runner.rank_candidates_by_ev(
    tickers=["XYZZ"],
    top_n=1,
    min_ev_dollars=-1e9,
    as_of=AS_OF,
    include_diagnostic_fields=True,
)
print(f"  rows={len(typo_df)}  drops={typo_df.attrs.get('drops', [])}")


# ---------------------------------------------------------------------------
# Ask 3 — "size this within a $250k book"
# ---------------------------------------------------------------------------
hline("Ask 3 — 'size this within a $250k book'")
print(
    "Trader request: 'I have $250k to deploy. Size this for me, then walk\n"
    " each opener through the EV-authority gate so we don't double-deploy\n"
    " a name or breach the sector / single-name caps.'\n"
)

ACCOUNT_SIZE = 250_000.0

# Step 1: select_book — pure post-processor over the ranking; §2-safe.
book = runner.select_book(
    account_size=ACCOUNT_SIZE,
    ranking=ranking,
    max_weight_per_name=0.25,  # no single name > 25% of $250k = $62.5k collateral
    min_roc=0.0,
)
print(f"select_book — {len(book)} positions chosen out of {len(ranking)} ranked.")
print(f"  attrs: {dict(book.attrs)}")

if not book.empty:
    book_cols = [
        "ticker",
        "strike",
        "premium",
        "collateral",
        "ev_dollars",
        "ev_per_day",
        "roc",
        "prob_profit",
        "sector",
    ]
    print(book[book_cols].to_string(index=False))

# Step 2: WheelTracker in STRICT mode (require_ev_authority=True).
# Connector wired so D17 hard-blocks see a live NAV.
print("\nWheelTracker constructed with:")
print("  initial_capital = 250_000")
print("  require_ev_authority = True   (D16 token gate)")
print("  connector wired (D17 live NAV mark-to-market)")
print("  min_nav_for_trading = 1_000")

tracker = WheelTracker(
    initial_capital=ACCOUNT_SIZE,
    require_ev_authority=True,
    connector=runner.connector,
    min_nav_for_trading=1_000.0,
)

# Inject a small held book so the dossier reviewer's R7/R8/R9/R10 have
# something to score against — otherwise PortfolioContext is empty and
# all four R-rules silently skip on missing data (Q3 semantics).
# Choice rationale: two existing short puts in the SAME sector as
# whatever the top survivor turns out to be, sized to ~5% NAV each so
# R10 will see the candidate as the THIRD same-name-or-sector position
# and the operator can observe R9/R10 firing.
held: list[dict] = []
seed_date = date.fromisoformat(AS_OF) - timedelta(days=10)
seed_exp = seed_date + timedelta(days=25)
# Use two ranked-but-not-top names as the held-book seed (real prices,
# real sectors). Falls back to no seed when ranking is empty.
if not ranking.empty and len(ranking) >= 3:
    seed_rows = [
        ranking.iloc[len(ranking) // 2].to_dict(),
        ranking.iloc[len(ranking) // 3].to_dict(),
    ]
    print("\nSeeding tracker with two existing short puts (so R7-R10 have evidence):")
    for s in seed_rows:
        # Issue a token, open the put — this exercises the strict-mode
        # path on the seed too, so failures are visible.
        try:
            seed_token = tracker.issue_ev_authority_token(s)
        except EVAuthorityRefused as e:
            print(f"  [seed] {s['ticker']} refused at issuance: {e}")
            continue
        opened = tracker.open_short_put(
            ticker=str(s["ticker"]),
            strike=float(s["strike"]),
            premium=float(s["premium"]),
            entry_date=seed_date,
            expiration_date=seed_exp,
            iv=float(s.get("iv", 0.25) or 0.25),
            ev_authority_token=seed_token,
            current_ev_dollars=float(s["ev_dollars"]),
            prob_profit=float(s.get("prob_profit", 0.0) or 0.0),
        )
        print(
            f"  [seed] {s['ticker']:>6s}  strike={s['strike']:>6}  "
            f"premium=${s['premium']}  opened={opened}"
        )
        held.append({"ticker": s["ticker"], "strike": s["strike"], "premium": s["premium"]})

print(f"\nTracker positions after seed: {list(tracker.positions.keys())}")
print(f"Tracker cash after seed: ${tracker.cash:,.2f}")

# Step 3: portfolio_context_snapshot — D17 PortfolioContext for dossiers.
spot_prices = {t: float(runner.connector.get_ohlcv(t)["close"].iloc[-1]) for t in tracker.positions}
ctx = tracker.portfolio_context_snapshot(
    spot_prices=spot_prices,
    today=date.fromisoformat(AS_OF),
)
print(
    f"\nPortfolioContext: NAV=${ctx.nav:,.2f}  "
    f"held_option_positions={len(ctx.held_option_positions)}  "
    f"stock_holdings={len(ctx.stock_holdings)}"
)

# Step 4: build dossiers for the top 20 with the PortfolioContext
# attached so R7-R10 fire live, and a null chart provider so R2 is
# observed as the dominant verdict_reason ("chart_context_missing").
reviewer = EnginePhaseReviewer()
chart_provider = _NullChartProvider()
dossiers = build_dossiers(
    ranking,
    chart_provider,
    reviewer=reviewer,
    timeframe="1D",
    top_n=20,
    as_of=None,
    portfolio_context=ctx,
)
print(f"\nDossiers built: {len(dossiers)}")
sub("Verdict distribution")
verdict_counts = Counter(d.verdict for d in dossiers)
reason_counts = Counter(d.verdict_reason for d in dossiers)
for v, c in verdict_counts.most_common():
    print(f"  verdict={v:>8s} : {c}")
sub("Verdict-reason distribution")
for r, c in reason_counts.most_common():
    print(f"  reason={r:>30s} : {c}")

sub("Top 5 dossiers (verdict + key fields + notes)")
for d in dossiers[:5]:
    row = d.ev_row
    print(
        f"\n  {d.ticker}  verdict={d.verdict}  reason='{d.verdict_reason}'  "
        f"ev=${row.get('ev_dollars')}  pp={row.get('prob_profit')}"
    )
    for note in d.review_notes:
        print(f"    note: {note}")


# Step 5: try to wire the top survivor into the tracker via the canonical
# `consume_ranker_row` helper. This is the full §2 path.
sub("§2 path — consume_ranker_row(top survivor)")
if top_survivor is not None and top_survivor["ticker"] not in tracker.positions:
    print(
        f"  candidate: {top_survivor['ticker']} {top_survivor['strike']}P  "
        f"premium=${top_survivor['premium']}  ev=${top_survivor['ev_dollars']}"
    )
    try:
        opened_top = tracker.consume_ranker_row(
            top_survivor,
            entry_date=date.fromisoformat(AS_OF),
        )
    except EVAuthorityRefused as e:
        opened_top = False
        print(f"  EVAuthorityRefused at issuance: {e}")
    print(f"  open_short_put returned: {opened_top}")
    print(f"  tracker positions now: {list(tracker.positions.keys())}")
    print(f"  tracker cash now: ${tracker.cash:,.2f}")
elif top_survivor is not None:
    print(f"  ! top survivor {top_survivor['ticker']} already held — skipping")

# Step 6: trace the last few audit-log entries so the doc can show
# what an operator sees when they ask "what just happened".
sub("EV-authority audit log (last 12 entries)")
for entry in tracker._ev_authority_log[-12:]:
    keys = sorted(entry.keys())
    summary = ", ".join(f"{k}={entry[k]}" for k in keys if k != "row")
    print(f"  {summary}")


# ---------------------------------------------------------------------------
# Ask 4 — "what's the downside if I get assigned"
# ---------------------------------------------------------------------------
hline("Ask 4 — 'what is the downside if I get assigned?'")
print(
    "Trader request: 'For the trade we just opened, walk me through the\n"
    " downside — distribution percentiles, CVaR, what regime we're in, and\n"
    " what the chart-side signals say about a bounce vs further drawdown.'\n"
)

if top_survivor is not None:
    t = top_survivor["ticker"]
    print(f"Anchor candidate: {t}")
    print(
        f"  spot={top_survivor['spot']}  strike={top_survivor['strike']}P  "
        f"premium=${top_survivor['premium']}  dte={top_survivor['dte']}d"
    )

    sub("Distribution shape (raw $ — total position, premium × 100 × contracts)")
    for key in ["pnl_p25", "pnl_p50", "pnl_p75", "cvar_5", "cvar_99_evt", "tail_xi", "heavy_tail"]:
        print(f"  {key:>16s} = {top_survivor.get(key)}")
    pnl_p25 = top_survivor.get("pnl_p25")
    pnl_p75 = top_survivor.get("pnl_p75")
    if pnl_p25 is not None and pnl_p75 is not None:
        try:
            iqr = float(pnl_p75) - float(pnl_p25)
            print(f"  IQR (P75 - P25) = ${iqr:.2f}  (post-multiplier P&L spread)")
        except (TypeError, ValueError):
            pass

    sub("Assignment-side framing")
    spot = float(top_survivor["spot"])
    strike = float(top_survivor["strike"])
    premium = float(top_survivor["premium"])
    contracts = 1
    assigned_basis = (strike - premium) * 100 * contracts
    notional = strike * 100 * contracts
    breakeven_move_pct = top_survivor.get("breakeven_move_pct")
    print(f"  spot                 = ${spot}")
    print(f"  strike               = ${strike}")
    print(f"  premium              = ${premium}")
    print(f"  notional if assigned = ${notional:,.2f}")
    print(f"  cost basis if assigned (strike - premium) × 100 = ${assigned_basis:,.2f}")
    print(f"  breakeven_move_pct (engine column) = {breakeven_move_pct}")
    print(
        f"  prob_profit / prob_assignment = "
        f"{top_survivor.get('prob_profit')} / {top_survivor.get('prob_assignment')}"
    )

    sub("Regime + dealer-positioning context")
    for key in [
        "hmm_regime",
        "hmm_multiplier",
        "hmm_realized_vol_252d_ann",
        "hmm_realized_return_252d_ann",
        "credit_regime",
        "credit_multiplier",
        "news_multiplier",
        "news_n_articles",
        "regime_multiplier",  # FINAL multiplier the engine applied
        "tail_widening_factor",  # F4 RV widening
        "dealer_regime",
        "dealer_multiplier",
        "gex_total",
        "gamma_flip_distance_pct",
        "nearest_put_wall_strike",
        "nearest_call_wall_strike",
        "skew_multiplier",
        "skew_source",
    ]:
        print(f"  {key:>32s} = {top_survivor.get(key)}")

    sub("Capital efficiency")
    print(f"  collateral = ${top_survivor['collateral']}")
    print(f"  roc        = {top_survivor['roc']}  (= ev_dollars / collateral)")
    ev_d = float(top_survivor["ev_dollars"])
    coll = float(top_survivor["collateral"])
    if coll > 0:
        roc_check = ev_d / coll
        print(f"  roc cross-check ev_dollars/collateral = {roc_check:.6f}")

# ---------------------------------------------------------------------------
# Section: §2 invariants observed
# ---------------------------------------------------------------------------
hline("§2 invariants — observed behaviour")

# (a) Token issuance refuses non-positive EV.
sub("(a) D16 leg 1: issue refuses ev_dollars <= 0")
neg_row = {
    "ticker": "ZZZZ",
    "strike": 100.0,
    "premium": 1.50,
    "dte": 35,
    "ev_dollars": -42.0,  # negative — must refuse
    "prob_profit": 0.40,
    "distribution_source": "test_neg",
    "iv": 0.30,
}
try:
    tok = tracker.issue_ev_authority_token(neg_row)
    print(f"  ! UNEXPECTED: token issued for negative EV: {tok}")
except EVAuthorityRefused as e:
    print(f"  ✓ EVAuthorityRefused raised: {e}")

# (b) Consume rejects stale EV.
sub("(b) D16 leg 2: consume rejects stale ev_dollars (positive at issue, zero at fire)")
stale_row = dict(neg_row)
stale_row["ev_dollars"] = 25.0  # positive at issue
stale_token = tracker.issue_ev_authority_token(stale_row)
print(f"  issued token for positive EV row: {stale_token[:16]}…")
# Now consume with stale (non-positive) current_ev_dollars.
opened_stale = tracker.open_short_put(
    ticker="ZZZZ",
    strike=100.0,
    premium=1.50,
    entry_date=date.fromisoformat(AS_OF),
    expiration_date=date.fromisoformat(AS_OF) + timedelta(days=35),
    iv=0.30,
    ev_authority_token=stale_token,
    current_ev_dollars=0.0,  # stale — must reject
    prob_profit=0.40,
)
print(f"  open_short_put returned: {opened_stale}  (expect False)")
print(f"  audit-log tail: {tracker._ev_authority_log[-1]}")

# (c) Dossier R1 blocks negative-EV candidate.
sub("(c) Dossier R1: negative EV → verdict=blocked, never upgrade")
neg_frame = pd.DataFrame(
    [
        {
            "ticker": "ZZNG",
            "spot": 100.0,
            "strike": 95.0,
            "premium": 1.10,
            "dte": 35,
            "iv": 0.30,
            "ev_dollars": -15.5,
            "prob_profit": 0.42,
            "distribution_source": "synthetic",
            "collateral": 9500.0,
            "roc": -0.0016,
        }
    ]
)
neg_dossiers = build_dossiers(neg_frame, _NullChartProvider(), top_n=1, portfolio_context=ctx)
for d in neg_dossiers:
    print(f"  {d.ticker}  verdict={d.verdict}  reason={d.verdict_reason}")
    for note in d.review_notes:
        print(f"    note: {note}")

# (d) Dossier R1a: non-finite EV.
sub("(d) Dossier R1a: non-finite EV → verdict=blocked, reason=ev_non_finite")
nan_frame = pd.DataFrame(
    [
        {
            "ticker": "ZZNF",
            "spot": 100.0,
            "strike": 95.0,
            "premium": 1.10,
            "dte": 35,
            "iv": 0.30,
            "ev_dollars": float("nan"),
            "prob_profit": 0.42,
            "distribution_source": "synthetic",
            "collateral": 9500.0,
            "roc": float("nan"),
        }
    ]
)
nan_dossiers = build_dossiers(nan_frame, _NullChartProvider(), top_n=1, portfolio_context=ctx)
for d in nan_dossiers:
    print(
        f"  {d.ticker}  verdict={d.verdict}  reason={d.verdict_reason}  ev_dollars={d.ev_dollars}"
    )
    print(f"    finite? {math.isfinite(d.ev_dollars)}")

# (e) Reviewer never upgrades — chart agreement on negative-EV row.
sub("(e) Reviewer never upgrades: even a 'perfect' chart leaves negative EV blocked")


class _PerfectChartProvider(ChartContextProvider):
    """Returns a perfect chart with spot == engine spot. R1 must still block."""

    def fetch(self, ticker, timeframe, as_of=None):  # type: ignore[override]
        return ChartContext(
            ticker=ticker,
            timeframe=timeframe,
            captured_at=datetime.utcnow(),
            screenshot_path=Path("fake_perfect.png"),  # is_ok() requires non-None
            source="test_perfect",
            visible_price=100.0,  # matches spot above
            visible_indicators={},
            error="",  # is_ok() falsy-checks `error`
        )


good_chart_dossiers = build_dossiers(
    neg_frame, _PerfectChartProvider(), top_n=1, portfolio_context=ctx
)
for d in good_chart_dossiers:
    print(f"  {d.ticker}  verdict={d.verdict}  reason={d.verdict_reason}  (perfect chart attached)")
    for note in d.review_notes:
        print(f"    note: {note}")


# ---------------------------------------------------------------------------
# Run footer
# ---------------------------------------------------------------------------
hline("Run complete")
print(
    f"Tickers in survivor frame : {len(ranking)}\n"
    f"Tickers dropped           : {len(drops)}\n"
    f"Dossiers built            : {len(dossiers)}\n"
    f"Tracker positions opened  : {len(tracker.positions)}\n"
    f"Audit-log entries         : {len(tracker._ev_authority_log)}\n"
)
print("Persona walkthrough driver finished.")
