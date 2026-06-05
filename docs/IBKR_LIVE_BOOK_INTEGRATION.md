# IBKR Live-Book Integration — Design Doc (DRAFT, no code yet)

**Status:** DESIGN ONLY — nothing in this document is implemented. It proposes
three `DECISIONS.md` entries: a **D24** that is in-scope under CLAUDE.md §2/§3, a
**D25** that is an explicit, consent-gated *scope expansion* (the exit-evaluator),
and a **D26** read-only personal **performance viewer** (in-scope, observational).
None is adopted. This is the written output of a brainstorming session; it exists
so the design is on the record before any branch touches code.

**Author context:** written against `main` @ the HEAD that shipped the
prob_profit Wilson-CI work (#317/#318) and the D17 cap-adoption status note
(#319). Seam line numbers are "as of this writing" and will drift — they are
pins to *find* the code, not contracts.

**Scope banner.** Everything in Part 2 is **read-only** and **downgrade-only**:
it gives the engine a live book to evaluate *against*, and never converts a
non-tradeable candidate into a tradeable one. No order-routing surface is
proposed anywhere. Part 3 (the exit-evaluator) is flagged as a genuine scope
frontier and is deliberately left un-adopted.

---

## 0. Motivation — the dormancy gap, and why IBKR closes it

`PROJECT_STATE.md` §3 ("D17 cap adoption status — verified 2026-06-01") records
the problem this integration exists to solve:

> the R9/R10 concentration caps are correctly **implemented and unit-tested**
> but **not active on any default operator path** — "documented protection ≠
> active protection". … there is **no default-armed live path today**. … Closing
> the gap is a forward step: route a future live path … through
> `make_live_book_tracker()`, and/or have the tracker auto-feed its book to the
> dossier reviewer.

The D17 soft-warns (R7 VaR, R8 stress + dealer-regime, R9 sector cap, R10
single-name cap) and the tracker's hard refusals only fire when a populated
`PortfolioContext` / live book is present. Today nothing populates one outside
tests and the S47 harness. **The engine has the guards but has never had a book
to point them at.**

### 0.1 The integration is the "future live path" the docs anticipated

Interactive Brokers, connected read-only, supplies exactly the missing inputs:

| Engine input it needs | IBKR read tool that supplies it |
|---|---|
| `nav` (nav for every gate) | `get_account_summary` → `net_liquidation` |
| `held_option_positions: list[dict]` (R9/R10 notional math) | `get_account_positions` |
| margin headroom / stress realism | `get_account_summary` → `available_funds`, `excess_liquidity`, `maintenance_margin` |
| live spot / IV for freshness checks | `get_price_snapshot`, `get_price_history` |
| realized-P&L reconciliation | `get_account_trades` |

### 0.2 Evidence it is not theoretical (live probe, 2026-06-05)

A read-only probe of the connected account produced a textbook case for every
dormant gate. Summarised here as the worked example the design must handle (full
numbers in Appendix A):

- **NAV $144,507**, but **available funds −$9,257**, **buying power $0**,
  **excess liquidity $4,746** — a margin-stressed book.
- **R10 (single-name 10% NAV) breached on every name with a short put:** CLS
  short-put notional ≈ **$212,500 (147% of NAV)**, MU ≈ $97k (67%), AMD $50k
  (35%), MRVL $29.75k (21%). Total CSP notional ≈ **$389k (269% of NAV)**.
- **R9 (sector 25% NAV) grossly breached:** AMD/CLS/MU/MRVL/NVDA/TSM are all
  semis / tech-hardware — the overwhelming majority of the book in one GICS
  sector.
- **Out-of-universe names present:** CNQ, ENB (Canadian / TSX) — the design must
  account for these as exposure-only, since the engine universe is S&P 500.

None of this is visible to the engine today. The integration makes it visible
*and* actionable by the existing reviewers — without touching the trio.

---

## 1. The one architectural constraint that shapes everything

**The IBKR MCP connection is an agent-time capability, not a runtime Python
SDK.** It lives inside a Claude Code session; `engine/` cannot call it mid-scan.

This is not a limitation to fight — it maps cleanly onto how the data layer
already works. The engine consumes **files on disk** (Bloomberg CSVs, Theta
parquets, the news-sentiment parquet). So the live book enters the same way:

```
  IBKR MCP  ──(agent or dedicated read-only puller)──▶  data_processed/ibkr/portfolio_snapshot.json
                                                                   │  point-in-time, timestamped, schema-versioned
                                                                   ▼
                                              engine/ibkr_portfolio_adapter.py   (NEW, outside the trio)
                                                                   │  parse → normalize → build PortfolioContext
                                                                   ▼
                            WheelRunner.build_candidate_dossiers(..., portfolio_context=ctx)
                                                    R7–R11 fire ▶ downgrade-only verdicts
                                                                   ▼
                                   dashboard "Live Book" panel  +  operator memo
```

Two acquisition modes, both read-only, decided per-deployment (proposed D24 picks
the default):

1. **Agent-produced snapshot (Phase 1 default).** A Claude session (or a thin
   `scripts/pull_ibkr_portfolio.py` driven through the MCP) writes the snapshot
   JSON. Zero new broker credentials in the engine. Matches the "point-in-time
   artifact on disk" discipline of `docs/DATA_POLICY.md`.
2. **Dedicated read-only puller (later, optional).** A standalone process using
   IBKR's own Client Portal / `ib_insync` API to refresh the snapshot on a
   schedule. Bigger lift, more moving parts, and closer to the §3 line (still
   read-only, but a live broker connection in-process) — explicitly deferred and
   gated behind its own decision.

**Why a file and not a live call:** point-in-time honesty (the snapshot is
timestamped and auditable, like every other tier-1 input), testability (fixtures
are just JSON), and it keeps the broker out of the hot decision path. The
SessionStart hook can warn on snapshot staleness exactly as it already does for
`sp500_ohlcv.csv`.

---

## 2. In-scope design (proposed D24)

### 2.1 New module: `engine/ibkr_portfolio_adapter.py` (outside the CI-gated trio)

Single responsibility: **IBKR snapshot JSON → the engine's existing types.** It
imports nothing from `ev_engine` / `wheel_runner` / `candidate_dossier`; it only
*produces* a `PortfolioContext` and a `held_option_positions` list that those
modules already accept. Target seams (as of this writing):

- `engine/portfolio_risk_gates.py:57` — `class PortfolioContext` (the D17 gate
  inputs R7–R11 read). **This is the target type**, *not* `advisors/schema.py:120`
  `PortfolioContext` (a different class for the advisor committee — the doc must
  not conflate them).
- `engine/portfolio_risk_gates.py:343` `check_sector_cap(...)`, `:416`
  `check_single_name_cap(...)`, `:529` `check_var(...)`, `:630`
  `check_stress_scenario(...)` — all take `held_option_positions: list[dict]` +
  `nav`. The adapter's job is to produce those two arguments faithfully.
- `engine/wheel_runner.py:3365` `WheelRunner.build_candidate_dossiers(...)` — the
  attach point; gains an optional `portfolio_context=` (or loads the snapshot
  when a path/env is set), mirroring how `engine_api.py`'s
  `_build_portfolio_context_from_params` builds a context from `nav` query params
  today.
- `engine/wheel_runner.py:357` `make_live_book_tracker(...)` — the canonical
  armed constructor with **zero non-test callers** (per #319). A live operator
  path is the intended first real caller.

### 2.2 Snapshot schema (illustrative — to be versioned)

```jsonc
{
  "schema_version": 1,
  "as_of": "2026-06-05T21:36:00Z",        // UTC, drives staleness warnings
  "base_currency": "USD",
  "account": {
    "net_liquidation": 144507.08,
    "total_cash": 105122.78,
    "available_funds": -9256.99,           // negative ⇒ margin-stressed
    "excess_liquidity": 4746.71,
    "maintenance_margin": 184250.73,
    "buying_power": 0.0
  },
  "positions": [
    {"symbol": "MU", "sec_type": "OPT", "right": "P", "strike": 970,
     "expiry": "2026-06-12", "qty": -1, "mark": 116.62, "avg_price": 35.69,
     "unrealized_pnl": -8093.0, "currency": "USD", "in_universe": true},
    {"symbol": "CNQ", "sec_type": "STK", "qty": 100, "mark": 63.84,
     "currency": "CAD", "in_universe": false}
    // ...
  ]
}
```

The adapter derives `held_option_positions` (notional = `strike × 100 × |qty|`
for short options) and `nav` (= `net_liquidation`, FX-normalized) from this.

### 2.3 The universe filter (a required design element, not an afterthought)

The live book contains names outside the engine's S&P-500 + wheel scope (CNQ,
ENB; and short *strangles* more aggressive than "short put + covered call").
Rule:

- **In-universe S&P-500 names** → full gate treatment (counted in NAV, sector,
  and single-name math; rankable).
- **Out-of-universe names** → **exposure-only**: they still consume NAV and count
  toward concentration denominators (a CAD energy position is real risk), but the
  engine never ranks or evaluates them, and never claims authority over them.
- The GICS `sector_map` already consumed by `check_sector_cap` is reused; the
  adapter must supply sectors for in-universe names and bucket the rest as
  `"non_universe"`.

### 2.4 The four in-scope tracks (all downstream of the adapter)

| Track | Deliverable | Wires into |
|---|---|---|
| **A. Live PortfolioContext** | Arm R7–R11 against the real book on every scan | `build_candidate_dossiers(portfolio_context=…)` |
| **B. Reconciliation** | IBKR positions/trades vs the `WheelTracker` ledger; surface drift | `make_live_book_tracker()`, `portfolio_context_snapshot()` (`wheel_tracker.py:1666`) |
| **C. Freshness overlay** | Live spot/IV vs the 77-day-stale CSVs; a live cousin of R3 (spot-mismatch) that downgrades a scan running on drifted data | new pre-scan guard / reviewer (downgrade-only) |
| **D. Dashboard "Live Book" panel** | NAV, margin/excess-liquidity, per-name & per-sector exposure vs R9/R10 caps, which gates trip, deep-ITM at-risk shorts | read endpoint on `engine_api` + `dashboard/` |

### 2.5 Proposed `DECISIONS.md` D24 (DRAFT — not adopted)

> **D24 — IBKR live-book feed via point-in-time snapshot (read-only, gate-arming).**
>
> **Decision.** Introduce a read-only IBKR → `data_processed/ibkr/portfolio_snapshot.json`
> point-in-time artifact and an `engine/ibkr_portfolio_adapter.py` (outside the
> trio) that builds a `engine/portfolio_risk_gates.py` `PortfolioContext` +
> `held_option_positions` from it. Attach the context to
> `build_candidate_dossiers`, arming R7–R11 against the operator's real book.
> No order-routing surface; no edits to `ev_engine` / `wheel_runner` reviewer
> logic / `candidate_dossier`; out-of-universe names are exposure-only.
>
> **Why.** Closes the #319 "documented protection ≠ active protection" gap by
> supplying the live book the D17 gates were built for, via the same
> file-on-disk, point-in-time pattern the rest of the data layer already uses —
> keeping the broker out of the hot decision path and out of §3.
>
> **Rejected alternatives.** (1) *In-process live IBKR API call during a scan* —
> puts a broker connection in the hot path, breaks point-in-time auditability,
> harder to test; deferred to an optional dedicated puller behind its own
> decision. (2) *Flip the library `WheelTracker` default to armed* — D22 already
> rejected this (moves pinned backtest snapshots); the live path arms via
> `make_live_book_tracker()` instead. (3) *Feed the book straight into the trio
> reviewers* — would edit CI-gated files and risk an authority leak; the
> `PortfolioContext` parameter is the sanctioned downgrade-only seam.
>
> **Pinned by.** (planned) `engine/ibkr_portfolio_adapter.py`,
> `tests/test_ibkr_portfolio_adapter.py` (snapshot→context fidelity, universe
> filter, FX normalization, never-rescues-negative-EV), a dossier test that R7–R11
> fire on an adapter-built context, and a SessionStart staleness check on the
> snapshot.

---

## 3. Scope frontier — the exit-evaluator (proposed D25, NOT adopted)

This part exists because the brainstorm surfaced that **the most valuable thing
for day-to-day operation is the thing the engine does not do.**

### 3.1 The gap

The engine is an **entry ranker**: `EVEngine.evaluate` scores *opening* a fresh
short put / covered call. It has **no exit-EV evaluator** — no first-class notion
of "should I roll, close, or take assignment on a position I already hold." S47's
finding put it exactly: **"TRUST entry, DISTRUST management."**

Our MU/GOOGL session was entirely a *management* problem. The MU short 970 put is
−$8,093 with the account at $0 buying power; the real decision is roll vs close
vs assign, and the engine is silent on it.

### 3.2 Why this is a scope expansion, not a feature add

CLAUDE.md §2's invariant is written around *candidates* flowing through
`EVEngine.evaluate` to a tradeable verdict. An exit-evaluator is a **different
kind of evaluation**: its inputs are an *open* position (cost basis, current
mark, assignment mechanics, margin impact), and its output is an *action on an
existing obligation*, not a new candidate. That is new surface area — it deserves
its own invariant, not a quiet extension of the entry path. Hence: design it on
the record, adopt it only with explicit consent.

### 3.3 Design sketch (for discussion, not commitment)

An `engine/exit_evaluator.py` (outside the trio) that, given an open position
from the IBKR snapshot, scores the three wheel exits on a **comparable, tail-aware
basis** — reusing existing machinery, never bypassing it:

- **Close now** — realize `(avg_price − mark) × 100`. Deterministic.
- **Take assignment** (puts) / **call-away** (covered calls) — effective basis
  `strike ∓ net_credit`; P&L distribution from the **same forward-distribution +
  POT-GPD tail** stack the entry path uses (`engine/forward_distribution.py`,
  `engine/tail_risk.py`), over the remaining DTE. Crucially, **margin-feasibility
  gated**: with `available_funds < 0` / `buying_power == 0`, assignment may be
  infeasible (forced liquidation) — the evaluator must read `excess_liquidity`
  from the snapshot and flag this, as in the live MU case.
- **Roll** — close + a fresh `EVEngine.evaluate` on the replacement strike/expiry,
  so the roll's *new* leg still passes the standard entry authority (no §2
  bypass: the new short is a normal candidate).

Output is advisory/ranked, surfaced in the memo + dashboard — **still no order
routing.** The human executes at IBKR.

### 3.4 Hard invariants the exit-evaluator must inherit

- The **roll's new leg is a candidate** and goes through `EVEngine.evaluate`
  unchanged — it cannot be rescued by "but I'm rolling."
- The evaluator is **advisory**; it never auto-acts.
- **Margin feasibility is a first-class gate**, not a footnote (the live probe
  shows why: the "optimal" path can be the one the broker won't let you take).

### 3.5 Proposed `DECISIONS.md` D25 (DRAFT — explicitly NOT adopted)

> **D25 — Position-management (exit) EV evaluator. STATUS: PROPOSED, NOT ADOPTED.**
>
> **Decision (proposed).** Add an advisory `engine/exit_evaluator.py` that scores
> roll / close / assign on held positions using the existing forward-distribution
> + tail-risk stack, margin-feasibility-gated from the IBKR snapshot, with any
> roll's new leg routed through `EVEngine.evaluate`. Advisory only; no order
> routing. Requires operator greenlight before implementation — it expands the
> engine's scope from entry-ranking to position management.
>
> **Why (the case for).** Management, not entry, is the operator's live pain
> (S47: "TRUST entry, DISTRUST management"); the IBKR feed makes the needed
> inputs available; reusing the entry stack keeps it tail-honest and §2-aligned.
>
> **Rejected / open alternatives.** (1) *Keep entry-only, leave exits manual* —
> the current state; lowest scope risk, highest operator burden. (2) *Bolt
> roll-logic onto the entry path* — rejected: conflates two evaluation kinds and
> risks a §2 bypass on the roll. (3) *A heuristic roll-rule (e.g. roll at 21 DTE
> / 50% profit)* — cheap but not tail-aware and not margin-aware; would mislead
> in exactly the high-IV, low-buying-power case (MU) where it matters most.
>
> **Pinned by.** (none yet — proposal stage; to be pinned on adoption.)

---

## 4. Hard lines preserved (the whole way through)

- **No order routing.** `create_order_instruction` / `delete_order_instruction`
  are never wired into the engine. Loop = **engine ranks → human executes at
  IBKR**.
- **Downgrade-only.** The live book can move a verdict negative→blocked /
  proceed→review; it can never rescue a negative-EV candidate. Dealer multiplier
  clamp `[0.70, 1.05]` untouched.
- **Trio untouched.** No edits to `ev_engine.py` / `wheel_runner.py` reviewer
  logic / `candidate_dossier.py`; integrate via the `PortfolioContext` parameter
  and the `make_live_book_tracker()` factory only.
- **Universe discipline.** Out-of-scope names are exposure-only; the engine never
  ranks or claims authority over non-S&P-500 / non-wheel positions.
- **Branch + PR + tests + a real D-number** for every code change; run the full
  suite for any decision-layer-adjacent touch (CLAUDE.md §4.3).

---

## 5. Phased plan & decision points

| Phase | Scope | Gating |
|---|---|---|
| **1** | Snapshot artifact + `ibkr_portfolio_adapter.py` + run real book through `portfolio_risk_gates`, emit R7–R11 verdicts (formalize the manual probe) | D24 adoption |
| **2** | Dashboard "Live Book" panel reading the snapshot (Track D) | after Phase 1 |
| **3** | Wire snapshot into live `/api/tv/dossier` so a real scan's candidates are gated against the book (Track A end-to-end) | after Phase 1 |
| **4** | Reconciliation (Track B) and/or the exit-evaluator (Track E / D25) | **explicit consent** (D25) |
| **PV** | Performance viewer (D26): `/api/portfolio/*` + `/(terminal)/portfolio` + risk overlay; agent layer phased (chat → in-dashboard). Reuses the D24 feed + existing trackers | D26 adoption; can run parallel to Phases 2–3 |

**Open decisions for the operator:**
1. Adopt **D24** (in-scope live-book feed)? Default acquisition mode = agent-produced snapshot.
2. Greenlight **D25** (exit-evaluator) as a scope expansion, or hold entry-only?
3. Snapshot refresh cadence + staleness threshold for the SessionStart warning.
4. FX policy for the multi-currency book (USD + CAD) in NAV / concentration math.
5. Adopt **D26** (read-only performance viewer)? Agent layer is phased per operator decision
   (Claude session now → in-dashboard `chat-panel` later). Also: persist daily snapshots
   for the equity curve, and ingest the Flex `CashTransactions` export for since-inception TWR.

---

## 6. Personal performance viewer (proposed D26 — design only)

This part exists because the brainstorm surfaced a second, complementary product
on the same IBKR feed: a **read-only personal portfolio performance viewer** — a
Sharesight / IBKR-PortfolioAnalyst-style surface, but **scoped to one book,
wheel-aware, and conversational.** It is the *realized, backward-looking*
counterpart to the engine's *forward-looking* ranker, and it is squarely
in-scope: purely observational, single-user, no EV authority, no order routing.

### 6.1 It is ~60–70% already built (unwired)

Three production analytics modules exist today with **zero UI**, plus a
charting-ready front end:

| Existing asset | Already computes | Status |
|---|---|---|
| `engine/portfolio_tracker.py` | TWR; 1D/1W/1M/3M/YTD/1Y/All returns; snapshot history; sector/asset allocation; **benchmark comparison** | live, no UI |
| `engine/wheel_tracker.py` | position lifecycle (short put→assignment→covered call→exit); realized/unrealized P&L; roll suggestions; win rate | live, no UI |
| `engine/performance_metrics.py` | Sharpe, Sortino, drawdown | live, no UI |
| `engine_api.py` `GET /api/portfolio` | per-ticker portfolio report | live, **not wired to UI** |
| `dashboard/` (Next.js 16, Recharts, Tailwind, shadcn/ui, SQLite+Drizzle) | the `(terminal)` route group (`/cockpit`, `/terminal`), a streaming chat panel, Ollama memos | live |

So the analytics brain, the chart library, the API server, and page routing all
exist. **D26 is wiring + a live data source — the same D24 IBKR snapshot.**

### 6.2 Three layers

1. **Visual layer** — a `/(terminal)/portfolio` page (sibling to `/cockpit`):
   equity curve + period returns vs SPY; holdings table with the wheel state
   machine (short-put / assigned-stock / covered-call); **premium-income view**
   (the metric brokers can't show because they don't know it's a wheel);
   sector/asset/currency allocation; realized P&L by name / week / month (the
   exact views surfaced manually in the 2026-06-05 session).
2. **Live risk overlay** — the same R7–R11 read from D24 (concentration vs
   R9/R10 caps, margin stress, VaR/stress). A performance viewer that also shows
   *where the book is about to hurt* — which no consumer platform does.
3. **Conversational agent layer (BOTH, phased — per operator decision).**
   Phase A: the Claude Code session is the query layer (the 2026-06-05 chat —
   "deposits?", "trades this week", "GOOGL P&L", "MU assignment math" — is the
   working prototype). Phase B: surface the same engine-backed query tools in the
   dashboard's existing `chat-panel.tsx`, routing portfolio questions to the
   trackers + IBKR snapshot in-app. Dashboard handles the ~10 standard views; the
   agent handles the long tail.

### 6.3 The novel, on-mission feature: closing the forecast→outcome loop

PROJECT_STATE records finding **I1**: `ev_dollars` has ≈0 rank-correlation with
realized P&L — the engine has never measured whether its *forecasts* pay off on
live trades. If the viewer **tags each position with the engine's original EV
verdict / prob_profit at entry**, it can finally compare forecast vs. realized
outcome on the real book — a feedback signal no broker platform can produce, and
a direct advance on the project's core open question.

### 6.4 Honest gaps (scope this right)

1. **Since-inception / money-weighted return needs the deposit ledger**, which
   the IBKR *MCP* does not expose (confirmed in-session). Ingest the IBKR **Flex
   `CashTransactions` export** as an on-disk artifact (same pattern as D24).
   Until then: mark-to-market P&L, period returns from snapshot deltas, and
   realized P&L from trades — but not a clean TWR/IRR since inception.
2. **The equity curve needs history to accumulate** — one snapshot is a point.
   Persist daily snapshots (a `portfolio_snapshots` table in the dashboard's
   existing SQLite/Drizzle DB, or parquet on disk); starts thin, optionally
   backfilled from trade history.
3. **Multi-currency (USD + CAD)** needs the same explicit FX policy as D24.

### 6.5 Architecture (reuses the D24 feed end-to-end)

```
IBKR (D24 snapshot JSON  +  Flex CashTransactions export)  ──▶  data_processed/ibkr/
        │
        ├─▶ portfolio_tracker / wheel_tracker / performance_metrics   (analytics)
        ├─▶ portfolio_risk_gates  R7–R11                              (live risk overlay)
        └─▶ portfolio_snapshots (accumulates → equity curve)
                        │
        engine_api:  /api/portfolio/{summary,positions,returns,income,risk,history}
                        │
        Next.js /(terminal)/portfolio  (Recharts)   +   chat-panel agent (NL queries)
```

### 6.6 Proposed `DECISIONS.md` D26 (DRAFT — not adopted)

> **D26 — Read-only personal performance viewer. STATUS: PROPOSED, NOT ADOPTED.**
>
> **Decision (proposed).** Wire the existing `portfolio_tracker` /
> `wheel_tracker` / `performance_metrics` analytics + the D24 IBKR snapshot into
> new `engine_api` `/api/portfolio/*` endpoints and a `/(terminal)/portfolio`
> Next.js surface, with a live R7–R11 risk overlay and a phased conversational
> agent (Claude session now; in-dashboard `chat-panel` later). Read-only,
> single-user, observational; no EV authority, no order routing. Optionally tags
> positions with their entry-time EV verdict to measure forecast vs. realized
> outcome (the I1 loop).
>
> **Why.** ~60–70% of it already exists unwired; it gives the operator the
> real-portfolio view brokers give *plus* the engine's wheel-native income view,
> live risk read, and forecast→outcome feedback — on the same IBKR feed D24
> already introduces.
>
> **Rejected alternatives.** (1) *Use a third-party tracker (Sharesight/IBKR
> PortfolioAnalyst)* — gives generic performance but no wheel-state, no premium
> income, no R7–R11 overlay, no forecast→outcome loop; the differentiators are
> exactly the engine-native parts. (2) *Recompute analytics in the React/TS
> frontend* — rejected: shadows Python logic, drifts from the trackers; the
> engine API stays the single source of truth. (3) *Block on a full
> since-inception TWR* — rejected: ship mark-to-market + period + realized views
> first; layer money-weighted return once the Flex `CashTransactions` ingest
> lands.
>
> **Pinned by.** (planned) new `/api/portfolio/*` endpoint tests, a
> `portfolio_snapshots` persistence test, a forecast-vs-outcome tagging test, and
> a viewer-never-claims-EV-authority guard (the viewer is observational only).

### 6.7 Scope note

D26 never converts data into a tradeable verdict and never routes orders; it
*reads* the book and *reports*. It is the descriptive complement to the engine's
prescriptive ranker, and it must keep that line crisp: realized P&L (what
happened) is displayed distinctly from `ev_dollars` (a forward tail-aware score
that, per I1, does not forecast realized P&L).

---

## Appendix A — Live probe (2026-06-05, read-only) as the worked example

Account: NAV **$144,507**; available funds **−$9,257**; buying power **$0**;
excess liquidity **$4,746**; maintenance margin **$184,251**; unrealized P&L
**−$37,014** (BASE). Multi-currency: USD $126,670 + CAD $24,863.

Short-put book (the R10 stress): CLS 415/420/430×3 (notional ≈ $212.5k, 147% of
NAV), MU 970 (mark $116.62, −$8,093, 67% of NAV), AMD 500 ($50k), MRVL 297.50
($29.75k). Total CSP notional ≈ $389k (269% of NAV). Sector: AMD/CLS/MU/MRVL/
NVDA/TSM all semis/tech-hardware → gross R9 breach. Out-of-universe: CNQ, ENB
(TSX) → exposure-only.

This is the fixture the Phase-1 adapter + gate run should reproduce.
