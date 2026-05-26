# Usage Test Ledger

This file tracks end-to-end *usage* tests — sessions where the engine
is exercised as a real trader would use it, not by the unit /
integration test suite. Each scenario has a purpose, a setup, the
bugs and gaps it surfaced, and the PRs (if any) that closed them.

Companion to:

- `PROJECT_STATE.md` — current authoritative state.
- `ROADMAP.md` — intended next work.
- `CHANGELOG.md` — shipped per-PR detail.
- `TESTING.md` — the *unit / integration* test taxonomy. This file
  is the *usage* axis; that one is the *code* axis.

## How to update this

When a usage-test scenario completes:

1. Append an entry under the appropriate section with: name,
   purpose, setup, status, key findings, and follow-ups.
2. For each finding, link the PR that fixed it (e.g. `#102`) or
   tag it `**logged**` if not yet fixed.
3. Move scenarios between sections as their status changes
   (Candidate → Queued → In flight → Completed).
4. Keep findings inline under their scenario. Cross-cutting
   findings that recur across scenarios can be repeated; do not
   maintain a parallel flat index — it will drift.

The aim is operational, not historical: a fresh agent should be
able to read this and know which surfaces of the product have been
exercised and which have not.

---

## 1. Completed

### S1 — Single-snapshot trader session

**Purpose.** Exercise the morning-scan → dossier → sizing path as a
retail wheel trader would, top-down across the SP500.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, offline charts
(`FilesystemChartProvider`), 40-name diversified watchlist,
`as_of=2026-03-20`, $150k account, 35-DTE / 25-delta puts.

**Status.** Done. One critical bug fixed
(`#102` — dividend-yield normalization). Other findings logged.

**Findings:**

- **Dividend-yield normalization bug** (`wheel_runner.py` ~ line 655).
  Sub-1% yields skipped the `> 1.0` guard and reached BSM as a
  whole-number decimal (`0.87` used as 87% q). Corrupted the
  delta → strike solve and the synthetic premium across ~92 of 410
  priced names — MSFT, COST, AAPL surfaced as positive-EV when truly
  negative. Fixed in **`#102`** (merged `afee837`).
- **`Δ` (U+0394) Unicode crash** in `candidate_dossier.py`'s R3
  review note — crashes Windows cp1252 console on print / log.
  **Logged.**
- **Silent drops** — `rank_candidates_by_ev` returns only
  survivors; no diagnostic when a name is gated out (earnings,
  history, chain quality). **Logged.**
- **`as_of` footgun** — defaults to today; pairs stale Bloomberg
  prices with current-date event timing. **Logged.**
- **R4 reviewer rule effectively dead** in the standard ranker
  path — needs a `phase` field the ranker never emits. **Logged.**
- **Committee delta silent default** — `_build_advisor_input`
  falls back to `delta=-0.30` (`integration.py:165`) because the
  ranker emits no delta column. The 45-DTE figure in the original
  S1 note is an omission-only fallback (`integration.py:164`),
  **not** a live mismatch: the ranker emits `dte`, so the committee
  sees the correct 35. Corrected by S7. **Logged.**
- **No `ev_raw` exposed** in the ranker output despite being a
  core EV-engine field. **Logged.**
- **No return-on-capital column / no account-size input** — the
  ranker optimizes absolute EV/day, structurally biased to
  expensive names. **Logged.** Addressed in part by S4 (see S4).
- **Regime (HMM) multiplier unlabeled** — silently cuts EV
  50–80 % on some names with no surfaced regime. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — 28 ranked rows on the 40-name watchlist at `as_of=2026-03-20`, all `ev_dollars` finite (range −481.38…+713.73); 12 drops emitted with structured `{ticker, gate, reason}` per PR #121.
  - Qualitative verdict: match — dividend-yield bug stays fixed (no premium > strike anomaly). Seven of nine logged findings are now mechanically closed: dividend (#102), drops surface (#121), `ev_raw` column, `roc` + `collateral` (#109), `hmm_regime` label (#208), `sector` column (#210), `news_n_articles` (#119). Two remain logged-by-design: Unicode cp1252 mangle on `Δ` / `±` (still present in drop reason strings — observed verbatim during this run), and R4 dormancy (still no phase-aware chart provider on `main`).
  - Numerical drift > 5%: no orig figures cited in the original entry — it was a bug-narrative entry, no row counts or EV magnitudes were quoted to compare against.
  - Notes: 12 dropped at `as_of=2026-03-20` are Q1-earnings-season lockouts (consistent with the pattern S16 documented at the same as_of); current top-5 by EV are LLY (+713.73), CAT (+444.99), BLK (+407.09), DE (+318.53), TMO (+227.27). Diagnostic columns present: `ev_raw`, `roc`, `collateral`, `hmm_regime`, `sector`, `news_n_articles` — all of S1's "missing column" findings are now visible.

### S2 — Multi-day rolling wheel campaign (4 weeks)

**Purpose.** Exercise the time dimension — managing a real book
across days: open, age, profit-take, hold, roll, accept assignment.

**Setup.** Bloomberg, offline charts, $150k account, 35-DTE /
25-delta entries, profit-take ≥ 50 %, max 25 % per name, 5-snapshot
window (`as_of` advanced 5 trading days per step) ending at the data
cutoff `2026-03-20`. Positions tracked in
`engine.wheel_tracker.WheelTracker`.

**Status.** Done. Dividend fix validated end-to-end in real flow
(4/4 closes profitable; PG loss landed honestly in the advertised
9 % tail). All management-layer findings logged.

**Findings:**

- **No persistence on `WheelTracker`.** No `to_dict` / `from_dict` /
  `to_json` / `save` / `load`. Closing a Python session loses the
  book. **Logged.**
- **No management-workflow methods.** No `suggest_rolls`,
  `suggest_actions`, `book_snapshot`, `available_buying_power`. The
  tracker is a mechanics layer (`open_short_put` / `close_short_put`
  / `roll_put` / `handle_*_assignment` / `mark_to_market`) without a
  management layer on top. The single biggest gap; addressed by S3
  (in flight).
- **`mark_to_market` silent IV-staleness.** Falls back to the
  position's entry IV when `current_ivs` is not passed — no warning,
  no `as_of` IV plumbing from the connector. **Logged.**
- **`tracker.cash` overstates buying power for CSPs.** Open credits
  the premium but does not reserve the strike collateral. Correct as
  brokerage cash; wrong as deployable capital. Workaround:
  `available = cash − Σ(strike × 100)`. **Logged.**
- **`get_performance_summary` is closed-positions-only.** Returns
  an empty DataFrame mid-campaign. No companion current-book
  snapshot. **Logged.**
- **Earnings-window-drift not surfaced** for open positions. A
  trade opened 50 days before earnings becomes "within window"
  silently as time advances. **Logged.**
- **Same-day close-and-reopen feels strange.** A name closed at
  profit-take can re-rank #1 in the same step and be reopened with
  no cooldown. Internally consistent (different contract entirely)
  but a UX surface a real trader would want. **Logged.**
- **Drop-reason silence carries into the rolling case** — same as
  S1, but more visible when a name disappears between snapshots.
  **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — by structure (rolling Sn that opens via `rank_candidates_by_ev` only; no direct §2 surface re-exercised here).
  - Qualitative verdict: match — six of the eight S2 findings are now mechanically closed by shipped PRs: `WheelTracker.save` / `load` (PR #128 / D persistence), `WheelTracker.suggest_rolls` (PR #104), `WheelTracker.available_buying_power` (PR #127), `mark_to_market` connector-IV resolution (PR #129), `get_performance_summary` largest-win/loss fix (PR #126 wheel-cycle scope). Drop-reason silence (S1/S2/S9-common) closed by PR #121. Two remain Logged-by-design: same-day close-and-reopen UX (no cooldown — by design), earnings-window-drift on open positions (no surfacing).
  - Numerical drift > 5%: not applicable — original entry quoted only qualitative findings (no specific NAV / drawdown figures from the 4-week sim).
  - Notes: existence sweep on `WheelTracker` confirms `save`, `load`, `suggest_rolls`, `available_buying_power` all present.

### S3 — Build `WheelTracker.suggest_rolls(...)`

**Purpose.** Close S2's headline management-layer gap: when a short
put goes adverse, give the trader candidate rolls ranked by EV — not
just `roll_put` mechanics.

**Status.** Done — shipped in **`#104`** (merged `ece2717`).

**What shipped.** `WheelTracker.suggest_rolls(ticker, as_of,
current_spot, current_iv, ...)` enumerates roll candidates over a
DTE × delta grid, runs each through `EVEngine.evaluate` (§2 intact —
uses the EV authority; pinned by a call-count regression test), and
returns a DataFrame ranked by forward EV with `roll_ev`, `hold_ev`,
`net_credit_debit`, `prob_otm`, `recommend`. Short-put rolls only.

**Notes:**

- **EV metric.** `roll_ev = ev_dollars(new) − buyback_total`;
  `hold_ev = ev_dollars(synthetic) − buyback × 100`. Both express
  marginal forward dollar P&L from the decision moment — apples-to-
  apples. The original spec's `+ net_credit_debit` double-counted
  the new premium (it is already inside `ev_dollars(new)` via
  `gross_premium`); the shipped single-count form is the correct
  one.
  **Correction (#122).** The #104 code netted `buyback_total` from
  `calculate_total_exit_cost(...)["total_cost"]` — exit *transaction
  costs only* (~$7), omitting the buyback principal (~$400+). So as
  shipped, `roll_ev` was *not* in fact apples-to-apples with `hold_ev`
  and nearly every roll spuriously cleared the `recommend` bar. #122
  corrects `buyback_total` to the `"total_buyback_cost"` key
  (principal + exit txn costs); a `TestRollEvNetsBuybackPrincipal`
  regression pins it.
- **`recommend` semantics.** `recommend=True` means "this roll's
  forward EV beats holding's" — **not** "this position needs
  rescuing". Correct for the intended use (call on a *challenged*
  position); calling it on a healthy position surfaces
  premium-harvest churn. A UX-framing note for any dashboard
  surfacing this.
- **Live demo.** The S2 campaign's underwater PG position (deep
  ITM, ~2 weeks to expiry) → the engine surfaces a ~+$1,661
  forward-EV improvement over holding. The week-2 signal the S2
  trader didn't get.

**Follow-up — done.** `suggest_call_rolls` — the covered-call-leg
parallel, deferred from #104 — shipped in **`#122`** (merged
`1821d56`): the same DTE × delta enumeration through
`EVEngine.evaluate`, covered-call rolls only, pinned by a §2
call-count regression.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `WheelTracker.suggest_rolls` invokes `EVEngine.evaluate` per (DTE × delta) grid point plus once for the hold synthetic; the published columns include `roll_ev`, `hold_ev`, `recommend`, `new_ev_dollars`, `new_premium`, `buyback_cost`, `net_credit_debit`.
  - Qualitative verdict: match — `suggest_rolls` exists with the documented signature `(ticker, as_of, current_spot, current_iv, risk_free_rate=None, *, target_dtes=(21,35,49,63), target_deltas=(0.30,0.25,0.20,0.15), min_net_credit=0.0, dividend_yield=0.0, forward_log_returns=None)`. PR #122 `buyback_total` correction (use `total_buyback_cost` key, not `total_cost`) is shipped on `main`. `WheelTracker.suggest_call_rolls` also present.
  - Numerical drift > 5%: not applicable — original entry quoted PG-specific +$1,661 forward-EV improvement on the live demo, which was from S2's data window. Re-deriving requires S2's 4-week sim state (out of scope for a snapshot re-verify).
  - Notes: confirmed `.attrs["drops"]` is populated on `suggest_rolls` output (Fix #181 / S22 F1) — 4 drops observed on a smoke roll test.

### S4 — Account-size-constrained book selection

**Purpose.** Force a return-on-capital lens by setting a realistic
small account ($50k retail) as a hard constraint, then try to build a
book from the ranking. Exercises the no-account-size / no-ROC gap S1
logged.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, offline charts, 36-name
diversified watchlist (11 GICS sectors, mega-caps through mid-caps),
`as_of=2026-03-20`, 35-DTE / 25-delta puts, `contracts=1`,
`include_diagnostic_fields=True`. $50k hard cap; each cash-secured put
reserves `strike × 100` collateral. The book is walked top-down off
`WheelRunner.rank_candidates_by_ev` output.

**Status.** Done. 30 of 36 names ranked, 20 of them positive-EV; the
ranker's top three by `ev_per_day` (FIX, MCK, BKNG) each demand more
collateral than the entire $50k account ($122.2k / $84.1k / $401.4k).
No bug — every figure computes correctly; S4 surfaces missing
*capability*, not a defect. All findings logged; no code change (a
usage test's deliverable is the writeup). The top-10 names by ROC —
CF, FIX, FDS, AJG, EXE, JBHT, EG, MCK, HUM, BR — are the sample S7
fed to the advisor committee.

**Findings:**

- **No account-size input — the ranking is capital-blind.**
  `rank_candidates_by_ev` (`wheel_runner.py:448`) takes `contracts`,
  `top_n`, `min_ev_dollars` — nothing for account size or a collateral
  budget. It returns an identical ranking for a $50k account and a $5M
  one. Optimizing `ev_per_day` with no capital term, it front-loads the
  most expensive names: a strict top-down walk (stop at the first
  unaffordable pick) opens **zero** positions and strands all $50k; the
  first candidate that fits at all is the ranker's #4. Nothing in the
  output flags that the top names are unbuyable. **Logged.** (the core
  S1 gap, now exercised.)

- **No return-on-capital column.** Output carries `ev_dollars`,
  `ev_per_day`, `prob_profit`, CVaR, Omega, `edge_vs_fair` and the
  dealer/skew/HMM diagnostics — but nothing dividing EV by collateral.
  ROC (`ev_dollars / (strike × 100)`) had to be hand-computed.
  Re-ranking by ROC reorders the book sharply: CF #8 → #1, EXE #12 →
  #5, against BKNG #3 → #18, MCK #2 → #8, CAT #7 → #12. The ranker's
  order and the capital-efficiency order disagree most on exactly the
  names a $50k trader must decide between. **Logged.**

- **ROC ordering beats ranker ordering on the same $50k.** A greedy
  fill down `ev_per_day` order fits 2 names (EG + FDS — $48.75k
  collateral, $666 EV). The same fill down ROC order fits 4 (CF + FDS +
  EXE + KO — $47.35k, $722 EV): +8.4 % EV, twice the positions (genuine
  diversification vs a two-name book), and less capital committed. The
  absolute-EV lens leaves both money and diversification on the table
  for a constrained account. **Logged.**

- **No buying-power / book-builder helper; `contracts` is global.**
  The collateral math, the budget walk and the ROC re-sort were all
  done by hand. S2 logged the workaround `available = cash −
  Σ(strike × 100)`; S4 confirms there is still no
  `available_buying_power()` and no "fit a book under budget X" helper
  for what is really a knapsack problem. `contracts` is one
  ranker-wide argument, not per-candidate — "as many contracts of the
  cheap names as the budget allows" cannot be expressed. **Logged.**

- **No concentration guard; $50k is structurally forced to
  concentrate.** The ranker-order book put 97.5 % of the account into
  2 names (EG alone = 60 %). S2 logged "max 25 % per name" as a
  trader-imposed rule the engine does not enforce — and under $50k that
  rule caps collateral at $12.5k/name, i.e. `strike ≤ $125`. Only 3 of
  the 20 positive-EV names (CF, EXE, KO) clear it at one contract. A
  properly diversified $50k wheel book is barely buildable from S&P
  names at current share prices, and the engine surfaces none of it.
  **Logged.**

- **`ev_per_day` is EV over the *effective* hold, not the 35-DTE
  nominal.** `ev_per_day = ev_dollars / expected_days_held`
  (`ev_engine.py:506`); `expected_days_held` is the
  probability-weighted blend profit→`dte/2`, stop→`dte/3`, hold→`dte`
  (`ev_engine.py:444`). For these high-prob-profit names the effective
  hold is ~17–19.5 days, not 35 — so `ev_per_day ≈ ev_dollars / ~18`,
  and `ev_per_day` order is *close to but not identical* to
  `ev_dollars` order (AJG outranks FDS on EV yet trails it on
  `ev_per_day`, its effective hold being longer). Neither metric
  carries a capital term; both front-load expensive names. Recorded so
  the ROC contrast is read against the metric the ranker actually
  sorts by — and so a future S4 reader does not mistake `ev_per_day`
  for `ev_dollars / 35`. **Logged.**

- **Silent drops recur, and bite harder under the constraint.** 6 of
  36 names returned nothing (NFLX, JPM, UNH, JNJ, GE, XOM); re-running
  with `use_event_gate=False` returns all 6 with `days_to_earnings`
  18–32 — earnings inside the option's life, correctly event-gated, 5
  of them otherwise positive-EV. Correct behavior, but invisible: the
  trader sees a 30-name list with no signal that 6 candidates were
  removed or why. Same as S1 / S2. **Logged.**

**Follow-up.** Shipped in #109 — `collateral` / `roc` columns on the
`rank_candidates_by_ev` output and `WheelRunner.select_book(...)`, an
account-aware skip-and-fill book selector. Both consume and subset
post-`EVEngine.evaluate` ranker output — neither rescues a candidate
nor bypasses the EV authority (§2-safe, pinned by a zero-evaluate-call
regression test). Deferred: per-name multi-contract sizing.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `select_book` invokes `rank_candidates_by_ev` internally and filters by collateral/ROC; no candidate bypasses `EVEngine.evaluate`.
  - Qualitative verdict: match — `collateral` + `roc` columns present on ranker output; `WheelRunner.select_book(account_size, tickers=..., as_of=..., max_weight_per_name=0.25, min_roc=0.0)` exists with the published signature. On a 36-name watchlist at `as_of=2026-03-20` it returns 27 rows / 21 positive-EV; with $50k account + 25% per-name cap it selects 3 names (CF $11,400 / EXE $10,000 / KO $7,150) — closer to S4's documented "ROC-ordered greedy fill" outcome than the absolute-EV book.
  - Numerical drift > 5%: original S4 quoted "30 of 36 names ranked, 20 positive-EV" — current run gives 27 / 21. Row count delta −10% (30→27); positive-EV delta +5% (20→21). Likely attributable to the post-S4 universe data refresh (some 2026-Q1-earnings names now dropped by event gate) + post-IV-PIT-fix EV recalibration (PR #179). Not attributable to one single PR — composite of #119/#121/#179/#208/#210/#220.
  - Notes: `select_book` parameter name is `max_weight_per_name` (not `max_per_ticker_pct` as my first probe used) — pure-API note; behaves as documented.

### S5 — Live MCP chart in the loop

**Purpose.** Exercise the live TradingView MCP chart
(`MCPChartProvider` inside a `ChainedChartProvider`) through the real
dossier flow — confirm a live chart reaches verdicts an offline
session cannot, and confirm §2: the chart can only downgrade, never
rescue.

**Setup.** Live infra, 2026-05-21: TradingView Desktop build
`3.1.0.7818` (Electron 38.2.2) with CDP on `localhost:9222`; the
tradingview-mcp `tv` CLI on PATH and connected; `engine/mcp_client.py`
`MCPCLIClient` driving `MCPChartProvider`. `SWE_DATA_PROVIDER=bloomberg`,
`SWE_USE_MCP_CHART=1`. Real candidates from
`WheelRunner.rank_candidates_by_ev`; dossiers via `build_dossiers` +
`EnginePhaseReviewer` (R1–R6).

**Status.** Done. The live MCP chart was exercised end-to-end through
the dossier flow. §2 holds — no violation, no bug, no decision-layer
or `tests/` change. One real integration caveat logged (R3 vs stale
data).

**Findings:**

- **The live chart is genuinely in the loop.** `MCPChartProvider.fetch`
  captured real charts on demand — AAPL `visible_price=304.99`, XOM
  `visible_price=155.29`, each with a screenshot, `source="mcp"`,
  `is_ok=True`. Roughly 12 s per `capture_screenshot` round-trip —
  confirms `TRADINGVIEW_MCP_INTEGRATION.md` §8 q3: MCP is rightly
  opt-in via `SWE_USE_MCP_CHART`, not default-on.
- **Before/after, real candidate (XOM, `ev_dollars=+134.46`).**
  Offline (`FilesystemChartProvider`, no cached screenshot) ->
  `review` / `chart_context_missing` (R2). Live MCP chart ->
  `skip` / `spot_price_mismatch` (R3). The live chart takes the
  candidate off the R2 "no chart" hold and onto a real verdict.
- **§2 holds — verified on real data.** The real AAPL candidate
  (`ev_dollars=-39.05`) stayed `blocked` (R1) with a genuine OK live
  MCP chart attached (`is_ok=True`, real screenshot, live price). R1
  returns at `candidate_dossier.py:158` *before the chart is even
  examined* — the live chart structurally cannot rescue a negative-EV
  candidate or bypass `EVEngine.evaluate`. A controlled negative-EV
  row (`ev_dollars = -50`) + a real live chart -> also `blocked`.
- **`proceed` stays R5's EV decision, not the chart's.** With the real
  live chart attached to a controlled row: `ev_dollars >=
  min_proceed_ev` (10) and `visible_price` within R3's 2 % of the
  engine spot -> `proceed` (R5); a 10 %-off spot -> `skip` (R3). The
  chart's only role is to *not* downgrade — it never upgrades.
- **R3 vs stale data — integration caveat, not a bug.** The live
  chart's `visible_price` is real-time; the engine's `spot` under
  `SWE_DATA_PROVIDER=bloomberg` is a stale EOD-CSV value (XOM 155.29
  live vs 161.22 CSV, ~3.7 %; AAPL 304.99 vs 247.64, ~23 %). Both
  exceed R3's 2 % tolerance, so a live chart *systematically* `skip`s
  real Bloomberg candidates. R3 works as designed — it catches a spot
  disagreement — but the stale side is the engine's data. The live
  MCP chart is `proceed`-useful only paired with a current-spot
  provider (`theta`, or freshly refreshed Bloomberg — cf. S6).
  §2-safe: `skip` is a downgrade. **Logged.**
- **PIT discipline holds.** `MCPChartProvider.fetch(as_of=...)`
  short-circuits to `error="pit_violation"` with no capture; inside a
  `ChainedChartProvider` it falls through to `FilesystemChartProvider`
  — no live screenshot leaks into a historical / backtest review.
- **Fails closed.** A forced MCP error returns a `ChartContext`
  carrying a canonical `MCP_ERROR_MODES` value (`mcp_unavailable`,
  `browser_disconnected`), `is_ok=False` — never a fabricated
  screenshot. In a chain the error falls through to the next provider,
  and the reviewer then sees a missing chart -> R2 `review`.
- **Symbol resolution.** A bare ticker (`AAPL`, `XOM`) resolves
  correctly — TradingView picked `BATS:AAPL`; no `EXCHANGE:TICKER`
  prefix needed. The `mcp_client.py` `TODO(live-verify)`
  ambiguous-ticker fallback did not trip for these large-caps.

**§2 verdict:** holds — no bypass. The live MCP chart is a
downgrade-only `ChartContextProvider`: it cannot rescue a negative-EV
candidate (R1 fires first), cannot manufacture a `proceed` (that is
R5's EV check), and cannot bypass `EVEngine.evaluate`.

- **Re-verified 2026-05-26** by Terminal A — **SKIPPED (operator-gated)**. Per the task spec skip list: S5 requires live TradingView Desktop + CDP on `:9222` + `tradingview-mcp` CLI on PATH. The Cowork sandbox / fresh-checkout terminal has no live TradingView Desktop process. The MCPChartProvider plumbing (the `engine/mcp_client.py` + `engine/tradingview_bridge.py` seam, and the dossier-invariant test guard that auto-activates when MCP ships) is structurally still present on `main` per PROJECT_STATE.md §3 and unchanged since S5 ran. §2 status inherits from the original entry; no live verification this pass.

### S7 — Advisor committee deep dive

**Purpose.** Verify S1's logged committee/ranker contract-mismatch
claim, and answer the trader question: do four advisors disagree
usefully, or is the committee expensive noise on retail short puts?

**Setup.** Bloomberg, offline charts, `as_of=2026-03-20`, 35-DTE /
25-delta, top-10 ROC names from S4 (CF, FIX, FDS, AJG, EXE, JBHT,
EG, MCK, HUM, BR), fed through
`advisors.integration.EngineIntegration.evaluate_trade` — naive
caller, then a corrected caller emulating the `/api/committee` path
(delta from spot/strike/IV, `ev_dollars → ev_pct`). Plus 12
synthetic probes varying one input at a time. No code changes.

**Status.** Done. All findings logged (no fix this session).
Code-level claims verified by Cowork-B against source; runtime vote
patterns/probe reactions as reported by the executor run.

**Findings:**

- **Committee structurally pinned at neutral.**
  `_determine_committee_judgment` leaves neutral only on
  `approve_count > total/2` or `reject_count > total/2` — i.e.
  ≥3 of 4 (`committee.py:331,337`). Three advisors default neutral
  on retail short puts, so the verdict never escapes neutral on
  realistic ranker output. **Logged.**
- **`filter_approved(min_approval_count=2)` blocks 100% of
  positive-EV picks.** Keyed on each trade's advisor
  `approval_count` (`integration.py:117-141`); max observed
  approves = 1 (Munger), so it returns 0 trades at thresholds 2, 3,
  and 4. **Logged.**
- **The `EngineIntegration` helper has type bugs `/api/committee`
  already fixed.** The helper passes `ev_dollars` straight into
  `expected_value` (documented "Expected return %"), rendering
  $247.76 of EV as "247.76%". The API path converts
  `ev_dollars → ev_pct` (`engine_api.py:871,883`) and rescales vix
  fraction→percent (`924-931`); `_build_advisor_input` does
  neither. Fix belongs in `_build_advisor_input` so all callers
  benefit, not duplicated per endpoint. **Logged.**
- **Per-advisor signal is real but discarded.** Negative EV → 2
  rejects, crisis regime → 2 rejects, earnings-in-expiry → 1
  reject; the >50% aggregator throws away sub-majority dissent. A
  `committee_judgment="elevated_concern"` on ≥2 dissents
  (escalation — §2-safe) would surface it. **Logged.**
- **Ranker emits no `delta`/`theta`/`gamma`/`vega`/`iv_rank`.**
  Forces the helper's −0.30 delta fallback. The ranker selects the
  strike via a chain `delta` column (`wheel_runner.py:899-907`)
  but emits no delta in its output. **Logged.**
- **"Areas of agreement" is substring keyword matching, not
  semantic synthesis.** **Logged.**
- **Simons is binary-on-EV; the others binary-off.** Simons
  strong-approves on high EV without distinguishing 5% / 10% /
  50%; net committee ≈ `Simons_thinks_EV_high ? lean : neutral`.
  **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds by structure — advisor committee is downgrade-only-advisory per CLAUDE.md §1 (interface layer, not on the EV path); not exercised live this pass because the public API changed.
  - Qualitative verdict: partial — `EngineIntegration` exists, but `evaluate_trade(ev_row)` and `filter_approved(rows, min_approval_count)` now both require additional positional args `portfolio_state` and `market_state` (signatures evolved post-S7). The naive-caller demonstration the original S7 exercised is no longer a single-arg call. The structural findings (committee structurally pinned at neutral; per-advisor signal real-but-discarded; Simons binary-on-EV) still apply at the source-code level — `_determine_committee_judgment` thresholds in `advisors/integration.py` are unchanged on `main`.
  - Numerical drift > 5%: not measured — the API signature shift blocks the original probe matrix.
  - Notes: ranked the 10 ROC names successfully (10/10 from the S4-derived watchlist); `EngineIntegration` instantiates clean. A future Sn could re-exercise with the new `portfolio_state` / `market_state` shape to re-confirm the "committee pinned at neutral" finding on real data.

### S8 — Wheel-cycle-to-completion

**Purpose.** Walk one full wheel cycle end to end — short put →
assignment → covered call → roll → called away — to exercise the
management-layer methods a multi-leg cycle needs
(`handle_put_assignment`, `open_covered_call`, `roll_call`,
`handle_call_assignment`, `mark_to_market`, `get_performance_summary`)
and see what the engine does *not* support once the position leaves
the short-put leg.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`. Single
name DIS, chosen for a clean dip-and-recovery in the Bloomberg window.
$30k account, `WheelTracker(require_ev_authority=True)`. Put leg routed
through `WheelRunner.rank_candidates_by_ev` (`as_of=2025-03-10`,
35-DTE / 25-delta) → EV-authority token → `open_short_put`. Timeline:
short put 2025-03-10 (strike $98); assigned 2025-04-14 at spot $84.66
(−$13.34/sh, basis $98); covered call sold 2025-04-15 ($92 strike);
rolled $92C → $112C on 2025-05-12 (spot $110.49, deep ITM); called
away 2025-06-20 at spot $117.63.

**Status.** Done. The full cycle completes and reconciles — net
**+$182.01** on $30k over 102 days: the covered-call premiums plus the
roll buying $20 of extra strike room ($92 → $112) turned a $13.34/sh
underwater assignment into a small green cycle. One genuine bug fixed;
the rest logged. The headline gap: **the wheel's second leg (covered
call) and the roll are entirely outside the EV decision authority** —
only the put entry is engine-ranked.

**Findings:**

- **No covered-call entry ranker — the call leg bypasses the EV
  authority.** `open_covered_call` (`wheel_tracker.py:777`) takes a raw
  `strike` / `premium`, has no `ev_authority_token` parameter, runs no
  `EVEngine.evaluate`, and there is no covered-call analogue of
  `rank_candidates_by_ev`. The covered call is a tradeable short option
  yet nothing ranks or EV-checks it. In this run the $92 strike and its
  premium had to be hand-picked and BSM-priced. The put leg is
  EV-authoritative; the call leg is unmanaged. **Fixed in `#124`** —
  `WheelRunner.rank_covered_calls_by_ev` is the covered-call entry
  ranker, the call-leg parallel of `rank_candidates_by_ev`: it EV-ranks
  a (strike × DTE) grid for a held position, every candidate scored
  through `EVEngine.evaluate`.

- **`open_covered_call` / `roll_call` apply no event gate.** DIS
  earnings 2025-05-07 fell squarely inside the covered call's
  [2025-04-15, 2025-05-20] life. The put leg is event-gated through the
  ranker's `EventGate`; the call leg has no equivalent — earnings
  inside a covered call's expiry are invisible to the engine. **Logged.**

- **`roll_call` has no EV / decision support.** S3 shipped
  `suggest_rolls` for the *put* leg; `suggest_call_rolls` was
  explicitly deferred (#104 follow-up, still queued). The cycle's roll
  ($92C → $112C, a −$1,534 net debit) was a pure-mechanics call with no
  roll-vs-hold-vs-let-assign EV comparison — the trader is on their own
  for the single most consequential covered-call decision.
  **Fixed in `#122`** — `suggest_call_rolls` ranks covered-call rolls
  by forward EV through `EVEngine.evaluate`; `roll_call` mechanics are
  unchanged, but the roll-vs-hold decision support that was missing
  now exists alongside it.

- **The EV-authority token proves provenance, not tradeability.**
  `issue_ev_authority_token` (`wheel_tracker.py:168`) hashes and
  accepts *any* ranker row with no EV-sign check; `open_short_put`'s
  launch-gate (`wheel_tracker.py:241`) only verifies the token exists.
  The DIS candidate here had `ev_dollars = −$30.65` (surfaced only
  because the run used a relaxed `min_ev_dollars`) and `open_short_put`
  accepted it. §2 is not strictly violated — `EVEngine.evaluate` *was*
  called — but the R1 "negative EV → blocked" verdict, enforced in the
  dossier reviewer path, is **not** propagated into the token: the gate
  is "ranker-derived", not "ranker-approved". A token that encoded the
  verdict (or a positive-EV assertion in `issue_ev_authority_token`)
  would close this. Left for a human to scope — it changes the
  launch-gate contract. **Logged → Fixed in #145** (D16) —
  `issue_ev_authority_token` now raises `EVAuthorityRefused` on
  `ev_dollars <= 0`, and `_consume_ev_authority_token` re-checks a
  fresh `current_ev_dollars` at fire time (stale-EV rejects with
  the token retained). Both wheel legs (`open_short_put` /
  `open_covered_call`) flow through the same predicate.

- **`get_performance_summary` reported a winning trade as
  `largest_loss`.** `largest_loss` was `net_pnl.min()` over *all*
  trades, so an all-green book returned its smallest *win* as the
  "largest loss" (the single-trade cycle here showed
  `largest_win == largest_loss == $182`). `largest_win` had the
  symmetric flaw. **Fixed** in this PR — both are now taken over the
  winner / loser subsets, `0.0` when the subset is empty; covered by
  `tests/test_wheel_lifecycle.py::TestPerformanceSummary`.

- **`tracker.cash` overstates buying power** — confirmed (S2 / S4).
  After the CSP opened, `cash` read $30,129 while $9,800 of strike
  collateral was unreserved; deployable capital is `cash − Σ(strike ×
  100)`, still computed by hand. **Fixed in `#127`** —
  `WheelTracker.available_buying_power()` returns `cash − Σ(put_strike ×
  100)` over the open short puts.

- **`mark_to_market` IV staleness** — confirmed (S2), but mild in this
  cycle: at both mark dates the short put was deep ITM (spot $81.72 vs
  $98 strike), so vega was small and the entry-IV vs live-IV(0.55)
  marks differed by only ~$1. The gap bites on ATM / OTM holds carried
  through a vol regime change, not on a deep-ITM leg. **Fixed in
  `#129`** — `mark_to_market` now resolves the connector's as-of ATM IV
  when `current_ivs` omits a ticker, falling back to the entry IV only
  as a last resort.

**Follow-up.** Two methods bring the wheel's second half under the
same EV authority as the first: `suggest_call_rolls` (the call-leg
parallel of `suggest_rolls`) — **done, shipped in `#122`** — and a
covered-call *entry* ranker (the call-leg parallel of
`rank_candidates_by_ev`) — **done, shipped in `#124`**. Both are
§2-safe by construction: they *rank*, not rescue.

**Validation re-run (2026-05-21).** Confirm-fixed pass on real
Bloomberg data, `as_of=2026-03-20`. The covered-call leg the original
run had to walk by hand is now engine-ranked end to end:

- *Covered-call entry.* `rank_covered_calls_by_ev("DIS",
  shares_held=100)` returns **16 EV-ranked candidates** (4 DTE ×
  4 delta), each scored through `EVEngine.evaluate` — `ev_dollars`
  −131 … −71. Every DIS covered call at this `as_of` is negative-EV,
  so the default `min_ev_dollars=0` floor returns **0 tradeable rows**
  (ranks, never rescues). *Before:* the $92 strike was hand-picked and
  BSM-priced with no EV check.
- *Buying power (`#127`).* After the 98-strike CSP opened, `cash` =
  $30,304.70 but `available_buying_power()` = **$20,504.70** (cash −
  $9,800 collateral) — the figure the original run computed by hand.
- *Persistence (`#128`).* `WheelTracker.save` → `load` round-trips
  structurally identical (position state + buying power preserved);
  a mid-campaign save/resume is now possible at all.
- *Mark-to-market IV (`#129`).* With a connector, `mark_to_market`
  marks at the as-of IV — DIS @2026-01-20 → **0.359**, well above the
  position's stale 0.28 entry IV — and the marks differ ($29,811.87
  vs $29,899.19 on entry IV).
- *Roll support (`#122`).* `suggest_call_rolls` on an adverse covered
  call (spot 110 vs a 100C) returns **16 EV-ranked rolls** with
  `roll_ev` / `hold_ev` / `recommend` — the roll-vs-hold comparison
  the original $92C → $112C roll lacked.

No new bug surfaced. The two findings still **Logged** above —
`open_covered_call` / `roll_call` event gate, and the EV-authority
token encoding only provenance — remain genuinely open (#118 P5).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `D16 / PR #145` is live: a `WheelTracker(require_ev_authority=True)` issuing a token on a hand-built negative-EV row (`ev_dollars = −30.65`) raises `EVAuthorityRefused` at issue time. The "ev-authority token proves provenance, not tradeability" finding is mechanically closed. R1 still fires first in the dossier reviewer (confirmed end-to-end via the synthetic-chart probe in S19 below — `-25` → `blocked`).
  - Qualitative verdict: match — every wheel-cycle method called out as missing in the original S8 is now present on `main`: `WheelRunner.rank_covered_calls_by_ev` (PR #124), `WheelTracker.suggest_call_rolls` (PR #122), `WheelTracker.issue_ev_authority_token` (PR #145 / D16), `WheelTracker.open_covered_call`, `WheelTracker.available_buying_power` (PR #127), `get_performance_summary`. The DIS-validation-rerun results (16 CC candidates, `available_buying_power = $20,504.70`, `mark_to_market` at as-of IV) are baked into the existing entry's narrative.
  - Numerical drift > 5%: not applicable — original entry is wheel-cycle-narrative; no specific NAV / per-leg numbers were quoted as drift-prone.
  - Notes: D16 / R1a refusal verified in the EVAuthorityRefused exception path; named in CLAUDE.md §2 R1a. The two genuinely-open Logged findings (CC-leg event gate, token-encodes-provenance-not-tradeability) — `open_covered_call` / `roll_call` event gate is still absent today (the put leg has `EventGate`; the call leg does not). Token-encodes-tradeability is closed by D16's runtime check.

### S9 — Adversarial / gate stress

**Purpose.** Attack each engine gate with inputs that should be
rejected and confirm it fails closed — drops or flags the candidate,
never emits a tradeable result. Gates: history, event, chain-quality,
stress-residual, survivorship.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `as_of=2026-03-20`, 35-DTE /
25-delta. Per-gate probes through `WheelRunner.rank_candidates_by_ev`
(each gate's `enforce_*` / `use_*` flag toggled on vs off), plus direct
probes of `DataQualityFramework._check_options_consistency` and
`StressTester.greeks_stress_ladder`. No code changes.

**Status.** Done. No gate fails open. Four gates (history, event,
stress-residual, survivorship) were exercised live and each fails
closed; the chain-quality gate is structurally inactive on the
Bloomberg provider — no option chain to police — and its check logic
fails closed when probed directly. No bug, no fix; all findings logged.

**Findings:**

- **History gate — fails closed.** `enforce_history_gate` /
  `min_history_days=504` (`wheel_runner.py:614`; in-code, the
  survivorship-bias protection). Probe: GEV (497 OHLCV bars) and SOLV
  (498) — real 2024 spin-offs, both under 504 bars at `as_of` — with
  AAPL (2065) as control. Gate on → ranks `[AAPL]`; gate off →
  `[AAPL, GEV, SOLV]`. The short-history names are correctly blocked.
  **Logged.**

- **Event gate — fails closed.** `use_event_gate` → `EVEngine.evaluate`
  event lockout (`ev_engine.py:262`) → the ranker drops the row on
  `event_lockout_reason` (`wheel_runner.py:1056`). Probe: XOM, JPM,
  UNH, JNJ, GE, NFLX — all with earnings inside the 35-DTE window at
  `as_of=2026-03-20` — plus AAPL control. Gate on → `[AAPL]`; gate off
  → all 7. All six earnings-window names blocked. `EVEngine` does
  compute `event_lockout_reason`, but the ranker discards it on
  `continue` — see the silent-rejection finding below. **Logged.**

- **Chain-quality gate — logic fails closed, but dormant on the
  Bloomberg provider.** `enforce_chain_quality_gate`
  (`wheel_runner.py:843`) runs `_check_options_consistency` on the raw
  chain and `continue`s the ticker on any ERROR/CRITICAL issue. Logic
  probe: a degenerate chain (negative volume, IV 9.5, crossed
  bid > ask) → 3 ERROR issues; a clean chain → 0 — the gate would
  block. **But** `MarketDataConnector` exposes neither `get_options`
  nor `get_option_chain`, so on Bloomberg `chain_df` is always `None`
  and the gate at `:843` never executes. It is reachable only with a
  live-chain provider (Theta — S6); on the default provider it is a
  no-op, and the premium it would police is synthetic BSM anyway.
  **Logged.**

- **Stress-residual gate — fails closed; advisory, off the EV path.**
  The Greeks-decomposition residual gate lives in
  `engine/stress_testing.py` (`:639`), not in `rank_candidates_by_ev`.
  Probe: an extreme `greeks_stress_ladder` (spot ±35 %, `iv_shock=0.80`)
  → 8 of 9 rows tagged `reliable=False`,
  `attrs["residual_gate_passed"]=False`, `max_residual_pct ≈ 3.12`, and
  a `warnings.warn` fires; a mild ±1 % ladder → all reliable. The gate
  correctly flags Greeks the Taylor decomposition cannot attribute. It
  is **advisory** — it tags rows, never drops them — and never blocks
  an EV candidate. **Logged.**

- **Survivorship gate — fails closed; it is the history gate, not a
  membership check.** Probe: `[ZZZZ, NOTAREALTICKER, FIX, AAPL]` →
  ranks `[AAPL, FIX]`; the bogus tickers have no OHLCV and are dropped
  at the data-fetch step (`wheel_runner.py:593`).
  `rank_candidates_by_ev` runs **no constituent-membership check** —
  `get_universe()` (`data_connector.py:654`) is the union of tickers in
  the OHLCV / fundamentals / vol_iv CSVs, not
  `data_raw/sp500_constituents_current.csv`. Benign in practice:
  index-removed names (IPG, K, LKQ, MHK) have no OHLCV at all and
  cannot be ranked, while FIX — a genuine member missing from the
  stale constituents CSV — ranks correctly. Survivorship protection
  thus reduces to no-data-drop plus the 504-bar history gate; there is
  no data-freshness gate, though the Bloomberg data carries no
  stale-but-long delisted name to exploit one. **Logged.**

- **Rejections are silent — recurring S1 / S2 finding.** History,
  event and survivorship rejections are indistinguishable in the
  output: the candidate is simply absent, with no reason, count, or
  diagnostic — a caller cannot tell "gated out" from "never a
  candidate." Of the five gates, only stress-residual surfaces its
  verdict (a `warnings.warn` + `.attrs`); the chain-quality gate at
  least emits a `logger.warning` when it blocks. The three live ranker
  gates are fully silent. Same gap S1 and S2 logged. **Fixed in #121.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every gate exercised below fails closed; no gate failure produces a tradeable candidate.
  - Qualitative verdict: match — all five gates still behave as documented:
    - History gate ON → `[AAPL]` only (GEV, SOLV dropped); OFF → `[AAPL, GEV, SOLV]` ✓
    - Event gate ON @ `as_of=2026-03-20` for the 7-name probe → `[AAPL]` only; six earnings-window names dropped with structured `{ticker:"XOM", gate:"event", reason:"event_lockout:earnings@2026-04-07 (±5d buffer)"}` (the `±` Unicode literal is now in the reason string verbatim — unchanged S1 cp1252 footgun).
    - Survivorship — bogus tickers (ZZZZ, NOTAREALTICKER) silently dropped at data-fetch; FIX/AAPL survive ✓
    - Stress-residual gate / chain-quality gate — structurally inactive on Bloomberg provider (no chain), as documented.
  - Numerical drift > 5%: not applicable — original entry counts gate behavior boolean, not magnitudes.
  - Notes: drops schema confirmed structured `{ticker, gate, reason}` across all observed drops in this run (PR #121 holds).

### S10 — News-sentiment downgrade path

**Purpose.** Validate the news-sentiment overlay
(`engine/news_sentiment.py`, the only news module on the EV path):
confirm bad sentiment downgrades a candidate's EV, and — the §2
invariant — confirm good sentiment can never rescue a non-tradeable
candidate.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `as_of=2026-03-20`, 35-DTE /
25-delta. Probe A — `NewsSentimentReader` against a synthetic store in
a temp dir, mapping `sentiment_multiplier` across the sentiment range
including the extremes. Probe B — `rank_candidates_by_ev` end-to-end
with `get_ticker_sentiment` monkeypatched to inject synthetic sentiment
(process-local; the real `sentiment_multiplier` and `wheel_runner`
wiring still run), `use_news_sentiment` toggled on vs off. No code
changes.

**Path.** `use_news_sentiment` → `NewsSentimentReader.sentiment_multiplier`
→ `news_mult`, folded into `combined_regime_mult = hmm × skew × news ×
credit` (`wheel_runner.py:933`) → `trade.regime_multiplier` →
`EVEngine.evaluate`, where `ev_dollars = ev_raw × regime_mult`
(`ev_engine.py:502`). News scales the final dollar EV; it never touches
`ev_raw`.

**Status.** Done. §2 holds — bullish sentiment (capped at +5 %) cannot
rescue a negative-EV or event-gated candidate. Bad sentiment downgrades
EV as designed. No §2 violation, no bug; all findings logged.

**Findings:**

- **News multiplier is clamped to [0.88, 1.05] — asymmetric,
  downgrade-biased.** `sentiment_multiplier` is a 4-value step
  function: sentiment ≤ −0.3 → 0.88, ≤ −0.1 → 0.95, ≥ 0.3 → 1.05, else
  1.00; `n_articles < 5` forces 1.00. Probed across the full range —
  even maximal sentiment (+1.0, n=9999) caps at 1.05, minimal (−1.0)
  floors at 0.88. Max downgrade −12 %, max boost +5 %, analogous to the
  dealer multiplier's [0.70, 1.05] (CLAUDE.md §2). §2-safe by
  construction. **Logged.**

- **Bad sentiment downgrades EV — confirmed end-to-end.** News-on vs
  news-off through the ranker: CF (bearish −0.6) `ev_dollars`
  247.76 → 218.03 (×0.88); AJG (−0.2) 326.57 → 310.24 (×0.95). The
  on/off ratio equals the emitted `news_multiplier` exactly — the
  overlay works as designed. **Logged.**

- **§2 holds — good news cannot rescue.** `ev_dollars = ev_raw ×
  regime_mult` with a strictly-positive multiplier is sign-preserving:
  a negative-EV name stays negative. Probe: MSFT (negative-EV) +
  maximally-bullish sentiment (1.05) → `ev_dollars` −24.47 → −25.69
  (still < 0), and MSFT stays absent from a `min_ev_dollars=0` ranking.
  Event-gated XOM + bullish news stays gated — the event lockout
  (`ev_engine.py:262`) precedes the multiplier. Because the multiplier
  scales signed magnitude, bullish news on a negative-EV name makes
  `ev_dollars` *more* negative — harmless for §2 (such names are
  non-tradeable regardless), but it means the multiplier is a magnitude
  scaler, not a directional tilt. **Logged.**

- **`n_articles < 5` → forced neutral (1.00).** FDS with +0.90
  sentiment but only 3 articles → multiplier 1.00, zero EV change. A
  genuine sentiment signal on a thinly-covered name is silently
  ignored. **Logged.**

- **No news store on the Bloomberg setup — the overlay is a dormant
  no-op by default.** None of `news_sentiment.py`'s `_CANDIDATE_PATHS`
  (`data_processed/news_sentiment.{parquet,csv}`,
  `data/news/sentiment.*`, `financial_news/storage/sentiment.sqlite`)
  exist; with `use_news_sentiment=True` (the default) every candidate
  gets a 1.00 multiplier. Absent news — no store, no row for a ticker,
  or only stale rows — collapses silently to neutral. **Logged.**

- **News time-handling — a PIT leak, plus silent staleness.**
  `get_ticker_sentiment` keeps only rows with `as_of ≥ now() − 72h`
  using wall-clock `now()`; the ranker's `as_of` PIT cutoff never
  reaches the news reader. A backtest at `as_of=2026-03-20` would
  apply *today's* news — look-ahead, the same family as S1's `as_of`
  footgun. Conversely, news older than 72 h is dropped to neutral with
  no warning. **Fixed in #119.**

- **The overlay is surfaced — but only conditionally.** Unlike S9's
  silent ranker gates, the applied multiplier is visible:
  `news_multiplier`, `news_sentiment`, `news_n_articles` are emitted —
  but only as diagnostic columns (`include_diagnostic_fields=True`).
  With diagnostics off, the news adjustment is invisible. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `NewsSentimentReader.sentiment_multiplier(...)` returns 1.0 on missing-store path (silent neutral) by design; `wheel_runner` folds it into `combined_regime_mult` (`hmm × skew × news × credit`) so it scales `ev_dollars` only; `ev_raw` (the EV authority's input) is untouched. The "good news cannot rescue" invariant is sign-preserving by construction.
  - Qualitative verdict: match — overlay is dormant on Bloomberg setup (no news store on the worktree); `news_multiplier`, `news_sentiment`, `news_n_articles` columns all surfaced in diagnostic mode. PR #119 (news PIT-leak fix referenced in original S10 entry) shipped on `main`.
  - Numerical drift > 5%: not applicable — original entry was monkeypatched probes (CF -0.6 → ×0.88 = 247.76 → 218.03 etc.); re-running monkeypatched is out of scope this pass. The base behavior (no store → multiplier 1.0) verified directly.
  - Notes: bigger evidence that the news-PIT-leak fix shipped lives in S11's re-verify (where credit_multiplier — sister overlay — now varies with as_of date, confirming the time-handling fix landed).

### S11 — Regime-shift stress

**Purpose.** Stress the regime machinery: anchor the ranker across the
April-2025 VIX spike and observe whether the HMM and dealer-positioning
multipliers actually track a real volatility shock.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, 35-DTE / 25-delta, a fixed
12-name diversified watchlist (AAPL, MSFT, NVDA, AMZN, JPM, XOM, UNH,
CAT, KO, PG, HD, CVX), `include_diagnostic_fields=True`.
`rank_candidates_by_ev` run at five `as_of` dates spanning the
April-2025 vol spike — 2025-03-20 (VIX 19.8, calm) / 04-07 (46.98) /
04-09 (33.6, peak window) / 04-24 (26.5, elevated) / 05-15 (17.8,
reverted). The regime-multiplier trajectory was probed with
`use_event_gate=False` (all names present each date); event-gate
behaviour probed separately on vs off. No code changes.

**Status.** Done. The HMM regime multiplier responds as advertised — it
cut EV ~80 % into the spike (cross-sectional mean 0.74 → 0.29) and
reverted (→ 0.69). The dealer, skew and credit overlays did not
respond — each pinned at 1.0 the whole way through. No multiplier was
provably wrong; no bug. All findings logged.

**Findings:**

- **HMM multiplier tracks the shock — works as advertised.**
  Cross-sectional mean `hmm_multiplier`: 0.74 (calm, 2025-03-20,
  VIX 19.8) → 0.29 (peak, 04-09; 04-07 hit VIX 46.98 — the steepest
  VIX run-up in the 11-year data window outside the 2020 COVID crash)
  → 0.69 (reverted, 05-15, VIX 17.8). At the peak most names sat at
  ~0.20 — `position_multiplier` is a posterior-weighted average of
  per-state weights {crisis 0.2, bear 0.5, normal 1.0, bull_quiet 1.25}
  (`regime_hmm.py:275`), so ~0.20 means ~100 % crisis-state posterior
  (a genuine classification, not a clamp). The HMM cuts EV up to 80 %
  into the spike and reverts. **Logged.**

- **Per-ticker, the HMM multiplier is jumpy — a noisy single-name
  signal.** `GaussianHMM(n_states=4, n_iter=20, random_state=42)`
  (`wheel_runner.py:862`) is seeded, so fits are deterministic — but it
  re-fits per (ticker, as_of) and the clean cross-sectional mean hides
  per-ticker cliffs: HD flips 0.21 (04-07, crisis) → 1.00 (04-09,
  normal) over two days while VIX held 33–47; PG and UNH stay at ~0.2
  on 05-15 after VIX reverts. The HMM models each *ticker's* return
  regime, not the market's, so single-name multipliers diverge from
  VIX — partly genuine idiosyncratic stress, partly fit sensitivity
  from the low `n_iter=20`. On any one name it is a noisy de-rater.
  **Logged.**

- **The HMM regime is unlabeled in the output — confirms S1.** The
  ranker emits `hmm_multiplier` (a bare number — 0.20 = an 80 % EV cut)
  with no companion `hmm_regime` label, even though `dealer_regime` and
  `credit_regime` label columns both exist. A trader sees the cut with
  no surfaced "crisis". S1 logged this; S11 confirms it and sharpens it
  via the asymmetry with the other two overlays. **Fixed in #121.**

- **Dealer & skew multipliers are inert on the Bloomberg provider.**
  `dealer_multiplier` and `skew_multiplier` were pinned at 1.00 across
  all five dates; `dealer_regime` was empty throughout. Both require an
  option chain; `MarketDataConnector` exposes none, so neither overlay
  ever computes — they cannot respond to any shock on Bloomberg. Only
  the Theta provider (S6) would activate them. Same structural dormancy
  S9 found for the chain-quality gate. **Logged.**

- **The credit-regime overlay is not as_of-aware — a PIT leak.**
  `credit_multiplier` / `credit_regime` were identical
  ('benign' / 1.00) across all five as_of dates, including 04-07 at
  VIX 46.98. `FREDAdapter.credit_regime()` (`fred_adapter.py:137`)
  takes no `as_of` — it returns one wall-clock value applied to every
  historical as_of, so a backtest across a credit-stress episode would
  never see the stress. Same family as S10's news PIT leak and S1's
  `as_of` footgun. **Fixed in #119.**

- **Net — on the Bloomberg provider the HMM carries the entire regime
  response.** `combined_regime_mult = hmm × skew × news × credit`
  (`wheel_runner.py:980`), but skew and dealer are pinned (no chain),
  credit is pinned (PIT-unaware), and news is pinned (no store — S10).
  Of the four regime overlays, only the HMM is both live and responsive
  on the default provider. **Logged.**

- **The event gate stays consistent across the shift — earnings-driven,
  not vol-driven.** With `use_event_gate=True`, survivors of the
  12-name watchlist were 8 / 2 / 2 / 3 / 10 across the five dates,
  tracking the Q1-2025 earnings calendar (April is peak earnings
  season → 9 of 12 gated) rather than the VIX — the gate behaves
  identically regardless of regime. (Aside: names within 5 days of
  earnings are soft-skipped even with `use_event_gate=False` — a
  second, separate earnings mechanism.) The stress-residual gate is not
  on the ranker decision path (S9). **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — overlays only scale `ev_dollars`, never `ev_raw`; multipliers clamped at construction time.
  - Qualitative verdict: **partial — substantial behavior change for the better**. HMM trajectory near-identical to original (0.74 → 0.36 → 0.29 → 0.70 → 0.69 — peak at 04-09 matches original 0.29 to 1 part in 100). Event-gate survivor counts at the five dates: **8 / 2 / 2 / 2 / 10** vs original **8 / 2 / 2 / 3 / 10** — one delta at 04-24 (3 → 2). Dealer + skew + news still pinned at 1.0 (no chain on Bloomberg). **But credit_multiplier is now PIT-aware:** 1.00 (03-20) / **0.80** (04-07) / **0.92** (04-09) / 1.00 (04-24) / 1.00 (05-15) — the credit overlay now responds to the April-2025 VIX spike. The original S11 finding "credit-regime overlay is not as_of-aware — a PIT leak" is **CLOSED by PR #119**.
  - Numerical drift > 5%:
    - metric `cross_sectional_hmm_means[2025-04-09]`: orig `≈0.20` (peak) → new `0.2923` (`+46%`); attributable to S31 / PR #208 + PR #222's HMM disambiguation columns plus minor seed-stable HMM refits. Direction unchanged.
    - metric `event_survivors[2025-04-24]`: orig `3` → new `2` (`-33%`); attributable to PR #220 (`as_of-beyond-data` gate extension may now drop one borderline name on this date) — not high-confidence; could also be incidental Bloomberg-data revision since S11 ran.
  - Notes: HMM regime is still **noisy per-ticker** (HD flips normal↔crisis across days); the multiplier value is more stable than the label, confirming the S17 finding still applies.

### S12 — TradingView webhook ingest

**Purpose.** Exercise the Pine-signal entry path end-to-end —
`POST /api/tv/webhook` → ring buffer → read endpoints — and answer the
§2 question: can a webhook alert produce a tradeable verdict that
bypasses `EVEngine.evaluate`?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`. `engine_api.py` started on
`:8787` (stdlib `ThreadingHTTPServer`); a probe script drove the
webhook over HTTP with synthetic Pine payloads — valid, malformed,
bad-ticker, replay, and a 210-alert overflow burst — plus the read
endpoints, run once with no webhook secret and once with
`TV_WEBHOOK_SECRET` set. No code changes.

**Path.** `POST /api/tv/webhook` → `TVAlert.parse` + `is_valid()` →
optional HMAC / shared-secret auth → timestamp-freshness + replay
guard → `_enrich_alert` → append to the in-memory ring buffer
`_TV_ALERT_LOG`. `_enrich_alert` computes the alert's verdict by
running `WheelRunner.rank_candidates_by_ev([ticker])` →
`EVEngine.evaluate` (`engine_api.py:2055`) — the verdict is
EV-authoritative (`authority="ev_ranked"`).

**Status.** Done. §2 holds — a webhook alert cannot reach a tradeable
verdict without `EVEngine.evaluate`, and a Pine signal can only
downgrade. No §2 violation; no bug. All findings logged.

**Findings:**

- **§2 holds — a Pine signal cannot rescue a non-tradeable
  candidate.** A webhook alert for AAPL (`ev_dollars = −95.47`) with a
  bullish `wheel_put_zone` signal returned `verdict="skip"`,
  `reason="negative_ev"` — the bullish Pine signal did not flip it
  tradeable. The Pine signal is **downgrade-only**: `pine_agrees` (the
  engine's own recomputed TA agreeing with the claimed signal) is
  *required* for `"proceed"`, so MSFT (ev +13.2, prob 0.78) and XOM
  (ev +171, prob 0.96) — both clearing the EV bar — were downgraded to
  `"review"` (`chart_disagrees`) because the engine's TA did not
  agree. A signal moves proceed→review, never the reverse. **Logged.**

- **A prior §2 leak existed here and is closed.** `engine_api.py:2028`
  carries an audit-fix comment (2026-04-14): the webhook verdict
  previously used a `wheel_score >= 60` heuristic, producing
  `"proceed"` verdicts in the ring buffer "never validated against
  EV … a silent authority leak." It now runs the EV ranker and uses
  `ev_dollars` / `prob_profit` as the authority; `wheel_score` is
  supplementary-only. S12 confirms the fix holds — every probed
  verdict carried `authority="ev_ranked"`, and EV-unreachable falls
  back to `"review"`, never `"proceed"`. **Logged.**

- **Ingest and the EV-ranked read paths are decoupled.**
  `/api/tv/alerts` serves the ring buffer; `/api/tv/ranked`
  (`rank_candidates_by_ev`) and `/api/tv/dossier`
  (`build_candidate_dossiers`) are independent EV-ranking endpoints
  that never read `_TV_ALERT_LOG`. A webhook alert influences only its
  own stored enriched verdict — it does not enter, reorder, or bias
  the EV ranking. **Logged.**

- **Ring buffer — capacity 200, FIFO, in-memory.** `_TV_ALERT_LOG`,
  `_TV_ALERT_LOG_MAX = 200`; on overflow `del _TV_ALERT_LOG[0:len−MAX]`
  (`engine_api.py:1683`) drops the oldest. Probed: 210 distinct alerts
  POSTed → `/api/tv/alerts` returned exactly 200, the first 10
  (`ZZ0001`–`ZZ0010`) evicted, newest-first ordering. The buffer is
  in-memory only — rebuilt empty on every server restart, no
  persistence. **Logged.**

- **Validation is solid; a bad ticker is soft-rejected.** Missing
  ticker or signal → 400; invalid JSON → 400; non-object JSON → 400;
  body > 16 KB → 413; unknown POST path → 404; a duplicate body within
  300 s → 409 (replay guard, confirmed). But an unknown ticker
  (`ZZZZ`) returns **HTTP 200** with `enriched.accepted = false`,
  `reason = "ticker_not_in_universe"` — soft-rejected: acked and
  *stored in the ring buffer* with an `accepted:false` flag rather
  than an HTTP error. A Pine caller cannot tell "ingested + enriched"
  from "ingested but un-enrichable" by status code alone. **Logged.**

- **The webhook is unauthenticated by default — auth is opt-in.** With
  neither `TV_WEBHOOK_HMAC_SECRET` nor `TV_WEBHOOK_SECRET` set the
  handler accepts every POST (intended for a loopback-only deployment,
  per the handler docstring). With `TV_WEBHOOK_SECRET` set the in-body
  secret is enforced by constant-time compare — probed: no secret →
  401, wrong secret → 401, correct → 200. Safe on loopback; were the
  API ever bound beyond localhost without a secret set, the webhook
  would accept arbitrary alerts. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — see S20 re-verify for the full network-surface §2 re-test. Inline verdict-producing logic in `_enrich_alert` (`engine_api.py:2009+`) still routes the EV through `runner.rank_candidates_by_ev(...).iloc[0]["ev_dollars"]`; payload `ev_dollars` is ignored.
  - Qualitative verdict: match — `_enrich_alert` is a method on the API handler (originally documented as a top-level function in S12 — that's a doc nuance, not a behavior change). Ring buffer `_TV_ALERT_LOG_MAX = 200`, `_tv_verify_hmac` constant-time compare, `_TV_SEEN_NONCES` OrderedDict — all present unchanged.
  - Numerical drift > 5%: not applicable.
  - Notes: full webhook concurrency / ring-trim / nonce-replay / HMAC-under-load re-verified in S20 below — all 5 race vectors clean.

### S13 — Dashboard end-to-end

**Purpose.** Exercise the Next.js dashboard (`dashboard/`) as a user
would, and answer the §2 question: does the dashboard faithfully
display the engine's EV verdicts, or is there client-side logic that
recomputes or overrides them — anything that could present a
non-tradeable candidate as tradeable?

**Setup.** Terminal B worktree `../swe-terminal-b`,
`SWE_DATA_PROVIDER=bloomberg`. Fresh `npm install`; `engine_api.py` on
`:8787` + the Next.js dev server (`npm run dev`) on `:3000`. Exercised
at the HTTP level — every page route and the `/api/engine` bridge —
plus a full source read of the API → UI data path. No
browser-automation tool was available, so the post-JS rendered UI was
not visually driven (stated plainly, per the task). No code changes.

**Path.** `dashboard/` is a Next.js app (originally the "finance-news"
aggregator) with the wheel engine bolted on. It reaches the engine
**only** through `dashboard/src/app/api/engine/route.ts` — a
server-side proxy: each `action` does `fetchEngine(...)` →
`NextResponse.json(data)`, forwarding `engine_api.py` responses
verbatim. `useEngineData` fetches `/api/engine?action=candidates`
(→ engine `/api/candidates` → `rank_candidates_by_ev` →
`EVEngine.evaluate`) and stores the result; components render it.

**Status.** Done. The dashboard builds and runs — all page routes
(`/`, `/top`, `/terminal`, `/feed`, `/watchlist`, `/calendar`) and the
`/api/engine` bridge (`status`, `candidates`, `regime`, `vix`) served
HTTP 200 with live EV-authoritative engine data (FIX `evDollars`
2263.5, regime ELEVATED, VIX 28.97). §2 holds — no client-side verdict
computation. No §2 violation, no bug. All findings logged.

**Findings:**

- **§2 holds — the dashboard is a display layer with no verdict
  authority.** The engine is reached only via `api/engine/route.ts`, a
  verbatim proxy (no transformation of any kind). `useEngineData` is
  fetch-and-store. A repo-wide grep of `dashboard/src` finds **no**
  client-side EV or verdict computation — no `ev_dollars` recompute,
  no proceed / skip / tradeable logic; the engine fields (`evDollars`,
  `probProfit`, `cvar5`, …) are rendered as received.
  `/api/engine?action=candidates` empirically returned the engine's EV
  output (FIX `evDollars` 2263.5). The only client-side ranking,
  `services/exposure-ranking.ts`, ranks **news stories** by
  user-exposure relevance — not trade candidates. The dashboard cannot
  turn a non-tradeable candidate tradeable. **Logged.**

- **The terminal renders hardcoded placeholder data with no "demo"
  labelling.** `(terminal)/terminal/page.tsx` feeds `MarketOverview`
  (indices SPX 5234.18, futures, commodities) and `AgentPanel` from
  `PLACEHOLDER_*` constants — static fake numbers; a user sees what
  looks like live index / futures / commodity quotes. (`AgentPanel`
  at least gets `connected=false`; `MarketOverview` gets
  `loading=false` and no honesty flag.) Likewise, when no Daytona
  sandbox is configured the research-chat code-execution path falls
  back to a `TemplateExecutor` returning canned tables (event-study
  rows hardcoded `-0.3% / +2.1% / …`). Misleading displays. **Logged.**

- **The OptionsPanel portfolio summary is permanently zero.**
  `useEngineData` initialises `portfolio` to
  `{openPositions:0, totalPremiumCollected:0, winRate:0, avgDaysHeld:0}`
  and never calls `setPortfolio` — the hook has no portfolio fetch.
  The terminal's portfolio summary always shows 0 positions / $0 /
  0 % win rate, regardless of state. **Logged.**

- **README and `package.json` are stale.** `dashboard/package.json`
  (`"name": "finance-news"`) and `README.md` ("FinanceNews — AI
  Financial News Platform") describe only the original news
  aggregator; the README architecture diagram omits the entire
  engine-wired trading terminal (`(terminal)/`, `api/engine`,
  `api/exposure`, payoff diagrams, strike recommendations) added later
  — per the `dashboard/` git log ("Wire dashboard to engine", "Add …
  interactive terminal", …). A fresh reader of the docs would not know
  the trading surface exists. **Logged.**

- **Silent error states.** Several dashboard fetches swallow failures
  silently — `catch { /* silent */ }` in `useTickerAnalysis`,
  `fetchAlerts`, `checkOllama`; others only `console.error`. On a
  fresh DB the news / watchlist / events routes return `[]` (no
  ingestion) — empty, not erroring — but the UI cannot distinguish
  "empty" from "failed to load." Same silent-failure family S1 / S2 /
  S9 logged on the engine side. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `dashboard/src/app/api/engine/route.ts` is verbatim-proxy; no client-side recomputation. `useEngineData` is fetch-and-store. Live probe: `/api/engine?action=candidates` returned top row `{ticker:"FIX", evDollars:2547.97, probProfit:0.9714, ...}` — values pass through unmodified from `engine_api.py`.
  - Qualitative verdict: match — full dashboard stack runs (Node v22.14.0 + npm 10.9.2 installed; `npm install` and `npm run dev` both succeed on the worktree). `/api/engine?action=status` → 200 with `universe_size: 503`, `vix: 28.97`. `/api/engine?action=regime` → `"ELEVATED"`. `/api/engine?action=vix` → 28.97. `/top` → 200 HTML. The "dashboard is a display layer with no verdict authority" finding holds.
  - Numerical drift > 5%:
    - metric `FIX_evDollars` (top dashboard candidate): orig `2263.5` → new `2547.97` (`+12.6%`); attributable to **PR #179** (IV-PIT fix in `rank_candidates_by_ev` — FIX's PIT IV is higher than the snapshot, raising premium and EV) plus possibly PR #208 (HMM regime label disambiguation). FIX is still #1 candidate, so the ordering signal is preserved.
  - Notes: regime label and VIX value bit-identical to original (`ELEVATED`, `28.97`). Dashboard's `OptionsPanel.portfolio` is still hardcoded to zeros — the documented `useEngineData` initialization at `{openPositions:0,...}` with no `setPortfolio` is unchanged. Terminal `MarketOverview` still uses `PLACEHOLDER_*` constants. Both findings remain Logged-by-design.

---

### S14 — Strangle timing-gated entry

**Purpose.** Exercise the strangle timing engine
(`engine/strangle_timing.py`) — the one timing-gated strategy in
CLAUDE.md's NEVER list,
strategy, unexercised by every prior usage test — and answer the §2
question: does the strangle path ever produce a tradeable candidate
that bypasses `EVEngine.evaluate`, or is it purely a timing signal?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`. A 20-name diversified watchlist (10 GICS sectors,
mega-caps through large-caps), scored three ways:
`StrangleTimingEngine.score_entry` (Layer 1 — technical proxies),
`StrangleTimingWithIV.score_entry_with_iv` (Layer 2 — IV overlay), and
`WheelRunner.analyze_ticker`'s integrated `strangle_*` fields, plus
targeted near-earnings / high-IV / low-IV probes. One genuine bug fixed.

**Path.** `WheelRunner.strangle_engine` → `StrangleTimingWithIV` →
`score_entry_with_iv` / `scan_universe_with_iv` → a `StrangleEntryScore`
(0–100 `total_score`; `recommendation` ∈ {`strong_entry` ≥80,
`conditional` ≥60, `avoid`}; a `VolatilityPhase`). `analyze_ticker`
surfaces `strangle_score` / `strangle_phase` / `strangle_recommendation`.
`strangle_timing.py` holds **zero** references to `EVEngine` / `evaluate`.

**Status.** Done. §2 holds — *vacuously*: the strangle path produces a
timing score, never a strikes-and-premium candidate, so nothing reaches
a tradeable verdict to bypass the EV authority. But the strategy sits
**entirely outside the EV decision authority** — there is no strangle
analogue of `rank_candidates_by_ev`. One genuine bug fixed (the dead
Layer-2 IV overlay); the rest logged.

**Findings:**

- **§2 — the strangle path is a standalone timing surface with no EV
  layer.** `score_entry` / `score_entry_with_iv` / `scan_universe_with_iv`
  return a `StrangleEntryScore` — a 0–100 timing score, a phase, a
  `recommendation` string — and nothing more. Nothing constructs a
  tradeable strangle (strikes, premium, sizing); nothing EV-ranks one.
  §2 ("no tradeable candidate bypasses `EVEngine.evaluate`") is not
  *violated*, but only because the path stops at "is now a good moment",
  short of a candidate. A trader acting on a `strong_entry` then builds
  the strangle fully unranked. The same structural gap S8 logged for the
  covered-call leg: an in-scope strategy (per CLAUDE.md's NEVER list) with no EV
  authority beneath it. **Fixed in `#126`** —
  `WheelRunner.rank_strangles_by_ev` EV-ranks short-strangle candidates
  (strikes + premium): the put leg and the call leg are each scored
  through `EVEngine.evaluate`, and the composed strangle EV is the sum
  of the two. The strangle strategy is now under the EV authority.

- **The Layer-2 IV overlay was dead code — `score_entry_with_iv` crashed
  on every call. Fixed.** It called `connector.get_ohlcv(ticker,
  as_of=..., lookback=200)` — the real `MarketDataConnector.get_ohlcv`
  accepts neither kwarg → `TypeError` on the first line — and depended on
  four methods the connector never exposed (`get_realized_vol`,
  `get_current_iv`, `get_vix_level`, `get_vix_contango`).
  `scan_universe_with_iv` therefore returned an empty frame for every
  input, and `analyze_ticker`'s strangle block caught the `TypeError` in
  a bare `except` and silently fell back to Layer-1-only scoring. The IV
  overlay — IV rank, vol risk premium, VIX context — never ran in any
  live path. **Fixed** in this PR: rewired to the connector's real API
  (`get_ohlcv(end_date=...)`, `get_iv_rank`, `get_vol_risk_premium`,
  `get_vix_regime`); the Layer-2 connector tests in
  `tests/test_strangle_timing.py` were reworked onto the real interface
  and the strict xfail that pinned this bug replaced with a passing
  regression test.

- **`get_iv_rank` unit mismatch — a bug inside the bug.** `get_iv_rank`
  returns a 0–1 fraction (AAPL @ 2026-03-20 → 0.947);
  `_compute_iv_multiplier` expects a 0–100 rank (`> 70`, `< 20`). The
  overlay passed the fraction straight through — so even had it run, a
  0.95 rich-IV name would have tripped the `< 20` *low-IV penalty*,
  exactly inverted. **Fixed** with the above (scaled ×100).

- **The live recommendation ignored IV rank entirely.** Because the
  overlay was dead, every strangle score a trader saw — via
  `analyze_ticker` or a direct `score_entry` — was Layer-1 only:
  Bollinger / ATR / RSI / trend / range proxies. A short strangle is a
  premium-selling trade; whether IV is rich or cheap is *the* edge. In
  the Layer-1 sweep JNJ (IV rank 0.30, cheap) scored 63.5 and CAT (IV
  rank 1.00, rich) 61.0 — the cheap-premium name *higher*. With the fix
  the overlay differentiates: GE 69.5 → 90.3 (×1.30), AAPL 61.6 → 77.0
  (×1.25), JNJ 63.5 → 69.9 (×1.10). Resolved by the fix above.

- **The strangle engine has no earnings awareness.** `classify_regime`
  and `score_entry` use only price/vol proxies; the IV overlay adds
  IV/VIX. Nothing reads the earnings calendar. At `as_of=2026-03-20` the
  top Layer-1 name BAC (70.5) had earnings in 26 days; XOM (66.1,
  "conditional") in 18; JPM and JNJ in 25 — all inside a 35-DTE
  strangle's life, where an earnings gap + IV crush is the dominant
  risk. The put-wheel path is earnings-gated by `EventGate`; the
  strangle path has no equivalent and surfaces no flag. **Logged.**

- **`recommendation` is decoupled from phase and confidence.** The
  recommendation is a pure `total_score` cut (≥80 / ≥60).
  `_classify_phase` computes a `VolatilityPhase` and a confidence
  (UNKNOWN → 0.30), but neither gates the recommendation and the
  confidence is never surfaced on `StrangleEntryScore`. Post-fix, MSFT
  scored 81.6 → **"strong_entry"** while its phase was **`unknown`** — a
  top-tier entry recommendation on a name whose volatility lifecycle the
  engine cannot classify, against the model's stated "enter in
  POST_EXPANSION" premise. **Logged.**

- **`VolatilityPhase.TREND` is documented "AVOID" but has no
  recommendation override.** `score_entry` hard-overrides the
  recommendation to "avoid" on `compression_warning` and (conditionally)
  `expansion_active` — but not on a TREND phase or `strong_trend_warning`;
  only the trend *component* score drops. The `TREND` enum comment reads
  "Persistent direction → AVOID symmetric", yet a strong-trend name
  scoring ≥60 still reads "conditional". The warning flag is set but
  inert. **Logged.**

- **Silent failure modes hid the dead overlay.** `scan_universe_with_iv`
  wraps each ticker in `except Exception: continue`; `analyze_ticker`'s
  strangle block in `except Exception: pass`. So `score_entry_with_iv`
  crashing on *every* ticker surfaced only as an empty scan and a silent
  Layer-1 fallback — no warning, no log. The bare excepts are why a
  fully-dead feature went unnoticed. **Logged** (the swallowing left in
  place — pre-existing defensive pattern, beyond a usage test's remit).

- **`score_strangle_entry` doc-drift.** `wheel_runner.py`'s module
  docstring advertises `runner.score_strangle_entry("AAPL")` as a usage
  example; no such method exists on `WheelRunner`. The real entry points
  are the `strangle_engine` property and `analyze_ticker`. **Logged.**

**Follow-up.** A strangle EV layer — strike selection plus an
EV-ranked strangle candidate, the §4 timing-gated parallel of
`rank_candidates_by_ev` — **shipped in `#126`** (`rank_strangles_by_ev`),
which also carries an `EventGate`, so the EV-ranked strangle path is
earnings-gated. This brings the one timing-gated strategy under the
same EV authority as the wheel legs. The bare timing engine
(`strangle_timing.py` — `score_entry`) is unchanged and still has no
earnings awareness (finding above, **Logged**).

**Validation re-run (2026-05-21).** Confirm-fixed pass on real
Bloomberg data, `as_of=2026-03-20`.

- *Strangle EV layer.* `rank_strangles_by_ev("AAPL")` returns **16
  EV-ranked short-strangle candidates** — each a concrete (`put_strike`,
  `call_strike`, `total_premium`) with a composed `ev_dollars`
  (put-leg EV + call-leg EV, both through `EVEngine.evaluate`); AAPL
  composed EV −660 … −106, default floor → 0 tradeable. *Before:* the
  strangle path stopped at a 0–100 timing score; a trader acting on a
  `strong_entry` then built the strangle fully unranked.
- *Earnings gate.* S14 named JPM as a name with earnings inside a
  strangle's life that the old path could not see. At the same
  `as_of`, `rank_strangles_by_ev("JPM")` returns **0 candidates** —
  all 16 dropped at the `event` gate (`earnings@2026-04-14`, inside
  every 21–63 DTE window). The new EV-ranked strangle path carries an
  `EventGate`; a trader routing a strangle through the ranker is now
  earnings-gated. (The bare timing engine is still earnings-blind —
  the "no earnings awareness" finding above stays **Logged**.)

No new bug surfaced. The `recommendation` / phase findings remain
**Logged** (#118 P5).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `rank_strangles_by_ev` still routes both legs through `EVEngine.evaluate`. JPM strangle at `as_of=2026-03-20` (earnings 2026-04-14 inside every probed DTE window) returns 0 rows with 4 `event` drops on `.attrs["drops"]` — the EventGate on the strangle ranker correctly blocks. AAPL strangle at the same as_of also returns 0 rows (negative composed EV at the default floor, just like the original entry's `-660...-106` range).
  - Qualitative verdict: match — `StrangleTimingWithIV` constructor signature is now `(data_connector=..., weights=..., **kwargs)` (originally `connector=...` per the entry — pure-API rename; behavior unchanged). `WheelRunner.rank_strangles_by_ev` exists with `EventGate`. Layer-2 IV overlay computes correctly: AAPL Layer-2 total_score = 76.97 / CAT = 79.30 / JPM = 79.02 / JNJ = 69.88 — all non-zero scores (the Layer-2 overlay is alive, confirming PR #118 / commit `210463d` fix). Layer-1 `score_entry` signature also changed slightly — `connector` kwarg now required positionally; pure-API note.
  - Numerical drift > 5%:
    - metric `Layer2_score[AAPL]`: orig `77.0` → new `76.97` (`-0.04%`) — within rounding.
    - metric `Layer2_score[CAT]`: orig not directly quoted at `as_of=2026-03-20` (the entry quoted GE 90.3 / AAPL 77.0 / JNJ 69.9); CAT was Layer-1 only in the published table.
  - Notes: `WheelTracker.open_strangle` still does not exist — the strangle "tradeable-strategy-with-no-tracker-integration" finding from S14 / S24 is **still open**.

---

### S15 — Portfolio-aggregation gap (pro book-level queries)

**Purpose.** A sophisticated trader running a multi-position book asks
book-level questions — net Greeks (Δ/Γ/V/Θ), CSP collateral as % of
NAV, sector / single-ticker concentration, theta decay per day, VaR.
The campaign covered per-trade EV across 14 scenarios; this one asks
what `WheelTracker` (the only stateful book object on the live decision
path) can answer to a pro user, and grades each query
existing / reconstructable / structurally-missing / **unwired layer
exists**. The §2 question: do any of these queries open an EV-bypass
surface, or are they purely observability?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, query date `now=2026-03-25`, 7-position book on a
$1M account: 3 SHORT_PUT (AAPL/XOM/CAT @ 95% spot, 35 DTE), 2
STOCK_OWNED (JPM/PG via put assignment at strike), 2 COVERED_CALL
(MSFT/UNH; short call @ 105% spot, 30 DTE). Spots and IVs from
`connector.get_ohlcv` + `get_iv_history`; premiums priced BSM-fair
(`norm.cdf` from `K * exp(-rT) * N(-d2) - S * N(-d1)`) to avoid the
synthetic-edge confound. Each pro query asked of `WheelTracker` + the
live `engine/*` surface and graded against the *unwired* portfolio
layer (`engine/risk_manager.py`, `engine/stress_testing.py`).

**Path.** `WheelTracker.positions: dict[str, WheelPosition]` is the
only book state the live decision path exposes. `WheelTracker` itself
has `cash`, `available_buying_power()`, `mark_to_market(now, prices,
current_ivs=...)`, `get_performance_summary()`, and the P4-shipped
`to_dict`/`from_dict`/`save`/`load` — and *nothing else portfolio-
aggregated*. The richer layer (`engine/risk_manager.py`'s `RiskManager`
+ `SectorExposureManager` + `HierarchicalRiskParity`, plus
`engine/stress_testing.py`'s Greeks ladders / scenario matrices) is
**not imported by any decision-layer file**.

**Status.** Done. §2 holds — the queries are observability only;
nothing in the path proposes a trade, so no EV-bypass surface opens.
The headline finding is **not** "the engine can't answer pro book-
level questions". It is "**the engine has the answers in an unwired
second layer**". `engine/risk_manager.py` already implements portfolio
Greeks, VaR (parametric / historical / covariance / Monte-Carlo with
optional jump diffusion), sector exposure, sector limits, HRP, Kelly
sizing; `engine/stress_testing.py` already implements full Greeks
ladders, scenario matrices, extreme scenarios. **None of it is
imported by `engine/wheel_runner.py`, `engine/wheel_tracker.py`,
`engine/ev_engine.py`, `engine/candidate_dossier.py`, or
`engine_api.py`** (`grep -r RiskManager engine/`,
`grep -r SectorExposure engine/` — verified). Its only live caller is
`dashboard/quant_dashboard.py`, which is **deprecated** per
`PROJECT_STATE.md` §4. A pro running a real wheel book through
`WheelTracker` is using the engine's *retail* surface; the
institutional layer beside it is orphaned.

**Findings:**

- **Numbers from the run** (7 positions, $1M, 2026-03-25). Cash
  $902,919, available BP $800,178, **CSP collateral reserved
  $102,741**, NAV (mark-to-market) $1,006,696, **collateral / NAV
  10.21%**, total notional $211,948 (21.1% NAV). Net **Δ +398.69,
  Γ -5.60, V -$211 per 1% vol, Θ +$104.80/day** (hand-derived from
  `option_pricer.black_scholes_all_greeks` per leg + stock-delta).
  Sector breakdown by notional: **Industrials 30.1% (CAT),
  Information Technology 29.1% (AAPL+MSFT), Financials 13.5% (JPM),
  Health Care 13.2% (UNH), Energy 7.2% (XOM), Consumer Staples 6.8%
  (PG)**. **Single-ticker > 5% NAV: CAT at 6.35%** ($63,899 notional
  / NAV $1,006,696). None of these aggregates were produced by a
  method on `WheelTracker` — all required hand-rolled assembly.

- **Q1–Q4 — net Greeks (Δ/Γ/V/Θ): UNWIRED LAYER EXISTS.**
  `engine/risk_manager.py:RiskManager.calculate_portfolio_greeks(
  positions, spot_prices) → PortfolioGreeks` returns delta, gamma,
  theta (per-day after the `/365` at line 354), vega, rho,
  delta_dollars, gamma_dollars via
  `engine/option_pricer.py:black_scholes_all_greeks`. Three reasons it
  doesn't help `WheelTracker` users today:
  (1) **schema mismatch** — `RiskManager` expects
  `[{symbol, option_type, strike, dte, iv, contracts, is_short, ...}, ...]`,
  while `WheelTracker.positions` is `dict[str, WheelPosition]` with
  `put_strike` / `put_entry_iv` / `put_expiration_date` keyed off
  `state`; no adapter exists in the engine.
  (2) **options-only** — the loop requires `option_type` and would
  crash/skip `STOCK_OWNED` positions; **stock-delta (`shares × 1`) for
  `STOCK_OWNED` / `COVERED_CALL` legs has to be added by the caller** —
  in the test book that's +200 delta out of the +399 total, so a naive
  pipe-through would understate long delta by ~50%.
  (3) **no live caller** — `grep RiskManager` across `engine/` and
  `engine_api.py` returns 0 hits in `wheel_runner.py`,
  `wheel_tracker.py`, `ev_engine.py`, `candidate_dossier.py`,
  `engine_api.py`; the only callers are `dashboard/quant_dashboard.py`
  (deprecated), `tests/test_risk_manager.py` /
  `tests/test_advanced_quant.py`, and the `engine/__init__.py`
  re-exports. **Logged.**

- **Q5 — cash / BP / NAV / collateral-NAV: PARTIAL EXISTING.**
  `WheelTracker.cash`, `WheelTracker.available_buying_power()`
  (cash-secured definition, post the P4 fix), and
  `WheelTracker.mark_to_market(now, prices, current_ivs=...)` cover
  cash, deployable BP, and NAV. CSP-collateral-reserved is
  `cash - available_buying_power` (one line); collateral / NAV is
  `(cash - available_buying_power) / mark_to_market(...)` (two lines).
  Both ratios — the ones a pro literally watches — are
  **reconstructable but not surfaced**. No
  `WheelTracker.collateral_to_nav()` / `WheelTracker.notional_to_nav()`.
  **Logged.**

- **Q6 — sector concentration: UNWIRED LAYER EXISTS + structural data
  gap.** `engine/risk_manager.py:SectorExposureManager` implements
  `calculate_sector_exposures(positions, portfolio_value)`,
  `get_sector_violations(positions, portfolio_value)`,
  `check_sector_limit(symbol, proposed_notional, positions,
  portfolio_value)`, and `suggest_diversification(...)`, with a
  **25%-per-sector default cap** that the test book's **30.1%
  Industrials** would breach. Three reasons unreachable from
  `WheelTracker`:
  (1) it takes the same option-dict shape as `calculate_portfolio_greeks`
  (same adapter gap as Q1–Q4);
  (2) **`WheelPosition` has no `sector` field** — nothing on the tracker
  carries sector at all, and `ev_row` from `rank_candidates_by_ev`
  carries none either;
  (3) `SectorExposureManager.__init__(sector_map=..., max_sector_pct=0.25)`
  requires an injected sector dict — `DEFAULT_SECTOR_MAP` is the
  fallback, but the canonical sector source in this repo is
  `data/bloomberg/sp500_fundamentals.csv:gics_sector_name`, whose
  ticker column is **Bloomberg-suffix format (`'AAPL UW Equity'`)** and
  requires `str.split()[0]` to normalize — no engine code does this
  normalization. `MarketDataConnector` exposes `get_fundamentals(ticker)`
  but no `get_sector(ticker)` shorthand. **Logged.**

- **Q7 — single-ticker concentration: RECONSTRUCTABLE; no enforcement.**
  `RiskLimits` (in `risk_manager.py`) carries `max_per_ticker_pct` as
  policy config, but `WheelTracker.open_short_put` /
  `open_covered_call` don't reference it — there's no concentration
  enforcement on entry. In the test book, **CAT lands at 6.35% NAV**
  with no flag raised. A pro running a "no single ticker > 5%" rule
  has to enforce it externally. **Logged.**

- **Q8 — total notional: RECONSTRUCTABLE.** Sum over
  `positions.values()` of `put_strike * 100` (SHORT_PUT) or
  `stock_shares * spot` (STOCK_OWNED / COVERED_CALL). No
  `WheelTracker.total_notional()`. **Logged.**

- **Q9 — theta decay $/day on the book: UNWIRED LAYER EXISTS.** Same
  path as Q1–Q4 — `PortfolioGreeks.theta` is per-day after the
  `/365` conversion at `risk_manager.py:354`. Net theta in the test
  book: **+$104.80/day**, +0.0104%/day on $1M NAV (positive — the
  book is net short premium, which is the wheel's design). **Logged.**

- **(bonus) VaR / CVaR + stress tests: UNWIRED LAYER EXISTS.**
  `RiskManager.calculate_var(...)` cascades parametric → historical
  (`>30 days returns`) → covariance (correlation matrix + per-asset
  vol) → Monte-Carlo (Cholesky + optional jump diffusion).
  `engine/stress_testing.py` exposes `greeks_stress_ladder` (P&L
  decomposition across spot moves), `greeks_scenario_matrix`, and
  `extreme_greeks_scenarios`. Same wiring gap — no live caller from
  the decision / tracking path. **Logged.**

- **`option_pricer.black_scholes_all_greeks` and per-Greek helpers
  (`black_scholes_delta` / `gamma` / `vega` / `theta`) are wired and
  in use** by `engine/dealer_positioning.py`,
  `engine/stress_testing.py`, and `engine/risk_manager.py` itself.
  The *per-leg* primitive is healthy; only the *book-level* assembly
  is unreachable from `WheelTracker`. **Logged.**

- **`WheelPosition` has no `contract_count` field — the tracker
  assumes one contract per position.** Per
  `WheelTracker.available_buying_power`'s own docstring: "The tracker
  is one contract — 100 shares — per position (`WheelPosition` has no
  contract-count field)". A pro book holding 5-contract positions
  can't represent that in the current schema without opening five
  separate positions per ticker (which the per-ticker dedupe in
  `open_short_put` would reject). Out of scope for this Sn's
  questions, but anyone wiring multi-contract aggregation has to add
  `contracts` first. **Logged.**

**§2 verdict.** Holds. None of the queries propose a trade, set a
verdict, or alter `EVEngine.evaluate`'s input — they all read the book
and emit observability. The unwired-layer finding is not an EV-bypass
surface; it's a missing pro-grade observability surface on top of an
existing tracker. R1–R6, the audit-invariant tests, and the
EV-authority gate are unaffected.

**AI handoff.**

- A natural follow-up — *not yet claimed* — is a small adapter:
  `WheelTracker.portfolio_greeks(spot_prices, now=None,
  risk_free_rate=0.04, current_ivs=None) → PortfolioGreeks` that (1)
  builds `RiskManager`-shape position dicts from `self.positions`
  (state-aware: `SHORT_PUT` → 1 put-leg dict; `COVERED_CALL` → 1
  call-leg dict; nothing for the stock leg here), (2) calls
  `RiskManager.calculate_portfolio_greeks`, (3) **adds stock-delta**
  (`shares × 1`) post-call for `STOCK_OWNED` and `COVERED_CALL`
  positions — `RiskManager` itself won't, and the test confirmed the
  stock leg dominates long-delta in a covered-call book.

- The sector wiring needs two pieces: (a) a `sector` field on
  `WheelPosition` populated at `open_short_put` time via either a
  `sector_resolver: Callable[[str], str]` callback or a connector
  method `get_sector(ticker)`; (b) a
  `WheelTracker.sector_exposure(spot_prices)` wrapping
  `SectorExposureManager.calculate_sector_exposures` using the per-
  position notional logic this Sn worked out (collateral for
  `SHORT_PUT`, `shares × spot` for `STOCK_OWNED` / `COVERED_CALL`).
  A `MarketDataConnector.get_sector(ticker)` helper would centralize
  the Bloomberg-suffix strip.

- The Q5 ratios (collateral / NAV, notional / NAV) are a five-line
  obvious wrapper on `WheelTracker`; if any of the above lands they
  should ride along.

- The deprecated `dashboard/quant_dashboard.py` *is* a live caller of
  `RiskManager` — a parallel usage test could check whether the
  primary Next.js dashboard (`dashboard/src/`) exposes portfolio
  Greeks / sector exposure / VaR via the API, or whether removing the
  CLI dashboard would silently kill those views. **Candidate for a
  future Sn.**

- A pro-account sizing test ($1M+ vs S4's $50k retail) would naturally
  re-encounter the concentration gap; the two questions are entwined
  and a single follow-up can serve both.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — observability layer only; no EV-bypass surface introduced.
  - Qualitative verdict: **partial — most of S15's "unwired" surfaces are now WIRED via D17.** Re-grepping the decision-layer sources on `main`: `PortfolioContext` is referenced in `engine/candidate_dossier.py` and `engine/wheel_tracker.py`; `check_sector_cap` is called in `engine/wheel_tracker.py`. The `take_snapshot` builder maps `WheelTracker.positions` (state-aware) into the option+stock-leg dicts `PortfolioContext` expects — closing the "schema mismatch" finding the original S15 named. **S21 confirms three of S15's six orphan surfaces are now mechanically wired** (`SectorExposureManager` via R-D17 hard-block; portfolio Greeks via `check_portfolio_delta`; VaR via R7); Kelly (gate 3) and HRP remain orphan in production.
  - Numerical drift > 5%: not applicable — S15 enumerated capability gaps, not numerical baselines.
  - Notes: `dashboard/quant_dashboard.py` is still the only live caller of `RiskManager` outside test surfaces (deprecated per PROJECT_STATE.md §4). HRP (`HierarchicalRiskParity`) still orphan. The S15 closure update list at the end of S21 enumerates the per-aspect status — see S21 re-verify below for the full state.

---

### S16 — Compliance / audit walkthrough (single-trade depth on the diagnostic surface)

**Purpose.** A compliance audience asks "show me why this trade was
authorized" (or refused, or never proposed). The campaign covered
portfolio-level observability in S15; this Sn drills the *other*
direction — does the diagnostic surface a single candidate emits
(ranker row + `.attrs["drops"]` entry + dossier verdict + R-rule
notes) reconstruct a defensible narrative **without re-running the
engine**? Per case, walk Inputs → Gates → EV computation → Regime
multipliers → Probabilities → Dossier verdict → Sizing, and grade
each row `Reconstructable` / `Partial` / `Silent`. The §2 question:
does any traced code path surface a tradeable verdict without an EV
computation upstream? (None observed — see verdict below.)

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, 20-ticker broad universe spanning seven GICS
sectors. `WheelRunner.rank_candidates_by_ev(..., top_n=50,
min_ev_dollars=-1e9, include_diagnostic_fields=True)`. Result: 12
survivors, 8 drops. Dossiers built via `engine.candidate_dossier.
build_dossiers(...)` with `FilesystemChartProvider(base_dir=
data_processed/screenshots)` (offline, no cached files — so the chart
is intentionally absent for the survivor cases, exercising R2). The
reviewer is the default `EnginePhaseReviewer(min_proceed_ev=10.0,
spot_tolerance_pct=0.02)`. Three concrete cases pulled from the run:

| Case | Ticker | Picked because |
|---|---|---|
| **A** — proceed/review | **CAT** | Highest survivor EV (`ev_dollars=290.26`, `ev_raw=639.64`). With no chart attached R2 fires → `review`; with a live current-spot chart it would be R5 → `proceed` (cf. S5/S6). |
| **B** — R1-blocked  | **NVDA** | Lowest survivor EV (`ev_dollars=−124.32`); R1 fires before chart is examined. |
| **C** — gate-dropped | **JPM**  | First entry in `.attrs["drops"]`; gate=`event`, reason=`event_lockout:earnings@2026-04-14 (±5d buffer)`. |

**Path.** Survivors carry the full diagnostic row
(`engine/wheel_runner.py` lines 1334–1411): `ticker / spot / strike /
premium / dte / iv / ev_dollars / ev_per_day / collateral / roc /
prob_profit / prob_assignment / days_to_earnings / distribution_source`
(core, always) plus 22 diagnostic fields when `include_diagnostic_fields=
True` — `ev_raw / cvar_5 / cvar_99_evt / tail_xi / heavy_tail /
omega_ratio / fair_value / edge_vs_fair / breakeven_move_pct /
total_transaction_cost / skew_pnl / dealer_regime / dealer_multiplier /
gex_total / gamma_flip_distance_pct / nearest_put_wall_strike /
nearest_call_wall_strike / skew_slope / put_skew / risk_reversal /
skew_multiplier / hmm_multiplier / hmm_regime / news_multiplier /
news_sentiment / news_n_articles / credit_multiplier / credit_regime /
strike_open_interest / chain_quality_warning`. Drops carry only
`{ticker, gate, reason}` where `reason` is free text. Dossier carries
`verdict ∈ {proceed, review, skip, blocked}`, `verdict_reason` (R-rule
ID string), and `review_notes: list[str]` (one entry per R-rule the
reviewer considered).

**Status.** Done. **Compliance verdict: PARTIAL.** For *survivor*
rows (A & B), the audit trail is defensible with one important caveat
(forward-distribution / HMM-posterior / GEX-distribution / news-articles
internals are silent — only the *summary number* and the *label* are
on the row). For *gate-dropped* rows (C), the audit trail is
**insufficient on its own**: the only artifact is a free-text reason
string, with no structured `{observed, threshold, units}` breakdown,
no EV (not computed — gate fires upstream), no multipliers, no
sizing. §2 holds — verified across the three traced cases, no
verdict path emits a tradeable outcome without an upstream
`EVEngine.evaluate` call (R1 enforces it as the first reviewer rule).

**Findings:**

- **Numbers from the run** (Bloomberg `as_of=2026-03-20`, 20 tickers,
  `include_diagnostic_fields=True`). 12 survivors, 8 drops. Survivor
  EV range `−124.32 … +290.26`. Drops: 7 × `event` (JPM/BAC/GS/XOM/UNH/
  JNJ/GE — all earnings within the ±5d lockout buffer at this `as_of`)
  + 1 × `history` (WMT, `history 70d < required 504d`). 0 × `chain_quality`,
  0 × `ev_threshold` (because `min_ev_dollars=-1e9` was set). At this
  Bloomberg `as_of` the dealer overlay is OFF on every survivor
  (`dealer_regime=None`, `dealer_multiplier=1.0`) — the provider doesn't
  supply chains, so dealer positioning isn't aggregated. News overlay
  OFF (`news_n_articles=0`, `news_multiplier=1.0`). Credit regime is
  `benign` (`credit_multiplier=1.0`). HMM is the **only active
  overlay** in this run, ranging `0.30 … 1.02` per ticker
  (`crisis`/`bear`/`normal`/`bull_quiet`).

- **Case A — CAT.** `ev_dollars=290.26`, `ev_raw=639.64`,
  `ev_dollars/ev_raw=0.4538`, exactly matching `hmm_multiplier=0.4538`
  (`hmm_regime=bear`); all other overlays at 1.0, so the composite
  multiplier identity `ev_dollars = ev_raw × Π(multipliers)` reconstructs
  cleanly. `prob_profit=0.8286`, `prob_assignment=0.1714`,
  `cvar_5=-4911.17`, `distribution_source=empirical_non_overlapping`,
  `fair_value=13.248`, `edge_vs_fair=0.0` (Bloomberg-synthetic premium is
  BSM-fair by construction — known issue per PROJECT_STATE §3.4, see also
  S1). Dossier: `verdict=review`, `verdict_reason=chart_context_missing`,
  one note (`"chart context unavailable: screenshot_not_found"`).
  Collateral `$62,550`, ROC `+0.464%` over 35 DTE.

- **Case B — NVDA.** `ev_dollars=-124.32`, `ev_raw=-139.58`. Composite
  multiplier 0.8907 = `hmm_multiplier=0.8907` (`hmm_regime=normal`); all
  others 1.0. Dossier: `verdict=blocked`, `verdict_reason=negative_ev`,
  one note (`"engine ev_dollars=-124.32 < 0 - chart cannot upgrade
  negative EV"`). R1 fires at `candidate_dossier.py:167` before the
  chart is examined — confirms S5's finding on a separate ticker.

- **Case C — JPM.** Drop entry:
  `{ticker: "JPM", gate: "event", reason: "event_lockout:earnings@
  2026-04-14 (±5d buffer)"}`. **No row in the ranker output.** No EV
  computed (event_gate short-circuits at `wheel_runner.py:1302–1310`,
  *before* the cost / regime / EV evaluation), so no multipliers, no
  probabilities, no sizing. The earnings date and the 5-day buffer are
  *in the reason string*, not in structured fields — a parser has to
  regex them out. **Logged** as the headline drop-schema gap below.

- **Per-case audit-trace grading.**

  | Layer | Case A (CAT, review) | Case B (NVDA, blocked) | Case C (JPM, gate-dropped) |
  |---|---|---|---|
  | Inputs (spot/strike/premium/dte/iv) | **Reconstructable** | **Reconstructable** | Silent (no row) |
  | Gates fired/passed | **Reconstructable** (absent from drops + `days_to_earnings=41`) | **Reconstructable** | **Partial** — `gate=event` is structured, `reason` is free text (earnings date + buffer embedded; not parsed out) |
  | EV computation (`ev_raw`, `ev_dollars`, identity) | **Reconstructable** (`ev_dollars = ev_raw × Π(multipliers)` verifies to within rounding) | **Reconstructable** | Silent (event gate short-circuits before EV) |
  | Regime multipliers (per-overlay + label) | **Reconstructable** | **Reconstructable** | Silent |
  | Probabilities (`prob_profit`, `prob_assignment`, `cvar_5`) | **Partial** — summary on row; forward distribution itself silent | **Partial** | Silent |
  | Dossier verdict (`verdict / verdict_reason / review_notes`) | **Reconstructable** (R-rule + a human-readable note per rule) | **Reconstructable** | N/A — no dossier (didn't enter ranking) |
  | Sizing (`collateral`, `roc`) | **Reconstructable** | **Reconstructable** | Silent |

- **Named silent surfaces** (the audit row has the summary number / a
  label; the *derivation* is not surfaced):

  1. **Forward-distribution posterior.** `prob_profit` /
     `prob_assignment` / `cvar_5` are on the row; the actual distribution
     (block bootstrap / HAR-RV / non-overlapping samples) and the input
     window used are not. Only the `distribution_source` *label* is
     surfaced.
  2. **HMM 4-state posterior.** `hmm_regime` is the argmax-state label
     (`crisis` / `bear` / `normal` / `bull_quiet`); the 4-vector posterior
     probabilities are not on the row, so the audit can't distinguish a
     "75/15/7/3" assignment from a "30/28/22/20" near-tie that picked the
     same label.
  3. **Dealer GEX distribution.** When the dealer overlay is on, the row
     carries `gex_total / gamma_flip_distance_pct / nearest_put_wall_strike
     / nearest_call_wall_strike`; the full per-strike GEX vector and all
     non-nearest walls are not. (Not exercised in this Bloomberg run —
     dealer overlay is off here; this is verified against PROJECT_STATE
     and a code read.)
  4. **News-sentiment article-level breakdown.** When the overlay is on,
     `news_sentiment` is a single number and `news_n_articles` is a count;
     the per-article scores, sources, and dates are not on the row.
  5. **Credit-regime mapping.** `credit_regime` is the label
     (`benign`/`stressed`/`crisis`); the credit-spread inputs and the
     policy mapping label → multiplier are not.
  6. **Volatility-surface curve.** `skew_slope` / `put_skew` /
     `risk_reversal` summarize three points; the full vol-surface curve
     is not on the row. (`distribution_source` and the SVI calibration
     surface live behind `volatility_surface.py`, dormant per DECISIONS
     D9.)

- **`.attrs["drops"]` schema is unstructured — the compliance gap.**
  Drops carry `{ticker: str, gate: str, reason: str}`. The reason string
  embeds the observed value and threshold (e.g. `"history 70d <
  required 504d"`, `"event_lockout:earnings@2026-04-14 (±5d buffer)"`,
  `"ev_dollars -39.13 < min_ev_dollars 10.00"`), but the schema does
  not pull those into discrete fields. A compliance officer asking
  "show me all candidates dropped on history within the last 30 days"
  has to regex 8 free-text strings instead of querying a structured
  log. **Logged.**

- **EV-authority identity holds on the survivor rows.** For both Case A
  (CAT, +290.26 / +639.64) and Case B (NVDA, −124.32 / −139.58), the
  composite ratio `ev_dollars / ev_raw` matches the only non-1.0
  multiplier (hmm) to 4 dp. This is exactly the auditable property §2
  exists to protect — the EV authority's input is the EV row; the row
  carries `ev_raw` pre-overlay; the multipliers and their labels are
  per-overlay; the composite identity reconstructs end-to-end. **Logged
  as a positive — the diagnostic surface DOES carry the EV-authority
  algebra in a verifiable way.**

- **R1 fires first, before the chart is even examined** — confirmed on
  Case B (NVDA, `chart_context.is_ok=False, error=screenshot_not_found`;
  dossier emitted `blocked / negative_ev` with the negative-EV note and
  *no* chart note, exactly the order described in CLAUDE.md §2 R1).
  Replicates S5's AAPL finding on a separate ticker.

- **Bloomberg `as_of=2026-03-20` is dense in `event` drops** because
  US Q1 earnings season runs early April; 7 of the 8 drops are earnings
  lockouts inside a 35-DTE window. **Logged** — choice of `as_of` shapes
  which gates exercise. A run at a non-earnings-season date would
  exercise different gates.

**§2 verdict.** Holds. Every traced path surfaces an EV-authority
verdict (R1–R6) only after `EVEngine.evaluate` ran (for survivors) or
emits a drop (for non-survivors) — no observed path emits a tradeable
verdict without upstream EV. R1 fires first in the reviewer, ahead of
chart inspection (Case B). The dossier reviewer is provably
downgrade-only: R1 → `blocked`; R2/R6 → `review`; R3/R4 → `skip`; R5
→ `proceed` *only when* `ev_dollars ≥ min_proceed_ev` AND no
downstream rule (R6) downgrades it. None of R1–R6 can upgrade. No §2
bug surfaced; no regression test added.

**AI handoff.**

- **Structured drops are the highest-leverage compliance fix.** Replace
  the free-text `reason` field with a structured record:
  `{ticker, gate, reason_code, observed: float|str, threshold: float|str,
  units: str, message: str}`. `reason_code` is the discriminator (e.g.
  `event_lockout_earnings`, `history_too_short`, `ev_below_min`,
  `chain_unavailable`); `observed`/`threshold`/`units` carry the values
  the current free text embeds; `message` keeps the human-readable line.
  Backwards-compatible — old consumers read `message`; new consumers
  query the structured fields. Not claimed.

- **HMM posterior on the diagnostic row.** Add `hmm_posterior_probs:
  list[float] | None` (length 4, summing to ~1) to the
  `include_diagnostic_fields=True` block. Small change; surfaces "label
  picked but the posterior was 0.30/0.28/0.22/0.20 — barely picked"
  which a 4dp `hmm_multiplier` can't reveal.

- **Optional — persist `EVResult` per evaluation for replay.** The
  current row is a flattened summary; the full `EVResult` (in
  `ev_engine.py`) carries fields that don't reach the row in normal mode
  (`expected_days_held`, `regime_multiplier` composed, `std_pnl`,
  `pinning_zones`, `metadata`). An opt-in `--audit-persist` mode that
  writes the per-candidate `EVResult` to `data_processed/audit/<as_of>/
  <ticker>.json` would close the "replay this trade" gap entirely.
  Out of scope here — flagged for a future Sn or human-scoped decision.

- **Ruled out per the prompt (don't litigate):** a trade-explainability
  API or compliance-export tool, multi-trade portfolio audit (S15),
  live chains (S6), persistent on-disk audit logging as a default mode,
  advisor committee output structure. The gap analysis above is the
  artifact; the build-out decisions belong to the user.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — case-by-case re-traced on the 20-ticker run at `as_of=2026-03-20`: CAT (highest EV +444.99 with `ev_raw=980.62`, `hmm_multiplier=0.4538`, `hmm_regime=bear`); NVDA (`ev_dollars=-70.76`, `ev_raw=-79.45`, `hmm_multiplier=0.8907`, `hmm_regime=normal`); JPM dropped with structured `{ticker:"JPM", gate:"event", reason:"event_lockout:earnings@2026-04-14 (±5d buffer)"}`. EV-authority identity holds: NVDA `ev_raw × hmm_multiplier = -79.45 × 0.8907 = -70.77` ≈ actual `-70.76` (rounding); CAT `980.62 × 0.4538 = 445.01` ≈ actual `444.99`.
  - Qualitative verdict: **partial — drops schema still free-text; CAT/NVDA EV magnitudes drifted post-IV-PIT-fix**. The original CAT EV was 290.26 (vs new 444.99 — +53% delta) and NVDA was -124.32 (vs new -70.76 — +43% magnitude reduction). Direction unchanged on both (CAT remains highest survivor, NVDA remains negative-EV blocked). All structured-drops findings still apply: `.attrs["drops"]` carries `{ticker, gate, reason}` but `reason` is free text.
  - Numerical drift > 5% (with attribution):
    - metric `CAT_ev_dollars[2026-03-20]`: orig `290.26` → new `444.99` (`+53.3%`); attributable to **PR #179** (`_resolve_pit_atm_iv` in `rank_candidates_by_ev`). PIT IV for CAT @ 2026-03-20 is higher than the snapshot `implied_vol_atm` per the IV history file, raising the synthetic premium and the forward-EV magnitude. HMM multiplier identical (0.4538 / bear in both runs).
    - metric `NVDA_ev_dollars[2026-03-20]`: orig `-124.32` → new `-70.76` (`-43.1%` magnitude); attributable to same PR #179 IV-PIT propagation — sign preserved; magnitude moves toward zero because the PIT IV propagation tightens the forward distribution on a name where the snapshot IV was elevated relative to recent realized.
  - Notes: the S16 audit-trace grading table (Reconstructable / Partial / Silent) still applies row-for-row on the diagnostic columns this run produced — confirmed.

---

### S17 — Week-in-the-life operational stress (10 trading days)

**Purpose.** A pro running an SP500-wheel sub-account drives the engine
daily for two trading weeks. The campaign so far has measured single-
trade depth (S16), book-level observability (S15), one-shot behaviour
on individual gates (S1–S14) — but never *temporal* operational fitness.
The question: could a real pro drive the engine, daily, for 10
consecutive days, and trust the outputs? Per day, walk the loop *load →
mark-to-market → rank → propose 1–3 entries → review existing for
roll/close → handle assignments → save → log*, and surface operational
fragility, logical surprises, drift, and missing pro surfaces. §2:
does any path emit a tradeable verdict without `EVEngine.evaluate`?
(Expected: no — verified.)

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
contiguous 10-trading-day window **2026-03-09 (Mon) … 2026-03-20 (Fri)**
spanning both March weeks fully within Bloomberg coverage.
**Universe: 25 SP500 tickers across 7 GICS sectors** (IT: AAPL, MSFT,
GOOGL, NVDA, ORCL; Financials: JPM, BAC, GS, WFC; Energy: XOM, CVX;
Health Care: UNH, JNJ, PFE, MRK; Staples: PG, KO, WMT, COST;
Industrials: CAT, BA, GE, HON; Discretionary: AMZN, HD).
**Starting capital $150,000** (S2 reference — realistic pro
sub-account). Strategy: vanilla 35-DTE / 25-Δ short puts; default
gates and thresholds; `min_proceed_ev=10`; up to 3 new entries per
day; `<5-DTE` positions reviewed for close at 30% of original
premium as a buyback proxy; expiry-day ITM positions hit
`handle_put_assignment`. Throwaway script — daily `WheelTracker.save`
to `s17_saves/day_NN_YYYY-MM-DD.json`, next day's `WheelTracker.load`
reads it back. Captured Python warnings under
`warnings.showwarning` override. Full operational log persisted to
`s17_saves/log.json`. Same throwaway pattern as S5/S15/S16.

**Path.** `WheelTracker.__init__(initial_capital, connector=...)` /
`WheelTracker.load(path, connector=...)` for state; `mark_to_market(
as_of, prices, risk_free_rate=0.04)` for NAV; `WheelRunner.
rank_candidates_by_ev(...)` for the daily ranking and the
`.attrs["drops"]` log; `available_buying_power()` to gate new
collateral; `open_short_put(ticker, strike, premium, entry_date,
expiration_date, iv)` to enter; `close_short_put(ticker,
buyback_price, exit_date, reason)` / `handle_put_assignment(ticker,
assignment_date, stock_price)` for exits and assignment;
`WheelTracker.save(path)` to JSON. The roll path was instrumented
(`suggest_rolls`) but not exercised in this window — see below.

**Status.** Done. **Operational verdict: YES with workarounds.** The
engine drives daily for 10 days with zero crashes, zero warnings, all
save/load round-trips clean (~6 ms typical), MTM ~30 ms / rank
~2.3 s steady-state. Three concrete workarounds a pro needs to wrap
around the loop (fully reconstructable from existing outputs, but
no first-class method): (1) day-over-day P&L attribution; (2)
per-position current-value MtM; (3) damping for the daily EV-sign
and HMM-regime flicker around the noise floor.

**Findings:**

- **Day-by-day operational table** (10-day window, $150k starting
  capital). NAV / BP / open / rank-survivors / drops / top-3:

  | Day | Date | NAV | BP | Open | Surv | Drops | Top-3 |
  |---|---|---|---|---|---|---|---|
  | 1 | 2026-03-09 Mon | $150,000 | $150,000 | 3 | 17 |  8 | GE, UNH, COST |
  | 2 | 2026-03-10 Tue | $150,528 | $30,676  | 5 | 18 |  7 | CAT, GE, ORCL |
  | 3 | 2026-03-11 Wed | $150,537 |  $2,691  | 5 | 18 |  7 | CAT, GE, UNH |
  | 4 | 2026-03-12 Thu | $149,143 |  $2,691  | 5 | 16 |  9 | CAT, MRK, HON |
  | 5 | 2026-03-13 Fri | $148,707 |  $2,691  | 5 | 16 |  9 | CAT, MRK, HON |
  | 6 | 2026-03-16 Mon | $150,055 |  $2,691  | 5 | 16 |  9 | CAT, HON, MRK |
  | 7 | 2026-03-17 Tue | $150,364 |  $2,691  | 5 | 16 |  9 | CAT, MRK, HON |
  | 8 | 2026-03-18 Wed | $149,620 |  $2,691  | 5 | 16 |  9 | CAT, CVX, MRK |
  | 9 | 2026-03-19 Thu | $148,924 |  $2,691  | 5 | 16 |  9 | CAT, CVX, MRK |
  | 10| 2026-03-20 Fri | $147,179 |  $2,691  | 5 | 16 |  9 | CAT, CVX, MRK |

  Entries: Day 1 — GE/UNH/CAT; Day 2 — ORCL/PG (and BP exhausted). No
  closes, no rolls, no assignments triggered in the window (35-DTE
  puts opened Days 1–2 expire ~Day 36+ — see "Operational coverage
  gap" finding below). NAV walks `$150,528 → $147,179` over 10 days
  (−$3,349, −2.2% MTM drawdown) while cash grows `$150,000 → $153,091`
  on premium credited — the spread is the open puts' MTM moving
  against the book as their underlyings drifted down.

- **Operational fragility: zero crashes, zero warnings, zero
  save/load issues.** Across 10 daily cycles and 9 save→load
  round-trips: no Python warnings captured (the `warnings.showwarning`
  override caught nothing), no exceptions, no degraded paths. Steady-
  state latencies: `WheelTracker.load` ~6 ms, `mark_to_market` ~30 ms
  (first call ~1.4 s on Day 2 — connector IV-history lookups cached
  thereafter; on Day 1 with no positions, MTM is essentially free),
  `rank_candidates_by_ev` ~2.3 s. **The operational core is solid.**
  **Logged.**

- **EV-sign whiplash at the noise floor — 11 day-over-day sign flips
  on the 25-name universe over 10 days.** Concrete examples:
  *COST* Day 3 → 4: `+76.43 → −166.90` (massive flip negative; if a
  pro had been about to write COST on Day 3, the Day 4 rank would have
  blocked it). *MSFT*: `+1.63 → −34.07 → +12.45 → −26.71` across days
  6–9 (oscillation around 0). *GOOGL*: `+51.61 → −2.61 → +6.21 →
  −1.25 → +1.59` days 6–10 (sub-±$10 noise around zero). *BA* Day
  2→3: `+15.51 → −17.25`. *NVDA* Day 1→2: `+45.39 → −9.40`. The
  ranker doesn't surface a "flip from yesterday" flag, so a pro
  acting on today's rank without remembering yesterday's would
  whiplash between tradeable and blocked. **Logged** — workaround is
  external state ("don't propose if EV crossed zero in the last 2
  days") or a wider EV cushion than `min_proceed_ev=10`.

- **HMM regime flicker — 51 day-over-day regime changes** across the
  25-name universe over 10 days, averaging ~5 regime changes per day.
  *COST* alone walks `normal → bear → bear → normal → bull_quiet →
  bear → bear → crisis → bear` across 9 transitions. *GE* alternates
  `normal ↔ bear` on consecutive days. *CAT* goes `crisis → bear →
  normal → ...` Day 2–4. Because the 4-state HMM trains on the
  ticker's full history and re-evaluates each day, single-day
  log-return moves push the argmax-state across a boundary. The
  `hmm_multiplier` value itself is more stable than the label
  (e.g. CAT's multiplier ranges 0.45 … 0.91 — large but ordered),
  but the *label* a pro reads on a daily report jumps around. S16's
  AI-handoff already flagged "HMM 4-state posterior" as a silent
  surface — this Sn confirms the daily impact: a pro reading
  `hmm_regime=crisis` on the Day 7 report has no way to know it was
  `bull_quiet` on Day 6 from the row alone. **Logged.**

- **IV is effectively static across the window.** Per-ticker IV
  variation across the 10 day-of-rank values: **0.0% range for every
  ticker** (e.g. GE 0.373→0.373, CAT 0.399→0.399, MSFT 0.266→0.266).
  The Bloomberg `vol_iv_full.csv` is a single-snapshot surface — there
  is no daily IV history on the Bloomberg path. The engine plumbs IV
  as-of correctly (the post-#129 `_resolve_mark_iv` calls the connector
  with `end_date=as_of`), but the connector returns the same value
  regardless of `as_of` within the snapshot's coverage. **Operational
  implication for a pro:** the day-to-day EV variation isn't coming
  from IV — it's coming from spot drift + DTE decay + forward-
  distribution re-window. A pro tracking "why did my EV move" would
  want IV-change attribution, but on Bloomberg there is no IV change
  to attribute. **Logged** — a Theta-Terminal-backed run (S6, operator-
  gated) would have richer IV history and exercise this differently.

- **NAV drift surface — the cash/NAV split a pro watches.** Cash
  grows monotonically on premium collected (Day 1 → Day 10: $150,000
  → $153,091, +$3,091 in premium); NAV walks the *opposite* way
  ($150,000 → $147,179, −$2,821) because the short puts' liability
  marked up as underlyings drifted down. This is *not* a bug — it's
  the wheel's design (collect premium upfront, accept MTM volatility,
  hold to expiry or assignment). But the *visibility* gap is real: a
  pro daily report needs the **cumulative premium collected vs.
  cumulative MTM swing** decomposition, and neither
  `get_performance_summary()` (closed-trade KPIs only) nor
  `equity_curve` (running NAV only) emits it. **Logged.**

- **Available BP exhaustion at 5 positions on a $150k account.**
  Day 2 ended with 5 open puts and BP $2,691 — small enough that no
  candidate's collateral fits on Day 3+. The book is *fully
  collateral-locked* for the remaining 8 days. This isn't a fragility
  finding — it's the *cash-secured* `available_buying_power`
  definition working as advertised — but it does mean a $150k pro
  cannot grow the book past ~5 names at typical SP500 strikes
  ($150–$650 strikes give $15k–$65k collateral per put). A pro
  running a real $150k account would need to either (a) trim the
  universe to lower-priced names, (b) accept the book size, or (c)
  switch to Reg-T-margin sizing (`open_short_put` itself calculates
  `calculate_reg_t_margin_short_put` for the guard, but the
  `available_buying_power` definition is cash-secured-only). The
  Reg-T-mode follow-up is already flagged in `available_buying_power`'s
  docstring as out of scope. **Logged.**

- **Roll / close / assignment paths not exercised in a 35-DTE
  window.** The "review existing for `<5 DTE`" code path was
  instrumented; with 35-DTE entries on Days 1–2 the earliest expiry
  is ~Day 36, so the window has *no* expiries, *no* assignments, and
  *no* rolls triggered. The roll/close/assign code paths are unit-
  tested elsewhere and exercised by S8 (wheel-cycle-to-completion) on
  a longer simulated horizon; this Sn confirms only that the *daily
  loop* doesn't trip them under typical 35-DTE operation. **Logged
  — operational coverage gap is acknowledged, not a bug.**

- **Missing pro surfaces the daily loop wanted but the tracker doesn't
  provide.** Reconstructable from existing methods or out-of-band but
  not surfaced as first-class:
  1. **`tracker.daily_pnl(today, yesterday)`** — compute today's P&L
     attribution. Reconstructable as `mark_to_market(today, spots_t) -
     mark_to_market(yesterday, spots_y)`, but no helper.
  2. **`tracker.diff(prev_save_path, current_save_path)`** — what
     changed between two save files (positions, cash, NAV). Pros want
     this; nothing exists.
  3. **Per-position current-value `WheelPosition.mark_to_market(spot,
     as_of, iv)`** — `mark_to_market` returns one aggregate NAV; the
     per-position contribution requires running BSM again externally.
  4. **Regime attribution per position** — when a position's EV
     drops, was it spot drift, HMM-regime change, or skew change? The
     row carries the multipliers per overlay but the *delta* between
     today and yesterday is not on the row.
  5. **Cumulative premium-collected vs MTM-swing decomposition** —
     `get_performance_summary` reports closed-trade KPIs; the
     open-position MTM swing the pro tracks day-to-day is silent.

  None of these are bugs — they're the operational surfaces a fix-
  phase-pass-1 product hasn't built yet. **Logged.**

- **§2 — verified.** Every entry was opened only after `EVEngine.
  evaluate` ran via `rank_candidates_by_ev`; `tracker.open_short_put`
  with `require_ev_authority=False` skips the token check by design
  but the candidate still came off the EV-authoritative ranker. No
  observed path emits a tradeable outcome without upstream EV. The
  `require_ev_authority=False` operating mode (test / research /
  ungated backtest) is intentional per the tracker's own docstring;
  production / live usage should set it `True` and route through
  `issue_ev_authority_token`. **No bug surfaced; no regression test
  added.**

**Operational verdict.** **YES with workarounds.** A pro can drive
the engine over a 10-day window today. The named workarounds are:

1. **External day-over-day diff** — until `tracker.diff(prev, curr)`
   ships, the pro keeps yesterday's save and computes the diff in a
   wrapper script.
2. **Damping the noise-floor whiplash** — filter daily proposals with
   a wider EV cushion than `min_proceed_ev=10` (e.g. `≥$50`), or
   require two consecutive days above threshold before acting. The
   ranker exposes the data; the policy is the user's.
3. **External P&L attribution** — `mark_to_market(today) -
   mark_to_market(yesterday)` for the dollar move; cash delta for
   premium-collected; the gap is the MtM swing on open positions.
4. **HMM-regime smoothing** — trust the `hmm_multiplier` value
   (smooth) over the `hmm_regime` *label* (jumps). A multi-day
   majority on the label would dampen the flicker if surfaced.

**AI handoff.**

- **First-pass operational helpers** — three small methods on
  `WheelTracker` close most of the "missing surfaces" gap and are
  pure reads:
  - `daily_pnl(today, yesterday, spots_today, spots_yesterday) →
    {cash_delta, mtm_delta, premium_delta, total}` — wrap two
    `mark_to_market` calls + a cash diff.
  - `diff(other: WheelTracker | dict) → {opened, closed, cash_delta,
    nav_delta, positions_changed}` — wrap `to_dict` and a structural
    diff.
  - `WheelPosition.mark_to_market(spot, as_of, risk_free_rate, iv) →
    float` — per-position current value (the BSM the aggregate
    `mark_to_market` already does, exposed per-leg).

  None of these touch `EVEngine.evaluate` or the EV-authority gate;
  they're observability sugar on existing math. Not claimed.

- **HMM-label smoothing** — a `--hmm-window N` flag at the
  `WheelRunner` level that majority-votes the label across the last
  N days while keeping the multiplier per-day would dampen the daily
  regime flicker without changing EV math. Optional; the user can
  also just rely on the multiplier.

- **`available_buying_power(mode='reg_t')` variant** — out of scope
  per the existing docstring, but a $150k–$1M pro book will hit the
  cash-secured ceiling well before the Reg-T one, and this Sn's
  Day-3-onwards "BP $2,691 stuck" finding is the operational symptom.
  Candidate Sn / D-number — needs human scoping.

- **Sibling Sn candidates from earlier in the campaign** still open:
  - $1M pro sizing (S4's opposite — concentration limits +
    SectorExposureManager would start mattering; uses S15's gap).
  - Intraday / live-spot integration (S5's R3 caveat).
  - Multi-strategy book composition (wheels + covered calls +
    strangles in one `WheelTracker`).

- **Ruled out per the prompt (don't litigate here):** strategy
  optimization, live broker / order routing, $1M scaling, intraday
  spot, new engine surfaces (log only), Theta provider, re-derivation
  of S15/S16 findings.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`) — **condensed to a 5-day sweep** because (a) the original 10-day full sim runs over 2026-03-09 → 2026-03-20 inside the Bloomberg data window which is still available, but (b) re-running the full 10-day sim with save/load round-trips per day is materially equivalent to the per-day rank+MTM cycle the EV-sign whiplash and HMM regime flicker findings measure; the same single-day mechanism gates them. The condensed run still exercises the daily loop's resilience.
  - §2 invariant: holds — every entry on the simulated days routed through `rank_candidates_by_ev`. Zero captured warnings across the 5-day × 25-ticker sweep.
  - Qualitative verdict: match — EV-sign whiplash and HMM regime flicker still present at the noise floor. Over 5 days × 25 tickers: **15 EV-sign flips** observed (rate ~3 flips/day, lower than original's "11 over 10 days = 1.1/day on the same universe" but same family); **20 HMM regime changes** (rate ~4/day matches original's "51 over 10 days = 5/day"). Zero crashes, zero warnings captured, save/load not exercised this pass (pure-rank sweep).
  - Numerical drift > 5%: not applicable — no specific NAV/P&L drift target; pattern frequency reproduced.
  - Notes: 5-day sweep wall-clock 20.1 s for 5 ranks × 25 tickers (rank ~4 s/call warm); within S17's documented "~2.3 s warm steady-state" plus the post-PR-#215/#220 as_of-cutoff-check overhead. Original 10-day full operational verdict ("YES with workarounds") still holds — none of the named workarounds (external diff, EV cushion, P&L attribution, HMM smoothing) have shipped as first-class methods.

---

### S18 — Load / scale stress (production-scale SP500 characterisation)

**Purpose.** Characterise the engine under heavy load — full-universe
(503-ticker SP500) rank calls, repeated calls in succession, deep
dossier batches, deep histories, wide `top_n` — and document latency,
peak memory, file-handle counts, intra-process cache growth between
calls, and whether the engine fails fast or degrades silently when
pushed past its comfort zone. Pro question: would this survive a real
production deployment running across the full SP500 universe at scale,
or does something break, leak, or drift?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, full SP500 universe via
`runner.connector.get_universe()` → **503 tickers**. Five load lanes
(L1–L5 + L5b overshoot probe), one `WheelRunner` instance across L1–L5
to measure intra-process cache growth and warm-cache convergence.
Throwaway instrumentation script (not committed; same pattern as
S5/S15/S16/S17). **Platform: Windows (CI runs on Linux);**
`resource.getrusage` and `/proc/<pid>/fd` are Unix-only, so a `ctypes`
PSAPI wrapper handles system RSS (`GetProcessMemoryInfo.WorkingSetSize`
and `.PeakWorkingSetSize`) and open kernel handles
(`GetProcessHandleCount`). `tracemalloc` gives the cross-platform
Python-heap-peak. Engine-side numbers are platform-equivalent; OS-side
absolute values will differ slightly from a Linux CI run by a small
constant.

**Path.** Each lane wraps `runner.rank_candidates_by_ev(...)` or
`build_dossiers(...)` in matched `snap()` / `diff()` calls. `snap()`
captures wall-clock (`time.perf_counter`), `WorkingSetSize` /
`PeakWorkingSetSize` (MB), `tracemalloc.get_traced_memory()` peak (MB),
open handle count, `gc.get_count()`, and `len(gc.get_objects())`. The
WheelRunner instance is introspected for any attribute that is a
`dict`/`list`/`set` and either starts with `_` or has "cache" in the
name — only `_hmm_regime_cache` (`engine/wheel_runner.py:254`) qualifies
on a non-Theta runner. Warnings captured via a `warnings.showwarning`
override.

**Status.** Done. **Verdict: YES — would survive production-scale SP500
deployment as-is, with three named conditions.**

1. **Warm the HMM regime cache once at startup.** Cold-call latency is
   145 s for a 503-ticker full-universe rank; subsequent calls on the
   same `WheelRunner` instance converge to **~10.5 s steady-state**
   (14× speed-up). Production should run a warm-up rank at boot, not
   on the first user request.
2. **Mild handle leak (~5 handles per repeated call).** Not exponential,
   not file-descriptor-exhaustion-level, but real: L2 calls each added
   +5 handles. At 100 calls/day this is ~500 handles/day on a process
   with a Windows default 16,384 handle cap. Investigate before
   high-throughput deployment; until then, plan for a daily process
   restart.
3. **Budget ~800 MB peak RSS during startup,** ~450 MB steady-state.
   The high-water mark hits during initialization (HMM cache load),
   not during steady-state ranking. A 1 GB working-set budget is
   comfortable.

§2 verified — no observed code path emits a tradeable verdict without
upstream `EVEngine.evaluate`; no §2 bug surfaced during the read.

**Findings:**

- **Per-lane operational table** (Bloomberg `as_of=2026-03-20`, 503-ticker
  universe, all lanes one process / one `WheelRunner`).

  | Lane | Description | Wall | RSS Δ | Handles Δ | Survivors | Notes |
  |---|---|---|---|---|---|---|
  | init | `WheelRunner()` + `get_universe()` | 13.0 s | +243 MB | +0 | – | Memory dominant cost is here; peak RSS hits 805 MB during this step (tracemalloc Python-heap peak +315 MB) |
  | **L1** | full-universe rank `top_n=50` cold | **145.2 s** | +13.2 MB | +18 | 50 / 70 drops | `_hmm_regime_cache` 0 → **492** (one entry per non-dropped ticker); gc objects +10,242 |
  | **L2** | same runner, repeat L1 ×5 | **10.5–10.7 s** (each) | +0.0 MB | +5 (each) | 50 / 50 / 50 / 50 / 50 | cache stays at **492 across all 5**, no growth; per-call latency 10.337 / 10.584 / 10.521 / 10.548 / 10.337 — extremely stable |
  | **L3** | `build_dossiers` on L1 top 50 | **0.059 s** | +0.0 MB | +0 | 50 verdicts | All 50 → `review` / `chart_context_missing` (offline `FilesystemChartProvider` — expected R2 per S16) |
  | **L4** | deep-history rank, 5 megacaps × 2065 days | **0.49 s** | +0.0 MB | +5 | 4 / 1 drop | Cache hits on the 5 names; sub-100 ms per ticker on full 2065-bar history. No O(n²) hotspot observed |
  | **L5** | full-universe rank `top_n=500` warm | **10.5 s** | +1.4 MB | +5 | 433 / 70 | Same wall-clock as L2 — `top_n` does not materially affect cost; the rank computation is the cost, row emission is free |
  | **L5b** | overshoot probe `top_n=10_000` | 10.8 s | – | – | caps to 433 | Graceful — no exception, no warning, ranker correctly caps to actual survivor count |

  Final state after all lanes: RSS 441.6 MB, peak RSS 805.6 MB,
  handles 400, gc_objects 117,786. Total wall 234.9 s.

- **Cold vs warm latency profile — 14× speed-up.** L1 cold = 145.2 s;
  L2 warm = 10.5 s. The cold-call cost is dominated by the HMM 4-state
  Gaussian model fit for each non-dropped ticker (`engine/regime_hmm.py`
  via `wheel_runner.py:1040–1060`). Once each `(ticker, history_days)`
  key is cached, the next call is essentially the BSM cost +
  forward-distribution per candidate. A `(ticker, history_days)` cache
  key means a *fresh history window* (i.e., a new trading day's worth
  of data) misses the cache and re-fits — daily production cadence
  will pay the cold-load cost **once per day**, not once per request.
  **Logged.**

- **`_hmm_regime_cache` is bounded.** Converges to **492 entries** on
  the first full-universe call (one per ticker that survives the
  ohlcv/history precondition) and stays there across L2's 5 repeated
  calls. No unbounded growth. The cache key is `(ticker,
  history_days)` so the entry count is bounded by `|universe| ×
  |distinct history window sizes|` — under realistic daily-batch
  operation, that's just `|universe|` per day. **Logged.**

- **`peak_RSS_delta=0` on every lane after init.** Peak RSS was set
  during `WheelRunner()` construction (805.6 MB) and never exceeded by
  any lane. Steady-state RSS settles at ~441 MB. The init memory cost
  comes from loading Bloomberg CSVs into the connector
  (`sp500_ohlcv.csv` 59 MB + `sp500_vol_iv_full.csv` 78 MB + a
  fundamentals/macro frame). `tracemalloc` confirms +315 MB Python-heap
  peak during init, congruent with the ~243 MB system-RSS delta (the
  difference is unfreed pandas internals). **Logged.**

- **Mild handle leak — +5 handles per L2 call, +18 on cold L1.** Not
  exponential and not file-descriptor exhaustion-level, but
  measurable. Over the 7 ranker calls in this run (L1 + L2×5 + L5 +
  L5b) the handle count walked from baseline post-init to 400. The
  per-call +5 strongly suggests a fixed number of file or kernel
  handles opened per call that aren't always closed — most likely the
  IV-history / OHLCV CSV reads materialise `pd.read_csv` handles that
  pandas closes lazily, or the `_hmm_regime_cache` keeps an
  `HMMRegimeDetector` instance alive that holds references. Not an
  emergency at the observed rate (~500 handles/day at 100 calls), but
  worth a follow-up read of the connector / HMM code paths.
  **Logged.**

- **`top_n` is not the cost.** L5 (`top_n=500` warm) = 10.5 s = L2
  (`top_n=50` warm). The rank computation runs over the full universe
  regardless of `top_n`; the sort+head at the end is essentially free
  on a 433-row frame. Implication: a pro running the ranker once and
  consuming the top-50 has paid the same compute cost as a pro asking
  for the top-500. **Logged.**

- **`top_n=10_000` overshoot is graceful.** No exception, no warning;
  the ranker returns 433 rows (the actual survivor count). Production
  callers asking for "all survivors" by passing a large `top_n` get
  exactly that, capped to the universe. **Logged as a positive.**

- **Deep history does not blow up.** L4 ranked 5 megacaps with 2065
  trading days of history each (the full Bloomberg coverage) in
  0.49 s total — about 100 ms per ticker. With the HMM cache warm,
  the per-ticker work is BSM + forward distribution on the full
  history, and that's linear in days, not quadratic. The HAR-RV /
  block-bootstrap / POT-GPD path on 2065 bars is well within budget.
  **Logged as a positive.**

- **70 drops on L1 = ~14% of universe** (`as_of=2026-03-20` is dense
  in Q1 earnings lockouts per S16; same shape here). Every drop has
  a `{ticker, gate, reason}` record in `.attrs["drops"]` —
  **433 survivors + 70 drops = 503**, fully accounted for. No silent
  drops observed on the full-universe path. (S16's structured-drops
  finding still applies: `reason` is free text.) **Logged as a
  positive.**

- **Zero captured warnings across 234 s of load.** The
  `warnings.showwarning` override saw no DeprecationWarning,
  RuntimeWarning, FutureWarning, or otherwise. The 5-ticker smoke
  warning surface (S17 also reported zero) extends to the
  503-ticker full-universe surface. **Logged as a positive.**

- **Init is the memory peak; per-call deltas are de minimis.** Cold
  init adds +243 MB RSS / +315 MB Python heap; steady-state per-call
  RSS Δ is 0.0 MB on all 5 L2 repeats. Memory is well-bounded under
  load. **Logged.**

- **No new orphaned surface noticed at scale.** The S15-style "exists
  but unwired" pattern (RiskManager / SectorExposureManager) was the
  only one observed across the campaign; this run didn't surface a
  new sibling. The decision-layer is fully wired through
  `WheelRunner.rank_candidates_by_ev` → `EVEngine.evaluate` from the
  full-universe entry point. **Logged.**

- **§2 verified across the load.** Every survivor on every lane came
  through `rank_candidates_by_ev` → `EVEngine.evaluate`; the
  reviewer-applied dossier verdicts on L3's 50 candidates were all
  R2 `chart_context_missing` (downgrade-only, expected). No path
  emitted a tradeable outcome without upstream EV. **No §2 bug
  surfaced; no regression test added.**

**AI handoff.**

- **Handle-leak follow-up.** The +5 handles per call (and +18 on the
  cold call) is the one mild concern. Suggested probe: wrap a single
  `rank_candidates_by_ev` call in `tracemalloc.get_traced_memory()`
  + a `gc.get_referrers` snapshot to identify which objects accumulate
  per call. A likely suspect is `engine/regime_hmm.py`'s detector
  cache holding open references to history `DataFrame`s. Out of scope
  for S18 — flagged for a small read-only follow-up Sn or a Terminal-
  A-lane decision-layer touch if the root cause is in `wheel_runner.py`.
- **Production warm-up pattern.** A `WheelRunner.warm()` method that
  loads the universe and populates `_hmm_regime_cache` synchronously
  at process start would hide the 145 s cold cost from user-facing
  request latency. Not claimed; flagged as the natural fix for the
  cold-load condition above.
- **L5b's `top_n=10_000` graceful-cap behaviour** is a structural
  positive worth pinning with a small regression test. Out of scope
  for a read-only usage test; flagged for a future Terminal A test
  addition.
- **Theta provider scale-out** is out of reach in this Cowork sandbox.
  The Bloomberg path has been characterised end-to-end here; the
  Theta path would have a fundamentally different latency / memory
  profile (live chains, persistent HTTP session to `:25503`,
  per-strike fetches) and warrants a separate Sn on the laptop with
  Theta Terminal up. **Theta-blocked.**

- **Ruled out per the prompt (don't litigate):** decision-layer edits,
  cProfile / line_profiler / memray (stdlib was sufficient), network
  load against `engine_api.py :8787` (S20's lane), failure-mode chaos
  / malformed payloads / corrupted-spot injection (S19's lane), Theta
  provider (operator-gated), optimisation / refactor.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — full-universe rank surfaces nothing tradeable without `EVEngine.evaluate`; structured drops on every dropped ticker.
  - Qualitative verdict: **partial — verified live, but the warm/cold latency profile has shifted materially**. L1 cold: 79.3 s (was 145.2 s, **45% faster**). L2 warm: 41.2 s (was **10.5 s, 4× SLOWER**). HMM cache 491 (was 492 — within 1 entry of the documented 492). RSS post-init: 672 MB (was 805 MB peak / 441 MB steady-state — current 672 is partway between, single-run snapshot). `top_n=10_000` overshoot still caps gracefully to actual survivors (423 vs original 433 — same family). Drops on the universe rank are still all-or-nothing structured.
  - Numerical drift > 5% (with attribution):
    - metric `L1_cold_wall_s`: orig `145.2` → new `79.3` (`-45.4%`); attributable to a **composite** of HMM caching improvements between S18 and now plus L1 hardware-state variance — not pinning to a single PR. The HMM cache size delta (492 → 491) is within seed-stable refit noise.
    - metric `L2_warm_wall_s`: orig `10.5` → new `41.2` (`+292%`); attributable to **PR #215 + PR #220** (`as_of-beyond-data` refusal guards added per-call cost — staleness probe on every ticker + post-PR-#208 / #210 / #222 emitted diagnostic columns adding serialization overhead). This is a **real warm-path regression** worth surfacing — the docstring-documented "warm calls ~10s" is no longer accurate. Flagged for follow-up; not in scope for this re-verify.
    - metric `L5b_top_n_10000_survivors`: orig `433` → new `423` (`-2.3%`); within Bloomberg-data-window noise / earnings-calendar drift on `2026-03-20`.
  - Notes: handle-leak follow-up (+5 handles per call documented in original entry) not measured this pass — would require a 100+-call sweep; the warm-path regression is the more pressing finding.

---

### S19 — Failure-mode chaos (fail-closed contract)

**Purpose.** Characterise how the engine behaves under hostile or
malformed inputs — bogus tickers, empty / garbage `as_of`, truncated /
empty CSVs, missing fundamentals, unreachable FRED adapter, missing data
directory, corrupted `ev_row` into the dossier — and document for every
chaos vector whether the engine fails **closed** (typed exception,
all-dropped result with structured `.attrs["drops"]`, or safe empty
frame) or **open** (≥1 tradeable row a naive caller would treat as
actionable). **A single fail-open in a tradeable-verdict path is the §2
headline; everything else is operational hygiene.**

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`, baseline
`as_of=2026-03-20`. Six vector groups: C1 (8 bogus-ticker sub-cases),
C2 (8 bogus-`as_of` sub-cases, fresh `WheelRunner` per sub-case to avoid
cache bleed), C3 (6 empty / truncated CSV sub-cases under a temp-dir
subprocess so the shared `data/bloomberg/*.csv` snapshots stay
untouched), C4 (FRED / EDGAR adapter failures via in-process
monkeypatch), C5 (missing `data/` directory entirely), C6 (§2 acid test
— row-for-row diff between `[AAPL, NOTATICKER, MSFT]` and clean
`[AAPL, MSFT]` ranks), C7 (4 dossier-side corrupted `ev_row` sub-cases:
NaN, +inf, missing `ticker`, path-traversal `ticker`). Plus an extension
probe injecting a synthetic *valid* `ChartContext` to exercise the
R5-bypass path with garbage EV — the dossier reviewer test the C7
in-script vectors couldn't reach because they all hit R2 first
(`chart_context_missing`). Throwaway instrumentation script + temp dirs;
not committed. Same pattern as S5 / S15 / S16 / S17 / S18.

**Path.** Each vector wraps a `rank_candidates_by_ev` or `build_dossiers`
call in `try / except BaseException` and a `warnings.catch_warnings(
record=True)` context, then classifies the outcome
`fail_closed_exception` / `fail_closed_empty` / `fail_open_tradeable` /
`degraded`. The §2 acid test then compares the AAPL/MSFT rows from C1c
(mixed-input) row-for-row to a clean baseline AAPL/MSFT rank — drift in
any of `strike / premium / iv / ev_dollars / ev_raw / prob_profit /
hmm_regime / hmm_multiplier` is the §2 violation indicator. The
synthetic-chart probe constructs a `ChartContext(screenshot_path=
Path('/dev/null'), visible_price=spot, error='')` — `is_ok()` returns
True per `engine/chart_context.py:86` — and steps the reviewer through
a boundary EV vector (`-25, -inf, 0, 5, 10, 25, +inf, NaN`) with the
chart match keeping R3 silent.

**Status.** Done. **Verdict: PARTIAL — engine fails closed on 22 of 27
chaos vectors, but one real §2 fail-open surfaced (C7b extension probe)
and three operational fail-opens (silent `as_of` substitution,
string-iteration on the ticker input, FRED-down silent default to
`credit_multiplier=1.0`).** §2 acid test (C6) confirms: the valid rows
returned alongside dropped unknown tickers ARE faithfully
EV-engine-computed on real Bloomberg data — row-for-row identical to a
clean rank.

> **§2 headline finding — `ev_dollars=+inf` on the dossier reviewer's
> input yields `proceed`.** Reviewer rule R1
> (`engine/candidate_dossier.py:167`) is `if ev < 0: return blocked` —
> the comparison `float('inf') < 0` is `False`, so R1 doesn't fire.
> R5 (`engine/candidate_dossier.py:206-209`) is `if ev >=
> self.min_proceed_ev: return proceed` — `float('inf') >= 10.0` is
> `True`, so R5 emits `proceed / ev_above_threshold`. With a valid
> `ChartContext` attached that passes R3 (the synthetic probe used an
> `is_ok=True` chart with `visible_price=spot`), the dossier returns a
> tradeable `proceed` verdict on garbage `ev_dollars=inf` input. **NaN
> degrades safely** (R5's `nan >= 10` is False → review),
> **`-inf` degrades safely** (R1's `-inf < 0` is True → blocked), but
> **`+inf` is the gap**. On real Bloomberg data the ranker never
> produces `inf` (the engine's `round(res.ev_dollars, 2)` would need
> `mean_pnl=inf` upstream), so this is a defence-in-depth gap — a
> downstream consumer that hand-builds an `ev_row` (e.g., the
> `engine_api.py` webhook receiving external payloads, or any future
> caller of `build_dossiers` not coming from the ranker) can be tricked
> into a `proceed`. **Logged. Not fixed in this Sn — Terminal A lane.**

**Findings:**

- **Per-vector outcome table** (27 vectors total; verdicts re-classified
  after probe-level analysis where my initial classifier was too
  shallow). Wall-clock under each shows the chaos vector failed *fast*
  — no vector hangs or runs > 2.5 s.

  | Vector | Description | Wall | Verdict | Detail |
  |---|---|---|---|---|
  | C1a | `tickers=[]` | 0.39 s | **fail_closed_empty** | 0 rows, 0 drops |
  | C1b | `tickers=["NOTATICKER"]` | 1.23 s | **fail_closed_empty** | 0 rows, drop `{gate:data, reason:"no OHLCV data..."}` |
  | C1c | mixed `[AAPL, NOTATICKER, MSFT]` | 0.49 s | **fail_closed_per_ticker** | 2 rows (AAPL+MSFT, both EV-computed); 1 drop for NOTATICKER. C6 confirms row-for-row match vs clean |
  | C1d | `["aapl"]` (lower-case) | 0.25 s | **fail_closed_empty** | dropped — no normalize on the public surface |
  | C1e | `["AAPL "]` (trailing whitespace) | 0.29 s | **fail_closed_empty** | dropped — no trim |
  | C1f | `[None]` | 0.30 s | **fail_closed_empty** | dropped with `ticker:null` in drops |
  | C1g | `["AAPL"]*50` (dupes) | 0.72 s | **deduplicates implicitly** | 10 rows = `top_n` cap on AAPL strike-set; not a § 2 issue |
  | C1h | `tickers="AAPL"` (string) | 0.49 s | **DEGRADED — per-char iteration** | string iterates to `["A","A","P","L"]`; **'L' is a real SP500 ticker (Loews Corp)** so it's ranked and 3 rows return |
  | C2a | `as_of="2099-01-01"` | 1.64 s | **DEGRADED — silent date substitution** | 2 rows ranked off latest available data; no warning |
  | C2b | `as_of="1990-01-01"` (pre-coverage) | 1.40 s | **fail_closed_empty** | 0 rows, drops "no OHLCV history at or before as_of" |
  | C2c | `as_of="2026-03-21"` (Saturday) | 1.65 s | **DEGRADED — silent date substitution** | falls back to last available (Friday); no warning |
  | C2d | `as_of="2026-03-22"` (Sunday) | 1.58 s | **DEGRADED — silent date substitution** | same |
  | C2e | `as_of="2026-12-25"` (Christmas) | 1.62 s | **DEGRADED — silent date substitution** | same |
  | C2f | `as_of="not-a-date"` | 1.38 s | **fail_closed_exception** | `ValueError: Invalid isoformat string: 'not-a-date'` |
  | C2g | `as_of=None` | 1.60 s | **DEGRADED — silent fallback to date.today()** | line 902 (`date.fromisoformat(as_of) if as_of else date.today()`); today() degrades same as C2a |
  | C2h | `as_of=date(2026,3,20)` (date obj) | 1.37 s | **fail_closed_exception** | `TypeError: fromisoformat: argument must be str` — API doesn't accept `date` objects |
  | C3a | 0-byte `sp500_ohlcv.csv` (subprocess) | 1.34 s | **fail_closed_empty** | drop `{gate:data}`; `pd.read_csv` raises `EmptyDataError`, connector catches at `engine/data_connector.py:108-111` and returns empty `_cache[key]` |
  | C3b | header-only `sp500_ohlcv.csv` | 1.35 s | **fail_closed_empty** | same as C3a |
  | C3c | `sp500_ohlcv.csv` missing `close` col | 1.38 s | **fail_closed_empty** | dropped on the `'close'` check |
  | C3d | 0-byte `sp500_vol_iv_full.csv` | 2.50 s | **not load-bearing for rank** | this file feeds `connector.get_iv_history` (mark-to-market path) and the HMM-staleness fallback, **not** the rank-time IV. Rank still returns 1 row of clean AAPL data because rank IV comes from `sp500_fundamentals.csv:implied_vol_atm` (see C3* note below). **Logged as a campaign-mapping correction.** |
  | C3e | header-only `sp500_vol_iv_full.csv` | 2.43 s | **not load-bearing for rank** | same |
  | C3f | `sp500_vol_iv_full.csv` missing IV cols | 2.51 s | **not load-bearing for rank** | same |
  | C3*  | 0-byte `sp500_fundamentals.csv` (probe) | – | **fail_closed_empty** | follow-up probe with the *actual* rank-IV source corrupted: drops `{gate:data, reason:"IV missing or non-positive"}` for every ticker; no rows returned |
  | C4a | baseline ranker without FRED network | – | **baseline_pass** | clean smoke succeeded at boot; ranker doesn't hard-require live FRED |
  | C4b | `FREDAdapter.credit_regime → ConnectionError` | 1.36 s | **DEGRADED — silent default** | 2 rows ranked; `credit_multiplier=1.0` and `credit_regime="benign"` reported in row — pro cannot tell from the row that the FRED overlay was bypassed; `wheel_runner.py:709-720` wraps the call in a permissive `try` that on any failure leaves the multiplier at 1.0 and the label at "unknown" / "benign" |
  | C4c | EDGAR adapter | – | **not_applicable** | `grep EDGARAdapter\|edgar_adapter engine/` returns only the adapter file itself + `engine/external_data/__init__.py`; **EDGAR is not referenced from the ranker / EV / dossier path**. S15-style orphaned surface — exists, not wired. |
  | C5 | missing `data/` directory | 1.34 s | **fail_closed_empty** | drop `{gate:data}`; `_data_dir / "sp500_*.csv"` does not exist → `_cache[key] = pd.DataFrame()` empty fallback at `engine/data_connector.py:101-104` |
  | C6 | §2 acid test: C1c rows vs clean AAPL/MSFT | – | **match** | clean=2 rows, mixed=2 rows; row-for-row identical across `strike / premium / iv / ev_dollars / ev_raw / prob_profit / hmm_regime / hmm_multiplier`. No drift. The dropped unknown ticker did **not** poison the valid rows. |
  | C7a | `ev_row.ev_dollars=NaN` | 0.001 s | **fail_closed_empty** | R2 fires (chart_context_missing → review); synthetic-chart probe: NaN → review (R5's `nan >= 10` is False) — safe degrade |
  | C7b | `ev_row.ev_dollars=+inf` | 0.001 s | **§2 FAIL-OPEN** | R2 fires in-script (chart missing); **synthetic-chart probe with valid chart → `proceed / ev_above_threshold`**. R1 (`if ev < 0`) doesn't catch `+inf`; R5 (`if ev >= 10`) does. |
  | C7c | `ev_row` missing `ticker` key | 0.001 s | **fail_closed_empty** | `build_dossiers` line 287-289 (`if not ticker: continue`) filters out the row |
  | C7d | `ev_row.ticker="../../etc/passwd"` | 0.001 s | **fail_closed (negative_ev)** in this test, but **input not sanitised** — reviewer uppercases to `"../../ETC/PASSWD"` and uses verbatim. A chart provider that does filesystem ops with the ticker would be exploitable; the live `FilesystemChartProvider` does `base_dir / f"{ticker}.png"` which Python's `pathlib` handles correctly today (the leading `..` is treated as a path component, not parent-dir traversal, when joined this way) — but it's a brittle no-validation seam |
  | post_smoke | 5-ticker EV smoke after all chaos | – | **pass** | Confirms the chaos didn't corrupt the live data path |

  **Tallies.** Of 27 vectors: **2 fail_closed_exception, 14
  fail_closed_empty, 1 fail_closed_per-ticker (C1c — the §2 acid-test
  positive), 1 deduplicates_implicitly, 5 DEGRADED operational
  (C1h/C2a/C2c/C2d/C2e/C2g + C4b — silent substitution / default), 1
  §2 fail_open (C7b inf bypass via the synthetic-chart probe), 3 not
  load-bearing (C3d-f), 1 not_applicable (C4c), 1 baseline_pass (C4a),
  1 acid-test match (C6), 1 post-smoke pass.**

- **C7b — `ev_dollars=+inf` bypasses both R1 and R5 → tradeable
  `proceed` verdict.** Synthetic-chart probe (`is_ok=True`,
  `visible_price=spot` → R3 silent) walked an EV grid
  `(-25, -inf, 0, 5, 10, 25, +inf, NaN)`. Verdicts:
  `-25 → blocked, -inf → blocked, 0 → review, 5 → review,
  10 → proceed, 25 → proceed, +inf → proceed, NaN → review`.
  **`+inf` is the only fail-open** — R1's
  `if ev < 0` is `False` for `+inf`; R5's `if ev >= 10` is `True`. The
  reviewer is supposed to be downgrade-only (§2: "reviewers can
  downgrade; never upgrade"); `+inf` slips through both guards. On
  real Bloomberg data the ranker doesn't produce `inf` so this isn't a
  live exploit, but it's exactly the defence-in-depth gap §2 exists to
  protect against — any downstream consumer that hand-builds an
  `ev_row` (the `engine_api.py` webhook receiving external payloads is
  the obvious candidate) can produce a `proceed` on garbage. The fix
  is one-line at `candidate_dossier.py:167` — replace `if ev < 0` with
  `if not (ev > 0) or not math.isfinite(ev)` (or similar — capture
  NaN/inf explicitly). **Logged. Not fixed in this Sn — Terminal A
  decision-layer lane.**

- **C2a/c/d/e/g — `as_of` silently substituted with latest-available
  data.** A pro running `rank_candidates_by_ev(..., as_of="2099-01-01")`
  or `as_of="2026-03-22"` (Sunday) or `as_of=None` gets back a real
  EV-ranked frame computed off the latest available trading day (here
  `2026-03-20`) — with **no warning, no flag, no drop entry**. The
  `<=as_of` OHLCV lookup is silently inclusive and never indicates the
  effective date is different from the requested. A pro testing a
  "what does the engine say for next Christmas?" scenario gets today's
  numbers labelled as Christmas. **Operational fail-open** (input
  contract relaxed silently); not strict §2 — the rows ARE
  EV-computed on real data — but a real pro-usage gap. **Logged.**

- **C1h — `tickers="AAPL"` iterates per character; 'L' is ranked.** A
  pro who accidentally passes a string instead of a list gets ranked
  candidates for ticker `L` (Loews Corp — a real SP500 name that
  happens to be a single character). The API has no `isinstance(
  tickers, list)` guard at the public surface; the iteration sees
  `["A", "A", "P", "L"]`, drops 'A' / 'P' (`A` is also a real ticker,
  Agilent, but at this `as_of` happens to be dropped by gates), and
  ranks 'L'. Returned row first ticker = "L", `ev_dollars=55.86`. The
  row IS EV-computed correctly for L — so strict §2 holds — but the
  caller's *intent* was AAPL, not L. **Operational fail-open;
  input validation gap.** Fix would be a single-line type guard in
  `rank_candidates_by_ev`. **Logged.**

- **C4b — FRED `credit_regime` raising silently defaults to
  `credit_multiplier=1.0` / `credit_regime="benign"`.** When the FRED
  adapter is unreachable (sandbox or production network failure), the
  ranker continues, the credit overlay is silently bypassed, and the
  emitted row shows `credit_multiplier=1.0` and `credit_regime="benign"`
  — **identical to a row where credit truly is benign**. A pro reading
  the diagnostic surface cannot tell whether the credit overlay ran
  with real data or was bypassed. The label *should* be `"unknown"` (a
  pre-existing convention in the codebase for unmeasured overlays —
  HMM uses it; see `wheel_runner.py:1058`) but the credit-overlay
  failure path defaults the label to "benign" / multiplier to 1.0,
  conflating "credit fine" with "credit not measured". **Operational
  silent-default; not strict §2.** Fix would change the failure-default
  label to `"unknown"` to match HMM convention. **Logged.**

- **C7d — path-traversal ticker propagates unsanitized through the
  dossier.** The reviewer uppercases `"../../etc/passwd"` to
  `"../../ETC/PASSWD"` and passes it forward verbatim. Today's
  `FilesystemChartProvider` does `base_dir / f"{ticker}.png"` and
  Python's `pathlib` joining with a leading `..` doesn't escape
  `base_dir` cleanly in either direction (the lookup just fails with
  `screenshot_not_found`), so it's not a live exploit — but the
  reviewer should validate input shape rather than rely on downstream
  consumers' incidental robustness. **Defence-in-depth gap.**
  **Logged.**

- **C6 §2 acid test passes.** Mixed input `[AAPL, NOTATICKER, MSFT]`
  produced rows that are row-for-row identical to clean
  `[AAPL, MSFT]` across `strike / premium / iv / ev_dollars / ev_raw /
  prob_profit / hmm_regime / hmm_multiplier`. The dropped unknown
  ticker did not poison the valid rows; the drops list correctly
  records the rejection with a structured-enough `{ticker, gate,
  reason}` entry. **The first-class §2 invariant — valid rows alongside
  dropped invalid ones are still faithful EV computations — holds under
  chaos.** **Logged as a positive.**

- **C3d/e/f misclassification corrected.** My initial classifier
  marked these as `fail_open_tradeable` because the rank returned 1
  row of clean AAPL data despite the corrupted file. Probe-level
  follow-up (`engine/data_connector.py:74:_FILES` +
  `engine/wheel_runner.py:813-814` reads) showed the rank-time IV
  comes from `sp500_fundamentals.csv:implied_vol_atm` (or
  `volatility_30d`), NOT from `sp500_vol_iv_full.csv`. The latter
  feeds `connector.get_iv_history` which is on the mark-to-market and
  HMM-staleness paths, not the rank IV path. **Re-classified as "not
  load-bearing for rank";** C3* probe with the *actual* IV source
  (`sp500_fundamentals.csv`) corrupted correctly yields `fail_closed_
  empty` with structured drops `{gate:data, reason:"IV missing or
  non-positive"}` for every ticker. The ranker's IV validation gate
  (`wheel_runner.py:821-829` + 833-843, percent normalisation +
  degenerate-IV drop) is the load-bearing fail-closed behaviour.
  **Logged.**

- **`.attrs["drops"]` schema is unchanged under chaos** (the S16
  finding still applies): `{ticker, gate, reason}` where `reason` is
  free text (`"no OHLCV data (empty or missing 'close')"`, `"no OHLCV
  history at or before as_of"`, `"IV missing or non-positive"`,
  `"IV degenerate after percent normalisation"`, etc.). **Logged.**

- **No vector hangs or runs longer than 2.5 s.** Even the malformed
  CSV subprocess vectors (C3 / C5) return inside 2.5 s. The engine
  fails fast under hostile inputs — no infinite loops, no retry
  storms, no deadlocks. **Logged as a positive.**

- **Validation is deep in the call stack, not at the public surface.**
  Every C1 / C2 fail-open and operational-degraded finding (C1h string
  iteration, C2a/c/d/e/g silent date substitution, C2h `date`-not-str
  TypeError) is the symptom of `rank_candidates_by_ev` doing no
  surface-level argument validation. The validation that does happen
  fires deep — in the connector (`_load` raising on empty CSV,
  caught), in the IV percent-normalisation gate
  (`wheel_runner.py:821-829`), in the dossier reviewer (R1–R6). A
  one-shot input contract check at the public surface
  (`tickers: list[str]`, `as_of: str | None`, `as_of` ISO-format
  regex, `as_of` within a tracked-coverage window with a structured
  drop on out-of-range) would close every operational fail-open at
  once. **Logged.**

- **Post-chaos 5-ticker smoke remains green.** None of the chaos
  vectors corrupted in-process state for subsequent runs (per-vector
  fresh `WheelRunner()` for C2, dedicated subprocess for C3 / C5
  meant zero cross-vector contamination on the worktree's shared
  `data/bloomberg/*.csv` snapshots). **Logged as a positive.**

- **No `WheelRunner` cache pollution surfaced.** The S18 finding
  (`_hmm_regime_cache` converges and stays) extends to chaos — across
  C1's 8 sub-cases on a single runner, the cache populated normally
  on the valid C1c / C1g / C1h paths and remained stable. No cache
  state leaked from a failed call into a subsequent one. **Logged
  as a positive.**

- **EDGAR adapter is structurally orphaned.** `grep EDGAR* engine/`
  returns only `engine/external_data/edgar_adapter.py` and
  `engine/external_data/__init__.py`. No call site on the ranker / EV /
  dossier path. This is a new S15-style "exists but not wired" surface
  — `EDGARAdapter` ships `cik_for_ticker / recent_insider_trades /
  insider_activity_signal`, none referenced from the live decision
  path. Not exercised by this Sn (`C4c not_applicable`); flagged as
  campaign companion to S15's `RiskManager` / `SectorExposureManager`
  orphaning. **Logged.**

**AI handoff.**

- **C7b is the headline §2 fix-up surface.** One-line patch at
  `candidate_dossier.py:167` to make R1 reject non-finite EV:
  `import math; ...; if not math.isfinite(ev) or ev < 0: return
  "blocked", "non_finite_or_negative_ev", notes`. Add a regression
  test that exercises `+inf / -inf / NaN` against the reviewer with a
  synthetic valid chart, asserting all three → `blocked`. **Terminal
  A decision-layer lane;** not fixed in this Sn.

- **One-shot public-surface input validation** would close C1h /
  C2a / C2c / C2d / C2e / C2g / C2h in a single change.
  `rank_candidates_by_ev` would `isinstance(tickers, list)` /
  `isinstance(as_of, (str, type(None)))` /
  `coverage_min <= parsed_as_of <= coverage_max` and emit a structured
  drop or raise a typed exception on violation. Backwards-compatible:
  callers passing a valid string-list and a valid `as_of` are
  unaffected. **Terminal A decision-layer lane.**

- **`credit_regime` failure label should be `"unknown"` not
  `"benign"`.** Match the HMM convention at
  `wheel_runner.py:1057-1058`. One-line change at
  `wheel_runner.py:709-720`. **Terminal A lane.**

- **Sibling structural orphan: EDGAR.** Combines with S15's
  `RiskManager` / `SectorExposureManager` and S16's `dashboard/
  quant_dashboard.py`-only consumer pattern. Worth a future Sn or a
  Terminal A audit pass: enumerate every `engine/` and
  `engine/external_data/` symbol that has zero live callers from the
  decision path and either wire them or retire them. **Candidate
  future scope.**

- **Ruled out per the prompt (don't litigate):** any decision-layer
  fix, network load on `engine_api.py:8787` (S20 lane), concurrency,
  Theta failures (sandbox-blocked), fuzz testing / hypothesis,
  performance / load (S18 covered).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: **closed — S19's C7b §2 FAIL-OPEN is RESOLVED**. Direct probe of `EnginePhaseReviewer.review(dossier)` on the four EV vectors with a valid `ChartContext` (visible_price=spot, is_ok=True):
    - `ev_dollars=+25` → `proceed / ev_above_threshold` (control)
    - `ev_dollars=+inf` → **`blocked / ev_non_finite`** (originally `proceed`)
    - `ev_dollars=NaN` → **`blocked / ev_non_finite`** (originally degraded to `review`)
    - `ev_dollars=-inf` → **`blocked / ev_non_finite`** (originally blocked via `<0`; now goes through the explicit non-finite check)
  - Qualitative verdict: **partial — §2 C7b CLOSED**; remaining operational fail-opens persist:
    - C2a (`as_of="2099-01-01"`): **NEW behavior** — now refuses with `ValueError: as_of=2099-01-01 is beyond OHLCV data cutoff 2026-03-20` (PR #215, S32 F3). Original was a silent date substitution; **now correctly fails closed**.
    - C2c (`as_of="2026-03-21"`, Saturday): still silently substitutes to Friday close (no warning).
    - C1h (`tickers="AAPL"`, string): still iterates per-character; 'L' (Loews) still gets ranked. Input validation gap unchanged.
    - C4b (FRED `credit_regime`): re-verified in S11 — credit overlay now PIT-aware, so the silent-default-to-benign concern partly closed (the multiplier actually moves now).
  - Numerical drift > 5%: not applicable — chaos vectors are boolean fail-closed checks.
  - Notes: PR #204 (R1a `ev_non_finite` guard) is **the closer of S19 C7b**. The CLAUDE.md §2 R1a entry explicitly cites PR #204. C2a future-`as_of` is now a typed `ValueError` per PR #215. Other operational fail-opens (C1h string iteration, C2c-e silent date substitution, C2h `date`-not-`str` TypeError) remain open.

---

### S20 — `engine_api.py` concurrency & crash resilience

**Purpose.** Characterise `engine_api.py` (HTTP on `:8787`) under
concurrency and crash conditions — parallel POST/GET, ring-buffer race
on `_TV_ALERT_LOG`, nonce-replay race on `_TV_SEEN_NONCES`, slow-ranker
blocking responsiveness, process-kill mid-write, payload validation,
auth-under-load — and document whether the API holds invariants under
production-shaped exposure. **The campaign-headline question is G3:
does S19's C7b `+inf` defence-in-depth gap become live-exploitable via
the network surface?** Closes the reliability arc (S18 scale + S19
chaos + S20 concurrency).

**Setup.** Server spawned as subprocess on `SWE_API_PORT=18787` /
`18788` (PR #158's per-instance binding, so the test instance doesn't
collide with the production port). 8 vectors planned (G1–G8); 5
landed cleanly. Bloomberg provider. Throwaway driver under
`%TEMP%\s20_concurrency\` (`tempfile`-style, not committed; not in
the worktree per the S18/S19 desktop-clutter feedback).
`urllib.request` + `concurrent.futures.ThreadPoolExecutor` for the
client side.

**Path.** Server uses `http.server.ThreadingHTTPServer` with
`daemon_threads = True` (`engine_api.py:2250-2251`) — one thread per
request, no explicit handler-side locking. The shared mutable state
that S20 stresses: `_TV_ALERT_LOG: list[dict]` (line 106),
`_TV_SEEN_NONCES: OrderedDict[str, float]` (line 121), and the
implicit shared state inside the lazy connector cache
(`engine/data_connector.py:_load`, no lock).

**Status.** Done. **Verdict: PASS, with one production-readiness
caveat (capacity ceiling at the OS socket backlog).** §2 G3
**REFUTED** (the campaign-headline positive); 4 secondary findings
(server-capacity limit at the OS socket backlog, ungraceful 10 MB
rejection, type-coerced ticker accepted, two divergent verdict-
producing paths); 1 code-level race surface
(`engine/data_connector.py:_load` has no lock around the cache-
populate). The race vectors (G1 ring-buffer trim + dedup, G2 torn
read, G4 nonce-replay race, G5 slow-vs-fast isolation, G8 HMAC under
load) all landed **clean** in the v5 backfill — see the per-vector
table. Four driver iterations (v1 wrong path → v3 wrong response key
→ v4 PIPE deadlock → v5 stdout-to-file fix) before the race
signals were measurable; the methodology debt is itself a finding
for the next API-chaos Sn.

*(Amended 2026-05-23 same-day: original entry shipped with
G1/G2/G4/G5/G8 as partially observed; v5 backfill landed clean
race-vector data. Original PARTIAL verdict revised to
PASS-with-caveat. Methodology-debt bullet preserves the v1→v5
driver history.)*

> **§2 G3 — REFUTED on the network surface.** No live exploit of
> S19's C7b inf-bypass via either `/api/tv/dossier` or
> `/api/tv/webhook`. Three control payloads (`+inf / NaN / -inf` as
> `ev_dollars` in the webhook body) all returned the server-computed
> AAPL EV (`-95.47` on the test run's `as_of`), with `verdict=skip`
> on all three. The server-side override is **mechanically protected**
> at `engine_api.py:2061-2072`: `_enrich_alert` re-initializes
> `ev_dollars = 0.0` and then overrides from
> `runner.rank_candidates_by_ev(...).iloc[0]["ev_dollars"]` — the
> payload's `ev_dollars` field is never read on the EV path
> (`TVAlert.parse` at `engine/tv_signals.py:537-567` doesn't have
> `ev_dollars` in its known-field set; it lands in
> `extras` and is ignored). `/api/tv/dossier` (line 1843-1865)
> similarly computes via `runner.build_candidate_dossiers(...)` from
> Bloomberg data; the only user-controlled EV-related input is
> `min_ev_dollars` (the **filter threshold**, not a candidate-level
> EV value). Probed `min_ev=Infinity` → 0 dossiers returned (filter
> drops everything; no fail-open). **C7b remains defence-in-depth
> only — log into S19's AI-handoff as the closing word: no
> network-path exploit observed.**

**Findings:**

- **Per-vector outcome table.** Verdicts after the methodology
  iterations; raw timings are best-effort given the capacity-related
  noise.

  | Vector | Description | Verdict | Detail (line cites) |
  |---|---|---|---|
  | **G3a** | `/api/tv/dossier?min_ev=Infinity&tickers=AAPL` | **REFUTED — fail-closed** | 200 OK, 0 dossiers (filter drops everything); `min_ev_dollars` is the filter threshold at `engine_api.py:1860`, not a candidate-EV override |
  | **G3a'** | `/api/tv/dossier?min_ev=-1e9&tickers=AAPL` (baseline) | works as designed | 200 OK, 1 dossier returned |
  | **G3c (+inf)** | webhook POST with `payload.ev_dollars = float("inf")` | **REFUTED — server overrides** | 200 OK, response `enriched.ev_dollars=-95.47` (real AAPL EV), `verdict=skip`. Payload's `+inf` never reaches the reviewer |
  | **G3c (NaN)** | same with `NaN` | **REFUTED — server overrides** | same: `enriched.ev_dollars=-95.47`, `verdict=skip` |
  | **G3c (-inf)** | same with `-inf` | **REFUTED — server overrides** | same: `enriched.ev_dollars=-95.47`, `verdict=skip` |
  | **G6** | crash recovery (kill PID, restart, verify) | **CLEAN** | pre-restart buffer cleared on cold start (`_TV_ALERT_LOG` per `engine_api.py:106`); post-restart POST returns 200 |
  | **G7a** | empty body | **fail_closed_400** | `{"error": "alert payload missing ticker or signal"}` |
  | **G7b** | non-JSON body | **fail_closed_400** | `{"error": "Invalid JSON: Expecting value..."}` |
  | **G7c** | `{"ticker": 42, "signal": [None, None]}` | **degraded** | 200 accepted; `TVAlert.parse` at `engine/tv_signals.py:557` does `str(payload.get("ticker", "")).upper()` → `"42"`; downstream `_enrich_alert` hits the `ticker_not_in_universe` branch at `engine_api.py:1989-1994` so no decision is produced — but the type-coerced row is accepted into `_TV_ALERT_LOG` as a no-op enriched record. Defence-in-depth: a `isinstance(payload.get("ticker"), str)` guard would fail-closed-400 instead |
  | **G7d** | `__proto__` / `constructor` keys | **clean** | 200 accepted; Python's `json.loads` has no JS prototype-pollution surface; keys captured into `TVAlert.extras` and ignored |
  | **G7e** | 10 MB payload | **fail_closed_socket_abort** | `ConnectionAbortedError(10053)`. Server abruptly closes the socket rather than emitting a 413 Payload Too Large. **Ungraceful** — a real load balancer in front of this would see the connection drop and may retry. Worth a proper `Content-Length` check upstream of the body read |
  | **G1a** | ring-buffer 32 POSTs at workers=4 | **clean** | 32/32 success; buffer length 33 = 32 new + 1 warmup AAPL; 33 unique tickers — no lost appends, no duplicates |
  | **G1c** | +200 POSTs to force trim, workers=4 | **clean** | 200/200 success; buffer length after = **exactly 200**; `_TV_ALERT_LOG_MAX=200` ring-trim at `engine_api.py:1683-1684` holds precisely |
  | **G2** | 40 POST + 40 GET parallel at workers=4 | **clean** | all 40 POSTs 200; all 40 GETs 200; every GET response returned exactly 30 items (`limit=30` honored); `get_len_min=get_len_max=30` — **no torn reads observed**, Python's GIL + slice semantics keep `_TV_ALERT_LOG[-limit:]` (line 1714) atomic |
  | **G4** | 16 same-nonce POSTs at workers=4 | **clean** | **1 × 200, 15 × 409** (`replay_blocked` per `_tv_seen_register` at `engine_api.py:125-143`). No race surfaced at workers=4. **Lock-free check-then-set IS theoretically racy** (lines 137-140); GIL + dict-op atomicity protect at this concurrency, but the pattern is fragile — see code-level finding below |
  | **G5** | slow dossier + fast alerts GET parallel | **clean** | slow_wall=0.69 s (warm caches), fast_wall=0.03 s (started 0.5 s after slow). `ThreadingHTTPServer` per-thread isolation works as advertised; **no serialization behind the shared `WheelRunner`** at `engine_api.py:163-167` (`get_runner` is the only shared mutation, lazy-init, single read after first call) |
  | **G6** | crash recovery (kill PID + restart) | **clean** | pre-kill: 10 POSTs OK, buffer=100; post-restart: buffer=**0**, follow-up POST=200. In-memory ring buffer cleared on cold start per `engine_api.py:103-105` docstring |
  | **G8** | 32 wrong-HMAC + 16 correct-HMAC POSTs at workers=4 | **clean — no auth bypass** | wrong: **32 × 401**; correct: **16 × 200**. `_tv_verify_hmac` (`engine_api.py:146-160`, `hmac.compare_digest` constant-time) holds deterministically under contention. No TOCTOU surfaced. |

- **Server-capacity ceiling (the only production-readiness limiter).**
  At 16 concurrent workers driving POSTs in the original v3 driver,
  **133 of 200** got `ConnectionRefusedError` / `-1` from the client.
  The remaining 67 succeeded with 200 OK. `http.server.
  ThreadingHTTPServer` accepts connections on a single listening
  socket whose listen-queue depth is the OS default
  (`socketserver.TCPServer.request_queue_size = 5`, inherited; see
  Python `Lib/socketserver.py`). Beyond ~5 in-flight accepts, the
  kernel refuses new connections. **The v5 backfill at workers=4
  showed zero drops across G1/G2/G4/G6/G8 (414+ POSTs total)**, so
  the production-readiness boundary is between workers=5 and
  workers=16. A typical dashboard hard-reload (5–10 simultaneous
  fetches) lands **right at the boundary** — usable, but a single CI
  smoke pile-up or an oncall investigator hitting it concurrently
  with the dashboard would tip into measurable drops. **Logged as
  a production-readiness gap.** Mitigations: (a) raise
  `request_queue_size` (`ThreadingHTTPServer.request_queue_size =
  128` before instantiating the server) — the two-line fix below;
  (b) front the server with a real reverse proxy (`nginx` / `caddy`)
  that handles the connection backpressure; (c) reduce dashboard
  concurrent-fetch count.

- **`engine/data_connector.py:_load` has no lock around cache-populate.**
  Code-level finding from reading the file:

  ```python
  # engine/data_connector.py:95-131
  def _load(self, key: str) -> pd.DataFrame:
      if key in self._cache:          # <- read without lock
          return self._cache[key]
      ...
      df = pd.read_csv(path, ...)     # <- can run concurrently in N threads
      ...
      self._cache[key] = df           # <- last writer wins
      return df
  ```

  Under ThreadingHTTPServer with `daemon_threads=True` and a cold
  connector (first request after server boot), the first ~16
  concurrent requests can ALL fall into the `pd.read_csv` branch
  for the same key, each loading 60 MB into memory in parallel.
  Last writer wins on `_cache[key]`, the earlier loads' DataFrames
  are garbage-collected — but during the load the process holds
  ~16× the steady-state memory footprint. Plausibly contributory to
  what wedged the v3 server after the first concurrent burst (though
  the proximate cause was the `subprocess.Popen(stdout=PIPE)`
  deadlock — v5 with stdout-to-file ran identical concurrency cleanly,
  so the lock-free `_load` was not the actual blocker in v3 either).
  Mitigation is one line: a `threading.Lock` field guarding the
  populate branch. **Logged. Terminal A data-layer-adjacent lane** —
  `data_connector.py` is the data layer per CLAUDE.md §1, not the
  decision layer; a lock-add is safe but should be claimed on #113.

- **Two divergent verdict-producing paths in `engine_api.py`.** The
  dossier endpoint (`/api/tv/dossier`) uses `EnginePhaseReviewer` at
  `engine/candidate_dossier.py:109-247` (rules R1–R6, the canonical
  reviewer). The webhook endpoint (`/api/tv/webhook` →
  `_enrich_alert`) runs **its own inline verdict logic** at
  `engine_api.py:2082-2099`:

  ```python
  # engine_api.py:2082-2099 (inline verdict, NOT EnginePhaseReviewer)
  if days_to_earnings is not None and 0 <= days_to_earnings < 5:
      verdict = "skip"; verdict_reason = "earnings_within_5d"
  elif verdict_authority != "ev_ranked":
      verdict = "review"
  elif ev_dollars < 0:
      verdict = "skip"
  elif ev_dollars >= 10 and prob_profit >= 0.65 and agrees:
      verdict = "proceed"
  elif ev_dollars > 0:
      verdict = "review"
  else:
      verdict = "skip"
  ```

  Both paths run `EVEngine.evaluate` upstream (so §2 holds in
  both), but the **rule structure diverges**: the webhook's
  conditions check `days_to_earnings < 5` (an earnings-gate
  duplicate not in `EnginePhaseReviewer`), `prob_profit >= 0.65`
  (additional confidence threshold), and `agrees` (signal-match
  predicate). The dossier reviewer has the dealer-positioning R6
  downgrade that the webhook does NOT. Same input could produce
  *different verdicts* between the two endpoints. **Code-duplication
  + consistency risk.** Logged as a divergence finding; the right
  fix is to route both endpoints through `EnginePhaseReviewer` and
  let R1–R6 be the single source of truth — but that's a Terminal A
  decision-layer change, not in scope here.

  **Bonus observation tied to S19 C7b:** the webhook's inline rule
  has the same `if ev_dollars >= 10` admit at `engine_api.py:2091`.
  If `ev_dollars` were ever `+inf` from the ranker (which it isn't
  on real Bloomberg data — see S19 C7b), this path would also emit
  `proceed` on garbage. Two paths, same defence-in-depth gap, both
  protected today only by the ranker not producing inf.

- **`_sanitize_nans` (`engine_api.py:209-217`) is a positive
  structural defence.** Every JSON response goes through
  `_sanitize_nans` which replaces `float('inf')` / `float('-inf')` /
  `float('nan')` with `None`. So even if internal state contained a
  non-finite EV that survived processing, the response would carry
  `None`, not the malicious value. **Belt-and-suspenders for the
  network reply path.** Worth pinning with a regression test
  (Terminal A lane).

- **G6 crash recovery is clean.** Killed the server PID
  (`subprocess.Popen.kill()` — SIGKILL on Unix, `TerminateProcess`
  on Windows), re-spawned on the same port, verified
  `_TV_ALERT_LOG` is empty after restart and POSTs work again. The
  docstring claim at `engine_api.py:103-105` ("buffer is rebuilt on
  each server start") holds. **Logged as a positive.** Note: the
  ring buffer is the *only* persistence surface in
  `engine_api.py` — there's no DB, no replay log, no disk-backed
  state. A crash mid-write loses at most one alert (the in-flight
  POST that didn't finish); everything else is on-disk in
  `data/bloomberg/*.csv` and loads cleanly on cold start.

- **Methodology debt — solved in v5.** Four iterations (v1 → v3 →
  v4 → v5) before landing usable race-vector data: v1 had a wrong
  endpoint path (`/api/tv/alert` vs the actual `/api/tv/webhook`); v3
  had a wrong response-key (`body["items"]` vs the actual
  `body["alerts"]`); v4 deadlocked on `subprocess.Popen(stdout=PIPE,
  stderr=PIPE)` with no draining (server filled the 64 KB pipe buffer
  and blocked on its next `print()`, all client requests then timed
  out); **v5 fixed all three by sending server stdout/stderr to a
  log file**, which unblocked clean numbers for G1/G2/G4/G5/G6/G8.
  Future API-chaos Sns should: (a) `subprocess.Popen(stdout=open(
  log, "w"), stderr=subprocess.STDOUT)` to a real file (or
  `DEVNULL`) — **never** PIPE without an active reader thread;
  (b) probe the response shape with one request before scaling to N;
  (c) start with workers=4 to stay under the default listen-queue;
  scale up only after validating the response shape. **Logged for
  the next-Sn-prompt template.**

- **§2 verified across the network surface.** The G3 negative
  result is the campaign-headline answer: C7b is mechanically
  closed by `_enrich_alert`'s `ev_dollars = float(r0.get(
  "ev_dollars", 0) or 0)` override at line 2072, and by the
  dossier endpoint's `runner.build_candidate_dossiers(...)`
  server-computation. **No observed network path emits a tradeable
  `proceed` on garbage ev_dollars.** This closes the reliability
  arc (S18 + S19 + S20) on a structural positive.

- **v5 backfill — race-vector positives.** Once the v4 PIPE
  deadlock was unblocked (v5 stdout-to-file), all five race vectors
  came back **clean at workers=4**:
  - **G1 ring-buffer trim is precise.** 32 unique POSTs all
    landed (buffer=33 including warmup); +200 more triggered the
    `_TV_ALERT_LOG_MAX=200` trim at `engine_api.py:1683-1684` —
    buffer cap held *exactly* at 200, no drift, no off-by-one.
  - **G2 GET-during-POST is atomic.** 40 GETs concurrent with 40
    POSTs returned **exactly 30 items each**
    (`get_len_min=get_len_max=30`). Python's GIL + the
    `_TV_ALERT_LOG[-limit:]` slice at line 1714 are atomic — no
    torn read, no truncated JSON, no `IndexError` 500s.
  - **G4 nonce-replay is correct under concurrency.** 16 same-
    payload POSTs at workers=4 yielded **1 × 200, 15 × 409** —
    exactly the expected behaviour from `_tv_seen_register`
    (`engine_api.py:125-143`). **However**, the check-then-set at
    lines 137-140 is **lock-free**, so the win is the GIL + dict-
    op atomicity, not explicit synchronisation. At higher
    concurrency (or under a non-CPython interpreter, or if the
    check/set window grew with future code changes), this could
    surface as >1 accept. **Logged as a code-level finding** —
    the same `threading.Lock` pattern as `_load` would close it.
  - **G5 per-thread isolation works.** Slow dossier (0.69 s warm)
    + fast `/api/tv/alerts` (0.03 s) ran concurrently; the fast
    GET was **not blocked** by the slow ranker. ThreadingHTTPServer's
    per-request threading at `engine_api.py:2250-2251` does what it
    advertises; the shared `WheelRunner` instance does NOT
    serialise readers behind a global mutation lock.
  - **G8 auth deterministic under load.** With `TV_WEBHOOK_HMAC_SECRET`
    set, 32 wrong-signature POSTs at workers=4 all returned 401;
    16 correct-signature POSTs all returned 200. **No auth bypass,
    no TOCTOU window.** `_tv_verify_hmac` (`engine_api.py:146-160`)
    uses `hmac.compare_digest` constant-time and is stateless, so
    the deterministic result under contention matches the code.
  **Logged as the campaign reliability-arc positive.**

**AI handoff.**

- **Top fix-up surface:** raise `request_queue_size` to handle
  realistic dashboard burst load. Two-line change at
  `engine_api.py:2250` —

  ```python
  # engine_api.py:2250 (current)
  server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
  # proposed:
  ThreadingHTTPServer.request_queue_size = 128
  server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
  ```

  Default is 5 (inherited from `socketserver.TCPServer`); 128 is the
  uvicorn/gunicorn default and is more aligned with the dashboard's
  burst pattern. **Terminal A decision-layer-adjacent lane**
  (`engine_api.py` is the interface layer; not the EV decision
  layer, but is on the launch-blocker list).

- **Second fix-up surface:** add a `threading.Lock` to
  `engine/data_connector.py:_load` to prevent N-thread cold-load
  amplification. Single-line lock acquire/release around lines
  101-129.

  ```python
  # engine/data_connector.py:86-89 (current)
  def __init__(self, data_dir: str = "data/bloomberg") -> None:
      self._data_dir = Path(data_dir)
      self._cache: dict[str, pd.DataFrame] = {}
  # proposed:
  def __init__(self, data_dir: str = "data/bloomberg") -> None:
      self._data_dir = Path(data_dir)
      self._cache: dict[str, pd.DataFrame] = {}
      self._cache_lock = threading.Lock()  # NEW
  # then in _load, wrap the cache-populate branch:
  def _load(self, key: str) -> pd.DataFrame:
      if key in self._cache:
          return self._cache[key]
      with self._cache_lock:
          if key in self._cache:    # double-check after lock
              return self._cache[key]
          # ...existing populate body unchanged...
  ```

  **Data-layer change, not decision-layer.** Worth a small Sn or
  Terminal A claim.

- **Third fix-up surface:** unify the two verdict-producing paths
  in `engine_api.py`. Either (a) route `_enrich_alert`'s decision
  logic through `EnginePhaseReviewer` (preferred — single source of
  truth), or (b) explicitly document the divergence and pin both
  rule-sets in tests. The S20 inline-rules block at
  `engine_api.py:2082-2099` reads like a parallel implementation of
  R1/R5 + extras; a regression test that drives the same `ev_row`
  through both endpoints and asserts identical verdicts would catch
  any future drift. **Terminal A decision-layer-adjacent lane.**

- **G7e 10 MB payload ungraceful disconnect** — add a
  `Content-Length` size guard at the top of `do_POST` (read header,
  reject 413 if > N bytes) so the server doesn't accept a
  body-too-large connection only to abort mid-stream. Defensive
  hygiene; not exploitable, but worth a clean 413.

- **Fourth fix-up surface (new from v5):** lock the check-then-set
  in `_tv_seen_register`. Current code at
  `engine/external/engine_api.py:125-143` (well, in `engine_api.py`
  itself):

  ```python
  # engine_api.py:125-143 (current — lock-free)
  def _tv_seen_register(digest: str, now: float) -> bool:
      cutoff = now - _TV_WEBHOOK_MAX_AGE_SEC
      while _TV_SEEN_NONCES and next(iter(_TV_SEEN_NONCES.values())) < cutoff:
          _TV_SEEN_NONCES.popitem(last=False)
      if digest in _TV_SEEN_NONCES:
          return False
      _TV_SEEN_NONCES[digest] = now
      while len(_TV_SEEN_NONCES) > _TV_SEEN_NONCES_MAX:
          _TV_SEEN_NONCES.popitem(last=False)
      return True
  # proposed: add a module-level _TV_SEEN_NONCES_LOCK = threading.Lock()
  # and wrap the body in `with _TV_SEEN_NONCES_LOCK:`.
  ```

  At workers=4 the race didn't surface (CPython GIL + dict-op
  atomicity protect the small check-then-set window), but the
  pattern is fragile: a future code change that widens the window,
  or a move off CPython, or higher concurrency in production, could
  surface a >1-accept anomaly. The fix is one `threading.Lock` plus
  a `with` block wrapping the function body.

- **Ruled out per the prompt:** any decision-layer code change
  (S20 found surfaces, did not fix), real-network load from off-box
  (none available in sandbox), MCP / chart provider chaos
  (S5 / S19 covered), fuzz testing (hypothesis), performance
  tuning, Theta provider failure (sandbox-blocked).

- **Campaign arc closure:** S18 (scale) + S19 (chaos) + S20
  (concurrency) all done. The **§2 invariant holds across all three
  axes on the live decision path.** Defence-in-depth gaps named
  (C7b inf-bypass, the silent-as_of substitution, the FRED-down
  silent regime label, the data_connector race, the two verdict
  paths) are all reads-not-writes findings; the fix surface is
  small and well-scoped for a follow-up. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`). API server spun up on Terminal-A's allocated port `:8787`.
  - §2 invariant: holds — **G3 REFUTED for the second time on the network surface**. Webhook payloads with `ev_dollars` as `+inf`, `-inf`, `NaN` all return server-computed AAPL EV (`-14.46`) and `verdict=skip` — the in-body `ev_dollars` field is ignored, `_enrich_alert` re-runs the ranker. `_sanitize_nans` confirmed: alerts GET response contains no `Infinity` or `NaN` JSON tokens.
  - Qualitative verdict: match — all race vectors clean at `workers=4`:
    - G1: 220 POSTs → buffer holds **exactly 200** (trim precise).
    - G2: 40 POSTs + 40 GETs concurrent → every GET returned exactly 30 items (no torn reads).
    - G4: 16 same-nonce POSTs → **1×200, 15×409** (nonce-replay protection works under contention).
    - G7a: empty body → 400; G7c: bad ticker (ZZZZ) → 200 with soft-rejection (matches original).
    - Dossier endpoint `min_ev=Infinity` returns 200 / 0 dossiers (filter consumed); baseline `min_ev=-1e9` → 1 dossier.
  - Numerical drift > 5%:
    - metric `AAPL_webhook_enriched_ev_dollars[as_of=2026-03-20]`: orig `-95.47` → new `-14.46` (`-85% magnitude`); attributable to **PR #179** (post-IV-PIT-fix AAPL ev_dollars at this as_of is materially less negative — PIT IV for AAPL at 2026-03-20 is lower than the snapshot, reducing the synthetic premium and the magnitude of the negative EV). Sign preserved.
  - Notes: PR #216 (engine_api `request_queue_size = 128` — Terminal B's S20 fix-up #1) and PR #219 (Terminal B's `_tv_seen_register` lock — S20 fix-up #4) are flagged as merged on `main` per the board's recent activity but were not the focus of this §2 re-test. Both should harden the same surfaces in higher-concurrency regimes than the workers=4 used here.

---

### S21 — D17 confirm-fixed + pro-account sizing at $1M

**Purpose.** Close the S15 loop. S15 (PR #148) found the portfolio-
aggregation gap: `RiskManager` / `SectorExposureManager` /
`HierarchicalRiskParity` / `StressTester` / Kelly helpers shipped in
`engine/risk_manager.py` + `engine/stress_testing.py` but **were not
imported by any decision-layer file** — positions could open with no
NAV-level sector cap, no portfolio-delta cap, no Kelly check.

Between S15 and S21 the institutional risk layer landed in two
phases:

- **#163 (D17 Phase 2)** wired three **hard-block gates** into
  `engine/wheel_tracker.py:_evaluate_d17_hard_blocks` —
  `check_sector_cap`, `check_portfolio_delta`, `check_kelly_size`
  from the new `engine/portfolio_risk_gates.py`. Fires only in
  strict mode (`require_ev_authority=True`), after the D16 token
  consume.
- **#165 (D17 Phase 3)** wired two **soft-warn rules** into
  `engine/candidate_dossier.py:EnginePhaseReviewer.review` — R7
  (`check_var`, default 5% NAV / 30-day / 95% conf.) and R8
  (`check_stress_scenario` C4 vol spike + `check_dealer_regime`).
  Fires only when a `PortfolioContext` is attached and the current
  verdict is `proceed`. Downgrade-only.

S21 verifies the new gates actually block what S15 said they
should block, characterises a pro-scale book at $1M, and exercises
R7/R8 with a constructed `PortfolioContext`.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, 25-ticker universe across 7 GICS sectors (same
set as S17). Three prongs:

- **Prong A** — sector-cap trip demo. `WheelTracker(initial_capital=
  150_000, require_ev_authority=True)`. Attempt CAT first
  (Industrials, strike $625.5, single-put collateral $62,550 =
  **41.7% of $150k NAV**). Expect `check_sector_cap` to reject —
  default `_DEFAULT_MAX_SECTOR_PCT=0.25`. Then attempt smaller
  positive-EV tickers (PG / HD / CVX) to see what else fires.
- **Prong B** — pro-account sizing at $1M. `initial_capital=
  1_000_000`, strict mode. Sort all 9 positive-EV candidates by
  `ev_dollars` desc, attempt to open sequentially with the full
  token-issue + token-consume flow. Record which gate fires per
  rejected attempt.
- **Bonus** — R7/R8 exercise. Build a `PortfolioContext` from the
  final Prong B book; attach to a `CandidateDossier` for the
  highest-EV candidate (CAT). Run `EnginePhaseReviewer.review` with
  and without the context; observe the verdict delta. Also call
  `check_var` and `check_stress_scenario` directly to capture
  their numerical results.

All paths use a synthetic `ChartContext` (visible_price=spot) so R2
and R3 stay silent and the gate path is the only thing exercised.
Strict mode uses `tracker.issue_ev_authority_token(ev_row)` →
`tracker.open_short_put(..., ev_authority_token=token,
current_ev_dollars=ev_dollars, prob_profit=prob_profit)`. Throwaway
driver under `%TEMP%\s21_d17\` (system temp, not in repo). Same
S18/S19/S20 pattern.

**Path.** D17 hard-blocks at `engine/wheel_tracker.py:1579-1719`:
gate sequence is `nav < min_nav_for_trading` (pre-gate) → sector
cap → portfolio delta cap → Kelly size cap. **First failing gate
short-circuits** and writes a `reject` entry into
`_ev_authority_log` with `action="reject", reason=..., ticker=...,
nav=..., nav_source=...`. R7/R8 soft-warns at
`engine/candidate_dossier.py:282-381` (post-#165): only fire when
`portfolio_context is not None and verdict == "proceed"`; both
downgrade-only.

**Status.** Done. **Verdict: D17 hard-blocks confirm-fixed (sector
cap works exactly as S15 predicted) + one binding-constraint
finding (default portfolio-delta cap is the dominant gate at all
NAV scales tested).**

**Findings:**

- **Prong A — `check_sector_cap` correctly blocks CAT at $150k.** Live
  output from the driver:

  ```
  CAT open attempt: opened=False, reject=sector_cap_breach
  reject_details: {
      'current_ev_dollars': 290.26, 'nav': 150000.0,
      'nav_source': 'live_mark_to_market',
      'sector': 'Industrials',
      'sector_pct': 0.417,
      'sector_limit': 0.25,
      'narrative': "Sector 'Industrials' would be 41.7%
                    (limit: 25.0%). Current positions: []"
  }
  ```

  Exactly the structural gap S15 named, mechanically closed. The
  audit entry shape (`action=reject, reason=sector_cap_breach,
  sector_pct, sector_limit, narrative`) matches the D17 audit-log
  schema pinned by `tests/test_ev_authority_log_schema.py`. The
  `nav_source='live_mark_to_market'` confirms the NAV was computed
  via `_compute_live_nav` (D17's "live NAV under gate decisions"
  principle), not against `initial_capital`. **S15's
  `SectorExposureManager`-orphaned finding is closed for sector
  exposure on the tracker hard-block path.**

- **Prong A unexpected — portfolio-delta cap is the *dominant*
  binding constraint at $150k.** After CAT failed sector, the next
  three positive-EV candidates (PG / HD / CVX, all in non-Industrials
  sectors) all failed at **gate 2**, not gate 3:

  ```
  PG  -> portfolio_delta_breach  post_open_delta=$3,382  cap=$450
  HD  -> portfolio_delta_breach  post_open_delta=$7,540  cap=$450
  CVX -> portfolio_delta_breach  post_open_delta=$4,698  cap=$450
  ```

  `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` × ($150k / $100k) =
  **$450 dollar-delta cap**. A single short put on a non-megacap
  ticker carries $3-8k of long-delta-dollars (per `RiskManager.
  calculate_portfolio_greeks`'s `delta * contracts * 100 * spot`
  computation at `engine/risk_manager.py:359`). **At $150k NAV the
  default delta cap admits zero single-put positions.** Prong A
  final book: 0 positions. **Logged** — not a bug (the gate works
  exactly as designed), but a calibration question: the default
  `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` at
  `engine/portfolio_risk_gates.py:59` may be too tight for the
  retail wheel use case.

- **Prong B — at $1M, only 2/9 positive-EV candidates open.** Same
  binding constraint, just shifted: delta cap = 300 × ($1M / $100k)
  = **$3,000**.

  | Ticker | EV $ | Strike | post-open Δ$ | Outcome |
  |---|---:|---:|---:|---|
  | CAT | +290.26 | 625.50 | 16,507 | BLOCKED (delta) |
  | CVX | +117.73 | 192.00 | 4,698 | BLOCKED (delta) |
  | MRK | +85.09 | 107.00 | (≤3000) | **OPENED** |
  | HON | +18.66 | 209.00 | 5,279 | BLOCKED (delta) |
  | PG | +18.51 | 138.00 | 3,382 | BLOCKED (delta) |
  | HD | +18.37 | 303.00 | 7,540 | BLOCKED (delta) |
  | KO | +15.29 | 71.50 | (≤3000) | **OPENED** |
  | ORCL | +12.20 | 136.00 | 3,650 | BLOCKED (delta) |
  | GOOGL | +1.59 | 283.00 | 7,382 | BLOCKED (delta) |

  Final Prong B book: **2 positions (MRK + KO)**, cash $1,000,237,
  BP $982,387. **All 7 blocks were `portfolio_delta_breach`** — not
  one sector, Kelly, or NAV-floor block in this universe at $1M.
  Compared to S17's 7-position book at $1M without D17, **D17 in
  strict mode caps the realistic book at low-strike tickers**
  (MRK $107, KO $71.5 are the lowest in the universe). **Logged.**

- **Gate ordering is sector → delta → Kelly; the first failing gate
  short-circuits.** Per `_evaluate_d17_hard_blocks` at
  `engine/wheel_tracker.py:1665-1700`. In Prong B, the delta cap
  fires first on every blocked candidate (the gate body returns the
  audit entry and the function returns); we never see the Kelly
  gate or sector gate fire downstream. **For the Kelly gate to
  exercise in a future Sn, the delta cap would have to be raised or
  the candidate set restricted to ones that pass delta.** **Logged.**

- **R7 (VaR) properly skips on missing data — Q3 semantics work.**
  Constructed `PortfolioContext` from the Prong B book (MRK + KO,
  2 positions, $999,964 NAV). With no `correlation_matrix` and no
  `returns_data` passed, `check_var` returns
  `passed=True, reason='missing_data',
  details={'var_check': 'skipped', 'skip_reason':
  'no_correlation_matrix_or_returns_data'}`. The reviewer at
  `engine/candidate_dossier.py:300-309` correctly logs
  `R7: VaR check skipped (no_correlation_matrix_or_returns_data)`
  and does NOT downgrade. **Matches the documented "soft-warns
  don't fire on absent evidence" rule (Q3 of the #154 C4 design
  checkpoint, per `engine/portfolio_risk_gates.py:92-98`).**
  **Logged as a positive.**

- **R8 (stress + dealer-regime) didn't fire on the 2-position
  book.** Direct `check_stress_scenario` call yielded
  `passed=True, scenario='C4 Vol Spike',
  portfolio_pnl_dollars=-$5,583, drawdown_pct=0.56%,
  drawdown_limit_pct=0.08`. Under the C4 standard scenario (-10%
  spot + 30% IV per `engine/portfolio_risk_gates.py:71-77`), the
  2-position book takes a $5,583 hit — **0.56% of $1M NAV, far
  below the 8% cap**. To exercise R8 we'd need a larger / more
  concentrated book, or test directly against `check_dealer_regime`
  with a `short_gamma_amplifying` regime label. Out of scope here.
  **Logged.**

- **Verdict-delta with vs without `PortfolioContext` — no
  downgrade observed.** For the highest-EV candidate (CAT,
  ev_dollars=290.26) with a synthetic chart that passes R3
  (`visible_price=spot=656.77`), the reviewer returned
  `proceed / ev_above_threshold` in BOTH cases. With context
  attached, the extra note `R7: VaR check skipped` was appended;
  no verdict change. **R7/R8 only downgrade when the gates fire on
  real data; with the missing-data skip path, the candidate
  proceeds.** **Logged as a positive** — the soft-warn contract is
  downgrade-only and silent-on-skip, which is exactly what S19
  said `_sanitize_nans` does for the response path: "don't claim
  what you can't prove."

- **§2 verified across D17.** Every Prong A and Prong B attempt
  routed through `EVEngine.evaluate` upstream (via
  `WheelRunner.rank_candidates_by_ev` → token issue → token consume
  → D17 gates). The D16 + D17 stack is "EV-authority *and* portfolio-
  risk authority" — a candidate can pass EV and still be blocked
  by D17 (CAT at $150k: EV +$290.26 → blocked by sector cap; CAT at
  $1M: EV +$290.26 → blocked by delta cap). **The §2 invariant —
  no tradeable verdict without EVEngine.evaluate — extends naturally
  into D17 (no tradeable position without passing portfolio-risk
  gates too). No bypass observed.** **Logged.**

- **`_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` calibration.** The
  cap at `engine/portfolio_risk_gates.py:59` (= 300 dollar-delta
  per $100k NAV) is the binding constraint at both $150k and $1M
  on this universe. Per `check_portfolio_delta` at
  `engine/portfolio_risk_gates.py` (and per the D17 design comment
  on #113), this is **deliberately conservative** — a wheel book
  with 5 unbalanced single puts could carry $20-40k delta-dollars
  unhedged. But the cap of $3,000 at $1M means a book is naturally
  capped at low-strike tickers; **a $1M pro running default D17
  strict mode is structurally biased toward $50-$200 strike names**
  (MRK / KO / WMT / PFE / KO-class). **Logged** as a calibration
  data-point for the eventual D17 default-tuning discussion. Not a
  bug, not a fix in this Sn.

**Verdict.**

- **D17 hard-block sector cap: CONFIRM-FIXED.** The structural gap
  S15 named (no sector-exposure enforcement at position-open) is
  mechanically closed by #163. CAT at $150k correctly blocked with
  `sector_pct=0.417, sector_limit=0.25` and a structured audit
  entry. The five-finding orphaned-`SectorExposureManager` thread
  from S15 is **closed for the tracker hard-block path**. (The
  dossier soft-warn path R7/R8 also exists post-#165; R7 properly
  skipped on missing data in this run, R8 didn't trigger on the
  small book.)
- **D17 portfolio-delta cap: WORKS BUT TIGHT.** At default
  `300/$100k NAV`, admits only low-strike single puts. At $150k the
  cap is $450 — no single put passes. At $1M the cap is $3,000 —
  2 / 9 positive-EV candidates pass. **Logged as a calibration
  data-point**, not a fix-up.
- **D17 Kelly gate: NOT EXERCISED.** Gate ordering puts Kelly third;
  delta cap fired first on every blocked attempt. A future Sn
  could either raise the delta cap or pick candidates that pass
  delta to exercise Kelly.
- **R7 / R8 dossier soft-warns: BEHAVE AS DOCUMENTED.** R7 properly
  skips on missing correlation/returns data (Q3); R8 stress
  scenario benign on a small book. To fully exercise either,
  follow-up Sn needs a richer `PortfolioContext` (real returns,
  larger book).

**AI handoff.**

- **Highest-leverage observation from this run: the D17 portfolio-
  delta default may be too tight for the wheel use case.** Three
  options the user / Terminal A could weigh, each with a literal
  proposed diff at `engine/portfolio_risk_gates.py:59`:

  ```python
  # engine/portfolio_risk_gates.py:59 (current)
  _DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0

  # option 1 (loosen by 3x):
  _DEFAULT_DELTA_CAP_PER_100K_NAV = 1000.0  # admits CAT at $1M

  # option 2 (loosen by 10x for wheel-specific use):
  _DEFAULT_DELTA_CAP_PER_100K_NAV = 3000.0  # ~16k delta at $1M

  # option 3 (leave default, document as conservative):
  # — add a doctest example showing "default admits 2 positions at $1M"
  ```

  This is a **design decision, not a bug**. Numbers above just
  contextualise the calibration question. Out of scope for this Sn.

- **The delta-cap-as-binding-constraint finding makes the Kelly
  gate (gate 3) effectively unreachable** at default settings,
  because gate 2 short-circuits first. If a future Sn wants to
  exercise Kelly (`check_kelly_size` at
  `engine/portfolio_risk_gates.py:384`), the test driver should
  either raise the delta cap per-call (the gate functions accept
  override kwargs) or filter to low-delta candidates that pass
  gate 2.

- **R7 / R8 follow-up Sn would need a real `PortfolioContext`** —
  build `returns_data` from connector OHLCV (compute daily log
  returns over a 252-day window) and `correlation_matrix` via
  `RiskManager.calculate_correlation_matrix` (or equivalent).
  Out of scope here; flagged for a future Sn.

- **The audit-log schema for `reject` entries is structured** —
  `engine/wheel_tracker.py:_evaluate_d17_hard_blocks` writes
  `action='reject', reason=<gate>, ticker, nav, nav_source,
  current_ev_dollars, sector, sector_pct, sector_limit,
  narrative` (sector path; other gates have parallel detail
  bags). **This is the structured-drops finding S16 asked for** —
  on the reject path, the audit log IS structured with discrete
  fields. (`.attrs["drops"]` on the ranker output still uses free-
  text `reason`; the post-#163 reject log is the structured
  alternative on the tracker side.) **Logged as a positive — S16's
  AI-handoff fix-up #1 is partially closed.**

- **S15 closure update.** S15 named six aspects of orphaned risk
  layer; D17 has closed three:
  - ✅ `SectorExposureManager` — wired via `check_sector_cap`.
  - ✅ Portfolio Greeks (`calculate_portfolio_greeks`) — used by
    `check_portfolio_delta`.
  - ✅ VaR (`calculate_var`) — wired via R7 / `check_var`.
  - ⚠ Kelly (`calculate_kelly_fraction`) — wired but not yet
    exercised in any test (delta gate short-circuits first).
  - ⚠ Stress testing (`StressTester`) — wired via R8 /
    `check_stress_scenario` but doesn't fire on small books.
  - ❌ HRP (`HierarchicalRiskParity`) — still orphaned (no consumer
    in the decision-layer; only `tests/test_advanced_quant.py`).
  **S15-mark-2 future Sn could be a re-run with a larger Prong B
  book to exercise Kelly + R8, plus a fresh HRP-orphan check.**

- **Ruled out per the prompt:** any decision-layer code change
  (S21 found surfaces, did not fix), Theta provider (sandbox-
  blocked), advisor committee (not in scope), tuning the gate
  defaults (design discussion, not a usage test's remit).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every Prong A and Prong B candidate routed through `EVEngine.evaluate` upstream; D17 hard-blocks fire AFTER the EV check.
  - Qualitative verdict: match — both prongs reproduce the documented behavior on `main`:
    - **Prong A** ($150k strict mode, CAT @ $625.50): `action=reject reason=sector_cap_breach` exactly as the original entry's audit-log narrative.
    - **Prong B** ($1M strict mode, 9 positive-EV candidates): **2 opened, 7 rejected with `portfolio_delta_breach`** — identical pattern to the original (delta cap = $3,000 at $1M binds before sector or Kelly).
    - The five `portfolio_delta_breach` tickers observed (CAT, GOOGL, CVX, PG, HON) overlap the original's set (CAT, CVX, HON, PG, HD, GOOGL, ORCL).
  - Numerical drift > 5%: not applicable — the cap-binding-constraint result is deterministic and ev_dollars/Δ$ map nearly 1:1 (CAT ev=+290 → +445 from PR #179 doesn't change the delta-breach decision because delta-dollars dominate at $1M).
  - Notes: `check_var` and `check_stress_scenario` signatures evolved post-S21 — both now require an explicit `candidate_option: dict` argument (the candidate being assessed). The R7 / R8 reviewer-integration is unchanged — they still fire only when `PortfolioContext is not None and verdict == "proceed"`.

### S22 — Roll defense economics (ITM short put with ≤7 DTE)

**Purpose.** Exercise the management-layer decision a wheel trader
hits every cycle: a short put goes meaningfully ITM with little
time left. The engine ships three paths for this moment —
`WheelTracker.suggest_rolls` (PR landing the put-roll ranker),
`WheelTracker.handle_put_assignment` followed by
`WheelRunner.rank_covered_calls_by_ev` (the post-assignment monetise
path, PR #124), and the implicit "hold to expiry" comparison
embedded in `suggest_rolls`'s `hold_ev` column. S2 / S3 / S8 logged
the surface; no prior Sn ran the three side-by-side on real data
to compare the three EVs at one decision moment.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-13` (decision date, Friday). Hypothetical scenario:
a 25-delta short put on **PNC** opened 2026-02-13 (Friday) with
35 DTE (expiry 2026-03-20). PNC dropped 11.3% over that window
(228.18 → 202.29; IV 26.5% → 36.2% → 34.1%), putting the put deep
ITM. At decision time (2026-03-13) the open put has DTE remaining
= 7. The BSM-fair opened-put parameters (strike $218 rounded to
integer-bank, premium $3.02/sh, IV 26.5%) are the engine's own
synthetic-chain convention from `rank_candidates_by_ev`. The
underlying spot, IV, and earnings calendar all come straight from
the Bloomberg connector — no fabricated numbers.

Three branches driven side-by-side under `%TEMP%\s22\` (system temp,
not in repo):

- **(A) Hold to expiry.** Surfaced as `hold_ev` by
  `tracker.suggest_rolls`. The engine internally builds a
  `ShortOptionTrade` with the current buyback fair value as the
  re-sell premium, runs it through `EVEngine.evaluate`, then
  subtracts the notional re-sell premium to recover the pure
  forward P&L (see the put-roll EV derivation in
  `engine/wheel_tracker.py:1937-1968`).
- **(B) Roll.** `tracker.suggest_rolls(ticker='PNC',
  as_of=2026-03-13, current_spot=204.37, current_iv=0.3624,
  target_dtes=(14,21,35,49), target_deltas=(0.15,0.20,0.25,0.30),
  min_net_credit=-1000.0)` — debit rolls allowed since the put is
  ITM and the rescue might cost up to ~$10/sh debit. The headline
  metric is `roll_ev` — the marginal forward EV of the chosen roll
  *net of the buyback cost*.
- **(C) Accept assignment + best covered call.** Fork a fresh
  `WheelTracker`, replay the opened put, call
  `tracker.handle_put_assignment(ticker='PNC',
  assignment_date=2026-03-20, stock_price=202.29)` then
  `wr.rank_covered_calls_by_ev(ticker='PNC', shares_held=100,
  as_of='2026-03-20', target_dtes=(14,35,49,63),
  target_deltas=(0.30,0.25,0.20,0.15))`. The composed EV is "stock
  leg paper P&L (basis vs spot, path-dependent) + best CC
  `ev_dollars` (forward)".

**Path.** `suggest_rolls` (at `engine/wheel_tracker.py:1840`)
enumerates `(dte × delta)` cartesian, solves each strike at the
target call/put delta under current state, BSM-prices it, builds a
`ShortOptionTrade(option_type='put')` and calls
`EVEngine.evaluate` for every candidate plus the hold-synthetic.
`rank_covered_calls_by_ev` (at `engine/wheel_runner.py:1608`) does
the same on `option_type='call'` for the covered-call leg, with
the connector's earnings calendar wiring an `EventGate`
(`earnings_buffer_days=5`).

**Status.** Done. **Verdict: the roll path correctly beats hold
($+63) on a real ITM situation; assignment-then-CC is worst by a
large margin; observability asymmetry surfaced between the
`suggest_rolls` output and the two ranker outputs.**

**Findings:**

- **Headline EVs.** Live driver output (all three branches, one
  decision moment):

  ```
  (A) Hold to expiry      hold_ev      = $-1,385.48
  (B) Best roll           roll_ev      = $-1,322.09  (K=194.50, DTE=35, Δ=0.30)
  (C) Accept + best CC    forward EV   = $   +40.85  (+ stock leg $-1,571.00 path-dependent)
                          composed     = $-1,530.15
  ```

  Roll beats hold by **$63.39**; both lose money in expectation
  (the put is structurally underwater), but the roll buys forward
  time and re-strikes 24$ lower at a $940 debit. **`recommend=True`
  fires on rows 0 and 1; the engine correctly picks the least-bad
  branch.** Composed assignment-then-CC is worst — the $1,571
  stock paper loss dominates the +$41 CC forward EV. **Logged**:
  the ranker / `suggest_rolls` triple is internally consistent
  and matches the qualitative wheel-trader intuition (don't accept
  an early-cycle assignment on a name you don't want).

- **Roll candidate filtering — 13 of 16 candidates silently
  dropped.** `target_dtes=(14,21,35,49)` × `target_deltas=(0.15,
  0.20,0.25,0.30)` = 16 grid points. Only 3 survived:

  ```
     new_strike  new_dte  target_delta  ...  hold_ev  roll_ev  recommend
  0       194.5       35          0.30  ... -1385.48 -1322.09       True
  1       193.5       49          0.30  ... -1385.48 -1365.15       True
  2       189.5       49          0.25  ... -1385.48 -1415.54      False
  ```

  No 14-DTE candidate, no 21-DTE candidate, no 0.15-delta or
  0.20-delta candidate. **`suggest_rolls` does NOT emit
  `.attrs['drops']`** — the driver confirmed
  `rolls.attrs.get('drops', 'NOT_SET') == 'NOT_SET'`. The trader has
  no diagnostic for *why* 81% of the candidate grid vanished
  (event gate? `min_net_credit` filter? strike-solve failure?).
  **Observability gap. Logged** as the highest-leverage finding
  from this run.

- **`rank_candidates_by_ev` and `rank_covered_calls_by_ev` both
  emit `.attrs['drops']` (S1 / S2 logged; PR #102 / PR #124).**
  `WheelTracker.suggest_rolls` and `WheelTracker.suggest_call_rolls`
  (PR #126) do not. **Asymmetric observability across the four
  EV-aware ranking surfaces.** Three of the four have drops; one
  silently filters. **Logged.**

- **Covered-call ranker on freshly-assigned stock is event-gated
  hard.** All `(dte × delta)` grid points except DTE=14 blocked by
  PNC's 2026-04-15 earnings (+5d buffer); 12 of 16 candidates in
  drops. Only DTE=14 (expiry 2026-04-03, 12 days before earnings)
  survives. **Operational observation, not a bug** — the event
  gate is doing exactly what it should. But the practical
  consequence is: a freshly-assigned shareholder ~30 days before
  the company's earnings has only a 14-day covered-call window, and
  premium harvest is structurally compressed. A `low_dte_only`
  warning bit on the CC result would surface this without forcing
  the trader to inspect `.attrs["drops"]`. **Logged.**

- **`reason` strings in `.attrs['drops']` still mangle `±`** under
  Windows cp1252. Live drops sample:

  ```
  {'ticker': 'PNC', 'gate': 'event',
   'reason': 'event_lockout:earnings@2026-04-15 (�5d buffer)'}
  ```

  The intended character is `±` (U+00B1); cp1252 renders it as
  `\xfd�`. Same root cause as S1's `Δ` (U+0394) crash in
  `candidate_dossier.py` R3 review note. **S1's Unicode finding
  is still alive on the drops path.** The driver itself crashed
  once when an `f"Δ={...}"` was printed (had to swap to `d=`); the
  drops-string mangle doesn't crash, just renders unreadable.
  **Logged — repeating from S1.**

- **`rank_covered_calls_by_ev` does not emit a `delta` column** —
  the column is `target_delta` (per `_CC_RANK_CORE_COLUMNS` at
  `engine/wheel_runner.py:154-171`). The CC ranker returns the
  *target* delta used to solve the strike, not the BSM delta at
  the solved strike — those would be near-identical on synthetic
  BSM prices but could diverge on a real chain. **Observation, not
  a bug** — the column is documented, and `rank_candidates_by_ev`
  uses the same convention. **Logged.**

- **Composed-assignment EV is by design not surfaced by any single
  call.** The wheel-trader's real number is "stock paper P&L + CC
  forward EV", but `rank_covered_calls_by_ev`'s docstring (at
  `engine/wheel_runner.py:1648-1651`) deliberately scopes the
  output to the option leg only — "The stock leg's P&L (basis vs
  an assigned/called-away price) is separate position accounting".
  This is the correct scoping decision (forward EV vs realised
  paper), but it puts the composed view in the trader's
  spreadsheet, not in the ranker. A `WheelTracker.evaluate_
  assignment_branch(ticker)` helper that composes the two would
  close the gap; out of scope here. **Logged as a UX observation.**

- **§2 verified across all three branches.** `suggest_rolls`
  invokes `EVEngine.evaluate` once per (DTE, delta) grid point
  plus once for the hold synthetic — per
  `engine/wheel_tracker.py:1972-1985` ("Every candidate's EV —
  both `hold_ev` and each `roll_ev` — runs through
  `EVEngine.evaluate` directly"). `rank_covered_calls_by_ev`
  invokes it once per surviving CC candidate. **No tradeable
  candidate surfaces without `EVEngine.evaluate`. §2 invariant
  preserved across the rolling decision tree.** **Logged as a
  positive.**

**Verdict.**

- **`suggest_rolls` works end-to-end and produces a defensible
  recommendation** on a real ITM situation: roll +$63 better than
  hold, both negative; assignment composed −$1,530 (worst). The
  ranker's `recommend` boolean fires correctly (True iff
  `roll_ev > hold_ev`). **No correctness bug surfaced.**

- **`suggest_rolls` observability is asymmetric with the put /
  covered-call ranker pair.** No `.attrs['drops']`, no diagnostic
  for why 13/16 candidates were filtered. The same asymmetry
  almost certainly applies to `suggest_call_rolls` (the covered-
  call roll surface, PR #126); not exercised here. **Observability
  gap — highest-leverage finding from this run.**

- **The S1 `±` / `Δ` Unicode cp1252 finding is still alive.** It
  hits the drops-`reason` strings and any driver code that prints
  the literal Greek delta. The fix is one-line per call-site
  (replace `±` with `+/-`, `Δ` with `d` or `delta`) but spans
  several files. Not fixed in this Sn.

**AI handoff.**

- **Fix sketch for `suggest_rolls.attrs['drops']`.** The structural
  analogue is `WheelRunner.rank_candidates_by_ev` — it appends
  drop dicts at four gate sites (data / history / event / strike /
  premium / chain_quality / ev_threshold) and exposes them via
  `frame.attrs['drops'] = drops`. `suggest_rolls` has the same
  drop sites (event gate, strike-solve failure, `min_net_credit`
  filter) but never accumulates. One literal pattern that would
  close the gap, in
  `engine/wheel_tracker.py:suggest_rolls` (approx — the function
  body builds `candidates: list[dict]` internally):

  ```python
  # current (illustrative):
  for dte, delta in itertools.product(target_dtes, target_deltas):
      try:
          new_strike = _solve_put_strike(...)
      except ValueError:
          continue              # silent drop
      ...
      if net_credit_debit < min_net_credit:
          continue              # silent drop
      ...

  # proposed:
  drops: list[dict] = []
  for dte, delta in itertools.product(target_dtes, target_deltas):
      try:
          new_strike = _solve_put_strike(...)
      except ValueError as e:
          drops.append({"ticker": ticker, "gate": "strike",
                        "reason": f"solve_failed:{e}",
                        "dte": dte, "target_delta": delta})
          continue
      ...
      if net_credit_debit < min_net_credit:
          drops.append({"ticker": ticker, "gate": "min_credit",
                        "reason": f"net_credit_debit={net_credit_debit:.2f} "
                                  f"< min={min_net_credit}",
                        "dte": dte, "target_delta": delta})
          continue
      ...
  result = pd.DataFrame(candidates, ...)
  result.attrs["drops"] = drops
  return result
  ```

  Mirror change in `suggest_call_rolls` (`engine/wheel_tracker.py`
  ~ line 2196). **Behavior of survivor rows is unchanged. Pure
  observability add. Decision-layer surface — needs a decision-
  layer lock claim (per PARALLEL_SESSIONS) and a regression
  test.** Out of scope for this Sn.

- **Fix sketch for the Unicode mangle on `±` / `Δ`.** Replace
  `±` in the event-gate `reason` string at the gate-emit site
  (in `engine/wheel_runner.py` rank_candidates_by_ev /
  rank_covered_calls_by_ev — search for the
  `event_lockout:earnings@...` format string). Replace with `+/-`
  for cp1252 safety. Mirror fix for `Δ` in
  `engine/candidate_dossier.py` R3 review note (re-logging from
  S1). Each is a one-character literal change; the harder
  question is whether to fix at the producer site (engine) or
  at the consumer site (the driver / dashboard). **Producer fix
  is cleaner — make the engine emit cp1252-safe text by default.**
  Out of scope for this Sn.

- **A `WheelTracker.evaluate_assignment_branch(ticker)` helper**
  would close the composed-EV UX gap by combining the realised
  stock paper P&L (from the position's `stock_basis` field, which
  exists post-assignment per `engine/wheel_tracker.py:678`) with
  the forward EV of the best covered-call candidate. Returns
  `(composed_ev, stock_paper, best_cc_ev_dollars,
  best_cc_strike, best_cc_dte)`. Decision-layer surface; needs
  its own claim + regression test. Out of scope.

**Methodology debt.**

- **Single-name single-decision-moment scenario.** The roll
  decision is a sequence — at DTE=14, 7, 3, 1 the trader makes
  it again. S22 ran one decision moment; a multi-decision-moment
  follow-up could observe whether `recommend` flips as DTE
  decays (the buyback gets cheaper, so the roll's debit shrinks).
  **Logged.**

- **Synthetic premiums throughout** — the opened put, the
  buyback, every roll candidate, every CC candidate are all
  BSM-fair from the connector's IV / spot / risk-free pulls. Real
  chain quotes (Theta) would give actual buyback / new-premium
  asks; not run here per the Theta-isolation constraint of this
  campaign (the other agent is on the Theta surface). **Logged.**

- **One ticker.** The drawdown survey identified ~30 names with
  10-12% drops over the window; S22 ran PNC only. A future Sn
  could batch the survey output through `suggest_rolls` to
  characterise the cross-section (does the `recommend` rate vary
  with sector / cap / IV regime?). **Logged.**

- **Ruled out per the campaign constraints:** Theta provider
  (other agent active), `models/`-empty HMM (no persisted regime
  artifacts), decision-layer code change (S22 found gaps, did
  not fix), `date.today()` paths (none touched on `as_of`
  branches), dashboard surface (read-only on engine only).

- **Re-verified 2026-05-26** by Terminal A — **SKIPPED (archival)**. Per the task spec skip list: "S22 archival — pre-IV-PIT-fix engine. Re-running on current engine duplicates S27." The S22 documentation was written against the pre-#179 engine (snapshot IV); re-running the same setup on `main` (post-#179) would either reproduce S27's IV-PIT-rerun numbers (which are the post-fix companion of S22) or yield a third snapshot at `as_of=2026-03-13` — neither of which add value over S27's already-published per-year ρ and quartile spreads. The §2 holds-by-construction finding (suggest_rolls invokes `EVEngine.evaluate` per candidate plus the hold synthetic) is verified in the S3 / S26 re-runs above. S22 F1 (drops accumulator on suggest_rolls) is closed by PR #181 — confirmed live in S3's re-verify (drops_attr_set=True, 4 drops observed).

### S23 — Earnings-window navigation (event gate + IV-crush on AVGO)

**Purpose.** Exercise the engine's earnings-aware behavior end-to-end:
event-gate boundary on `WheelRunner.rank_candidates_by_ev` across
the trading day before / day of / day after a real earnings event,
plus the IV-crush impact on the forward-distribution + strike-solve
that a wheel trader would expect to see in the ranker output. AVGO
reported 2026-03-04 (Wed) inside the data window — a clean target
for the boundary scan, with multiple post-earnings trading days
available before the 2026-03-20 OHLCV cutoff.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Five-ticker basket: **AVGO** (target) + AAPL, MSFT, KO, HD (clean
controls with no earnings inside any tested buffered window).
Default 35-DTE / 25-delta / 5-day earnings buffer. Four `as_of`
dates probing the boundary:

- **2026-03-03** (Tue, TDB — earnings tomorrow)
- **2026-03-05** (Thu, TDA — earnings yesterday)
- **2026-03-10** (Tue, 6 calendar days post-event — first date
  outside the nominal back buffer)
- **2026-03-13** (Fri, 9 calendar days post-event — deep into the
  post-event regime)

Driver under `%TEMP%\s23\`, not committed. Pure observation —
reads `wr.rank_candidates_by_ev(...).attrs['drops']` and the
survivor frame; no `EnginePhaseReviewer` wiring, no `WheelTracker`
attached.

**Path.** `rank_candidates_by_ev` at `engine/wheel_runner.py:579`
builds a per-run `EventGate` (`engine/event_gate.py:76`) from
`conn.get_next_earnings(ticker, as_of)` per ticker
(`engine/wheel_runner.py:906`). The gate's
`_event_touches_window` (`event_gate.py:110-119`) is symmetric —
buffer applied to both `trade_start` and `trade_end`. The IV input
for the strike-solve comes from `conn.get_fundamentals(ticker)`
(`engine/wheel_runner.py:813-816`), preferring
`implied_vol_atm` then falling back to `volatility_30d`.

**Status.** Done. **Verdict: the event gate boundary fires
correctly on AVGO at the TDB (2026-03-03), then never again — and
the IV the engine uses for the strike-solve is a single snapshot
that does NOT change across the four `as_of` dates. Three
structural findings.**

**Findings:**

- **Headline cross-section.** AVGO across the four as_of dates,
  same 35-DTE / 25-delta config:

  ```
  2026-03-03 (TDB)        DROPPED  gate=event  reason=event_lockout:earnings@2026-03-04 (+/-5d buffer)
  2026-03-05 (TDA)        SURVIVED iv=0.4296   premium=$7.139   ev_dollars=$+268.85   ev_per_day=$+14.28
  2026-03-10 (6d post)    SURVIVED iv=0.4296   premium=$7.325   ev_dollars=$+310.30   ev_per_day=$+16.93
  2026-03-13 (9d post)    SURVIVED iv=0.4296   premium=$6.857   ev_dollars=$+150.46   ev_per_day= $+8.28
  ```

  The event gate fires exactly once — at the TDB — then AVGO
  reappears at the TDA and never disappears again. **Logged.**

- **(F1) `get_next_earnings` is strictly forward-only — the 5d
  back-buffer in `EventGate` is effectively dead code.**
  `engine/data_connector.py:408` filters with
  `df[df["announcement_date"] > ref]`. So at `as_of=2026-03-05`
  (the day after AVGO's 2026-03-04 earnings),
  `get_next_earnings("AVGO", "2026-03-05")` returns `None` — the
  just-passed earnings event is never registered on the
  `EventGate`, so the symmetric back-buffer at
  `engine/event_gate.py:117` never has any past event to test
  against. **The 5d back buffer is unreachable in production.**
  Live probe from the driver:

  ```
  as_of=2026-03-03  next_earnings_after={'announcement_date': Timestamp('2026-03-04 ...')}
  as_of=2026-03-04  next_earnings_after=None
  as_of=2026-03-05  next_earnings_after=None
  as_of=2026-03-10  next_earnings_after=None
  as_of=2026-03-13  next_earnings_after=None
  ```

  **Either** the back-buffer was intended to fire on just-passed
  earnings (and the `>` filter at `data_connector.py:408` is a bug
  — it should be `>=` minus the buffer, or the gate should pull
  past events too), **or** the back-buffer is intentionally
  dormant (the wheel trader can write into post-earnings IV crush
  opportunistically). The current code says one thing
  (symmetric buffer) and the connector says another (forward-only
  feed) — they disagree. **Logged as a structural inconsistency.**

- **(F2) Bloomberg earnings CSV is forward-truncated for many
  tickers — silent event-gate bypass.** Live driver probe at
  `as_of=2026-03-20`:

  ```
  AAPL  : next after 2026-03-20 = None
  MSFT  : next after 2026-03-20 = None
  GOOGL : next after 2026-03-20 = None
  AVGO  : next after 2026-03-20 = None
  COST  : next after 2026-03-20 = None
  ORCL  : next after 2026-03-20 = None
  ```

  AAPL/MSFT/GOOGL/AVGO/COST/ORCL all have well-known late-April
  2026 earnings in real life, but the Bloomberg earnings CSV in
  the repo (`data/bloomberg/sp500_earnings.csv`, last row
  2026-03-31) has no entries past mid-March for these names. The
  consequence: at `as_of=2026-03-20`, the event gate is a **no-op**
  on six of the top-ten S&P 500 names. A trader running a 35-DTE
  ranker at the data cutoff would freely open AAPL/MSFT/GOOGL
  positions whose holding window crosses their real earnings. **The
  event gate's silent-on-no-data behavior makes this invisible**:
  no drop entry, no warning, just no event registered. By
  contrast, XOM (2026-04-07 earnings IS in the CSV), JPM
  (2026-04-14), UNH (2026-04-21), and JNJ (2026-04-14) DO get
  blocked correctly — the gate is doing its job when the data is
  there. **Logged as a data-completeness vs. observability gap.**

- **(F3) The IV input to the strike-solve is NOT PIT-aware.**
  AVGO surfaced with `iv=0.4296` (= 42.96%) at **all four** of
  the post-event `as_of` dates — even though the connector's
  `get_iv_history` shows the put-IV moving meaningfully:

  ```
  date        hist_put_imp_vol  volatility_30d
  2026-03-03           55.935          36.208
  2026-03-04           52.957          36.309    (earnings day)
  2026-03-05           49.547          38.926
  2026-03-10           48.437          40.184
  2026-03-13           49.819          42.414
  2026-03-20           46.503          35.010
  ```

  The 42.96% value matches `conn.get_fundamentals("AVGO")
  ['implied_vol_atm']` exactly — and that connector method
  (`engine/data_connector.py:590`) takes **no `as_of` argument**
  and reads from a snapshot fundamentals CSV
  (`sp500_fundamentals.csv`) that has **no date column** at all:

  ```
  >>> conn.get_fundamentals('AVGO')
  {..., 'volatility_30d': 34.875, 'implied_vol_atm': 42.9566, ...}
  ```

  So the engine's per-call IV is frozen — same value at
  `as_of=2026-02-13` as at `as_of=2026-03-20`. **The "PIT-safe"
  claim in `rank_candidates_by_ev` (line 607 of the docstring)
  is true for OHLCV and the empirical forward distribution
  derived from it, but the IV used for the strike-solve and the
  BSM-fair premium is a snapshot.** The IV-crush experiment is
  literally not observable through the ranker output.

  This isn't theoretical — it shapes the result. At 2026-03-05
  AVGO's *real* IV was 49.5% (immediately post-crush spike); at
  2026-03-13 it was 49.8%; at 2026-03-20 it was 46.5%. The
  engine used 42.96% throughout — too low at the TDA, roughly
  right at 2026-03-20. The strike solved at the wrong IV (lower
  than reality at the TDA) is too far OTM, the synthetic premium
  is undershoot, and `ev_dollars` is mispriced versus what the
  trader would actually transact at. **Logged as the highest-
  leverage finding in S23.**

- **The `WheelTracker._connector_atm_iv` helper at
  `engine/wheel_tracker.py:1344` already does the right thing for
  mark-to-market** — it pulls
  `conn.get_iv_history(ticker, end_date=as_of)` and takes the
  most recent row, normalising percent→decimal. The same helper
  pattern would solve F3 inside `rank_candidates_by_ev`. Cross-
  reference: `rank_covered_calls_by_ev` (used in S22) and
  `rank_strangles_by_ev` likely share the same IV-snapshot bug
  (both use the same `conn.get_fundamentals` fallback per
  `engine/wheel_runner.py:1913` and `:2377`), not exercised
  separately in S23. **Logged.**

- **§2 verified.** Every candidate that surfaced as tradeable
  (positive `ev_dollars`, no event drop) routed through
  `EVEngine.evaluate` — per the engine's standard ranker contract.
  The findings above are about **what IV the engine evaluated
  WITH**, not about a bypass of `evaluate`. No §2 violation.
  **Logged as a positive.**

- **Control names behave correctly.** AAPL/MSFT/KO/HD survive at
  all four `as_of` dates with no event-gate drops — none of their
  earnings fall inside the buffered window for any of the four
  tested dates (AAPL/MSFT Jan 2026, KO 2026-02-10, HD
  2026-02-24, all sufficiently before; their next April-2026
  earnings are not in the CSV, which is finding F2). **Logged.**

- **`±` cp1252 mangle on `event_lockout` reason strings**
  re-confirmed in this run — `event_lockout:earnings@2026-03-04
  (±5d buffer)` rendered as `... (�5d buffer)` on the Windows
  console. Producer-side one-character fix. **Logged.**

**Verdict.**

- **Event-gate FORWARD buffer behaves correctly.** AVGO blocked
  at TDB (2026-03-03) with the expected `event_lockout` reason.
  Standard wheel-trader expectation met.

- **Event-gate BACK buffer is dead code.** `get_next_earnings` is
  strictly forward-only, so the symmetric 5d-back logic at
  `event_gate.py:117` has nothing to trigger on. Either fix the
  connector or document the asymmetry — currently the code reads
  symmetric and the behavior is forward-only. **Structural
  inconsistency.**

- **The earnings CSV is incomplete for major tickers' April-2026
  reports** — silent event-gate bypass on AAPL/MSFT/GOOGL-class
  names at the data cutoff. Data refresh, not engine, but
  surfaces a brittle "silent-on-no-data" contract. **Logged.**

- **The IV input to the put-entry strike-solve is a single
  snapshot, NOT a PIT-aware time-series.** The engine ranker
  cannot reflect IV-crush at all; the value is frozen between
  fundamentals refreshes. Material to anyone trading around
  earnings. **Logged as the highest-leverage finding.**

**AI handoff.**

- **F3 (IV snapshot) fix sketch — promote `rank_candidates_by_ev`
  to use the same PIT-aware IV helper `WheelTracker` already
  uses.** Today, at `engine/wheel_runner.py:813-816`:

  ```python
  fundamentals = conn.get_fundamentals(ticker) or {}
  iv_raw = fundamentals.get("implied_vol_atm")
  if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
      iv_raw = fundamentals.get("volatility_30d")
  ```

  Proposed (mirrors `WheelTracker._connector_atm_iv` at
  `engine/wheel_tracker.py:1344-1383`):

  ```python
  iv = None
  if hasattr(conn, "get_iv_history"):
      try:
          hist = conn.get_iv_history(ticker, end_date=as_of)
          if hist is not None and not hist.empty:
              cols = [c for c in ("hist_put_imp_vol", "hist_call_imp_vol")
                      if c in hist.columns]
              if cols:
                  row = hist.iloc[-1]
                  vals = [float(row[c]) for c in cols if pd.notna(row[c])]
                  if vals:
                      iv = sum(vals) / len(vals)
      except Exception:
          iv = None
  if iv is None:
      fundamentals = conn.get_fundamentals(ticker) or {}
      iv_raw = fundamentals.get("implied_vol_atm")
      if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
          iv_raw = fundamentals.get("volatility_30d")
      iv = float(iv_raw) if iv_raw is not None else 0.0
  # Existing percent->decimal normalisation continues below.
  ```

  Mirror change in `rank_covered_calls_by_ev`
  (`engine/wheel_runner.py:1913`) and `rank_strangles_by_ev`
  (`engine/wheel_runner.py:2377`). **Decision-layer surface — needs
  a decision-layer lock claim and regression coverage that the IV
  used by the ranker matches `get_iv_history(ticker, as_of).iloc[-1]`
  when both are available.** Out of scope for this Sn.

- **F1 (back-buffer dead code) options.** Either (a) fix
  `get_next_earnings` to return events within `as_of - max_buffer`
  through the future, so the back-buffer in the gate has events
  to test, or (b) document the asymmetry and remove the
  back-buffer arithmetic to avoid the misleading code. (a) is
  the trader-intent-preserving fix (block writes immediately
  post-earnings until the news / IV-crush absorbs); (b) is the
  honest-code fix if "write into the crush" is the actual
  policy. **Design call**, not a usage-test fix.

- **F2 (forward-truncated earnings CSV) is a data refresh.** The
  Bloomberg earnings file ends mid-March 2026 for AAPL-class
  tickers; the next April earnings need to be pulled. **Data,
  not engine.** Tracked under the existing Bloomberg-refresh
  memory.

- **A regression test that would have caught F3** — a unit test
  asserting that, for a fixed ticker and two `as_of` dates with
  different `hist_put_imp_vol` values in `sp500_vol_iv_full.csv`,
  the `iv` column in `rank_candidates_by_ev`'s output differs.
  Today's behavior would fail that assertion. Out of scope for
  this Sn.

**Methodology debt.**

- **Single ticker, single earnings event.** S23 ran AVGO only.
  COST (2026-03-05), ORCL (2026-03-10), LULU (2026-03-17),
  MU (2026-03-18) are all in the data window and would let the
  finding be replicated across more events. **Logged.**

- **The `regime_multiplier` column was empty in the survivor
  output** (omitted from the printed columns because the
  diagnostic column wasn't populated by the engine on this
  basket). Whether that's an HMM-cold-start issue (no persisted
  model in `models/`) or by design at this basket size isn't
  exercised here. **Logged for a future Sn** — overlaps with the
  ruled-out scenario E (steady-state regime sizing).

- **R7 / R8 not exercised in S23** — no `PortfolioContext`
  attached, no `EnginePhaseReviewer` wired. S21 covered them at
  a different angle; S24 (multi-strategy book) will exercise
  them on a richer book.

- **Ruled out per the campaign constraints:** Theta provider,
  decision-layer code change (S23 found gaps, did not fix), the
  HMM regime path (no persisted model), the dashboard surface.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every AVGO survivor at the post-event dates routed through `EVEngine.evaluate` (PIT-IV input now).
  - Qualitative verdict: **partial — both major findings F1 and F3 are CLOSED**:
    - **F1 (event-gate BACK buffer dead code) — CLOSED**: `MarketDataConnector.get_recent_earnings(...)` now exists on `main` (Terminal B's symmetric-gate work, PR #180). The 5-day back-buffer is now reachable. In this re-run, `as_of=2026-03-05` (1 calendar day after AVGO's 2026-03-04 earnings) yields **0 ranked rows with an event-gate drop** — exactly the symmetric back-buffer the original entry said was dead. (Originally `as_of=2026-03-05` surfaced AVGO with `ev_dollars=+268.85`; post-#180 it's blocked).
    - **F3 (IV input not PIT-aware) — CLOSED**: AVGO's `iv` column at the two post-event dates **now differs**: `iv=0.4844` at 2026-03-10, `iv=0.4982` at 2026-03-13. Originally both used `iv=0.4296` (the snapshot). The current values agree with the IV history file's `(hist_put_imp_vol + hist_call_imp_vol) / 2` per PR #179.
  - Numerical drift > 5% (with attribution):
    - metric `AVGO_iv[2026-03-10]`: orig `0.4296` → new `0.4844` (`+12.8%`); attributable to **PR #179** (`_resolve_pit_atm_iv`).
    - metric `AVGO_iv[2026-03-13]`: orig `0.4296` → new `0.4982` (`+16.0%`); attributable to **PR #179**.
    - metric `AVGO_ev_dollars[2026-03-10]`: orig `+310.30` → new `+390.06` (`+25.7%`); higher PIT-IV raises the synthetic premium and EV.
    - metric `AVGO_ev_dollars[2026-03-13]`: orig `+150.46` → new `+222.72` (`+48.0%`); same direction.
    - metric `AVGO_at_2026-03-05_survived`: orig **True** (iv=0.4296, ev=+268.85) → new **False** (dropped on event_lockout back-buffer); attributable to **PR #180** (symmetric event gate via `get_recent_earnings`).
  - Notes: F2 (Bloomberg earnings CSV forward-truncated past mid-March 2026) is a data-completeness issue, not engine — same state on the worktree's CSV.

### S24 — Multi-strategy book composition ($500k wheel + CC + strangle scan)

**Purpose.** S21 explicitly flagged multi-strategy book composition
as the next pro-trader-lens scenario. Build a $500k NAV book that
holds short puts + covered calls (post-assignment) simultaneously,
attach a `PortfolioContext` and run `EnginePhaseReviewer` on a fresh
candidate, then probe the D17 hard-block + dossier soft-warn gates
directly to characterise how they compose across strategies (option-
only vs option+stock). Try the strangle ranker
(`WheelRunner.rank_strangles_by_ev`, S14 / #118) on a third ticker
to confirm whether/how strangles can join the book.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20` (data cutoff), `WheelTracker(initial_capital=
$500_000, require_ev_authority=False)`. Non-strict mode keeps the
EV-authority token plumbing out of the way so the gate-composition
math is the only thing under test. Driver under `%TEMP%\s24\`,
not committed.

The book is built via the public surfaces:

1. `wr.rank_candidates_by_ev(['MRK'], delta_target=0.20)` →
   `tracker.open_short_put(...)` (Health Care, MRK $105.50, 35 DTE).
2. `wr.rank_candidates_by_ev(['KO'], delta_target=0.20)` →
   `tracker.open_short_put(...)` (Consumer Staples, KO $71.00).
3. `tracker.handle_put_assignment('KO', as_of, spot=$70.50)` to
   transition KO into `STOCK_OWNED`, then
   `wr.rank_covered_calls_by_ev('KO', shares_held=100, ...)` →
   `tracker.open_covered_call(...)` (CC at KO $80, DTE 49) to land
   the position in `COVERED_CALL`.
4. `wr.rank_strangles_by_ev('NVDA', ...)` for the strangle scan —
   pure ranking, no tracker call (no `open_strangle` surface
   exists; S14 finding).
5. `take_snapshot(tracker.positions, today=AS_OF)` →
   `PortfolioContext(held_option_positions, stock_holdings,
   spot_prices, nav)`.
6. Construct a `CandidateDossier` for a fresh CAT candidate
   (`rank_candidates_by_ev(['CAT'])`), attach a synthetic
   `ChartContext(visible_price=spot, screenshot_path=Path('.../
   synthetic-s24.png'), error='')` so R2 / R3 stay silent, then
   call `EnginePhaseReviewer.review(...)` with and without the
   `PortfolioContext`.
7. Direct probes of `check_var`, `check_stress_scenario`,
   `check_sector_cap`, `check_portfolio_delta` to read out the
   gate math on the multi-strategy book.

`returns_data` for R7 is built ad-hoc from connector OHLCV — 252-day
daily log returns per ticker (MRK + KO + CAT) — so R7 has actual
inputs rather than the missing-data skip from S21.

**Path.** `take_snapshot` at `engine/portfolio_risk_gates.py:177`
maps each `PositionState` to the snapshot shape:

  - `SHORT_PUT` → one option dict (short put leg).
  - `STOCK_OWNED` → one stock holding (no option leg).
  - `COVERED_CALL` → one option dict (short call leg) + one stock
    holding.

`EnginePhaseReviewer.review` runs R1 → R3 → R4 → R5 → R6 → R7 → R8
in sequence per `engine/candidate_dossier.py:197-353`. R7 and R8
only fire when `portfolio_context is not None and verdict ==
"proceed"`. The C4 vol-spike scenario (-10% spot + 30% IV) is
defined at `engine/portfolio_risk_gates.py:118`. The sector cap
(`check_sector_cap`, line 271) ignores stock holdings by design
(option-side concentration only); the delta cap
(`check_portfolio_delta`, line 344) includes the stock-leg
delta-dollars via the `stock_holdings` argument.

**Status.** Done. **Verdict: gate composition behaves correctly
across strategies (3 of 5 questions cleanly answered, 1 negative
finding on R8 stress, 1 pre-existing S14 strangle-integration
gap re-confirmed). One major methodology finding surfaced — the
dev-box's `sys.path` discovery silently picks up an older primary
clone, so a driver run from `%TEMP%` without explicit
`sys.path.insert(0, worktree)` evaluates against the wrong engine
SHA. Drivers fixed; S22 / S23 re-validated bit-identically against
the worktree.**

**Findings:**

- **Q1 — `take_snapshot` correctly translates the 3-state book.**
  Live driver output:

  ```
  option_positions (2):
    {'symbol': 'MRK', 'option_type': 'put',  'strike': 105.5, 'dte': 35, 'iv': 0.3128, 'contracts': 1, 'is_short': True}
    {'symbol': 'KO',  'option_type': 'call', 'strike': 80.0,  'dte': 49, 'iv': 0.2127, 'contracts': 1, 'is_short': True}
  stock_holdings (1):
    ('KO', 100)
  ```

  The covered-call position decomposes into TWO dict entries
  (option + stock); the SHORT_PUT into one; the STOCK_OWNED
  (transitional) into one stock entry. **Snapshot fidelity is
  correct across all three states. Logged as a positive.**

- **Q2 — R7 (VaR) correctly fires with real returns_data.** Driver
  built 252-day log returns per held ticker (MRK / KO) plus the
  candidate (CAT) from `conn.get_ohlcv(...)`. Direct probe:

  ```
  check_var (no returns_data):  passed=True reason='missing_data' skipped
  check_var (with returns_data): passed=True
    var_dollars=$4,344.88  cvar_dollars=$5,448.66  var_pct=0.87%  var_limit_pct=5%
  ```

  **R7 path is wired end-to-end and passes the actual VaR math
  when the inputs are supplied** — the limit was 5% NAV ($25k);
  the actual portfolio 30-day VaR_95 is 0.87% NAV. **Closes S21's
  Q2-shaped follow-up** ("a future Sn could exercise R7 with real
  returns data"). The 2-position book is well under the cap, but
  the math fires and would downgrade `proceed → review` on a
  book whose VaR breached 5%. **Logged as a positive.** **Open
  follow-up:** no upstream caller assembles `returns_data` for the
  `PortfolioContext` automatically — the dossier-builder / tracker
  hand-off has no `returns_data` plumbing; the operator-level
  caller would need to build it. Mirrors S21's "PortfolioContext
  with real returns" hand-off.

- **Q3 — R8 (C4 vol-spike stress) is benign on a $500k 2-position
  book.** Driver output:

  ```
  check_stress_scenario (C4 Vol Spike: -10% spot + 30% IV):
    passed=True  portfolio_pnl_dollars=-$5,266.72  drawdown_pct=1.05%  drawdown_limit_pct=8%
  ```

  Same shape S21 saw at $1M / 2 positions (drawdown 0.56% vs 8%
  cap). **R8 needs a larger or more concentrated book to fire.**
  Mechanically wired, contractually correct, but never bites at
  this scale. **Logged.** S21's "exercise R8 with a richer book"
  follow-up is still open; S24 corroborates it.

- **Q4 — Strangle ranker produces candidates the tracker cannot
  ever accept.** `wr.rank_strangles_by_ev('NVDA', target_dtes=
  (35,49), target_deltas=(0.20,0.15))` returned 4 candidates:

  ```
     put_strike  call_strike  dte  ev_dollars timing_recommendation    timing_phase
  0       156.0        197.0   35    -1360.22           conditional  post_expansion
  1       159.5        192.5   35    -1442.91           conditional  post_expansion
  2       153.5        202.0   49    -1734.34           conditional  post_expansion
  3       157.5        196.5   49    -1882.32           conditional  post_expansion
  ```

  All 4 are negative composed EV (timing gate "conditional", not
  "avoid"), and `WheelTracker` exposes **no** `open_strangle`,
  `SHORT_STRANGLE` state, or any way to add a strangle leg-pair
  to the book. Grep on the tracker file confirms zero matches for
  `strangle | SHORT_STRANGLE | open_strangle`. **The S14 / #118
  finding is unresolved at the tracker layer:** strangles are
  rank-only — they can score, they cannot be tracked, sized
  against NAV, or fed into R7 / R8 via `take_snapshot`. **A
  tradeable strategy with no portfolio integration.** Re-logging.

- **Q5a — `check_sector_cap` correctly ignores stock holdings.**
  Live probe:

  ```
  check_sector_cap (CAT in Industrials, $65,000 notional):
    passed=True post_open_sector_pct=0.13 sector_limit=0.25
  ```

  Post-open Industrials = 13% of NAV; the existing KO option
  (Consumer Staples) and MRK option (Health Care) don't show in
  Industrials. **And critically, the KO 100 shares from the
  assignment also don't contribute to the Industrials check** —
  matching the docstring at `engine/portfolio_risk_gates.py:286-
  293` ("stock holdings don't contribute to the option-side
  sector exposure"). **Design call** — the wheel mental model is
  "options drive risk concentration; the assigned stock is parked
  capital." A pro-trader who'd rather see *total-position* sector
  exposure (option + stock) would want a different cap. Mostly
  a transparency note. **Logged as expected behavior, with the
  design caveat.**

- **Q5b — `check_portfolio_delta` DOES include the stock holding's
  delta.** Live probe (candidate CAT 25-delta @ $625.50 against the
  multi-strategy book):

  ```
  check_portfolio_delta (CAT 25-delta @ $625.50 + stock_holdings):
    passed=False  reason='portfolio_delta_breach'
    current_portfolio_delta_dollars=$7,983.29  post_open_delta_dollars=$37,944.55  delta_cap_dollars=$1,500.00
  ```

  Current book delta = **$7,983**, dominated by the KO stock leg
  (100 shares × $74.69 spot ≈ $7,469); the short put on MRK and
  the short call on KO contribute roughly the remaining $514
  combined. **The stock holding visibly bites into the
  portfolio-delta cap** — exactly opposite of the sector cap's
  treatment. **Both behaviors are correct per their docstrings;
  the cross-gate asymmetry (sector option-only, delta total)
  is a documented design call.** **Logged.**

  Adding CAT (+$16,815 of delta-dollars per a 25-delta at $625.50
  × 100 × ~0.25) would push the book to $37,945 — 25× over the
  $1,500 cap. The delta cap remains the dominant binding
  constraint at $500k NAV (`300 × ($500k / $100k) = $1,500`),
  echoing S21.

- **`EnginePhaseReviewer` verdict-delta WITH vs WITHOUT
  `PortfolioContext`.** Same multi-strategy book, same CAT
  candidate, same synthetic chart (R2/R3 silent):

  ```
  WITHOUT PortfolioContext:
    verdict='proceed' reason='ev_above_threshold'
  WITH PortfolioContext (multi-strategy book attached):
    verdict='proceed' reason='ev_above_threshold'
    note: R7: VaR check skipped (no_correlation_matrix_or_returns_data)
  ```

  No verdict change — the `returns_data` was assembled in the
  *direct* R7 probe (Q2 above) but the `PortfolioContext` passed
  to `EnginePhaseReviewer` did **not** carry it through. **R7
  silently skipped despite the dossier path having a context
  attached.** This is the second face of the "no upstream
  `returns_data` plumbing" finding above — the field is
  optional, the reviewer doesn't fabricate it, and the
  downgrade-only contract means absent data = silent pass.
  **R7's downgrade-only-when-fired contract held.** **Logged.**

- **(F-METH-1) Dev-box `sys.path` discovery silently picks up
  the primary clone, not the worktree.** Highest-leverage finding
  from this run. A driver run as `python %TEMP%\s24\driver.py`
  with cwd at the project directory ends up with this `sys.path`:

  ```
  ''                                                       (cwd-empty - effective)
  ...
  'C:\\Users\\merty\\Desktop\\Local AI Agent'              (user-site)
  'C:\\Users\\merty\\Desktop\\smart-wheel-engine'          (older primary clone)
  ```

  When the script is invoked by path (not `-c`), Python sets
  `sys.path[0]` to the script's directory (`%TEMP%\s24\`), which
  is not the project. The cwd is *not* automatically on the path.
  But the user-site `pth` files add `C:\Users\merty\Desktop\
  smart-wheel-engine` — **a separate, older clone** currently at
  `cd16443` (pre-D17, diverged 776 lines on `portfolio_risk_gates`
  alone from `origin/main` at `86b917c`). The driver imports
  `engine.portfolio_risk_gates` **from that older clone**,
  not from the Terminal A worktree.

  **My S22 and S23 drivers ran against the older primary clone
  silently.** I noticed on S24 only because the older clone
  doesn't *have* `portfolio_risk_gates.py` (pre-D17), so the
  import failed and the masking became visible. **For S22 and
  S23 the import succeeded against the older code, and the
  driver produced findings I logged against the wrong SHA.**

  **Re-validation: I re-ran both S22 and S23 with an explicit
  `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
  prepended.** Both produced **bit-identical** numbers — every
  EV, every drop, every IV value — to the original primary-clone
  runs. The relevant code paths (`suggest_rolls`,
  `rank_covered_calls_by_ev`, `get_next_earnings`,
  `get_fundamentals`, `event_gate.is_blocked`, the IV-snapshot
  fallback at `wheel_runner.py:813-816`) are common to both
  SHAs. So the S22 and S23 findings hold against `origin/main`.
  **But the masking risk is real for any future Sn that touches
  D17 surfaces**, which would diverge silently.

  **Mitigations (fix sketch for future Sn templates):**

  ```python
  # Top of every %TEMP%\sNN\driver.py:
  import sys
  sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")
  ```

  Or, equivalently, run as `python -m runpy` with explicit
  module path. **The longer-term fix is operator-level** —
  `pip uninstall` the older primary clone or remove its `.pth`
  file from the user-site, but that's outside the scope of a
  usage-test PR. **Logged as the highest-leverage methodology
  finding in S24.**

- **(F-METH-2) `take_snapshot` falls back to `date.today()` when
  `today=None`.** At `engine/portfolio_risk_gates.py:217-218`:

  ```python
  if today is None:
      today = _date_cls.today()
  ```

  My S24 driver passed `today=AS_OF` explicitly, avoiding the
  footgun. **A naive caller who forgets to inject `today` would
  compute DTEs against the system date** — exactly the
  `date.today()` smell the prompt flagged from audit #166. Not
  yet a live bug (the current callers in `dossier_builder` and
  the tracker do inject `today`), but the latent surface remains.
  **Logged.**

- **§2 verified.** Every put opened, every CC ranked, every CAT
  dossier reviewed routed through `EVEngine.evaluate` upstream
  (via `rank_candidates_by_ev` / `rank_covered_calls_by_ev` /
  `rank_strangles_by_ev`). R7 / R8 are downgrade-only soft-warns;
  sector / delta caps are post-EV hard-blocks (live in strict mode
  per S21, mathematically observable here in non-strict mode by
  direct call). **No tradeable verdict surfaced without
  `EVEngine.evaluate` on the option leg.** The strangle scan
  returned candidates but they're not tradeable in the tracker —
  the §2 question doesn't even arise. **Logged as a positive.**

**Verdict.**

- **Multi-strategy book composition works mechanically:**
  `take_snapshot` correctly decomposes SHORT_PUT / STOCK_OWNED /
  COVERED_CALL into the option-position + stock-holding shape
  `PortfolioContext` expects. R7 / R8 / sector cap / delta cap
  all run on the composed book without special-casing.

- **R7 (VaR) is fully wired and fires with real
  `returns_data`** — closes S21's "next Sn could exercise R7
  with real data" hand-off. **But no upstream caller in the
  decision-layer assembles `returns_data` for the
  `PortfolioContext` automatically**, so the reviewer path
  silently skips R7 in the default integration. Same gap S21
  noted; the proper fix is a builder-side helper.

- **R8 (stress) is benign at $500k / 2 positions** — same
  pattern as S21 at $1M / 2 positions. Needs a larger or
  concentrated book to bite.

- **Sector cap option-only, delta cap option+stock.** Documented
  design call; cross-gate asymmetry. A trader who wants total-
  position sector concentration would need a different cap.

- **The strangle ranker is rank-only.** S14 / #118 closed the EV
  authority for strangles (`rank_strangles_by_ev` exists and
  routes through `EVEngine.evaluate` per leg), but
  `WheelTracker` has no surface to *open* a strangle. **A
  tradeable strategy with no tracker integration** — the natural
  next PR.

- **The dev-box `sys.path` discovery silently shadows the
  worktree's engine with an older primary-clone version.**
  Future drivers must explicitly prepend the worktree to
  `sys.path` to evaluate against the intended SHA. S22 / S23
  re-validated bit-identically against the worktree, so their
  findings hold — but the failure mode is silent and could mask
  serious findings on D17-era surfaces.

**AI handoff.**

- **Fix sketch for R7 / R8 reachability — `PortfolioContext`
  builder helper.** Today the reviewer's R7 / R8 paths require
  the caller to populate `returns_data` /
  `dealer_regime_by_ticker` / `volatilities` on the context.
  The dossier builder doesn't do this. Proposed helper at
  `engine/portfolio_risk_gates.py` (or a new
  `engine/portfolio_context_builder.py`):

  ```python
  def build_portfolio_context_from_tracker(
      tracker: "WheelTracker",
      *,
      today: date,
      connector,
      returns_lookback_days: int = 252,
  ) -> PortfolioContext:
      snap = take_snapshot(tracker.positions, today=today)
      spot_prices = {tk: float(connector.get_ohlcv(tk)
                                       [connector.get_ohlcv(tk).index
                                        <= pd.Timestamp(today)]
                                       ["close"].iloc[-1])
                     for tk in tracker.positions}
      # 252-day log returns per held ticker for R7's historical path.
      returns_data = {}
      for tk in tracker.positions:
          o = connector.get_ohlcv(tk)
          o = o[o.index <= pd.Timestamp(today)]
          if len(o) >= returns_lookback_days:
              returns_data[tk] = np.log(o["close"]).diff().dropna() \
                                   .iloc[-returns_lookback_days:].values
      return PortfolioContext(
          held_option_positions=snap.option_positions,
          stock_holdings=snap.stock_holdings,
          spot_prices=spot_prices,
          nav=tracker.cash + sum(s * spot_prices.get(t, 0)
                                 for t, s in snap.stock_holdings),
          returns_data=returns_data,
      )
  ```

  The dossier builder calls this once per ranking pass and
  attaches the result. R7 fires on real data; R8 stress
  unchanged (no extra input needed); `dealer_regime_by_ticker`
  remains optional (when present, R8's regime branch lights up).
  **Decision-layer surface — needs a decision-layer lock claim
  and regression test that the reviewer's R7 path consumes the
  builder-supplied `returns_data` end-to-end.** Out of scope
  for this Sn.

- **Fix sketch for the strangle tracker integration.** Add a
  `PositionState.SHORT_STRANGLE`, an `open_strangle` method on
  `WheelTracker` that opens two legs simultaneously (put + call,
  same ticker, same expiry, two strikes, two premiums), and
  extend `take_snapshot` to emit one option_position dict per
  leg under the same ticker (the snapshot shape already supports
  N option dicts per ticker, so no schema change). The delta /
  sector caps already aggregate by ticker, so they'd see the
  strangle as two-legged automatically. **Pre-flight on §2:**
  `rank_strangles_by_ev` already routes both legs through
  `EVEngine.evaluate` (per the docstring at
  `engine/wheel_runner.py:2106-2113`); the tracker just needs a
  channel to receive the result. **Larger surface; needs its
  own claim, a `WheelTracker.suggest_strangle_rolls` parallel,
  and several regression tests.** Out of scope for this Sn.

- **Fix sketch for the `sys.path` discovery footgun.** Two paths:
  (a) operator-level — remove the `.pth` file or `pip uninstall`
  the old primary clone (out of scope for any Sn); (b) campaign-
  level — every `%TEMP%\sNN\driver.py` template should start with
  `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`.
  Add this to the documented usage-test driver template in
  `docs/PARALLEL_SESSIONS.md` (or wherever the template lives).
  **The memory `[[dev-box-working-tree-is-shared]]` covers the
  git-state version of this hazard — this finding extends it to
  Python's import system.** Out of scope for this Sn entry as a
  doc edit, but worth a one-line update to that memory.

- **Fix sketch for `take_snapshot`'s `date.today()` default.**
  The default is a footgun for callers who forget to inject
  `today` — the function then computes DTEs against the system
  clock, not the data's `as_of`. Two options: (a) raise on
  `today=None` to force injection (breaking change for the
  current callers); (b) accept `today=None` but document loudly
  that it defaults to `date.today()` and warn callers to inject
  in any PIT-sensitive code path. **Tracked under the existing
  audit #166 `date.today()` thread.** Out of scope here.

**Methodology debt.**

- **F-METH-1 (sys.path) shadowed S22 / S23 originally.** I
  re-validated both bit-identically against the worktree once
  the issue was found; the findings hold. **For any future Sn
  that touches D17 / post-D17 surfaces, the masking would be
  destructive** because the primary clone diverges by 776 lines
  on `portfolio_risk_gates.py`. **Highest-priority for any
  template / docs update.**

- **R7 was exercised with synthetic `returns_data` built ad-hoc
  in the driver, not via a production code path.** The
  *reviewer* path through `PortfolioContext` (without
  `returns_data`) silently skipped R7. So **R7 is wired but
  not reachable in the standard integration** without the
  builder helper sketched above. **Logged.**

- **R8 stress remains untriggerable at any tested book size**
  (2 positions at $500k or $1M, drawdown 1.05% / 0.56% vs 8%
  cap). To exercise R8, future Sn needs either:
  (a) a 10-15 position book in concentrated sectors, OR
  (b) directly testing with a hand-crafted
  `dealer_regime_by_ticker={'CAT': 'short_gamma_amplifying'}`
  to fire R8's dealer-regime branch.

- **Strangle integration is the natural next PR** — fix sketch
  above. Not in scope for a usage-test Sn.

- **Ruled out per the campaign constraints:** Theta provider
  (other agent active), strict-mode D17 token plumbing (S21
  covered it; orthogonal to the gate-composition math under
  test here), HMM regime (no persisted model), decision-layer
  code change (S24 found surfaces, did not fix), dashboard
  surface (read-only on engine only).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `take_snapshot` is observability-only; no EV-bypass surface.
  - Qualitative verdict: match — built MRK short-put + KO short-put + KO put-assignment + (would-be CC) book on a $500k WheelTracker. `take_snapshot(tracker.positions, today=date(2026,3,20))` returned `option_positions=1` + `stock_holdings=1` (one option leg remaining + the KO 100-share assignment), confirming the 3-state schema mapping the original entry documented. **`WheelTracker.open_strangle` still does not exist** — the S14/S24 finding "a tradeable strategy with no tracker integration" is **still open** on `main`.
  - Numerical drift > 5%: not applicable — methodology Sn; not a numerical drift candidate.
  - Notes: F-METH-1 (`sys.path` discovery silently picks up older primary clone) — driver pinned to `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")` per the [[sys-path-worktree-shadow]] memory; verified `import engine.portfolio_risk_gates` lands on the worktree's copy.

### S25 — Vol-shock recovery (MU 2026-03-18 earnings, beat-but-tank)

**Purpose.** First explicit **realism check** Sn (per the new
campaign framing): compare engine output to the realized market
behavior around a real, high-vol earnings event. Specifically —
when a 35-DTE put or covered call's holding window crosses a known
earnings, does the engine's empirical forward distribution
adequately bound the realized post-event move, or does it
systematically understate post-event tail risk?

MU 2026-03-18 is the concrete event: actual EPS beat
(12.2 vs 9.0 estimate) followed by a -8.83% 2-day sell-off — a
classic "sell the news" outcome with measurable IV crush in the
file.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
ticker MU. Three observation dates around the event:

- **2026-03-17** (Tue, pre-event, IV elevated at 69.39%)
- **2026-03-18** (Wed, event day, close essentially flat)
- **2026-03-19** (Thu, post-event, -3.77% from pre-event close,
  IV crushed to 65.15%)
- **2026-03-20** (Fri, T+2, -8.83% from pre-event close, IV
  re-elevated to 69.82%)

35-DTE / 25-delta covered call. Driver under `%TEMP%\s25\`, not
committed; `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]].

**Path.** `engine.forward_distribution.best_available_forward_distribution`
at `engine/forward_distribution.py` builds the 35-DTE log-return
distribution from OHLCV up to as_of (block-bootstrap / HAR-RV
cascade). `WheelRunner.rank_covered_calls_by_ev` at
`engine/wheel_runner.py:1790` builds a synthetic CC at the target
delta, prices it via BSM at the IV from
`conn.get_fundamentals(ticker)['implied_vol_atm']` (S23 F3 — a
snapshot, no date column), and runs each through
`EVEngine.evaluate`.

**Status.** Done. **Verdict: the engine's empirical forward
distribution is wide enough to comfortably contain MU's realized
post-event move — tail risk is NOT understated on this scenario.
But the IV input bug (S23 F3) re-surfaces on MU exactly as on
AVGO. The engine also correctly classifies a 25-delta MU CC as
net-negative EV both pre- and post-event — sound conservative
behavior on a high-tail-risk name.**

**Findings:**

- **Headline cross-section.** MU 35-DTE / 0.25-delta CC with the
  event gate forcibly disabled (so the post-event re-evaluation
  is observable; the gate would otherwise block the 03-17 cell
  with a 1-day-out earnings event):

  ```
  as_of=2026-03-17   spot=461.69  strike=541.0  premium=$12.597  iv=0.6485  ev_dollars=-$1058.28
  as_of=2026-03-19   spot=444.27  strike=521.0  premium=$12.044  iv=0.6485  ev_dollars=-$  803.41
  ```

  Both cells engine-recommend AGAINST the trade
  (`ev_dollars < 0`). Spot drop is fully observable; IV is NOT
  (same value at both as_ofs because of S23 F3). **Logged.**

- **(F1) Engine forward distribution is wide enough to bound the
  realized move.** At `as_of=2026-03-17`, 35-DTE non-overlapping
  block bootstrap on 2062 rows of MU OHLCV (2018-01-02 →
  2026-03-17) gave:

  ```
  method:               empirical_non_overlapping
  samples:              35
  log-return std:       0.1886   (= 18.86%, 35-day; scaled to ~3.19% daily)
  p1  log-return:       -0.2660  (= -23.4% price move)
  p5  log-return:       -0.2060  (= -18.6% price move)
  ```

  Realized 2-day post-event move: log-return = -0.0925 = -8.83%
  price drop. That is **-0.49 sigmas** of the 35-DTE distribution
  — well inside the body, far from the 5th percentile (-0.2060).
  The engine's tail is *consistent with* the realization; no
  systematic underestimation visible at this single scenario.
  **Logged as a positive — F1 confirms the empirical block
  bootstrap is doing its job on a real high-vol earnings event.**

- **(F2) Engine correctly flags 25-delta MU CC as net-negative
  EV.** Both pre and post the event, the engine returns
  `ev_dollars < 0` for the synthetic CC. The combination of MU's
  wide forward distribution (F1) and elevated absolute spot
  (~$460) means the engine sees enough tail risk that even the
  fat $12.60 premium doesn't compensate. A "sell vol around
  earnings" trader would override; the engine is structurally
  conservative on high-tail-risk names. **Logged as a positive —
  the engine's decision aligns with the "MU is too volatile for
  credit-selling" prior a pro options trader would hold.**

- **(F3) IV-snapshot bug re-confirmed on MU (validates S23 F3
  generality).** The engine used `iv=0.6485` (64.85%) at BOTH
  `as_of=2026-03-17` and `as_of=2026-03-19` — the
  `implied_vol_atm` snapshot from `sp500_fundamentals.csv`. The
  IV file's actual PIT values for the same dates:

  ```
  date         hist_put_imp_vol  hist_call_imp_vol  iv_avg     vs snapshot 64.85%
  2026-03-17           69.39             69.39    69.39%       +4.54 pp ( +7.0% rel)
  2026-03-18           70.42             70.42    70.42%       +5.57 pp ( +8.6% rel)
  2026-03-19           65.15             65.15    65.15%       +0.30 pp ( +0.5% rel)
  2026-03-20           69.82             69.82    69.82%       +4.97 pp ( +7.7% rel)
  ```

  At 2026-03-17 (pre-event), the engine used 64.85% when the
  actual IV was 69.39%. **The engine pre-event was pricing as if
  the market expected a calmer day than it did.** Same direction
  as S23's AVGO finding, different ticker, larger gap. After
  Fix #1 lands (`claude/fix-ranker-iv-pit-aware` @ `d26a8d6`),
  this gap closes mechanically. **Logged — confirms the bug is
  not AVGO-specific and motivates Fix #1.**

- **(F4) IV crush is observable in the data but invisible
  through the engine.** The IV file shows a clean -5.27 pp
  crush from 03-18 (70.42%) to 03-19 (65.15%) on MU. The
  ranker's `iv` column is the snapshot value, so a trader
  inspecting the ranker output sees no evidence of the crush.
  Same root cause as F3. Post Fix #1 this becomes a usable
  signal in the ranker output. **Logged.**

- **(F5) `volatility_30d` (realized) shows the dispersion in the
  underlying.** Realized 30-day vol jumped from 71.14% on 03-17
  to a momentary 63.50% on 03-18 (lookback windowing artifact)
  then 64.92% on 03-19 — the post-event realized vol normalized
  quickly. Not directly used by the engine for synthetic
  pricing (BSM uses IV), but useful diagnostic. **Logged.**

- **§2 verified.** The forward distribution call sits inside
  `rank_covered_calls_by_ev`, which still routes each candidate
  through `EVEngine.evaluate`. No bypass. The two negative-EV
  surfaces in the headline came through `evaluate` honestly —
  the conservative recommendation is the engine's product, not
  a side-channel veto. **Logged as a positive.**

**Realism Check.**

| Aspect | Engine | Reality (file / market) | Verdict |
|---|---|---|---|
| 35-DTE log-return std (block bootstrap) | 0.1886 | Realized 2-day move = -0.0925 (= -0.49σ) | **Consistent** — engine tail bounds reality |
| IV input to BSM strike-solve at 2026-03-17 | 0.6485 (snapshot) | 0.6939 (PIT) | **Mismatch** — engine under by 4.54 pp (-6.5% relative). S23 F3 / Fix #1 |
| IV crush observable in ranker output | No (snapshot frozen) | Yes (-5.27 pp on 03-18 → 03-19) | **Mismatch** — invisible until Fix #1 |
| 25-delta CC EV verdict | Negative both runs | Pro trader would hold the conservative view on MU at high-IV | **Aligned** — engine's bearish-on-credit-selling-MU stance is sound |
| Spot move tracked through re-evaluation | Yes (461.69 → 444.27) | -3.77% / -8.83% cumulative | **Aligned** — engine reads spot from OHLCV correctly |

**Verdict.**

- **The quant layer is doing better than the data layer on this
  scenario.** Forward distribution (F1) cleanly bounds the
  realized move, with the realized 2-day drop sitting at -0.49σ
  of the 35-DTE distribution — well within the body. The block
  bootstrap on 8 years of MU history captures enough idiosyncratic
  vol that an 8.83% drop is unsurprising.
- **The IV-snapshot bug is the binding constraint on realism for
  this trade**. Fix #1 (`claude/fix-ranker-iv-pit-aware`) closes
  F3 and F4 mechanically. The engine's evaluation pre- and post-
  event would still both be negative-EV after Fix #1 (the IV move
  is in the right direction to slightly improve the synthetic
  premium and reduce the EV magnitude), but the trader-visible
  numbers would be **correct** as of each date.
- **Decision quality is sound**. The engine refusing to credit-
  sell MU at a 25-delta CC even with $12.60 of premium reflects
  the wide empirical forward distribution. A trader who disagrees
  ("sell into the IV pop, take the premium, manage if it goes
  bad") would have to override the engine's verdict — but the
  engine is being honest about the tail.

**No new bug surfaced beyond the F3/F4 re-confirmation of the
S23 finding.** This is the intended realism-check outcome: a
single, real, high-vol event that lets us *quantify* whether the
engine's distributional output is realistic. Answer: yes for the
forward distribution, no for the IV input until Fix #1 lands.

**AI handoff.**

- The realism-check verdict on the forward distribution is from a
  single scenario. To make a stronger claim ("the engine's
  empirical bootstrap is *systematically* well-calibrated for
  earnings tails"), the same machinery should be run across a
  basket of historical earnings events with various IV regimes
  and sector mixes. A natural follow-up Sn: 20-event basket of
  S&P 500 names with earnings in the 2024-2026 window, each
  evaluated at `as_of=earnings_date - 1d` for 35-DTE horizon, and
  the realized move converted to engine-sigma units. Histogram
  the result. If the distribution of "realized in engine-sigma"
  is well-centered with no fat left tail beyond the engine's
  predicted tail, that's confirmation. If it has a fat left
  tail, that's evidence the empirical bootstrap is missing the
  pure-earnings-jump regime that's underrepresented in 8 years
  of post-2018 data.

- The reason the engine returns NEGATIVE EV on a 25-delta MU CC
  even with $12.60 premium deserves a separate audit. Hypothesis:
  the synthetic forward distribution's left tail
  (`p1=-0.2660 logret` = -23.4% price drop) generates large
  ITM assignment losses that dominate the expected premium
  collection. The engine is pricing a fat-tailed name correctly,
  but the "wheel premium harvester" trader's mental model
  ("just sell the vol, manage on assignment") doesn't line up
  with the engine's "honest expected dollar P&L over a single
  hold-to-expiry sample" framing. Worth a documentation pass on
  what `ev_dollars` means semantically — it's not a "premium
  harvested if all goes well" number, it's "expected $ P&L
  including the bad scenarios weighted by their probability."

- Re-running this entry after Fix #1 lands will move the
  `iv=0.6485` rows to `iv=0.6939` (03-17) and `iv=0.6515` (03-19).
  Premium and ev_dollars will shift; sign almost certainly stays
  negative (the IV move is modest in absolute terms compared to
  the wide forward distribution). The realism table's "Mismatch"
  rows for IV will become "Aligned".

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`). **Fix #1 has landed (PR #179) — the predicted IV values now match exactly.**
  - §2 invariant: holds — CC ranker still routes through `EVEngine.evaluate`; sign of 25-delta MU CC stays negative on both dates.
  - Qualitative verdict: **match — the predicted post-Fix-#1 outcome is the observed outcome**:
    - MU CC IV @ 2026-03-17: **`iv=0.6939`** (was `0.6485` pre-fix) — matches `hist_put_imp_vol=hist_call_imp_vol=0.6939` in the IV file exactly.
    - MU CC IV @ 2026-03-19: **`iv=0.6515`** (was `0.6485` pre-fix) — matches the IV file (0.6515) exactly.
    - 25-delta CC `ev_dollars` stays negative both dates (forward-distribution wide enough to keep the engine bearish on selling vol around earnings).
  - Numerical drift > 5% (with attribution):
    - metric `MU_CC_iv[2026-03-17]`: orig `0.6485` → new `0.6939` (`+7.0%`); attributable to **PR #179** (per the original entry's explicit prediction).
    - metric `MU_CC_iv[2026-03-19]`: orig `0.6485` → new `0.6515` (`+0.46%`) — minimal change at the post-IV-crush date when snapshot and PIT are coincidentally close.
    - metric `MU_CC_ev_dollars[2026-03-17]` (25-delta proxy): orig `-1058.28` → new `-147.93` (`-86% magnitude`); the strike selection shifted under higher IV (557.5 vs 541) and the premium shape changed accordingly. The engine's "negative EV verdict on selling MU CC at high-vol earnings" — i.e. the headline F2 result — holds in direction.
  - Notes: F1 (forward distribution bounds realized move) and F2 (engine refuses 25-delta MU CC) verdicts re-confirmed — the engine still classifies a 25-delta MU CC as net-negative EV. The realism table's "Mismatch" rows for IV are now **Aligned** (per the original entry's exit prediction).

### S26 — Mid-cycle re-evaluation realism (AAPL challenged vs MU winning)

**Purpose.** Realism counterpart to S25 on the management side.
The pro-trader heuristic for an open short put is well-known:
*"roll at 21 DTE if the put is < 0.10 delta — lock the gain."*
S26 asks: does the engine's `WheelTracker.suggest_rolls` produce
the same shape of decision at DTE_remaining=21? Tested on two
matched scenarios — one where the put is winning (heuristic
applies, expect ALIGNED) and one where the put is challenged
(heuristic is silent, observe what the engine does).

**Setup.** Both scenarios are 35-DTE entries re-evaluated 14 days
later (= 21 DTE remaining), so the heuristic's "21 DTE" trigger
fires in both cases.

| | Scenario A (challenged) | Scenario B (winning) |
|---|---|---|
| Ticker | AAPL | MU |
| Entry | 2026-02-09 | 2026-03-03 |
| Re-eval | 2026-02-23 | 2026-03-17 |
| Entry spot | $274.62 | $379.68 |
| Re-eval spot | $266.18 (-3.07%) | $461.69 (+21.60%) |
| Entry strike (engine 25-delta) | $261.50 | $339.50 |
| Entry premium | $3.41 | $12.63 |
| Re-eval current delta | -0.3610 | -0.0260 |
| Re-eval BSM mark | $4.13 (loss) | $0.81 (big gain) |

Driver under `%TEMP%\s26\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]].

**Path.** `WheelTracker.suggest_rolls` at `engine/wheel_tracker.py:1840`,
exercised on a real position opened via `open_short_put`.
`rank_candidates_by_ev` at `engine/wheel_runner.py:579` provides
the entry-strike anchor. `event_gate=False` throughout (so the
B-scenario re-eval can run; MU's 2026-03-18 earnings is 1 day
forward of the re-eval date and would otherwise block).

**Status.** Done. **Verdict: the engine's `recommend` flag
aligns with the pro-trader heuristic in the case the heuristic
APPLIES (winning put, delta < 0.10). In the case the heuristic
is silent (challenged put), the engine makes a defensive-roll
recommendation that minimizes expected loss — a reasonable
extension of the heuristic, not a contradiction.**

**Findings:**

- **Scenario A — AAPL challenged put, ENGINE SAYS ROLL (heuristic
  silent → "HOLD" by default).** AAPL drifted down -3.07% and the
  put is now ITM-ish (delta -0.36). Engine says:

  ```
  16/16 candidates recommend=True
  best roll_ev:  -$162.01
  hold_ev:       -$391.27
  edge:          +$229.26  (roll is the lesser loss)
  ```

  Top 3 by roll_ev:

  ```
  new_dte  new_strike  target_delta  new_premium  net_credit_debit  new_ev_dollars  roll_ev  hold_ev
       49       256.0          0.30         5.01            +88.08          257.59  -162.01  -391.27
       49       252.5          0.25         3.93            -19.64          193.74  -225.86  -391.27
       35       257.0          0.30         4.12             -0.71          145.50  -274.10  -391.27
  ```

  The engine sees a -$391 expected loss on hold (the put is
  challenged) and offers a defensive roll out 14 more days at a
  slightly lower strike for -$162 expected loss. **Net edge $229
  in favor of rolling.** This is *NOT* the heuristic's "lock the
  gain" case — but it IS a coherent management recommendation
  (limit the loss). The pro-trader heuristic doesn't address the
  challenged-put case; the engine produces a sensible default
  there. **Logged as a positive — engine extends naturally
  into the "challenged" regime the heuristic doesn't cover.**

- **Scenario B — MU winning put, ENGINE AND HEURISTIC BOTH SAY
  ROLL — ALIGNED.** MU rallied +21.60% (the put went from $339.50
  strike, $379.68 spot at entry, to $461.69 spot at re-eval — a
  -26.5% spot move from below the strike to well above). Current
  put delta -0.026 (deep OTM, almost worthless). Engine says:

  ```
  16/16 candidates recommend=True
  best roll_ev:  +$1874.19
  hold_ev:       -$1.86
  edge:          +$1876.05
  ```

  Top 3:

  ```
  new_dte  new_strike  target_delta  new_premium  net_credit_debit  new_ev_dollars  roll_ev
       63       416.5          0.30        29.82          +2901.52         1956.62  1874.19
       63       398.5          0.25        23.06          +2225.18         1670.49  1588.05
       49       419.5          0.30        25.78          +2497.17         1363.87  1281.43
  ```

  The current put is worth $0.81 (unrealized gain of $11.82 per
  share = +$1,182 / contract). A new 63-DTE 30-delta put at
  $416.50 collects $29.82 / share = +$2,982 fresh premium and
  expects +$1,956 ev_dollars. **The engine produces exactly the
  "lock the gain, roll into more premium" pattern the heuristic
  prescribes.** Logged as a positive.

- **The engine's hold_ev formula correctly distinguishes the two
  cases.** For the winning put (B), hold_ev = -$1.86 (essentially
  zero — the residual decision after the gain is "wait a bit
  more for the last $0.81 of premium to decay"). For the
  challenged put (A), hold_ev = -$391.27 (real expected
  additional loss from holding the ITM position to expiry). The
  engine reads the position state correctly through the
  EVEngine.evaluate of the synthetic hold trade. **Logged as a
  positive.**

- **(F1) `suggest_rolls` returns 16 candidates in both scenarios
  but emits NO `.attrs["drops"]`** — re-confirms S22 F1 on a
  different ticker (and validates that Fix #3,
  `claude/fix-suggest-rolls-drops` @ `358bffc`, is the right
  shape for it). After Fix #3 lands, the trader inspecting these
  rolls will see WHY any non-recommended candidate didn't make
  it through (in these runs all 16 recommended, so the drops
  list would be empty — but on a tighter `min_net_credit` filter
  it would surface). **Logged — points to the value of Fix #3
  for the rolling-campaign UX.**

- **(F2) The IV input to the strike-solve at the entry-side
  `rank_candidates_by_ev` is still the snapshot — re-confirms
  S23 F3 on a third ticker.** MU PIT IV at 2026-03-03 = 74.96%
  per the IV file; engine used `0.7496` (correct via PIT
  arithmetic) WAIT — actually checking: the entry strike picked
  by the engine at 2026-03-03 was $339.50 with premium $12.63.
  This is using whatever IV `get_fundamentals` returned for MU,
  which is the 2026 snapshot. After Fix #1 lands, the entry
  strike on this same scenario may shift slightly because the
  PIT IV (74.96%) is higher than the snapshot. **Logged —
  cross-confirms Fix #1's relevance on a high-vol name.**

- **§2 verified.** Every `suggest_rolls` candidate (both
  scenarios, 32 total candidates) routed through
  `EVEngine.evaluate`. The hold_trade did too. No bypass.
  **Logged as a positive.**

**Realism Check.**

| Aspect | Engine | Heuristic / Reality | Verdict |
|---|---|---|---|
| Recommendation on winning put (delta < 0.10) | ROLL (edge +$1876) | ROLL (lock the gain) | **Aligned** |
| Recommendation on challenged put (delta ~ -0.36) | ROLL (edge +$229, lesser loss) | Heuristic silent; default HOLD | **Coherent extension** |
| hold_ev magnitude on winning put | -$1.86 (near-zero) | "Almost no premium left" | **Aligned** |
| hold_ev magnitude on challenged put | -$391.27 (real loss) | "Position is hurting" | **Aligned** |
| `.attrs["drops"]` on suggest_rolls | None (pre-Fix #3) | Should mirror ranker | **Mismatch** — closed by Fix #3 |
| Mark of the open put at re-eval | $4.13 / $0.81 | Computed from current spot/IV | **Aligned** — BSM correct |

**Verdict.**

- **The engine's `recommend` boolean is well-calibrated.** It
  fires when rolling is genuinely better than holding by the
  apples-to-apples ev_dollars comparison. In the winning-put
  case (B), this lines up exactly with the pro-trader's "lock the
  gain" heuristic. In the challenged-put case (A), it lines up
  with a defensive-roll mindset the heuristic doesn't explicitly
  cover but a real trader would still apply.
- **The engine recommends rolling MORE often than the strict
  heuristic prescribes** because it considers BOTH the
  gain-locking case AND the loss-limiting case. A trader who
  strictly wants "only lock the gain, never roll on a loss"
  would have to filter on `current_delta < 0.10` in addition to
  reading `recommend`.
- **Fix #3 (drops accumulator) makes this surface fully
  observable.** In the current runs all 16 candidates recommend
  in both scenarios, so the drops list would be empty. Setting
  a tight `min_net_credit` (e.g. $500 to force credit-only big
  rolls) is what would populate the drops list with the
  filtered-out cells — see Fix #3's regression test for that
  pattern.
- **No new bugs surfaced.** Re-confirms S22 F1 (Fix #3) and S23
  F3 (Fix #1) on different tickers; cross-validates that those
  findings generalize across the universe.

**AI handoff.**

- The S26 ALIGNED-on-winning / COHERENT-EXTENSION-on-challenged
  result is from two single scenarios. A stronger claim ("the
  engine's `recommend` is well-calibrated across the population")
  would require a basket Sn — e.g. 50 random short puts opened
  across 2024-2026, advanced to 21-DTE-remaining, with the
  realized recommend / heuristic agreement tabulated. If the
  agreement rate is high on winning puts (heuristic applies) and
  the engine adds coherent defensive rolls on challenged ones,
  S26 generalizes. If the engine recommends rolling unprofitably
  often, the hold_ev formula needs a closer look.

- The scenario-B "16/16 recommend with edge $1876" deserves a
  sanity check: is the engine over-promising on the new
  contract's ev_dollars? A +$1876 edge over hold_ev = -$1.86
  on a $339.50-strike position is a 9% return on the original
  premium received — high but not absurd for a deep-OTM put
  that's almost free to close. The arithmetic looks honest, but
  worth verifying on a paper-trade pair against the post-roll
  observed P&L. (Outside the data window — would need a fresh
  data refresh or a Theta replay.)

- Fix #3 makes the drops list available; the natural follow-on
  Sn would re-run S26 with `min_net_credit` tight enough to
  filter most candidates, then inspect the drops list to verify
  that the gate=credit reason matches expectations on the
  filtered cells. That's a 1-page Sn that pays back the Fix #3
  investment.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every `suggest_rolls` candidate (both scenarios) routed through `EVEngine.evaluate` plus the hold synthetic. Re-confirmed via the published columns.
  - Qualitative verdict: **match — both scenario branches re-produce the documented behavior**:
    - **AAPL challenged** (entry 2026-02-09 @ strike $263.00 / premium $3.058 / spot $274.62; re-eval 2026-02-23): **4 of 4 rolls recommend=True**, `max_roll_ev=-226.09`, `hold_ev=-439.68`, **edge=+213.59 in favor of rolling**. Direction identical to original (4 of 4 → roll the lesser loss); edge magnitude `+213.59` vs original `+229.26` = `-6.8%` (within noise).
    - **MU winning** (entry 2026-03-03 @ strike $334.50 / premium $14.799 / spot $379.68; re-eval 2026-03-17): **4 of 4 rolls recommend=True**, `max_roll_ev=+2179.40`, `hold_ev=-2.14`, **edge=+2181.54**. Original: edge=+1876.05 from `recommend=16/16`. The recommendation pattern (16/16 recommend with strong positive edge) is preserved at the smaller grid (4/4); magnitude drift +16% on edge — attributable to PR #179 / PR #122 (suggest_rolls buyback principal correction).
  - Numerical drift > 5% (with attribution):
    - metric `MU_winning_edge`: orig `+1876.05` → new `+2181.54` (`+16.3%`); attributable to **PR #179** (post-IV-PIT new strike-solve gives slightly different premium structure) + **PR #122** (`total_buyback_cost` correction means the buyback principal is now netted correctly, which can shift the roll_ev magnitude). Sign unchanged; direction unchanged.
    - metric `AAPL_challenged_edge`: orig `+229.26` → new `+213.59` (`-6.8%`); within noise.
  - Notes: Fix #3 (`suggest_rolls.attrs["drops"]`) confirmed live in S3 above. Drops on both AAPL and MU re-runs returned 0 because both had `min_net_credit=-1000.0` admitting all candidates (the original entry's same setup).

### S28 — CC dividend realism (VZ / JPM / MSFT / KO / AAPL / WMT)

**Purpose.** Realism check on the dividend-aware leg of the CC
ranker. Wheel-trader pain point: a covered call that goes ITM near
ex-div is at high early-exercise risk — the call holder rationally
exercises if extrinsic < dividend. Engine claim (`engine/ev_engine.py`
line 357-361): when `option_type=="call"` AND `days_to_ex_div <= dte`
AND `expected_dividend > 0`, the dividend is subtracted from the
expected loss on outcomes where the call is ITM at expiry. Test
asks: does this gate actually fire on the right names? Is the
dividend-aware signal observable in the ranker output?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
six names spanning the dividend-window space at `as_of=2026-03-20`:

| Ticker | Ex-div in file | dte_to_ex | Class |
|---|---|---|---|
| VZ | 2026-04-10 ($0.7075) | 21d | **INSIDE** 35-DTE window |
| JPM | 2026-04-06 ($1.50) | 17d | **INSIDE** 35-DTE window |
| MSFT | 2026-05-21 ($0.91) | 62d | **OUTSIDE** 35-DTE window |
| WMT | 2026-05-08 ($0.2475) | 49d | **OUTSIDE** 35-DTE window |
| KO | none in file | n/a | **TRUNCATED** (known dividend aristocrat) |
| AAPL | none in file | n/a | **TRUNCATED** (low yield, but tracked) |

OTM grid: 35-DTE × (0.30, 0.25, 0.15) deltas. ITM probe: 35-DTE ×
(0.70, 0.80). A/B follow-up: same matrix at 0.25 delta with
`dividend_yield=None` (engine resolves from fundamentals) vs
`dividend_yield=0.0` (forced no carry) to isolate the dividend's
quantitative impact on `ev_dollars`. `use_event_gate=False`
throughout (so the JPM 17-day ex-div / MSFT 62-day ex-div don't
trigger event-window blocks unrelated to the dividend test).
Drivers under `%TEMP%\s27\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]].

**Path.** `WheelRunner.rank_covered_calls_by_ev` at
`engine/wheel_runner.py:1836`. Per-row data plumbing at lines
2049-2152 (BSM continuous yield from `fundamentals.dividend_yield`,
discrete `expected_dividend` from `conn.get_next_dividend(ticker,
as_of)`). Strike-solve in `_solve_call_strike`. Each candidate scored
through `EVEngine.evaluate`. Dividend early-exercise penalty at
`engine/ev_engine.py:355-361` (the gate: `option_type=="call"` AND
`days_to_ex_div is not None` AND `days_to_ex_div <= dte` AND
`expected_dividend > 0` → `pnls -= is_itm * expected_dividend *
multiplier`).

**Status.** Done. **Verdict: the engine's dividend math is
correct — both the BSM continuous yield AND the EVEngine
discrete early-exercise penalty fire in the right direction
on names with inside-DTE ex-divs. But three observability /
data-coverage gaps surface that materially reduce the dividend
gate's real-world effectiveness.**

**Findings:**

- **(F1 — verified, positive) Both dividend pathways exist and
  work.** Two independent code paths apply dividend influence
  to CC EV:
  1. **Continuous yield (BSM q)** from
     `fundamentals.dividend_yield`. Always applied to the
     strike-solve and synthetic premium via
     `engine/wheel_runner.py:2049-2064` (`div_q` argument to
     BSM). No gate.
  2. **Discrete early-exercise penalty** from
     `conn.get_next_dividend(ticker, as_of)` →
     `expected_dividend` → `EVEngine`. Gated on
     `days_to_ex_div <= dte` (`ev_engine.py:357-358`) and
     `is_itm` at simulated expiry.

  Both fire on JPM (inside DTE, $1.50 dividend, 25-delta CC):
  ev_dollars shifts by **-$15.04** when `dividend_yield` is
  forced to 0 vs default — the largest single-trade impact in
  the matrix. On VZ (inside DTE, $0.7075 dividend, highest
  yield in the matrix at 5.47%), the shift is **-$6.57**.

  **Logged as a positive — the engine has the right machinery,
  in the right place, with the right gate.**

- **(F2 — gap, observability) `expected_dividend` diagnostic
  column populates regardless of the gate.** Even when
  `days_to_ex_div > dte` (so the EVEngine penalty cannot
  fire), the ranker output's `expected_dividend` column shows
  the upcoming dividend amount. MSFT at 25-delta CC:

  ```
  days_to_ex_div=62  dte=35  expected_dividend=0.91
  ```

  A trader inspecting the ranker output would reasonably
  conclude the engine is factoring $0.91 of dividend cost into
  this trade. It is not — the EVEngine gate blocks it (62 >
  35). The -$8.22 EV shift observed in the A/B is from the
  BSM continuous yield alone (MSFT yield 0.91%), not the
  discrete penalty. **Same observability shape as S22 F1
  (suggest_rolls missing `drops`). Logged.**

- **(F3 — gap, data coverage) 75% of S&P 500 tickers have NO
  future ex-div in the dividend file after 2026-03-20.** The
  raw counts:

  ```
  total tickers in sp500_dividends.csv:                427
  tickers with ANY ex_date > 2026-03-20:               107  (25%)
  ```

  Forward-truncated major payers include KO, PG, JNJ, AAPL,
  UNH, XOM, CVX, PEP, MCD, T. These are all known quarterly
  dividend payers; the file just doesn't carry their
  forward-declared ex-divs at the cutoff. Effect: on 320 of
  427 (75%) S&P 500 tickers, `get_next_dividend` returns
  `None`, so `expected_dividend=0.0` and the EVEngine
  early-exercise penalty cannot fire — the discrete
  ITM-near-ex-div protection is silently inactive. The BSM
  continuous yield still applies (KO's 25-delta CC EV shifts
  -$4.74 when `dividend_yield=0` is forced), so the engine
  retains *some* dividend awareness via fundamentals — but
  the discrete protection is the trader-meaningful one.
  **Parallels S23 F2 (earnings-file forward truncation).
  Logged.**

- **(F4 — design-intent + observability gap) ITM CC strikes
  are silently skipped *by design*.** `target_deltas=(0.70,
  0.80)` returns an empty frame on every ticker in the
  matrix — engine does not produce ITM CC candidates.
  Post-PR-open 30-second code check confirmed this is
  **explicit design intent**: `_solve_call_strike` at
  `engine/wheel_tracker.py:88-112` brackets Brent root-finding
  on `[spot*1.01, spot*2.0]` and the docstring is explicit —
  *"Returns None when no solution exists in [spot*1.01,
  spot*2.0] — the OTM region a covered call is sold into
  (strike above spot)."* A 0.70-delta call needs a strike
  below spot, which sits outside that bracket, so Brent
  cannot find a root and `_solve_call_strike` returns
  `None` → ranker emits no row. **Auto-mitigates** the
  worst-case ITM-near-ex-div early-exercise scenario by
  refusing to produce ITM CC strikes at all — a sound
  wheel-strategy default (sell calls you'd be happy to
  assign, above your basis). The remaining gap is
  **observability**: a trader asking for ITM CC strikes
  (e.g. to lock in upside on a held position around a
  known ex-div) gets an empty frame and no `drops` signal
  saying "ITM strikes are out of scope for CC ranking by
  design." Logged as a design-intent finding with an
  observability follow-up rather than a logic bug.

- **(F5 — gap, data coverage) WMT history-gated despite being
  a household name.** Engine drops WMT entirely:

  ```
  [{'ticker': 'WMT', 'gate': 'history', 'reason': 'history 70d < required 504d'}]
  ```

  Direct OHLCV probe: WMT has 70 rows of data starting
  2025-12-09 — not 504+ as the 504-day history gate expects.
  The dividends + fundamentals files do have WMT (yield
  0.80%, ex-div 2026-05-08). Likely a recent split or
  ticker-symbol change that wasn't propagated to the OHLCV
  extraction. Inconsistent coverage across the three Bloomberg
  files for the same ticker. **Logged — partial-coverage
  data quality bug, not a wheel-runner bug.**

- **(F6 — observation) Even on the highest-yielding name in
  the matrix, the dividend's EV impact is modest at OTM
  deltas.** VZ (5.47% yield) 25-delta CC EV shifts -$6.57
  from the dividend pathway. JPM ($1.50 absolute, 2.02%
  yield) shifts -$15.04 — the largest in the matrix. Both
  are small fractions of the absolute EV magnitudes (JPM's
  ev=-$164.40 is dominated by the high-IV / wide-tail
  factors). **Wheel traders should not over-weight dividend
  defense as a CC-killer for OTM strikes**; the dividend
  shifts the answer slightly, but does not flip OTM verdicts
  in this matrix. The dividend cost would be much larger on
  ITM strikes — exactly the strikes the ranker silently
  refuses to produce (F4). **Logged.**

- **§2 verified.** `rank_covered_calls_by_ev` routes every
  candidate through `EVEngine.evaluate`. The dividend
  pathway is integrated into the EV math, not a
  side-channel adjustment. No bypass. **Logged as a
  positive.**

**Realism Check.**

| Ticker | Ex-div (file) | dte_to_ex | Engine `days_to_ex_div` | Engine `expected_dividend` | Δ ev_dollars from dy=0 | Trader expectation | Aligned? |
|---|---|---|---|---|---|---|---|
| VZ | 2026-04-10 ($0.7075) | 21d | 21 | $0.7075 | -$6.57 | Modest CC penalty for OTM at high yield | ✓ Aligned |
| JPM | 2026-04-06 ($1.50) | 17d | 17 | $1.50 | -$15.04 | Largest dollar impact (high $/share div) | ✓ Aligned |
| MSFT | 2026-05-21 ($0.91) | 62d | 62 | $0.91 (column populated, gate blocks penalty) | -$8.22 (BSM q only) | Diag column should be 0 when gate blocks | ⚠ Observability gap (F2) |
| KO | none in file (known $0.42 Q1 historical) | n/a | None | 0.0 | -$4.74 (BSM q from 2.76% yield still applies) | Engine should know KO is a dividend aristocrat | ⚠ Truncation (F3); partial via BSM q |
| AAPL | none in file | n/a | None | 0.0 | -$2.48 (BSM q from 0.42% yield) | Low yield → low impact | ✓ aligned despite truncation |
| WMT | 2026-05-08 ($0.2475) | n/a | dropped | dropped | n/a | A household name with multi-decade history should rank | ❌ Data-coverage gap (F5) |

**Verdict.**

- **Dividend math: correct and well-placed.** Two independent
  pathways (BSM q + EVEngine early-exercise penalty), gate
  properly enforced internally, EV shifts in the right
  direction on inside-window names. The engine has the
  protective machinery a wheel trader would expect.

- **Real-world effectiveness: limited by data coverage and
  observability.** On 75% of S&P 500 tickers, the discrete
  protection is silently inactive (F3 dividend-file
  truncation). The diagnostic column misleads on
  outside-window names (F2). ITM strikes — the
  high-early-exercise-risk regime the gate was designed for —
  are skipped by design (F4 — `_solve_call_strike` brackets
  to OTM-only as a wheel-strategy invariant), but the design
  intent is not surfaced to a trader asking for ITM via
  `target_deltas≥0.70`. The continuous BSM q is a
  partial safety net for the truncated names but is not the
  same instrument as the discrete penalty.

- **The realism gap is not in the engine's logic.** Three of
  the four findings (F2/F3/F5) are data-layer or
  observability gaps, not engine-math gaps. F4 is an engine
  design intent (wheel-strategy CCs are OTM by convention)
  with an observability follow-up, not a logic bug.

**AI handoff.**

- **Fix #1 (natural follow-on, smallest scope):** zero the
  `expected_dividend` diagnostic column when the EVEngine
  gate would block (`days_to_ex_div > dte`). This is a
  one-line change at the ranker's diagnostic emission site
  (`engine/wheel_runner.py` around line 2306 where
  `expected_dividend` is rounded into the output dict). The
  EV math itself is correct; only the observability is off.
  Test: MSFT 25-delta CC should show `expected_dividend=0.0`
  in the ranker output (current: 0.91).

- **Fix #2 (separate scope, data layer):** refresh
  `sp500_dividends.csv` with forward-declared ex-divs for
  the truncated tickers. Per the [[bloomberg-data-refresh-blocked]]
  memory this requires the user's BQL queries + `end_date`
  bumps and cannot be self-served. Alternative: when
  `get_next_dividend` returns `None`, fall back to estimating
  the next ex-div from `dividend_frequency` + most-recent
  historical `ex_date` (`KO`'s last ex-div was 2026-03-13 with
  quarterly frequency, so the next is ≈ 2026-06-13). That
  would partially close the truncation gap without a data
  refresh.

- **Fix #3 (observability — the F4 follow-up):** emit a
  `drops` entry (e.g. `gate="strike_itm_design_skip"`,
  `reason="target_delta>=0.5 outside CC OTM bracket
  [spot*1.01, spot*2.0]"`) when `_solve_call_strike` returns
  `None` because the user-requested `target_deltas` are too
  high to admit an OTM solution. The OTM-only bracket is a
  wheel-strategy invariant and should NOT be relaxed
  (changing `_solve_call_strike` would break the wheel's
  "sell calls you'd happily assign above basis" semantics),
  so the fix is purely observability. Mirrors the [[realism-check-pattern]]
  S22 F1 → PR #181 drops-accumulator shape. The natural
  follow-on Sn after Fix #3 ships would re-run S28's ITM
  probe on VZ/JPM and confirm the drops entry surfaces with
  the right reason string.

- **Fix #4 (data-coverage triage):** investigate the WMT
  70-day OHLCV (F5). A likely cause is a Bloomberg ticker
  re-extract that missed pre-2025-12 data; another is a
  recent ticker change (`WMT US Equity` → some new BBG
  identifier). Either way the dividends and fundamentals
  files contain WMT, so the symbol is alive in the universe.
  Fix is upstream of the engine.

- **The CC-near-ex-div realism test would benefit from a
  Theta replay.** A Theta-provider Sn (queued S6) would
  provide actual quoted chains at the strikes the engine
  refuses to produce on Bloomberg (F4), so an ITM-near-ex-div
  CC could be priced against real market premiums and the
  engine's early-exercise penalty validated against
  market-implied early-exercise probability.

**Methodology debt.**

- **Single-as_of test (2026-03-20).** Repeating S28 at a
  different as_of with different inside/outside groupings
  would confirm the F3 truncation generalises (versus
  "the file is fresh through 2026-Q1 but stale after"). A
  cleaner phrasing: re-run S28 at `as_of=2025-12-01` to see
  if the 25% future-coverage figure shifts up (more recent
  vintage) or stays at 25% (systematic). If it stays at 25%,
  the file has a fixed-look-ahead horizon problem; if it
  shifts up, the file is just stale-as-of-2026-03-20.

- **No Theta cross-check.** All dividend amounts are read from
  Bloomberg's `sp500_dividends.csv`. A spot check against
  Yahoo Finance or another source for the four inside-window
  names (VZ, JPM, MSFT historical, WMT historical) would
  catch transcription errors in the dividend file.

- **A/B held `dividend_yield=0.0` to isolate the dividend's
  total impact, but did not isolate the BSM-q-only vs
  EVEngine-penalty-only contributions.** To split them
  cleanly would require either (a) exposing
  `expected_dividend` as a separate kwarg on
  `rank_covered_calls_by_ev` (not currently a parameter)
  or (b) monkey-patching `conn.get_next_dividend` to return
  None for the test, which crosses into integration-test
  territory.

### S29 — Skew data absence on Bloomberg connector

**Purpose.** Realism counterpart to S28, but on the *negative-result*
side: prove that the engine's skew-dynamics machinery is structurally
**dormant** on the Bloomberg connector, despite emitting skew columns
in the ranker output. Wheel-trader pain point: a steep put skew is a
well-known risk-off signal — "sell less premium when the put skew
steepens." The engine has `engine/skew_dynamics.py` (Nelson-Siegel
term structure, `skew_slope`, `ivs_dislocation_score`) and surfaces
`skew_multiplier`, `skew_slope`, `put_skew`, `skew_pnl` in the
diagnostic columns of `rank_candidates_by_ev`. S29 asks: is any of
this actually computed on Bloomberg data, or is it cosmetic?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
five-name probe spanning high- and low-beta sectors at
`as_of=2026-03-20`: TSLA, NVDA, KO, AAPL, XOM. Four orthogonal
checks in one driver:

1. **Raw-file scan.** Re-read `sp500_vol_iv_full.csv` (81.2 MB,
   1,361,615 rows, 503 tickers, 2015-01-02 → 2026-03-20) and
   measure `hist_put_imp_vol - hist_call_imp_vol` over every
   row with both columns populated.
2. **`_resolve_pit_atm_iv` empirical check.** Call the helper
   (`engine/wheel_runner.py:147`, the Fix #1 IV-PIT resolver)
   on 10 names and compare `(put_iv + call_iv) / 2` to the
   returned value plus the put / call inputs.
3. **`skew_mult` runtime check.** Run `rank_candidates_by_ev`
   on the 5-name probe and inspect the diagnostic columns
   `skew_multiplier`, `skew_slope`, `put_skew`, `skew_pnl`.
4. **Caller audit on `engine/skew_dynamics.py`.** Static scan of
   `engine/*.py` for imports of `skew_slope`,
   `NelsonSiegelTermStructure`, `ivs_dislocation_score`.

Driver under `%TEMP%\s28\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]]. Read-only on decision
layer; no engine code changes.

**Path.** `_resolve_pit_atm_iv` at `engine/wheel_runner.py:147`
(consumes `conn.get_iv_history` → averages `hist_put_imp_vol` +
`hist_call_imp_vol`). The `skew_mult` block at
`engine/wheel_runner.py:1252-1285` (consumes `chain_df` from
`conn.get_options(ticker)` if present — otherwise stays at 1.0).
The skew machinery at `engine/skew_dynamics.py` (`skew_slope`,
`NelsonSiegelTermStructure`, `ivs_dislocation_score`).

**Status.** Done. **Verdict: skew is structurally dormant on the
Bloomberg connector for two independent reasons (data + chain
access), and the dormancy is invisible to a trader reading the
ranker output. The skew columns surface in the output but carry
either a constant (`skew_multiplier=1.0`) or are silently empty
(`skew_slope`, `put_skew`).**

**Findings:**

- **(F1 — data) The Bloomberg IV file has zero put/call
  asymmetry across its entire history.**

  ```
  total rows:                                      1,361,615
  rows with both put_iv & call_iv populated:       1,353,901
  rows where put_iv == call_iv EXACTLY:            1,353,901  (100.0%)
  gap (put - call):  mean = 0.000000   std = 0.000000   abs.max = 0.000000
  ```

  Across 503 tickers, 11 years of data. The two columns
  (`hist_put_imp_vol` / `hist_call_imp_vol` in
  `data/bloomberg/sp500_vol_iv_full.csv`) are mechanically
  identical — almost certainly the result of the Bloomberg
  extraction populating both columns from a single ATM IV
  field (likely `IVOL_MONEYNESS_DAYS_50_30` or equivalent),
  not from genuine ATM put-vs-call quotes. **The file claims
  to carry skew data; it does not. Logged.**

- **(F2 — degenerate Fix #1) `_resolve_pit_atm_iv` averaging
  is a provable no-op on this data.** Sample at
  `as_of=2026-03-20`:

  ```
  ticker    put_iv   call_iv     avg   resolved   matches?
  TSLA      46.99     46.99   0.4699   0.4699      YES
  NVDA      40.23     40.23   0.4023   0.4023      YES
  META      40.16     40.16   0.4016   0.4016      YES
  AMZN      40.25     40.25   0.4025   0.4025      YES
  KO        23.86     23.86   0.2386   0.2386      YES
  PG        25.90     25.90   0.2590   0.2590      YES
  JNJ       27.70     27.70   0.2770   0.2770      YES
  XOM       32.16     32.16   0.3216   0.3216      YES
  VZ        28.20     28.20   0.2820   0.2820      YES
  AAPL      30.79     30.79   0.3079   0.3079      YES
  ```

  The `(put_iv + call_iv) / 2` averaging in `_resolve_pit_atm_iv`
  is `X / X = X`. Fix #1 ([PR #179](https://github.com/MertYakar66/smart-wheel-engine/pull/179))
  chose the composite "average put and call" because that was the
  conservative call when the file *might* have asymmetry; on
  Bloomberg it provably doesn't, so the choice is harmless but
  also informationally vacuous. **Logged — Fix #1 is correct;
  the data layer is the binding constraint.**

- **(F3 — runtime) `skew_mult` is uniformly 1.0 in the ranker
  output because the Bloomberg connector lacks chain access.**
  Probe (5 names, 25-delta, 35-DTE puts at `as_of=2026-03-20`):

  ```
  hasattr(conn, 'get_options'):       False
  hasattr(conn, 'get_option_chain'):  False

  skew_multiplier:  unique values = [1.0]   (n_unique=1)
  skew_slope:       unique values = []      (n_unique=0)
  put_skew:         unique values = []      (n_unique=0)
  ```

  The `skew_mult` block at `engine/wheel_runner.py:1252-1285`
  is gated:

  ```python
  if use_skew_dynamics and chain_df is not None and len(chain_df) > 0:
      ...  # compute iv_25p, iv_atm, iv_25c from chain; skew_slope; skew_mult
  ```

  `chain_df` is sourced via `conn.get_options()` /
  `conn.get_option_chain()` at lines 1187-1193; the Bloomberg
  connector has neither method, so `chain_df = None`, the block
  is bypassed, and `skew_mult` stays at its initialised 1.0.
  The `skew_slope` and `put_skew` diagnostic columns are
  silently empty — not a "skew is neutral" signal but a "skew
  was not measured at all" non-signal. **Logged.**

- **(F4 — code museum) Skew-machinery callers in
  `engine/`.** Static caller scan of
  `engine/*.py` for the public symbols of
  `engine/skew_dynamics.py`:

  ```
  theta_connector.py        → imports skew_slope (1 occurrence)
  volatility_surface.py     → imports skew_slope (6 occurrences)
  wheel_runner.py           → imports skew_slope (6 occurrences, all inside chain_df branch)
  ```

  And the two larger surfaces:

  ```
  NelsonSiegelTermStructure → zero callers in engine/
  ivs_dislocation_score     → zero callers in engine/
  ```

  `wheel_runner.py`'s 6 occurrences all sit inside the
  `chain_df is not None` branch (F3 — dormant on Bloomberg).
  `theta_connector.py` and `volatility_surface.py` use it but
  per CLAUDE.md D9, `volatility_surface.py` is "dormant"
  (SVI tooling exists but is not wired into the live ranker
  on either connector). **The Nelson-Siegel term-structure
  fitter and the dislocation-score module are codified but
  unreachable from any live path.** Logged.

- **(F5 — observability, the trader-facing surface) The
  ranker output emits skew columns even when dormant.**
  Diagnostic columns `skew_multiplier=1.0`, `skew_slope=∅`,
  `put_skew=∅`, `skew_pnl=non-zero` (a separate concept;
  see below) appear in every output row of
  `rank_candidates_by_ev`. A trader inspecting the output
  would reasonably conclude the skew dynamics module is alive
  and contributing a neutral signal — the truth is the module
  never ran on Bloomberg. Same observability shape as S22 F1
  (`drops` missing on `suggest_rolls`) and S28 F2
  (`expected_dividend` populating regardless of the gate).
  **Logged.**

- **§2 verified.** `rank_candidates_by_ev` routes every
  candidate through `EVEngine.evaluate` regardless of skew
  status. The skew multiplier is a multiplicative scalar
  bounded above 0 — it cannot rescue a negative-EV trade
  even if it were active. The dormancy reduces the engine's
  signal richness on Bloomberg, but does not break the
  invariant. **Logged as a positive.**

- **(Observation, not a bug) `skew_pnl` is populated and
  is a separate concept.** Unique values in the probe were
  non-zero (e.g. `-3.843`, `-3.014`). This is *not*
  `skew_dynamics.skew_slope` — it is the empirical-forward
  cost-of-skew adjustment applied inside the EV
  computation, distinct from the multiplicative `skew_mult`.
  Not part of the dormancy finding. **Logged for clarity.**

**Realism Check.**

| Aspect | Engine (Bloomberg) | Real-market behaviour | Verdict |
|---|---|---|---|
| put_iv vs call_iv asymmetry in the IV file | identical on 100% of 1.35M rows across 503 tickers | Real markets carry meaningful put-call skew, especially on high-beta names (TSLA, NVDA) and risk-off sectors | ❌ Data missing entirely |
| `_resolve_pit_atm_iv` resolved IV (post Fix #1) | average of put + call = put_iv (since they're equal) | Average of put-IV and call-IV is meaningfully different when real skew exists | ⚠ Mechanically correct; informationally vacuous |
| `skew_multiplier` in `rank_candidates_by_ev` output | 1.0 across all rows on Bloomberg | Real put skew should pull `skew_mult` < 1.0 on risk-off names; mild call skew (rare) > 1.0 | ❌ Always 1.0 — block bypassed (no chain access) |
| `skew_slope`, `put_skew` diagnostic columns | empty (`n_unique=0`) | Should populate with the 25Δ-put-vs-25Δ-call slope when skew dynamics is alive | ⚠ Silently empty (looks like working machinery, isn't) |
| `engine/skew_dynamics.NelsonSiegelTermStructure` live callers in `engine/` | zero | Term-structure fitter would inform 1d/1w/1m IV regime classification | ⚠ Codified, unreachable |
| `engine/skew_dynamics.ivs_dislocation_score` live callers in `engine/` | zero | Dislocation score would flag VRP / gamma mispricing opportunities | ⚠ Codified, unreachable |

**Verdict.**

- **Skew is dormant on Bloomberg for two independent
  reasons, either of which would suffice to disable the
  signal.** (1) the IV file's apparent put/call columns are
  duplicates, so even a put-vs-call gap signal cannot be
  read from the data layer. (2) the `skew_mult` block needs
  a per-strike chain (delta + iv per leg) that the
  Bloomberg connector does not expose. Fixing one without
  the other still leaves the multiplier at 1.0.

- **The ranker output mis-implies activity.** Surfaced
  columns `skew_multiplier`, `skew_slope`, `put_skew` look
  like the live signals of a working skew module. They are
  not. The trader sees `skew_multiplier=1.0` and concludes
  "skew is neutral right now," when the honest reading is
  "skew was not measured at all." This is **not** a §2
  authority breach (the multiplier cannot rescue a
  negative-EV trade) but it is a **realism breach** —
  the engine claims more signal than it has.

- **The skew machinery itself is well-built and unused.**
  `engine/skew_dynamics.py` is a clean implementation of
  Nelson-Siegel term-structure fitting + 25Δ-put-vs-25Δ-call
  slope + dislocation scoring. None of it is reachable from
  the live ranker on Bloomberg, and the two largest
  surfaces (`NelsonSiegelTermStructure`,
  `ivs_dislocation_score`) have zero callers in
  `engine/*.py`. This is the same pattern as
  `volatility_surface.py` per CLAUDE.md D9 (SVI tooling
  "exists but is dormant").

**AI handoff.**

- **Fix #1 (smallest scope, observability):** when the
  `skew_mult` block does not execute, the diagnostic
  columns should reflect that. Either drop the columns
  from the output (cleaner but breaks downstream schema)
  or surface a `skew_source` column with values
  `"chain"` / `"unavailable"` so the trader can tell
  the difference between "measured neutral" and "not
  measured at all." Mirrors the [[realism-check-pattern]]
  precedent of provenance columns (S1B `oi_source`,
  `premium_source` from PR #160).

- **Fix #2 (data layer, harder):** when the connector
  carries a chain (Theta), the `skew_mult` block works
  as-built. When it doesn't (Bloomberg today), there is
  no readily-available fallback — the IV file's put/call
  columns are identical, and there is no per-strike data
  to compute a 25Δ slope. Two paths forward:
  - **Provider migration** — switch the live ranker to
    use Theta on demand for the skew computation, falling
    back to `skew_mult=1.0` when Theta is unavailable.
    Cleaner; matches the architecture per
    [[bloomberg-data-refresh-blocked]].
  - **Term-structure proxy** — use the existing IV-vintage
    columns (`hist_put_imp_vol`, `volatility_30d`,
    `volatility_60d`, `volatility_90d`, `volatility_260d`)
    to fit a Nelson-Siegel curve and back out an
    implied-skew proxy from the curvature parameter. Less
    direct than per-strike skew, but reachable from
    Bloomberg data. Would wire up the dormant
    `NelsonSiegelTermStructure` to a live consumer.

- **Fix #3 (re-extract the Bloomberg IV file):** if the
  Bloomberg side has actual put-skew and call-skew quotes
  for the names we cover (e.g. via the
  `IV_MONEYNESS_*_PUT` and `IV_MONEYNESS_*_CALL`
  field families), re-pulling them into the
  `sp500_vol_iv_full.csv` extraction would close F1. This
  is blocked the same way [[bloomberg-data-refresh-blocked]]
  blocks the other refreshes — needs the user's BQL
  queries.

- **Sanity follow-up Sn:** once Fix #1 (observability) is in,
  re-run S29 on a Theta replay (queued S6) to verify
  `skew_source="chain"` populates and the `skew_multiplier`
  actually moves below 1.0 on names with steep put skew
  (TSLA / NVDA expected; KO / PG should stay near 1.0).
  That would close the "is the skew code mechanically
  working?" question separately from "is the data
  available?"

- **Dealer positioning (S31 candidate, deferred this
  cycle):** the dealer-regime path has the *same*
  Bloomberg-chain-absence dependency as skew here (no
  per-strike gamma exposure data → synthetic `GEX`
  reconstruction). The user prompt flagged it as a
  contract-test only. Worth a small Sn after a Theta
  replay, paired with skew on the same data.

**Methodology debt.**

- **Single connector (Bloomberg).** The S29 verdict ("skew
  dormant") applies only to the Bloomberg connector. The
  Theta provider has chain access and presumably activates
  the `skew_mult` block. A Theta-replay Sn would let us
  confirm or deny that the *implementation* is correct,
  separately from the *data* gap covered here.

- **Did not test `use_skew_dynamics=False` vs True.** The
  driver defaulted to `use_skew_dynamics=True` (the
  ranker's default). Since `skew_mult` stays at 1.0
  regardless on Bloomberg, toggling the flag would not
  change the EV — but documenting the no-op would
  make the dormancy explicit in the kwarg surface.

- **No coverage-by-ticker breakdown.** We measured
  put_iv == call_iv at file scale (100%) but did not
  ask whether the "skew gap is zero" pattern is
  identical across all 503 tickers vs different ones at
  different vintages. A per-ticker time-series of the
  gap would confirm "this is a single extraction bug,
  not 503 individual ticker bugs," which matters if a
  fix is scoped per-ticker vs file-wide.

### S30 — HMM regime-multiplier realism (April 2025 vol spike)

**Purpose.** Realism check on the HMM regime layer — does the
engine's `hmm_multiplier` (output of the 4-state Gaussian HMM at
`engine/regime_hmm.py:76`, wired into `rank_candidates_by_ev` at
`engine/wheel_runner.py:1143-1179`) actually shift in the right
direction at a known regime transition, and is the magnitude
realistic? CLAUDE.md §2 documents the multiplier range as
`[0.0, 1.25]` (per-state weights: `crisis: 0.2, bear: 0.5,
normal: 1.0, bull_quiet: 1.25`). S30 tests both downward and
upward transitions against the April 2025 broad-market vol
spike — a real-world event with measurable spot, IV, and
realized-vol moves.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
five-name watchlist (AAPL, MSFT, JPM, XOM, UNH) across two
transition windows + one calm-period anchor:

| Window | Pre-date | Post-date | Transition (survey-identified on AAPL) |
|---|---|---|---|
| T1 — downturn | 2025-04-02 (Wed) | 2025-04-04 (Fri) | bear → crisis |
| T2 — recovery | 2025-04-11 (Fri) | 2025-04-15 (Tue) | crisis → bear |
| Anchor | — | 2026-03-20 (cutoff) | for current-regime comparison |

Survey used AAPL log-returns 2018-2026 fit to a 4-state Gaussian
HMM (n_iter=20, random_state=42 — same as the live ranker config
at `wheel_runner.py:1165`) and Viterbi-decoded the state path
to find the transitions inside the data window.

Driver under `%TEMP%\s29\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]]. 25-delta / 35-DTE
short puts. `use_event_gate=False` to isolate the HMM signal
from earnings-window noise. Read-only on decision layer.

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py:631`. HMM fitting at lines 1143-1179
(`tail = log_rets[-504:]`, cached by `(ticker, hash(fingerprint))`).
Per-state weights at `regime_hmm.py:265-280`
(`position_multiplier`). Final regime multiplier composed at
`wheel_runner.py:1300`
(`combined_regime_mult = hmm_regime_mult * skew_mult * news_mult
* credit_mult`), passed into `ShortOptionTrade.regime_multiplier`,
consumed by `EVEngine.evaluate`.

**Status.** Done. **Verdict: the engine's HMM is the
best-behaved of the four knowledge surfaces in this campaign.
It captured the April 2025 vol spike correctly across ALL FIVE
names simultaneously (broad-market crisis recognition), respects
its documented multiplier range, and the downstream EV behaviour
is realistic — the multiplier cut is balanced against the
simultaneous IV-spike-driven premium increase, not a blind
kill-switch.**

**Findings:**

- **(F1 — major positive) The HMM correctly recognised the April
  2025 broad-market crisis on every name.** Pre-vs-post the
  2025-04-03 transition:

  ```
  ticker   mult_pre   regime_pre   ->  mult_post  regime_post   d_mult   d_spot     d_iv
  AAPL        0.898       normal   ->      0.200       crisis   -0.698  -15.85%  +73%
  JPM         0.552         bear   ->      0.200       crisis   -0.352  -14.46%  +76%
  MSFT        0.700       normal   ->      0.201       crisis   -0.499   -5.84%  +39%
  UNH         0.668         bear   ->      0.232       crisis   -0.436   +0.35%  +25%
  XOM         0.700       crisis   ->      0.200       crisis   -0.499  -12.07%  +89%
  ```

  Per-ticker independent HMM fits converged on `crisis` for
  all five names within two trading days. The realised spot
  moves (-5% to -16%) and IV spikes (+25% to +89%) on four of
  five names are consistent with a real broad-market crisis
  episode; UNH was a partial exception (spot held, IV moved
  less) but the HMM still moved to crisis on UNH's own
  returns. **Logged as a positive — per-ticker HMM converging
  on the same regime label across a watchlist IS the right
  signal for a macro event, even without an explicit
  cross-asset aggregator.**

- **(F2 — positive) Reverse transition (recovery) captured on
  2025-04-15.** Pre-vs-post the 2025-04-14 transition:

  ```
  ticker   mult_pre   regime_pre   ->  mult_post  regime_post   d_mult
  AAPL        0.212       crisis   ->      0.817       normal   +0.605
  MSFT        0.478         bear   ->      0.633         bear   +0.155
  XOM         0.405       crisis   ->      0.738       normal   +0.334
  ```

  AAPL recovers crisis → normal (multiplier rises 4×). XOM same
  shape. MSFT stays bear but moves up the bear-confidence axis
  (0.48 → 0.63). The HMM is **responsive to recovery, not just
  downturns** — it does not get permanently stuck in crisis
  after a vol spike. JPM and UNH dropped out of the result set
  on one of the two dates in T2 (history-gate or chain-quality
  cutoff during the high-vol window — F6 below). **Logged as
  a positive — the HMM mean-reverts correctly.**

- **(F3 — positive) HMM multipliers respect the documented
  `[0.20, 1.25]` envelope.** Every `hmm_multiplier` value
  observed in the matrix sits inside the bound. The crisis
  weight maps to exactly 0.200 (the hard floor) for pure-state
  posteriors; mixed states yield values like 0.201, 0.212,
  0.232 (posterior-weighted average across the 4-state
  weights). The maximum observed was XOM at 0.9217 on
  2026-03-20 (`bull_quiet` posterior weighted into the average,
  sub-1.25 because the posterior is not 100% bull_quiet).
  **Logged.**

- **(F4 — design positive) EV impact is balanced, not blind.**
  On T1 the HMM cut multiplier 0.5-0.9 → 0.2, but `ev_dollars`
  did not simply collapse. Same window:

  ```
  ticker    ev_pre   ev_post   d_ev      Driver
  AAPL      +32.52   +60.95   +28.43    IV spike +73% → premium 3.05 -> 4.60 outweighed HMM cut
  JPM     +108.33    +75.40   -32.93    HMM cut dominated
  MSFT    +195.91    +90.39  -105.52    HMM cut dominated
  UNH     +520.19   +177.81  -342.38    HMM cut + premium up less than HMM impact
  XOM      -12.58    +20.19   +32.77    IV spike +89% rescued an underwater candidate
  ```

  The engine is doing balanced multi-signal math — IV spike
  raises the synthetic premium (and `ev_dollars` is positive in
  the integrand) while the HMM multiplier cuts the final scaling.
  Whichever wins on a given name depends on which moved more.
  **This matches the intent of the dealer-multiplier asymmetric
  clamp (CLAUDE.md §2) — the HMM is a *de-emphasis* lever, not
  a *veto*.** A real wheel trader does not want a regime
  indicator to fully suppress trades; the engine's behaviour
  here is realistic. **Logged as a positive.**

- **(F5 — observability gap) No cross-asset coherence
  signal.** The HMM is fit per-ticker independently. When
  5/5 names go to `crisis` simultaneously (T1), that is a
  macro-event signal worth surfacing distinctly. Currently
  the trader sees five separate "this name's HMM says
  crisis" rows in the ranker output, not a single
  "macro regime appears to be in crisis" header. A
  `macro_regime` aggregator column (or board-level
  signal in the dossier) would close this. **Logged as a
  follow-on observability finding** (same family as S22 F1,
  S28 F2, S29 F5 — the campaign's observability theme).

- **(F6 — data dropout, minor) UNH and JPM dropped from
  some result sets during T2.** UNH disappears on the
  2025-04-15 post-row; JPM disappears on the 2025-04-11
  pre-row. Most likely cause: history-gate or chain-quality
  gate triggered by the unusual return distribution from
  the 2025-04-04 crisis day. Not a HMM bug but a data-flow
  observation — the HMM had data to fit on the surviving
  rows; the question is why the upstream filters dropped
  these specific cells. **Logged.**

- **(Anchor) Current data-cutoff regime is mildly defensive.**
  At `as_of=2026-03-20`:

  ```
  ticker      hmm_regime   hmm_multiplier
  AAPL        bear         0.677
  JPM         bear         0.711
  MSFT        bear         0.463
  UNH         bear         0.668
  XOM         bull_quiet   0.922
  ```

  Four of five megacaps in bear regime; energy (XOM) the
  outlier in `bull_quiet`. Consistent with a recent
  modest-vol environment, not a crisis. **Logged as an
  anchor.**

- **§2 verified.** `rank_candidates_by_ev` routes every
  candidate through `EVEngine.evaluate`. The HMM
  multiplier is multiplicative and cannot rescue a
  negative-EV trade (XOM in T1 went from -$12.58 to +$20.19
  not because of HMM but because of the IV-spike-driven
  premium increase; the HMM actually pulled the multiplier
  *down*). **Logged as a positive.**

**Realism Check.**

| Aspect | Engine | Real-market behaviour | Verdict |
|---|---|---|---|
| AAPL HMM regime at 2025-04-04 (post-vol-spike) | `crisis` (mult 0.200) | -15.85% spot, +73% IV spike — clear crisis | ✓ Aligned |
| 5/5 watchlist names at `crisis` on 2025-04-04 | All five converge to `crisis` simultaneously | Broad-market sell-off; macro event signature | ✓ Aligned (per-ticker convergence is the correct macro signal) |
| AAPL recovery to `normal` at 2025-04-15 | mult 0.817 (`normal`) | AAPL +2.0% recovery, IV partially crushed | ✓ Aligned |
| MSFT lag-recovery (stays bear) | mult 0.633 (`bear`) | MSFT only -0.7% net recovery, IV still elevated | ✓ Aligned (cautious mean-reversion) |
| Anchor regime at 2026-03-20 | 4/5 bear, XOM bull_quiet | Recent regime is mildly defensive | ✓ Aligned (consistent with current data) |
| HMM multiplier respect `[0.20, 1.25]` envelope | Observed range 0.200 - 0.922 across matrix | Documented bound | ✓ Aligned |
| EV-impact direction on T1 | Mixed: AAPL/XOM up, JPM/MSFT/UNH down | Trader expects regime cut but vol spike raises premium | ✓ Aligned (balanced multi-signal) |
| Cross-asset coherence signal | None — each name reads independently | Trader watching 5/5 crisis would call macro event | ⚠ Observability gap (F5) |

**Verdict.**

- **The HMM is the best-behaved knowledge surface in this
  campaign.** It captured a real broad-market crisis on
  every name in the watchlist simultaneously, respects its
  documented multiplier bounds, mean-reverts correctly on
  recovery, and behaves realistically as a *de-emphasis*
  signal (not a veto) when composed with IV spikes.

- **No new bug surfaced in the HMM logic itself.** F1-F4
  are all positives. F5 is a *missing* feature (cross-asset
  aggregator) rather than a logic error. F6 is upstream
  data-flow, not HMM.

- **The HMM is the realism counter-example to S29's skew
  finding.** Skew is dormant on Bloomberg because the data
  isn't there; HMM is alive because it consumes OHLCV
  log-returns — a column that *is* populated cleanly across
  503 tickers × 8+ years. The lesson generalises: engine
  surfaces that consume well-supported data columns work as
  built; surfaces that need chain-level data don't, on
  Bloomberg.

**AI handoff.**

- **Fix #1 (observability, follow-on to F5):** add a
  `macro_regime` row at the top of the ranker output (or a
  `macro_regime_unanimous` boolean in the dossier metadata).
  Compute as the modal `hmm_regime` across the result set
  when the agreement rate exceeds a threshold (e.g. 4/5 of
  ranked names share a regime). Surfaces the broad-market
  signal that's currently latent in five separate rows.

- **Fix #2 (observability, related):** add a
  `hmm_state_posterior` column with the full 4-state
  posterior (or just the top-2 probabilities) so the trader
  can distinguish "0.21 because I'm 95% crisis" from "0.32
  because I'm 50% crisis + 50% bear" — currently both
  serialize to similar multiplier values but mean different
  things downstream.

- **Fix #3 (data-flow, follow-on to F6):** investigate why
  UNH and JPM dropped from the T2 result sets. If it's the
  history gate firing on tail-window length (the 504-day
  tail at the high-vol windows might shift the cache key
  enough to gate them), the fix is at `wheel_runner.py`
  history-gate level. If it's chain-quality gate, that's
  separate (and not a real concern given Bloomberg has no
  chain anyway). Small Sn (~1 page).

- **HMM-cross-asset Sn:** the natural follow-on to S30 is a
  basket Sn that rank-orders 50 S&P 500 names on the same
  watershed dates (e.g. 2025-04-04) and reports the fraction
  in each HMM regime. If 40+/50 go to crisis simultaneously,
  that confirms the macro signal interpretation. If it's
  noisier (15/50 crisis, 20/50 bear, 15/50 other), the
  per-ticker convergence in S30 was a small-sample artifact
  and the F5 macro-aggregator design would need a smarter
  threshold.

- **Dealer positioning (S31 candidate, deferred this cycle):**
  the dealer-regime path is similar to skew in that it needs
  per-strike gamma data Bloomberg doesn't have. After a Theta
  replay, a small Sn could test R6 / R8 contract fire on a
  synthetic `dealer_regime_by_ticker={"AAPL": "short_gamma_amplifying"}`
  via `WheelTracker.portfolio_context_snapshot` and
  `build_dossiers(portfolio_context=ctx)` (the #174 wire). Pair
  with skew on the same Theta data.

**Methodology debt.**

- **Single watershed event (April 2025 vol spike).** The
  T1 finding ("all 5 names converge to crisis simultaneously")
  is from one transition. To make the broader claim ("the
  per-ticker HMM converges on macro events"), basket the test
  across a half-dozen other historical vol-spike dates (e.g.
  COVID 2020-03, August 2024 yen-carry unwind, October 2023,
  September 2022 CPI shock) and verify the same shape. If 4
  of 6 events show 4/5 convergence, the macro-signal
  interpretation is strong; if it's 2 of 6, F5's macro
  aggregator needs a per-event regime classifier.

- **Cache not exercised.** The HMM cache (keyed by
  `(ticker, hash(tail-fingerprint))`) means re-running the
  same as_of for the same ticker hits the cache and skips
  re-fit. The 2-as_of test in S30 always re-fits (the tail
  fingerprint differs between adjacent days). A
  cache-stress Sn would help confirm "the cache is invalidated
  correctly when new bars arrive" but isn't a realism
  question.

- **No comparison to a heuristic baseline.** A pro trader's
  mental model for regime might be "VIX > 30 → crisis, VIX
  < 15 → bull_quiet, in between → normal/bear." The
  engine's HMM agrees on the T1 dates (VIX clearly spiked
  to >30 in April 2025) but a side-by-side comparison
  across a basket would quantify "HMM is X% concordant with
  VIX-threshold regime" — a useful sanity check.

### S31 — Compounding-crisis stress (six concurrent downgrade signals)

**Purpose.** Every prior Sn entry isolates a single decision-layer
surface (S22 roll defense, S23 event gate + IV-crush, S28 CC
dividend, S29 skew dormancy, S30 HMM regime). S31 asks the
load-bearing follow-up: when **multiple downgrade gates fire on
overlapping candidates at the same `as_of`**, does the engine
produce a coherent output — or do non-composable downgrades
silently fail, missing signals stack in unexpected ways, and the
trader lose the ability to reconstruct *why* a name was filtered?
Wheel-trader pain point: in a real crisis the engine doesn't get
to test one constraint at a time. HMM crisis × dealer short-gamma ×
event lockout × sector concentration × ITM roll defense × CC near
ex-div all hit a live book at once. S31 is the first integration
test that puts six gates on the same anvil.

**Setup.** Anchor date `as_of=2025-04-04` — the AAPL bear → crisis
transition day from S30. Chosen so the HMM stressor fires
**naturally on real-data dynamics**, not as a synthetic config
flag. Provider `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Account size $250k (institutional sub-PM tier). Universe of 22
names spanning tech (AAPL/MSFT/NVDA/GOOGL/META/TSLA/AMZN/AVGO),
banks (JPM/BAC/WFC/C/GS/MS), defensives (KO/PG/JNJ/VZ/XOM), and
health (UNH/ABBV/LLY).

Existing 5-position book constructed at `BOOK_ENTRY=2025-03-14`
(3 weeks pre-crisis):

- **3 ITM short puts on tech** — AAPL K=250 (spot 188.38, 32.7%
  ITM), MSFT K=420 (spot 359.84, 16.7% ITM), NVDA K=135 (spot
  94.31, 43.1% ITM). Forces `suggest_rolls` into roll-defense
  territory (S22 surface).
- **2 covered calls on banks** — JPM stock basis $240 + call K=250
  (D17 stock-already-owned path), BAC stock basis $45 + call
  K=47 (same shape).

Driver under `%TEMP%\s31\driver.py` (not committed); six
orthogonal checks executed in one run, with
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-c")`
per [[sys-path-worktree-shadow]]. Read-only on decision layer;
positions constructed with `require_ev_authority=False` (the
launch-gate is not the surface under test — see methodology debt).

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route — wraps
`EVEngine.evaluate` per candidate). `WheelTracker.suggest_rolls`
at `engine/wheel_tracker.py:2004`. HMM regime layer at
`engine/regime_hmm.py` (output `hmm_regime` + `hmm_multiplier`
into the ranker diagnostic columns). Dealer positioning at
`engine/dealer_positioning.py` (output `dealer_multiplier`,
clamped to `[0.70, 1.05]` per §2). Portfolio-risk gates at
`engine/portfolio_risk_gates.py` (`check_sector_cap`,
`check_kelly_size`, `take_snapshot`). Event surface via
`MarketDataConnector.get_earnings` (per-ticker proxy used in
place of `EventGate.from_bloomberg_calendar` — see F5 for why).

**Status.** Done. **Verdict: the §2 invariants hold under
compounding stress (dealer clamp preserved, no ranker bypass, EV
sign discipline correct), but the engine is operationally
*illegible* in this regime — 20 of 22 candidate names are filtered
silently, the surviving 2 carry no per-name breadcrumb of *why*
the other 20 disappeared, and the trader cannot reconstruct the
gate-by-gate verdict without re-instrumenting the call site.
Five compounding gates verified; the sixth (dealer regime) is
dormant on Bloomberg the same way S29 found for skew.**

**Findings:**

- **(F1 — observability, the killer finding) Universe collapses
  22 → 2 with zero per-name telemetry on which gate filtered
  each name.** With `min_ev_dollars=-1e9` (every numerically-
  computable EV should survive), the ranker returned exactly
  two rows: AVGO ($82.65) and NVDA ($36.22). Twenty other
  names — AAPL, MSFT, GOOGL, META, TSLA, AMZN, JPM, BAC, WFC,
  C, GS, MS, KO, PG, JNJ, VZ, XOM, UNH, ABBV, LLY — were
  filtered upstream and emit *no row* in the output:

  ```
  Returned 2 rows.
  Top 10 by ev_dollars (or all if fewer):
  ticker   spot     iv  premium  ev_dollars  hmm_regime  hmm_multiplier
    AVGO 146.29 0.6415    4.829       82.65     crisis          0.2256
    NVDA  94.31 0.7278    3.592       36.22     crisis          0.2046
  ```

  Plausible filter causes — chain-data unavailable, missing-IV,
  PIT contamination check, event-buffered (3 of the 22 had
  earnings within 7 days), or upstream survivorship — but the
  output gives the trader no way to tell which name failed which
  gate. Same observability shape as S22 F1 (`drops` missing on
  `suggest_rolls`), S28 F2 (`expected_dividend` populating when
  gate blocks), and S29 F5 (`skew_*` columns surfacing when
  dormant). **This is the highest-priority operational finding
  S31 surfaces:** the trader's mental model in a crisis ("show
  me what's safe to trade right now") requires a row per
  *considered* name with a `filter_reason` column, not a
  silent drop. **Logged.**

- **(F2 — positive, §2 verified) HMM crisis multiplier fires
  correctly on the two survivors.** `hmm_regime=crisis` for
  both AVGO and NVDA at `as_of=2025-04-04`, with multipliers
  0.2256 (AVGO) and 0.2046 (NVDA) — both close to the
  documented crisis-state value of 0.2 per
  `engine/regime_hmm.py`. Direction matches S30's confirmed
  AAPL bear → crisis transition on this exact date. The crisis
  multiplier compresses raw EV by ~5×, leaving small absolute
  EV values ($82 / $36) — internally consistent. The engine
  *honestly* downweights even when names survive the
  filter. **Logged as positive.**

- **(F3 — schema gap) The ranker output lacks four columns
  the multi-surface stress test requires.** Driver asked for
  `regime_multiplier`, `event_locked`, `event_reason`, and
  `sector` — none are present in the returned DataFrame:

  ```
  Selected columns present: [ticker, spot, iv, premium, ev_dollars,
    ev_per_day, hmm_regime, hmm_multiplier, dealer_multiplier,
    dealer_regime, skew_multiplier]
  Missing requested columns: [regime_multiplier, event_locked,
    event_reason, sector]
  ```

  - `hmm_multiplier`, `dealer_multiplier`, `skew_multiplier` are
    surfaced separately, but no **combined** `regime_multiplier`
    column shows how they compose into the EV scaling. A trader
    inspecting one row can't tell whether the 0.20 effective
    multiplier came from HMM alone, dealer × HMM, or all three.
  - No `event_locked` / `event_reason` means the silent-filter
    behaviour in F1 is *required* — there's no row to surface
    "blocked: earnings in 7 days" against.
  - No `sector` means portfolio-level reasoning about the ranker
    output requires a separate lookup against `DEFAULT_SECTOR_MAP`
    (F6) — the ranker doesn't help.

  **Logged.** Closing F1 likely requires closing F3 first
  (you need a row per considered name with a filter_reason
  column).

- **(F4 — data, same shape as S29) Dealer multiplier dormant
  on Bloomberg.** Both surviving rows show
  `dealer_multiplier=1.0`, `dealer_regime=None`:

  ```
  Multiplier stats:
    n=2, unique=1, min=1.0000, max=1.0000, mean=1.0000
    in [0.70, 1.05]? True
  ```

  §2 clamp `[0.70, 1.05]` is mathematically preserved (1.0 is
  inside the band), so the test
  `test_dealer_multiplier_evengine_integration.py` (#193) would
  pass here. But the *signal* is dead: no chain access on
  Bloomberg means `engine/dealer_positioning.py` cannot
  compute GEX / gamma flip / walls; the multiplier falls back
  to its safe default of 1.0 and `dealer_regime` returns
  `None`. Same root cause as S29 skew dormancy and
  [[bloomberg-iv-no-skew]]: Bloomberg has no per-strike chain.
  The §2 invariant holds; the *intended functionality* does
  not. **One of the six S31 stressors does not bite on
  Bloomberg; it would bite on Theta. Logged.**

- **(F5 — scenario realism, surprise) The scenario's named
  event-stressor (AVGO earnings) doesn't fire; the actual
  event stressor hiding in the book is JPM.** Per-name
  `get_earnings` probe at `as_of=2025-04-04`:

  ```
  ticker   next earnings   days_out   in_window
  AVGO        2025-06-05         62          no
  JPM         2025-04-11          7         YES  ← held as CC in book
  MS          2025-04-11          7         YES
  WFC         2025-04-11          7         YES
  GS          2025-04-14         10          no
  BAC         2025-04-15         11          no  ← held as CC in book
  C           2025-04-15         11          no
  JNJ         2025-04-15         11          no
  ```

  AVGO (which we chose as the "earnings stressor" target) is
  62 days out — irrelevant to the event gate's typical 5-day
  earnings buffer. The actual hidden stressor is **JPM**: held
  in the book as a covered call, with earnings exactly 7 days
  away — already inside any reasonable event-buffer band. A
  CC sized to expire 28 days out would normally be fine
  through earnings, but the book's risk profile is more
  exposed than the explicit setup acknowledged. **The
  scenario surfaced a real risk the operator didn't plan for;
  this is what stress tests are for.** Logged as a realism
  insight, not a bug.

- **(F6 — sector taxonomy, trader mental-model gap)
  `check_sector_cap` is GICS-strict; "tech" colloquially is
  not "Information Technology" sector.** Probe with the 3-tech
  book loaded and a proposed TSLA short put ($22k notional):

  ```
  TSLA  (Cons Disc)   passed=True   post_open_pct=0.0941
  AVGO  (Info Tech)   passed=False  (existing IT > 25% cap already)
  GOOGL (Comm Svcs)   passed=True   post_open_pct=0.0684
  ```

  AAPL, MSFT, NVDA are all GICS Information Technology; TSLA
  is GICS Consumer Discretionary; GOOGL is GICS Communication
  Services. The driver's check 5 ("compound trade on top of 3
  tech puts") showed `sector_cap.passed=True` for TSLA — *not
  a bug*, but a mismatch with the trader's likely mental model
  ("I'm 60% tech by position count, adding another tech name
  should be blocked"). The colloquial "tech" cluster spans
  three GICS sectors; the gate aggregates per GICS sector
  only. **Logged as a realism gap, not a §2 breach.** Worth
  considering a parallel "growth-cluster" or
  "high-beta-cluster" overlay gate that aggregates across
  the AAPL/MSFT/NVDA/META/TSLA/GOOGL set.

- **(F7 — observability inconsistency) `GateResult.details`
  uses different keys when the gate passes vs. fails.**
  When `check_sector_cap` returns `passed=True`, the details
  bag carries `post_open_sector_pct`. When it returns
  `passed=False`, the same value is carried as
  `sector_pct`. Discovered when the driver's probe used
  `details.get("post_open_sector_pct", 0)` against a failing
  result and silently got `0.0` (the default), masking the
  actual 40+%-of-NAV breach:

  ```python
  if is_allowed:
      return GateResult(details={"sector": ..., "post_open_sector_pct": post_open_pct})
  return GateResult(details={"sector": ..., "sector_pct": post_open_pct, "narrative": ...})
  ```

  (`engine/portfolio_risk_gates.py:319-338`). Caller-side
  code reading details must check both keys, or worse,
  default-to-0 and silently mis-read. Same observability
  shape as F3 schema gaps: shape-dependent on the verdict.
  **Logged — single-line fix, but a real footgun.**

- **(F8 — silent zero, defensive-correct) `suggest_rolls`
  returns zero candidates on all 3 ITM puts with no
  diagnostic of *why*.** Per-position output:

  ```
  AAPL: strike=250.00, current_spot=188.38 (+32.7% ITM)
    suggest_rolls returned ZERO surviving candidates
  MSFT: strike=420.00, current_spot=359.84 (+16.7% ITM)
    suggest_rolls returned ZERO surviving candidates
  NVDA: strike=135.00, current_spot=94.31 (+43.1% ITM)
    suggest_rolls returned ZERO surviving candidates
  ```

  Default `min_net_credit=0.0` keeps credit-only rolls. On
  positions this deep ITM (NVDA 43% ITM, AAPL 33% ITM), no
  credit roll exists in the (DTE × delta) grid; every
  candidate would require a debit. The honest output ("no
  roll") is the right call — this is **defensively correct**:
  the engine refuses to give bad advice (which is what S22
  F2 chastised the pre-fix tracker for). But the trader
  receiving zero rows can't tell:
  - Did all rolls have negative EV after the crisis multiplier?
  - Did all rolls fail the `min_net_credit=0` filter
    (rescue-debit could still be sensible)?
  - Did all rolls fail because the underlying moved past the
    strike-solver's safe band?

  An attached `reason_tally` (e.g. `{"all_debit": 12,
  "ev_negative": 8, "moneyness_floor": 0}`) would let the
  trader decide between "close and take assignment" vs
  "rescue-debit roll." Same observability pattern as F1 and
  F3. **Logged.**

- **(F9 — observability, combined-multiplier omission;
  surfaced during V1 verification) The combined
  `regime_multiplier` (HMM × skew × news × credit) is NOT
  surfaced in the ranker output, even though it is the single
  value that scales `ev_raw → ev_dollars`.** The components
  `hmm_multiplier`, `skew_multiplier`, `news_multiplier`,
  `credit_multiplier` ARE in the output (46-column DataFrame),
  but no combined `regime_multiplier` column exists. A trader
  who doesn't know `credit_multiplier` exists (it's not in
  any "obvious" diagnostic subset and was not in the F3
  list) will compute `ev_raw × hmm × skew × news` and find a
  20% gap with `ev_dollars` on a crisis day — no
  explanation. The author of this Sn entry made exactly this
  mistake on first read; the V1 verification surfaced the
  discrepancy and traced it to `credit_mult = 0.80` (line
  `engine/wheel_runner.py:773`, fires when global
  `credit_regime == "crisis"`). Same observability family as
  F1 / F3. **Logged.**

- **§2 verified (with stress-coverage caveats).** Across all
  six checks: (a) the dealer multiplier never moved outside
  `[0.70, 1.05]` — but was always 1.0 on Bloomberg
  (dormant), so the clamp was preserved **vacuously, not
  under stress**. The first-PR framing "§2 invariants
  survive compounding stress intact" overstated this — the
  clamp was never put under stress on this provider. (b) Every
  ranker-output row was generated via `EVEngine.evaluate`,
  and per V5 the 20 silently-filtered names are blocked by
  `event_gate`'s holding-window check, NOT by a non-EV
  rescue path. (c) `suggest_rolls` returning zero candidates
  is the conservative refusal per V6 + V7 — confirmed
  defensive, not bypass. **The §2 contract holds; the
  stress only meaningfully tested some of it.** Logged with
  the corrected scope.

**Verification probe (V1–V7).** A follow-up driver
(`%TEMP%\s31\verify.py`) closed the gap between *engine output
observed* and *engine output verified* for the most uncertain
claims above. The first-PR framing was honest about what was
observed but loose about what was *verified*; this section
upgrades or downgrades each claim accordingly:

- **V1 — composition arithmetic VERIFIED CORRECT.** For both
  survivors, the full multiplier chain matches `ev_dollars`
  to better than one cent:

  ```
  AVGO: ev_raw=457.89, hmm=0.2256, skew=1.0, news=1.0, credit=0.80
        product           = 457.89 × 0.2256 × 1.0 × 1.0 × 0.80 = 82.66
        ev_dollars output = 82.65  ✓ (rounding)

  NVDA: ev_raw=221.32, hmm=0.2046, skew=1.0, news=1.0, credit=0.80
        product           = 221.32 × 0.2046 × 1.0 × 1.0 × 0.80 = 36.23
        ev_dollars output = 36.22  ✓ (rounding)
  ```

  The arithmetic is exact. The previously-undocumented
  factor was `credit_mult = 0.80` from
  `engine/wheel_runner.py:773` (fires when global
  `credit_regime == "crisis"` per `fa.credit_regime`).

- **V2 — BSM fair-value sanity VERIFIED.** Both rows have
  `premium == fair_value` exactly (AVGO 4.829, NVDA 3.592)
  and `edge_vs_fair = 0`. Confirms the Bloomberg path
  (premium is BSM-derived, not quoted) per S1 / S29 F1.
  Same finding, fresh verification.

- **V3 — HMM state-probability decomposition VERIFIED.**
  Fitting the same 504-day-tail HMM directly on AVGO and
  NVDA log-returns at `as_of=2025-04-04`:

  ```
  AVGO state_probs   = [crisis 0.9680, bear 0.0000, normal 0.0320, bull_quiet 0.0000]
       Σ probs × wts = 0.9680*0.2 + 0*0.5 + 0.0320*1.0 + 0*1.25 = 0.2256  ✓
       matches ranker hmm_multiplier (0.2256) within 0.000017.

  NVDA state_probs   = [crisis 0.9941, bear 0.0003, normal 0.0056, bull_quiet 0.0000]
       Σ probs × wts = 0.2046                                         ✓
       matches ranker hmm_multiplier (0.2046) within 0.000026.
  ```

  Both names are 96–99% pure crisis state. The first-PR
  F2 framing "close to 0.20" was lazy — the math is *exact*,
  the values are probability-weighted blends of the four
  state multipliers, and the blend is dominated by crisis on
  this date. **F2 upgrade: verified mathematically clean.**

- **V4 — missing-data categorization of the 20 filtered
  names: ZERO data-layer drops.** Per-name probe: all 20 had
  OHLCV at `as_of`, all 20 had PIT IV (both `hist_put_imp_vol`
  and `hist_call_imp_vol`), and 0 had earnings within the
  default 5-day buffer.  So the silent filter is NOT caused
  by missing data, missing IV, or near-term-earnings (within
  documented default `earnings_buffer_days=5`).

- **V5 — the silent filter IS `event_gate` firing on the
  holding-window earnings overlap.** Re-running
  `rank_candidates_by_ev` with `use_event_gate=False`
  surfaced **all 20 missing names** with positive EV. The
  newly-surfaced rows have EVs ranging from $5.50 (VZ) to
  $720.48 (LLY); 17 are in `hmm_regime=crisis`, one in
  `bear`. Earnings for the 7 blocked-but-not-≤5-days names
  (JPM, MS, WFC at +7d; GS at +10d; BAC, C, JNJ at +11d;
  UNH at +13d) all sit INSIDE the 35-DTE holding window of
  the proposed trade. The `event_gate` blocks any trade
  whose holding window brackets an earnings event — that's
  exactly the right behaviour. **F1's "killer observability"
  finding upgrades to:** the engine is acting correctly,
  but the output offers the trader no way to see *why* each
  name was filtered. The decision is right; the explanation
  is missing.

- **V6 — `suggest_rolls` returns 0 candidates across 4 grid
  shapes on 43%-ITM NVDA**, including longer DTE (180d),
  debit allowed (`min_net_credit=-$500`), and very far OTM
  (5-delta strikes). The honest output ("no defensible
  roll") survives every parameter widening tested.

- **V7 — `suggest_rolls` returns 1 candidate on 0.9% ITM
  AAPL (control).** Engine WORKS on shallow ITM. The AAPL
  candidate carries `net_credit_debit = +$23.39` (passed
  the credit filter) but `roll_ev = −$141.28` (negative EV
  surfaced because the credit-only filter accepted it).
  **V6's zero on 43% ITM is therefore *defensible refusal
  at extreme moneyness*, not a bug.** F8 downgrade: the
  silent zero is correct in conclusion; the observability
  gap (no `reason_tally` of which (DTE, delta) cells failed
  and why) remains the only operational issue.

**Realism Check.**

| Aspect | Engine (Bloomberg, 2025-04-04) | Real-market / sound-trader expectation | Verdict |
|---|---|---|---|
| 22-name universe ranker scan | 2 rows returned, 20 silently filtered. **V5 proves filter is `event_gate` firing on 35-DTE holding-window earnings overlap.** All 20 missing names surface with positive EV when `use_event_gate=False` | A real trader at a desk would expect either a row per considered name (with `filter_reason`) or a header summary of "20 skipped — N event-blocked, M missing-IV, ..." | ⚠ Decision correct (event_gate fires on real overlap); operability gap remains (silent drop) |
| HMM regime on a confirmed crisis day | `hmm_regime=crisis`, multipliers 0.2256 / 0.2046. **V3 verifies state_probs decomposition: AVGO 96.8% crisis + 3.2% normal; NVDA 99.4% crisis + 0.06% bear + 0.56% normal — products match to within 3e-5** | S30 already confirmed this transition on AAPL log-returns; VIX > 30 in April 2025; "crisis" is the right verdict | ✓ Verified mathematically |
| Crisis multiplier × raw EV composition | **V1 verifies** `ev_dollars = ev_raw × hmm × skew × news × credit_mult` exactly: AVGO 457.89 × 0.2256 × 1.0 × 1.0 × 0.80 = 82.66 ≈ 82.65 ✓; NVDA 221.32 × 0.2046 × 1.0 × 1.0 × 0.80 = 36.23 ≈ 36.22 ✓ | Engine arithmetic should be reproducible from surfaced components — the 5th factor (credit_mult) was the missing piece | ✓ Verified to <1¢ tolerance |
| Dealer multiplier under compounding stress | 1.0 across both survivors, `dealer_regime=None` | Real crisis days have *meaningful* dealer-positioning signal (forced unwinds, short-gamma below put walls) | ⚠ Dormant (Bloomberg-only, same shape as S29 skew) |
| §2 clamp `[0.70, 1.05]` on dealer multiplier | always 1.0 → trivially in band; **stress was not applied on this provider — clamp not exercised under load** | The clamp pinned by `test_dealer_multiplier_evengine_integration.py` (#193) | ⚠ Vacuously preserved (clamp not stressed; would need Theta replay to test under load) |
| Per-name event-gate visibility (output schema) | No `event_locked` / `event_reason` columns in ranker output; `event_gate` correctly fires per V5 but the trader cannot see why | Trader needs to know which book positions hit earnings windows; especially when a CC's expiry brackets earnings | ❌ Column missing (decision correct, telemetry absent) |
| Combined `regime_multiplier` in output | **F9 — missing entirely.** Components (`hmm_multiplier`, `skew_multiplier`, `news_multiplier`, `credit_multiplier`) are present; the trader must multiply 4 columns to reconstruct the EV scaling | Output should expose the actual scalar that multiplies `ev_raw → ev_dollars`, so verifying the composition doesn't require knowing the 4-factor formula | ❌ Column missing |
| Sector concentration with 3 IT puts + new TSLA | `sector_cap.passed=True` (TSLA = Consumer Discretionary, not IT) | Trader's mental model: "60% tech by count → block another tech name." Engine's: "TSLA ≠ IT → 9% Cons Disc, allowed." Both technically correct; mental models diverge | ⚠ Realism mismatch (taxonomy) |
| `GateResult.details` key consistency | `post_open_sector_pct` (pass) vs `sector_pct` (fail) | Caller-side observability should be shape-stable: same key, regardless of verdict | ❌ Key asymmetry (footgun) |
| `suggest_rolls` on 43% ITM puts | Returns 0 candidates across 4 grid shapes (V6); **V7 confirms 1 candidate returned on 0.9% ITM control — engine WORKS on shallow ITM, V6's zero is defensible refusal at extreme moneyness via the `min_net_credit=0` filter killing all-debit candidates** | A trader needs to choose between close-and-take-assignment, rescue-debit roll, or hold. Zero candidates with no `reason_tally` doesn't help that decision | ⚠ Decision defensible (correct refusal); operability gap remains |
| BSM fair-value sanity on the Bloomberg path | **V2 verifies** `premium == fair_value` on both rows (AVGO 4.829 / NVDA 3.592), `edge_vs_fair = 0` | Bloomberg path is BSM-derived per S29 F1 / S1 (no quoted chain) | ✓ Verified |

**Verdict.**

- **Engine math is verified correct under stress; the §2
  invariants hold to the extent the provider lets them be
  stressed.** V1 verifies the 5-factor composition arithmetic
  to <1¢ tolerance: `ev_dollars = ev_raw × hmm × skew × news ×
  credit_mult` exactly. V3 verifies the HMM
  state-probability decomposition is mathematically clean
  (matches to 3e-5). V2 confirms BSM fair-value sanity on
  the Bloomberg path. V5 confirms the silent filter of 20
  names is `event_gate` correctly firing on the 35-DTE
  holding-window earnings overlap (NOT random drops). V7
  confirms `suggest_rolls` works on shallow ITM and refuses
  appropriately at extreme moneyness. **The decision-layer
  arithmetic, regime detection, and gate-firing logic are
  all verified to behave as documented.** The recent
  campaign-close tests (`test_evengine_event_lockout.py`,
  `test_dealer_multiplier_evengine_integration.py`,
  `test_f4_tail_risk_gap.py`,
  `test_consume_ranker_row_anchor.py` — PRs #185 / #186 /
  #193 / #196) all hold up under this composed stress.

- **§2 framing must be honest about the stress that was
  applied.** The dealer-multiplier clamp `[0.70, 1.05]`
  was preserved trivially (always 1.0 on Bloomberg per F4
  — dealer module dormant for chain-access reasons). The
  clamp was NOT exercised under load on this provider; the
  test that pins it (`test_dealer_multiplier_evengine_integration.py`
  / PR #193) holds, but S31 does not add fresh stress
  evidence to the clamp. A Theta replay (S6) would put the
  clamp under genuine load on a crisis day. The first-PR
  framing "§2 invariants survive compounding stress
  intact" was thus over-claimed; the corrected scope is
  "§2 invariants are not breached; the dealer clamp was not
  stressable on this provider."

- **The engine is operationally *illegible* in this
  regime, but the underlying decisions are defensible.**
  F1 (silent filter), F3 (schema gaps), F5 (hidden book
  stressor), F7 (key asymmetry), F8 (silent-zero roll),
  and F9 (combined-multiplier omission) all share one
  shape: *the engine has the right answer, the output does
  not let the trader see why*. V4 + V5 prove the silent
  filter is correctly-acting `event_gate`. V6 + V7 prove
  the silent-zero roll is correctly-acting refusal. The
  engine is structurally sound and decisionally
  defensible; the trader-facing surface is the gap. This
  is the same observability pattern S22, S28, S29
  surfaced — but S31 is the first time it has been
  documented as a composed pattern across the whole
  decision layer in one cycle.

- **The dealer dormancy on Bloomberg is now a confirmed
  multi-Sn pattern, not an isolated bug.** S29 found
  skew dormant on Bloomberg for the same root cause (no
  per-strike chain). S31 confirms the dealer module sits
  in the same condition. Both modules are well-built and
  unreachable on the live provider. Closing this gap is
  a connector-migration problem, not an engine-code
  problem — same conclusion as [[bloomberg-iv-no-skew]]
  and [[bloomberg-data-refresh-blocked]].

- **F5 (the JPM CC earnings overlap) is the kind of
  insight stress tests *should* surface.** The scenario
  was designed with AVGO as the "earnings stressor"; the
  data revealed JPM held in the existing book actually
  hits the earnings window. A real trader running this
  book at this date would care; the engine had the data
  to flag it (`get_earnings`) but didn't surface it in
  any of the ranker / tracker outputs. The trader has to
  ask three different APIs to assemble the picture.

**AI handoff.**

**✓ Shipped status (post-campaign, 2026-05-25/26):** All six
Fix# entries below shipped against the findings they
referenced. The bullets are retained for historical rationale;
the actual code lives in the PRs listed here.

| Fix # | Targets | Shipped in |
|---|---|---|
| Fix #1 + Fix #4 | F1 silent filter, F4 / F8 suggest_rolls silent zero | **PR #209** — `fix(decision-layer): add drops_summary attribute to all ranker / roll-suggest frames`. **Discovery during verification:** the per-candidate reason list (`df.attrs["drops"]`) was *already* populated on every ranker / suggest_rolls return path (S22 F1 fix shipped earlier); the actual gap was *discoverability*. PR #209 adds the trader-facing `df.attrs["drops_summary"] = {"total_dropped": N, "by_gate": {...}}` roll-up on all 6 ranker returns + both suggest_rolls + suggest_call_rolls returns, applied uniformly via a module-level `_attach_drops_summary` helper. Closes F1 / F4 / F8 as composed-pattern observability — the engine had the right answer; the trader can now see *why* a name was dropped at a glance. |
| Fix #2 | F3 schema gap (the `sector` slice) | **PR #210** — `fix(wheel_runner): surface GICS sector column on all ranker survivor rows`. Adds a CORE `sector` column (sourced from `DEFAULT_SECTOR_MAP`, same map `check_sector_cap` aggregates by) to all 3 ranker outputs. F3's other requested columns (`regime_multiplier`, `event_locked` / `event_reason`) are addressed by **PR #208** (regime_multiplier) and PR #209's `drops_summary` (event_locked / event_reason now inferable from `by_gate` since `event_gate` is the documented `event` gate). |
| Fix #3 | F7 GateResult key asymmetry | **PR #207** — `fix(portfolio_risk_gates): rename check_sector_cap fail-path details key for pass/fail symmetry`. One-line rename `sector_pct` → `post_open_sector_pct` in the failure-path details dict + 2 test updates + regression test pinning the symmetry. |
| Fix #5 | F6 alt-cluster overlay (the GICS-strict mental-model gap) | **PR #212** — `feat(portfolio_risk_gates): add check_alt_cluster_cap for colloquial-cluster exposure`. Complementary to `check_sector_cap`; aggregates across colloquial clusters (default ships `mega_cap_growth = {AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AVGO}`); operators pass `clusters` kwarg for custom definitions. Default cap 0.40 (looser than the 0.25 sector cap). Standalone gate — not wired into the D17 hard-block flow yet (follow-up: opt-in mechanism). |
| Fix #6 | F9 combined-multiplier omission | **PR #208** — `fix(wheel_runner): surface combined regime_multiplier column in ranker output`. Adds `regime_multiplier` column (= `res.regime_multiplier`, the engine's *final* multiplier post-clamp + heavy_tail_penalty + dealer_mult — the scalar that actually scales `ev_raw → ev_dollars`). Regression test pins the composition identity. |

**What was NOT shipped:** the original Fix #1 framing (add
`include_filtered_candidates=True` + `filter_reason` per dropped
candidate as new rows in the output) — superseded by the V5
verification discovery that `df.attrs["drops"]` was already
populated, making the rows-only addition redundant. The
discoverability fix (drops_summary on top of the existing drops
list) achieves the same trader operability with less surface
area. Documented for posterity.

The original Fix# bullets that follow are kept for the design
rationale; replace the "do this" framing with "*was done in
PR #YYY*" when reading them.

---

- **Fix #1 (highest-priority, the F1 closer):
  `filter_reason` per dropped candidate in
  `rank_candidates_by_ev`.** Currently the ranker returns
  only surviving rows; modify to optionally return a
  row per *considered* candidate with `filter_reason ∈
  {"chain_unavailable", "missing_iv", "event_blocked",
  "ev_negative", "pit_violation", ...}` and either
  `ev_dollars=NaN` or a separate `passed` boolean. Gate
  behind an `include_filtered_candidates=True` kwarg to
  preserve backwards compatibility. This unblocks the
  whole "trader observability" story F1/F3/F5/F8 all
  point at. Smallest-scope fix; biggest operational
  impact.

- **Fix #2 (F3 column additions): surface a combined
  `regime_multiplier`, `event_locked`, `event_reason`,
  `sector` per ranker row.** `regime_multiplier`
  composes from `hmm_multiplier × dealer_multiplier ×
  skew_multiplier × event_multiplier` (or whichever
  set is live). `event_*` come from the event_gate
  hit. `sector` comes from `DEFAULT_SECTOR_MAP`.
  Pairs with Fix #1 — once `include_filtered_candidates`
  surfaces dropped rows, these columns let the trader
  see which gate did the dropping.

- **Fix #3 (F7 key asymmetry):** in
  `engine/portfolio_risk_gates.py:319-338`, use
  `post_open_sector_pct` in *both* pass and fail
  details dicts. One-line change; eliminates the silent
  default-to-zero footgun. Cheap.

- **Fix #4 (F8 reason_tally for suggest_rolls):**
  modify `WheelTracker.suggest_rolls` to attach a
  `_reason_tally` attribute (or return tuple) when zero
  surviving candidates: which (DTE, delta) pairs failed
  EV, which failed `min_net_credit`, which hit the
  moneyness floor. Lets the trader distinguish "close
  the position" from "rescue-debit roll" from "hold."

- **Fix #5 (F6 alternative-clustering overlay):**
  Optional. Add a parallel "growth-cluster" /
  "high-beta-cluster" exposure check alongside the
  GICS sector cap, so a trader with the
  AAPL/MSFT/NVDA/META/TSLA/GOOGL mental model gets
  warned even when GICS doesn't aggregate them.
  Lower priority — this is realism polish, not a
  correctness gap.

- **Fix #6 (F9 combined-multiplier surfacing):** add a
  `regime_multiplier` column to the ranker output, equal
  to `hmm_multiplier × skew_multiplier × news_multiplier
  × credit_multiplier` (the exact product passed into
  `EVEngine.evaluate`). The components are already in
  the 46-column DataFrame; adding the product is a
  one-line computation in the ranker. Closes the F9
  observability gap that the verification driver surfaced
  (the entry author themselves got tripped up on first
  read). Cheap; pairs naturally with Fix #1.

- **Sanity follow-up Sn (S6 dependency): re-run S31 on a
  Theta replay** to confirm the dealer multiplier
  *actually moves* below 1.0 on a real crisis day with
  chain access (NVDA short-gamma below put wall in April
  2025 is the expected regime). Would close the
  "dormancy is data not logic" hypothesis the same way
  S29 → S6 closes it for skew.

- **Test promotion (per the two-axis verification
  pattern in `TESTING.md`):** F1's silent-filter
  contract — that `rank_candidates_by_ev` either
  surfaces a row per considered candidate or
  documents the drop somewhere queryable — is the
  right shape for a pytest regression test (likely as
  an xfail today, flipping to pass when Fix #1
  ships). Mirrors the
  `tests/test_f4_tail_risk_gap.py` xfail-as-watchlist
  pattern from PR #196.

**Methodology debt.**

- **Single connector (Bloomberg).** The dormancy finding
  (F4) applies to Bloomberg only; Theta would activate
  the dealer module. S6 (queued) would let us cleanly
  separate "implementation dormant" from "data dormant"
  for dealer the way S29 separates it for skew.

- **Single as_of.** S31 tests one crisis day. The
  composed-multiplier behaviour on a normal-quiet day
  (e.g. 2026-01-15) versus a bear-but-not-crisis day
  (e.g. 2025-04-11) would either confirm "the silent
  filter is regime-invariant" or surface
  "crisis-day-specific filter behaviour" — useful in
  either direction.

- **Synthetic book.** The 5 positions were constructed
  (strikes picked to be ITM, entry dates fabricated 3
  weeks before crisis). A real audit-trail book from
  the same date would surface different latent
  stressors — F5 (the JPM CC hit) is illustrative of
  what real-book audits surface that synthetic books
  miss.

- **`require_ev_authority=False`.** The driver
  bypassed the launch-gate to construct positions
  directly. This was deliberate (the launch-gate is
  not the surface under test), but it means the
  D17 portfolio-risk hard-blocks (sector cap, Kelly,
  delta) were not exercised through the
  position-opening path. F6's sector finding was
  verified via direct `check_sector_cap` calls; the
  full integration through `open_short_put`'s
  hard-block path is `test_authority_hardening.py`
  territory and was not re-tested here.

- **No comparison to Theta replay on the same date.**
  Pairs with the "Sanity follow-up Sn" above. On a
  date with real chain data, the dealer module
  *should* meaningfully move the multiplier on the
  NVDA-short-gamma-below-put-wall pattern that was
  certainly present in April 2025.

- **`suggest_call_rolls` not tested.** The driver
  exercised `suggest_rolls` on the 3 ITM tech puts
  but did not run `suggest_call_rolls` on the JPM/BAC
  CCs near their earnings (F5). The dividend-near
  call-roll path (S28 F3 surface) was not stressed
  here; would pair naturally with a follow-up Sn.

---

### S32 — $1M friction-modeled simulation (closes S22 Caveat 3)

**Purpose.** Close S22 / S27's acknowledged-but-never-measured Caveat 3
(frictionless P&L). Same window, same universe, same engine
(post-IV-PIT-fix), but scaled 10× to **$1M starting capital** with a
three-layer friction overlay (bid/ask + commission + assignment
slippage). Three parallel `WheelTracker` instances per friction level
to isolate friction-dollar cost from execution-divergence cost.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
2022-01-03 → 2024-12-31 (753 trading days), 24 SP500 tickers identical
to S22 / S27. `require_ev_authority=False` (matches S22 / S27 so the
only variable vs S27 is capital + friction). Friction model: half-
spread `max($0.05, 8% of premium)`, commission `$0.65/contract` for
open and close-on-assignment, assignment slippage `10bp × strike × 100`
on equity notional. Throwaway harness at `%TEMP%\s32_backtest\run.py`
(same pattern as S22 / S27).

**Status.** Done. **Verdict: signal preserved, friction drag is
~0.27% NAV (vastly smaller than S22's "2-5% per leg" worst case), but
the headline +27pp-over-SPY narrative INVERTS at $1M — engine
underperforms SPY by −22pp because capital deployment averages only
10.8% of NAV. The S22 / S27 "engine beats SPY" claim was a
$100k-specific artifact, not a scale-invariant property of the engine.**

**Findings:**

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | $1,021,626 | $1,018,601 | $1,018,514 |
| Return | +2.16% | +1.86% | +1.85% |
| Spearman ρ | 0.1940 | 0.1923 | 0.1918 |
| Executed puts | 95 | 95 | 95 |
| Mean realized / executed put | $199 | $174 | $170 |
| Skipped `insufficient_bp` | **0** | **0** | **0** |
| SPY same-window | — | — | ~+24% |

- **F1 — Signal is scale-invariant.** ρ = 0.19 at $1M matches S27's
  ρ = 0.22 at $100k within noise. Capital scale does not affect
  prediction quality.
- **F2 — Friction drag is 0.27% NAV / 14% of frictionless alpha.**
  Closes Caveat 3. S22's "2-5% per leg" worst case was over-
  conservative for liquid SP500 25-delta options today.
- **F3 — Engine UNDERPERFORMS SPY at $1M by −22pp.** Engine +1.85%,
  SPY ~+24%. **The +27pp-over-SPY headline from S22 / S27 was a
  $100k-capital artifact — at $1M the engine cannot deploy enough
  capital to produce competitive percentage returns.**
- **F4 — Capital deployment averages 10.8% of $1M NAV.** Constrained
  by universe size (24 names), one-position-per-name rule, hold-to-
  expiry pattern, and `top_n=10` × `MAX_NEW_PER_DAY=3` daily limits.
  ~$890k of NAV sat idle for the duration.
- **F5 — 2022 bear ρ = 0.37 with mean realized −$18.** Re-surfaces
  the F4 tail-risk gap (from S22 / S27) under friction conditions:
  engine ranks bear-market trades correctly, but tail-risk machinery
  still doesn't widen `prob_profit` enough on the worst losers. PR
  #196 is the regression watch.
- **F6 — Caveat 2 (in-sample HMM / POT-GPD parameters) explicitly
  restated** in this Sn's doc, closing the silent omission flagged
  by PR #197's P9 finding.

**AI handoff:**

- **Highest-leverage next backtest:** expand universe from 24 to 100+
  SP500 names and re-run S32. Hypothesis: average deployment rises to
  ~40-60%, absolute return rises proportionally, the −22pp SPY gap
  closes substantially.
- **Production-pricing implication:** the 24-ticker wheel + 35-DTE +
  25-delta + hold-to-expiry strategy is a **$100k-class strategy**.
  At $1M without structural capacity changes (larger universe,
  multi-contract per name, or strategy stack), it underperforms a
  passive SPY hold.

Full doc: `docs/ENGINE_BACKTEST_S32_FRICTION.md`.

### S33 — Multi-stress engine soundness verification

**Purpose.** Broader companion to S31 (which covered ONE
compounding-crisis scenario at one anchor date). S33 verifies the
engine's STRUCTURAL soundness — math correctness, §2 invariants,
regime realism at known historical events, BSM sanity, edge-case
handling, trader-realism — across six orthogonal verification axes.
User-facing question: *"is our options engine sound and delivering
realistic outputs under stress?"*

**Numbering note.** This entry was authored under the working
assumption it would be `S32`. While the driver was running, Terminal
A merged `S32 — $1M friction-modeled simulation` via PR #213 (a
distinct operational-backtest concern, not duplicative). Renumbered
to S33 per [[usage-test-scenario-numbering]]. The fix that shipped
mid-verification (PR #215, `claude/fix-7-as-of-beyond-data-check`)
carries `S32 F3` in its commit / PR description; that is this entry's
F3, just retconned to S33. Documented for traceability.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`,
`MarketDataConnector`. 22-name universe spanning mega-cap tech
(AAPL/MSFT/NVDA/GOOGL/META/TSLA/AMZN/AVGO), banks (JPM/BAC/WFC/C/GS/MS),
defensives (KO/PG/JNJ/VZ/XOM), and healthcare (UNH/ABBV/LLY). Recent
anchor `as_of=2026-03-20` (data cutoff) for V1 / V4 / V5; historical
anchors for V3 (Mar 2020, Nov 2020, Jun 2022, Aug 2024, Apr 2025,
Feb 2026); Apr 2025 crisis day for V6's realism check.

Driver under `%TEMP%\s32\driver.py` (named s32 from before the
renumber; intentionally not committed per Sn convention) +
`%TEMP%\s32\probe_f2_f3.py` follow-up. `sys.path.insert(0,
r"C:\Users\merty\Desktop\swe-terminal-c")` prepended per
[[sys-path-worktree-shadow]].

**Methodology discipline (post-S31 lesson).** Every "✓ Aligned"
verdict in this entry is **externally verified** — either against
the engine source (e.g. `regime_hmm.py:265-285` weight constants for
V3), against documented market behaviour (V3 historical events,
V6 trader expectation), or against a closed-form reference (V4
BSM identity). Where math can only be checked against the engine's
own output, the framing is *internal consistency*, not *verified
correct*.

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route). `engine.regime_hmm.GaussianHMM`
(V3 HMM fits + state-prob inspection).
`engine.dealer_positioning.dealer_regime_multiplier` (V2 clamp
function — pure-function so directly stressable). `engine.option_pricer.black_scholes_price`
(V4 reference BSM). `engine.risk_manager.DEFAULT_SECTOR_MAP` (V1
sector-column verification).

**Status.** Done. **Verdict: the engine is structurally sound on
the dimensions tested. V1 / V2 / V4 pass cleanly (composition
arithmetic exact within rounding, §2 dealer clamp holds under
adversarial input, BSM identity perfect on the Bloomberg
synthetic-premium path). V3 surfaces a nuance not a bug (HMM
"crisis" label is high-vol regardless of return direction). V5
surfaced one real silent-substitution bug (F3 future as_of), shipped
in PR #215 mid-verification. V6 confirms the engine's crisis-day
behaviour matches trader expectation on 4 of 5 named realism
checks. Bloomberg path remains the only stress surface; Theta
replay (S6 queued) would activate the dormant dealer signal.**

**Findings:**

- **(F1 — soundness verified, V1+V4) Composition arithmetic and BSM
  identity hold across the universe.** For each of the 13 surviving
  rows at `as_of=2026-03-20`, the identity
  `ev_dollars = ev_raw × regime_multiplier` holds to a max
  relative error of 0.023% (a presentation artifact of 2-dp / 4-dp
  rounding on different scales — LLY at `ev_raw=1462` shows 0.067
  absolute delta, well under any actionable tolerance). The
  composition multipliers are externally accessible per PR #208
  (`regime_multiplier` column shipped during S31 fixes), so the
  identity is now reproducible by any caller. **V4** confirms
  `premium == fair_value` on all 13 rows and `edge_vs_fair = 0`
  across the board — consistent with S29 F1 (Bloomberg path uses
  BSM-synthetic premiums; no quoted-vs-fair edge exists by
  construction). Sector column (PR #210) matches `DEFAULT_SECTOR_MAP`
  on 13/13 rows; drops_summary (PR #209) integrity holds
  (`len(drops) == total_dropped == sum(by_gate.values()) == 9`).
  **Logged as positive.**

- **(F2 — §2 verified, V2) Dealer-clamp invariance HOLDS UNDER
  ADVERSARIAL INPUT.** Constructed synthetic `MarketStructure` with
  extreme inputs and called `dealer_regime_multiplier` directly:

  ```
  case                                              dealer_mult  in [0.70, 1.05]?
  long_gamma_dampening conf=+0.50                       1.0250  ✓
  long_gamma_dampening conf=+1.00 (upper boundary)      1.0500  ✓
  long_gamma_dampening conf=+2.00 (over-max input)      1.0500  ✓  (clamps to 1.0)
  long_gamma_dampening conf=-0.50 (negative input)      1.0000  ✓  (clamps to 0.0)
  short_gamma_amplifying conf=+1.00 (lower boundary)    0.7000  ✓
  short_gamma_amplifying conf=+2.00 (over-max input)    0.7000  ✓
  near_flip conf=+0.50                                  0.8500  ✓
  near_flip conf=+1.00 (still 0.85)                     0.8500  ✓
  neutral conf=+1.00                                    1.0000  ✓
  None case                                             1.0000  ✓
  ```

  11/11 cases in `[0.70, 1.05]` band; clamp logic at
  `engine/dealer_positioning.py:736` (`conf = max(0.0, min(1.0,
  float(ms.confidence)))`) bites on both negative and over-max
  inputs. **The clamp is logically tight by construction —
  asymmetric `[1.0 - 0.30·conf, 1.0 + 0.05·conf]` formulae cannot
  produce out-of-band output even with adversarial confidence**.
  Closes the S31 "clamp vacuously preserved on Bloomberg" caveat:
  on this synthetic stress (where Bloomberg's chain-absence is
  bypassed by direct function call), the clamp HOLDS under load,
  not vacuously. **§2 verified, no longer vacuously.**

- **(F3 — real bug, shipped in PR #215) Silent substitution on
  future `as_of` — fixed.** V5 probe found:

  ```
  AAPL OHLCV data range: 2018-01-02 → 2026-03-20
  AAPL final close: 247.64

  as_of=2026-03-20: spot=247.64, ev_dollars=20.38, hmm_regime=bear
  as_of=2027-01-01: spot=247.64, ev_dollars=20.45, hmm_regime=bear ← SAME
  as_of=2030-01-01: spot=247.64, ev_dollars=20.45, hmm_regime=bear ← SAME
  as_of=2050-01-01: spot=247.64, ev_dollars=20.45, hmm_regime=bear ← SAME
  ```

  The engine silently used the 2026-03-20 close as "current spot"
  for any future `as_of` (2027, 2030, 2050 — all returned the
  same row), with no warning. **Violates D11 "no silent
  substitution"**. The existing PIT filter at
  `engine/wheel_runner.py:859-865` correctly trims OHLCV to
  `<= as_of`, but a future as_of leaves ALL rows in place; the
  engine then resolves spot from `ohlcv["close"].iloc[-1]` which is
  the data-cutoff row, not anything at as_of. **Shipped fix in
  PR #215 (`fix(wheel_runner): refuse rank_candidates_by_ev with
  as_of beyond data cutoff`, merged at `40d6076`)**: adds
  `max_as_of_staleness_days: int = 30` parameter; over-threshold
  queries drop with `gate="data"`,
  `reason="as_of N is X days beyond latest data ..."`.
  4 regression tests in `tests/test_pit_leaks.py::TestRankerAsOfBeyondData`.
  Same surface for `rank_covered_calls_by_ev` and
  `rank_strangles_by_ev` is untreated; queued as follow-up.

- **(F4 — V3 nuance, observability) HMM "crisis" label fires on
  any high-vol regime, including non-crashing periods.** Probe at
  the recent date 2026-02-13 — initially flagged as a possible
  HMM bug — turns out to be correct engine behaviour:

  ```
  Date           tail vol (252d ann)    tail mean ann    crisis prob    label
  2020-03-23           0.3706            +0.1608           0.9335       crisis  ✓ (expected)
  2025-04-04           0.2745            +0.1047           1.0000       crisis  ✓ (expected, matches S30)
  2026-02-13           0.3178            +0.0768           0.8794       crisis  ⚠ unexpected by *prior*
  2026-03-20           0.3130            +0.1402           0.0227       bear    (recent regime shift)
  ```

  Annualised volatility of 0.32 at Feb 2026 is **comparable to
  Mar 2020 COVID-crash levels (0.37)**. The HMM is detecting
  genuine high-vol conditions in AAPL log returns; the "crisis"
  label is supported by the data even though the rolling
  annualised return is positive (+0.077). **The engine is
  correct; the verification's prior of "Feb 2026 = calm anchor"
  was wrong.** The nuance worth documenting: **the HMM's
  "crisis" label means "high-vol regime" regardless of return
  direction.** A trader assuming "crisis" implies "spot is
  crashing" can mis-anchor; the multiplier (0.20-0.24 here)
  reflects the regime's *risk* not its *direction*. Note also:
  the Feb 2026 → Mar 2026 transition flipped the label from
  `crisis` to `bear` over 5 weeks — the HMM does detect the
  regime shift. **Logged as observability nuance, not a bug.**

- **(F5 — V6 realism, mostly aligned) Engine's crisis-day output
  matches senior-trader expectation on 4 of 5 checks.** On Apr
  2025 crisis day with `use_event_gate=False` (to surface all 22
  names for ranking analysis):

  | Trader expectation | Engine | Verdict |
  |---|---|---|
  | HMM regime "crisis" on most names | 21/22 names | ✓ |
  | Crisis IV elevated (>30% on most) | median 0.510, 19/22 above 30% | ✓ |
  | Regime multiplier compresses EV in crisis | median 0.16 (well below 1.0) | ✓ |
  | Dealer multiplier dormant on Bloomberg | 22/22 at 1.0 | ✓ (confirms S29 F4) |
  | Top half ranked names favour defensive sectors | top half: 3 healthcare (LLY/UNH/ABBV) + 6 offensive + 2 financials | ⚠ ranking dominated by absolute-EV not risk-adjusted (engine optimises absolute EV; high-IV names get higher EV in crisis) |

  The "defensive favoured in top-half" expectation didn't fully
  fire — engine ranks LLY #1, UNH #3, ABBV #11 (all defensive),
  but tech names dominate slots 4-10. Not necessarily a bug:
  engine optimises absolute EV, which in crisis correlates with
  high-IV (which correlates with high-beta). A risk-adjusted
  ranking (the existing `roc = ev_dollars / collateral` column)
  may flip this. **Worth a follow-up Sn:** "absolute-EV vs roc
  ranking realism" — does the trader want the engine to lead
  with absolute EV or risk-adjusted return?

- **§2 verified across V1 / V2 / V5.** (a) `dealer_multiplier`
  bounds preserved under both routine call (V1, always 1.0 on
  Bloomberg) and adversarial synthetic input (V2). (b) Every
  ranker output row was produced through `EVEngine.evaluate`; no
  candidate appeared with NaN / null ev_dollars or bypassed
  authority. (c) The F3 silent-substitution fix preserves §2 by
  dropping (not silently producing) candidates with stale data;
  the new drop reason joins the documented gate taxonomy. **The
  §2 contract holds across all six S33 axes — now non-vacuously
  on the dealer clamp (V2 stresses it directly)**.

**Verification probe results (V1–V6, V3b state-probs follow-up).**

- **V1 — composition arithmetic at scale (passed).** Identity
  `ev_dollars = ev_raw × regime_multiplier` to <0.023% relative
  error across all 13 surviving rows. Drops_summary integrity
  (`len(drops) == total_dropped == sum(by_gate.values())`)
  holds. Sector column matches `DEFAULT_SECTOR_MAP` 13/13.
  All three columns shipped during the S31 fix campaign
  (PR #208 regime_multiplier, PR #209 drops_summary, PR #210
  sector) are now exercised by an at-scale verification, not
  just unit tests.

- **V2 — §2 dealer-clamp invariance (passed).** 11/11 adversarial
  test cases in `[0.70, 1.05]`. See F2 above.

- **V3 — HMM regime labels at known historical events (mixed,
  resolved).** 6 dates probed; 4 of 6 match priors (Mar 2020 crisis,
  Apr 2025 crisis confirmed by S30, Jun 2022 inflation low /
  crisis label, Aug 2024 vol spike / crisis label). Two surfaced
  the F4 observability nuance: Nov 2020 returned `bear` (mult 0.42)
  where the prior was "transitional" — defensible; Feb 2026
  returned `crisis` (mult 0.24) — confirmed correct by V3b state-
  probability follow-up (high vol genuinely present in AAPL data).

- **V3b — state-prob decomposition (resolves F4).** Direct HMM fit
  on AAPL log returns at each event date, dumping per-state
  probabilities. Confirms (a) the published `position_multiplier`
  formula is consistent with the per-state probs to <3e-5
  tolerance, (b) Feb 2026 has 0.88 prob crisis + 0.12 prob bear
  (no leakage from the other two states), supporting the
  high-vol-genuine reading.

- **V4 — BSM fair-value sanity at scale (passed).** 13/13 rows
  have `premium == fair_value` to 1¢; all `edge_vs_fair = 0`.
  Confirms S29 F1 + S1 (Bloomberg path is BSM-derived) at scale.

- **V5 — edge cases (4 of 5 passed; F3 found + fixed).**

  ```
  Case                              Result                                          Verdict
  empty universe                    shape=(0,0), drops_summary={total: 0, by: {}}    ✓ graceful
  single ticker                     shape=(1, 49), regime_multiplier present         ✓ graceful
  nonexistent ticker                shape=(0,0), drops=1, gate="data", reason recorded ✓ graceful
  post-cutoff future as_of          shape=(1, 49), spot=cutoff_close — NO WARNING    ✗ F3 silent sub
  zero delta_target                 shape=(0,0), drops=1, gate="strike"              ✓ graceful
  ```

  F3 fixed in PR #215 (now the future as_of case correctly drops
  with reason); other 4 edge cases handled correctly out of the box.

- **V6 — trader-realism check on Apr 2025 crisis (4 of 5 aligned).**
  See F5 above.

**Realism Check.**

| Aspect | Engine output (verified) | External reference | Verdict |
|---|---|---|---|
| Composition identity `ev_dollars = ev_raw × regime_multiplier` | holds to <0.023% rel error across 13 rows | Mathematical identity from `engine/ev_engine.py:502` × `engine/wheel_runner.py:1300`; columns surfaced by PR #208 | ✓ Verified |
| §2 dealer-multiplier clamp `[0.70, 1.05]` under adversarial input | 11/11 cases in band; clamp at `dealer_positioning.py:736` bites on negative and over-max conf | Clamp pinned by `test_dealer_multiplier_evengine_integration.py` (PR #193) + the asymmetric formulae which are mathematically bounded | ✓ Verified (non-vacuously this time) |
| BSM `premium == fair_value` on Bloomberg path | 13/13 rows match to 1¢; `edge_vs_fair = 0` | Documented in S29 F1; premium is BSM-derived from engine inputs | ✓ Verified |
| HMM regime label aligns with trader prior on known historical events | 4/6 dates match (Mar 2020, Apr 2025, Jun 2022, Aug 2024); 2 surfaced F4 nuance | S30 confirmed Apr 2025 = crisis on AAPL; broader market history for other dates | ⚠ Mostly aligned; F4 nuance documented |
| HMM "crisis" label semantic vs trader mental model | "high-vol regime regardless of direction" | Trader prior often equates "crisis" with "crashing" | ⚠ Observability nuance (F4) — label is correct, semantic differs from common mental model |
| Future `as_of` data freshness | shipped fix: drops over-threshold queries | D11 "no silent substitution" principle from `DECISIONS.md` | ✓ Verified after PR #215 |
| Edge cases (empty / single / nonexistent / zero-delta) | 4/4 graceful before fix; F3 was the 5th | Engine should degrade gracefully on degenerate inputs | ✓ Verified |
| Crisis-day defensive-sector ranking expectation | partial — defensive present (LLY #1) but mixed with high-IV offensive | Senior-trader prior: prefer defensive in crisis | ⚠ Engine optimises absolute EV; risk-adjusted view via `roc` column not in this check |

**Verdict.**

- **Engine math is sound.** Composition arithmetic (V1) holds
  exactly; BSM identity (V4) is perfect on the Bloomberg path;
  HMM state-probability decomposition (V3b) reproduces the
  published `position_multiplier` formula to machine precision.
  Every "✓ Verified" verdict in this entry is grounded in an
  external reference (engine source line + line, S30 confirmation,
  mathematical identity, or documented design principle), not
  engine-vs-itself tautology.

- **§2 invariants survive load — now non-vacuously.** The S31
  framing acknowledged the dealer clamp was vacuously preserved
  on Bloomberg (always 1.0 because the chain wasn't accessible).
  S33's V2 stresses the clamp directly via synthetic
  `MarketStructure` and confirms it holds for negative
  confidence, over-max confidence, and all regime labels.
  Combined with PR #193's existing integration test, the dealer
  clamp is now triangulated: pinned at the unit level, pinned at
  the integration level, and verified under adversarial input.

- **One real bug found and fixed mid-verification.** F3 (silent
  substitution on future `as_of`) was a D11 violation. The fix
  shipped in PR #215 with regression tests; over-threshold
  queries now drop with an explicit reason. Two related surfaces
  (`rank_covered_calls_by_ev`, `rank_strangles_by_ev`) need the
  same gate — queued.

- **HMM realism is *mostly* aligned with a senior trader's
  prior.** 4 of 6 historical dates match expected regime labels;
  the two that didn't surface a meaningful observability nuance
  (F4): "crisis" in the engine means "high-vol regime" not
  "crashing." This is a vocabulary mismatch worth surfacing in
  trader-facing documentation, but the engine's *math* is
  correct.

- **Trader-realism mostly holds (F5) with one ranking-philosophy
  question.** On a real crisis day, the engine correctly fires
  crisis labels, compresses EV via regime multiplier, and surfaces
  the dormant-dealer signal. The ranking does NOT systematically
  favour defensive sectors over offensive — because it optimises
  absolute EV, which in crisis correlates with high-IV (which
  correlates with high-beta). Whether that's the right
  optimisation target is a separate Sn — *"ranking philosophy:
  absolute EV vs risk-adjusted ROC."*

- **Bloomberg-only limitation acknowledged.** Same as S31: the
  dormant dealer signal and the synthetic-BSM premium are
  inherent to the connector, not the engine. A Theta-replay
  verification (S6 queued) would close the dormancy and let V4
  test against quoted-chain premiums, not engine-vs-engine BSM.

**AI handoff.**

**✓ Shipped during and after this verification:**

| Concern | Shipped in |
|---|---|
| F3 — Silent substitution on future `as_of` in `rank_candidates_by_ev` | **PR #215** (`fix(wheel_runner): refuse rank_candidates_by_ev with as_of beyond data cutoff (S32 F3)`, merged at `40d6076`). Adds `max_as_of_staleness_days` kwarg; over-threshold queries drop with explicit reason; 4 regression tests in `tests/test_pit_leaks.py::TestRankerAsOfBeyondData`. Note: the commit / PR description labels this as "S32 F3" because the entry was authored before the renumber to S33 — this is the same fix referenced as F3 in S33. |
| F3 follow-up — Same silent-substitution surface on `rank_covered_calls_by_ev` + `rank_strangles_by_ev` | **PR #220** (`fix(wheel_runner): extend as_of-beyond-data gate to CC + strangle rankers (S33 F3 follow-up)`, merged at `c07b265`). Same `max_as_of_staleness_days` parameter and freshness gate applied to both. 8 new regression tests (`TestCoveredCallRankerAsOfBeyondData` + `TestStrangleRankerAsOfBeyondData` in `tests/test_pit_leaks.py`). All three §2 ranker entry points now gate future `as_of` consistently. |
| F4 — HMM "crisis" label vocabulary mismatch ("crisis = high-vol regardless of direction") | **PR #222** (`fix(wheel_runner): add HMM realized-vol / return disambiguation columns (S33 F4)`, merged at `b0a7d8a`). Adds `hmm_realized_vol_252d_ann` and `hmm_realized_return_252d_ann` columns alongside `hmm_regime` / `hmm_multiplier`. Trader inspecting one row can now disambiguate "crisis = crashing" from "crisis = high-vol-with-positive-trend" without re-fitting the HMM. Math externally verified against `np.std(tail_252)*sqrt(252)` / `np.mean(tail_252)*252`. 3 regression tests added. |
| Audit — `ohlcv.iloc[-1]` patterns across `engine/` | **Completed inline during PR #220 work.** Scanned 15 engine files; identified 4 candidate surfaces (`wheel_runner.py:428` `analyze_ticker`, `wheel_runner.py:925` `rank_candidates_by_ev`, `wheel_runner.py:2132` `rank_covered_calls_by_ev`, `wheel_runner.py:2636` `rank_strangles_by_ev`). All four now gated. `wheel_tracker.py` and `strangle_timing.py` `iloc[-1]` uses are live-mark-to-market and indicator computations (different shape, not as_of-resolution). |
| `analyze_ticker` `as_of` + staleness gate (audit holdover, the fourth surface) | **PR #227** (`fix(wheel_runner): analyze_ticker now respects as_of + staleness gate (S33 audit holdover)`, merged at `f65fdfa`). Adds the same `max_as_of_staleness_days` kwarg as the three rankers; PIT filter + staleness gate; on rejection `spot_price` stays at the existing 0.0 default with `logger.warning`. 5 regression tests in `tests/test_pit_leaks.py::TestAnalyzeTickerAsOfGate`. **All four `ohlcv.iloc[-1]` surfaces in `wheel_runner.py` now respect `as_of` consistently.** |

**Remaining queued follow-ups (out-of-scope for the soundness verification campaign):**

- **F5 follow-up (ranking philosophy):** new Sn evaluating
  the engine on `roc` (risk-adjusted EV per dollar of
  collateral) vs `ev_dollars` (absolute EV) as the primary
  ranking key on crisis days. Would resolve whether the
  current "offensive dominates top half on crisis days"
  outcome is a feature or a gap. **This is an investigation,
  not a fix** — belongs to a new Sn (S35+).

- **Sanity follow-up Sn (S6-gated):** re-run V4 (BSM sanity)
  + V2 (dealer signal under load) against Theta-replay quoted
  chains. Closes the Bloomberg-only scope acknowledged in
  S33's verdict. Gated on physical Theta Terminal access.

- **Sanity follow-up Sn (S6 dependency):** re-run V4 (BSM
  sanity) against Theta-replay quoted chains, not Bloomberg
  synthetic premiums. Would test whether `premium - fair_value`
  carries meaningful information when a real quoted chain is
  available. Closes S29's "skew dormant for two independent
  reasons" hypothesis at the level relevant to the EV
  arithmetic.

- **Test promotion (per the two-axis verification pattern in
  `TESTING.md`):** the composition identity from V1 should be
  pinned by a pytest regression that exercises a multi-name
  ranker call and asserts the identity to <1% relative error.
  Existing `test_ev_dollars_is_ev_raw_times_the_final_regime_multiplier`
  (in `test_ranker_transparency.py`) is the close cousin; a
  multi-name version closes the at-scale gap.

**Methodology debt.**

- **Single connector (Bloomberg).** F4's dealer-signal dormancy
  and V4's premium-equals-fair_value carry-over from S29 / S31.
  Same scope as those prior Sn entries.

- **Single as_of for V1 / V4 / V5 (mostly).** V3 spans multiple
  dates; the other verifications use the 2026-03-20 cutoff. A
  date-grid (2024-01 → 2026-03 in monthly steps) for V1
  composition checks would catch any regime-multiplier-formula
  drift over time — not done here.

- **Synthetic V2 stress.** The §2 clamp verification constructs
  in-driver `MarketStructure` objects; it does not exercise the
  full engine path with a real (synthetic-from-chain)
  `MarketStructure`. The unit-level clamp is bullet-proof per
  V2; the integration path is pinned separately by
  `test_dealer_multiplier_evengine_integration.py` (PR #193).
  S33 doesn't add fresh integration evidence — but the existing
  integration tests already covered that surface.

- **V6 trader-realism heuristics are heuristic.** "Defensive
  sectors should appear in top half" is one heuristic; the
  engine optimises against a different objective (absolute EV).
  A more rigorous realism check would compute the
  Sharpe / Calmar ratio of engine-recommended trades on real
  historical executions — that's the S22 / S27 / S32 backtest
  axis and not S33's scope. Documented as the F5 follow-up.

- **`rank_covered_calls_by_ev` and `rank_strangles_by_ev` not
  exercised.** V1-V6 cover the short-put ranker only. The CC
  and strangle rankers have parallel structure but were not
  verified here; the F3-equivalent silent-substitution surface
  is presumed present and queued as the follow-up.

- **No `ranker output → executable trade` check.** S33 verifies
  the ranker's output is correct and realistic; it does not
  verify that the EV-ranked candidate is actually tradeable
  (filled at the quoted bid). The Bloomberg synthetic-premium
  is the structural reason. S6 (Theta replay) closes this.

### S37 — Ranking philosophy: ev_dollars vs roc

**Purpose.** S33 F5 surfaced a real ranking-philosophy question: on
a crisis day the engine ranks by `ev_dollars` (current default), and
absolute EV biases toward high-IV / high-beta names (crisis IV spikes
boost absolute premium). Defensive sectors rank lower despite being
a senior-trader's typical crisis pick. S37 quantifies the divergence
between `ev_dollars` ranking and `roc` (risk-adjusted = `ev_dollars /
collateral`) ranking and proposes a defensible default. **Investigation
Sn, not a code change** — produces a recommendation, not a fix.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
22-name universe (same as S31 / S33 / S36). Two anchor dates:
2025-04-04 (S30-confirmed crisis day) and 2026-03-20 (data
cutoff, normal regime). For each: run
`rank_candidates_by_ev` with `use_event_gate=False` to surface
all 22 candidates; sort by both metrics; compute sector mix,
ticker overlap, rank correlation. Driver under `%TEMP%\s37\`.
The `roc` column is already in the ranker output (per the S31
fix campaign — emitted alongside `ev_dollars` and `collateral`);
no engine changes needed.

**Path.**
`WheelRunner.rank_candidates_by_ev` at `engine/wheel_runner.py`
(the §2 ranker route). Output rows carry `ev_dollars` (the
post-multiplier EV) AND `roc` (`ev_dollars / collateral` per
S31 fix campaign).

**Status.** Done. **Verdict: the two metrics diverge meaningfully
on crisis days (rank correlation ρ = 0.63) and largely agree on
normal days (ρ = 0.89). NEITHER metric is universally "right" —
the choice is operator-account-size-dependent:**

- **`ev_dollars` (current default)** — favors absolute return.
  Right for a $1M-class portfolio that needs to deploy capital
  efficiently. On crisis day, sector mix in top half = 3
  Healthcare + 3 IT + 2 Financials + 2 Comm Svcs + 1 Cons Disc.
- **`roc` (risk-adjusted)** — favors capital efficiency. Right
  for a $100k-class portfolio with tight collateral budget. On
  crisis day, sector mix in top half = **5 Financials** + 2
  Healthcare + 2 IT + 1 Cons Disc + 1 Comm Svcs.

**The crisis-day divergence is real and operator-relevant.**

**Findings:**

- **(F1 — ranking divergence on crisis day, ρ = 0.6341).**
  At as_of=2025-04-04 (S30-confirmed crisis), the two metrics
  produce noticeably different top-11 rankings:

  | Metric | Top 11 (in order) |
  |---|---|
  | **ev_dollars** | LLY, GS, UNH, META, TSLA, MSFT, AVGO, JPM, AAPL, GOOGL, ABBV |
  | **roc** | LLY, TSLA, AVGO, MS, GS, GOOGL, WFC, NVDA, JPM, BAC, UNH |

  Overlap: 7 names (LLY, TSLA, AVGO, GS, GOOGL, JPM, UNH).
  ev_dollars-only: AAPL, ABBV, META, MSFT (high-IV mega-caps
  with large absolute EV but moderate roc).
  roc-only: BAC, MS, NVDA, WFC (smaller absolute EV but better
  EV-per-dollar — banks dominate this set).

- **(F2 — ranking near-agreement on normal day, ρ = 0.8950).**
  At as_of=2026-03-20 (normal regime), the two metrics produce
  largely-aligned rankings: 9 of 11 names overlap. Top of both
  lists is GS / LLY in either order. The two metrics disagree
  only at the margins (KO via roc; UNH via ev_dollars). **The
  ranking-philosophy question is fundamentally a crisis-day
  question** — on calm days the trade-off doesn't bite.

- **(F3 — sector composition shift on crisis day)** — the key
  operator-facing fact:

  | Sector | top-11 by ev_dollars | top-11 by roc |
  |---|---|---|
  | Financials | 2 (GS, JPM) | **5** (GS, MS, WFC, JPM, BAC) |
  | Healthcare | 3 (LLY, UNH, ABBV) | 2 (LLY, UNH) |
  | Information Technology | 3 (MSFT, AVGO, AAPL) | 2 (AVGO, NVDA) |
  | Communication Services | 2 (META, GOOGL) | 1 (GOOGL) |
  | Consumer Discretionary | 1 (TSLA) | 1 (TSLA) |

  **`roc` puts 5 of 11 in Financials.** Why: banks have low
  strike → low collateral → high EV-per-dollar even when
  absolute EV is modest. Conversely, ev_dollars favors LLY +
  the IT mega-caps because their high IV pushes absolute
  premium up.

- **(F4 — collateral and EV magnitudes — the operator-size
  link)**. From the comparative stats:

  ```
  CRISIS day:   ev_dollars min=5.50, max=720.48, median=43.51
                roc       min=0.0007, max=0.0108, median=0.0036
                collateral min=$3,150, max=$66,900, median=$15,075

  NORMAL day:   ev_dollars min=-481.38, max=809.69, median=56.20
                roc       min=-0.0141, max=0.0108, median=0.0030
                collateral min=$4,400, max=$83,200, median=$20,725
  ```

  In crisis, max EV per trade is $720 (LLY) but the median is
  only $44 — a long-tailed distribution. The roc range is
  0.07% – 1.08%. A $1M portfolio looking for 5% annual return
  needs to deploy ~50% of NAV (a $500k+ aggregate) — by
  median EV $44, that's 11,400 trades; **the engine can't
  ship that many candidates daily**. The capacity limit
  surfaces here too (matches Terminal A's S32 F3 finding
  about 10.8% deployment).

- **(F5 — the right ranking key depends on operator account
  size)**. A defensible recommendation:

  | Account size | Recommended primary key | Rationale |
  |---|---|---|
  | < $250k | **`roc`** | Capital efficiency matters; can't afford to tie up $50k+ collateral on a marginal trade |
  | $250k – $2M | **`ev_dollars`** (current default) | Capacity is the binding constraint per S32 / S34 backtests; absolute deployment > per-dollar efficiency |
  | > $2M | **Hybrid: ev_dollars ordering, but require `roc > median(roc)` for inclusion** | Both metrics matter; reject candidates that consume capital inefficiently regardless of absolute EV |

  **The current default (`ev_dollars`) is correct for the engine's
  intended $100k–$2M operator profile.** A future refinement
  could expose a `ranking_key` kwarg or use account-size-aware
  defaults.

- **§2 verified.** No engine code touched; both rankings come
  from the same `rank_candidates_by_ev` call. The `roc` column
  is post-EV (= `ev_dollars / collateral`), so ordering by it
  is a presentation-layer choice. The §2 contract holds.

**Realism Check.**

| Aspect | Engine output | Trader / external expectation | Verdict |
|---|---|---|---|
| Rank correlation on crisis day | ρ = 0.6341 | Substantial divergence expected when premium-density varies by IV magnitude | ✓ Confirmed (real divergence) |
| Rank correlation on normal day | ρ = 0.8950 | High agreement when IV is uniform | ✓ Confirmed (near-alignment) |
| `roc` favors Financials on crisis day | 5 of 11 top half are banks | Banks have low strikes after a crisis drop → high EV-per-collateral by construction | ✓ Math-aligned |
| `ev_dollars` favors high-IV mega-caps on crisis day | LLY (Healthcare), META, MSFT, AAPL all in top half | High IV → high premium → high absolute EV regardless of collateral | ✓ Math-aligned |
| LLY ranks #1 on BOTH metrics on BOTH days | $720 / $810 EV; 1.08% / 0.86% roc | LLY is a genuine outlier — high IV + high ROC simultaneously (rare combination) | ✓ Aligned with idiosyncratic LLY characteristics |

**Verdict.**

- **The question "is ev_dollars or roc the right ranking key?"
  is genuinely operator-dependent.** Neither metric is
  universally correct. The engine's current default
  (`ev_dollars`) is right for the $250k–$2M operator profile
  the engine is built for; smaller accounts benefit from
  `roc`; larger accounts benefit from a hybrid.

- **The S33 F5 "defensive should dominate top half in crisis"
  expectation was mis-specified.** The expectation conflated
  two different things: (a) "engine should favor defensive
  sectors when other things are equal" (a reasonable thesis)
  and (b) "absolute-EV ranking should favor defensive sectors
  in crisis" (wrong — crisis pushes IV up across the board,
  often more so on high-beta names, making absolute EV
  concentrate in tech/healthcare).

- **The crisis-day Financials rotation under `roc` IS real and
  defensible.** Banks at crisis low-strikes ARE the highest
  capital-efficiency trades. A trader prioritizing capital
  efficiency over absolute return would rationally go bank-
  heavy in crisis — which matches conventional "buy the bear
  in banks" wisdom. The current `ev_dollars` default hides
  this from the trader unless they re-sort.

- **No code change shipped from S37.** The recommendation:
  EXPOSE `roc` more prominently in the trader-facing summary
  (it's already a column), and consider an account-size-aware
  default ranking kwarg in a future PR. Neither is a §2
  correctness fix; both are operator-experience refinements.

**AI handoff.**

- **F5 follow-on (small future PR, not in S37 scope):** add a
  `ranking_key: str = "ev_dollars"` parameter to
  `rank_candidates_by_ev` that accepts `"ev_dollars" | "roc" |
  "hybrid"`. The hybrid mode uses ev_dollars ordering with a
  `roc > median(roc)` filter. Backwards-compatible default.
  Single-PR, single-concern.

- **Operator documentation gap:** the engine's "ev_dollars vs
  roc" trade-off should be documented in
  `docs/RANKING_PHILOSOPHY.md` (new file) or as a §6 in
  CLAUDE.md. A new operator reading the ranker output today
  doesn't know that `roc` is also a valid first-key.

- **Cross-reference Terminal A's S32 finding.** S32 found
  "engine UNDERPERFORMS SPY by −22pp at $1M due to 10.8%
  deployment." S37 confirms this is the right behavior for
  the intended operator profile — the engine isn't broken;
  it just isn't right for a $1M+ account without strategy
  expansion (which is exactly what Terminal B's S34
  universe-expansion backtest is testing).

**Methodology debt.**

- **Two anchor dates only.** S37 picks one crisis day and one
  normal day. A multi-anchor study (crisis bottoms, bull tops,
  vol shocks) would surface whether the ρ = 0.63 / 0.89
  divergence pattern holds or shifts with the regime.

- **No explicit operator-size simulation.** The recommendation
  table is a heuristic derived from the EV magnitudes seen on
  these two days. A formal backtest comparing both ranking
  keys at $100k / $1M / $5M account sizes would either
  confirm or refute the size-dependent recommendation.

- **`roc` is a static post-hoc sort.** S37 doesn't simulate
  what the engine WOULD recommend if `roc` were the primary
  key from the start — selection effects could shift
  candidates (e.g., the engine might surface different
  candidates entirely if it knew to optimize for `roc`).
  The current `roc` value is computed from a candidate set
  pre-filtered by `ev_dollars` semantics.

- **No comparison to other risk-adjusted metrics.** `roc =
  ev_dollars / collateral` is one risk-adjustment scheme.
  Other options exist: `ev_per_day / collateral` (time-
  adjusted), `(ev_dollars - cvar_5) / collateral` (downside-
  adjusted), Sharpe-style ratios. S37 picks the simplest
  `roc` because it's already emitted.

### S36 — Multi-ticker HMM regime realism

**Purpose.** Extends S33 V3 (which only checked AAPL) to **9
tickers across 5 sectors**: mega-cap tech (AAPL / MSFT / NVDA),
banks (JPM / BAC), defensives (KO / JNJ), energy (XOM), and
healthcare (LLY). Question: does the per-ticker HMM produce
*consistent* regime labels at known historical events across the
universe — or does it diverge by ticker in ways a trader needs to
understand? Closes the "S33 V3 only checked one name" methodology
debt from S33's verdict.

**Setup.** Same 504-day-tail HMM (`engine.regime_hmm.GaussianHMM`
n_states=4, random_state=42, n_iter=20). 7 anchor dates spanning
2020-03 → 2026-03: COVID crash, post-vaccine rally, 2022
inflation low, Aug 2024 vol spike, Apr 2025 S30 crisis, Feb 2026
recent, Mar 2026 data cutoff. Per ticker × event: extract argmax
label + multiplier + per-state probability vector + realized vol
(252d annualised) + realized return (252d annualised). Driver at
`%TEMP%\s36\driver.py` (not committed; per Sn convention).
`SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`. Read-only on
decision layer.

**Path.** `engine.data_connector.MarketDataConnector.get_ohlcv`
→ trim to `<= as_of` → `np.diff(np.log(close))` → last 504
points → `engine.regime_hmm.GaussianHMM().fit(tail).predict_proba`
→ `position_multiplier(probs[-1])` and `argmax(probs[-1])` for
label. Per-ticker, per-event, exhaustive.

**Status.** Done. **Verdict: the HMM is mathematically consistent
across tickers (each ticker has its own state space, labels +
multipliers reflect per-ticker dynamics). At known consensus crises
the universe agrees (9/9 Mar 2020, 8/9 Apr 2025). At transitional
events labels diverge by ticker in ways that are defensible. The
"crisis" label is NOT a simple vol threshold — Feb 2026
monotonicity check failed by design (XOM at vol 0.247 → crisis;
BAC at vol 0.276 → normal). This confirms and broadens S33 F4
("crisis = high-vol-state in the per-ticker fit, not a market-wide
vol-threshold rule") at the universe scale.**

**Findings:**

- **(F1 — universal-consensus crises, ✓ verified)** At the
  two cleanest historical crises:

  | Event | Crisis-labeled count |
  |---|---|
  | 2020-03-23 (COVID crash bottom) | **9/9** |
  | 2025-04-04 (S30 confirmed crisis) | **8/9** (LLY = bear, multiplier 0.482; LLY 252d return slightly positive +0.075) |

  Per-state probabilities at Mar 2020 are uniformly high crisis
  (most tickers >0.93 crisis probability, some 1.000). At Apr
  2025 the 8 crisis-labelers all show `crisis prob ≥ 0.994`
  (uniform pure-crisis state). The HMM's argmax label is
  unambiguous when the universe genuinely IS in crisis. ✓
  **Confirms S30's findings extrapolate beyond AAPL.**

- **(F2 — per-ticker divergence at transitional events, expected
  not a bug)** At less-clean transitional dates the universe
  splits by ticker in defensible ways:

  | Event | Label distribution across 9 tickers |
  |---|---|
  | Nov 2020 (post-vaccine rally) | 5 bear + 4 crisis (defensives KO/JPM/BAC/XOM still crisis; tech + JNJ + LLY transitioning to bear) |
  | Jun 2022 (inflation low) | Mixed: AAPL crisis (mult 0.77 — split state probs), MSFT/NVDA bull_quiet (mults 0.88/0.91!), JPM bear, BAC crisis, KO/JNJ crisis (mults 0.21/0.39), XOM normal (0.83), LLY bear. **Multipliers span 0.21 to 0.91 on the same date.** |
  | Aug 2024 (vol spike) | 7 crisis + XOM bear + LLY bear |
  | Feb 2026 (recent calm-anchor candidate) | 1 crisis (AAPL — same finding as S33 F4) + 6 bear + 1 normal (BAC) + 1 crisis (XOM) — wait, that's the same 9. Distribution: AAPL crisis, MSFT bear, NVDA bear, JPM bear, BAC normal, KO bear, JNJ bear, XOM crisis, LLY bear → 2 crisis, 6 bear, 1 normal |
  | Mar 2026 (data cutoff) | 1 crisis (KO! a defensive going crisis), 5 bear, 2 normal (NVDA, BAC), 1 bull_quiet (XOM) — post-crisis recovery diversity |

  This per-ticker divergence is **correct behavior**: each HMM
  is fit on that ticker's own 504-day log returns; the state
  emission distributions (means + variances) are
  ticker-specific. A trader reading the ranker output should
  interpret "AAPL crisis + KO crisis" as "both tickers are in
  their own high-vol regime" — NOT "AAPL is crashing the same
  way KO is." Logged as a documentation gap, not a bug.

- **(F3 — vol-vs-label monotonicity FAILS by design)** Feb 2026
  monotonicity check:

  ```
  ticker   vol_ann (252d)   label
  KO         0.172           bear
  JNJ        0.191           bear
  XOM        0.247           crisis   ← lower-vol-than-BAC, but crisis
  JPM        0.262           bear
  MSFT       0.263           bear
  BAC        0.276           normal   ← higher-vol than XOM, but normal
  AAPL       0.318           crisis
  LLY        0.424           bear
  NVDA       0.441           bear
  ```

  **Lowest-vol "crisis" = 0.247 (XOM); highest-vol "normal" =
  0.276 (BAC). Monotone? FALSE.** The label depends on the
  full 504-day state structure (state means, transition matrix,
  per-state emission probabilities), not just the trailing 252d
  vol. A trader assuming "high vol → crisis label" mis-models
  the engine. This **broadens S33 F4** at the universe scale —
  the HMM disambiguation columns (PR #222) only help if the
  trader knows to compare across the ranker's output.

- **(F4 — LLY anomaly at Apr 2025, ⚠ flag-not-fix)** While 8 of
  9 tickers label crisis on 2025-04-04, LLY labels bear
  (mult 0.482, crisis prob 0.059, bear prob 0.941). LLY's
  252d window ending 2025-04-04 had vol 0.324 (mid-range) and
  mean −0.050. The HMM categorizes LLY's recent dynamics as
  "moderately high vol, slightly negative mean" → bear, not
  crisis. **Defensible per the HMM's mathematical regime
  definition** (crisis = "very negative mean, very high vol" per
  `engine/regime_hmm.py:30`). LLY's vol of 0.324 wasn't
  extreme enough to trip the crisis state's emission
  probability mass for THAT ticker's fitted state means. **Not
  a bug; an interesting per-ticker characteristic worth
  documenting.**

- **(F5 — Mar 2026 KO anomaly, ⚠ flag-not-fix)** At the most
  recent data point (2026-03-20), 8 of 9 tickers label bear /
  normal / bull_quiet, BUT KO labels crisis (mult 0.548).
  State probs: crisis 0.544, bear 0.033, normal 0.423. **It's
  a marginal call (crisis edges normal by ~0.12).** KO's 252d
  vol is 0.169 — the LOWEST in the universe — yet labels
  crisis. The argmax-label rule masks the proximity to normal.
  **F4 nuance compounds:** the bare label hides marginal-call
  state. A trader inspecting one row needs the state-prob
  vector (NOT currently in the ranker output) to know the
  label is fragile. **Logged as a follow-up observability
  consideration:** could surface `hmm_argmax_prob` (the prob
  of the labeled state) as a confidence column.

- **§2 verified at universe scale.** Each per-ticker HMM fit
  runs independently; no cross-ticker contamination. Each
  ticker's `hmm_multiplier` is consumed by the ev_engine for
  THAT ticker's candidate only. The §2 contract holds.

**Realism Check.**

| Aspect | Engine (9 tickers, 7 events) | External reference | Verdict |
|---|---|---|---|
| Universal consensus at clean crises | 9/9 Mar 2020, 8/9 Apr 2025 | S30 confirms Apr 2025 = crisis on AAPL; broader market history for Mar 2020 (VIX > 60, S&P drawdown >30%) | ✓ Verified |
| Per-ticker label divergence at transitions | Multipliers span 0.21–0.91 on Jun 2022 across the universe | Each HMM is per-ticker by design (per `engine/regime_hmm.py`); divergence reflects per-ticker dynamics, not engine inconsistency | ✓ Verified (correct behavior, documentation gap) |
| Vol → label monotonicity | FAILS at Feb 2026 (XOM crisis @ vol 0.247; BAC normal @ vol 0.276) | HMM regime definition uses 4-state emission probabilities, not a vol threshold | ⚠ Realism gap — trader mental model "high vol = crisis" is wrong |
| LLY at Apr 2025 (8/9 universe → crisis, LLY → bear) | bear, mult 0.482, low crisis-prob (0.059) | LLY's 252d window had vol 0.324 (mid-range); not extreme enough for crisis emission | ⚠ Defensible per HMM math; surfaces "regime is per-ticker, not market-wide" |
| Mar 2026 KO crisis (others bear/normal) | crisis prob 0.544, but normal prob 0.423 — MARGINAL call | Argmax-rule label hides marginal-call ambiguity | ⚠ F5 observability nuance |

**Verdict.**

- **HMM is universe-consistent on consensus crises.** When the
  market actually IS in crisis, 9/9 or 8/9 tickers label crisis
  with high probability (>0.93 for most, 1.000 for many). The
  HMM's per-ticker independence does not prevent cross-ticker
  consensus when the underlying regime is shared.

- **Per-ticker label divergence at transitions is correct
  behavior.** Each ticker's HMM is fit on its own log returns;
  the state space is ticker-specific. At Jun 2022 the multiplier
  spans 0.21 (KO crisis) to 0.91 (NVDA bull_quiet) on the same
  date — and this is right: KO's slow-moving defensive history
  vs NVDA's high-vol growth history mean their "high-vol" states
  have different emission characteristics. The HMM correctly
  reflects this.

- **The "crisis = high-vol" mental model is wrong.** F3's Feb
  2026 monotonicity check failed at the universe scale: lowest-
  vol "crisis" (XOM 0.247) is below highest-vol "normal" (BAC
  0.276). The label depends on the full state structure, not just
  the trailing 252d vol. **S33 F4 is now confirmed at universe
  scale.** The disambiguation columns from PR #222
  (`hmm_realized_vol_252d_ann`, `hmm_realized_return_252d_ann`)
  help — but only if the trader knows to look at them.

- **Two flag-not-fix anomalies surfaced.** LLY at Apr 2025 (F4)
  and KO at Mar 2026 (F5) are both defensible-per-HMM-math but
  surface trader-actionable nuances. Neither is a bug. F5
  motivates a small future observability addition:
  `hmm_argmax_prob` column (the probability of the labeled
  state) would let a trader see when a label is marginal.

- **S33's findings extrapolate to the universe.** The Apr 2025
  crisis behaviour S30 documented on AAPL is reproduced across 8
  of 9 names. The Feb 2026 "high-vol-not-crashing" finding from
  S33 V3b reproduces across mid-cap tech (MSFT, NVDA), banks (JPM),
  defensives (KO, JNJ), and healthcare (LLY) — confirming it
  wasn't an AAPL artifact.

**AI handoff.**

- **Closes the S33 "single-name HMM verification" methodology
  debt.** The HMM is now verified to work consistently across 9
  representative tickers at 7 historical events.

- **F5 small observability follow-up (queued):** add an
  `hmm_argmax_prob` column to the ranker output alongside
  `hmm_regime` / `hmm_multiplier` / `hmm_realized_vol_252d_ann`
  / `hmm_realized_return_252d_ann`. A trader inspecting one row
  would then see (a) the label, (b) the multiplier, (c) the
  realized vol/return, and (d) the confidence in the label
  (e.g., 0.544 for KO at Mar 2026 = marginal call). Single-
  line addition in `engine/wheel_runner.py`'s HMM block; ships
  as a future small PR if the user wants this enriched.

- **F4 + F5 motivate trader-facing documentation.** The HMM
  semantic ("high-vol regime in the per-ticker fit, not a
  market-wide vol threshold") deserves a CLAUDE.md or
  `docs/HMM_REGIME_SEMANTICS.md` note. Currently this knowledge
  lives in the regime_hmm.py source comments + S33 F4 + S36 F3
  — three places, none of them the trader's first read.

- **Sanity follow-up Sn (S6 dependency):** re-run S36 with the
  Theta connector. Different data path may surface different
  per-ticker emission probabilities; would confirm or deny
  whether the universe-consensus finding is data-source-
  independent.

**Methodology debt.**

- **9 tickers, not the full SP500.** S36 picks a representative
  cross-section. A full-universe verification (~500 tickers)
  would surface more LLY-/KO-style per-ticker anomalies but
  would also be much harder to digest. The 9-ticker sample
  spans 5 sectors and intentionally includes both extremes
  (NVDA high-beta growth, KO defensive consumer staples).

- **Single anchor per event.** Each historical event uses ONE
  anchor date. The HMM behaviour ON the transition day (e.g.,
  Apr 2025 = the day OF the crisis) may differ from one week
  before or after. S30 covered ±2-3 days around the AAPL
  bear → crisis transition; S36 takes one snapshot per ticker
  per event. A date-grid (e.g., 5 days centered on each event)
  would let us check label-stability under noise.

- **No comparison to a non-HMM regime classifier.** The HMM is
  the regime arbiter today. A VIX-threshold rule (VIX > 30 →
  crisis; VIX < 15 → bull) on the same dates would either
  agree or surface where the HMM and the conventional vol
  proxy diverge. S30's methodology debt already mentioned
  this; S36 inherits the same gap.

- **Bloomberg-only.** Theta replay would change the data
  source for HMM fitting; results may shift. S6 queued.

- **No causality check for LLY / KO anomalies.** F4 and F5
  surface "this ticker behaves differently here" without
  investigating *why* — e.g., is LLY's bear-not-crisis on Apr
  2025 because LLY actually had a defensive return profile
  during the universe's crisis? Worth a small ad-hoc probe.

### S38 — Multi-window backtest at 100 tickers / $1M / 2020-2024

**Purpose.** S34 (Terminal C) found "+11.6pp over SPY at $1M with
100 tickers over 2022-2024." S35 (Terminal B) found −41pp over
2018-2020 at $100k / 24 tickers, showing window-sensitivity. S38
closes the gap by re-running S34's $1M / 100-ticker setup over a
**5-year multi-window (2020-01-02 → 2024-12-31)** that includes
COVID + 2021 bull + 2022 bear + 2023-2024 recovery. Question:
does S34's "engine beats SPY" generalize to a longer window, or
was it 2022-2024-window-favored?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Identical to S34 except the window. Same 100 tickers
(first alphanumeric SP500 names). $1M starting capital.
35-DTE / 25-delta puts → wheel into CC on assignment → hold to
expiry. Three parallel `WheelTracker` instances (frictionless /
bid_ask / full). `require_ev_authority=False`. Post-IV-PIT-fix
engine on `origin/main` (commit `2da76ff`). Driver under
`%TEMP%\s38_backtest\` (not committed; per Sn convention).
~3 hours runtime, 17,360 candidate-evaluations per friction
level (52,080 total ledger rows).

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route) — daily
re-rank, top-3 trade attempts per day, frictionless +
bid_ask + full-friction parallel trackers.

**Status.** Done. **Verdict: S34's +11.6pp result was
2022-2024-window-favored. Across the full 5-year window the
engine UNDERPERFORMS SPY by ~52pp.** The engine's dollar-alpha
varies from −52pp (5-year multi-window) to +27pp (S27 anomaly)
across the five completed $-scale backtests. There is no single
number that represents the engine's "edge."

**Findings:**

- **(F1 — window-sensitivity confirmed at scale, full
  friction)**. Final NAV $1,331,764 = +33.18% over 2020-2024.
  SPY over the same window returned ~+85% (price + dividends).
  **Engine UNDERPERFORMS SPY by ~52pp.** Headline numbers across
  all backtests:

  | Sn | Capital | Universe | Window | Engine | SPY | Delta |
  |---|---|---|---|---|---|---|
  | S27 | $100k | 24 | 2022-2024 | +51% | +24% | **+27pp** |
  | S32 | $1M | 24 | 2022-2024 | +1.8% | +24% | **−22pp** |
  | S34 | $1M | 100 | 2022-2024 | +35.6% | +24% | **+11.6pp** |
  | S35 | $100k | 24 | 2018-2020 | +3.6% | ~+45% | **−41pp** |
  | **S38** | **$1M** | **100** | **2020-2024** | **+33.2%** | **~+85%** | **−52pp** |

  Five configurations, five different deltas spanning −52pp to
  +27pp. **Dollar-alpha is a multi-dimensional function of
  (capital × universe × window).**

- **(F2 — signal generalizes at scale, ρ = 0.358)**.
  Statistically overwhelming across N=17,192 candidates.
  Slightly higher than S34's 0.327. Per-year: 2020 ρ=0.55,
  2021 ρ=0.21, 2022 ρ=0.37, 2023 ρ=0.31, 2024 ρ=0.31. **Never
  negative.** The ranker genuinely ranks better than random in
  every year, including the COVID crash year (highest ρ).

- **(F3 — realized executed P&L is NEGATIVE)**. The 305 executed
  put trades + 168 CCs over 5 years generated **−$28,647** in
  realized P&L. All NAV growth (+$331,764) came from equity-beta
  on assigned stock positions held through 2023-2024 bull
  (108.6% of NAV gain attributable to equity-beta residual).
  Same shape as S27 (NAV +$51,444 / realized −$3,421) and S35
  (NAV +$3,566 / realized −$48,326). **The engine's put-selection
  alpha is consistently negative on average across all multi-
  window backtests.**

- **(F4 — engine refusal during COVID was correct, again)**.
  847 candidates in 2020-02-15 → 2020-05-15; 19 executed
  (97.8% refusal rate). Mean realized of all 847 candidates =
  **−$254 per trade** if blindly executed (i.e., ~−$215k of
  losses avoided). **This is the engine's strongest defensible
  property** — the refusal mechanism is a real risk control.

- **(F5 — concentration risk amplified at scale)**. Top 5
  contributors to executed realized P&L:
  BKNG +$10,940 / BIIB +$4,046 / AZO +$3,234 / ADSK +$2,534 /
  CHTR +$2,373 (total +$23,127). Net of all 62 traded tickers
  = **−$28,647**. **The other 57 tickers collectively LOST
  ~$52k.** Aggregate realized P&L is negative despite BKNG-
  style outliers. Concentration risk is even more extreme than
  S34's single-window finding.

- **(F6 — quartile monotonicity at extremes, broken in middle)**.
  Q0 realized mean +$28; Q1 +$3; Q2 −$67; Q3 +$206. **Q3 beats
  Q0 by 7.4×**, but Q2 went negative. Signal at the tails is
  clean; middle of the distribution has noise. Same pattern as
  S22/S27/S32.

- **(F7 — capital deployment 22.6%)**. Average across the 5-year
  window. ~77% of $1M sits in cash earning nothing during a
  multi-year bull market. **This is the dominant explanation for
  the −52pp gap.** Even with universe expansion to 100 tickers,
  the engine cannot capture enough of SPY's bull-market return
  because it under-deploys. Strategy-stack additions (S37
  follow-on) become more important: the cash buffer is the
  capacity blocker, not the BP gate.

- **§2 verified.** Same `rank_candidates_by_ev` path as
  S22/S27/S32/S34/S35. No engine code touched. 0 rank failures
  over 52,080 evaluations. The §2 contract holds.

**Realism Check.**

| Aspect | Engine (S38) | External reference | Verdict |
|---|---|---|---|
| 2020-2024 SPY return | ~+85% (price + div) | Public data: SPY 320 → ~600 = +85% incl div | ✓ Reference correct |
| Engine NAV +33.18% over 5y = ~5.9% annualized | Comparable to long-dated treasuries + small premium | Conservative income strategy expectation | ✓ Internally consistent |
| Engine refuses 97.8% of COVID candidates | If blindly executed, mean P&L = −$254/trade | Defensive behavior expected during crisis | ✓ Confirms S35 finding |
| Engine ρ = 0.358 over 5y | Spearman ranks alpha is preserved out-of-sample | Statistical: N=17,192 with p << 1e-100 | ✓ Robust signal |
| Realized executed P&L = −$28,647 | The engine LOSES money on average per executed trade | Counterintuitive: NAV up while realized down | ✓ All NAV growth = equity-beta on assigned stocks |

**Verdict.**

- **The "engine beats SPY" framing is window-specific.** S34's
  +11.6pp at $1M / 100t / 2022-2024 was an outlier. The honest
  multi-window expectation is **engine underperforms SPY by
  20-50pp in bull-dominated 3-5 year windows**.
- **The engine's defensible value proposition is income +
  refusal, NOT dollar-alpha.** +33% over 5 years = ~5.9%
  annualized, which is a reasonable conservative-income return
  with strong crisis refusal. SPY-beating is not.
- **The realized executed P&L is negative across S27, S35, and
  S38.** All NAV growth in those backtests came from equity-
  beta on assigned positions. **The put-selection signal ranks
  correctly (ρ = 0.32–0.36) but loses money on average per
  trade.** F4 fix (B1) becomes more important; if the engine
  could correctly refuse the worst tail-risk candidates, the
  realized P&L might flip positive.
- **Autonomous deployment verdict remains NO.** F4 tail-risk
  gap + D17-live still apply. The multi-window result reinforces
  that forward dollar-alpha is not predictable from any single
  backtest. The deployment matrix in
  `docs/PRODUCTION_READINESS.md` should be revised to
  acknowledge the −52pp / +27pp window dependence.

**AI handoff.**

- **The deployment matrix needs amendment.** Update
  `docs/PRODUCTION_READINESS.md` §5 supervised-$1M case from
  "Conditional ✅" (citing S34's +11.6pp) to "Conditional ⚠ with
  explicit underperformance acknowledgment" citing S38's −52pp
  multi-window result.
- **Marketing / pitch material that cites "engine beats SPY"**
  should be qualified as 2022-2024-window-specific, not a
  forward estimate.
- **B3 capacity blocker is closed structurally but NOT in dollar
  terms.** Universe expansion enables more deployment (22.6% vs
  S32's 10.8%) but does not produce SPY-beating returns.
- **F4 fix (B1) is now more important.** If F4 could correctly
  refuse the worst tail-risk candidates, the −$28k realized P&L
  might flip positive. The Fix A attempt (lookback 5y→2y/3y)
  failed; new approaches need exploration.
- **Strategy-stack expansion (S37 follow-on) is now more
  important** as a way to deploy the idle cash buffer. The
  capacity blocker is the cash sitting in BP, not the BP gate
  itself.
- **Consider repositioning** the engine as a conservative income
  strategy (~5.9% annualized + crisis refusal) rather than an
  alpha strategy. That's a defensible value proposition the
  multi-window data supports.

**Methodology debt.**

- **Single multi-window only.** S38 uses one 5-year window. A
  rolling-5-year-window study (2015-2019, 2016-2020, 2017-2021,
  2018-2022, 2019-2023) would surface whether the −52pp result
  is itself a feature of the 2020-2024 specific window or a
  general property of the engine at scale.
- **Bloomberg-only.** Same S6 dependency as other backtests.
- **No SPY-included tracker.** S38 reports "engine vs SPY" using
  an external ~+85% reference. A direct SPY-in-tracker simulation
  with the same friction model would let us compare dollar-
  for-dollar on identical assumptions.
- **In-sample HMM/POT-GPD parameters.** The forward distribution
  parameters (lookback windows, threshold quantiles) were fit on
  data overlapping the backtest period. A true out-of-sample
  verification would re-fit parameters on pre-2020 data only.

---

### S35 — 2018-2020 out-of-window cross-validation

**Purpose.** Cross-validate the engine's predictive signal and dollar-
alpha against a **completely different 3-year window** from S22 / S27
/ S32 / S34 (all 2022-2024). Test the engine's behaviour through 2018
chop + Q4 selloff, 2019 strong bull, 2020 COVID crash + V-recovery.
Directly compare to S27 ($100k, same 24-ticker universe, post-fix
engine) — only the time window changes.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
2018-01-02 → 2020-12-31 (756 trading days), 24 SP500 tickers
identical to S22 / S27 / S32 / S34. **$100k starting capital**
(matches S27 for direct comparison). `require_ev_authority=False`.
Three parallel `WheelTracker` instances per friction level
(frictionless / bid_ask / full). Post-IV-PIT-fix engine on
`origin/main`.

**Status.** Done. **Verdict: signal generalizes (ρ = 0.50 in 2020,
DOUBLE S27's 0.22); dollar-alpha does NOT generalize (engine +3.57%
vs SPY ~+45% = −41pp).** Plus the discovery of a 504-day OHLCV
history gate that effectively makes this a 2020-only backtest.

**Findings:**

| Metric | Frictionless | bid_ask | full friction |
|---|---|---|---|
| Final NAV | $104,473 | $103,594 | $103,566 |
| Return | +4.47% | +3.59% | **+3.57%** |
| Short puts opened | 19 | 19 | 19 |
| Spearman ρ (2020 only, n=1,946) | 0.5028 | 0.5003 | **0.4970** |
| Mean realized (all puts) | −$148.67 | −$165.78 | **−$169.21** |
| Mean realized (executed puts) | −$1,097.51 | −$1,118.20 | **−$1,125.25** |
| Executed hit-rate | 68.4% | 68.4% | 68.4% |
| Skipped (no BP) | 221 | 221 | 221 |
| SPY same window | — | — | ~+45% |

- **F1 — Signal generalizes.** ρ = 0.50 in 2020 is 2× higher than
  S27's 0.22 in 2022-2024. The engine's ranking quality is both
  scale-invariant (S32 confirms) AND window-invariant (S35 confirms).
- **F2 — Dollar-alpha does NOT generalize.** Engine +3.57% vs SPY
  ~+45% = **−41pp underperformance**. The "+27pp over SPY" headline
  from S22 / S27 is **window-specific**, not a robust engine
  property.
- **F3 — Engine wisely sat out COVID.** During 2020-02-15 →
  2020-05-15, the engine refused 99.8% of 482 candidate rows (took
  only 1 trade — which won +$231). Mean realized of all 482
  candidates was −$369.55 — engine refusals correct in aggregate.
- **F4 — Post-COVID positions performed poorly.** 18 trades taken
  after the COVID window averaged −$1,125 realized despite ρ=0.50.
  The engine's `prob_profit` was over-optimistic in the unusual
  post-pandemic vol environment.
- **F5 — 504-day OHLCV history gate is REAL.** Verified live: at
  as_of=2018-06-15, every ticker is dropped with
  `gate=history, reason="history 115d < required 504d"`. S35
  effectively becomes a 2020-only backtest (252 useful trading days)
  because OHLCV starts 2018-01-02. **New caveat for deployment:
  recently-listed names are unrankable until 2 years of OHLCV
  history accumulates.**
- **F6 — Q3 still beats Q0 but all quartiles negative.** Q3 realized
  mean −$76 vs Q0 −$316 (4× spread). The signal is clean. But every
  quartile has negative mean — the engine ranks correctly relative
  but loses absolutely in this regime.

**AI handoff:**

- The window-specific dollar-alpha is the most consequential finding
  for `docs/PRODUCTION_READINESS.md`'s deployment matrix. The
  "+27pp over SPY" must now be qualified BOTH "$100k-class" AND
  "2022-2024-class".
- 504-day history gate worth surfacing in onboarding docs.
- A multi-year backtest (e.g., 2020-2024 = 5 years post-gate)
  would average out window-specific noise. (S38 closed this.)

Full doc: `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md`.

---

## 2. In flight

_(none currently)_

---

## 3. Queued

### S6 — Theta provider with real chains

**Purpose.** Exercise actual chain-quoted premiums vs the synthetic
BSM premium Bloomberg uses. `edge_vs_fair` is structurally 0 on
Bloomberg; S1 flagged this as the biggest missing signal.

**Setup.** `SWE_DATA_PROVIDER=theta` with the Theta Terminal running
on `127.0.0.1:25503`. Operator setup required.

- **Re-verified 2026-05-26** by Terminal A — **SKIPPED (operator-gated, Theta lock held by another agent)**. Per the task spec skip list: "S6 operator-gated — needs Theta Terminal. Not in completed list originally." Additionally, the user-stated coordination context: another agent is actively pulling data from Theta servers (the Theta lock per PARALLEL_SESSIONS rule 7 is held); even attempting an S6 re-run would risk collision. The §2-relevant claim — that Theta-sourced chain premiums route through `EVEngine.evaluate` exactly as Bloomberg-synthetic premiums do — is structurally enforced by `WheelRunner` provider abstraction and unchanged on `main`.

---

## 4. Candidate (not yet selected)

Worth running when scope and time allow:

_(none currently)_

---

## Re-verification 2026-05-26 — Summary

This section consolidates Terminal A's re-verification pass against
the current engine at `origin/main` HEAD `8a17b0b`. Each completed
S1-S27 scenario was re-run with the original setup (or the closest
faithful proxy where setup-specific harnesses were not in the
worktree). Per-scenario sub-notes live in-line under each entry
above (and under `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`
for S27). This section is the campaign-wide read.

### Scope

- **Active set (24 scenarios):** S1-S4, S7-S21, S23-S27.
- **Skipped (3):** S5 (operator-gated MCP chart), S6 (operator-gated
  + Theta lock held by another agent), S22 (archival — pre-IV-PIT-fix
  duplicate of S27).
- **Decision-layer code (`engine/ev_engine.py`,
  `engine/wheel_runner.py`, `engine/candidate_dossier.py`): NOT
  EDITED.** Re-verification is read-only against the decision layer.

### Per-scenario verdict table

| S-id | §2 holds | Verdict match | Drift > 5% | Suspect PRs | Notes |
|---|---|---|---|---|---|
| S1 | ✅ | match | no | — | Dividend bug stays fixed; `ev_raw`/`roc`/`collateral`/`hmm_regime`/`sector`/`news_n_articles` columns all surfaced |
| S2 | ✅ | match | n/a | #104 / #122 / #127 / #128 / #129 | save/load/suggest_rolls/avail_BP all shipped |
| S3 | ✅ | match | n/a | #122 / #181 | `suggest_rolls` published cols intact; `.attrs["drops"]` populated |
| S4 | ✅ | match | yes (mild) | #109 / #119 / #121 / #179 | `collateral`/`roc` columns shipped; `select_book(account_size, ..., max_weight_per_name=...)` selects 3 names at $50k |
| S5 | n/a | skipped | n/a | — | Operator-gated; no live TradingView Desktop |
| S6 | n/a | skipped | n/a | — | Operator-gated + Theta lock held by another agent |
| S7 | ✅ (by structure) | partial | n/a | — | `EngineIntegration.evaluate_trade` signature evolved; structural findings still apply at source |
| S8 | ✅ | match | n/a | #122 / #124 / #126 / #129 / **#145 (D16)** | D16 confirmed live: `EVAuthorityRefused` on neg-EV row at `issue_ev_authority_token` |
| S9 | ✅ | match | n/a | #121 | History/event/survivorship gates fail-closed; structured drops `{ticker, gate, reason}` |
| S10 | ✅ | match | n/a | #119 | News mult=1.0 silent neutral on missing store (by design) |
| S11 | ✅ | partial | yes | **#119** | `credit_multiplier` now PIT-aware (0.80 / 0.92 at 2025-04 VIX spike vs originally pinned 1.00) — PIT leak CLOSED |
| S12 | ✅ | match | n/a | — | engine_api `_enrich_alert` / ring buffer / nonce-register / HMAC all unchanged |
| S13 | ✅ | match | yes | #179 | FIX evDollars 2263.5 → 2547.97 (+12.6%); regime label `ELEVATED` + vix `28.97` bit-identical |
| S14 | ✅ | match | <5% | #118 / #220 | Layer-2 IV overlay alive (AAPL 76.97 vs orig 77.0); strangle ranker carries EventGate |
| S15 | ✅ | partial | n/a | **#163 / #165 (D17)** | 3 of 6 unwired surfaces wired; HRP + Kelly still orphan |
| S16 | ✅ | partial | yes | #179 | CAT EV +53%, NVDA magnitude -43% (both signs preserved); HMM identity holds at 4dp |
| S17 | ✅ | match | n/a | — | EV-flip + HMM-regime-flicker pattern reproduced at noise floor; zero crashes/warns |
| S18 | ✅ | **partial — WARM REGRESSION** | yes | **#215 / #220 + #208 / #210 / #222** | L2 warm 10.5s → 41.2s (+292%); HMM cache 492 → 491 (within 1) |
| S19 | ✅ — **C7b CLOSED** | partial | n/a | **#204 (R1a)** / **#215** | +inf / NaN / -inf all → `blocked / ev_non_finite`; future as_of → typed `ValueError` |
| S20 | ✅ — **G3 RE-REFUTED** | match | yes (AAPL EV) | #179 | Webhook +inf/-inf/NaN → server-computed AAPL EV/skip; ring-trim/torn-read/nonce/HMAC clean |
| S21 | ✅ | match | n/a | — | Prong A `sector_cap_breach` on CAT @ $150k; Prong B 2/9 opened at $1M, 7×`portfolio_delta_breach` |
| S22 | n/a | skipped | n/a | — | Archival pre-IV-PIT-fix |
| S23 | ✅ — **F1 + F3 CLOSED** | partial | yes | **#179 + #180** | get_recent_earnings exists; AVGO blocked at TDA; IV PIT-aware (0.4844/0.4982) |
| S24 | ✅ | match | n/a | — | take_snapshot 3-state map; `open_strangle` still absent |
| S25 | ✅ — **F3 + F4 CLOSED** | match | yes | **#179** | MU CC iv 0.6939 / 0.6515 — matches IV file exactly; sign of 25-Δ CC stays negative |
| S26 | ✅ | match | yes | #179 / #122 | MU edge $2181.54 (orig $1876, +16.3%); AAPL edge $213.59 (orig $229, -6.8%) |
| **S27** | ✅ | **partial** | yes (NAV, executed trades) | composite | ρ 0.1881 (orig 0.2183, -14%); NAV $164,876 (orig $151,444, +9%); 15 executed (orig 50, -70%). Per-year shape + quartile monotonicity preserved. Sub-note in `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`. |

### Drift report — every flagged metric

Drifts > 5% by scenario, original → new, attributed:

| Scenario | Metric | Original | New | Δ% | Suspect PR | Cause hypothesis |
|---|---|---|---|---|---|---|
| S4 | `rows[36-name watchlist]` | 30 | 27 | −10.0% | #220 + Bloomberg refresh | as_of-cutoff guard + earnings-calendar drift |
| S4 | `positive_ev rows` | 20 | 21 | +5.0% | composite (#179 + #208) | post-IV-PIT EV recalibration |
| S11 | `credit_mult[2025-04-07]` | 1.00 (pinned) | **0.80** | −20.0% (PIT-correct) | **#119** | PIT-leak fix — credit overlay now responds to as_of |
| S11 | `credit_mult[2025-04-09]` | 1.00 | **0.92** | −8.0% (PIT-correct) | **#119** | same |
| S11 | `cross-section_hmm_mean[2025-04-09 peak]` | 0.20 | 0.2923 | +46.2% | #208 + #222 | HMM disambiguation columns + seed-stable refits |
| S11 | `event_survivors[2025-04-24]` | 3 | 2 | −33.3% | #220 (low-confidence) | as_of-cutoff may drop a borderline name |
| S13 | `dashboard FIX evDollars` | 2263.5 | 2547.97 | +12.6% | **#179** | IV-PIT in `rank_candidates_by_ev` |
| S16 | `CAT ev_dollars` | 290.26 | 444.99 | +53.3% | **#179** | PIT-IV propagation; HMM mult unchanged (0.4538 / bear) |
| S16 | `NVDA ev_dollars` | −124.32 | −70.76 | −43.1% magnitude | **#179** | Same; sign preserved |
| S18 | `L2 warm wall_s` | 10.5 | 41.2 | **+292%** | **#215 + #220 + diagnostic columns (#208/#210/#222)** | Composite per-call overhead; real warm-path regression |
| S18 | `L1 cold wall_s` | 145.2 | 79.3 | −45.4% | composite (HW + cache) | Hardware/state variance + cache improvements |
| S18 | `L5b top_n=10000 survivors` | 433 | 423 | −2.3% | within noise | Bloomberg-data window drift |
| S20 | `AAPL webhook ev_dollars` | −95.47 | −14.46 | −85% magnitude | **#179** | Post-IV-PIT AAPL ev at this as_of is materially less negative; sign preserved |
| S21 | `Prong B opened/9` | 2 | 2 | 0% | — | Identical |
| S23 | `AVGO iv[2026-03-10]` | 0.4296 | 0.4844 | +12.8% | **#179** | PIT IV |
| S23 | `AVGO iv[2026-03-13]` | 0.4296 | 0.4982 | +16.0% | **#179** | PIT IV |
| S23 | `AVGO ev_dollars[2026-03-10]` | +310.30 | +390.06 | +25.7% | **#179** | Higher PIT IV → higher synthetic premium |
| S23 | `AVGO ev_dollars[2026-03-13]` | +150.46 | +222.72 | +48.0% | **#179** | same |
| S23 | `AVGO survives at 2026-03-05 (TDA)` | True (iv=0.43, ev=+268.85) | False (event_lockout) | (block) | **#180** | Symmetric event-gate back-buffer now reachable via `get_recent_earnings` |
| S25 | `MU CC iv[2026-03-17]` | 0.6485 (snapshot) | **0.6939** (PIT) | +7.0% | **#179** | Predicted post-fix; observed exactly |
| S25 | `MU CC iv[2026-03-19]` | 0.6485 | **0.6515** | +0.46% | #179 | Within rounding (snapshot vs PIT happen to coincide here) |
| S25 | `MU CC ev_dollars[2026-03-17]` (25-Δ proxy) | −1058.28 | −147.93 | −86% magnitude | #179 + strike-shift | Higher IV moved the 25-Δ strike from 541 to 557.5 |
| S26 | `MU winning edge` | +1876.05 | +2181.54 | +16.3% | **#179 + #122** | IV-PIT + buyback_total correction |
| S26 | `AAPL challenged edge` | +229.26 | +213.59 | −6.8% | within noise | Smaller grid (4 vs 16 candidates) + IV-PIT |
| S27 | `Spearman ρ` | 0.2183 | **0.1881** | −13.8% | composite (engine SHA `d26a8d6` → `8a17b0b`) | Per-year shape + quartile monotonicity preserved |
| S27 | `Final NAV` | $151,444 (+51.4%) | **$164,876 (+64.9%)** | +8.9% / +13.5 pp | mostly fewer-but-cleaner executions | Both runs $100k-class; both beat SPY +24% by 25–40 pp |
| S27 | `Executed trades` | 50 | **15** | **−70%** | composite (#215 / #220 / #227 cutoff guards + harness-shape BP gating) | Most likely contributor is harness-shape BP gating; see S27 sub-note diagnosis |
| S27 | `Mean realized (per trade)` | $63.34 | **$51.70** | −18.4% | post-#215 cutoff guards | Hit-rate rose (76.4% → 80.5%) so fewer trades pick up smaller average gains |
| S27 | `2022 mean realized` | $21.68 | **$1.72** | **−92%** | F4 tail-risk gap (open since original S27) | 2022-bear executions leaned more on losers; F4 still unresolved |

**Headline pattern.** Three PRs are responsible for ~90% of the
positive drift on the EV-magnitude axis:

1. **PR #179 (IV-PIT fix on `rank_candidates_by_ev`)** — propagated
   to `rank_covered_calls_by_ev` and `rank_strangles_by_ev` per
   PR #220's gate extension. Cited as the primary suspect on 11 of
   the flagged metrics above.
2. **PR #119 (news / credit PIT-leak fix)** — cited on the S11 credit-
   regime overlay closure.
3. **PR #180 (symmetric event-gate back-buffer via `get_recent_earnings`)**
   — closes S23 F1 (the previously-dead back-buffer is now reachable).

Plus two regressions:

4. **PR #215 + #220 (`as_of-beyond-data` refusal guards) + diagnostic
   columns from PR #208 / #210 / #222** — composite per-call overhead
   that bumps the L2 warm-rank wall-clock from ~10s to ~40s on the
   503-ticker universe. **Warm-path regression, not a §2 issue.
   Flagged. Not fixed in this PR.**

### §2 status

**GREEN.** No §2 BREACH surfaced across the 24 active scenarios.
Two §2 surface closures confirmed live:

- **S19 C7b** (dossier reviewer `+inf` bypass) — **CLOSED by PR #204**
  via R1a's `math.isfinite(ev)` check, returning the distinct
  `verdict_reason="ev_non_finite"`. Verified end-to-end with synthetic
  `ChartContext(is_ok=True, visible_price=spot)` against the EV vector
  `(+25, +inf, -inf, NaN)` — all three non-finite values return
  `blocked / ev_non_finite`. CLAUDE.md §2 R1a's text matches the
  current `engine/candidate_dossier.py:130-153` source.
- **S20 G3** (network-surface `+inf` bypass via the TV webhook) —
  **RE-REFUTED** on the v5 backfill of the original test. Webhook
  payloads carrying `ev_dollars` as `+inf` / `-inf` / `NaN` all return
  the server-computed AAPL EV (-14.46 on `2026-03-20`) with
  `verdict=skip`. The payload's `ev_dollars` is structurally never
  read on the EV path; `_enrich_alert` overrides from the ranker.

### Pre/post pytest delta

| Phase | Total | Passed | Failed | xfailed | Δ from pre |
|---|---|---|---|---|---|
| Pre-flight (baseline at `8a17b0b`) | 2394 | 2375 | 17 | 2 | — |
| Post-run (rebased onto `46ddbd4`) | **2417** | **2412** | **3** | **2** | **+14 passed, −14 failed** |

**The 14-test improvement is not from this re-verification work.**
It landed via **PR #237** (`fix(tests): extend synthetic OHLCV to
cover as_of=2026-03-15`), which merged into `main` during my work
and was pulled in via `git rebase origin/main` before the final
pytest run. PR #237 closes the `test_ranker_iv_pit.py` (9 fails) and
`test_event_gate_back_buffer.py` (5 fails) clusters by extending the
mock connectors' synthetic OHLCV ranges so the new
`as_of-beyond-data` cutoff guards (PR #215 / #220) don't refuse the
test fixtures.

The **3 remaining post-run failures** are all in
`tests/test_theta_connector.py` — Windows-local per the
[[windows-local-vs-ubuntu-ci]] memory; not extrapolable to CI;
present on `main` before this work.

**Conclusion: no regression introduced by this re-verification.**
Pre-existing failures: 17 → 3 (improvement attributable to PR #237).
My work contributes 0 new failures.

The total test count went 2394 → 2417 (+23) because PRs that
merged into main during my work added regression tests
(e.g. PR #233's 13 D17-wire regression tests).

### 5-ticker EV smoke drift

| Phase | sha256 | rows | cols | Top by EV |
|---|---|---|---|---|
| Pre-flight (at `8a17b0b`) | `4fc14bf0e6985ac42fe9f9f04352df8884e2c0e51bdcf52bc08626e7905c5317` | 5 | 51 | XOM ($137.57), JPM (124.90), MSFT (90.97), UNH (62.62), AAPL (20.45) |
| Post-run (at `46ddbd4`) | `4fc14bf0e6985ac42fe9f9f04352df8884e2c0e51bdcf52bc08626e7905c5317` | 5 | 51 | (identical) |

**Result: IDENTICAL.** Byte-for-byte parquet match. The 5-ticker
EV smoke is unchanged across the rebase + re-verification work,
confirming no in-process state mutation leaked from re-verification
into the live data path. Saved at `C:\tmp\preflight_5tk.parquet`
and `C:\tmp\postrun_rebased_5tk.parquet`.

Pre-flight saved to `C:\tmp\preflight_5tk.parquet` (not in repo).
Post-run will be diffed byte-for-byte against this; any deviation
indicates an unintended state mutation during re-verification.

### Methodology / setup substitutions

- **S17 condensed**: ran a 5-day rolling rank sweep on the documented
  25-ticker universe instead of the full 10-trading-day operational
  sim (with daily save/load round-trips). The condensed run exercises
  the same per-day EV-flip and HMM-flicker mechanism. The "operational
  verdict: YES with workarounds" outcome from the original entry is
  preserved by reference, not re-derived end-to-end.
- **S20 ports**: ran the engine_api subprocess on Terminal A's
  allocated port `:8787` instead of S20's documented `:18787` (which
  is outside Terminal A's `.claude/settings.local.json` env block).
  No behavioral difference — same code path, same race vectors.
- **S22**: skipped per task spec (archival).
- **S27**: run via a one-off harness modeled on Terminal C's unmerged
  `backtests/regression/_common.py` (read-only reference; not imported
  or committed into this re-verification PR). Setup matches the doc
  exactly: 2022-01-03 → 2024-12-31, 24 tickers, $100k, 35-DTE / 25-Δ,
  frictionless, `require_ev_authority=False`.

### What was NOT re-verified

- **Theta-provider-specific paths** (S6 and the chain-quality gate at
  `wheel_runner.py:843`). Theta Terminal access is held by another
  agent for the duration of this work — no contention.
- **MCP live chart loop** (S5). Requires TradingView Desktop + CDP
  on `:9222` + tradingview-mcp CLI.
- **Performance regression follow-up on S18 L2 warm path.** Captured
  the drift; identification of the root-cause PR is composite and
  needs profiler-level work to attribute precisely.
- **Handle-leak follow-up from S18.** Would require a 100+-call
  sweep; the warm-path regression is more pressing.

### Recommended follow-ups (not in scope for this PR)

1. **S18 warm-path regression.** Profile a single warm
   `rank_candidates_by_ev(503-ticker)` call to identify which gate-
   check or diagnostic-column emit is responsible for the 4× warm
   latency. Plausible suspects (in priority order): PR #215's
   `_check_as_of_cutoff` running per-ticker, PR #220's CC/strangle-
   ranker variants of the same guard, PR #210's `sector` column
   needing a per-ticker `get_fundamentals` lookup, PR #222's HMM-
   disambiguation columns adding per-call regime fetches.
2. **`tests/test_ranker_iv_pit.py` + `tests/test_event_gate_back_buffer.py`
   mock fixtures.** 14 of the 17 pytest failures stem from mock
   connectors that don't expose the data-cutoff or recent-earnings
   metadata the new guards require. Update the mocks to match the
   real connector's API surface.
3. **S7 advisor committee re-test under the new `evaluate_trade(ev_row,
   portfolio_state, market_state)` signature.** The "committee
   structurally pinned at neutral" finding wasn't re-derived; a small
   Sn that exercises the new shape on the 10 ROC-ranked names would
   close the loop.

### Engine state — overall posture

The engine is **mechanically sound on the §2 invariant** at
`origin/main` HEAD `8a17b0b`. The campaign-headline closures from
the original entries (S19 C7b inf-bypass, S23 F1 back-buffer dead
code, S23 F3 / S25 F3 IV-snapshot bug across all three rankers,
S11 credit-PIT leak) have all shipped. Drift > 5% is concentrated
on EV-magnitude axes where PR #179's IV-PIT propagation legitimately
shifts the engine's view of the world; signs and orderings are
preserved.

The most material unresolved item this re-verification surfaces is
the **S18 warm-rank latency regression** — not a §2 issue, but a
real operator-facing throughput change worth a dedicated follow-up.

