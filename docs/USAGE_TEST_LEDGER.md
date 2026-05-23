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
  launch-gate contract. **Logged.**

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

---

## 4. Candidate (not yet selected)

Worth running when scope and time allow:

_(none currently)_
