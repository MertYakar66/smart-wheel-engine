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

### S27 — CC dividend realism (VZ / JPM / MSFT / KO / AAPL / WMT)

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

- **(F4 — gap, observability) ITM CC strikes are silently
  skipped.** `target_deltas=(0.70, 0.80)` returns an empty
  frame on every ticker in the matrix — engine does not
  produce ITM CC candidates. `_solve_call_strike` (and the
  downstream chain-clipping) likely floors the solved strike
  to OTM. **Auto-mitigates** the worst-case ITM-near-ex-div
  early-exercise scenario, but **silently** — there's no
  signal in the output saying "ITM strikes skipped." A
  trader who deliberately wants to write ITM covered calls
  (e.g. to lock in upside on a held position with a known
  ex-div) gets an empty frame and no explanation. **Logged.**

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
  are silently skipped (F4). The continuous BSM q is a
  partial safety net for the truncated names but is not the
  same instrument as the discrete penalty.

- **The realism gap is not in the engine's logic.** Three of
  the four findings (F2/F3/F5) are data-layer or
  observability gaps, not engine-math gaps. F4 is an engine
  scope-of-output gap rather than a logic bug.

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

- **Fix #3 (engine surface, may not be wanted):** add a
  `cc_strike_floor: str | None = "otm"` kwarg to
  `rank_covered_calls_by_ev` that controls whether ITM
  strikes can be produced. Default to current behaviour
  ("otm"); allow `"any"` for traders who want ITM as an
  explicit choice. Couples with a drops entry
  (`gate="strike_itm_skipped"`) so the silent skip becomes
  observable. The natural follow-on Sn after Fix #3 ships
  would re-run S27's ITM probe on VZ/JPM with the new flag
  and confirm the early-exercise penalty actually fires in
  the EV math.

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

- **Single-as_of test (2026-03-20).** Repeating S27 at a
  different as_of with different inside/outside groupings
  would confirm the F3 truncation generalises (versus
  "the file is fresh through 2026-Q1 but stale after"). A
  cleaner phrasing: re-run S27 at `as_of=2025-12-01` to see
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
