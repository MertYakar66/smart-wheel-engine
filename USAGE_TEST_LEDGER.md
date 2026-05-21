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
(`engine/strangle_timing.py`) — CLAUDE.md §4's one timing-gated
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
  covered-call leg: an in-scope strategy (CLAUDE.md §4) with no EV
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

## 2. In flight

_(none currently)_

---

## 3. Queued

### S5 — Live MCP chart in the loop

**Purpose.** Exercise the just-shipped TradingView MCP integration
(Stages 1–3 + live-verify fixes + `tv quote` price wiring) in a real
dossier flow.

**Setup.** TradingView Desktop running with CDP on
`localhost:9222`; tradingview-mcp `tv` CLI on PATH;
`SWE_USE_MCP_CHART=1`. Unlocks the `proceed` verdict that offline
sessions cannot reach. Operator setup required.

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
