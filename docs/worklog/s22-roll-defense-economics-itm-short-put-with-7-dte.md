---
id: S22
title: Roll defense economics (ITM short put with ≤7 DTE)
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

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
