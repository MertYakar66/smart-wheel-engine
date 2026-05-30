---
id: S8
title: Wheel-cycle-to-completion
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

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
