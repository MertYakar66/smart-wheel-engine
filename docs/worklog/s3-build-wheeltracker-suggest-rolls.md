---
id: S3
title: Build `WheelTracker.suggest_rolls(...)`
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

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
