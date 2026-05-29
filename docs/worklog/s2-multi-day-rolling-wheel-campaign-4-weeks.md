---
id: S2
title: Multi-day rolling wheel campaign (4 weeks)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

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
