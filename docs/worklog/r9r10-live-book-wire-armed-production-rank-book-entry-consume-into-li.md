---
id: r9r10-live-book-wire
title: Armed production rank->book entry (consume_into_live_book) wires make_live_book_tracker R9/R10 onto the live wire
kind: feature
status: in-flight
terminal: UltraCode
pr:
decisions: []
date: 2026-06-01
headline: New WheelRunner.consume_into_live_book pairs the make_live_book_tracker factory (R9 sector 25% + R10 single-name 10%, refusal-only) with the consume_into_tracker rank->book wire, so an over-concentrated open is REFUSED end-to-end on a live path — closing the "factory has zero callers" gap (heavy-verify Category A). Additive, §2-safe (refusal-only; D16 launch gate still refuses negative-EV); touches the wheel_runner trio so it carries a lane-claim + needs the independent §2 second-read.
surface:
  - engine/wheel_runner.py
  - tests/test_ranker_tracker_wire.py
---

## Goal
<!-- What we set out to do, and why. -->

Close heavy-verify 2026-05-31 **Category A**: the D17 concentration caps
(R9 sector 25% NAV, R10 single-name 10% NAV) were armed in code but
**dormant on every live path**. The canonical production factory
`engine.wheel_runner.make_live_book_tracker` (D22 / #303) sets
`enforce_sector_cap` + `enforce_single_name_cap`, but a grep showed it
had **zero callers** outside its own def, so the production rank→book
wire (`consume_into_tracker`) ran an *unarmed* default tracker. The
critic's example: a trader who fills the top ranked names loads e.g. a
$83k LLY put = 33% of a $250k account with no refusal.

## What we tried
<!-- Approaches, in the order we tried them. -->

1. **Grounded the seam first.** Confirmed the card's premise needed
   refining: the cap *mechanisms* are fully built + tested, not missing.
   - Dossier R9/R10 soft-warns already fire in `build_candidate_dossiers`
     when a `portfolio_context` is passed (`candidate_dossier.py` R9
     :461-497, R10 :499-532) — `build_candidate_dossiers` has accepted
     `portfolio_context` since C3 (#... `test_ranker_tracker_wire.py`).
   - Tracker R9/R10 hard-caps already fire at `open_short_put` →
     `_evaluate_d17_hard_blocks` when `enforce_*` flags are on; the
     `make_live_book_tracker` factory sets them, pinned by
     `test_production_tracker_caps.py`.
   - `consume_ranker_row` (the path `consume_into_tracker` loops) issues
     the D16 token then calls `open_short_put`, so the hard-caps fire on
     the full rank→book wire **if the tracker is factory-built**.
2. Concluded the only gap is **no production caller pairs the factory
   with the wire**. Added the missing paired entry.

## What worked

`WheelRunner.consume_into_live_book(...)` — a thin, additive production
entry: builds the tracker via `make_live_book_tracker` (caps armed) and
delegates to `consume_into_tracker`, returning `(tracker, outcomes)`. An
over-concentrated open is refused end-to-end; a diversified open passes.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

- **Adding `portfolio_context` to `rank_candidates_by_ev`** (the card's
  literal framing). Rejected — the EV ranker does NOT run the dossier
  reviewer; R9/R10 live in `build_candidate_dossiers` (soft-warn) and the
  tracker (hard-cap). Threading context into the EV ranker would do
  nothing there, and re-implementing R9/R10 inside the ranker would
  duplicate the reviewer (CLAUDE.md wants ONE reviewer). The caps are
  inherently book-relative (concentration vs held positions), so they
  belong at booking / dossier-with-context, not the candidate-by-candidate
  EV layer.

## How we fixed it
<!-- The approach that shipped. -->

`engine/wheel_runner.py`:
- New module sentinel `_USE_RUNNER_CONNECTOR` so the method can tell
  "caller didn't specify connector" (→ use the runner's own) from an
  explicit `connector=None` (→ connector-less book; the caps are
  notional/NAV-based and fire without one — the data-free test path).
- New `WheelRunner.consume_into_live_book(*, entry_date, initial_capital=
  100_000.0, connector=<sentinel>, rank_kwargs=None, top_n_to_consume=
  None, expiration_date=None, **tracker_kwargs)`. Builds the tracker via
  `make_live_book_tracker(...)`, runs `consume_into_tracker` through it,
  returns `(tracker, outcomes)`.

`tests/test_ranker_tracker_wire.py`: new `TestConsumeIntoLiveBook` (7
tests) — caps-armed on the built tracker; over-concentration refused
end-to-end with `single_name_breach` on the audit log; the *same* open
accepted on an unarmed default tracker (control proving it's the caps);
diversified open succeeds; negative-EV still refused at the D16 launch
gate (§2 no-rescue); signature pin.

§2: additive and refusal-only. The method composes the already-§2-safe
factory + wire; it has no EV path of its own, never touches
`ev_raw` / `ev_dollars` / `prob_profit`, and the D16 launch gate inside
`consume_ranker_row` still refuses `ev_dollars <= 0`. It only ADDS
refusals — it cannot rescue a non-tradeable candidate.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

- `pytest tests/test_ranker_tracker_wire.py tests/test_production_tracker_caps.py`
  → **27 passed** (7 new live-book tests + existing wire/cap tests).
- `ruff check` + `ruff format --check` on both changed files → clean.
- Reproduced: $200-strike put = $20k = 20% of $100k NAV → refused on the
  live wire (`refusal_reason="tracker_rejected"`,
  `_ev_authority_log[-1]["reason"]=="single_name_breach"`); same row opens
  on the unarmed default tracker.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- **Trio change → carries a lane-claim + needs the independent §2
  second-read before merge.** PR body must carry the lane-claim block
  naming `engine/wheel_runner.py` (CI `check_lane_claim.py` reads
  `PR_BODY`).
- The **dossier soft-warn** arming (threading a `portfolio_context` into
  the rank→dossier path so R9/R10 *downgrade* ranked candidates, not just
  refuse at booking) is a complementary follow-up — `build_candidate_dossiers`
  already accepts the param; the gap is production callers (e.g. the
  stateless `/api/tv/dossier`) supplying a context.
- `enforce_delta_cap` / `enforce_kelly_cap` stay OFF in the factory
  (delta cap mis-calibrated; deferred) — out of scope here.
