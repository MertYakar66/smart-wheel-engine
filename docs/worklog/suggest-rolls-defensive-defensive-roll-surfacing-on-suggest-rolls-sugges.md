---
id: suggest-rolls-defensive
title: Defensive-roll surfacing on suggest_rolls / suggest_call_rolls (S47 F-S47-1)
kind: fix
status: in-flight
terminal: UltraCode
pr:
decisions: []
date: 2026-06-01
headline: suggest_rolls / suggest_call_rolls no longer go silent on a challenged position — an opt-in include_defensive surfaces credit-gate-failing (debit) rolls flagged defensive=True (each scored through EVEngine.evaluate), and .attrs["defensive"] always reports how many defensive rolls exist so the credit-only default is never a silent zero.
surface:
  - engine/wheel_tracker.py
  - tests/test_suggest_rolls_defensive.py
---

## Goal
<!-- What we set out to do, and why. -->

Close the highest-value half of the S47 trust verdict — *"TRUST IT FOR
ENTRY, DISTRUST IT FOR MANAGEMENT."* S47 (`docs/worklog/s47-...md`)
reproduced that `WheelTracker.suggest_rolls` returns an **empty frame**
on a challenged (ITM) short put under the default
`min_net_credit=0.0`: every defensive roll is a debit, so the
credit-only filter prunes all of them. A trader on defaults reads the
empty frame as "no action available" when the honest answer was "N
defensive rolls exist, all gated by the credit filter." Same gap on the
covered-call leg (`suggest_call_rolls`) for a call the stock has rallied
through.

Scope for this merge-unit: surface the *already-enumerated-but-credit-
suppressed* defensive rolls. Deliberately **off the §2 trio** — the fix
lives entirely in `engine/wheel_tracker.py` (NOT `ev_engine.py` /
`wheel_runner.py` / `candidate_dossier.py`). The more invasive
same/near-strike ITM roll-out enumeration (S47 F-S47-3) and the
basis-aware covered-call grid (F-S47-2, which touches `wheel_runner.py`
= trio) are deferred to follow-on PRs.

## What we tried
<!-- Approaches, in the order we tried them. -->

1. **Flip the default to surface defensive rolls.** Rejected — see below.
2. **Opt-in `include_defensive` + always-on visibility (shipped).**

## What worked

The shipped design (see "How we fixed it"): an opt-in keyword that
surfaces the suppressed rolls flagged `defensive=True`, plus a
`.attrs["defensive"]` summary that is populated on **every** return path
regardless of the flag — so even the bare credit-only default advertises
that defensive rolls exist.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

- **Flipping the documented credit-only default to surface debit rolls
  by default.** The `min_net_credit=0.0` docstring explicitly defends
  "keeps only credit-rolls (the conventional wheel discipline)."
  Silently changing what the default trade-set contains is a
  product-*policy* call, not a bug fix, and it would change the meaning
  of every existing caller's output. Left the policy lever to the
  operator; fixed only the *silence* (the actual S47 harm) by making the
  suppressed count always visible and the surfacing opt-in. The operator
  can flip the default in review if desired.

## How we fixed it
<!-- The approach that shipped. -->

`engine/wheel_tracker.py`, mirrored across `suggest_rolls` (put) and
`suggest_call_rolls` (call):

- New keyword-only `include_defensive: bool = False`.
- The credit gate now computes `is_defensive = net_credit_debit <
  min_net_credit`. When `is_defensive and not include_defensive` it drops
  to the existing `gate="credit"` log exactly as before (legacy path
  byte-for-byte). When `include_defensive=True`, the candidate is instead
  **scored through `EVEngine.evaluate`** and appended as a row with
  `defensive=True`.
- New `defensive` column appended to `_ROLL_COLUMNS`; credit rolls carry
  `defensive=False`.
- `_attach_drops_summary` (the single chokepoint at all 8 return sites)
  now also attaches `.attrs["defensive"] = {"available", "surfaced",
  "suppressed", "included"}`. `available == surfaced + suppressed`
  because a defensive roll is always exactly one of surfaced (in-frame)
  XOR suppressed (in the credit drops).

§2 safety: the change only chooses *which candidates to score* and how
to *label/surface* them. Every emitted row — credit or defensive — still
passes through `EVEngine.evaluate`; `recommend` stays `roll_ev >
hold_ev`; nothing rescues a negative-EV trade. The §2 evaluate-count
invariant (`call_count == 1 + len(df)`) holds in both modes.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

Inline smoke (challenged put strike 95 / spot 82, 21 DTE elapsed):

```
include_defensive=False | rows=0  | defensive={'available':16,'surfaced':0, 'suppressed':16,'included':False}
include_defensive=True  | rows=16 | defensive={'available':16,'surfaced':16,'suppressed':0, 'included':True}
   all 16 surfaced rows defensive=True, min net_credit_debit=-1244.28, all roll_ev finite
```

Tests (interpreter `Python312`, `SWE_DATA_PROVIDER=bloomberg`):

- `pytest` on all four roll test files → **61 passed** (15 new in
  `test_suggest_rolls_defensive.py` + 46 existing, no regression).
- Adversarial review (3 independent lenses — §2 / correctness /
  edge-tests, each grounded on the diff) → **§2 verdict: holds** on all
  three; no blocker/major findings. Acted on its findings: replaced a
  vacuous credit-label test with a natural credit+defensive **mix**
  fixture (spot 98 / strike 95 → 3 credit + 13 defensive rows),
  de-guarded the call-leg surfacing test, and tightened the
  `available` / `recommend` docstrings (`available` counts only rolls
  that *reached* the credit gate; `recommend=True` on a defensive roll
  is least-bad-vs-hold, not an absolute buy).
- `ruff check` + `ruff format` on both changed files → clean.
- Full suite: see PR / commit `Tested:` block.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- **F-S47-3 (next slice, off-trio):** enumerate the canonical
  same/near-strike roll-out-in-time. The strike grid currently drops
  `new_strike >= current_spot` (put) / `<= current_spot` (call), so the
  defensive "roll out at the same strike for a credit" is never produced.
- **F-S47-2 (§2-gated, needs lane-claim):** basis-aware covered-call
  strike grid in `engine/wheel_runner.py::rank_covered_calls_by_ev` (it
  takes `shares_held` but no `cost_basis`/`min_strike`, so it never
  proposes a call at/above an underwater basis).
- **Default policy:** whether to flip `include_defensive` to `True` by
  default (or have the dashboard/trader-session caller pass it) is an
  operator decision — the silent-zero is now reported either way.
