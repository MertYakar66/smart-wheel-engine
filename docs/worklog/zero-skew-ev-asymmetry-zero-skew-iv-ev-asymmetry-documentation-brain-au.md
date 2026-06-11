---
id: zero-skew-ev-asymmetry
title: Zero-skew IV EV asymmetry documentation (brain-audit follow-up)
kind: docs
status: shipped
terminal: UltraCode
pr:
decisions: [D9]
date: 2026-06-11
headline: "docs(skew): document the zero-skew EV asymmetry — short-put conservative (25Δ premium understated 13–41%), covered-call optimistic (~6–12% overstated) — brain-audit M-dim3 follow-up in DECISIONS.md D9 and docs/DATA_ACQUISITION_ROADMAP.md §1"
surface:
  - DECISIONS.md
  - docs/DATA_ACQUISITION_ROADMAP.md
---

## Goal

The 2026-06-11 brain audit (dimension 3 — Pricing & Greeks) flagged the
zero-skew limitation as *asymmetric* and under-documented. The existing docs
describe zero-skew as "uniformly conservative" — which is wrong for the
covered-call leg. This worklog records where the corrective note was placed
and why.

## What we tried

Candidate locations for the note:

1. **`DECISIONS.md` D9** — the canonical record of why SVI/skew is dormant.
   Operators who want to understand the zero-skew decision come here.
   Post-audit annotations on decisions are conventional (see D23 pattern).
2. **`docs/DATA_ACQUISITION_ROADMAP.md` §1 "Honest Bloomberg limits"** —
   the paragraph at the top of that file already describes `put_iv ==
   call_iv` for ~100% of rows, but gave no EV-impact direction. This is
   where operators reading CC ranker output will encounter the limitation.
3. `docs/DATA_POLICY.md` — considered but rejected: it documents data
   tiers and refresh procedures; the IV asymmetry is an *engine-impact*
   note, not a data-ops note. The IV row is already present in the
   capability matrix without a suitable host section.
4. `docs/IBKR_EV_CALIBRATION.md` — considered but rejected: it is a
   retrospective calibration report, not a forward-looking operator guide.

## What worked

Two locations chosen (one decision-layer annotation + one data-ops
paragraph), matching the brain audit's recommendation that the note live
"where an operator reading CC ranker output would actually encounter it."

## What didn't

Scattering the note across three or more files — the audit's finding is a
single quantitative statement; one paragraph per audience (decision-layer
reader / data-roadmap reader) is sufficient.

## How we fixed it

**`DECISIONS.md` D9** — added a post-audit annotation block immediately
before the `---` separator, following the exact pattern of D23's
"Post-ship validation" block.

**`docs/DATA_ACQUISITION_ROADMAP.md` §1** — appended two sentences to the
existing "The IV file is ATM-only" bullet, adding the direction of bias
per leg and a pointer to the audit report.

## Evidence

Source numbers from `docs/BRAIN_AUDIT_2026-06-11.md` §3 (dimension 3):

> the zero-skew IV limitation is *asymmetric* — short-put EV is
> conservative (25Δ put premium understated 13–41% vs a real smile) but
> **covered-call EV is optimistic (~6–12% overstated)**. The docs treat
> zero-skew as uniformly conservative; they shouldn't.

No code changed. `check_manifest_coverage` passes (worklog directory is
glob-covered; no new non-worklog file introduced). Worklog index regenerated.

## Unresolved / handoff

The underlying data gap (no smile on Bloomberg) is tracked as T0-2 in
`docs/DATA_ACQUISITION_ROADMAP.md`. Wiring the deep 5×5 IV-surface archive
(T0-1, "highest ROI") would eliminate the put-leg under-statement without a
new pull; a full moneyness grid pull (T0-2) would eliminate both biases.
Until then the quantitative ranges from the audit (13–41% put under-stating,
6–12% CC over-stating) are the operator's calibration haircut.
