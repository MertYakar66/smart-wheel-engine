---
id: supervised-block-worklist
title: Supervised-block worklist consolidation (post brain-audit fix wave)
kind: docs
status: done
terminal: major-session
pr:
decisions: []
date: 2026-06-11
headline: Single routing doc (docs/SUPERVISED_BLOCK_WORKLIST.md) consolidating every operator-gated item into Block A (Terminal/data) before Block B (coordinated EV re-baseline)
surface: [docs/SUPERVISED_BLOCK_WORKLIST.md]
---

## Goal
After the brain-audit fix wave (#405 WIP landing, #407 D16 token binding,
#408 as_of=None frontier gate, #410 zero-skew docs note), everything left in
the queue deliberately waits for the operator — but the items were scattered
across issues (#339/#354/#355/#357/#369/#372/#378/#402), the brain-audit
report (M2/M4), the D19/D21/recal scope draft, and session memory. Produce one
routing document so the next supervised session starts from a checklist, not
an archaeology dig.

## What we tried
Direct synthesis from: the 2026-06-08 batch decision (data queue has no free
forward progress; batch A+B+C into one Terminal session),
`docs/bloomberg_refresh_runbook.md` (plan-of-record for the pull),
`docs/REBASELINE_D19_D21_RECAL_SCOPE.md` (the entanglement analysis), the
brain-audit M4 scoping run (2026-06-11 Fable/Explore pass over
`engine/transaction_costs.py`).

## What worked
The two-block structure with a hard ordering: Block A (Terminal/data; moves
the data) strictly before Block B (coordinated decision-layer re-baseline;
re-pins against that data). Doing B first pays the ~4h re-baseline tax twice.

## What didn't
- First draft claimed the D19/D21/recal scope doc was "local, uncommitted" —
  wrong; it is tracked on main (the untracked copy in the primary clone is a
  stale-branch artifact). Corrected before commit.
- An "IBKR Phase-1 PR" queue item turned out to be a phantom from a stale
  memory-index line — Phases 1+3+4 and the live Gateway puller all merged
  2026-06-08 (#359/#362/#368). Dropped.

## How we fixed it
`docs/SUPERVISED_BLOCK_WORKLIST.md`: Block A = data queue (#339/#355/#354/
#357) + reserved (E)s (#369/#372/#378) + NFLX 10× mis-scale + the M4-enabling
option-volume capture + post-refresh hygiene. Block B = D21 + D19 +
recalibration (operator draft) + brain-audit M2 (widening coverage) + M4
(size-impact wiring, same cost block as D19) + #402 re-pin + the
fingerprint-pins-OHLCV-only gap. Plus the standing operator decisions that are
not session work (Task Scheduler trigger time, dashboard/ restore note).

## Evidence
- M4 scoping verdict (2026-06-11): sqrt impact term inert at every production
  call site (no caller passes `adv_contracts`); ungated wiring is EV-moving
  even at 1 contract (S27/S32/S34/S35 + smoke all run contracts=1); the input
  the model wants (per-contract option ADV) exists in no production data
  source — hence A9 (capture/proxy) before B5 (wiring).
- #402 s27 drift independently reproduced from a third environment on
  2026-06-11 (local slow-lane run: ev_mean −19.898 vs pinned −19.915).

## Unresolved / handoff
The worklist itself. Strike items as they merge; archive the doc when both
blocks land.
