---
id: pr-backlog-clear-2026-07-04
title: "PR backlog cleared: #414/#413/#457/#462 merged, #415 closed superseded, July self-skip pinned"
kind: fix
status: shipped
terminal: X
pr: 474
decisions: []
date: 2026-07-04
headline: "Operator-directed backlog clear: all 5 open PRs resolved against post-#472 main (2 straight merges, 2 rebuilt-then-merged, 1 closed as superseded), plus the third instance of the silent-self-skip class pinned to a dated as_of"
surface:
  - engine/wheel_runner.py
  - tests/test_macro_event_gate_wiring.py
  - tests/test_token_param_binding.py
---

## Goal

Operator: "merge all in without any errors." Five open PRs, several
predating the #472 frontier bump that changed the ground under them.
Method: parallel per-PR fitness assessment (adversarially verified for
the close/hold verdicts), then serial merges easiest-first so each
harder branch updates against the freshest main.

## What we tried / worked

- **#415 (theta docs) CLOSED superseded** — main's copies (via #422,
  `164d6c5`) are strictly newer: branch content PLUS the 2026-06-25
  disk-verified outcome tables. Merging would have deleted them.
- **#414 (validation docs) MERGED `c9d9506`** — sole carrier of the
  top-20 mega-cap validation milestone; clean after a branch refresh.
- **#413 (theta BRKB fix) MERGED `c58065a`** — the concatenated-form
  fix was never on main; main's own docs cited "merge BRKB (PR #413)"
  as the pending step.
- **#457 (splice classifier) MERGED `07962ed`** — its base pointed at
  CLOSED #455's dead branch (the "GitHub will retarget" assumption
  broke when #455 closed unmerged). Retargeted to main, dropped its
  stale #455 data payload (5-file conflict, main taken wholesale),
  grafted the 197-line classifier block onto main's re-pinned
  integrity test. The classifier is validated by the fresh data: 0
  artifacts with KLAC 10:1 / CRWD 4:1 / DD / HON splits present and
  correctly back-adjusted — it now auto-catches the next pull's
  splices with no allowlist edit. Bonus: the merge fixed main's stale
  FILE_MANIFEST row for `test_w1_data_wiring.py` (still described the
  lifted strict-xfail).
- **#462 (macro event-gate) MERGED `9f23135`, default-OFF** — the
  hold was an *arming* decision, not a merge defect; "merge all"
  resolves it. Re-anchored `_register_macro_events` into #464's
  de-silenced D6-1 registration flow across both conflicted rankers
  (all 4 signatures + 3 call sites verified), added the missing
  TESTING.md taxonomy row (the branch's sole CI failure), and re-dated
  the WIRING_CAMPAIGN §3A claim. **Arming remains operator-gated**:
  `use_macro_event_gate=True` still empties the book under
  whole-window semantics.

## What didn't

- **The silent-self-skip class struck a third time.** The #462
  worktree suite showed 29 skips vs the 28 baseline; the delta was
  `test_token_param_binding.py`'s escape-hatch test self-skipping
  ("No AAPL row") because July earnings season locks AAPL on the live
  path — pre-existing on main, masked in earlier censuses by the
  flaky Theta-larder skip offsetting the count. Both #462's flag pair
  and this test are now pinned to dated `as_of=2026-06-04`, where the
  book is deterministic and the tested property is date-agnostic.
  Running census lesson: count-matching is not name-matching.

## Evidence

- Assessment: 5 parallel verdict agents + adversarial re-verification
  of the close/hold verdicts (~357k tokens); #457's classifier logic
  independently replicated against main's data pre-merge.
- Every merge at 9/9 CI with CLEAN state re-verified post-main-move;
  #462 additionally carried a full in-worktree fast suite (3,457
  passed / 0 failed) before push, wheel_runner being trio.
- End state: **0 open PRs**; main green through `9f23135`.

## Unresolved / handoff

- 13 unmerged remote branches remain (no PRs) — mostly historical
  carriers whose content is on main or superseded (e.g.
  `data/bloomberg-refresh-2026-06-02` REVERTS engine per memory;
  `claude/rescue-2026-06-15-fixes` is an unreviewed preservation).
  Deletion/keep triage table delivered in-session; operator call.
- Macro-gate ARMING (entry-proximity semantics) remains the
  operator-gated follow-up documented in WIRING_CAMPAIGN §3A.
