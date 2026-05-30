<!-- PR body — see COMMIT_GUIDE.md §3. Omit any empty section (a missing
     section is silence; "N/A" is noise). Name your task card id if allocated. -->

<!-- DECISION-LAYER CLAIM — REQUIRED only if this PR edits
     engine/ev_engine.py, engine/wheel_runner.py, or engine/candidate_dossier.py.
     Claim the file on board #113 first, then edit the block below to name the
     real path (replace the placeholder). CI (scripts/check_lane_claim.py) fails
     a decision-layer PR without a matching claim. Leave the placeholder as-is on
     a non-decision-layer PR — the gate does not run, and the placeholder is not
     a real path so it can never auto-satisfy the gate. -->
<!-- lane-claim
files: engine/<the-decision-layer-file-you-edit>.py
board: https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-NNN
-->

## Summary
<!-- 2-3 bullets: what, why, scope. Task card id (e.g. C7-A) if allocated. -->

## Changes
<!-- file.py — what changed (one bullet per concrete change) -->

## Why
<!-- motivation / constraint / the past incident this prevents -->

## Tests
<!-- exact commands run + what they verified -->

## §2 surface
<!-- Touches the EV decision authority (CLAUDE.md §2)? If yes, how the
     downgrade-only / no-rescue invariant is preserved. If no, say "no". -->

## Tried but rejected
<!-- alternatives considered + why not (omit if none) -->

## Unresolved
<!-- noticed-but-not-fixed; follow-ups (omit if none) -->

## AI handoff
<!-- what the next agent should look at next (omit if none) -->
