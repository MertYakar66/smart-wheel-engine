<!-- PR body — see OPERATING_MODEL.md §9.7; the Run Summary (§4.4) goes in a PR comment. Omit any empty section (a missing
     section is silence; "N/A" is noise). Link the campaign issue if the run belongs to one. -->

<!-- DECISION-LAYER CLAIM — REQUIRED only if this PR edits
     engine/ev_engine.py, engine/wheel_runner.py, or engine/candidate_dossier.py.
     Edit the block below to name the real path(s) (replace the placeholder) and
     link the campaign issue or the Execution Prompt that authorised the edit.
     CI (scripts/check_lane_claim.py) fails a decision-layer PR without a
     matching claim. Leave the placeholder as-is on a non-decision-layer PR —
     the gate does not run, and the placeholder is not a real path so it can
     never auto-satisfy the gate. -->
<!-- lane-claim
files: engine/<the-decision-layer-file-you-edit>.py
campaign: <link to the campaign issue or the Execution Prompt comment>
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
<!-- Touches the EV decision authority (OPERATING_MODEL.md §7 Decision integrity / §9.2)? If yes, how the
     downgrade-only / no-rescue invariant is preserved. If no, say "no". -->

## Tried but rejected
<!-- alternatives considered + why not (omit if none) -->

## Unresolved
<!-- noticed-but-not-fixed; follow-ups (omit if none) -->

## AI handoff
<!-- what the next agent should look at next (omit if none) -->
