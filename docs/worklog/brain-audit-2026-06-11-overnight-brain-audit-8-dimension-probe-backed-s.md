---
id: brain-audit-2026-06-11
title: Overnight brain audit: 8-dimension probe-backed soundness review + full-suite green + evidence archive
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-11
headline: 8/8 dimensions SOUND_WITH_CAVEATS, zero new HIGH/CRITICAL; EV integral exact to <5e-9; §2 verified fresh (560 launch-blocker tests); suite 3,126 green; 4 new MEDIUMs dispositioned.
surface: [docs/BRAIN_AUDIT_2026-06-11.md, docs/HEAVY_VERIFY_FINDINGS_2026-06-09.md, docs/verification_artifacts/brain_audit_2026-06-11/, docs/verification_artifacts/efix_ab_2026-06-10/]
---

## Goal

Operator mandate (away 14h): "thorough analysis on whether the engine's logic is
safe and sound … verify the brain, then start testing." Full campaign report:
`docs/BRAIN_AUDIT_2026-06-11.md`.

## What we tried

8-dimension multi-agent review panel (workflow `wf_b1f6f6aa-2c9`, 8 reviewers,
1.07M tokens) against `origin/main @ 7f9dc10` in a pinned worktree — EV core
math, forward distribution, pricing/Greeks, tail/regime overlays, costs, §2
integrity, calibration evidence, data foundation. Every claim probe-executed;
adversarial verification armed for any NEW HIGH/CRITICAL (never triggered).
The three most load-bearing probes re-executed first-hand by the session.

## What worked

- EV integral hand-replicates to <5e-9 on all 21 EVResult fields incl. a −30%
  crash path; BSM parity 8.5e-14; IV percent→decimal proven end-to-end.
- §2 firewall fresh-verified: all candidate paths route through evaluate,
  R1–R11 downgrade-only, dealer clamp bounded; 560 launch-blocker tests green.
- Full suite 3,126 passed / 0 failed (4m03s).
- The JPM 5-ticker-smoke "missing row" is the event lockout correctly refusing
  to sell through 2026-07-14 earnings — safety reflex confirmed, not a defect.
- Panel falsified the session's own stale briefing (frontier is 2026-06-04,
  not 2026-03-20) — independence demonstrated.

## What didn't

No soundness breaks found. Four NEW MEDIUMs (all bounded, dispositioned in the
report §3): D16 token not parameter-bound at consume; RV widening covers only
the put-entry ranker; `as_of=None` bypasses the staleness gate (CTRA/LW rank on
stale spots); size-impact cost model has zero callers.

## How we fixed it

Nothing engine-side was changed autonomously — the MEDIUMs are §2-adjacent and
routed to the supervised decision-layer/re-baseline queue per the report's
dispositions. This PR archives the evidence: campaign report, the previously
untracked heavy-verify register, the 8 A/B backtest payloads + comparator
(evidence for issue #402), and the 23 probe scripts.

## Evidence

- `docs/BRAIN_AUDIT_2026-06-11.md` (report);
  `docs/verification_artifacts/brain_audit_2026-06-11/` (probes);
  `docs/verification_artifacts/efix_ab_2026-06-10/` (payloads + comparator).
- Workflow transcripts under the session subagents dir (`wf_b1f6f6aa-2c9`).

## Unresolved / handoff

- The four MEDIUM dispositions (report §3) await the operator's supervised
  block; the zero-skew asymmetry docs note (CC EV ~6–12% optimistic) should
  land wherever the smile limitation is documented.
- Snapshot drift re-pin remains tracked in issue #402.
