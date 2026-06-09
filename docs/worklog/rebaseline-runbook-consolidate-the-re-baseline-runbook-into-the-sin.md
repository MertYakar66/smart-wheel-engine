---
id: rebaseline-runbook
title: Consolidate the re-baseline runbook into the single authoritative session checklist (data + 3 (E) fixes + re-baseline)
kind: docs
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: NEXT_DATA_SESSION_RUNBOOK elevated from data-queue plan to the one ordered checklist for the supervised re-baseline session — universe data → 3 (E) trio/risk-gate fixes → re-baseline → frontier re-picks → (D) pulls
surface: [docs]
---

## Goal
After the 2026-06-09 data-test audit landed W14–W37 (8 PRs) and opened the three
(E) issues (#372 R9→GICS, #369 #363 IV-gate fallback, #378 IV-staleness +
rate-fallback), make `docs/NEXT_DATA_SESSION_RUNBOOK.md` the **single
authoritative checklist** for the supervised Bloomberg re-baseline session — so
the operator runs one ordered pass and the (E) fixes land **before** the
snapshot re-pin (they move EV output; re-pinning first would force a second
~4 h re-baseline). Docs-only prep; no engine/data work (all (E)/(D) are gated
behind the §2 ceremony / a producer change).

## What we tried
Extended the existing #365 runbook (Phase A/B/C, data-queue-only) rather than
rewrite it — preserved the CASY spec pointer, the 10-name blue-chip table, the
A3 frontier-refresh line edits, and the re-baseline command block; restructured
into five strictly-ordered phases and inserted the (E) fixes as Phase 2.

## What worked
The governing rule "everything that moves the frontier or the EV output lands
before the snapshot re-pin" cleanly orders the whole session and explains *why*
the (E) fixes sit between the universe data and the re-baseline. An
at-a-glance ordered table (phase · what · Bloomberg? · §2?) front-loads the
order so the operator can't re-pin early.

## What didn't
n/a (docs consolidation). One honesty correction folded in: **W28**
(`edge_vs_fair` ≡ 0) is grouped with the (D) pulls but **cannot** be cleared by
a Bloomberg pull — it needs a market-mid option-premium producer the connector
lacks (C4). Flagged as BLOCKED rather than implying it's a simple pull.

## How we fixed it
- Phase 2 details each (E) fix with issue #, the W-finding + audit §ref, the
  exact file:line (`risk_manager.py:1755`/`1579`; `wheel_runner.py:1082`/`1101`,
  `:2418`; `data_integration.py:323`; `_resolve_pit_atm_iv` `wheel_runner.py:153-202`),
  the fix direction, the §2/held-for-review note, and the **characterization-test
  flip** (W17 for #372, W27 for #369) so the pinning tests move with the fix.
- #372 carries the W19 GICS-quality fallback (GICS primary → DEFAULT_SECTOR_MAP
  for the 95 NaN/seam-leavers → counted `'Unknown'`, never a silent collapse).
- Phase 3/4 add the frontier-coupled re-pins the audit introduced: `EXPECTED_FRONTIER`
  **and** the data-test `FRONTIER` constant (both suites), full-universe 480/31,
  W16/W30 JPM re-pick (iff the frontier moved; W15/W32 sign controls are robust).
- Phase 3 notes the single re-pin absorbs three things at once: Phase-1 data, the
  #363 `ev_mean` serving-logic re-pricing (raw-byte fingerprint stays silent by
  design — caught by W14), and the Phase-2 (E) fixes.

## Evidence
- Verified `origin/main @ 9847edc` is the live HEAD with all audit PRs (#370–#380)
  merged; decision trio byte-identical across `d0cdcde..9847edc` (§2 held).
- Issue titles/state pulled live: #372/#369/#378 all OPEN, exact framing matched.
- File:line refs grepped against the worktree source (not memory).
- Touches: `docs/NEXT_DATA_SESSION_RUNBOOK.md` (rewrite), `FILE_MANIFEST.md`
  (row description), this fragment. No `.py`, no data, no trio.

## Unresolved / handoff
- The runbook is the plan-of-record; **execution waits for a live Terminal
  session** (operator-driven). Nothing else lands until then.
- The three (E) fixes are trio/risk-gate PRs — each lane-claimed + held for the
  operator's §2 panel review; do **not** grab them autonomously.
- Branch `claude/rebaseline-runbook` pushed for review; no PR opened (per the
  ask-before-opening-PRs standing rule) — open on the operator's go.
