---
id: rec2-onpath-failloud
title: Heavy-verify rec #2 closure: widening neutral guard + on-path non-finite containment pins
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-10
headline: realized_vol_ratio returns the documented no-fire 1.0 on a non-finite ratio (was: NaN silently hit max widening 1.15); Sites A/C adjudicated NO_CHANGE/EXPLICIT_KEEP and pinned.
surface: [engine/forward_distribution.py, engine/option_pricer.py, tests/test_f4_rv_widening.py, tests/test_forward_distribution_invariants.py, tests/test_option_pricer.py]
---

## Goal

Close recommendation #2 of the 2026-06-09 heavy-verify campaign
(`HEAVY_VERIFY_FINDINGS_2026-06-09.md` §5/§7): the three on-evaluate-path sites that
swallow non-finite input the same way the just-merged (E) fixes (#382/#384/#386) did —
`forward_distribution.py` empirical `np.log` (Site A), `realized_vol_widening_factor`
NaN-ratio fall-through to the 1.15 cap (Site B), `option_pricer._validate_inputs`
non-finite sigma/T pass-through (Site C).

## What we tried

A 12-agent probe→design→adversarial-verify Workflow panel (wf_57ef58ab-517), one
pipeline per site, every claim backed by an executed probe against the real
Bloomberg CSV (1,014,920 rows: exactly 4 NaN closes — BIIB 2020-11-06/2023-06-09,
TPL 2019-05-16/2019-07-09 — zero non-positive, zero inf).

## What worked

- **Site B — the one real fix.** A non-positive close inside the trailing window
  poisons `np.log` → nan std → nan ratio; `nan < threshold` is False so every
  downstream guard is skipped and `min(max_widening, nan)` returns **1.15 max
  widening from garbage**. Guard: non-finite ratio → the function's own documented
  no-fire 1.0 (docstring already promised it for every other degenerate route).
  House precedents: wheel_runner HMM-failure → 1.0, ev_engine non-finite regime
  mult → 1.0, R7-R10 "soft-warns don't fire on absent evidence". Verifier: APPROVE;
  bit-identity on 4,088 real-data probes + 39 fixture replicas; guard is dead code
  on real data (structurally: only zero/neg/inf closes reach it, none exist, and
  the data-integrity suite pins that) → determinism-null, no re-baseline needed.
- **Site A — NO_CHANGE, now pinned.** The unguarded `np.log` is the center of a
  deliberate five-layer design (dropna → isfinite filter → min_samples cascade →
  ev_engine re-filter → R1a). New pins: negative-close filtered like zero (was the
  one unpinned matrix cell), NaN-dropna splice semantics (295→294 — the LIVE BIIB
  path; an ffill "cleanup" would silently change shipped EV), min_samples cascade
  demotion, plus a do-not-clamp comment at the log line.
- **Site C — EXPLICIT_KEEP, now pinned.** Non-finite sigma/T price to honest NaN
  with no raise in every scalar entry point; the NaN EV is hard-blocked by R1a
  (`ev_non_finite`, PR #204). FAIL_LOUD probe-refuted: no per-candidate try/except
  at any decision-layer call site → a raise aborts the whole rank run and erases
  the funnel audit trail. Pinned with `TestNonFiniteSigmaTContract` (full mixed
  {nan, +inf, finite}² matrix minus finite-finite) + a do-not-raise contract
  comment in `_validate_inputs`.

## What didn't

- FAIL_LOUD at any of the three sites: breaks the shipped W41 filter pin (Site A),
  kills whole rank scans on one corrupt ticker (Sites A/C), fights the deliberate
  R1a design (Site C).
- A pre-log positive-price filter (Site A): probe-proven NOT behavior-preserving —
  it splices a return across the corrupt bar (294 vs 293 survivors) and shifts the
  non-overlapping sampling phase.
- NEUTRAL_FALLBACK for the pricer: sigma has no identity element; substituting any
  finite vol fabricates a finite EV from unparseable input — upgrade-shaped.

## How we fixed it

One branch, decision trio untouched: the 9-line Site B guard + docstring bullet in
`engine/forward_distribution.py`, contract comments at Site A's log line and Site
C's validator, 10 new behavior pins across the three existing test files (no new
test file → no TESTING.md/FILE_MANIFEST churn).

## Evidence

- Probe values pinned before writing: clean/NaN/zero/neg overlapping counts
  295/294/293/293; zero-close ratio→1.0 widen→1.0 (was nan→1.15 pre-guard);
  all-greeks 12/12 NaN no-raise; sigma=-0.5/-inf still raise.
- `py -3.12 -m pytest tests/test_f4_rv_widening.py
  tests/test_forward_distribution_invariants.py tests/test_option_pricer.py -q`
  → 90 passed. ruff check/format clean.
- Workflow transcripts under the session subagents dir (wf_57ef58ab-517); probe
  scripts under `swe-rec2/_rec2_probes/` (untracked scratch).

## Unresolved / handoff

- Latent Site C residual (verifier-flagged, needs §2 consent — NOT grabbed): a
  finite market-mid premium + NaN IV + empirical forward returns yields FINITE
  `ev_dollars` with NaN confined to `fair_value`/`edge_vs_fair`; R1a only inspects
  EV, so such a candidate would PROCEED. Unreachable today (ranker premium is
  BSM-coupled to IV, `premium_source='synthetic_bsm'`), becomes real if a
  market-mid premium path lands.
- Snapshot drift on main (s27/s32/s34 `ev_mean`/`spearman_p` vs the #338 baseline,
  s35 green; prime suspect #363 IV gate) — documented in
  `HEAVY_VERIFY_FINDINGS_2026-06-09.md` §9, needs the revert-isolation + re-pin
  protocol in a supervised session.
