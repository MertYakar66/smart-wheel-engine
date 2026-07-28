---
id: validation-v2-freeze-replay
title: V2 parameter freeze-replay (C1) — amnesia PIT proof, freeze snapshot lock, frozen-knowledge replay
kind: verification
status: complete
terminal:
pr:
decisions: []
date: 2026-07-12
headline: The "PIT-clean by construction" assertion behind every reported edge becomes an experiment — physical data truncation must reproduce the ranker byte-for-byte, the C1 HMM/GPD freeze snapshot gets a committed refit lock, and a frozen-at-cutoff replay bounds how much rank quality and risk honesty depend on refit recency.
surface: [backtests/freeze_replay.py, scripts/run_freeze_replay.py, tests/test_freeze_replay.py, tests/fixtures/freeze_replay/freeze_snapshot_24t.json, docs/VALIDATION_PHASE_PLAN.md]
---

## Goal

Close C1 (`docs/PRODUCTION_READINESS.md`: "Out-of-sample parameter freeze —
snapshot parameters at a cutoff date, run a held-out backtest; assert the
frozen parameters reproduce the snapshot's classifier output") as the V2
workstream of the validation phase. The design + six falsifiable
expectations were pre-registered in `docs/VALIDATION_PHASE_PLAN.md` section
5 in a commit that precedes all V2 code. Measurement-only; trio untouched.

## What we tried

- Freezing the static hand-set constants "as of a past cutoff": rejected —
  they did not exist then and the hand-tuning is not reproducible; their
  honest treatment is the parameter_oos train-refit (done) and the V3
  plateau sweep. V2's surface is the ONLINE-fit artifacts (HMM, POT-GPD,
  the forward-distribution scenario set).
- Freezing the HMM inside the engine via a patched `GaussianHMM`: rejected
  — the constructor site has no ticker identity to key frozen parameters
  by. The HMM freeze is applied offline instead (fit at cutoff, posterior
  per date, recombine `ev_raw x clamp(mult)` in numpy — the parameter_oos
  rederivation convention).
- Truncating every data file for the amnesia test: rejected — the earnings
  calendar is legitimately forward-looking (the event-lockout gate needs
  future earnings dates), `sp500_credit_risk.csv` turned out dateless (a
  snapshot file — inherent PIT limitation, nothing to truncate), and the
  credit-regime de-rank reads FRED network series that truncation cannot
  cover (disabled on both sides; PIT slice code-verified separately).
  Tier-1 truncates the five unambiguous market series; everything else is
  copied intact (identical on both sides, so it cannot create false
  diffs). Both scope corrections were recorded in the plan doc BEFORE any
  run.

## What worked

`backtests/freeze_replay.py` — V2-a amnesia protocol (byte-preserving
truncation via dtype=str round-trip, fresh `WheelRunner(data_dir=...)` per
pass, A/A determinism control before the A/B full-vs-truncated diff, exact
NaN-aware comparison of every diagnostic column + the drops list); V2-b C1
freeze snapshot (per-ticker HMM start_prob/trans_mat/means/stds/labels/
posterior/multiplier with the production recipe, plus a canonical
return-space POT-GPD fit) with a pure differ + refit verifier; V2-c frozen
replay (context-manager patch of `best_available_forward_distribution` to
the cutoff — wheel_runner imports it function-locally so the module
attribute patch takes effect with zero trio changes — emitting V1's
`TAIL_TABLE_COLUMNS` so `tail_exceedance.full_report` runs unchanged on
the frozen table) + the frozen-vs-production comparison (rank tiers with
block-7 clustered CIs, paired deltas, months-since-cutoff drift buckets,
distribution-source mix). `scripts/run_freeze_replay.py` driver
(amnesia / snapshot / verify / frozen-build / compare). 26 fast engine-free
tests + the slow-lane C1 refit lock.

## What didn't

- Two pre-registration mechanics errors, both caught and corrected in the
  plan doc BEFORE any run: `sp500_credit_risk.csv` was listed tier-1 but
  is dateless (moved to copied-intact with a documented PIT limitation),
  and the original section 5.3 text lumped the F4 vol-ratio into the
  freeze (it is a market-state reading and stays live at T). The 5.4
  expectations were never touched.
- **Harness artifact found by the V2-c-100t run (terminal, died twice per
  protocol):** `frozen_hmm_multipliers` guarded the per-date
  `predict_proba` but left the one-time `hmm.fit(tail0)` unguarded —
  contradicting its own docstring. BIIB carries NaN closes on halt days
  (2020-11-06, 2023-06-09); the 2023-06-09 one sits inside the
  504-return tail at the canonical cutoff and trips the engine's
  deliberate #386 non-finite guard, which `wheel_runner` catches but the
  harness did not. Never surfaced at 24t (BIIB not in UNIVERSE_24).
  The expensive 46-min 100t engine pass had already completed and its
  table is valid; only the cheap multiplier step crashed.

## How we fixed it

Scope corrections recorded in `docs/VALIDATION_PHASE_PLAN.md` 5.1/5.3
with explicit "pre-run correction" markers. The BIIB crash: the fit is
now guarded exactly like wheel_runner's own fit site (degrade to the
neutral 1.0/"unknown"); the same latent gap in `hmm_snapshot_at` /
`gpd_snapshot_at` is guarded too (degrade to no-entry); pinned by two
regression tests using a poisoned mini-monolith (which also had to learn
the AUDIT-VIII P1.5 rotated-column fact — the served close is the CSV's
"high"). `frozen-build` now reuses an existing frozen table so the
terminal's resume costs minutes, not 46. C1 lock re-verified after the
change (fixture reproduces; UNIVERSE_24 has no poisoned tails).
Interpretation note for the 100t compare: BIIB's ~198 frozen rows carry
the neutral multiplier in `frozen_ev_frozen_hmm` — symmetric with
production, where the engine's own catch degraded BIIB the same way.

## Evidence

- `python3 -m pytest tests/test_freeze_replay.py -q` -> 26 passed + the
  C1 lock (1 slow-lane test) passing once the fixture landed.
- Pre-registration: `docs/VALIDATION_PHASE_PLAN.md` section 5, committed
  before any V2 code; section 5.4 expectations frozen.
- **V2-a PASS (2026-07-12)**: A/A determinism clean and A/B
  full-vs-truncated byte-identical at all 5 regime-spanning dates
  (2022-06-15, 2023-11-15, 2024-08-06, 2025-04-15, 2026-05-01) x
  UNIVERSE_24 — zero diffs in any diagnostic column or drop. No
  market-data leakage through OHLCV / vol-IV / treasury / VIX /
  liquidity into the rank path.
- **V2-b REPRODUCED (2026-07-12)**: 24/24 tickers' HMM + GPD refit
  matches the committed 56 KB fixture at rtol 1e-7.
- **V2-c DONE (2026-07-12)**: 1,793 frozen rows joined the V1-a
  production capture row-for-row over 144 holdout dates (scenario
  knowledge frozen at 2023-06-30, up to ~2.9y stale). Top-5 tier rank
  rho SURVIVES the freeze (+0.17 prod vs +0.19-0.21 frozen, CIs
  overlap); pooled cvar_5 breach barely moves (1.51% -> 1.78%, both
  PASS); the degradation lives elsewhere — violation clustering ac1
  0.10 (p=0.08) -> 0.56 (p=0.0005), top-bin population 79 -> 178 rows
  (breach 7.6% -> 8.4%), traded-region breach 3.9% -> 5.4%, all-candidate
  rho turns significantly negative. Refit recency buys violation
  independence and selection discipline, NOT pooled tail-frequency
  accuracy — sharpening I3-E (the rear-view lag dominates even a 3-year
  freeze; tail honesty needs a forward-looking input). Expectations 4/5
  falsified in their pre-registered pooled forms; scorecard in the plan
  doc section 5.6.
- **V2-c-100t DONE (2026-07-12, terminal)**: 18,837 rows joined
  row-for-row over 359 dates. THE decisive answer: the S34-class
  top-tier edge survives the freeze — top15 production +0.294 [0.205,
  0.379] vs frozen +0.366 [0.290, 0.442], both consistent with the
  committed +0.371 [0.25, 0.49]. Top-bin inflation replicates (755 ->
  1,205 rows, breach 7.15% -> 7.80%); traded-region crosses the bound
  (3.42% -> 5.20%); the 24t all-candidate inversion does NOT replicate
  (retro-flagged small-menu noise); clustering already at the floor on
  the dense grid (cadence caveat recorded). Full numbers:
  `docs/VALIDATION_PHASE_PLAN.md` section 5.7.
- Full numbers: `docs/VALIDATION_PHASE_PLAN.md` sections 5.6-5.7.

## Unresolved / handoff

- **V2 CLOSED 2026-07-12** — all four runs done (V2-a PASS, V2-b
  REPRODUCED, V2-c 24t + 100t); C1 closed in full. Workstream verdict:
  the reported top-tier edge does not depend on refit recency; refit
  recency's value is selection discipline + (sparse-grid) violation
  independence; freezing's cost lands exactly in V1's weakest strata
  (top-bin / traded-region tails).
- Next: V3 (parameter-plateau sweep) design + pre-registration.
