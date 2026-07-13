# Validation phase — closing findings record (2026-07-12 → 2026-07)

**Status: DRAFT — V1–V5 CLOSED and final; V6 read in flight** (the one
pre-registered lockbox spend, `scripts/run_v6_lockbox.py`, running on the
deep-data terminal). §7 of this doc and the plan's §9.3 + §2 ledger row
complete when the report lands. Everything above §7 is final.

This is the phase roll-up. The full per-workstream record — designs,
pre-registered expectations, scorecards, and result tables — lives in
`docs/VALIDATION_PHASE_PLAN.md` (V1 additionally in
`docs/VALIDATION_V1_TAIL_EXCEEDANCE_FINDINGS_2026-07-12.md`). Nothing here
supersedes those; this doc exists so the phase's deployment-relevant
conclusions can be read in one sitting.

**Discipline held throughout:** every workstream pre-registered its design
and falsifiable expectations in a commit BEFORE any run; falsified
expectations are recorded as falsified (V2-c #4/#5, V3 #2, V4 #5, V5 #4's
trough clause); every harness is measurement-only (read-only ranker calls,
no production default mutated, nothing fed back into EV/verdicts/gates);
the decision-layer trio was untouched; **no engine change ships from this
phase** — findings go to the re-baseline queue (§5).

---

## 1. The phase verdict in one paragraph

The engine's reported edge is real where it claims to be and honest about
its own construction: the ranking pipeline is leakage-free by experiment
(V2-a), reproducible to rtol 1e-7 (V2-b), its top-tier edge survives a
~2.9-year parameter freeze at the real menu scale (V2-c), and none of its
hand-set constants is a fitted peak (V3). Its risk numbers pass their
pooled frequency tests (V1) — and the phase's sharpest results are about
where that pooled honesty decomposes: tail optimism concentrates exactly
in the highest-confidence, actually-traded rows (winner's curse, F-V1-2),
entry-time conditioning is structurally blind to crisis onset (F-V1-1),
violations arrive in bursts (F-V1-3), breaches run severe (F-V1-4), and
those measured blind spots COMPOSE — the gate stack admits ruin-class
calm-onset books (F-V5-1). Capacity is capital-bound at $1M and
proxy/participation-bound at $10M with impact ≤ ~2.5% of premium (V4).
The load-bearing crisis protection is not a detector — it is the CSP
mandate's inability to be forced out (V5-b). V6 tests whether the refusal
mechanism generalizes to 2008; its verdicts drop into §7 verbatim.

## 2. Per-workstream verdicts (one block each; numbers final)

**V1 — tail-risk exceedance (CLOSED 2026-07-12).** Official verdict
`OVERALL: PASS` on both scales (100t: p25 17.75% vs 25% nominal,
conservative; cvar_5 pooled 2.32% vs the 5% ES bound, CI [1.65%, 3.10%]).
The PASS decomposes into F-V1-1/2/3/4 (§5) plus F-V1-5 (calibration
structure: conservative pooled / vacuous middle / optimistic top). One
harness artifact (the short-put win point-mass making mid-quantile
coverage vacuous) was found and fixed during V1-a before V1-b ran.

**V2 — parameter freeze-replay, C1 (CLOSED 2026-07-12).** The amnesia
test PASSed with zero diffs on all five dates (full-pipeline PIT proof by
physical truncation); the freeze snapshot REPRODUCED 24/24 tickers at
rtol 1e-7 and is now a committed slow-lane lock; the frozen replay at
100t (18,837 joined rows, 359 dates, up to ~2.9y stale) preserved the
tradeable-tier edge — top15 rho production +0.294 [0.205, 0.379] vs
frozen +0.366 [0.290, 0.442], both consistent with the committed +0.371.
Verdict: **the reported top-tier edge does not depend on refit recency or
on any parameter the past could not have had.** What refit recency
actually buys: selection discipline (the freeze mints 1.6–2.25× more
top-bin rows, breaching worse) and, on sparse calm grids, violation
independence — staleness expresses itself exactly where V1 said the
engine is weakest.

**V3 — parameter-plateau sweep (CLOSED 2026-07-13).** No PEAK verdict
anywhere: no constant in the swept surface is a fitted artifact. F4 is
PLATEAU on both axes at both scales — because it is **barely
load-bearing** (fire rate moves ~5× with zero measured tail-coverage
change; the OFF-control differs by ~2 breaches in 1,596 rows). The POT /
ξ-gate / penalty surface is unpowered everywhere observed (`heavy_tail`
fired 0.00% at both scales; the POT diagnostic became powered at 100t,
4.12% ≥ the 2% floor, but touches no decision input — sweep deferred with
that rationale). The R11 sweep produced F-V3-1 (§5).

**V4 — capacity curve (CLOSED 2026-07-13).** At the S34-class
configuration: **$1M is capital-bound at every contract size** — BP
refusals from N=1 at 100 names (the book holds ~47 concurrent positions);
market impact is not the binding constraint. At $10M the book becomes
proxy/participation-bound with a clean interior impact knee at the thin
proxy (r=1e-5, N*=10, zero BP refusals), the linearity A/A is substantive
(return/N to 2.15e-12), per-contract capture declines monotonically, and
modeled impact stays ≤ ~2.5% of premium everywhere. The knee's location
is proxy-dominated — the data-grounded number is deferred to the Theta
option-volume pull (E-13) by design, because a data-free point claim was
pre-registered as dishonest. Cross-platform replication: the Mac arm
matched the Windows arm byte-for-byte across all 16 grid points.

**V5 — reverse stress (CLOSED 2026-07-13).** The hindsight adversary
bounded by the entry-time gate stack found 8 ruin dates (≥ 25% NAV in one
cycle), every one a COVID-onset entry; the two deepest books lose
36.5% / 36.4% of NAV from entries at VIX 13.7 / 14.2, at 3.18–7.07× the
modeled book CVaR (F-V5-1, §5). The top-bin-only adversary is bounded
below ruin (max 19.2% NAV). V5-b resolved margin procyclicality honestly:
100% of the COVID book finished ITM, yet the measured trough is 10.03%
NAV / terminal 7.54% (lower bound: intrinsic-only marking,
capture-limited book) — while the Reg-T levered twin is called on day 13
of COVID at ×1.0. **The classical margin spiral is structurally absent
under the CSP mandate; the inability to be forced out is the quantified,
load-bearing protection.**

**V6 — lockbox spend (spec committed; read in flight).** See §7.

## 3. The through-line: three independent arrivals at one fact

The phase's central structural finding was reached three separate ways:

1. **F-V1-1 (from the breach side):** Feb–Apr 2020 calm-entry rows breach
   84.4% at ~3.4× severity and land in the calm/elevated strata *by
   construction* — no entry-time VIX rule can see them.
2. **F-V3-1 (from the reviewer's side):** R11's flagged region shows
   breach lift ≤ 1 at the shipped cutoff even with 2020 in-sample; the
   over-confidence mass it exists for sits in CALM entries at both scales.
3. **V2-c + V3-b (from the parameter side):** freezing the trailing
   window ~3 years barely moves pooled coverage, and F4's trailing-vol
   reaction channel fires 5× more or less with zero coverage change.

One sentence: **trailing/entry-time signals cannot buy tail honesty at
crisis onset; only damage-bounders (R10-class) and the CSP mandate act
there.** V6's H2 pre-registers this as a prediction for 2008.

## 4. What the risk numbers are, precisely

Honest pooled, optimistic exactly where the money goes. Pooled cvar_5
breach 1.28%/2.32% (24t/100t) vs the 5% bound — PASS with room. But
top_bin (prob_profit > 0.90) breaches 3.55%/6.32% (the latter exceeding
the bound point-wise, WARN), traded region 2.81%/4.92%, ratios to pooled
~2.7×/2.1× stable across scales, replicated on a third window cut by
V2-c. Breach severity: median 1.87×, mean 2.93×, 29.8% of breaches ≥ 3×
modeled ES (100t). Violations cluster at the permutation floor (I3-E
formal at both scales). prob_profit is pooled-honest (miscalibration is
bin-local and cancels).

## 5. The re-baseline queue (consolidated; nothing applied)

Inputs to the next re-baseline discussion — measured facts, not patches:

- **F-V1-1 — onset blindness.** Entry-VIX conditioning is structurally
  blind to crisis onset; calm-entry crisis rows breach 84.4% at ~3.4×
  severity (BA entered at VIX 14 realized 7.8× modeled cvar). Lever
  class: damage-bounders (R10) and forward-looking tail inputs (IV), not
  faster trailing fits.
- **F-V1-2 — winner's curse.** Top-bin tail optimism, 6.32% breach vs the
  5% bound at 100t, AZO-robust, 94/98 names; replicates at 24t and on the
  V2-c holdout cut. Selection concentrates tail optimism exactly where
  the money goes.
- **F-V1-4 — breach severity.** ~30% of breaches ≥ 3× modeled ES; the
  2026-06-15 anecdote is now a statistic. cvar_5 understates
  conditional-on-breach loss in exactly the strata that trade.
- **F-V3-1 — R11 mis-aim.** The VIX=25 cut sits below the measured
  lift>1 crossover (~27.5+); the D23 over-confidence gap is inverted at
  every prob=0.90 cell at both scales. Not a removal recommendation
  (downgrade-only; cost is bounded sizing-down) — but any R11
  recalibration starts from this.
- **F-V5-1 — the blind spots compose.** The R10 + R9 + collateral stack
  admits up to 36.5%-NAV single-cycle damage assembled at calm-VIX onset
  entries at 3.2–7.1× modeled book CVaR; sector diversification inside
  one short-vol strategy does not diversify a market-wide onset; only the
  top-bin confidence filter bounds worst-case below ruin (19.2%) in this
  history.

Documented structural constraints riding with the queue: the tracker is
**contract-blind** (multi-contract accounting is internally inconsistent;
a real multi-contract tracker is a D-series decision — V4 §7.0), and the
Almgren-Chriss impact term is **dormant in production** (V4 armed it
harness-side; B5 owns calibration; E-13 owns the data-grounded ADV).

## 6. What this phase did NOT establish (read before citing it)

- **No premium realism before ~2016.** V6 premiums are synthetic BSM from
  deep IV panels — selection/refusal/assignment evidence only; NAV and
  dollar P&L from any pre-2016 run are not evidence (plan §2.4).
- **No data-grounded capacity knee.** V4's knee is conditional on the
  stock-ADV proxy ratio by design; the real number waits on E-13.
- **The ≥ 20% assignment-wave trough clause is neither established nor
  refuted** — the measured 10.03% NAV is biased low on two stacked axes
  (intrinsic-only marking; capture-limited book). Optional follow-up: a
  full-ranking, time-value-marked replay.
- **Tier-2 amnesia scope.** V2-a's PIT proof covers the tier-1 market
  series; schedule-type files and the (disabled) FRED credit series
  remain outside it (triage pass noted, non-blocking).
- **POT/ξ/penalty calibration.** Unpowered at both scales on this
  provider (heavy_tail never fired); sweeps deferred until the surface
  activates (Theta window), not validated.
- **R11's crisis-onset value as designed (D23/i11) was not contradicted**
  — F-V3-1 measures a different contrast (flagged-vs-unflagged
  held-to-expiry); both can be true. The finding constrains
  recalibration, not the original decision record.

## 7. V6 — the lockbox spend *(PENDING — completes on report landing)*

Spec frozen at plan §9 (commit `bc1446a`); driver pinned with in-code
H-verdicts (`1473857`); the verdict math test-pinned before the spend
(`tests/test_v6_lockbox.py`). Slice: deep panels 2007-01-03 → 2009-06-30,
PIT universe incl. delisted, max_universe=100; 1998/LTCM stays locked.
Hypotheses, frozen: **H1** refusal engages in the Oct–Dec 2008 grind
(EV-positive rate ≤ 0.5× the 2007 baseline); **H2** refusal LAGS the
Lehman cliff (opens/day in the 10 tdays ending 2008-09-12 ≥ 0.7× August's
— CONFIRMS onset blindness generalizes; its falsifier is good news and
will be reported as such); **H3** the mandate carries the book through
the assignment wave (≥ 50% ITM in some Sep–Dec 2008 month; delisting
mechanism census report-only); **H4** rank-vs-realized rho, report-only
under the synthetic-premium caveat.

> **[SLOT — filled verbatim from the executor's H-verdict block +
> `v6_report.json`; plan §9.3 and the §2 ledger row complete in the same
> commit.]**

---

*Phase artifacts: harnesses under `backtests/` (tail_exceedance,
freeze_replay, param_plateau, capacity_curve, reverse_stress,
survivorship), drivers under `scripts/run_*.py`, fast-lane tests under
`tests/`, one committed fixture
(`tests/fixtures/freeze_replay/freeze_snapshot_24t.json`). All run
outputs gitignored under `data_processed/validation/`; executor-machine
reports quoted into the plan doc at closure.*
