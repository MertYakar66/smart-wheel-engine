# Validation phase — closing findings record (2026-07-12 → 2026-07-14)

**Status: FINAL — all six workstreams CLOSED.** The lockbox read was
spent 2026-07-14 (attempt 3; the first two attempts crashed in driver
post-processing before any result surfaced — plan §9.2 run record — so
the spend semantics held). Full V6 record: plan §9.3.

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
mandate's inability to be forced out (V5-b). And the lockbox spend (V6)
sealed that conclusion on a regime the tuning window never saw: in
2007–2009 the engine did NOT refuse in the grind (H1 FAIL), traded at
3.15× the August rate INTO the Lehman cliff (H2 confirmed the
onset-blindness prediction), and survived through 93.3% peak-month
assignment absorbed and held (H3 PASS) — avoidance is not where survival
comes from; the mandate is (F-V6-1).

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

**V6 — lockbox spend (CLOSED 2026-07-14).** One read, 2007-01-03 →
2009-06-30, PIT universe including 8 ranked delisted names, everything
through `rank_candidates_by_ev`, synthetic BSM premiums (selection/
refusal/assignment evidence only — NAV is not evidence). H1 **FAIL**
(refusal did not engage in the grind — EV-positive rate at the top of
the book was exactly 1.0 every month from 2007-07 through window end;
ratio 1.077 vs the ≤ 0.5 bound), H2 **CONFIRMED_BLIND** (3.15× August's
open rate into the Lehman eve), H3 **PASS** (peak monthly assignment
93.3%, Oct 2008, n=345; the delisting-loss mechanism exercised on 8
names), H4 report-only (+0.315 [0.265, 0.363] within-menu rho through
the crisis). Full verdicts + post-hoc measurement notes: plan §9.3.

## 3. The through-line: now four independent arrivals at one fact

The phase's central structural finding was reached four separate ways:

1. **F-V1-1 (from the breach side):** Feb–Apr 2020 calm-entry rows breach
   84.4% at ~3.4× severity and land in the calm/elevated strata *by
   construction* — no entry-time VIX rule can see them.
2. **F-V3-1 (from the reviewer's side):** R11's flagged region shows
   breach lift ≤ 1 at the shipped cutoff even with 2020 in-sample; the
   over-confidence mass it exists for sits in CALM entries at both scales.
3. **V2-c + V3-b (from the parameter side):** freezing the trailing
   window ~3 years barely moves pooled coverage, and F4's trailing-vol
   reaction channel fires 5× more or less with zero coverage change.
4. **F-V6-1 (from deep history, pre-registered as a prediction):** on
   2007–2009 — a regime no part of the configuration ever saw — the
   engine traded at full appetite through the grind (H1 FAIL) and at
   3.15× the August rate into the Lehman cliff (H2, whose CONFIRMED
   outcome was the pre-registered generalization of F-V1-1).

One sentence: **trailing/entry-time signals cannot buy tail honesty at
crisis onset; only damage-bounders (R10-class) and the CSP mandate act
there** — and V6 showed the mandate doing exactly that job in 2008
(H3: the assignment wave absorbed and held).

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
- **F-V6-1 — EV-sign refusal is not a crisis defense.** On 2007–2009
  (the one lockbox read; a regime the tuning window never saw) the top
  of the book stayed EV-positive through the entire Oct–Dec 2008 grind
  (H1 FAIL), entries continued into the Lehman cliff at 3.15× the
  August rate (H2), and survival was carried entirely by
  assignment-and-hold (H3: 93.3% peak monthly assignment) — the
  deep-history generalization of the F-V1-1 + F-V5-1 structure.
  Mechanism note: elevated IV inflates the credit side of EV while the
  trailing forward distributions lag the regime — the same two-sided
  failure F-V1-1 measured at COVID onset. Lever class: identical to
  F-V1-1's (damage-bounders and forward-looking tail inputs, not EV
  recalibration). Caveats bounding the claim: top-of-book censoring —
  **now RETIRED** by the V6-r1 full-menu re-read (plan §10.3: un-censored
  at `top_n=100`, the grind EV-positive rate is still 0.98, ratio 1.95;
  the FAIL is unconditional) — and premium coupling (plan §9.3 note 2,
  §2.4), which remains the sole standing bound and waits on real option
  marks (E-13 / Theta).

Documented structural constraints riding with the queue: the tracker is
**contract-blind** (multi-contract accounting is internally inconsistent;
a real multi-contract tracker is a D-series decision — V4 §7.0), and the
Almgren-Chriss impact term is **dormant in production** (V4 armed it
harness-side; B5 owns calibration; E-13 owns the data-grounded ADV).

## 6. What this phase did NOT establish (read before citing it)

- **No premium realism before ~2016.** V6 premiums are synthetic BSM from
  deep IV panels — selection/refusal/assignment evidence only; NAV and
  dollar P&L from any pre-2016 run are not evidence (plan §2.4).
- **H1's refusal metric was top-of-book-censored — now RESOLVED.** The
  V6 rank log held exactly the top 15 of ~100 candidates per date, so
  the EV-positive rate saturated at 1.0 whenever ≥ 15 names cleared zero.
  The V6-r1 full-menu re-read (`top_n=100`, plan §10.3) un-censors it:
  the grind EV-positive rate is still **0.98** (ratio grind/baseline
  1.95, verdict `CAVEAT_RETIRED`), and 0 of the 650 grind-and-surround
  dates thinned below the opens appetite — so the FAIL is **unconditional
  at every menu depth**, not an artifact of top-15 logging. What remains
  is the premium bound: because EV sign is premium-dependent, the §2.4
  synthetic-premium discount still bites H1 harder than H2/H3, and that
  is the only standing caveat on F-V6-1.
- **No data-grounded capacity knee.** V4's knee is conditional on the
  stock-ADV proxy ratio by design; the real number waits on E-13.
- **The ≥ 20% assignment-wave trough clause is `RETIRED_PRACTICAL`** (was
  "neither established nor refuted"). The V5-b-full follow-up (plan
  §10.3) ran the promised full-ranking, time-value-marked replay: COVID
  troughs at 10.03% NAV even on the doubled-IV (tv_x2) leg, so per the
  frozen rule x2.0 < 20% → RETIRED. Both suspected low-biases are
  disposed of, not merely bounded: the TV mark equals intrinsic when the
  book is deep-ITM at the trough (so marking was not the limiter), and
  the full-menu books are byte-identical to the capture-limited ones (so
  capture was not the limiter). The real reason the trough stays small is
  the engine's own `ev_dollars > 0` gate, which fills only ~32% of NAV at
  the COVID eve — the pre-registered ≥80%-saturation expectation was
  falsified on all four windows, in the engine's favor.
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

## 7. V6 — the lockbox spend (CLOSED 2026-07-14)

Spec frozen at plan §9 (commit `bc1446a`); driver pinned with in-code
H-verdicts (`1473857`); the verdict math test-pinned before the spend
(`tests/test_v6_lockbox.py`). Slice: deep panels 2007-01-03 → 2009-06-30,
PIT universe incl. delisted, max_universe=100; 1998/LTCM stays locked —
and remained unread through all three attempts. The read was spent on
attempt 3 (2026-07-14, driver transport fixed between attempts with SPEC
and H-functions byte-identical — plan §9.2 run record): 9,750 ranked
rows, 650 dates, 245 opens, 73.6 min.

**Verdicts (frozen rules, reported as computed):**

- **H1 — refusal engages in the grind: FAIL.** Grind/baseline
  EV-positive ratio 1.077 vs the ≤ 0.5 bound; the rate is exactly 1.0
  every month from 2007-07 through 2009-06 (only early 2007 dips). The
  refusal mechanism did not generalize by the pre-registered metric —
  the engine's tradeable menu never thinned below its own 3-opens/day
  appetite at any point in Oct–Dec 2008.
- **H2 — refusal lags the cliff: CONFIRMED_BLIND** (the pre-registered
  F-V1-1 generalization). Opens/day ran at **3.15×** the August-2008
  rate over the 10 trading days into the Lehman eve (threshold ≥ 0.7;
  small counts — 2 vs 3 opens on a BP-saturated book — direction
  unambiguous, magnitude noisy).
- **H3 — the mandate carries the book: PASS.** Peak monthly ITM 93.3%
  (Oct 2008, n=345; Sep 34.9% → Oct 93.3% → Nov 52.0% → Dec 19.4%) —
  the COVID assignment wave (100% of 27 puts) generalizes at 13× the
  sample size. H3b: 8 delisted PIT-only names were ranked (last bars
  2007-03 → 2008-12); the delisting-loss mechanism exercised, not
  vacuous.
- **H4 — selection sanity (report-only):** per-date within-menu rho
  **+0.3154**, block-7 CI [+0.2645, +0.3627], 83.5% of 650 dates
  positive — the ordering signal holds through the deepest regime in
  the readable data, under the synthetic-premium caveat.

**What V6 changes:** H1's FAIL retires any residual notion that the EV
gate is a crisis detector — refusal-by-EV-sign did not happen in the
grind, full stop (with the §9.3 censoring/premium caveats bounding how
far that claim extends). H2+H3 land exactly where the modern-era record
pointed: blind into the cliff, survived by the mandate. Together:
**F-V6-1** (§5). The engine's honest crisis story after this phase is a
composition of R10-class damage bounds, the top-bin confidence bound
(V5-a's 19.2% ceiling), and the CSP mandate's absorb-and-hold — with
selection ordering (H4) as the through-crisis alpha claim, and nothing
in the stack pretending to see onset coming.

---

*Phase artifacts: harnesses under `backtests/` (tail_exceedance,
freeze_replay, param_plateau, capacity_curve, reverse_stress,
survivorship), drivers under `scripts/run_*.py`, fast-lane tests under
`tests/`, one committed fixture
(`tests/fixtures/freeze_replay/freeze_snapshot_24t.json`). All run
outputs gitignored under `data_processed/validation/`; executor-machine
reports quoted into the plan doc at closure.*
