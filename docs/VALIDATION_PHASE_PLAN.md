# Validation phase — formal reliability audit of engine outputs

**Status:** ACTIVE (opened 2026-07-12). **Owner:** the operator; executed as a
brain/terminal split — one session designs harnesses and verifies results
(the brain), the operator's terminal runs the expensive passes (the
executor). **Scope discipline:** every workstream here is measurement-only
(CLAUDE.md section 2) — harnesses call the ranker read-only, mutate no
production default, and feed nothing back into EV, verdicts, or gates. The
decision-layer trio is untouched throughout.

This plan is the coordinating doc for the phase; per-workstream findings get
their own dated docs + worklog fragments as they land. It operationalizes the
gaps identified against desk practice in the 2026-07-12 validation
discussion: the repo is already strong on falsification culture
(pre-registration, leave-one-crisis-out, adversarial verification,
fingerprinting) and specifically weak on (a) multiplicity-deflated
statistics, (b) parameter freeze-replay, (c) risk-model exceedance testing,
and (d) execution/capacity realism beyond one contract at mid.

---

## 1. Workstreams (priority order)

| # | Workstream | Question it answers | Status | Artifacts |
|---|---|---|---|---|
| V1 | **Tail-risk exceedance validation** — Kupiec POF + date-clustered CIs + violation-clustering tests on the engine's own `pnl_p25/p50/p75`, plus the ES-bound breach test + severity on `cvar_5` | Are the engine's per-candidate risk numbers statistically honest, per regime? | **CLOSED 2026-07-12** — findings: `docs/VALIDATION_V1_TAIL_EXCEEDANCE_FINDINGS_2026-07-12.md` | `backtests/tail_exceedance.py`, `scripts/run_tail_exceedance.py`, `tests/test_tail_exceedance.py` |
| V2 | **Parameter freeze-replay (C1)** — snapshot every tuned artifact as-of a cutoff, replay forward touching nothing | Does any reported edge survive with parameters the past could actually have had? | **CLOSED 2026-07-12** — all four runs done; results §5.6-§5.7 | `backtests/freeze_replay.py`, `scripts/run_freeze_replay.py`, `tests/test_freeze_replay.py`, `tests/fixtures/freeze_replay/` |
| V3 | **Parameter-plateau sweep** — perturb every static constant in the `docs/PARAMETER_OOS.md` Phase-0 inventory +/-20-50%; require plateaus, not peaks | Is the configuration a fitted artifact? | **designed + pre-registered (§6)** | `backtests/param_plateau.py` (planned) |
| V4 | **Capacity curve** — re-run S34-class backtests at 5/10/25 contracts with the Almgren-Chriss impact term armed and OI-capped fills | Where is the knee of edge-vs-deployed-dollars? | queued | (extends `backtests/regression/_common.py` friction overlay) |
| V5 | **Reverse stress** — cheapest-path-to-ruin search, starting from the known blind spots (calm-VIX single-name gap on a top-bin name; margin procyclicality) | What breaks the book that no gate catches? | queued | (new) |
| V6 | **Lockbox spend** — one pre-registered deep-history run (1998/2008, delisted names included) of the refusal mechanism | Does crisis refusal generalize to regimes the tuning window never saw? | gated on V1-V3 + protocol below | `SWE_DEEP_HISTORY` panels |

## 2. The lockbox protocol (reserved-data discipline)

The deep-history panels (`data/bloomberg/deep/`, 1994 floor + ~1,015
delisted names) and the unwired broad-pull datasets are **reserved
validation data**, managed as a budget:

1. **No casual reads.** No experiment touches deep history without a
   pre-registered spec (hypothesis, metric, pass/fail criterion, and the
   exact command) committed *before* the run — the SIM-200K pattern.
2. **Reads are counted.** Each deep-history experiment gets a row in the
   ledger below. After a read, that slice is validation data, not lockbox
   data; its p-values are discounted accordingly in any write-up.
3. **One headline spend.** The final deployment-grade claim about crisis
   refusal gets exactly one lockbox run (V6), reported whatever it says.
4. **Known limit, stated up front:** there are no option prices before
   ~2016 (Theta larder floor), so pre-2016 runs validate *selection,
   refusal, and assignment behavior* on synthetic premiums — never premium
   realism, VRP capture, or dollar P&L. A 2008 run demonstrating the
   refusal mechanism is evidence; its NAV is not.
5. **Broad-pull wiring pays a multiplicity toll:** any new feature wired
   toward EV needs pre-registered acceptance criteria (the supervised
   Phase 1-3 discipline already in `docs/PHASE1_E_TRIO_EXECUTION_SPEC.md`),
   and the count of candidate features *tried* is recorded so the survivor's
   evidence can be deflated honestly.

**Lockbox ledger:** (append-only)

| Date | Slice read | Spec committed at | Result doc |
|---|---|---|---|
| — | — | — | — |

## 3. V1 — tail-risk exceedance: design summary

Full rationale in `backtests/tail_exceedance.py`'s module docstring. In
brief: capture the ranker's modeled tail block (`pnl_p25/p50/p75`, `cvar_5`,
plus `prob_profit`/`n_scenarios`/`distribution_source`/entry-VIX) over a
PIT date grid with `min_ev_dollars=-1e9` (calibration needs the losers),
forward-replay each row to held-to-expiry realized P&L with the locked
S27/S34 convention, then test:

- **Quantile coverage** (p25/p50/p75): Kupiec (1995) POF for the pooled
  count, a **date-clustered bootstrap CI** as the honest significance
  (rows sharing an entry date share one market path — the same
  overlap-inflation caveat `docs/PRODUCTION_READINESS.md` applies to rank
  rho), and a **violation-clustering permutation test** on the date-level
  violation-rate series (the panel-honest Christoffersen stand-in; crisis
  procyclicality I3-E predicts bursts).
- **CVaR bound + severity**: for any distribution `P(X < ES_5) < 5%`, so
  breach rate significantly above 5% is unambiguous tail understatement;
  conditional severity (`realized/cvar_5` among breaches) formalizes the
  2026-06-15 stress-test's 3-4.5x anecdote. Strata: entry-VIX band
  (calm <= 15 / elevated <= 25 / crisis > 25, the W-series convention),
  top-bin membership (`prob_profit > 0.90`), traded region
  (`ev_dollars > 0`).
- **Pooled prob_profit coverage**: sum-of-heterogeneous-Bernoulli z — one
  pooled statistic complementing the binned reliability tables the repo
  already publishes.

Verdict ladder per test: INSUFFICIENT (n < 30) / PASS / WARN (point rate
above nominal, CI straddles) / FAIL (clustered-CI lower bound above
nominal). Coverage *below* nominal (model too conservative) is reported but
never fails — for a short-vol book the too-fat-modeled-tail direction is
safe.

**Documented comparison offset:** realized P&L is gross of entry costs and
the $5 ITM assignment fee (the locked `_forward_replay_realized_pnl`
convention, kept for cross-study comparability); modeled quantiles are net.
The few-dollar offset favors the engine on lower-tail tests, so FAILs are
conservative evidence and marginal PASSes are not proof.

**Expected findings (pre-registered, falsifiable):** given the calibration
record we expect p50/p75 roughly honest, the cvar_5 breach rate to PASS the
pooled 5% bound in calm but WARN/FAIL in the crisis stratum, breach
severity well above 1x, and strong violation clustering (the I3-E
procyclicality). If instead everything PASSes flat across strata, that is
*suspicious* (the offset + survivorship of the current universe) and the
100t run must arbitrate.

## 4. Runs

| Run | Config | Grid | Where | Status |
|---|---|---|---|---|
| V1-a | `24t` (UNIVERSE_24) | 2022-01-03 -> 2026-05-21, every 5 bdays | sandbox (brain) | **DONE 2026-07-12** |
| V1-b | `100t` (UNIVERSE_100) | 2020-02-03 (COVID entry) -> 2026-05-21, every 2 bdays | terminal | **DONE 2026-07-12** |

### V1-a results (2026-07-12; 2,735 rows / 229 dates / 2,735 resolved)

Console + `report_24t.json` (gitignored run dir). Headlines:

- **Harness finding first:** the initial analyze produced byte-identical
  p50/p75 violation rates (0.245) — the short-put **win point mass**: with
  `prob_profit >= 1 - nominal` the modeled quantile sits ON max profit and
  continuous coverage is vacuous. Fixed same-day: quantile tests now
  restrict to informative rows (`prob_profit < 1 - nominal`, an entry-time
  stratum, PIT-clean) and report exclusions loudly. Post-fix: p50/p75 are
  honestly INSUFFICIENT on this book (every row has prob_profit >= 0.5 —
  25-delta puts); p25 runs on 808 informative rows.
- **p25 coverage: PASS, conservative** — violations 17.1% vs nominal 25%
  (clustered CI [0.136, 0.208]).
- **cvar_5 breach: PASS pooled** — 1.28% vs the 5% ES bound (35/2,735;
  clustered CI [0.75%, 1.86%]). Note the *model-implied* breach ceiling at
  N=35 scenarios is ~1/35 ~ 2.9%, tighter than the generic 5% bound —
  against that tighter yardstick the strata below are less comfortable.
- **The strata tell the real story:** calm 0.55% / elevated 1.64% / crisis
  1.18% — but **top_bin (prob_profit > 0.90): 3.55%** and **traded region
  (ev_dollars > 0): 2.81%** — the engine's highest-confidence and actually-
  tradeable rows breach their modeled tails at 2-3x the pooled rate, at or
  above the ~2.9% model-implied ceiling. Selection concentrates tail
  optimism exactly where the money goes (consistent with the W3/I1 top-bin
  record; a winner's-curse signature).
- **Breach severity:** median realized/cvar_5 = **1.36x**, mean excess
  **-$1,462** per breach — real losses run well past the modeled expected
  shortfall when they breach, though the median multiple is far below the
  stress-test's 3-4.5x single-name anecdotes (those live in the worst
  cases, which the report lists).
- **Violation clustering: confirmed, strongly** — date-level lag-1
  autocorr 0.40-0.67 with permutation p < 0.001 on every test. Violations
  arrive in bursts (the pre-registered I3-E procyclicality signature),
  formally established. Frequency PASSes; *independence* does not — the
  risk numbers are regime-blind in exactly the documented way.
- **prob_profit pooled: honest** (z = 0.94, p = 0.35) — miscalibration is
  bin-local, cancels pooled, as the record predicted.
- **Pre-registration scorecard:** clustering + severity + pooled-honesty
  expectations confirmed; the "crisis stratum WARN/FAIL" expectation was
  NOT borne out on this window — note the 24t grid starts 2022-01 and so
  contains no crisis *onset* (the 2022 grind's vol was already in the
  trailing distributions, and F4 widening fires there). V1-b starts
  2020-02-03 specifically to arbitrate this.

### V1-b results (2026-07-12; 40,201 rows / 822 dates / 100 names; 101.9-min build on the operator terminal, zero rank failures)

Full findings + per-stratum verdicts + triage:
**`docs/VALIDATION_V1_TAIL_EXCEEDANCE_FINDINGS_2026-07-12.md`** (the V1
closure doc). One-paragraph summary: official verdict `OVERALL: PASS`
(p25 17.75% vs 25% conservative; cvar_5 pooled 2.32% vs the 5% bound, CI
[1.65%, 3.10%]) — and the PASS decomposes into the phase's sharpest
findings so far: (F-V1-1) entry-VIX conditioning is structurally blind
to crisis onset — Feb-Apr 2020 calm-entry rows breach 84.4% at ~3.4x
severity (BA entered at VIX 14 realized 7.8x modeled cvar), landing in
the calm/elevated strata by construction, so the pre-registered "crisis
stratum WARN/FAIL" was falsified *as stratified* while its underlying
hypothesis was confirmed sharper — no entry-time VIX rule (incl. R11)
can see these rows, only damage-bounders (R10) act on them; (F-V1-2)
winner's-curse concentration replicates at 100 names with top_bin 6.32%
now exceeding the 5% bound point-wise (supplementary clustered CI
[4.15%, 8.72%] straddles -> WARN), traded_region 4.92%, ratios to pooled
matching 24t (2.7x/2.1x), AZO-robust, 94/98 names; (F-V1-3) violation
clustering at the permutation floor (ac1 0.966 cvar / 0.761 p25,
p=0.0005) — I3-E formal at both scales; (F-V1-4) severity: median 1.87x,
mean 2.93x, 29.8% of breaches >= 3x — the 2026-06-15 anecdote is now a
statistic. Benign anomalies (BK->BNY seam, BNY/CASY history gate,
AZO worst-case dollar-ranking mechanic) verified and recorded. Every
headline number independently recomputed from the captured table by the
executor session.

**Acceptance for V1 closure: MET (2026-07-12).** Both reports generated;
findings doc written with per-stratum verdicts; every WARN/finding
triaged (all engine findings; the workstream's one harness artifact —
the win point mass — was fixed during V1-a before V1-b ran). No engine
change ships from this workstream directly; F-V1-1/2/4 are queued as
re-baseline inputs and the forward-looking-tail research question.

---

## 5. V2 — parameter freeze-replay (C1): design + pre-registration

**Written 2026-07-12, before any V2 code or run exists.** The expectations in
§5.4 are the falsifiable part; they must not be edited after the first run.

### 5.0 What C1 asks, and what surface this actually tests

`docs/PRODUCTION_READINESS.md` C1: *"Out-of-sample parameter freeze.
Implement HMM / POT-GPD parameter freeze + replay infrastructure — snapshot
parameters at a cutoff date, run a held-out backtest; new test that asserts
the frozen parameters reproduce the snapshot's classifier output."*

The engine's parameter surface splits three ways, and freeze-replay only
applies to one of them:

1. **Static hand-set constants** (regime weights, F4 trigger, R11 cutoffs,
   heavy-tail penalty, POT threshold — the `docs/PARAMETER_OOS.md` §1
   inventory). These cannot be "frozen as of a past cutoff" — they did not
   exist then and the hand-tuning procedure is not reproducible. Their
   honest treatment is the train-only re-fit (done for the regime overlay —
   optimism gap +0.096) and the plateau sweep (**V3**), not V2.
2. **Online-fit artifacts** — refit per `as_of` from trailing data: the
   4-state HMM (means/stds/trans_mat/labels, `wheel_runner.py` ~L1999), the
   POT-GPD tail fit (ξ/β on scenario losses, `ev_engine.py` ~L507), the
   empirical forward-distribution cascade itself (the scenario set IS the
   fitted object), and the F4 vol-ratio. **This is V2's surface.** The
   record calls these "PIT-clean / leakage-free *by construction*"
   (`docs/PARAMETER_OOS.md` §1) and the §7.2 surviving top-15 edge rests
   on exactly that claim ("the edge lives in `ev_raw`, an online-fit,
   PIT-clean signal") — an *assertion*, never yet an *experiment*.
3. **Market-state readings** — spot, premium, IV, VIX, HY OAS at `as_of`.
   Never frozen: freeze-replay freezes fitted *knowledge*, not the market.

Three sub-workstreams:

### 5.1 V2-a — the amnesia test (full-pipeline PIT proof by data truncation)

**Claim under test:** the ranker's output at `as_of = T` depends on no
market-data row dated after `T`.

**Method.** Build a truncated copy of the data directory in which each
unambiguous market time-series CSV is physically cut to `date <= T`
(tier-1 set: `ohlcv`, `vol_iv`, `treasury`, `vix`, `liquidity`).
Everything else is copied intact (`earnings` — the event-lockout gate
legitimately needs the *future* earnings calendar; `dividends`,
`fundamentals`, `corporate_actions`, `broad_pull/` — mixed semantics,
identical in both runs so they cannot create false diffs; a stricter
tier-2 pass that truncates these too and *triages* diffs is a noted
follow-up, not part of the tier-1 assertion).

*Pre-run mechanics corrections (2026-07-12, before any run):*
(i) `credit_risk` was listed tier-1 at pre-registration but turned out to
be a **dateless snapshot file** (no date column exists to truncate) — it
is copied intact and recorded as an inherent PIT limitation of that
dataset, not a truncation target. (ii) The credit-regime de-rank reads
FRED **network** series (`fred_adapter.credit_regime`), which file
truncation cannot cover and whose availability could differ between runs;
the amnesia harness therefore passes `use_credit_regime=False` on BOTH
sides — its PIT slice is code-enforced and unit-tested separately — and
records the exclusion. Then:

1. **A/A determinism control** — two fresh `WheelRunner(data_dir=<full>)`
   instances, same `as_of`, full diagnostic fields, option-premium rail
   pinned off: outputs must be identical. (Without this, A/B diffs are
   uninterpretable.)
2. **A/B** — `WheelRunner(data_dir=<full>)` vs
   `WheelRunner(data_dir=<truncated@T>)` at `as_of = T`: the full ranked
   frame (every diagnostic column, exact float equality) and the drops
   list (gate + reason per ticker) must be identical.

Grid: ~5 regime-spanning `as_of` dates (calm / elevated / crisis-adjacent /
near-frontier) × `UNIVERSE_24`. Runtime is minutes; runs in-sandbox.

**Why this test is load-bearing:** it covers every leak vector at once —
including the ones code review rates "clean": the PIT slice in
`wheel_runner` sits inside a swallowed-exception block (`except: pass`),
the HMM fits on the sliced frame rather than taking `as_of` itself, and
any connector-level cache or fallback that quietly serves unsliced data
would be invisible to per-function unit tests but cannot survive a
physical truncation diff.

### 5.2 V2-b — the C1 freeze snapshot + reproducibility lock

**Claim under test:** the fitted classifier state at a cutoff is a stable,
reproducible artifact — not an accident of environment, seed, or silently
restated data.

At cutoff **T0 = 2023-06-30** (the parameter_oos canonical split date, for
coherence), per `UNIVERSE_24` ticker, snapshot to a committed fixture
(`tests/fixtures/freeze_replay/`):

- **HMM**: `start_prob`, `trans_mat`, `means`, `stds`, `state_labels`,
  `converged`, the posterior at T0, and the resulting
  `position_multiplier` — fit with the production recipe
  (`GaussianHMM(n_states=4, n_iter=20, random_state=42)` on the last 504
  log-returns `<= T0`).
- **POT-GPD**: `fit_gpd_tail` on the *return-space* losses of
  `best_available_forward_distribution(<=T0)` — ξ, β, threshold,
  `n_exceedances`, `converged` — plus the source array's method, length,
  and endpoint fingerprint. (The engine's own GPD fits on trade-space
  scenario *P&L*, which is strike/premium-dependent; the return-space fit
  is the canonical per-ticker lock, and the trade-space fit inherits its
  determinism from the frozen scenario array. Stated so nobody mistakes
  the fixture for the engine's exact in-situ numbers.)

A test refits both from the committed CSVs and asserts reproduction within
tight tolerance. This is C1's literal definition of done, and it converts
"seed=42 so it's deterministic" into a regression lock that fires on data
restatement, numpy/scipy behavior drift, or a refactor that changes the
classifier.

### 5.3 V2-c — the freeze-replay study (held-out backtest with frozen knowledge)

**Question:** how much of the engine's rank quality and risk honesty
depends on the *recency* of the online fits? Freeze all fitted knowledge
at T0 and replay the leakage-certified holdout.

**Mechanics (measurement-only, one lever).** The single engine-level
freeze lever is `best_available_forward_distribution`: a harness-scoped
monkeypatch (the `docs/PARAMETER_OOS.md` §6 invariant-3-sanctioned
pattern, applied inside a context manager) substitutes `as_of = T0` while
the ranker runs at `as_of = T`. Because the frame passed in is already
sliced `<= T`, the internal `<= T0` slice yields exactly the scenario set
the past could have had. Everything downstream inherits the freeze with
no further surgery: `ev_raw`, `prob_profit`, `pnl_p25/p50/p75`, `cvar_5`,
the trade-space GPD fit (`cvar_99_evt`, ξ, heavy-tail penalty).
*Pre-run mechanics correction (2026-07-12, before any run):* the F4
vol-ratio widening (`realized_vol_widening_factor` /
`realized_vol_widened_log_returns`) is a separate call and stays **live
at `T`** — rv30/rv252 is a market-state reading (§5.0 category 3), like
spot and IV, not fitted knowledge; the realistic stale-operation scenario
is "current market readings, stale fitted models." The pre-registered
sentence above originally lumped F4 into the freeze; corrected before any
run. The HMM freeze is applied **offline** (the parameter_oos
rederivation convention): fit on returns `<= T0`, posterior on the tail
`<= T`, `frozen_ev_dollars = frozen_ev_raw × clamp(frozen_mult)` —
recombined in numpy on the captured columns. Spot, premium, IV, strike
selection stay at `T` (market state, §5.0). Credit-regime de-rank stays
at `T` for the same reason.

**Grid:** the parameter_oos-canonical holdout (`as_of >= 2023-08-20`,
every 5 bdays, `UNIVERSE_24`, 35-DTE/25-Δ), T0 = 2023-06-30, leakage
certificate conventions carried over. The frozen pass emits the same
`TAIL_TABLE_COLUMNS` schema as V1, so `tail_exceedance.full_report` runs
unchanged on the frozen table.

**Comparison (frozen vs production, identical dates, rows joined on
(date, ticker), asymmetric drops reported):** per-date cross-sectional
rank ρ (all candidates + top-tier, the §7 parameter_oos convention);
p25 coverage; cvar_5 breach rate + strata + severity; violation
clustering; pooled prob_profit z; `distribution_source` tier mix; and the
drift of each gap as a function of time-since-cutoff (bucketed by months
since T0).

**Scale:** 24t in-sandbox (~10-20 min build). A 100t frozen replay is
terminal-scale; go/no-go decided at the V1-b debrief.

### 5.4 Pre-registered expectations (falsifiable; do not edit after first run)

1. **V2-a: zero A/B diffs.** Falsifier: *any* row/column/drop difference.
   A diff is market-data leakage into the rank path — the
   highest-severity possible finding of this phase; every reported edge
   (S27/S34/S35, parameter_oos §7.2) is quarantined pending triage.
2. **V2-b: exact reproduction** (allclose, rtol ≤ 1e-7) of the HMM and
   GPD snapshot from committed data. Falsifier: drift — an
   environment/data-restatement finding (not an engine finding), but it
   invalidates "seed-pinned ⇒ stable" until explained.
3. **V2-c-i: rank ordering survives the freeze.** Holdout per-date ρ
   (frozen) ≈ ρ (production), difference within date-clustered CI noise —
   because the surviving edge lives in `ev_raw`'s cross-sectional
   ordering, and a stale scenario set shifts *levels* more than
   *orderings*. Falsifier: frozen ρ collapses toward 0 → the reported
   edge depends on refit recency; material operational finding (a week
   of stale data would degrade live ranking).
4. **V2-c-ii: risk coverage degrades under the freeze, asymmetrically.**
   Frozen cvar_5 breach rate > production breach rate, concentrated in
   elevated/crisis entry-VIX strata, and widening with time-since-cutoff.
   This is the I3-E mechanism made explicit — the trailing window can no
   longer catch up at all. Falsifier: frozen ≈ production on risk
   coverage → continuous refitting adds almost no tail protection, which
   *sharpens* I3-E (the lag dominates even a two-year freeze) and makes
   the case that tail honesty needs a forward-looking input (IV), not a
   faster rear-view mirror.
5. **V2-c-iii: frozen prob_profit turns optimistic** (pooled z
   significantly > 0) in windows whose realized vol regime sits above the
   T0 regime, and pessimistic below it.
6. **V2-c-iv: `distribution_source` mix shifts down-tier** in the frozen
   pass (less usable history at T0 for the same ticker).

**Interpretation guard, stated up front:** V2-c degradation under the
freeze is NOT an engine defect — production refits every `as_of`, so the
frozen configuration is not the shipped one. What V2-c bounds is (a) the
*value added by refit recency* (the honest content of "our fits are
online"), and (b) the risk of operating on stale data. Only V2-a and
V2-b failures indict the engine as shipped.

### 5.5 Runs

| Run | Sub-workstream | Where | Status |
|---|---|---|---|
| V2-a | amnesia (5 dates × 24 names, tier-1) | sandbox (brain) | **DONE 2026-07-12 — PASS** |
| V2-b | freeze snapshot @ 2023-06-30 + lock | sandbox (brain) | **DONE 2026-07-12 — REPRODUCED** |
| V2-c | frozen replay, 24t holdout grid | sandbox (brain) | **DONE 2026-07-12** |
| V2-c-100t | frozen replay, 100t (`--config 100t`; production side = the V1-b `tail_table_100t.csv`; block-13 CIs) | terminal | **DONE 2026-07-12** (after one harness fix — the BIIB halt-day NaN guard, `20a9f3f`) |

### 5.6 Results

**V2-a (2026-07-12): PASS — zero diffs on every date.** The A/A
determinism control was clean at all five dates (the ranker is exactly
reproducible given a data directory), and the A/B full-vs-truncated
outputs were identical on every diagnostic column and every drop:

| as_of | regime flavor | ranked rows | drops | A/A | A/B |
|---|---|---|---|---|---|
| 2022-06-15 | crisis-adjacent | 15 | 9 | identical | identical |
| 2023-11-15 | calm | 20 | 4 | identical | identical |
| 2024-08-06 | post vol-spike | 16 | 8 | identical | identical |
| 2025-04-15 | tariff-crisis window | 3 | 21 | identical | identical |
| 2026-05-01 | near-frontier | 22 | 2 | identical | identical |

Pre-registered expectation 1 confirmed: **no market-data row dated after
`as_of` influences the rank path** through any tier-1 series (OHLCV,
vol/IV, treasury, VIX term structure, liquidity). This converts
`docs/PARAMETER_OOS.md` §1's "leakage-free by construction" from an
assertion into an experiment for those paths. Scope limits stand as
documented in §5.1: schedule-type files and the FRED credit series are
outside the tier-1 assertion (the latter disabled on both sides), and a
tier-2 triage pass remains the noted follow-up. (Incidental: 2025-04-15
ranks only 3 names — 21 dropped, dominated by the April earnings-season
event lockout; correct behavior, noted so nobody mistakes it for a data
problem.)

**V2-b (2026-07-12): REPRODUCED — 24/24 tickers.** The committed fixture
(`tests/fixtures/freeze_replay/freeze_snapshot_24t.json`, 56 KB) holds
every UNIVERSE_24 ticker's fitted HMM (production recipe) and canonical
return-space GPD fit at 2023-06-30; a full refit from the committed CSVs
matches at rtol 1e-7. Pre-registered expectation 2 confirmed. The lock now
runs in the slow lane (`test_c1_lock_refit_reproduces_committed_snapshot`,
`backtest_regression` marker) and will fire on silent data restatement,
numpy/scipy behavior drift, or a classifier refactor. C1's
"snapshot + assert reproduction" definition of done is met; the
"held-out backtest" half is V2-c.

**V2-c (2026-07-12): the freeze barely hurts — and where it does hurt is
diagnostic.** 1,793 frozen rows joined the production (V1-a) capture
row-for-row over 144 holdout dates (2023-08 -> 2026-05, every 5 bdays;
scenario knowledge frozen at 2023-06-30, i.e. up to ~2.9 years stale).
Report: gitignored `freeze_replay_report_24t.json`. Headlines:

- **Rank: the tradeable-tier edge fully survives a multi-year freeze**
  (expectation 3 CONFIRMED). Per-date cross-sectional rho, top-5 tier:
  production +0.174 [block-7 CI +0.054, +0.283] vs frozen +0.188-0.207
  across the three frozen variants — statistically indistinguishable,
  nominally *higher* frozen. The frozen-HMM variant matches the live-HMM
  variant to the third decimal (the overlay is as inert as parameter_oos
  found). Refit recency is NOT where the top-tier edge comes from —
  consistent with the "edge lives in the parameter-light `ev_raw`" story,
  and operationally: a stale-fit week would not degrade live top-tier
  ranking. New wrinkle: the ALL-candidate cross-section goes from
  statistically zero (production -0.058 [-0.165, +0.039]) to
  significantly negative frozen (-0.142 to -0.150, CI excluding 0) — the
  freeze corrupts the ordering of the *untradeable tail*, not the head.
- **Risk frequency: pooled coverage is freeze-insensitive** (expectation
  4's pre-registered form largely FALSIFIED, and that is the finding).
  cvar_5 breach 1.51% -> 1.78% pooled (both PASS); p25 violations 14.7%
  -> 11.9% (frozen *more* conservative); drift by time-since-cutoff is
  minimal (12-24m: 2.76% -> 2.92%; 24m+: 1.61% -> 2.25%). A ~3-year-stale
  scenario set produces nearly the same pooled tail-breach frequency as a
  daily-refit one. Per the pre-registered interpretation: the trailing
  window's lag dominates — refitting daily buys almost no *pooled*
  frequency protection, sharpening I3-E and the case that tail honesty
  needs a forward-looking input (IV), not a faster rear-view mirror.
- **Where refit recency DOES protect: violation independence.**
  Production violation clustering on this (crisis-onset-free) holdout:
  ac1 0.10, permutation p = 0.079 — not significant. Frozen: **ac1 0.56,
  p = 0.0005** — strongly clustered. The live refit absorbs slow regime
  drift well enough to de-cluster violations on a calm-to-elevated
  window; the frozen model's violations arrive in bursts. (Reconciles
  with V1-a full-window clustering ac1 0.40-0.67: refitting de-clusters
  slow drift but cannot de-cluster crisis *onsets* — the two results
  bracket exactly what the rear-view mirror can and cannot do.)
- **The freeze redistributes confidence INTO the over-confident bin**
  (expectation 5 falsified in its pooled form, confirmed in a sharper
  one). Median prob_profit shift is 0.0000 and pooled z is identical
  (+4.88 vs +4.84 — both *conservative* on this window: 1,488 wins
  observed vs ~1,404 expected), but 55.6% of joined rows move by > 5pp
  symmetrically, and the top bin (prob_profit > 0.90) more than doubles
  in population: 79 production rows -> 178 frozen, with breach rate
  7.6% -> 8.4% — stale calm-vintage distributions mint more
  high-confidence picks in exactly the bin the record shows is
  over-confident. Traded-region breach rises 3.89% -> 5.42%. Selection
  concentration, not mean drift, is how staleness expresses itself.
- **Distribution-source mix** (expectation 6): confirmed, small — 0.7% ->
  4.2% of cells fall to the overlapping tier under the freeze.
- Corroboration for V1: on this holdout subwindow the PRODUCTION capture
  itself shows top_bin breach 7.6% (n=79) and traded_region 3.9% (n=540)
  vs 1.5% pooled — the V1-a winner's-curse concentration replicates on a
  different window cut, before V1-b even reports.

**Scorecard against §5.4 (unedited):** (1) CONFIRMED, (2) CONFIRMED,
(3) CONFIRMED, (4) largely falsified in its pooled-frequency form — the
degradation channel is violation clustering + top-bin population, not
pooled breach rate; the pre-registered falsifier interpretation applies,
(5) falsified as a mean shift; the effect is symmetric churn plus top-bin
concentration, (6) confirmed (small). No V2 finding indicts the shipped
engine (V2-a and V2-b both clean); the V2-c findings characterize the
*value of refit recency*: it buys violation independence and selection
discipline, not pooled tail-frequency accuracy.

### 5.7 V2-c-100t results (2026-07-12; terminal) + V2 closure

18,837 frozen rows joined the V1-b production capture row-for-row (zero
asymmetric drops) over 359 holdout dates (2023-08-22 -> 2026-05-20,
every 2 bdays; scenario knowledge frozen at 2023-06-30, up to ~2.9y
stale; block-13 moving-block CIs for the every-2-bday overlap). One
harness artifact en route: the BIIB halt-day NaN close tripped the
engine's #386 non-finite guard at the harness's one unguarded fit call —
fixed + regression-pinned (`20a9f3f`); BIIB carries the neutral
multiplier in the frozen-HMM signal, symmetric with the engine's own
degrade on those dates. Headlines:

- **THE decisive answer: the S34-class top-tier edge survives the
  freeze at the real menu scale.** top15 per-date rho: production
  +0.294 [0.205, 0.379] vs frozen +0.366 [0.290, 0.442] (top5 +0.394 ->
  +0.437) — heavily overlapping CIs; the correct statement is *stable
  under the freeze*, not "improved". Both consistent with the committed
  §7.2 holdout top-15 edge (+0.371 [0.25, 0.49]). Refit recency is NOT
  where the tradeable-tier edge comes from — now established with ~48
  candidates/day and up to ~2.9 years of scenario staleness. The
  frozen-HMM recombination (+0.348) sits on the live-HMM variant
  (+0.339): the overlay is inert at 100t too.
- **The 24t all-candidate inversion does NOT replicate** — frozen "all"
  is +0.034 [-0.050, +0.122], statistically zero like production's. The
  24t tail-ordering inversion is retro-flagged as likely small-menu
  noise, not a real freeze effect.
- **Top-bin inflation replicates directionally:** the freeze mints 755
  -> 1,205 top-bin rows (1.60x; 24t: 2.25x) breaching at 7.15% -> 7.80%
  (both already above the 5% bound on this holdout window — F-V1-2
  corroborated on a third window cut). Traded-region breach rises 3.42%
  -> 5.20%, crossing the bound (24t: 3.89% -> 5.42%, near-identical).
  Pooled 1.55% -> 2.02%, both PASS. Severity mean multiple 1.82x ->
  3.43x (deep tail dragged; medians 1.38x -> 1.46x). Staleness drift
  widens with time-since-cutoff, largest at 24m+ (2.27% -> 3.12%).
- **The 24t clustering contrast does not reproduce — with a cadence
  caveat.** Production is already at the permutation floor on this
  dense grid (ac1 0.749) and frozen is marginally higher (0.776). The
  every-2-bday cadence mechanically raises date-level autocorrelation
  vs 24t's every-5-bday grid (adjacent dates share most of their option
  life), so cross-run ac1 LEVELS are not comparable; the within-run
  statement stands — freezing adds little clustering where production
  violations are already bursty.
- p25 frozen slightly MORE conservative (17.3% -> 15.8%), replicating
  24t; prob_profit pooled strongly conservative on this calm-heavy
  holdout in BOTH variants (z ~ +10.7 / +10.4; un-clustered-z caveat);
  54.3% of rows move > 5pp in prob_profit with medians exactly 0.0 and
  per-date ev_raw ordering agreement rho 0.673 — symmetric churn under
  a preserved head, same as 24t. distribution_source down-tier 0.73% ->
  4.94% (expectation 6, same magnitude as 24t).

**V2 closure.** All four runs done (V2-a PASS, V2-b REPRODUCED, V2-c
24t + 100t). C1 is closed in full: freeze + replay infrastructure
shipped with a committed reproduction lock, and the held-out frozen
backtest run at both scales. The workstream's verdict, stated once:
**the engine's reported top-tier edge does not depend on refit recency
or on any parameter the past could not have had** (the online fits are
leakage-clean by experiment, V2-a; the fitted state is reproducible,
V2-b; the frozen replay preserves the tradeable-tier edge, V2-c) —
while refit recency's real value is selection discipline (top-bin
population control) and, on sparse calm grids, violation independence.
What freezing costs shows up exactly where V1 said the engine is
weakest: the top-bin and traded-region tail strata.

---

## 6. V3 — parameter-plateau sweep: design + pre-registration

**Written 2026-07-12, before any V3 code or run exists.** The §6.4
expectations and plateau criteria are the falsifiable part; do not edit
after the first run. Measurement-only: every swept value enters via the
§6-sanctioned harness monkeypatch pattern (`docs/PARAMETER_OOS.md` §6,
context-manager-scoped) and no production default changes.

### 6.0 Disposition of the Phase-0 inventory

The sweep question ("plateau or fitted peak?") only applies to constants
that (a) were hand-set with full-history visibility AND (b) actually act
on the ranked path on the Bloomberg provider. Verified against source:

| Constant (shipped) | Disposition |
|---|---|
| Regime weights `{0.2, 0.5, 1.0, 1.25}` | **already out-of-parameter tested** — parameter_oos §5: refit collapses OOS, overlay adds no OOS value; V2-c: overlay inert under freeze. Not re-swept. |
| Regime clamp `[0.0, 1.25]`, dealer clamp `[0.70, 1.05]` | **safety rails, not fitted optima** (and the dealer surface is inert on Bloomberg — PARAMETER_OOS §6). Not swept. |
| HMM recipe (4 states, 20 iters, seed 42) | **online-fit machinery** — V2-b locks its reproducibility; state-count sensitivity deferred (a different study, not a plateau sweep). |
| **F4 widening: threshold 1.30, slope 0.20, cap 1.15** | **SWEPT (engine passes)** — the marquee in-sample-tuned constant ("chosen to not invert S27 ρ"). Fires on ~14% of dates; acts on every scenario-derived output. |
| **R11 cutoffs: VIX 25.0, top-bin prob 0.90** | **SWEPT (offline)** — a pure function of captured columns (`vix_entry`, `prob_profit`); realized outcomes already in the V1 tables at both scales. Complements the committed D23/i11 leave-one-crisis-out. |
| POT-GPD threshold 95.0 / min-exceedances 15 / heavy-tail ξ-gate 0.3 / penalty 0.5 | **ACTIVATION-GATED** — the GPD only runs at >= 200 scenarios, but ~99% of Bloomberg rows ride the N~35 non-overlapping tier. §6.2 first *measures* activation (the ranker emits `tail_xi`/`heavy_tail`); if < 2% of ranked rows carry a converged GPD fit, the POT/ξ/penalty sweeps are recorded **NOT POWERED on this provider** (the PARAMETER_OOS §6 dealer-clamp treatment) and deferred to a Theta-window follow-up. The ξ-gate/penalty halves are offline-recomputable from a capture carrying `tail_xi` when activation permits. |
| Block-bootstrap block 5 / MC block 21 / n_sims 10000 | Bootstrap tier: same activation gate (tier fired on 0% of V1-a rows). MC constants: reporting layer, off the EV path. Not swept unless activation shows otherwise. |
| (Scoping remark) profit-target 0.50 / stop 2.0 / slippage 0.20 | Not in the Phase-0 inventory; they belong to the D19 exit-cost re-baseline discussion, recorded here so the omission is explicit, not accidental. |

### 6.1 V3-a — R11 cutoff sweep (offline, both scales)

Grid: `vix_threshold` x `top_bin_prob` over {20.0, 22.5, 25.0*, 27.5,
30.0} x {0.85, 0.90*, 0.95} (* = shipped). For each cell, on a V1 tail
table (24t sandbox; 100t on the terminal, seconds): flagged-set size,
flagged cvar_5 breach rate + mean realized P&L vs the unflagged
remainder of the top bin, lift, and date-clustered CIs. R11's value
claim is that the flagged region is materially WORSE than what it
leaves behind — the sweep asks whether that lift sits on a shelf around
(25.0, 0.90) or on a spike.

### 6.2 V3-b — activation diagnostic + F4 engine sweeps (sandbox)

**Activation diagnostic first** (one cheap unpatched pass, ~20 dates x
24 names, reading `tail_xi` / `heavy_tail` / `n_scenarios` /
`distribution_source` / `cvar_99_evt` off the ranked frame): records the
GPD/bootstrap activation rates that decide the §6.0 gates.

**F4 sweeps** — one axis at a time, engine pass per value (grid: 24t
universe, 2020-02-03 -> frontier-capped, every 10 bdays — breach-rich,
onset-inclusive, ~13 min/pass; V1's TAIL_TABLE schema so V1 statistics
run unchanged; `tail_widening_factor` in the table gives the fire rate
directly):

- axis `threshold`: {1.10, 1.20, **1.30**, 1.40, 1.50} (slope/cap pinned)
- axis `cap`: {1.00 = F4 OFF (the control), 1.075, **1.15**, 1.225, 1.30}
  (threshold/slope pinned)

Primary metric per value: cvar_5 breach rate in the elevated+crisis
entry-VIX strata (the strata F4 exists to protect), pooled breach as
secondary, p25 coverage as the conservatism check, fire rate for
mechanism visibility. **Guard metric:** top-5/top-15 per-date rho with
block-clustered CIs — the original calibration constraint was "do not
invert rank quality"; a swept value that improves tails by destroying
rank is not an improvement.

### 6.3 Plateau criteria (pre-registered, per axis)

Let SE be the date-clustered bootstrap SE of the primary metric.
Comparing the shipped value against its +/-1-step neighbors:

- **PLATEAU** — both neighbors' primary metric within 2 SE of shipped,
  and no swept value beats shipped by > 2 SE on the primary metric
  while keeping the guard metric inside its shipped CI.
- **CLIFF** — exactly one neighbor degrades by > 2 SE (report which
  side; a cliff at the edge of a shelf is a documented sensitivity, not
  automatically a fitted artifact).
- **PEAK / FITTED-ARTIFACT** (the falsifier) — shipped is a strict
  local optimum: BOTH neighbors degrade the primary metric by > 2 SE.
  This is the signature that the constant was tuned to the sample.

### 6.4 Pre-registered expectations (falsifiable)

1. **F4 threshold and cap sit on plateaus, not peaks.** The published
   calibration rationale is "gentle by design", not "optimal" — if
   either axis shows the PEAK signature, the F4 calibration is a fitted
   artifact and goes to the re-baseline queue.
2. **The F4-OFF control (cap 1.00) degrades tail coverage measurably**
   in the elevated/crisis strata (F4 earns its keep) — falsifier: OFF
   is indistinguishable from shipped, in which case F4 is ornamental on
   this grid and that is recorded (consistent with the V2-c finding
   that trailing-window recency buys little pooled frequency).
3. **R11's lift surface is a shelf**: the flagged region is worse than
   the unflagged top bin across a broad neighborhood of (25.0, 0.90),
   not just at the shipped point. Falsifier: lift concentrates at the
   shipped cell only. (Context, already known and not contradicted: V1
   F-V1-1 showed onset rows enter at CALM VIX — R11's lift can only
   exist for elevated-entry rows; the sweep measures what R11 CAN see,
   it cannot fix what it structurally cannot.)
4. **GPD/bootstrap activation on Bloomberg is < 2%** of ranked rows —
   in which case POT threshold / ξ-gate / penalty / block-length are
   recorded NOT POWERED here (an honest scope statement, not a pass).

### 6.5 Runs

| Run | What | Where | Status |
|---|---|---|---|
| V3-act | activation diagnostic | sandbox (brain) | pre-registered |
| V3-a-24t | R11 sweep on `tail_table_24t.csv` | sandbox (brain) | pre-registered |
| V3-a-100t | R11 sweep on `tail_table_100t.csv` | terminal (seconds) | pre-registered |
| V3-b-thr | F4 threshold axis (4 passes + shared baseline) | sandbox (brain), ~1 h | pre-registered |
| V3-b-cap | F4 cap axis (4 passes) | sandbox (brain), ~1 h | pre-registered |

Acceptance for V3 closure: disposition table honored; sweep tables +
per-axis plateau verdicts recorded; activation rates recorded; any PEAK
verdict triaged to the re-baseline queue. No engine change ships from
this workstream.

### 6.6 Results

**Activation diagnostic (2026-07-12; 8 spot dates x 24 names, 87 rows):**
`gpd_fit_rate` 2.30% (exactly the 2 overlapping-tier rows),
`heavy_tail` fires **0.00%**, `n_scenarios >= 200` 0.00%. Read together
with the full captures (V1-a mix: 0.73% overlapping at 24t; V1-b: 11.4%
at 100t), activation is **scale-dependent**: effectively NOT POWERED at
24t, potentially powered at 100t — a 100t activation spot-check rides
the next terminal card. The ξ-gate/penalty pair is unpowered everywhere
observed (the heavy-tail flag never fired), so expectation 4 is
CONFIRMED in substance at 24t with the 100t caveat recorded; the POT /
ξ / penalty / block-length sweeps stay deferred per §6.0.

**V3-a R11 sweep, 24t (2026-07-12; n=2,735):** the shipped cell (25.0,
0.90) sits on a SHELF — no cell in the 5x3 grid is special — but the
shelf's content is the finding: **breach lift <= 1 across the entire
grid** (shipped cell: flagged rows breach 0.0% vs 7.1% for the calm
unflagged top-bin), flagged mean realized BEATS unflagged (+$368/row at
the shipped cell), and the D23 over-confidence gap is SMALLER in the
flagged region (+0.067) than in what R11 leaves untouched (+0.108). On
a window with no crisis onset, the danger R11 guards against lives
BELOW its VIX threshold — the F-V1-1 onset-blindness seen from the
reviewer's side. Expectation 3's shelf/spike framing got a third
outcome (a shelf of INVERTED lift). NOT a verdict on R11 yet: D23/i11
validated it on onset windows this capture does not contain; **V3-a-100t
(2020-02 start) arbitrates** — if the flagged region shows lift > 1 and
a larger over-confidence gap there, R11 is doing its documented job on
the windows it was built for and is simply inert on calm regimes; if
the inversion persists even with 2020 in-sample, R11's conditioning is
mis-aimed and that goes to the re-baseline queue.

*(F4 threshold/cap sweep results: pending — 9 engine passes in flight.)*
