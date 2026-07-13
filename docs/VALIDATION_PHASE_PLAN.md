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
| V3 | **Parameter-plateau sweep** — perturb every static constant in the `docs/PARAMETER_OOS.md` Phase-0 inventory +/-20-50%; require plateaus, not peaks | Is the configuration a fitted artifact? | **CLOSED 2026-07-13** — no PEAK anywhere; F4 PLATEAU x2 at both scales; F-V3-1 (R11 mis-aim) to the re-baseline queue; results §6.6-§6.7 | `backtests/param_plateau.py`, `scripts/run_param_plateau.py`, `tests/test_param_plateau.py` |
| V4 | **Capacity curve** — contract ladder with the (dormant) Almgren-Chriss impact term armed via a swept stock-ADV proxy; participation-capped fills | Where is the knee of edge-vs-deployed-dollars, as a function of the proxy assumption? | **CLOSED 2026-07-13** — pilot + both 100t NAV arms; $1M capital-bound at every rung, $10M proxy-bound with an interior impact knee; results §7.4-§7.5 | `backtests/capacity_curve.py`, `scripts/run_capacity_curve.py`, `tests/test_capacity_curve.py` |
| V5 | **Reverse stress** — cheapest-path-to-ruin search, starting from the known blind spots (calm-VIX single-name gap on a top-bin name; margin procyclicality) | What breaks the book that no gate catches? | **designed + pre-registered (§8)** | `backtests/reverse_stress.py` (planned) |
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

**V3-b F4 sweeps, 24t grid (2026-07-13; 9 passes, 1,971 rows each,
2020-02 -> 2026-05 every 10 bdays; elev+crisis stratum n=1,596):**

| threshold | fire rate | elev+crisis breach | | cap | fire rate | elev+crisis breach |
|---|---|---|---|---|---|---|
| 1.10 | 28.8% | 1.63% | | 1.00 (OFF) | 0.0% | 1.75% |
| 1.20 | 20.0% | 1.63% | | 1.075 | 13.5% | 1.75% |
| **1.30*** | 13.5% | 1.63% | | **1.15*** | 13.5% | 1.63% |
| 1.40 | 9.1% | 1.63% | | 1.225 | 13.5% | 1.63% |
| 1.50 | 6.0% | 1.75% | | 1.30 | 13.5% | 1.63% |

**Verdicts: PLATEAU on both axes** (guard rho ok everywhere; no
DOMINATED value). Expectation 1 CONFIRMED — the F4 calibration is not a
fitted artifact; the shipped values sit on broad shelves. Expectation 2
**falsified in substance**: the F4-OFF control differs from shipped by
0.12pp of breach rate — TWO breaches in 1,596 rows — nowhere near 2 SE.
The sharper reading of the flat rows: the threshold axis moves the fire
rate by ~5x (28.8% -> 6.0%) with ZERO change in measured tail coverage,
and the cap effect saturates immediately (1.075 ~ OFF; >= 1.15 all
identical). **F4 is a plateau because it is barely load-bearing** — the
gentle 1.15x widening rarely flips a breach, whichever rows it fires on.
This is the third independent arrival at the same conclusion (V2-c:
freezing the trailing window barely moves pooled coverage; F-V1-1: the
misses live where trailing/entry-time signals can't see): the
trailing-vol reaction channel is not where tail honesty comes from.
Recorded as a finding, not a defect — F4's design goal ("gentle, never
invert rho") is met, and nothing here motivates touching it.

---

## 7. V4 — capacity curve: design + pre-registration

**Written 2026-07-13, before any V4 code or run exists.** §7.4 expectations
are frozen after the first run. Measurement-only: the harness is a sibling
driver in the sanctioned `r10_strict_driver.py` copy-the-driver pattern
(imports `_common` helpers, zero engine changes); any "better" sizing
implied by the results is a reported finding, never shipped.

### 7.0 Discovered constraints (recon 2026-07-13, recorded before design)

1. **The tracker is contract-blind.** `WheelTracker` hardwires one
   contract everywhere: premium credit (`open_short_put` books
   per-100-shares), BP reserve (`available_buying_power` reserves
   `strike*100`, no multiplier), assignment (flat 100 shares), and the
   scored `_forward_replay_realized_pnl` (`x100`, never `x contracts`).
   Consequently `run_backtest(contracts=N>1)` is **internally
   inconsistent** (N-contract BP gate + N-contract friction debits against
   a 1-contract book) — never exercised beyond 1 in any locked study
   (S27/S32/S34/S35 all `contracts=1`). Recorded as a finding; no fix
   ships from V4 (a real multi-contract tracker is a D-series decision).
2. **A no-impact capacity curve is a straight line by construction:**
   per-share premium/spread are contract-invariant and every dollar term
   (commission, assignment slip, collateral, notional) is exactly linear
   in N — if the same trades fire, total P&L is exactly N x the
   1-contract P&L. The curve can only bend through (a) the BP gate
   (loose mode) and (b) size-dependent impact, which the engine models
   (`calculate_slippage`'s Almgren-Chriss sqrt term,
   `transaction_costs.py:149-153`) but which is DORMANT — no production
   caller supplies `adv_contracts` (BRAIN_AUDIT 2026-06-11 item 4).
3. **Strict mode is excluded from V4** — HT-D established that
   `portfolio_delta_breach` dominates refusals (92.1%) under a delta cap
   that is documented as structurally miscalibrated for a wheel book
   (`docs/HEAVY_R10_STRICT_SCALE.md` F6: assigned stock alone saturates
   it and strict froze after 2020). Strict-mode capacity is bounded by
   that miscalibration, not by market capacity; it is already
   characterized by HT-D and would swamp the signal here. V4 runs LOOSE
   (the S27/S32/S34 canonical mode). R10's NAV-relative admissibility
   (N contracts of strike K needs NAV >= N*K*1000) is reported as
   context arithmetic, not simulated.
4. **No option-volume data exists in this checkout** — Bloomberg carries
   no option OI/volume (manual-OMON tier); the Theta `option_history`
   panels that would carry per-contract ADV are not materialized. Stock
   ADV exists (`sp500_liquidity.csv` `avg_vol_30d`, shares, 100/100
   UNIVERSE_100 coverage through 2026-07-02, served by `get_liquidity`
   with caller-supplied `end_date` for PIT). Worklist A9 sanctions the
   stock-ADV proxy but supplies NO option/stock ratio number — so V4's
   deliverable is the knee **as a function of the proxy assumption**,
   never a point estimate. The data-grounded knee is explicitly deferred
   to the Theta option-volume pull (acquisition plan E-13).

### 7.1 Method — capital-equivalent contract ladder, shared rank, engine-native impact overlay

**Capital-equivalent scaling (exact).** For ladder point N, run a plain
1-contract tracker at `capital = BASE/N`. Every cash flow in the
1-contract book (premium, $0.65 commission, Reg-T margin, assignment
cash, $5 ITM fee, BP reserve) is exactly 1/N of the N-contract world, so
returns, gating sequence, and NAV path (x N) are identical to a true
N-contract book under the linear cost model — the size-dependence enters
ONLY through the overlay, which prices each fill at the TRUE order size
N. This sidesteps constraint 7.0(1) without touching the tracker.

**Shared rank.** One daily `rank_candidates_by_ev` serves every
(N, proxy) tracker (the `run_backtest_multi_friction` /
`r10_strict_driver` pattern) — the rank is capital- and
impact-independent, so a 12-tracker ladder costs ~1x the dominant rank
bill (~20 min for 24t x 2022-2024).

**Impact overlay (engine-native, no double count).** On every option
fill (short-put open, covered-call open), on top of `full` friction:
`impact_per_share = calculate_slippage(mid=premium_raw, bid_ask_spread=0,
"sell", num_contracts=N, adv_contracts=adv(ticker, date))` — passing
spread 0 isolates the engine's own sqrt size term
(`k * mid * sqrt(N/adv)`, shipped `k=0.10`), while the spread cost stays
with the `full` overlay (composition stated; no double count).
**Participation cap:** a fill is REFUSED when
`N > p_cap * adv_contracts` (p_cap = 0.10, the desk convention);
refusals are logged per (N, proxy). Assignment stock-leg impact is a
documented second-order omission.

**The proxy (pre-registered).** `adv_contracts(ticker, date) = r x
avg_vol_30d(ticker, <= date)` with **r swept over {1e-5, 1e-4, 1e-3}**
contracts per share of stock ADV — spanning thin (AZO-class: ~30
contracts/day at r=1e-4 on ~300k shares) to liquid (AAPL-class: ~4,000
at r=1e-4 on ~40M shares). Central case r=1e-4. `k` stays at the
shipped 0.10 (its own calibration is worklist B5, needs fill data).

**Ladder:** N in {1, 5, 10, 25}. **Pilot config:** UNIVERSE_24, BASE
$1M, 2022-01-03 -> 2024-12-31 (the S32 window), `full` friction,
top_n=10, max_new_per_day=3 — 12 trackers, one shared rank, ~25 min
in-sandbox. 100t replication optional after the pilot (one shared rank,
~4 h — terminal-scale).

**Metrics per (N, r):** final NAV (rescaled x N) + total return; gross
premium collected vs impact dollars (impact share); net premium capture
in BASE-world dollars (the capacity curve proper); executed opens;
BP-gate refusals; participation-cap refusals; mean deployment. Knee =
argmax over N of net capture at each r.

### 7.2 Pre-registered expectations (falsifiable)

1. **Linearity control:** with the overlay disabled, `return_pct` is
   exactly PROPORTIONAL to N while no BP refusal has fired (the same
   1-contract book on 1/N the capital — the linear segment of the curve
   by construction), and any deviation from proportionality coincides
   exactly with the first BP refusal. A deviation WITHOUT a BP refusal
   is a harness bug, not a finding (the A/A of this study).
   *Pre-run correction (2026-07-13, before any run):* originally
   written as "identical returns at every N", which contradicted
   expectation 5 — identical returns hold only per-position, not at the
   portfolio level where deployment scales with N. Corrected to the
   proportionality form; expectation 5 unchanged.
2. **Impact bends the curve monotonically:** per-contract net capture
   declines in N at every r; impact share of premium grows ~sqrt(N).
3. **The knee location is proxy-dominated:** N*(r) shifts by at least
   one ladder step across the r grid — establishing that a data-free
   capacity point-claim would be dishonest (the reason E-13 exists).
4. **Thin names bind first:** participation-cap refusals concentrate in
   the lowest-ADV names (AZO-class), echoing R10's role as the
   damage-bounder on exactly those names.
5. **BP refusals appear between N=5 and N=25 at 24t** (S32 baseline:
   10.8% deployment at N=1 => ~9x headroom; max_new_per_day continues to
   throttle below that).

### 7.3 Runs

| Run | What | Where | Status |
|---|---|---|---|
| V4-pilot | 24t ladder {1,5,10,25} x r {1e-5,1e-4,1e-3} + linearity control | sandbox (brain) | pre-registered |
| V4-100t | 100t replication (decided on pilot results) | terminal | gated on pilot |
| V4-100t-10m | supplementary NAV arm at BASE $10M (added 2026-07-13 after the pilot, BEFORE either 100t run): at $1M the BP gate confounds the impact knee — the pilot showed BP refusals from N=5 — so a $10M arm lets the impact/participation knee show clean. Expectation 6 (pre-registered with this row): the BP knee scales out ~proportionally with BASE while the impact share of premium does NOT move (impact depends on N and ADV, not NAV), so at $10M the binding constraint at high N becomes participation/impact; falsifier: returns stay BP-dominated at $10M, meaning capacity is tighter than the deployment arithmetic suggests. Connects to the R10 admissibility arithmetic (25 contracts of a $200-strike needs ~$5M NAV). | terminal | pre-registered |

Acceptance for V4 closure: linearity control clean; the knee-vs-r table
+ per-(N,r) metrics recorded; constraint 7.0(1) (tracker
contract-blindness) triaged into the findings record; disposition of the
deferred data-grounded knee (E-13) restated. No engine change ships.

### 7.4 V4-pilot results (2026-07-13; 24t, $1M, 2022-2024, 16 grid points, 68-min run)

Linearity control: **PASS** — formally trivial (only N=1 ran unthrottled)
but the load-bearing half held: every deviation from proportionality
coincides with BP refusals (none without). The control ladder:

| N (control) | return% | opens | bp_refused |
|---|---|---|---|
| 1 | +7.03 | 99 | 0 |
| 5 | +15.95 | 69 | 363 |
| 10 | +11.86 | 42 | 978 |
| 25 | +35.72 | 31 | 1,379 |

**Scorecard against §7.2:** (1) PASS. (2) inconclusive at this scale —
impact share runs only 0.25%-3% of premium and is swamped by composition
noise (below). (3) CONFIRMED — knee shifts with r (r=1e-5 -> N*=10;
r=1e-4 and 1e-3 -> N*=25, the ladder edge). (4) confirmed directionally
(participation refusals: 155 -> 1,438 across N at the thin proxy).
(5) **FALSIFIED in direction** — BP refusals begin already at N=5 (363),
not between 5 and 25: the linear segment at $1M/24t ends before N=5
(instantaneous concurrency binds long before the 10.8% average-deployment
arithmetic suggested).

**The emergent finding that matters most:** beyond the BP knee at 24
names, the curve is a **concentration lottery, not a capacity signal**.
The control ladder is non-monotone (+7.0 -> +16.0 -> +11.9 -> +35.7):
at N=25 the scaled book holds 1-3 positions and the return is whichever
names it happened to ride — consistent with the E1/E3 record (equity
beta + single-name dominance). The same mechanism produces the apparent
paradox of an impact arm BEATING its control (+43.3 vs +35.7 at N=25,
r=1e-4): participation refusals redirect capital to a different draw
(40 opens vs 31), and a different draw can luck out. **Reading the
knee_n cells at high N as capacity evidence would be dishonest at this
universe size.** What the pilot establishes: the harness mechanics are
validated (proportionality, refusal accounting, sqrt-ish impact-share
growth: 0.25 -> 0.50 -> 0.70 -> 0.86% at r=1e-3 vs sqrt-N 1:2.2:3.2:5),
the BP knee at $1M/24t sits between N=1 and N=5, and the capacity
question proper needs the 100-name breadth to average composition
effects. **V4-100t: GO (recommended)** — next terminal card
(`run --config 100t`, one shared rank, ~4-5 h), after the V3 overnight
bundle lands.

### 6.7 V3 100t results (2026-07-13; overnight executor, 4h40m chain, all rc=0) + V3 closure

Provenance: the executor's `v3_overnight_report.md` (uncommitted, repo
root on the overnight machine) — preflight at `6bbf756`, zero tracked
modifications, all artifacts gitignored; capture rebuilt from the branch
+ committed CSVs alone.

**(a) R11 arbitration — 2020 in-sample only PARTIALLY restores the
premise; shipped cutoffs are mis-aimed.** At (25.0, 0.90): breach lift
0.96 (parity — up from the 24t inversion's 0.0, still <= 1) and the
over-confidence gap is STILL inverted (flagged +0.075 < unflagged
+0.111). Lift crosses 1 only ABOVE the shipped threshold (27.5 -> 1.43,
30.0 -> 1.74); the gap half is inverted at every prob=0.90 cell. Per the
pre-registered arbitration rule (§6.6), this goes to the **re-baseline
queue as F-V3-1**: R11's VIX=25 cut sits below the crossover where
elevated-vol top-bin picks actually degrade, and the over-confidence
mass R11 exists for (D23) sits in CALM entries at both scales — the
same structural fact as F-V1-1, now measured from the reviewer's side
on 40k rows with a real onset in-sample. Interpretation guard: this
does NOT contradict D23/i11 (different metric — forward-regime top-bin
realized rate vs flagged-vs-unflagged held-to-expiry contrast; both can
be true), and it is NOT a removal recommendation — R11 is downgrade-only
and its mis-aim costs bounded sizing-down of some fine trades. It is a
finding that any R11 recalibration discussion must start from: the
threshold that would make its lift real is ~27.5+, and no VIX cut
reaches the calm-entry over-confidence.

**(b) Activation — POWERED at 100t for the POT diagnostic; the EV-path
half stays unpowered.** gpd_fit_rate 4.12% >= the 2% floor (full-table
overlapping tier 11.44%) — the scale-dependent flip §6.6 predicted.
`heavy_tail` fired 0.00% everywhere observed, so the ξ-gate/penalty pair
(the only part of the POT surface that touches EV) remains NOT POWERED
at both scales. Disposition: the POT-threshold sweep is now *eligible*
at 100t but would test the calibration of a diagnostic column
(`cvar_99_evt`) on ~4-11% of rows, not a decision input — deferred with
that rationale; revisit if heavy_tail ever activates (Theta window).

**(c) F4 — PLATEAU on both axes at 100t.** The OFF-control (cap 1.00)
worsens elev+crisis breach by 0.064pp (~4 breaches in 6,273, ~0.10 SE)
vs shipped — directionally worse, not measurably. Replicates the 24t
verdicts and re-falsifies expectation 2: F4 is not a fitted artifact
AND is barely load-bearing, at both scales.

**V3 CLOSED 2026-07-13.** Acceptance met: dispositions honored (§6.0);
sweep tables + per-axis verdicts recorded at both scales (§6.6-§6.7);
activation rates recorded with the powered/unpowered split; no PEAK
verdict anywhere (no constant in the swept surface is a fitted
artifact); one finding triaged to the re-baseline queue (F-V3-1, the
R11 mis-aim). Workstream verdict, stated once: **the hand-set constants
survive perturbation — the E5 anxiety was concentrated on a surface
that is either on plateaus (F4), inert on this provider (POT/ξ/penalty,
dealer clamp), already-tested (regime overlay), or mis-aimed in a way
no threshold tuning fixes (R11, whose target lives below its own
conditioning variable).**

### 7.5 V4-100t results (2026-07-13; both NAV arms, 133 min each, Windows executor) + V4 closure

Provenance: `capacity_report_100t.json` + `capacity_report_100t_10m.json`
(gitignored, executor machine); zero rank failures; meta pins the
pre-registered constants; working tree HEAD-honest at `6897759`.

**$1M arm — the BP gate is the whole story at this NAV.** BP refusals
begin at N=1 (34 even on the 1-contract control; controls: N1 34 / N5
3,035 / N10 3,883 / N25 4,799) — question (a) answered OPPOSITE to the
card's framing: 100-name breadth does not delay the BP knee, the richer
flow (top-15, 3/day) saturates a $1M book at EVERY rung. The linearity
control passes only degenerately (0 unthrottled points). The knee table
(N* = 10/10/25 across r) is BP-shaped, and the 24t concentration
lottery persists at high N (two impact arms beat their own controls —
possible only as a different-draw effect). Impact share <= 2.45% of
premium in every cell.

**$10M arm — expectation 6 CONFIRMED 3/3; the falsifier rejected; the
A/A substantive.** (a) The BP knee scales out EXACTLY with BASE —
first refusal moves from N=1 to between N=5 and N=10; structural
identity makes it exact (the $10M/N=10 tracker IS the $1M/N=1 tracker;
byte-identical outputs). (b) Impact share is NAV-invariant at matching
cells — byte-identical where the executed book matches (e.g.
0.02112103... in both reports); every cross-arm difference traces to
book composition, never NAV in the impact formula. (c) At N=25 the
binding constraint is participation, not BP (4,186-4,813 participation
vs 0-28 BP refusals at r <= 1e-4); the control ladder turns monotone
(+3.3 -> +16.7 -> +33.4 -> +42.7 — the lottery was a small-book
artifact); per-contract capture declines monotonically up the ladder
(r=1e-4: 2.02 -> 1.11 -> 0.85 -> 0.38 %-per-N), expectation 2 finally
confirmed clean. Linearity A/A: N=1/N=5 unthrottled with return/N
agreeing to 2.15e-12; every deviation above coincides with BP refusals.
**First clean impact knee of the workstream:** the r=1e-5 column runs
with ZERO BP refusals and still bends at an interior N* = 10
(+0.85 -> +1.31 -> +1.90 -> +1.43) — a pure participation/impact knee,
unconfounded.

**Structural aliasing (recorded so evidence is not double-counted):**
capital enters only as BASE/N and impact only as N/(r x ADV), so the
$10M cell (N, r) equals the $1M cell (N/10, r/10) wherever both exist —
confirmed byte-for-byte on three cells, which doubles as a
determinism cross-check across two concurrent runs.

**Scorecard (7.2 + expectation 6):** (1) PASS — substantive at $10M.
(2) confirmed at $10M (monotone per-contract decline; sqrt(N)
impact-share growth holds approximately where books are stable —
1 : 2.08 : 2.63 : 3.65 vs sqrt-N's 1 : 2.24 : 3.16 : 5 at the cleanest
column — and is censored from above elsewhere: the participation cap
refuses exactly the highest-impact fills, truncating the law's own
tail). (3) confirmed (knee monotone in r at both scales and both
NAVs). (4) confirmed. (5) FALSIFIED at both scales — the knee arrives
earlier than deployment arithmetic suggests; at $1M/100t there is no
unthrottled rung at all. (6) confirmed 3/3.

**Replication addendum (2026-07-13, after closure):** the Mac
executor independently ran the $1M arm (1h34m, rc=0) — every reported
cell matches the Windows arm byte-for-byte (control ladder, throttled
points, knees, impact shares, refusal counts): a full cross-platform
determinism replication of the 782-day 16-tracker run, on top of the
aliased-cell check. The Mac report adds two mechanism details for the
record: the $1M/N=1 book at 100 names holds ~47 concurrent positions
(why BP binds from the first rung), the N=25 book only ~4 (why high-N
cells are draws); and `priced_out` = `no_adv_fills` = 0 across all 16
points (full ADV coverage; impact never consumed an entire premium).

**V4 CLOSED 2026-07-13.** Acceptance met: linearity control clean
(substantively at $10M); knee-vs-r recorded at two NAVs; constraint
7.0(1) (tracker contract-blindness) triaged to the findings record (a
real multi-contract tracker is a D-series decision); the data-grounded
knee remains deferred to the Theta option-volume pull (E-13), with the
capacity statement properly conditional: **at the S34-class
configuration, $1M is capital-bound at every contract size (market
impact is not the binding constraint); at $10M the book becomes
proxy/participation-bound with an interior knee at the thin proxy, and
modeled impact stays <= ~2.5% of premium everywhere** — the knee's
location is proxy-dominated, exactly why a data-free point claim was
pre-registered as dishonest.

---

## 8. V5 — reverse stress: design + pre-registration

**Written 2026-07-13, before any V5 code or run exists.** §8.3
expectations are frozen after the first run. Measurement-only: pure
offline computation over captured tables + OHLCV paths; no ranker
re-runs, no engine changes; the adversary's findings are reported damage
bounds, never sizing advice.

### 8.0 What reverse stress means here

Classical stress asks "given this scenario, what happens to the book?"
Reverse stress asks "what is the cheapest path to unacceptable damage
that the gate stack PERMITS?" The phase's closed workstreams hand V5 a
measured blind-spot composite instead of hypotheticals: calm-VIX onset
entries that no entry-time rule can see (F-V1-1), thinnest modeled
tails exactly in the top-confidence/traded region (F-V1-2), breach
severity ~30% >= 3x modeled ES (F-V1-4), sub-R10 per-name sizing as the
only damage bounder, and V4's fact that a $1M book runs BP-saturated.
**Ruin-class threshold (pre-registered): realized book loss >= 25% of
NAV within one held-to-expiry cycle.** NAV = $1M throughout.

**Interpretation guard, stated up front:** the V5-a adversary selects
on REALIZED outcomes — deliberate hindsight. Reverse stress measures
what the gate stack *permits*, an upper bound on composition damage;
it is NOT a claim the strategy would assemble that book. Only
gate-admissibility is entry-time-honest (every constraint checked uses
entry-time quantities).

### 8.1 V5-a — worst admissible book (offline search on the V1-b capture)

For each entry date in `tail_table_100t.csv`: eligible rows = engine-
tradeable (`ev_dollars > 0`, resolved outcome); per name the adversary
may take up to `floor(0.10 x NAV / (strike x 100))` contracts (the R10
cap — the sub-R10 concentration channel), subject to the R9 sector cap
(25% NAV per GICS sector; sector from the connector's fundamentals
snapshot — a documented current-labels approximation; unknown sector =
counted loudly and exempted, mirroring Q3 missing-evidence semantics)
and total collateral <= NAV (the cash-secured budget). The adversary
greedily maximizes realized LOSS per collateral dollar (greedy is the
pre-registered search — a documented lower bound on the true optimum,
i.e. conservative). Two variants: unrestricted-tradeable, and TOP-BIN
ONLY (`prob_profit > 0.90` — can the engine's highest confidence still
be composed into ruin?). Outputs per date: worst-book loss (% NAV),
modeled book CVaR (sum of `n_c x cvar_5`) for the model-vs-realized
gap, names/sectors/VIX-band; plus the cross-date distribution,
ruin-date list, and entry-VIX strata.

### 8.2 V5-b — margin procyclicality, resolved honestly (replay on crisis windows)

Recon fact first: the tracker's BP reserve is FULL collateral
(`strike x 100` — cash-secured by construction), while its admission
check uses Reg-T (~20%). Under the CSP mandate the classical margin
spiral is therefore **structurally absent** — cash is reserved in full
at entry and cannot be called away. V5-b decomposes the named blind
spot into what remains:

1. **Assignment-wave stress** — build the BP-saturated engine-chosen
   book (rows ranked by `ev_dollars`, 1 contract per name until
   collateral exhausts NAV) on four pre-registered crisis eves:
   2020-02-19, 2022-01-03, 2024-07-31, 2025-04-01. Daily replay over
   the 35-day cycle with **intrinsic-only marking** (put mark =
   `max(0, K - S_t) x 100` — no time value, so trough damage is a
   LOWER bound; conservative toward the engine, stated). Report: max
   book drawdown (% NAV), assignment fraction at expiry, and the
   **trough-liquidation counterfactual** (panic exit at the worst mark)
   vs held-to-expiry.
2. **Levered counterfactual** — the same book run at Reg-T initial
   margin instead of full cash (what the CSP mandate protects
   against): daily maintenance = Reg-T recomputed on the day's spot x
   a stressed-margin multiplier swept over {1.0, 1.25, 1.5} (broker
   tightening); record the first margin-call day and shortfall per
   window x multiplier.

### 8.3 Pre-registered expectations (falsifiable)

1. **Ruin is admissible.** The V5-a worst admissible book reaches
   ruin-class (>= 25% NAV) on at least the COVID-onset entry dates,
   with those dates concentrated in the CALM/elevated entry-VIX bands
   (the F-V1-1 composite). Falsifier: the gate stack caps worst-case
   composition damage below 25% everywhere — the gates are sufficient
   against historical single-cycle composition risk.
2. **The model-vs-realized gap composes.** On ruin dates, worst-book
   realized loss >= 3x the modeled book CVaR (F-V1-4 at book level).
3. **Confidence does not protect.** The top-bin-only adversary still
   reaches >= 15% NAV damage on onset dates.
4. **The CSP mandate is the load-bearing safety feature.** Assignment
   fraction >= 50% of the saturated book in the COVID window, with the
   trough-liquidation counterfactual >= 20% NAV while held-to-expiry
   damage is materially smaller — i.e. the book survives BECAUSE
   nothing can force the exit. Falsifier: trough and terminal damage
   are close — holding power is not what saves the book.
5. **The levered counterfactual fires.** At stress multiplier 1.5, the
   Reg-T book's first margin call arrives within ~15 trading days of
   2020-02-19; at 1.0 it survives all windows except (at most) COVID.

### 8.4 Runs

| Run | What | Where | Status |
|---|---|---|---|
| V5-a | worst-admissible-book search, both variants, on `tail_table_100t.csv` | terminal (Windows; minutes) | pre-registered |
| V5-b | assignment-wave + levered counterfactual, 4 windows x 3 multipliers | terminal (Mac; minutes) | pre-registered |

Acceptance for V5 closure: both reports recorded; the ruin-date list +
gate-permission statement written up; any "gates insufficient" finding
triaged to the re-baseline queue alongside F-V1-1/2/4 and F-V3-1; the
CSP-mandate statement (structural absence of the margin spiral)
recorded as a documented design fact. No engine change ships.

### 8.5 V5-a results (2026-07-13; Windows executor, ~1 min, first attempt)

Provenance: `reverse_stress_search_100t.json` (gitignored) over the
untouched V1-b capture (40,201 rows / 822 entry dates; 429 dates carried
an admissible adversary book); HEAD-honest at `1f3a2ba`.

**Expectation 1 CONFIRMED — the falsifier is rejected: ruin-class
composition damage IS admissible through the R10 + R9 + collateral gate
stack.** 8 ruin dates (>= 25% NAV in one cycle), every one a COVID-onset
entry (2020-02-11 -> 2020-03-02); nothing from the 2022 bear or any
other window reaches ruin. Depth and per-date ruin rate concentrate in
the CALM band exactly as F-V1-1 predicts — the two deepest books lose
**36.5% / 36.4% of NAV from entries at VIX 13.7 / 14.2**, and the calm
band's ruin rate (3/59 = 5.1%) exceeds elevated (0.7%) and crisis
(3.6%). Honest nuance, recorded: 3 of 8 ruin dates sit in the crisis
band (VIX 27.9-39.2, the late-Feb escalation) — "concentrated in
calm/elevated" holds for depth and rate, not as an exclusive location
claim. The worst book is diversified-LOOKING: 12-14 names across 6+
sectors, Financials exactly at the R9 25% cap, every name within R10 —
**sector diversification inside one short-vol strategy does not
diversify a market-wide onset.**

**Expectation 2 CONFIRMED 8/8:** realized loss exceeded the modeled book
CVaR by 3.18-7.07x on every ruin date, largest on the calm entries
(7.07x / 6.25x) — F-V1-4's per-row severity composes at book level; the
thin tails stack, they do not diversify away.

**Expectation 3 CONFIRMED, with the bound recorded:** the top-bin-only
adversary reaches >= 15% NAV on three onset dates (19.2% / 16.5% /
15.1%) — the engine's highest confidence composes into ~1/5-of-NAV
damage — but NEVER ruin-class (max 19.2%, q99 15.3%): `prob_profit >
0.90` does bound worst-case composition below 25% in this history,
even while those picks stay severely mis-modeled (x CVaR up to 7.40,
the highest multiple of the run).

Recorded observations: (i) the greedy fills residual budget with
profitable names once losers are exhausted — a concrete mechanism behind
the pre-registered "greedy = conservative lower bound" (true permitted
damage >= reported; immaterial on ruin dates, where losers saturate the
budget). (ii) The R9 unknown-sector exemption was live but immaterial
(one ticker, BK; never in a worst book). (iii) Connective tissue with
F-V3-1: the top-bin worst dates at VIX 27.9-39.2 sit ABOVE R11's
measured lift>1 crossover (~27.5) — R11 is not part of the V5-a
admissibility stack, and where it CAN see, it would have flagged exactly
those books; the calm ruin books (36.5% at VIX 13.7) remain unreachable
by any VIX rule — the two findings tell one story.

*(V5-b: pending — Mac executor.)*
