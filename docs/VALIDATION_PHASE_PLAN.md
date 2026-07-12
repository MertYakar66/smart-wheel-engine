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
| V1 | **Tail-risk exceedance validation** — Kupiec POF + date-clustered CIs + violation-clustering tests on the engine's own `pnl_p25/p50/p75`, plus the ES-bound breach test + severity on `cvar_5` | Are the engine's per-candidate risk numbers statistically honest, per regime? | **harness shipped; 24t run in flight** | `backtests/tail_exceedance.py`, `scripts/run_tail_exceedance.py`, `tests/test_tail_exceedance.py` |
| V2 | **Parameter freeze-replay (C1)** — snapshot every tuned artifact as-of a cutoff, replay forward touching nothing | Does any reported edge survive with parameters the past could actually have had? | **designed + pre-registered (§5)** | `backtests/freeze_replay.py` (planned) |
| V3 | **Parameter-plateau sweep** — perturb every static constant in the `docs/PARAMETER_OOS.md` Phase-0 inventory +/-20-50%; require plateaus, not peaks | Is the configuration a fitted artifact? | queued | (extends `backtests/parameter_oos.py`) |
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
| V1-b | `100t` (UNIVERSE_100) | 2020-02-03 (COVID entry) -> frontier-capped, every 2 bdays | terminal | task card issued |

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

Terminal command for V1-b (~2-4 h expected; write access only to the
gitignored `data_processed/validation/`):

```bash
export SWE_DATA_PROVIDER=bloomberg
python scripts/run_tail_exceedance.py full --config 100t
# then report back: data_processed/validation/tail_exceedance/report_100t.json
```

Acceptance for V1 closure: both reports generated; findings doc written with
per-stratum verdicts; any FAIL triaged into (a) engine finding -> candidate
for the D19/D21-style re-baseline queue, or (b) harness artifact -> fix and
re-run. No engine change ships from this workstream directly.

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
(tier-1 set: `ohlcv`, `vol_iv`, `treasury`, `vix`, `credit_risk`,
`liquidity`). Schedule-type files are copied intact (`earnings` — the
event-lockout gate legitimately needs the *future* earnings calendar;
`dividends`, `fundamentals`, `corporate_actions`, `broad_pull/` — mixed
semantics, identical in both runs so they cannot create false diffs; a
stricter tier-2 pass that truncates these too and *triages* diffs is a
noted follow-up, not part of the tier-1 assertion). Then:

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
the trade-space GPD fit (`cvar_99_evt`, ξ, heavy-tail penalty), and the
F4 factor. The HMM freeze is applied **offline** (the parameter_oos
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
| V2-a | amnesia (5 dates × 24 names, tier-1) | sandbox (brain) | pre-registered |
| V2-b | freeze snapshot @ 2023-06-30 + lock | sandbox (brain) | pre-registered |
| V2-c | frozen replay, 24t holdout grid | sandbox (brain) | pre-registered |
| V2-c-100t | frozen replay, 100t | terminal | gated on V1-b debrief |
