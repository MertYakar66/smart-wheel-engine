# Parameter-OOS validation gate

**Question this answers:** every locked backtest (S27/S32/S34/S35) reports the
*shipped* engine's rank quality. S35 is out-of-**window** — the same fixed
parameterization on an unseen *period* — but the engine's static magic numbers
(regime state weights, dealer clamp, POT-GPD threshold, R11 cutoffs, heavy-tail
penalty) were all hand-set with full-history visibility (**caveat E5**,
`docs/BACKTEST_REGRESSION_CAMPAIGN.md`). So no locked number is
out-of-**parameter**. This gate measures how much of the reported edge survives
when the one parameter surface that is *active* on the Bloomberg provider — the
regime (HMM state-weight) overlay — is re-selected on a training partition and
evaluated on a temporally-disjoint hold-out.

**Bottom line.** On the 24-name config over 2020–2025, (1) the engine's
cross-sectional rank-vs-realized ρ is **not stable across windows** — it
oscillates by calendar year `+0.25 / −0.23 / +0.25 / −0.03 / −0.11 / +0.16` and
its 2020–2025 pooled value is `+0.005` (z=0.21, **indistinguishable from zero**),
even though the locked S27/S34 windows (2022–2024) report `0.18 / 0.31`; and
(2) **re-fitting the regime overlay on a training partition manufactures a
statistically-real in-sample ρ (`+0.112`, z≈3.8) that collapses to noise
out-of-parameter (`+0.016`, z≈0.4) — an optimism gap of `+0.096`**, with
train-optimal weights (`crisis 2.0, bear 0.1, bull_quiet 2.0`) that invert the
shipped `{0.2, 0.5, 1.0, 1.25}`. The shipped overlay adds no robust OOS rank
value over `ev_raw` (both ≈ 0). This is consistent with **E1** (equity-beta
dominance), **E5** (parameter-in-sample), and **i9** (regime tuning does not
generalize). Full numbers + caveats in [§5 Results](#5-results).

> **Scope / invariants.** This is a *measurement harness*. Its outputs feed no
> trade, no ranking, and no `ev_dollars` (CLAUDE.md §2). It **calls**
> `WheelRunner.rank_candidates_by_ev` read-only and does every re-weighting
> offline in numpy on the captured `ev_raw` column. **No production parameter
> default is changed** — a re-selected value that differs from the shipped one
> is reported as a *finding*, never shipped (task invariant 3). The
> decision-layer trio is untouched (CI-gated).

---

## 1. The overfitting surface (Phase 0 inventory)

Every static constant chosen with full-history visibility, with its shipped
value and whether it is fit **online per `as_of`** (PIT-clean, low over-fit
risk) or **static hand-set** (the over-fit surface). Verified against source.

| Parameter | `file:line` | Shipped value | Fit mode | Over-fit risk |
|---|---|---|---|---|
| Regime weight — crisis | `engine/regime_hmm.py:303` | `0.2` | **static** | High — crisis realized rate swings 0.37–0.93 (i9); few crises to fit |
| Regime weight — bear | `engine/regime_hmm.py:304` | `0.5` | **static** | Medium |
| Regime weight — normal | `engine/regime_hmm.py:305` | `1.0` | **static** (gauge) | Low (reference level) |
| Regime weight — bull_quiet | `engine/regime_hmm.py:306` | `1.25` | **static** | Medium |
| Regime multiplier clamp | `engine/ev_engine.py:552-553` | `[0.0, 1.25]` | **static** | Low (safety rail) |
| HMM fit (means/cov/trans) | `engine/wheel_runner.py:1999` | `n_states=4, n_iter=20, seed=42` | **online** per `as_of` on trailing returns | Low — PIT-clean; leakage-free by construction |
| F4 vol-widening trigger | `engine/forward_distribution.py:~529` | ratio `≥ 1.30`, slope `0.20`, cap `1.15` | **static** (ξ,β of the tail fit online) | Medium — value chosen to not invert S27 ρ (in-sample tuned) |
| POT-GPD threshold | `engine/tail_risk.py:58` | `95.0` pct | **static**; GPD ξ/β fit online (MLE) | Low-Medium |
| POT min exceedances | `engine/tail_risk.py:85` | `15` | **static** | Low |
| Heavy-tail ξ gate | `engine/tail_risk.py:225` | `ξ > 0.3` | **static** | Medium |
| Heavy-tail penalty | `engine/ev_engine.py:260` | `0.5` | **static** | Medium |
| Dealer clamp | `engine/dealer_positioning.py:738-739` | `[0.70, 1.05]` | **static** | **Not testable on Bloomberg — inert (dealer_multiplier ≡ 1.0; no GEX/flow feed)** |
| R11 VIX threshold | `engine/candidate_dossier.py:65` | `25.0` | **static** | Medium — already leave-one-crisis-out validated (D23/i11) |
| R11 top-bin prob | `engine/candidate_dossier.py:64` | `0.90` | **static** | Medium |
| Block-bootstrap block | `engine/forward_distribution.py:153` | `5` | **static** | Low |
| MC block / n_sims | `engine/monte_carlo.py:88-89` | `21` / `10000` | **static** | Low (reporting layer) |

**What is testable parameter-OOS on the available data.** On the Bloomberg
provider, `skew_multiplier`, `news_multiplier`, `credit_multiplier`, and
`dealer_multiplier` are all `1.0` (no chain / no flow / D18 news no-op), so
`ev_dollars ≈ ev_raw × clamp(hmm_multiplier)`. **The regime (HMM state-weight)
overlay is the only active tuned surface** — and it is the marquee E5 risk (the
i9 leave-one-crisis-out story is precisely about regime recalibration failing to
generalize). This gate therefore targets the regime overlay directly. The
dealer clamp, POT-GPD, F4, and R11 parameters do not move the cross-sectional
rank-vs-realized ρ on Bloomberg data (they act on tails / verdicts / prob_profit,
or are inert), so a clean rank-ρ hold-out cannot score them — they are marked
**not established** here and speced for a full re-fit in [§6](#6-what-is-not-established).

---

## 2. Method

One production engine pass ranks `UNIVERSE_24` on a sampled business-day grid
(`2020-06-01 → 2025-06-30`, every 8th b-day) and forward-replays every ranked
row to its held-to-expiry realized P&L — the **same** rho signal
(`spearman(ev_dollars, realized_pnl)`) that `_common._compute_metrics` locks for
S27/S34. The diagnostic columns `ev_raw`, `hmm_regime`, `regime_multiplier` are
captured so the overlay can be **re-derived offline** without re-running the
engine.

* **Phase 1 — walk-forward out-of-window scorecard.** The captured span is cut
  into `N` contiguous rolling folds, each scored with the *fixed shipped*
  `ev_dollars`: per-fold ρ, hit-rate, mean realized P&L, Brier, ECE. This shows
  whether the fixed parameterization's rank quality is *stable across windows*
  (the online HMM/POT-GPD components are PIT-clean; the static constants saw all
  folds during hand-tuning, so this is out-of-window, **not** out-of-parameter —
  that is Phase 2).

* **Phase 2 — parameter hold-out.** Split the rows into TRAIN
  (`as_of ≤ 2023-06-30`) and a disjoint HOLDOUT (`as_of ≥ 2023-08-20`), with an
  embargo wide enough that **every train option expires before the first
  holdout ranking date** ([§3](#3-no-leakage-proof)). Re-select the regime
  overlay on TRAIN ONLY, two ways:
  * **per-regime scalars** — grid-search scalars keyed on the dominant-state
    label (`normal` pinned to 1.0 as gauge) that maximise TRAIN ρ; maps 1:1 to
    the four shipped weights;
  * **tilt exponent γ** — `ev_raw × (ev_dollars/ev_raw)^γ`; γ=0 is no overlay,
    γ=1 is shipped. Dials the full multiplier tilt up/down; exercises the entire
    overlay, not just the per-label approximation.

  Then evaluate the TRAIN-fit parameters on the HOLDOUT and report the optimism
  gap = `TRAIN ρ at the re-fit optimum − HOLDOUT ρ at that same re-fit`.

---

## 3. No-leakage proof

The airtight, per-row-checkable invariant enforced by
`parameter_oos.assert_no_leakage` and pinned in the snapshot's
`leakage_certificate`:

> **every TRAIN row's option expires strictly before the earliest HOLDOUT
> ranking date.**

A row's realized P&L becomes knowable only at its option's expiration. If every
train option has expired before any holdout ranking decision, then no train
outcome can carry information dated at/after a holdout decision — so a parameter
re-fit on train literally cannot have "seen" the holdout. The certificate
records `max_train_expiration_date`, `min_holdout_ranking_date`, and the gap in
days; the test `test_snapshot_leakage_certificate_is_clean` asserts
`max_train_expiration_date < min_holdout_ranking_date`.

The embargo window `(2023-06-30, 2023-08-20)` — ~51 calendar days, wider than
the 35-DTE option horizon (~37 calendar days business-adjusted) — is what
creates that gap. Rows falling inside the embargo are dropped. Rows whose option
never resolved (realized P&L NaN, e.g. near the data frontier) are ignored by
every ρ/scorecard computation.

**The shipped constant is not leakage-free — by design.** The production weights
were hand-tuned on full history, so they *did* see the holdout. That is exactly
the optimism we are measuring: the honest counterfactual is the TRAIN-only re-fit
evaluated on the holdout it never touched, which the table below reports beside
the shipped constant.

---

## 4. Reproduce

```
# 1) expensive one-time engine pass -> committed rank-table fixture
python scripts/run_parameter_oos.py build
# 2) fast pure-numpy analysis -> committed snapshot JSON
python scripts/run_parameter_oos.py analyze
```

* Fixture: `tests/fixtures/param_oos/rank_table_24t.csv`
* Snapshot: `backtests/regression/snapshots/param_oos_regime_24t.json`
* Locks: `tests/test_parameter_oos.py` — fixture-independent leakage/identity
  unit tests (fast lane); fixture↔snapshot recompute lock (fast lane); engine
  regeneration lock (`backtest_regression` marker, slow).

---

## 5. Results

Snapshot: `backtests/regression/snapshots/param_oos_regime_24t.json`
(`UNIVERSE_24`, 2020-06-01 → 2025-06-30, every 8th b-day, 35-DTE/25-δ, 1957
resolved rows). ρ standard error ≈ `1/√(n−1)`; a result is "significant" here at
|z|≥2.

### 5a. Honest table — reported (in-sample) vs out-of-parameter

Train = `as_of ≤ 2023-06-30` (n=1170); holdout = `as_of ≥ 2023-08-20` (n=753),
leakage-certified (§3). ρ SE ≈ 0.029 (train) / 0.037 (holdout).

| Variant | TRAIN ρ | HOLDOUT ρ | Note |
|---|---|---|---|
| `ev_raw` (no overlay, γ=0) | +0.020 | −0.026 | baseline: forward-dist EV alone |
| **shipped** weights `{0.2,0.5,1,1.25}` (γ=1) | +0.037 | **−0.024** | production constant (saw holdout when tuned) |
| TRAIN-refit per-regime scalars | **+0.112** | +0.016 | out-of-parameter; z(train)≈3.8 |
| TRAIN-refit tilt exponent γ | +0.064 | +0.002 | out-of-parameter |

* **Optimism gap (regime scalars)** = **+0.096** (train +0.112 → holdout +0.016;
  ~86% of the apparent train edge does not survive)
* **Optimism gap (tilt γ)** = +0.063
* **Re-fit beats shipped on holdout?** technically yes (+0.016 > −0.024) but both
  are within 1 SE of zero — neither has a real OOS edge
* **Overlay adds OOS value beyond `ev_raw`?** no (shipped −0.024 ≈ ev_raw −0.026)
* **Train-optimal weights** = `{crisis: 2.0, bear: 0.1, normal: 1.0,
  bull_quiet: 2.0}` (γ*=3.0) — implausible and inverting the shipped prior
  (crisis 0.2→2.0), the signature of a sample-fit optimum.

**Reading it.** The only statistically-real ρ in the table is the *train* re-fit
(+0.112). It is manufactured: grid-searching four scalars on 1170 rows finds a
combination that orders the train outcomes, but that combination carries no
information about the holdout (ρ +0.016, z≈0.4). The shipped constant does not
rescue this — it is itself ≈0 out-of-sample. **The regime-overlay parameter
holds little robust cross-sectional rank signal; what looks like edge when you
tune it is optimism.**

### 5b. Walk-forward per-fold scorecard (fixed shipped `ev_dollars`)

Out-of-*window* stability of the shipped parameterization (online HMM/POT-GPD are
PIT-clean; static constants saw all folds — E5). ρ SE ≈ 0.05 per fold.

| Fold | Period | n | ρ | hit | mean $ | Brier | ECE |
|---|---|---|---|---|---|---|---|
| F1 | 2020-06 → 2021-06 | 398 | −0.014 | 0.935 | +153.7 | 0.077 | 0.105 |
| F2 | 2021-06 → 2022-06 | 381 | +0.045 | 0.774 | −31.6 | 0.177 | 0.059 |
| F3 | 2022-06 → 2023-06 | 385 | +0.116 | 0.730 | −27.5 | 0.203 | 0.090 |
| F4 | 2023-06 → 2024-06 | 398 | **−0.138** | 0.824 | +38.9 | 0.152 | 0.072 |
| F5 | 2024-06 → 2025-06 | 410 | +0.020 | 0.790 | −70.7 | 0.174 | 0.085 |
| **pooled** | 2020-06 → 2025-06 | 1957 | **+0.005** | 0.812 | +15.6 | 0.156 | 0.048 |

Per-fold ρ spans −0.14 → +0.12 (range 0.25, ≈5 SE) — the fixed-parameter rank
edge is **not stable out-of-window**; the pooled +0.005 is the average of large
sign-flipping oscillations, not a small steady edge.

**Reconciliation with S27/S34/S35.** These are not in conflict — rank-ρ is
strongly window- and universe-dependent: S35 `0.50` (2018–2020), S27 `0.36`
(2022) decaying to `0.06` (2024), S34 `0.29–0.35` (100-name, 2022–2024). The
locked numbers sit on *favorable* windows; the 24-name pooled 2020–2025 value is
≈0. Two disclosed caveats on the absolute level (they do **not** affect the
optimism-gap or instability findings, which are within-methodology differences):
(i) **universe size** — 24 names is structurally noisier than S34's 100 (a
100-name replication would likely land nearer S34's stable 0.3; speced, not run);
(ii) **sampling density** — every-8-b-day sampling raises per-year SE to ≈0.05, and
my calendar-year ρ (2022 +0.25, 2023 −0.03, 2024 −0.11) diverges from S27's
daily-sampled per-year (0.36/0.19/0.06) by up to ~3 SE in 2023–24. A daily-sampled
replication is the natural robustness follow-up.

### 5c. Regime-conditional holdout (Phase 3, consistent with i9)

The committed **leave-one-crisis-out** study
(`archive/2026-07/HEAVY_VERIFY_2026-05-31_I9_FIX_GENERALIZATION.md`;
driver `docs/verification_artifacts/campaign_2026-05-31/i9_calibration_generalization.py`)
found the regime overlay's core failure mode: the **(crisis, top-bin) realized
rate is wildly unstable across crises — 0.57 (2020) / 0.93 (2021) / 0.83 (2022)
/ 0.37 (2025), a 56.5pp spread** — so no static or regime-conditional
recalibration of the overlay generalizes to an unseen crisis. **Do not
contradict this.** This gate does not re-derive a calendar leave-one-crisis-out
result (the sampled span `2020-06 → 2025-06` contains only the 2022 crisis; the
2020 COVID crash predates it — a fresh calendar LOCO needs that earlier window
and is already covered by the committed i9 study). Instead it reports, from the
**leakage-clean** temporal holdout, the fixed overlay's ρ/calibration **broken
down by regime label** — which surfaces the same crisis-bin unreliability
without a leaky cross-time claim.

| Holdout regime | n | ρ | hit | mean $ | Brier | ECE |
|---|---|---|---|---|---|---|
| crisis | 112 | −0.067 | 0.768 | −129.6 | 0.190 | 0.088 |
| bear | 221 | +0.026 | 0.765 | −146.6 | 0.184 | 0.055 |
| normal | 243 | +0.091 | 0.823 | +42.8 | 0.153 | 0.076 |
| bull_quiet | 177 | **−0.222** | 0.887 | +114.4 | 0.128 | 0.131 |

The overlay's rank signal is where i9 predicts it is weakest: the **crisis** bin
has negative OOS ρ (−0.067) and loses money on average (−$130/contract), and the
**bull_quiet** bin inverts hard (ρ −0.222, ≈3 SE at n=177) — higher-`ev_dollars`
calm-bull candidates did *worse*. Only `normal` (ρ +0.091) is weakly positive.
This is the cross-sectional echo of i9's finding that the (crisis, top-bin)
realized rate is unstable across crises — the overlay cannot be relied on to
order outcomes precisely in the regimes that matter. (Per-bin n is small; treat
signs as directional, magnitudes as noisy.)

### 5d. Split-robustness of the optimism gap

Guards against a cherry-picked split date — the re-fit is repeated at four
leakage-certified splits (`split_robustness` in the snapshot; every row
`leakage_free`).

| train ≤ | holdout ≥ | n (train/hold) | refit train ρ | refit holdout ρ | **gap** | shipped holdout ρ |
|---|---|---|---|---|---|---|
| 2022-12-30 | 2023-02-20 | 978 / 941 | +0.112 | +0.008 | **+0.104** | −0.037 |
| 2023-06-30 | 2023-08-20 | 1170 / 753 | +0.112 | +0.016 | **+0.096** | −0.024 |
| 2023-12-29 | 2024-02-20 | 1371 / 542 | +0.094 | +0.035 | **+0.059** | −0.014 |
| 2024-06-28 | 2024-08-19 | 1565 / 350 | +0.059 | +0.091 | −0.032 | +0.037 |

The optimism gap is **positive and material for 3 of 4 splits** (+0.06 … +0.10):
a train re-fit consistently manufactures ρ ≈ +0.1 that does not survive OOS, and
the **shipped constant is ≤ 0 out-of-sample in all three**. The exception is the
last split, whose holdout is the smallest (n=350, SE ≈ 0.053) and latest — there
the (already weak, +0.059) train re-fit happens to generalize; within noise, and
not evidence against the pattern. **The finding is not an artifact of the
canonical split.**

---

## 6. What is NOT established (and the spec to establish it)

* **Dealer clamp `[0.70, 1.05]` — not testable on Bloomberg.** `dealer_multiplier
  ≡ 1.0` (no GEX/options-flow feed), so the clamp never fires. Establishing it
  needs a Theta options-flow window and a re-run capturing a live dealer signal.
* **POT-GPD threshold / F4 trigger / heavy-tail penalty / R11 cutoffs — not
  scored by rank ρ.** These act on tail CVaR, `prob_profit`, or the reviewer
  verdict, not the cross-sectional `ev_dollars` ordering, so a rank-ρ hold-out is
  the wrong instrument. A proper parameter hold-out for each requires **re-running
  the engine per grid value with the constant monkeypatched in the harness**
  (permitted by invariant 3) and a metric matched to the parameter (tail
  coverage / Brier / per-contract P&L). That is a compute-heavy follow-up, speced
  but not run here. R11 specifically is *already* leave-one-crisis-out validated
  (D23/i11) — extending that into the locked set is Phase 3.
* **Full per-fold rolling re-fit** (re-select the overlay independently in every
  walk-forward fold, not just one train/holdout split) would tighten the
  parameter-stability estimate. The single split is the conservative first cut;
  the rolling re-fit reuses `refit_regime_scalars` per fold and is a direct
  extension.
* **Portfolio NAV per fold** is a per-contract-P&L *proxy* here (`mean_realized`);
  true capital-constrained NAV needs the full `_common.run_backtest` tracker per
  fold (expensive) and is deferred.

---

## 7. 100-name replication (daily sampling) — does the 24-name null generalize?

The §1–§6 gate ran on `UNIVERSE_24`. This section replicates it on
`UNIVERSE_100` — the universe where the strongest *in-sample* signal lived
(**S34: in-sample ρ 0.313**, 2022–2024) — at **daily** cadence, to decide whether
the 24-name null (pooled OOS ρ ≈ 0) generalizes or an out-of-parameter edge
appears on the wider, S34 universe. Same shared harness
(`backtests/parameter_oos.py`), same leakage-certified split (`train ≤ 2023-06-30`,
`holdout ≥ 2023-08-20`) — only the universe, cadence, and significance treatment
differ. Snapshot `param_oos_regime_100t.json`; fixture `rank_table_100t.csv`;
locks `tests/test_parameter_oos_100t.py`; driver `scripts/run_parameter_oos_100t.py`.

### 7.0 The independence correction (why daily N is not more evidence)

Daily sampling creates heavily **overlapping** forward windows (a 35-DTE option
opened today and tomorrow share ~34 days of the same price path) and recurs the
**same 100 names** every day. So the pooled row count (~90k) massively
**overstates independent trials** — a naive z on that N is dishonestly tight.
Two corrections, both in the shared library:

* **Per-date cross-sectional ρ** — Spearman(`ev_dollars`, realized) *within* each
  as_of's candidate menu, then aggregated across dates. This answers "did the
  signal rank *today's* menu?" on genuinely date-level draws.
* **Moving-block bootstrap** — every CI resamples whole as_of **dates**, never
  individual rows (so same-day cross-sectional dependence is carried by
  construction), AND resamples them in **contiguous blocks of ~25 chronological
  dates** (`block_len` ≈ the 35-DTE horizon in trading days). Individual-date
  resampling (`block_len=1`) still treats serially-correlated nearby dates as
  independent and reports optimistically-tight intervals; the block breaks that
  serial dependence. The effective independent-unit count is therefore
  `n_dates / block_len` — e.g. ~53 blocks over ~1326 daily dates, **not** ~90k
  rows. The snapshot records both the block CI and the naive `block_len=1` CI for
  the holdout so the widening is explicit.

### 7.1 Sampling actually used

**Daily** — 1326 distinct as_of dates, 2020-06-01 → 2025-06-30, 63,187 ranked
rows (all resolved). No sub-sampling. (Finest cadence; the full daily pass took
~2.9 h, checkpointed per 6-month batch.)

### 7.2 The decisive result — the null does NOT blanket-generalize; a top-tier edge SURVIVES

**Verdict.** Measured across **all** ~48 candidates/day, the 100-name pooled ρ is
≈ 0 (holdout −0.077, block-CI95 [−0.19, +0.02]) — the 24-name null *does* hold for
the full cross-section. **But that all-candidate number hides a real,
out-of-window rank edge at the tradeable top tier.** The engine only ever *opens*
its highest-EV candidates, and there ρ is strongly positive and survives into the
holdout with honest block-bootstrap CIs that **exclude zero:**

| Segment | top-5 ρ [block-CI95] | top-15 ρ [block-CI95] | top-50 ρ | all (~48) ρ |
|---|---|---|---|---|
| TRAIN (2020-06 → 2023-06) | +0.560 [0.46, 0.64] | +0.354 [0.25, 0.45] | +0.221 | +0.022 |
| **HOLDOUT (2023-08 → 2025-06)** | **+0.597 [0.48, 0.70]** | **+0.371 [0.25, 0.49]** | +0.102 [−0.02, 0.23] | −0.077 [−0.19, 0.02] |
| S34 window (2022–2024) | +0.525 [0.43, 0.62] | **+0.316 [0.22, 0.41]** | +0.102 | −0.062 |

Four things make this a *first-class surviving edge*, not an artifact:

1. **It reproduces S34 exactly.** On the S34 window at top-15, ρ = **+0.316**
   (S34 reported **+0.313**, identical n=10,896) — the pipeline is faithful.
2. **It survives out-of-window — stable, neither decaying nor improving.** HOLDOUT
   top-15 = **+0.371** (block-CI [0.25, 0.49], ~20 effective blocks) is
   *statistically indistinguishable* from TRAIN (+0.354, block-CI [0.25, 0.45]):
   the intervals overlap almost entirely, so the near-equal point estimates are
   **not** evidence of an OOS *improvement* — only that the edge does **not decay**
   out-of-window. The block bootstrap is the honest interval (individual-date
   resampling would be ~3× tighter — see §7.0).
3. **It is BREADTH, not one name (E3).** HOLDOUT top-15: dropping BKNG is a no-op
   (0.371 → 0.370); leave-one-name-out range **[0.351, 0.404]** across **97
   distinct names**; BKNG P&L share 0.1%. Removing *any* single name leaves ρ ≈
   0.37. (On the *all-candidate* set the ρ is ≈ 0 and E3 is likewise flat — LOO
   [−0.024, +0.000] — so there is no hidden single-name edge there either.)
4. **It lives in `ev_raw`, not the tuned overlay — so it is robust to the entire
   E5 parameter surface.** Ranking the holdout top-15 by the *untuned, PIT-clean
   core* `ev_raw` gives ρ **+0.378** — essentially identical to `ev_dollars`
   +0.371. The regime overlay that §1–§6 showed does **not** generalize (optimism
   gap on 100t is +0.066, same story) is simply **not where the edge is**. The
   edge is the block-bootstrap/forward-distribution EV picking the best ~15 of ~48
   names — a parameter-light, online-fit signal with no static-constant overfit
   surface.

**Why the edge decays with `top_n` and why 24 names couldn't see it.** ρ falls
monotonically as the menu widens: +0.60 (top-5) → +0.37 (top-15) → +0.10 (top-50)
→ ≈ 0 (all). The engine's EV discriminates its *best* picks from a broad field
but cannot rank the low-EV tail it would never trade. The 24-name universe only
surfaces ~15 candidates/day — there is **no broad field to select the best 15
from**, so its "all-candidate" ρ (#484: +0.005) *is* its top tier, and it is ≈ 0.
The 100-name universe is what makes the selective top-tier edge measurable. The
two results are fully consistent: **different subsets of the same monotone curve.**

### 7.3 What is / isn't established (100-name)

* **Established (leakage-clean, honestly powered):** a breadth, out-of-window,
  block-CI-excludes-zero rank edge at the tradeable top tier (holdout top-15
  +0.37 [0.25, 0.49]), located in the parameter-light core EV. Every OOS number
  carries the per-row leakage certificate (§3) and a moving-block CI (§7.0).
* **Not a profitability claim.** ρ measures *rank ordering* of held-to-expiry P&L,
  not dollar edge; caveats E1 (equity-beta dominance) and E3 (dollar story) are
  unchanged — this is skill at *ranking* the top tier, not a return guarantee.
* **Out-of-window, not fully out-of-parameter.** The static constants saw all
  history (E5). This is mitigated for *this* edge because it sits in `ev_raw`
  (online-fit, not the static overlay), but a strict per-fold parameter re-fit of
  the forward-distribution knobs was not run — labelled, not claimed.
* **Regime overlay still adds no OOS value** on 100 names either (optimism gap
  +0.066; the re-fit collapses out-of-parameter exactly as in §5).
