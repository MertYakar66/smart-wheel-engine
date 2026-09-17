# V1 findings — tail-risk exceedance validation (closed 2026-07-12)

**Workstream:** V1 of the validation phase (`docs/VALIDATION_PHASE_PLAN.md`
§1/§3/§4). **Harness:** `backtests/tail_exceedance.py` (Kupiec POF,
date-clustered bootstrap CIs, violation-clustering permutation test,
ES-bound `cvar_5` breach test + severity, pooled heterogeneous-Bernoulli
prob_profit z). **Runs:** V1-a (24 names, 2022-01 -> 2026-05, every 5
bdays, 2,735 rows, sandbox) and V1-b (100 names, 2020-02-03 -> 2026-05-21,
every 2 bdays, **40,201 rows / 40,187 resolved**, operator terminal,
101.9-minute build, zero rank failures). Run artifacts are gitignored
(`data_processed/validation/tail_exceedance/`); every headline V1-b number
was independently recomputed by the executor session from the captured
table using the harness's exact breach definition and reproduced
byte-for-byte, and supplementary CIs used the harness's own
`date_clustered_rate_ci` (n_boot=2000, seed=12345). Both official reports
carry `OVERALL: PASS`; the findings below are what the PASS is made of.

**Scope discipline:** measurement-only throughout (CLAUDE.md §2). No
engine change ships from this workstream; candidates for change are
*queued*, not applied.

---

## 1. Results at a glance

| Statistic | V1-a (24t) | V1-b (100t) |
|---|---|---|
| rows resolved | 2,735 | 40,187 |
| p25 coverage (nominal 25%) | 17.1% — PASS-conservative | 17.75% (n=11,032 informative; CI [16.1%, 19.5%]) — PASS-conservative |
| p50 / p75 | INSUFFICIENT (win point mass) | INSUFFICIENT (n=26 / 0 informative) |
| cvar_5 breach, pooled (bound 5%) | 1.28% — PASS | 2.32% (931 breaches; CI [1.65%, 3.10%]) — PASS |
| — stratum vix_calm | 0.55% | 2.51% (n=8,013) |
| — stratum vix_elevated | 1.64% | 1.91% (n=24,133) |
| — stratum vix_crisis | 1.18% | 3.33% (n=8,041) |
| — stratum top_bin (prob>0.90) | 3.55% | **6.32%** (n=3,609) |
| — stratum traded_region (ev>0) | 2.81% | **4.92%** (n=12,595) |
| breach severity (median / mean realized÷cvar) | 1.36x / — | 1.87x / 2.93x |
| violation clustering (date ac1, perm p) | 0.40-0.67, p<0.001 | **0.966 (cvar), 0.761 (p25), p=0.0005 (floor)** |
| prob_profit pooled z | +0.94 (honest) | +3.40 (conservative: 32,386 obs vs 32,118 expected wins) |

## 2. Findings (with triage)

### F-V1-1 — Entry-VIX conditioning is structurally blind to crisis onset
**(the headline; pre-registered expectation falsified *as stratified*,
confirmed in a sharper form). Triage: engine finding + measurement-design
insight; NOT a harness artifact.**

The pre-registered expectation was "cvar_5 WARN/FAIL in the crisis
stratum" once a true onset (Feb 2020) was in-sample. The stratified
verdict says otherwise — vix_crisis breaches at 3.33%, under the bound.
The executor's supplementary (hindsight-labeled, diagnostic-only) slices
show why, and it is worse than the expectation, not better:

- Feb-Apr 2020 **calm-entry** rows breach **84.4%** (n=179, median
  severity 3.38x); **elevated-entry** 74.2% (n=205); crisis-entry 22.5%
  (n=915). Onset window pooled: 39.2% (CI [0.25, 0.52]).
- The two worst non-AZO cases are BA entered 2020-02-13/17 at **VIX
  14.15/13.68** — calm-labeled — realizing **7.8x / 7.1x** modeled cvar_5.
- 151 of the 201 calm-stratum breaches are 2020 rows: the calm stratum's
  elevated pooled rate (2.51%) is mostly the onset in disguise.
- The 2020 crisis-*labeled* entries run 6.11% but the stratum is diluted
  to 3.33% by 2022-grind crisis entries (1.56% — trailing distributions
  already widened; V1-a's explanation, confirmed).

Entry-VIX banding is PIT-honest — and that is precisely the finding: **at
onset, the entry-time market state is calm, so no entry-time VIX
conditioning (stratum, gate, or reviewer) can see the miss coming.** R11
(VIX > 25 top-bin size-down) could not have fired on BA in Feb 2020; no
threshold re-tuning fixes this. Damage-bounding mechanisms that do not
require prediction — R10's single-name cap, sector caps — are the only
levers that act on these rows, consistent with
`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10. Queued (not applied): the case for
a *forward-looking* tail input (IV-based) rather than a faster trailing
window — independently sharpened by V2-c, which showed a 3-year-stale
scenario set produces nearly the same pooled breach frequency as a daily
refit (`docs/VALIDATION_PHASE_PLAN.md` §5.6).

### F-V1-2 — Winner's-curse concentration replicates and reaches WARN
**Triage: engine finding (selection-layer tail optimism), replicated
across universes; not single-name, not a harness artifact.**

top_bin 6.32% vs pooled 2.32% (2.7x) and traded_region 4.92% (2.1x) —
the 24t ratios (2.8x / 2.2x) replicate almost exactly at 4x the name
count and 15x the rows. New at 100t: the top-bin **point rate exceeds
the generic 5% ES bound itself**, with supplementary date-clustered CI
[4.15%, 8.72%] straddling the bound — WARN-shaped under the harness's
own ladder semantics (the official report emits stratum rates, not
stratum verdicts; see §5 follow-ups). Breadth-checked: excluding AZO
moves nothing (6.32% / 4.96%), and breaches touch 94 of 98 names. The
engine's highest-confidence, actually-tradeable rows are where modeled
tails are thinnest relative to reality. Cross-reference: V2-c showed
freezing the scenario set *doubles* the top-bin population — selection
discipline is the thing refit recency buys, and this stratum is where
its absence would show first.

### F-V1-3 — Violations cluster at the permutation floor
**Triage: engine finding (regime-blind risk numbers), the I3-E
procyclicality signature, now formal at both scales.**

Date-level lag-1 autocorrelation of the violation rate: 0.966 (cvar_5)
and 0.761 (p25), permutation p = 0.0005 — the floor (no permutation
reached the observed value). Violations arrive in regime-length bursts.
Frequency PASSes; **independence emphatically does not** — the harness
ladder has no independence verdict (deliberately), so this is recorded
here as a named finding rather than a verdict. The V2-c freeze-replay
localizes the mechanism: live refitting de-clusters slow regime drift
(holdout ac1 0.10) but nothing de-clusters onsets.

### F-V1-4 — Breach severity: the 3-4.5x anecdote is now a statistic
**Triage: engine finding (magnitude, not just frequency).**

Among the 931 V1-b breaches: median realized/cvar_5 1.87x, mean 2.93x,
p90 6.06x, max 35x; **29.8% of breaches run >= 3x** the modeled expected
shortfall. The 2026-06-15 trader stress-test's "3-4.5x" single-name
anecdote is now a ~30%-of-breaches population statistic. Mean excess
-$1,480 per breach (24t: -$1,462 — stable).

### F-V1-5 — Calibration structure: conservative pooled, vacuous middle, optimistic top
**Triage: mixed — mostly confirms the known record; one
window-dependence note.**

p25 is conservative at both scales (17-18% vs 25%) — the safe direction,
never a FAIL by design. p50/p75 are honestly INSUFFICIENT: at 25-delta
the win point mass leaves n=26 / n=0 informative rows at 100t — these
quantiles are untestable on this book, not "passing". Pooled prob_profit
flips sign across windows (V1-a z=+0.94 honest; V1-b z=+3.40
conservative — more wins than forecast over 2020-2026) while the top bin
stays over-confident in tail terms (F-V1-2) — pooled calibration masks
bin-local structure, exactly as the W-series record predicted.

## 3. Benign anomalies (verified by the executor, recorded so nobody re-triages them)

- 14 unresolved rows: 7x BK (the BK->BNY rename seam at the 2026-03-23
  reconstitution — OHLCV ends 2026-03-20) + 7x CAG (OHLCV ends
  2026-06-04). Correctly excluded as unresolved, no spot fabricated.
- BNY and CASY absent from the table (98/100 names): 71 days of history
  vs the 504-day gate — correct gate behavior.
- The worst-cases list being 8/10 AZO is the dollar-ranking mechanic
  (AZO is a $3-4k stock); breach *rates* are AZO-robust (F-V1-2).
- distribution_source mix: 35,587 non-overlapping / 4,600 overlapping;
  ~49 rows/date, proportional to 24t. Only 56% of rows carry
  n_scenarios=35, so V1-a's "~2.9% model-implied ceiling" footnote is
  per-row at 100t; the generic 5% bound is the yardstick used.

## 4. Per-stratum verdict table (V1 closure requirement)

| Test / stratum | Verdict | Basis |
|---|---|---|
| p25 pooled | **PASS** (conservative) | official ladder |
| p50, p75 | **INSUFFICIENT** (point-mass vacuity — untestable on a 25-delta book) | official ladder |
| cvar_5 pooled | **PASS** | official ladder (CI [1.65%, 3.10%] < 5%) |
| cvar_5 vix_calm / elevated / crisis | **PASS** with F-V1-1 caveat — the calm stratum's rate is mostly disguised onset | rates + supplementary slices |
| cvar_5 top_bin | **WARN** | point rate 6.32% > bound; clustered CI [4.15%, 8.72%] straddles |
| cvar_5 traded_region | **WARN-lean** | 4.92% pressing the bound from below at n=12,595 |
| violation independence | **named finding F-V1-3** (no ladder verdict by design) | perm p at floor, both scales |
| prob_profit pooled | **PASS** (window-dependent sign; see F-V1-5) | official z |

**Triage summary:** every WARN/finding is an **engine finding**; none is
a harness artifact (the one harness artifact of the workstream — the win
point mass — was found and fixed during V1-a, before V1-b ran). Nothing
here mandates an engine change on its own; F-V1-1/2/4 are queued as
inputs to the D19/D21-style re-baseline discussion and to the
forward-looking-tail research question. R10/R11 stand as-is: R11 is
useful where it can see (elevated-VIX top-bin), and F-V1-1 documents
that no entry-time VIX rule can see onset — which is R10's job.

## 5. What V1 does NOT establish, and follow-ups

- **Not established:** VaR-level coverage (the engine emits no VaR_5
  series; the ES bound is the generic `P(X < ES_5) < 5%`, valid for any
  distribution but loose); premium realism (synthetic-BSM rail pinned
  off); dollar-P&L implications (rank/coverage statistics only); any
  claim about the two 2026-added names (BNY, CASY — history-gated out).
- **Comparison offset stands:** realized P&L is gross of entry costs +
  the $5 ITM fee (locked convention) — favorable to the engine on
  lower-tail tests, so the WARNs above are conservative evidence.
- **Follow-ups queued (non-blocking):** (i) stratum-level clustered CIs +
  verdicts inside `cvar_breach_report` so the official report emits what
  the supplementary pass computed; (ii) a diagnostic (hindsight-labeled,
  clearly non-PIT) onset-window slice in the report for exactly the
  F-V1-1 blind spot; (iii) the V1-a/V1-b prob_profit sign flip as a
  window-dependence exhibit for the multiplicity discussion (V6
  write-up).

**V1 acceptance (plan §4): met.** Both reports generated; per-stratum
verdicts above; triage complete. Workstream closed 2026-07-12.
