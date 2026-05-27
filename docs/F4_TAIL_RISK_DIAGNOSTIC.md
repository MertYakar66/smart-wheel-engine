# F4 tail-risk gap — diagnostic + fix plan (2026-05-26)

**Status:** diagnostic complete. Three hypotheses tested against the
COST April 2022 case live. **Root cause identified.** Fix is scoped
but NOT shipped in this doc — it is a quant-layer change that
warrants its own PR with full regression coverage.

**Reviewer:** Terminal B, fresh session.
**Scope:** characterise mechanically why `prob_profit = 0.8333`
stayed constant across COST's 31.5% drop in April-May 2022, in
both the pre-fix and post-fix engines. The phenomenon is the
single highest-leverage engineering finding from S22 / S27 / S32
and is the **B1 blocker** in `docs/PRODUCTION_READINESS.md`.

**Engine SHA at diagnostic:** `origin/main` @ `e504801` (post-IV-PIT-fix
+ all overnight campaign PRs).

---

## TL;DR

**Root cause:** `engine.forward_distribution.empirical_forward_log_returns`
with the default `lookback_years=5.0` and `non_overlapping=True` at
`horizon_days=35` yields only ~30 samples for a typical SP500 name.
After PIT-filtering (`<= as_of`), those samples are **bitwise
identical day-by-day during a short event window** — the 14-day
unfolding of an adversarial drawdown moves the cutoff edge by only
14 days, but the non-overlapping sampling at 35-day stride sees
that as **zero new sample points** until the cutoff has advanced by
≥ 35 trading days. The forward distribution therefore literally
cannot react to a mid-event spot move.

**Mechanism (verified live):**

| `as_of` | strike | spot | n_rets | rets_mean | rets_std | 5%-tail | P(OTM@expiry) |
|---|---|---|---|---|---|---|---|
| 2022-04-01 | 553.00 | 575.57 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-04 | 552.50 | 575.13 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-05 | 553.00 | 575.32 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-06 | 562.00 | 584.79 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-07 | 584.50 | 608.05 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-08 | 576.50 | 600.04 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-11 | 562.00 | 584.67 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-12 | 558.50 | 581.36 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-13 | 568.00 | 591.09 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |
| 2022-04-14 | 567.50 | 590.39 | 30 | +0.0342 | 0.0795 | −0.0923 | **0.8333** |

Every column except `as_of`, `strike`, and `spot` is **byte-for-byte
identical**. The 5y lookback at 35-day stride does not see the
unfolding event because non-overlapping sampling discards intermediate
data.

This matches the live S27 backtest's reported values exactly:
`prob_profit = [0.8333] × 10` with `distribution_source =
empirical_non_overlapping`. The diagnostic reproduces the engine's
behaviour.

---

## 1. Method

The diagnostic reproduces `WheelRunner.rank_candidates_by_ev`'s
forward-distribution call chain in isolation, varying only
`lookback_years`. The throwaway script is at
`_f4_diagnostic.py` in the worktree root during this session (deleted
before commit per the throwaway-script convention).

For each of the 10 COST candidates that S22 / S27 / S32 reported,
we call:

```python
empirical_forward_log_returns(
    ohlcv=conn.get_ohlcv("COST"),
    horizon_days=35,
    as_of=<entry_date>,
    lookback_years=<varied>,
    min_samples=10,
    non_overlapping=True,
)
```

and record `n_rets`, mean, std, 5th-percentile (the "tail" loss),
and `P(log_return > log(strike / spot))` — the probability that a
short put expires OTM given the empirical sample.

---

## 2. Lookback sensitivity — the diagnostic

### 2.1. Default 5y lookback (what production uses today)

**`P(OTM) = 0.8333` is constant across all 10 dates.** The 30-sample
empirical distribution is essentially frozen — the 14 days of cutoff
advance moves the non-overlapping 35-day sampling grid by less than
one stride.

Mathematical reason: with `horizon_days=35` and `non_overlapping=True`,
the sampler takes returns at indices `[0, 35, 70, ...]` from the
log-price array, filtered to `<= as_of`. Advancing `as_of` by 14
trading days does not add any new index unless the new edge happens
to land on a multiple of 35. In 14 days, that's a 14/35 = 40%
probability per day, but the 30-sample distribution is barely
perturbed even when a new sample IS added.

### 2.2. 2y lookback — partial responsiveness

`n_rets ≈ 14`. Some date-to-date variability emerges (P(OTM) shifts
from 0.857 to 1.000) but the sample is so small that estimates are
noisy and the engine could swing wildly on a single new sample.

### 2.3. 1y lookback (overlapping fallback engages)

`n_rets ≈ 220` because at 1y the `min_samples=10` threshold for
non-overlapping fails (1y / 35d ≈ 7 samples), so the engine falls
back to the **overlapping** branch which gives 220 samples.

P(OTM) varies day-by-day (0.8676 → 0.8682) but only marginally because
adding 1 day to a 220-sample distribution barely moves the median.
**Crucially, the 1y window REFLECTS recent vol better** — `rets_std`
varies from 0.0755 to 0.0774 across the 14-day episode, the 5%-tail
from −0.0955 to slightly less negative. The distribution is moving,
but not enough to change the verdict.

### 2.4. 6-month lookback

`n_rets ≈ 92` (overlapping). **P(OTM) = 0.6848 throughout the
episode** — 18pp lower than the 5y default. The 6-month window
overweights recent vol (which by 2022-Q1 was already elevated from
the December 2021 sell-off and Russia-Ukraine).

### 2.5. 3-month lookback

`n_rets ≈ 29` (overlapping). P(OTM) **oscillates 0.83 → 1.00** as
the cutoff sweeps — the window is too short and over-responds to
the most-recent few weeks of calm pre-drawdown spot action. The
engine would be biased OPTIMISTIC right before the drop because the
preceding 3 months looked calm.

---

## 3. Why the engine reports identical prob_profit across the COST window

The S27 rank log shows `prob_profit = [0.8333] × 10` with
`distribution_source = empirical_non_overlapping` for all 10 COST
candidates. My diagnostic reproduces this exactly with the engine's
default parameters. **The engine is doing what its code says — the
defect is in the parameters, not the math.**

Two compounding factors:

1. **Sparse sampling at long horizon.** 35-day non-overlapping
   sampling at 5y lookback = ~30 samples. The distribution barely
   updates day-by-day.
2. **Calm 2017-2022 sample.** Of those 30 samples, the dominant
   regimes were the post-2018 recovery + post-COVID bounce. Only
   2-3 samples reflect actual bear-market 35-day moves.

The HMM regime classifier IS detecting the regime shift (the rows
show `hmm_regime` flipping `normal → crisis → bear` mid-window),
but the regime detection is consumed DOWNSTREAM by
`regime_multiplier` in `EVEngine`, which adjusts the FINAL
`ev_dollars` number, not the FORWARD DISTRIBUTION ITSELF. So the
engine knows "this is a bear regime" but still computes
`prob_profit` from a calm-market sample.

---

## 4. Three candidate fixes — ranked by tractability and risk

### Fix A (recommended) — shorter default lookback + force overlapping below floor

**Change:** `empirical_forward_log_returns` defaults
`lookback_years=5.0 → 2.0`; when non-overlapping yields fewer than
`min_samples`, automatically fall through to overlapping (already
the existing cascade behaviour in `best_available_forward_distribution`,
but the lookback applies to BOTH paths).

**Mechanism:** 2y / 35d ≈ 21 non-overlapping samples; below the
`min_samples=20` floor on edge cases. The cascade falls through to
overlapping which gives ~470 1-day-step 35-day forwards. P(OTM)
becomes much more responsive to recent vol.

**Expected effect (from §2.3 above):** P(OTM) on COST April 2022
moves from 0.8333 (5y default) toward 0.7-0.8 (1y / 2y range). Not
a dramatic shift on its own, but combined with Fix B below it
should push the worst trades below the proceed-threshold.

**Risk:** lower in 2022 bear but possibly lower across the WHOLE
backtest. The S22 / S27 / S32 ρ ≈ 0.22 might shift; magnitudes
might decrease overall. **A full backtest re-run is required** to
confirm the change doesn't degrade the signal in calm markets.

**Cost:** small. Single-line default change; well-bounded test
exposure.

### Fix B (recommended, complement to A) — regime-conditioned distribution widening

**Change:** when the HMM regime classifier reports `crisis` or
`bear`, widen the empirical distribution's tail before computing
`prob_profit`. Two mechanical options:

- **B1:** scale the std-dev of empirical returns by a regime
  multiplier (e.g., `crisis → ×1.5`, `bear → ×1.25`). Pure
  rescaling — the empirical samples themselves stay; only their
  variance is bumped.
- **B2:** Resample with a weight bias toward historical samples
  whose own HMM regime was `crisis` / `bear`. More principled but
  needs an HMM-state column threaded through the historical OHLCV.

**Risk:** B1 is cheap and incremental. B2 is more principled but
requires re-fitting / re-classifying historical regimes once and
threading that state through the connector.

**Cost:** B1 is ~30 lines + tests. B2 is a multi-week effort.

### Fix C (research-grade, lower priority) — POT-GPD threshold sensitivity

**Change:** the POT threshold today is fixed at the 95th percentile.
Under calm-regime historical samples, the 95th percentile is too
mild (it's the worst 5% of a calm sample, not the worst 5% of a
regime-aware sample). Either:

- **C1:** lower the percentile (e.g., 90th) so more observations
  are classified as tail and contribute to the GPD fit.
- **C2:** select the threshold by minimum mean excess (the
  standard POT diagnostic; see McNeil-Frey-Embrechts 2005).

**Risk:** lower percentile means more samples per fit but each
exceedance is smaller; the GPD shape parameter changes. C2 is
more principled but requires running the mean-excess plot
diagnostic at every rank call (a single extra `O(n)` pass).

**Cost:** small. But this only matters if the tail FIT itself is
being used downstream — current code uses the empirical 5%-tail
directly for the `prob_profit` calc. So Fix C is a no-op for the
F4 finding unless the EV engine starts consuming the POT-GPD output
in `prob_profit`.

---

## 5. Proposed minimal fix (single PR scope)

Ship **Fix A + Fix B1 together** in one PR with a tight regression
test suite. Specifically:

1. **`engine/forward_distribution.py`:** lower `lookback_years`
   default from 5.0 → 2.0 in `empirical_forward_log_returns` AND in
   `best_available_forward_distribution`'s arguments to it.
2. **`engine/wheel_runner.py` or `engine/ev_engine.py`:** when the
   HMM regime classifier reports `crisis` or `bear`, multiply the
   empirical std-dev by a regime factor (1.5 / 1.25). Implementation
   could live at the EV-engine boundary so the forward distribution
   stays a pure function of OHLCV + as_of.
3. **`tests/test_f4_tail_risk_gap.py`:** extend with the COST April
   2022 sequence. Assert `prob_profit` moves DOWN at least 10pp
   somewhere in the 14-day window (from 0.8333 to < 0.75) as the
   drawdown unfolds.
4. **Re-run S22 / S27 / S32 backtests** on the fix branch. Confirm:
   - Overall Spearman ρ doesn't drop below 0.15 (the F4 fix shouldn't
     destroy the bull-market signal).
   - 2022 mean realized P&L improves (the engine refuses more bad
     trades during bear).
   - The COST 2022-04 mean realized loss decreases (from S32's
     −$7,500 to something less negative).

### Definition of done (B1 in `PRODUCTION_READINESS.md` §6)

| Criterion | Today | After fix (target) |
|---|---|---|
| COST 2022-04-14 `prob_profit` | 0.833 | < 0.65 |
| COST 2022-04 mean realized loss | −$7,500 | > −$5,000 |
| Overall Spearman ρ (post-fix) | 0.22 | ≥ 0.15 |
| Full pytest tests/ -q | 2,374 + new | new tests added, no regression |
| `tests/test_f4_tail_risk_gap.py` extended assertions | regression-watch only | live failure-mode assertion |

---

## 6. What this diagnostic does NOT do

- **Does not ship the fix.** Fix A + B1 is a quant-layer change.
  Touching `engine/forward_distribution.py` defaults is technically
  not in the §2 decision-layer file list, but the function is
  consumed by all three rankers + the regime detector — a careful
  PR with full backtest re-run is warranted, not a one-session
  rush.
- **Does not modify any engine code.** This is a documentation
  artifact + a script that lives in `_f4_diagnostic.py` (worktree
  root, deleted before commit per throwaway-script convention).
- **Does not exhaustively characterise Fix B2 or Fix C.** Those are
  research questions worth their own diagnostic passes; this doc
  pins the immediate path (Fix A + B1).

---

## 7. Sources

- **`docs/USAGE_TEST_LEDGER.md`** §S22 (PR #178 closed) — original F4
  finding.
- **`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`** §F4 — confirmed
  unchanged across IV-PIT fix.
- **`docs/PREDICTIVE_VALIDITY_REVIEW.md`** P5 — meta-verified
  finding.
- **`docs/ENGINE_BACKTEST_S32_FRICTION.md`** §F5 — F4 re-surfaces
  under friction; same `prob_profit = 0.833` constant observed.
- **`tests/test_f4_tail_risk_gap.py`** (PR #196) — current regression
  watch; passes today only because it asserts the GAP exists, not
  that it's fixed.
- **`docs/PRODUCTION_READINESS.md`** §3 Blocker B1 — the deployment
  consequence.
- **This doc** — the mechanism.

---

## 8. Action items for the engine team

1. Open a PR `claude/fix-f4-tail-risk-widening` implementing Fix A + Fix B1.
2. Extend `tests/test_f4_tail_risk_gap.py` with the COST 2022-04
   live-failure-mode assertion above.
3. Re-run S22 / S27 / S32 on the fix branch; produce a side-by-side
   doc.
4. If overall ρ drops below 0.15 in the calmer years, iterate on
   the regime multiplier (try 1.25 for `crisis` instead of 1.5; or
   make it data-driven from the realized-vol ratio).
5. When the fix lands, `docs/PRODUCTION_READINESS.md` §3 Blocker B1
   gets marked resolved and the §5 deployment matrix gets re-evaluated.

This diagnostic closes the **understanding** part of the F4 gap.
The **implementation** part is the next PR.

---

## 9. Fix A attempt result — INSUFFICIENT (added 2026-05-26)

A single-session attempt at the proposed Fix A (lookback shortening)
was made on branch `claude/fix-f4-tail-risk-lookback` and **reverted
without merging**. Result: **Fix A alone does NOT close the F4 gap;
in fact it makes the engine MORE optimistic on the COST 2022-04
case, not less.** Documenting the negative result here so future
work does not re-attempt the same path.

### What was tried

Three variants tested live against the F4 regression suite + the
COST 2022-04-04 case via direct ranker probe:

| Variant | COST 2022-04-04 prob_profit | distribution_source |
|---|---|---|
| Baseline (lookback=5.0, min_samples=20) | **0.8333** (the F4 gap) | empirical_non_overlapping |
| `lookback_years=3.0` (everything else unchanged) | 0.8333 (unchanged) | empirical_non_overlapping |
| `lookback_years=2.0` (everything else unchanged) | **0.8635 → 0.8771 (WORSE)** | empirical_overlapping (cascade fell through) |
| `min_empirical_samples=40` (forces overlapping at 5y) | **0.8766 (WORSE)** | empirical_overlapping (~1300 samples at 5y) |

### Why Fix A alone fails

The F4 gap manifests at the engine's default lookback. Naive
shortening (5y → 2y) at the COST 2022-04 case produces a sample
window of **2020-04 to 2022-04** — which is dominated by:

- COVID recovery V-shape (April 2020 → end 2020): big up moves
- 2021 strong bull market
- Just the very beginning of 2022 vol pickup

This 2-year window is **MORE BULLISH than the 5-year window** that
included pre-COVID grind. The engine's empirical forward
distribution therefore implies HIGHER `prob_profit` (engine more
optimistic), not lower.

Similarly, forcing the cascade to use the overlapping path (via
`min_empirical_samples=40`) gives ~1300 sample points at 5y. The
larger sample doesn't tilt toward the recent vol — it averages
across the same 5-year history with more granularity. Result: same
~0.87 prob_profit on COST 2022-04.

### What this proves

The F4 root cause is **NOT** the lookback window size or the
NOS-vs-overlapping cascade. The root cause is the **historical
backward-look** itself: any sample of past returns will fail to
predict idiosyncratic single-name drawdowns like COST 2022-04
(supply chain + inflation surprise) or UNH 2024-11 (regulatory
event).

The only way to widen the engine's distribution for these cases
is **regime-conditioned widening** (Fix B1) — multiply the
empirical std by a regime-dependent factor when the HMM regime
classifier flags `crisis` or `bear`, OR widen when the underlying's
30-day realized volatility is materially elevated versus its
historical baseline. This requires:

1. Threading the HMM regime label (or a recent-vol ratio) into
   the forward-distribution computation.
2. Calibrating the widening factor (1.25, 1.5, data-driven, etc.)
   against the S22 / S27 / S32 / S34 / S35 backtest set to confirm
   signal preservation in calm regimes.
3. Updating `tests/test_f4_tail_risk_gap.py` to flip the
   regression-watch assertions after validation.

**This is research-level engineering, not a single-session fix.**
Estimated effort: 3-6 hours implementation + 4-8 hours backtest
re-runs + 2 hours documentation. Should be its own scoped PR with
explicit owner ownership.

### Implications for `docs/PRODUCTION_READINESS.md`

§3 Blocker B1 cannot be marked resolved with Fix A alone. The
deployment matrix's "autonomous" verdicts remain blocked. The
"supervised at $1M with 100-ticker universe" conditional verdict
in §5 still requires F4 fix (B1 closure) before promoting to
unconditional.

### Sources for the negative result

- `engine/forward_distribution.py` defaults verified at baseline
  via `inspect.signature(...)` — `lookback_years=5.0`,
  `min_empirical_samples=20`.
- Live ranker probe (`_f4_probe.py`, throwaway, deleted before
  commit) on COST 2022-04-04 and 2022-04-14 across baseline +
  three variant settings.
- F4 regression suite (`tests/test_f4_tail_risk_gap.py`): all 5
  pass + 2 xfail across all variants. The regression-watch
  assertions are not sensitive enough to detect "engine became
  more optimistic" — they only assert the gap exists with a
  permissive threshold (`cvar_5 > -10% of collateral`).

---

## 10. Fix B1+C attempt rolled back — research-grade calibration needed (2026-05-27)

Following §9, a Fix B1 (regime-conditioned widening) and Fix C
(worst-of-two evaluation) attempt was shipped on
`claude/fix-f4-regime-conditioned-widening` (PR #253), validated
against the named F4 cases via unit-test pins, and **rolled back
after the S27 backtest validation surfaced a Spearman ρ inversion**.
Documenting the negative result here so future work does not
re-attempt the same calibration.

### What was tried (commits a055349, ec3d51b — both reverted)

**Fix B1** (a055349): multiplied the empirical forward-log-return
std by `regime_widening_factor(p_crisis, p_bear)` immediately before
`EVEngine.evaluate`. Factor bounded to `[1.0, 1.5]`. Sign-preserving
(factor ≥ 1.0). Mean-preserving (only widens spread). Wired in
`engine/wheel_runner.py::rank_candidates_by_ev`.

**Fix C** (ec3d51b): added a "worst-of-two" evaluation. When
widening fires, the ranker evaluates `EVEngine.evaluate` on both the
NOS-widened sample (Fix-B1 baseline) and an overlapping-widened
sample (~1225 samples at 5y/35d default; unlocks the POT-GPD
heavy-tail fit which is gated on `len(pnls) >= 200`). The
more-conservative result (lower `ev_dollars`) wins. Designed to
preserve the COST 2022-04 partial close while deepening the UNH
2024-11 close.

### Unit-test pins closed the named cases

Live probe on Fix B1+C @ ec3d51b vs pre-fix baseline @ 9f0afaf:

| Case | Pre-fix | Post Fix-B1+C | Status |
|---|---|---|---|
| UNH 2024-11 | 0.857 / +\$114.53 | 0.7725 / **−\$118.82** | F4 gap CLOSED (cvar past −10% of coll) |
| COST 2022-04 | 0.833 / +\$62.88 | 0.833 / **−\$25.31** | partial — ev_dollars flips |
| AAPL 2026-02 (control) | 0.857 / +\$5.50 | 0.7355 / −\$55.81 | flagged via legitimate HMM p_crisis=0.879 |

All three unit-test classes in `tests/test_f4_tail_risk_regression.py`
passed. Launch-blocker subset 93/93 passed. Full pytest (excl.
backtest_regression + slow): 2,428 passed, 2 xfailed.

### Backtest regression surfaced the failure mode

S27 IV-PIT backtest (`tests/test_backtest_regression.py::test_backtest_matches_snapshot[s27_ivpit_24t_100k]`):

| Metric | Pre-fix (snapshot) | Post Fix-B1+C | Δ |
|---|---|---|---|
| Spearman ρ (aggregate) | **+0.1881** | **−0.1450** | **INVERTED** |

This is not a degradation — the ranking signal flipped sign.
The engine's "best" picks underperform its "worst" picks across
2,400+ trades in the 2022-2024 window.

### Diagnosis: HMM widening over-fires across calm-bull periods

A fast Spearman probe (905 ticker-date pairs across 2022-2024) with
**Fix B1 only** (Fix-C reverted) confirmed Fix-B1 is also the root
regression — not just Fix-C:

```
Spearman rho(ev_dollars, realized_pnl) = -0.0940  (p=0.0046)
positive ev: 132 / 905
widening factor mean: 1.185
no-widening rate: 14 / 905 (1.5%)
```

The HMM fires `crisis` / `bear` labels on **98.5% of probed pairs**.
Reading the HMM source: the K=4 GMM labels `crisis` as the
HIGHEST-VARIANCE state, not "actual systemic crisis." During the
2022-2024 window, the HMM appears to oscillate between `bear` and
`crisis` labels for moderate-vol equity ranges that are NOT
realized tail events. The widening multipliers
(`crisis_weight=0.5`, `bear_weight=0.25`) were calibrated against
UNH 2024-11 and the AAPL 2026-02 control — both genuine tail
periods — but the *frequency* of widening firing in non-tail
periods was not measured before the calibration.

The result: widening pulls ev_dollars down on a per-ticker basis
that's orthogonal to forward returns. Per-ticker bias creates
ranking shifts → signal degradation → eventual sign flip.

### Why naïve fixes don't recover the signal

Tried during the rollback investigation:

1. **Worst-of-two adding overlapping**: makes the signal MORE
   degraded (S27 ρ from −0.094 to −0.145) because the larger
   overlapping sample is even more sensitive to mean-shift
   artifacts than NOS.
2. **Single-evaluation B1-only**: still inverted (ρ = −0.094).
3. **Raising the fire threshold** (e.g., widening_factor > 1.20):
   would skip the COST 2022-04 case (factor 1.09), losing the
   partial close.
4. **Halving the weights** (crisis 0.5 → 0.25, bear 0.25 →
   0.125): would not close UNH 2024-11 (factor 1.16 vs needed
   1.32 for the prob_profit drop).

The named F4 cases (UNH at 0.27 crisis + 0.72 bear, COST at 0.14 +
0.09) need DIFFERENT calibration from the broader window, where the
HMM mostly reports 0.1–0.4 crisis + 0.1–0.3 bear on calm periods.
A static multiplier cannot satisfy both.

### What a correct fix needs

A working F4 fix must:

1. Distinguish "true tail-regime" from "moderate elevated vol"
   beyond the HMM's K=4 label alone. Candidate signals:
   - Ticker-specific 30-day realized vol vs. its 1y baseline
     (idiosyncratic regime signal).
   - VIX level / VIX term structure as a market-wide regime
     signal independent of the per-ticker HMM.
   - HMM posterior threshold (e.g., fire only when p_crisis >
     0.7) tuned against the full backtest, not against single cases.
2. Be calibrated against the **full backtest set** (S22 / S27 /
   S32 / S34 / S35) such that Spearman ρ does not drop below the
   pre-fix +0.188 baseline.
3. Close the named F4 cases (UNH, COST) without firing on the
   calm-bull plurality.

This is a multi-week research effort with explicit
"keep-ρ-non-negative" + "close-named-cases" co-objectives. Not a
single-PR fix.

### What was preserved through the rollback

- **All non-§2 work**: the `f4_baseline_driver.py` and
  `f4_baseline_2026-05-26_raw_output.txt` from PR #245 still
  reproduce the pre-fix engine exactly.
- **The diagnostic chain**: §1-9 above (root cause, lookback
  sensitivity, why Fix A failed) is unchanged.
- **The validation harness**: `tests/test_f4_tail_risk_gap.py` and
  the S27 backtest regression test continue to pin the GAP.

PR #253 converted to draft and left open as a research record.
The implementation work + verification artifacts are reachable in
the branch's `Reverted` commits for future agents to inspect.

### Status of `docs/PRODUCTION_READINESS.md` §3 Blocker B1

**Not resolved.** F4 widening attempt rolled back. The B1 blocker
remains open and requires the research-grade redesign above. No
engine code changed. Engine is at `origin/main` predictive-signal
parity.
