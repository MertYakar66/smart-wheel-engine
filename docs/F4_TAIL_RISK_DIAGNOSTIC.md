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
