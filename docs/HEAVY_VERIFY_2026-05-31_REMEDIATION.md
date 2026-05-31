# Heavy-Verify Campaign 2026-05-31 — Remediation plan (triage + sequencing)

**What this is:** the actionable, sequenced plan derived from the campaign's findings.
It is **observe-and-document** — no engine fix is applied here; this routes each finding
to its correct treatment and gates the dangerous one. `engine/` remains untouched.

The six "negative findings" are **four different kinds of thing** and must not be
lumped into one "fix it" pass. The sort *is* the recommendation.

| Cat | Findings | Kind | Right action | §2-output change? | Status |
|---|---|---|---|---|---|
| **A** | I3-A risk caps off by default | Cheap wiring, high-consequence | Arm the caps | **No** (adds refusals only) | **held** by operator |
| **B** | I1 top-bin over-confidence + I3-E crash procyclicality | Real model bug (one root cause) | Structural fix, coordinated re-baseline | **Yes** (touches prob_profit→EV) | **gated** (see I9) |
| **C** | I2 bull-lag + I1 dollar-EV-isn't-a-forecast | What the strategy IS | Document / reframe; no engine change | No | **this doc** |
| **D** | I8 monthly-vs-daily drawdown | Measurement artifact | Daily marking in the reporting layer | No | **done** (I8) |
| +  | I3-D earnings-gate PIT look-ahead | Harness/data correctness | Fix gate to use as-of-known dates, or document | No | proposed |

> **Note on labels:** "dollar-EV isn't a forecast" is an **I1** finding (reconciled by
> I6-B), NOT I4. **I4 (the §2 probe) HELD — it is a positive, nothing to fix there.**

---

## A — Arm the risk caps (safe, high-consequence; operator has not greenlit yet)

I3-A: the D17 caps (sector 25% / single-name 10% / portfolio-delta / Kelly) only fire
when `require_ev_authority=True`, which the ranker/book path never sets → dormant.
Arming them is **§2-safe** (refusals/downgrades only; never touches `ev_raw`/`ev_dollars`).

**But not all four are equal (I3-B):**
- **R9 sector + R10 single-name: arm now.** Clean, additive refusals. Closes the
  174-of-73-month concentration breaches.
- **Portfolio-delta + Kelly: calibrate first.** The delta cap is $300 per $100k NAV
  ($3,000 at $1M); a single assigned 1,000-share lot is ~50× it. For a *wheel*, where
  assignment is the expected path, arming it as-calibrated would refuse essentially
  every post-assignment book. Re-calibrate (per-NAV scaling, or exclude
  expected-assignment stock) before arming.

**Process:** touches `wheel_tracker` / `wheel_runner` (decision-layer-adjacent) → goes
through the explicit-ask + decision-layer lane-claim path, not the plain docs gate.
**Status: operator held this in the triage decision — not started.**

## B — The over-confidence / procyclicality fix (GATED — do not re-baseline yet)

I1 and I3-E are one root cause: the empirical forward distribution lags the regime at
transitions, so `prob_profit` is most over-confident exactly at crisis onset. This is
an **EV-authority-output** change → a coordinated re-baseline event.

**The gate result (I9) — the demonstrated fix does NOT clear it.** I6-C's recalibration
layer looked promising (OOS ECE 3.17→1.29pp) but on a **benign 2024-2026 window**. I9's
leave-one-crisis-out / regime-holdout tests show recalibration **does not generalize to
unseen crises** (2020 crash top-bin stays 0.84-predicted vs 0.72-realized; crisis
realized rate is unstable across crises, 0.37–0.93, 56.5pp spread). **So:**
1. **Do NOT build the re-baseline around a recalibration layer.** It would polish the
   calm tape and leave the crisis over-confidence (the dangerous part) intact.
2. **The remaining candidate is structural:** wire POT-GPD (`engine/tail_risk.py`,
   currently unwired) into `prob_profit` so it responds to *current* realized-tail
   conditions, not a backward-fit label. **Untested** — needs an engine prototype.
3. **Gate the structural fix on the same I9 test** (leave-one-crisis-out generalization)
   BEFORE committing.
4. **Then bundle** the validated fix with the already-deferred **D19 (exit costs) +
   D21 (trading-bar horizon)** into ONE re-baseline cycle: re-run backtests, refresh
   the published `prob_profit` matrix, re-pin the exact-EV invariant tests, gate the
   merge. One re-baseline, not three.

**B-routing update (I10).** A scoping study tested B1 (probability fix) vs B2 (behavioral
transition gate) head-to-head. Result: **neither *naive* form clears the bar.** The
crisis instability is real (≥26pp between well-powered cells, CIs non-overlapping), but
**no simple PIT signal detects the transition cleanly** — `rv_ratio` is highest at the
2020 *recovery* (the best entry) and the 2022 bear-onset looks calm by drawdown; a crude
gate conflates onset / recovery / sustained-bear. So: (a) do NOT commission a
single-signal gate (naive B2) or the recalibration (naive B1); (b) if a fix is pursued
it is a **multi-feature *onset* detector**, acceptance = the 3-way separation under
leave-one-crisis-out (I10 P2) — a *research* task; (c) the robust default needing no
detector is a §2-clean, downgrade-only **risk-budget rule** — cap `prob_profit > 0.90`
size in elevated-vol regimes (the defensive-sleeve posture). See
`HEAVY_VERIFY_2026-05-31_I10_B1_VS_B2_SCOPING.md`.

## C — Reframe (document; not a fix — this IS the strategy)

I2 bull-lag and the I1 dollar-EV-magnitude caveat are **structural to short-premium
writing** (the CBOE PUT-index profile). "Fixing" bull-lag means changing the strategy
— on CLAUDE.md's NEVER list. The action is **framing**, recorded here as authoritative:

1. **The engine is a defensive premium sleeve, not a bull-market growth substitute.**
   It earns its keep in down/sideways/high-rate tape (2022 bear +27pp vs passive; 2020
   crash +10pp; at ~0.4–0.6× the index drawdown) and structurally lags strong bulls
   (−19 to −26pp). Size it as a complement to long equity beta.
2. **`ev_dollars` is a risk-aware ranking score, NOT a dollar-profit forecast.** It has
   ≈0 rank-correlation with realized $ P&L (I1); its value is in *selection* (top-K
   beats random, I6-B). **Stop presenting `ev_dollars` magnitude as expected profit in
   docs/dashboard.** Where a ranking/headline number is shown, prefer `prob_profit`
   (best risk-adjusted selector) or `ev_roc`, and label `ev_dollars` as a
   tail-risk-adjusted score.

**Propagation status (applied 2026-05-31):**
- `PROJECT_STATE.md`: **DONE** — added a "what the engine is / isn't" defensive-sleeve
  framing block (+ the `ev_dollars`-is-a-ranking-score-not-a-profit-forecast note + a
  pointer to this campaign) at the top of the file, after the deployment-gate warning.
- `dashboard/`: **N/A — checked, already clean.** The dashboard surfaces a 0–100
  composite `score` (`trade.score.toFixed(1)`) plus mechanical `premium`/`maxProfit`; it
  does **not** render `ev_dollars`/`ev_roc`/`prob_profit` magnitude as expected profit
  anywhere. So Category C's display-side concern does not apply to the current UI — there
  is nothing to relabel. (If a future dashboard iteration adds an EV-dollar column, label
  it "tail-adjusted EV score, not a profit forecast" and prefer `prob_profit`/`ev_roc`
  for sort.)

## D — Daily marking in the reporting layer (DONE)

I8 delivered this: `i8_daily_risk.py` reproduces the canonical backtest book and marks
it daily, giving the true intra-month drawdown (crash −20.6% vs the −2.6% monthly
figure). **The honest risk view already exists.** Optional follow-up: make daily-mark
the default risk output of the canonical backtest harness (`backtests/regression/`,
observe-side). No engine change.

## I3-D — Earnings-gate PIT look-ahead (harness/data correctness)

The event gate reads a static calendar of *realized* `announcement_date`s and filters
to `> as_of`, so for a historical `as_of` it can use a date that wasn't yet known.
Conservative direction (gate can only *remove* candidates → cannot inflate returns),
but per-month candidate counts aren't strictly PIT. **Fix:** source the gate from an
as-of-known *scheduled* earnings calendar (or snapshot the calendar by vintage); or
accept-and-document. Low priority; observe-side. No EV-output effect.

---

## Sequenced bottom line
1. **Now, safe, observe-side:** this doc (C framing) + I8 (D, done) + the I3-D note.
2. **Soon, §2-safe but decision-adjacent (operator greenlight + lane-claim):** arm
   R9/R10. Calibrate delta/Kelly before arming them.
3. **Gated, do NOT start until validated:** the B structural fix — prototype POT-GPD
   into `prob_profit`, pass the I9 leave-one-crisis-out test, THEN bundle with D19+D21
   in a single EV-authority re-baseline.

Nothing in categories A or B is started; everything here is documentation. The
campaign's mandate — observe-and-document, never fix under the same pass — is intact.
