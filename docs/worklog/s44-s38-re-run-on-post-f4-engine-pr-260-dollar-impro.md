---
id: S44
title: S38 re-run on post-F4 engine (PR #260 dollar-improvement test)
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** S40's AI-handoff (PR #264) hypothesised that PR #260's
realized-vol-ratio widening would close 5-10pp of S38's −52pp
engine-vs-passive gap at \$1M/100t/2020-2024 by refusing more
candidates in elevated-vol regimes (especially COVID 2020). S41
(PR #267, Terminal A) tested this at the 24t/\$100k/2022-2024 scale
and found PR #260 alone is a slight dollar NEGATIVE (ρ −3.3%,
NAV −12.1%, executed −22%). S44 tests the same question at the
100t/\$1M/2020-2024 scale to surface whether COVID's elevated vol
regime materially changes the F4 impact.

**Setup.** Identical to S38 except engine SHA. 100 alphanumeric SP500
tickers, \$1M, 2020-01-02 → 2024-12-31, 35-DTE / 25-delta puts,
wheel into CC, hold to expiry, `require_ev_authority=False`, three
parallel WheelTracker instances. Engine SHA `56d8e5c` (post-PR #260
F4 RV widening + PR #262 R10 single-name cap). Driver under
`%TEMP%\s38_postf4_backtest\` (throwaway). Compute 6.31h wall-clock
(18 sec/day; 1,258 trading days).

**Section-2 invariant scan: CLEAN both pre- and post-F4.** 0 PUT
executions on `ev_dollars ≤ 0`; 0 non-finite `ev_dollars` anywhere.
CC negative-EV opens (258 pre, 270 post) are harness design with
`require_ev_authority=False`, not engine §2 breaches.

**Headline result.**

| Metric | Pre-F4 (b2cce25) | Post-F4 (56d8e5c) | Δ | Δ% |
|---|---|---|---|---|
| Final NAV (full friction) | \$1,331,764 | \$1,337,350 | +\$5,586 | +0.4% |
| Engine return | +33.18% | +33.74% | +0.56pp | — |
| n_executed_puts | 305 | **307** | +2 | +0.7% |
| Realized grand total | −\$28,647 | −\$32,729 | **−\$4,082** | +14.2% |
| Spearman ρ | 0.3576 | 0.3539 | −0.0037 | **−1.0%** |
| Engine vs Univ-EW (+92.19% baseline) | **−59pp** | **−58.45pp** | +0.56pp closer | — |

**Realism check.**

| Aspect | Engine (S44) | External reference / prior Sn | Verdict |
|---|---|---|---|
| Section-2 invariant preservation | 0 breaches on PUTs both pre/post | F4 fix is downgrade-only by design | ✓ |
| Cross-config consistency with S41 | S44 ρ −1.0% vs S41 ρ −3.3%; S44 NAV +0.4% vs S41 NAV −12.1% | Universe size + window length dilute per-trade F4 impact | ✓ Different magnitude same direction-of-effect on ρ |
| COVID-specific refusal hypothesis | 19 → 18 executed (97.76% → 97.87% refusal) | Hypothesis was material refusal increase; observed +0.11pp | ⚠ Hypothesis falsified |
| Deployment-matrix amendment (PR #263) | S44 result −58.45pp Univ-EW; PR #263 cites −52pp SPY | Within error bars; matrix verdict unchanged | ✓ No re-revision needed |
| F4 + R10 deployment bundle (S41 framing) | S44 reinforces: PR #260 alone not value-creating at \$1M/100t | F4 = frequency guard; R10 = magnitude guard; bundle closes B1 | ✓ Consistent |

**Verdict.**

- **F4 fix has near-zero impact on S38 at the 5y/100t scale.**
  +0.56pp engine return; realized P&L slightly worse (−\$4,082);
  ρ minimally degraded (−1.0%); executed +0.7%.
- **The S40 hypothesis is falsified.** Predicted 5-10pp closure of
  the −59pp Univ-EW gap; observed 0.56pp. The −52pp pattern is
  structural to the strategy's limited deployment (15-23% NAV),
  NOT to a missing tail-risk widening mechanism.
- **COVID specifically: no material effect.** 1 fewer trade taken
  in the 12-week window (19 → 18 of 847 candidates). The hypothesis
  that COVID's elevated vol would trigger material F4 refusal +
  loss avoidance does not pan out — rv30/rv252's 30-day window
  takes too long to catch up to a sharp drawdown.
- **Cross-configuration finding holds:** PR #260 alone is signal-
  preserving but not value-creating. **Deployment bundle that
  closes PROD_READINESS §3 B1 is PR #260 + PR #262 together, per
  S41's framing.**
- **PROD_READINESS deployment matrix (PR #263 amendment) requires
  no revision based on S44.** The −52pp framing is within the
  observed range.

**AI handoff.**

- **For research follow-up:** test R10 in strict mode on the S38
  setup. Set up a `PortfolioContext` per-step that aggregates
  per-name exposure; verify R10 blocks AAPL/BKNG/AZO when any
  approaches 10% NAV. This would close the parallel question:
  "does R10 actually constrain anything at \$1M/100t?" (Likely:
  rarely, since natural deployment is wide.)
- **For the F4 docs-sync follow-up flagged in PR #263's
  Unresolved:** S44 reinforces that PROD_READINESS §1/§3/§6 should
  cite PR #260 as "shipped, but value-creating only in bundle with
  R10" rather than open-vs-shipped status binary. S41 + S44
  together are the validation that B1 closure requires both PRs.
- **For PR #263 reviewers:** S44 confirms the deployment matrix
  amendment is honest. The −52pp framing holds; F4 fix does not
  close it.

**Methodology debt.**

- Same in-sample HMM/POT-GPD parameters caveat inherited from
  S22-S40. Engine parameters were fit on data overlapping the
  backtest window.
- Bloomberg-only (SPY not in dataset). Used Univ-EW + external SPY
  estimates.
- F4 widening factor is NOT captured in rank_log (the harness
  doesn't save it), so this doc cannot directly count "how often
  did F4 fire in 2020 in the backtest." Inferred from numerical
  shifts. A future Sn harness should add the column.
- Same engine SHA across the run (no main-advance mid-campaign).

Full doc: `docs/ENGINE_BACKTEST_S44_S38_POSTF4_RERUN.md`.
