# Smart Wheel Engine — Institutional Audit Report

**Date:** 2026-04-06  
**Reviewer posture:** Quant/risk/software production audit (institutional standards)

---

## Executive Summary

Smart Wheel Engine is a strong, unusually feature-rich wheel-strategy platform with impressive breadth: options pricing (including higher-order Greeks and American approximations), multi-method risk, Monte Carlo engines, stress testing, portfolio lifecycle tracking, committee-style trade review, and a significant test suite. The architecture and module coverage are materially ahead of most retail-grade wheel tools. However, several issues prevent immediate "institutional-grade" launch readiness: inconsistencies in event calendars, unit-consistency risk in vega-based stress attribution, simplified/heuristic assumptions where claims suggest stronger rigor (e.g., LSM basis labeling, SVI arbitrage enforcement, TWR/FIFO semantics), and dependency/collection hygiene gaps in full test discovery. The product is a high-potential foundation that needs targeted hardening before launch.

---

## Strengths (Top Competitive Advantages)

1. **Depth of quantitative stack**: BSM, BAW American approximation, full first/second/third-order Greeks, IV inversion with Newton + Brent fallback are all implemented in a cohesive pricing module.

2. **Risk architecture breadth**: Multiple sizing methods, portfolio Greeks, parametric/historical/covariance/Monte Carlo VaR pathways, sector/correlation/drawdown controls, and stress scenario framework provide layered defense.

3. **Lifecycle completeness for wheel**: Dedicated wheel state machine, transaction-cost modeling, assignment handling, and mark-to-market liability accounting are present.

4. **Strong simulation suite**: Block bootstrap + jump diffusion + LSM provides multiple lenses for robustness and assignment risk.

5. **Solid software quality signal**: Large automated test base with strong pass rates for core quant/risk modules.

6. **Differentiated decision layer**: Multi-persona advisor committee and confidence/rationale outputs are distinctive versus typical execution-only wheel engines.

7. **Data governance intent**: Feature store includes lineage, metadata, and point-in-time filtering interfaces.

---

## Weaknesses / Gaps

1. **Event-calendar consistency drift**: Different subsystems encode conflicting 2026 FOMC dates (engine calendar vs macro calendar), creating operational/event-filter risk.

2. **TWR methodology is simplified**: Return calculation adjusts only net flows over period rather than true subperiod geometric linking around each external flow.

3. **FIFO claim mismatch**: Sell-side realized P&L uses average cost basis while comments/documentation call it FIFO.

4. **SVI arbitrage controls are partial**: Butterfly check warns but does not enforce hard constraints; no explicit calendar no-arbitrage optimization across expiries.

5. **LSM documentation/implementation mismatch**: Code comments reference Laguerre basis but implementation uses simple polynomial basis.

6. **Stress calibration mismatch vs stated severe-history framing**: Default "historical" shocks are relatively mild versus canonical event magnitudes in the prompt.

7. **Full-suite test collection not clean in this environment**: Collection fails on missing `pydantic`, reducing confidence in plug-and-play reproducibility.

---

## Critical Issues (Must-Fix Before Launch)

### 1. Calendar Integrity (Must Fix Immediately)

- Align all FOMC/CPI/NFP calendars from one authoritative source pipeline and enforce single source of truth.
- Current inconsistency can produce false trade blocks/permits.

**Specific discrepancy found:**
- `engine/event_calendar.py` hard-codes 2026 FOMC dates including **Nov 4 / Dec 16**
- `financial_news/calendar/macro_calendar.py` uses **Oct 28 / Dec 9**
- These conflicting dates create scheduling behavior mismatches across subsystems

**Reference sources for validation:**
- Federal Reserve (FOMC): https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- BLS 2026 schedule (CPI/NFP): https://www.bls.gov/schedule/2026/

### 2. Vega Unit Normalization Across Risk/Stress Paths

- Standardize whether vega is per 1 vol point or per 100 vol points and enforce via typed wrappers/tests.
- Inconsistent conventions can materially under/over-state stress P&L.

**Specific finding:** Vega is defined per 1% vol in pricer; stress paths include vega P&L formulas that should be unit-audited for consistency across modules.

### 3. Performance-Metric Correctness Claims

- Either implement true GIPS-style TWR subperiod chaining and true FIFO lot accounting, or relabel methods to avoid model-risk from incorrect reporting.

**Specific findings:**
- Portfolio tracker labels realized P&L as FIFO but computes using average cost basis
- TWR is implemented as a single flow-adjusted period return rather than full subperiod chain-linking

### 4. Arbitrage Governance for Vol Surface

- Promote arbitrage checks from warnings to constraints/rejections where needed; add calendar-arbitrage checks across tenors.

**Specific finding:** SVI butterfly check warns but does not enforce hard no-arbitrage rejection, and calibration is unconstrained beyond box bounds.

### 5. Release-Quality Dependency Hygiene

- Ensure required libs are pinned/installed for clean full test collection in baseline CI/dev environment.

---

## Recommendations (Prioritized)

### P0 (Launch Blockers)

1. **Unify macro calendar subsystem** with signed data source + regression tests against official 2026 releases.

2. **Create "Greeks unit contract"** document + assertions; add cross-module tests specifically for vega/rho scaling consistency.

3. **Fix accounting semantics**: implement true FIFO lots and true flow-segmented TWR.

4. **Harden vol-surface no-arb**: explicit butterfly + calendar arbitrage constraints in calibration acceptance.

### P1 (High Value, Near-Term)

5. **Recalibrate stress presets** to historically grounded joint shocks (spot/vol/rates/corr) and document methodology.

6. **Clarify LSM basis language** (or switch to actual Laguerre basis with orthogonalization options).

7. **Add model validation notebooks**: benchmark BAW/LSM/IV solver vs reference libraries and published cases.

### P2 (Medium-Term)

8. **Regime-aware correlation engine** (rolling + shrinkage + crisis override) for VaR and sizing.

9. **Formal model risk controls**: challenger models, drift monitors, and periodic recalibration policy.

---

## Model/Implementation Gaps Detail

### LSM Basis Mismatch
- **Claim:** Comments say "Laguerre basis"
- **Reality:** Implementation uses simple polynomial powers (`x**d`)
- **Impact:** Naming confusion; potential numerical stability differences vs true orthogonal basis

### SVI Arbitrage Controls
- **Claim:** Arbitrage-free vol surface
- **Reality:** Butterfly check warns but does not enforce hard constraints; calibration uses box bounds only
- **Impact:** Potential for arbitrageable surfaces in production

### TWR/FIFO Semantics
- **Claim:** FIFO realized P&L, TWR performance
- **Reality:** Average cost basis for P&L; single-period flow adjustment for TWR
- **Impact:** Audit/compliance risk if reported as true FIFO/GIPS-TWR

---

## Detailed Scoring

| Category | Weight | Score | Weighted |
|---|---:|---:|---:|
| Mathematical Correctness | 25% | 82 | 20.50 |
| Software Engineering Quality | 15% | 85 | 12.75 |
| Risk Management Robustness | 20% | 80 | 16.00 |
| Production Readiness | 15% | 74 | 11.10 |
| Feature Completeness | 15% | 91 | 13.65 |
| Competitive Differentiation | 10% | 88 | 8.80 |
| **Overall** | **100%** |  | **82.80 / 100** |

### Score Justification

- **Math (82):** Core formulas and safeguards are strong, but important consistency/rigor caveats exist (LSM basis labeling, arbitrage enforcement depth, unit risks).

- **Engineering (85):** Good modularity and extensive tests; some quality debt (duplicate lines in verification helper section and dependency fragility in full collection).

- **Risk (80):** Broad toolkit and sensible limits; needs tighter parameter governance and unit-standardization.

- **Production (74):** Close, but calendar consistency + accounting semantics + dependency hygiene are launch-critical.

- **Completeness (91):** Excellent end-to-end wheel coverage including lifecycle and analytics.

- **Differentiation (88):** Advisor committee + integrated quant/risk/news stack is a notable moat.

---

## Testing Summary

**Passed:**
```
pytest -q tests/test_option_pricer.py tests/test_risk_manager.py tests/test_monte_carlo.py tests/test_stress_testing.py tests/test_portfolio_tracker.py tests/test_wheel_cycle.py tests/test_signals.py tests/test_regime_detector.py tests/test_advisors.py
```

**Blocked:**
```
pytest --collect-only -q tests
```
Collection blocked by missing `pydantic` for `tests/test_bloomberg_loader.py` and `tests/test_data_pipeline.py`.

---

## Final Verdict

**Promising and strong, but not yet institutional-launch ready.**

Current state fits the **"Strong. Minor-to-moderate improvements needed before launch"** band at **82.8/100**. If the P0 items above are resolved with evidence-backed validation, this can credibly move into upper-80s readiness and become one of the more compelling quantitative wheel engines in market.

---

## How to Reach 100/100 (Concrete Upgrade Plan)

To earn a true **100/100 institutional readiness score**, the bar is not "more features"; it is **verified correctness + operational resilience + governance evidence**. The following must be delivered and independently verified:

### 1. Calendar & Event Integrity (Required)

- Build a single source-of-truth event service consumed by both `engine/event_calendar.py` and `financial_news/calendar/macro_calendar.py`.
- Auto-ingest official calendars (Fed/BLS) with checksum/versioning and alert on drift.
- Add regression tests that fail if 2026 FOMC/CPI/NFP dates diverge from approved golden files.
- **Acceptance standard:** Zero cross-module calendar discrepancies for supported years.

### 2. Greeks Unit Contract + Risk P&L Consistency (Required)

- Define one canonical convention for all Greeks (especially vega/rho scaling) and enforce via typed wrappers.
- Add cross-module invariants so stress P&L, risk manager VaR approximations, and pricing greeks all agree on units.
- Add finite-difference backchecks for stress ladder decomposition (delta/gamma/vega/theta/rho) across spot/vol/rate shocks.
- **Acceptance standard:** Max decomposition error < 1% of full revaluation for test scenarios.

### 3. Accounting Correctness (Required)

- Implement true lot-level FIFO realization (not average-cost shorthand).
- Implement true GIPS-style TWR with geometric chain-linking between each external cash-flow break.
- Add deterministic fixture tests covering deposits, withdrawals, dividends, partial sells, and multi-lot exits.
- **Acceptance standard:** Audit-traceable reconciliation to independent calculator on all fixture cases.

### 4. Vol Surface No-Arbitrage Hardening (Required)

- Enforce butterfly and calendar no-arbitrage constraints during SVI calibration (reject/repair invalid fits).
- Add wing-behavior guardrails and monotone term-variance checks across expiries.
- Add stress-time recalibration diagnostics with fail-safe fallbacks.
- **Acceptance standard:** 0 arbitrage violations on calibration benchmark set.

### 5. Model Validation & Benchmarking (Required)

- Produce model validation pack:
  - BSM/Greeks vs textbook/known values
  - BAW vs high-step binomial benchmark grid
  - LSM vs binomial for American options (with/without dividends)
  - IV solver robustness across moneyness/tenor grids
- **Acceptance standard:** Predefined error tolerances met and documented with reproducible notebooks.

### 6. Production Reliability & SDLC Controls (Required)

- Resolve dependency hygiene so full suite installs/collects cleanly in CI and local bootstrap.
- Expand CI matrix (Python versions + OS where relevant), include mutation/property tests for quant invariants.
- Add SLOs/alerts for pipeline freshness, calendar ingestion, and risk calc latency.
- **Acceptance standard:** Green CI for full test suite, reproducible env build, and defined on-call runbooks.

### 7. Governance, Model Risk, and Operational Controls (Required)

- Establish formal model risk policy: versioned assumptions, challenger models, periodic recalibration, decommission criteria.
- Add pre-trade and post-trade controls with hard risk kill-switches and change-management approvals.
- Add incident simulation (tabletop + replay drills) for data outage, stale vol, bad calendar, and extreme market gap.
- **Acceptance standard:** Signed governance artifacts and successful control-drill outcomes.

### Suggested Milestone Targets

- **Phase A (2-4 weeks):** Calendar unification, Greeks unit contract, dependency fixes, CI hardening.
- **Phase B (3-5 weeks):** FIFO/TWR correctness, SVI no-arb enforcement, stress decomposition validation.
- **Phase C (3-6 weeks):** Full model-validation pack, governance controls, reliability drills.

If all acceptance standards above are met with evidence, the system can justify a **90-95+** institutional readiness score; reaching a strict **100/100** requires sustained live-operational proof (stability + control effectiveness), not only code completeness.

---

*End of Audit Report*
