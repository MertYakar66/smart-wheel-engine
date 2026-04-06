# Smart Wheel Engine — Institutional Audit Report

**Audit date:** 2026-04-06  
**Scope:** Engine, advisors, data pipeline, backtests, dashboard, tests, docs  
**Auditor stance:** Independent code-first review (implementation over claims)

---

## Executive Summary

The Smart Wheel Engine is a strong quantitative prototype with several genuinely institutional design elements (multi-method VaR, American option approximation, regime-aware sizing, advisor committee, and broad test surface). It is **not yet institutional-launch ready** due to a cluster of P0 gaps concentrated in unit consistency, accounting methodology, event-calendar source-of-truth, and test reproducibility/environment hardening.

### Overall assessment
- **Strength:** The architecture demonstrates unusually broad quantitative ambition for a wheel framework (option Greeks through 3rd order, LSM, jump diffusion, covariance/MC VaR, scenario engines).
- **Risk:** Multiple modules still contain "research-grade" simplifications and inconsistent conventions that can materially alter live P&L and risk interpretation.
- **Verdict:** Promising platform requiring targeted remediation before institutional deployment.

### Top 3 strengths
1. **Breadth of quantitative tooling** across pricing, risk, stress, and simulation.  
2. **Risk infrastructure depth** (parametric/historical/covariance/MC VaR, CVaR, correlation-aware logic).  
3. **Strong CI structure and large test corpus** including property-based and anti-lookahead tests.

### Top 3 critical issues
1. **Unit and decomposition inconsistencies** (theta/vega handling and documentation mismatches in stress/risk paths).  
2. **Portfolio accounting limitations** (average-cost realization and simplified non-GIPS TWR while docs/claims imply institutional rigor).  
3. **Calendar/event governance gap** (hardcoded FOMC years only; CPI/NFP schedule integrity not centralized).

---

## Detailed Findings by Module

## A) Option Pricing Engine (`engine/option_pricer.py`)
### What works well
- Black-Scholes-Merton with continuous dividend yield is implemented in standard form.
- Deterministic edge handling exists for `T<=0` / `sigma<=0`.
- Full Greek stack implemented in one pass (`black_scholes_all_greeks`).
- IV solver uses Newton-Raphson with Brent fallback.
- Includes BAW-style American approximation and finite-difference American Greeks.

### What needs improvement / risks
- **Vega docstring convention confusion:** function returns `.../100` but text says "per 1% change" then "divide by 100 for 1 vol point"; this can mislead downstream consumers.
- **Sigma=0 American behavior simplification:** immediate intrinsic returned for both call/put; deterministic carry-consistent handling is not symmetric with no-dividend call logic.
- **BAW sensitivity:** critical-price Newton derivatives are complex; no direct benchmark calibration test against reference trees across moneyness/rates/dividend corners in this file.

### Missing pieces
- No explicit finite-difference audit harness for 2nd/3rd order Greeks in production code path.
- No formal numeric guardrails for very small `T`/`sigma` beyond basic branch logic.

---

## B) Risk Management (`engine/risk_manager.py`)
### What works well
- Kelly criterion implementation follows canonical form and uses half-Kelly + cap.
- Greeks aggregation correctly uses sign and contract multiplier.
- VaR stack includes parametric, historical, covariance, and Monte Carlo methods.
- Correlation matrix PSD repair and Cholesky/eigendecomp fallback are implemented.

### What needs improvement / risks
- Parametric VaR currently uses aggregate proxy exposures and simplified assumptions for cross-effects.
- Historical VaR uses equal-weighted return proxy for multi-asset data when per-position mapping unavailable.
- Gamma/CVaR adjustment choices are heuristic and not uniformly documented as model assumptions.

### Missing pieces
- Stronger model-risk controls for method selection (auto-block risky approximations for concentrated books).
- More explicit confidence-level and horizon governance in a policy layer.

---

## C) Monte Carlo Simulation (`engine/monte_carlo.py`)
### What works well
- Fixed-block and stationary bootstrap implemented with configurable block length.
- Jump-diffusion uses compensated drift and compound Poisson jumps.
- LSM workflow (path generation, ITM regression, backward exercise) is present.
- Seeding strategy supports reproducibility.

### What needs improvement / risks
- **LSM basis mismatch:** comments and function label mention Laguerre, but implementation uses plain polynomial basis.
- Bagholder metrics depend on path interpretation assumptions (assignment timing proxy) not fully parameterized.

### Missing pieces
- Regression diagnostics (conditioning, basis selection by AIC/BIC, monotonic boundary smoothing).
- Benchmarking against binomial/finite-difference American prices across stress grid.

---

## D) Volatility Surface (`engine/volatility_surface.py`)
### What works well
- SVI parameters are bounded and butterfly condition is enforced (error or warning depending on strict mode).
- Variance-space expiry interpolation is correctly preferred over direct IV interpolation.
- SLSQP calibration with constraints exists.

### What needs improvement / risks
- **Calendar-arbitrage enforcement absent** across expiries (monotonic total variance in time not globally checked).
- Extrapolation policy outside known expiries is simplistic (first/last expiry reuse).

### Missing pieces
- Global surface no-arbitrage validation (butterfly + calendar jointly).
- Calibration robustness diagnostics (fit error decomposition by strike buckets).

---

## E) Stress Testing (`engine/stress_testing.py`)
### What works well
- Broad scenario library (historical + hypothetical) and custom scenario support via `Scenario` inputs.
- Full repricing and Greeks decomposition interfaces are available.

### What needs improvement / risks
- **Theta decomposition bug risk:** annual theta multiplied by days without explicit `/365` conversion in ladder path.
- `correlation_shock` is stored in scenario schema but not actually integrated into repricing path.
- Historical magnitudes are static constants and not linked to versioned calibration dataset.

### Missing pieces
- Reconciliation report proving Greek-sum vs full repricing error bounds by shock size.
- Crisis-correlation engine ("toward 1") actually used in portfolio-level scenario path.

---

## F) Portfolio Tracker (`engine/portfolio_tracker.py`)
### What works well
- Rich transaction type support and snapshot-based analytics pipeline.
- Realized/unrealized tracking and high-water-mark drawdown logic present.

### What needs improvement / risks
- **Realized P&L uses average cost only** (explicitly noted), not FIFO/LIFO lot accounting.
- **TWR is explicitly simplified and non-GIPS** despite institutional positioning.
- Sharpe uses excess return with hardcoded 4% risk-free assumption rather than configurable curve.
- Claimed CAGR/Calmar metrics are not fully exposed in implementation.

### Missing pieces
- Tax-lot ledger with selectable FIFO/LIFO/Specific-ID.
- True chain-linked subperiod GIPS TWR around each external flow.

---

## G) Wheel Lifecycle (`engine/wheel_tracker.py`)
### What works well
- State machine captures main wheel cycle transitions.
- Assignment basis includes premium and assignment fee adjustment.
- Transaction-cost model integrated, including commission and assignment fee assumptions.

### What needs improvement / risks
- Margin check uses simplified approximations (e.g., underlying approximation at strike during entry).
- Lifecycle regression tests are thin; one file in `tests/test_wheel_cycle.py` is script-like print flow.

### Missing pieces
- Stronger invariants around every transition (especially overlapping expiries and partial closes).

---

## H) Advisor Committee (`advisors/`)
### What works well
- Distinct personas, structured judgments, confidence scale, and committee majority logic.
- Scorecard includes precision/recall/F1 and calibration constructs.

### What needs improvement / risks
- Simons expected-value significance uses highly simplified SE proxy; t-stat interpretation is heuristic.
- Hardcoded ticker quality/speculative sets and sector maps require governance lifecycle.
- Concentration thresholds are encoded in helper heuristics rather than centralized policy config.

### Missing pieces
- Periodic retraining/calibration protocol and champion/challenger framework.

---

## I) Event Calendar (`engine/event_calendar.py`)
### What works well
- Event model and query/filter interfaces are coherent.
- Position-span event checks (symbol + macro) are implemented.
- 2026 FOMC dates in code match Federal Reserve schedule.

### What needs improvement / risks
- CPI/NFP release schedules are not generated/maintained as first-class calendar sources.
- Source-of-truth remains partially hardcoded by year for FOMC.

### Missing pieces
- Centralized external calendar ingestion + validation + automatic yearly rollover.

---

## J) Signal Generation (`engine/signals.py`)
### What works well
- Modular signal classes, composability, and configurable thresholds through constructors.

### What needs improvement / risks
- Several defaults remain hardcoded at class-init level, with no global policy registry.
- Lookahead safety depends entirely on upstream context construction.

### Missing pieces
- Signal provenance metadata (as-of timestamps, lag assertions) for backtest auditability.

---

## K) Regime Detection (`engine/regime_detector.py`)
### What works well
- Volatility/trend/term-structure regime decomposition is clear.
- Position-size multipliers are explicitly tied to detected regimes.

### What needs improvement / risks
- Regime taxonomy differs from requested labels (uses `ELEVATED/HIGH` vs exact `HIGH_VOL/NORMAL` naming).
- Potential bias control is external: detector assumes supplied series are properly time-trailing.

### Missing pieces
- Regime drift monitoring and confusion-matrix tracking against realized outcomes.

---

## L) Data Pipeline (`data/`)
### What works well
- Feature store includes as-of filtering hooks and lineage metadata.
- Data quality framework has contracts, schema checks, ranges, and anomaly checks.
- Orchestrator supports DAG stages, retries, checkpoints, and metrics summaries.

### What needs improvement / risks
- Operational dependency/environment drift: local test run failed due missing `pydantic` in current runtime despite requirement declarations.
- Some observability claims rely on logger patterns rather than unified tracing backend wiring across all tasks.

### Missing pieces
- Enforced deployment parity checks (runtime package validation gate before tests/jobs).

---

## M) Backtesting (`backtests/simulator.py`, `backtests/walk_forward.py`)
### What works well
- Wheel state progression is wired into the backtester.
- Explicit limitations are documented in simulator header.
- Walk-forward framework includes anchored/rolling/purged split logic with embargo support.

### What needs improvement / risks
- Simulator admits placeholder assumptions (constant IV reconstruction, simplified fills/slippage).
- Backtest realism is constrained by synthetic valuation approach until richer option history is integrated.

### Missing pieces
- Full option chain historical mark-to-market engine and borrow/assignment microstructure detail.

---

## N) Dashboard (`dashboard/quant_dashboard.py`)
### What works well
- Exposes pricing, Greeks, IV solving, and portfolio/risk integration paths.

### What needs improvement / risks
- Input validation appears thin for invalid user payloads and state mismatches.
- Kelly and risk views depend on externally supplied context quality.

### Missing pieces
- Defensive schema validation at dashboard boundary and clearer error surfaces.

---

## O) Test Coverage (`tests/`)
### What works well
- Large suite with unit + property + anti-lookahead-style tests in repository.
- CI workflow includes broad coverage and dedicated quant test stages.

### What needs improvement / risks
- Local run did not complete due environment dependency issue (`pydantic` missing).
- Some lifecycle tests are script-like and assertion-light.
- Published "170+ passing" claim is stale relative to current suite size and should be version-stamped.

### Missing pieces
- Mandatory coverage publication artifact per build and seeded stochastic replay reports.

---

## P) Documentation (`docs/`)
### What works well
- Governance and model-card structure is present.
- Greeks unit contract is explicit and useful.

### What needs improvement / risks
- Some docs assert institutional status ahead of current implementation gaps.
- Unit wording (especially vega/theta conventions) should be harmonized with all code comments.

### Missing pieces
- Living model inventory with owner, validation cadence, and retirement criteria per module.

---

## Cross-Cutting Concerns

## 1) Mathematical Correctness
- Core BSM formulas look consistent and dividend-adjusted.
- Advanced formulas (BAW, 3rd-order Greeks, LSM) are present but require tighter independent benchmark harnesses.

## 2) Unit Consistency
- Canonical Greek contract exists, but stress decomposition and docstrings are not uniformly aligned.
- Theta day/year conversions are a recurring fragility point.

## 3) Numerical Stability
- Many guardrails exist (`max`, floors, deterministic branches), but extreme parameter regime testing is incomplete.

## 4) Lookahead Bias
- Framework includes anti-lookahead structures (as-of filtering, walk-forward embargo), but enforcement depends on caller discipline.

## 5) Production Readiness
- CI/CD is present and substantial.
- Packaging/dependency and runtime parity controls need tightening (local environment mismatch observed).

---

## Competitive Analysis (2026 snapshot)

### Relative to OptionAlpha / tastytrade / ORATS / QuantConnect
- **Where Smart Wheel Engine is stronger:** code-level extensibility, custom risk math, advisor committee architecture, and integrated simulation stack in one codebase.
- **Where incumbents remain stronger:** data operations maturity, live brokerage plumbing reliability, UX robustness, and production monitoring/tooling.
- **Differentiators:** multi-persona committee + quant risk stack + wheel lifecycle tracking in a unified internal framework.

---

## Prioritized Remediation Plan

## P0 — Must fix before launch
1. Enforce unit-consistent Greeks decomposition (theta/vega) with invariant tests.
2. Implement institutional accounting modes (FIFO/LIFO/lot ledger + GIPS TWR chain-linking).
3. Centralize event calendar source-of-truth and auto-refresh (FOMC/CPI/NFP with validation).
4. Promote environment parity gate to block runs when required dependencies are missing.

## P1 — Fix within 30 days post-launch
1. Calendar-arbitrage checks in vol surface across expiries.
2. Expand LSM benchmark suite and explicit basis-function governance.
3. Improve dashboard input validation and error ergonomics.
4. Add policy-config registry for advisor/signal thresholds.

## P2 — Roadmap
1. Full historical option chain MTM in backtests.
2. Regime model drift monitoring and retraining loop.
3. Enhanced scenario engine with dynamic cross-asset correlation stress.

---

## Scoring Table

| Category | Weight | Score | Weighted |
|---|---:|---:|---:|
| Mathematical Correctness | 25% | 78 | 19.50 |
| Software Engineering Quality | 15% | 82 | 12.30 |
| Risk Management Robustness | 20% | 76 | 15.20 |
| Production Readiness | 15% | 72 | 10.80 |
| Feature Completeness | 15% | 84 | 12.60 |
| Competitive Differentiation | 10% | 86 | 8.60 |
| **Final Weighted Score** | **100%** |  | **79.00 / 100** |

### Launch readiness interpretation
- **79** falls in the "promising but needs moderate remediation before launch" band.

---

## Verdict Statements

1. **This product IS NOT ready for institutional launch.**
2. **This product IS among the most promising quantitative wheel engines in the market, provided P0 remediations are completed.**
