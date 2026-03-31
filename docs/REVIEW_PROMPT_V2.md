# Smart Wheel Engine - Comprehensive Audit Prompt (V2)

## Context

You are a **Senior Quantitative Researcher** with expertise in:
- Derivatives pricing and Greeks
- Volatility modeling (realized vol estimators, implied vol surfaces)
- Systematic options strategies (wheel strategy, CSP, covered calls)
- Statistical analysis and machine learning for trading
- Production-grade financial software engineering

You are tasked with conducting a **rigorous, critical review** of the Smart Wheel Engine - an algorithmic trading system for executing the wheel strategy (selling cash-secured puts, taking assignment, selling covered calls).

This is a **professional audit**. Your job is to find errors, not to praise. Be skeptical. Challenge assumptions. Verify calculations against academic sources.

---

## IMPORTANT: Recent Fixes Applied

The following critical issues were identified in a prior audit and have been **fixed in commit `828ca6e`**. Your task is to:
1. **Verify** each fix is correctly implemented
2. **Run behavioral tests** to confirm fixes work
3. **Identify any remaining issues** not yet addressed

### Fixes to Verify

#### Fix 1: RSI Edge Case Handling
**Location**: `src/features/technical.py` (lines 50-73)
**Fix Applied**: RSI now correctly returns:
- `100.0` when avg_loss == 0 (all gains, no losses)
- `0.0` when avg_gain == 0 (all losses, no gains)
- `50.0` only when both are zero (no movement)

**Verification Test**:
```python
from src.features.technical import TechnicalFeatures
import pandas as pd
close = pd.Series([1,2,3,4,5,6,7,8,9,10], dtype=float)  # Monotonic uptrend
result = TechnicalFeatures.rsi(close, window=3).iloc[-1]
assert result == 100.0, f"RSI should be 100.0 for uptrend, got {result}"
```

#### Fix 2: VIX Auto-Scaling (Points vs Decimal)
**Location**: `src/features/regime.py` (lines 156-161)
**Fix Applied**: `vol_regime()` now auto-detects if VIX is in point format (e.g., 20) or decimal format (e.g., 0.20) using median > 1.0 check, and normalizes to decimal before thresholding.

**Verification Test**:
```python
from src.features.regime import RegimeDetector, VolRegime
import pandas as pd
rv = pd.Series([0.2, 0.2, 0.2])
vix_points = pd.Series([20.0, 25.0, 30.0])  # Point format
result = RegimeDetector.vol_regime(rv, vix_points).tolist()
# Should be [NORMAL, ELEVATED, ELEVATED], NOT [CRISIS, CRISIS, CRISIS]
assert VolRegime.CRISIS not in result, f"VIX points wrongly classified as CRISIS: {result}"
```

#### Fix 3: IV Rank Formula Corrected
**Location**: `data/feature_pipeline.py` (line 568)
**Fix Applied**: IV Rank now uses proper formula `(IV - Min) / (Max - Min) * 100` via `VolatilityFeatures.iv_rank()` instead of percentile-count formula.

**Verification Test**:
```python
from src.features.volatility import VolatilityFeatures
import pandas as pd
iv = pd.Series([10, 30, 20, 30, 25], dtype=float)
result = VolatilityFeatures.iv_rank(iv, lookback=5).iloc[-1]
# For 25 in window [10,30,20,30,25]: (25-10)/(30-10)*100 = 75.0
assert result == 75.0, f"IV Rank should be 75.0 (min-max), got {result}"
```

#### Fix 4: Forward Label NaN Preservation
**Location**: `src/features/labels.py` (lines 193-198, 246-250, 297-299, 451-455)
**Fix Applied**: All forward-looking labels now use `.astype(float)` instead of `.astype(int)` and explicitly preserve NaN for rows with insufficient future data.

**Verification Test**:
```python
from src.features.labels import LabelGenerator
import pandas as pd
import numpy as np
price = pd.Series([100, 101, 102, 103, 104], dtype=float)
result = LabelGenerator.forward_return_binary(price, periods=2).tolist()
# Last 2 should be NaN (insufficient forward data)
assert np.isnan(result[-1]) and np.isnan(result[-2]), f"Tail should be NaN: {result}"
```

#### Fix 5: VaR Multi-Day Compounding
**Location**: `engine/risk_manager.py` (lines 385-392)
**Fix Applied**: Historical VaR now uses proper return compounding `(1+r1)*(1+r2)*...-1` via rolling product instead of simple sum.

**Verification Test**:
```python
import pandas as pd
returns = pd.Series([-0.01, 0.02])  # -1%, +2%
compound = (1 + returns).prod() - 1  # (0.99)*(1.02) - 1 = 0.0098
simple_sum = returns.sum()  # 0.01
assert abs(compound - 0.0098) < 0.0001, "Compound return should be 0.0098"
assert compound != simple_sum, "Should use compounding, not sum"
```

#### Fix 6: Division-by-Zero Guards
**Locations**:
- `src/features/vol_edge.py` (line 69): `iv_rv_zscore` - divides by `std.replace(0, np.nan)`
- `src/features/options.py` (line 52): `unusual_volume_score` - divides by `std.replace(0, np.nan)`
- `src/features/options.py` (line 201): `premium_yield` - guards `strike <= 0 or days_to_expiry <= 0`
- `src/features/labels.py` (line 163): `premium_capture_rate` - guards `premium_collected == 0`

**Verification Test**:
```python
from src.features.vol_edge import VolatilityEdge
import pandas as pd
import numpy as np
# Constant IV-RV spread means std=0
iv = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2])
rv = pd.Series([0.15, 0.15, 0.15, 0.15, 0.15])
result = VolatilityEdge.iv_rv_zscore(iv, rv, window=3).iloc[-1]
assert np.isnan(result), f"Should be NaN when std=0, got {result}"
```

---

## Project Overview

**Smart Wheel Engine** is a Python-based options trading system with:

1. **Data Pipeline** (`data/`): Bloomberg data ingestion, consolidated CSV loading, feature store with Parquet persistence
2. **Feature Engineering** (`src/features/`): 10 feature modules computing volatility, technical, options flow, regime, and ML labels
3. **Trading Engine** (`engine/`): Black-Scholes pricing, Greeks, risk management, position tracking, signals
4. **ML Models** (`ml/`): Gradient boosting for entry timing
5. **Backtesting** (`src/backtest/`, `backtests/`): Event-driven wheel strategy simulation
6. **Dashboard** (`dashboard/`): Next.js terminal UI with real-time engine data

---

## Your Review Tasks

### 1. VERIFY RECENT FIXES

Run all six verification tests above and confirm:
- [ ] RSI returns 100.0 on monotonic uptrend
- [ ] VIX points [20, 25, 30] do NOT classify as CRISIS
- [ ] IV Rank returns 75.0 for the test case (min-max formula)
- [ ] Forward labels preserve NaN in tail positions
- [ ] VaR uses compounding (0.0098), not sum (0.01)
- [ ] Division by zero returns NaN, not inf or error

### 2. MATHEMATICAL CORRECTNESS

Review all mathematical formulas for correctness. Verify against authoritative sources.

**Volatility Estimators** (`src/features/volatility.py`):
- [ ] Close-to-close realized volatility: Is annualization factor correct (sqrt(252))?
- [ ] Parkinson volatility: Verify factor `1/(4*ln(2))` - cite source
- [ ] Garman-Klass volatility: Verify formula `0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2`
- [ ] Yang-Zhang volatility: Verify k coefficient formula `0.34/(1.34 + (n+1)/(n-1))`
- [ ] IV Rank: Verify `(IV - Min)/(Max - Min) * 100` implementation
- [ ] IV Percentile: Verify percentile rank formula (count below / total)

**Options Pricing** (`engine/option_pricer.py`):
- [ ] Black-Scholes with continuous dividend yield (Merton model): Verify d1, d2 formulas
- [ ] Delta formula: `exp(-qT) * N(d1)` for calls - correct?
- [ ] Gamma formula: Verify the divisor `S * sigma * sqrt(T)`
- [ ] Theta formula: Verify all three terms for calls and puts
- [ ] Vega scaling: Returns vega per 1% (divided by 100) - is this standard?
- [ ] Edge cases: T=0, sigma=0 handling

**Risk Metrics** (`engine/risk_manager.py`):
- [ ] Kelly Criterion: `f = (p*b - q) / b` where b = win/loss ratio - verify
- [ ] VaR parametric: Delta-normal approximation assumptions
- [ ] VaR historical: Verify compounding implementation is correct
- [ ] CVaR calculation: Is expected shortfall computed correctly?

**Technical Indicators** (`src/features/technical.py`):
- [ ] RSI: Uses Wilder's smoothing (EMA span = 2*window - 1) - verify equivalence
- [ ] RSI edge cases: Verify 100/0/50 behavior for all-gains/all-losses/no-movement
- [ ] MACD: Standard 12/26/9 parameters and EMA calculation
- [ ] Bollinger Bands: +/-2 sigma from 20-day SMA
- [ ] ATR: True Range definition and smoothing method
- [ ] Hurst Exponent: Verify R/S analysis implementation

**Regime Detection** (`src/features/regime.py`):
- [ ] VIX auto-scaling: Verify median > 1.0 detection works for edge cases
- [ ] Vol regime thresholds: Are 10%/15%/25%/35% bands appropriate after scaling?
- [ ] Trend regime thresholds: Are 20%/15% momentum thresholds reasonable?
- [ ] Composite regime score weighting: Is the scoring justified?

### 3. STATISTICAL VALIDITY

**Feature Engineering**:
- [ ] Is there any lookahead bias in feature computation?
- [ ] Are rolling windows correctly aligned (backward-looking only)?
- [ ] Are percentile calculations correct (handling edge cases)?
- [ ] Is forward-looking label generation point-in-time correct?

**Label Generation** (`src/features/labels.py`):
- [ ] Forward max drawdown: Is vectorized version mathematically equivalent to loop?
- [ ] Forward realized vol: Correct shift alignment for "future" returns?
- [ ] Touch strike probability: Is vectorized reversed-rolling approach correct?
- [ ] NaN preservation: Are all forward labels preserving NaN for insufficient data?

**Edge Score** (`src/features/vol_edge.py`):
- [ ] Is the 40/35/25 weighting of components justified?
- [ ] Is sigmoid normalization `100 / (1 + exp(-5*(ratio-1.1)))` appropriate?
- [ ] Division guards: Verify all divisions have zero-guards

### 4. QUANT LOGIC & STRATEGY

**Volatility Risk Premium (VRP)**:
- [ ] Is `IV - RV` the correct formulation for VRP?
- [ ] Should we use forward-looking RV or historical RV for live trading?
- [ ] Is 21-day RV the right horizon to compare with ATM IV?

**Wheel Strategy Logic**:
- [ ] Is delta of 0.30 optimal? What does research say?
- [ ] Is 30-45 DTE the optimal range? Cite sources
- [ ] 50% profit target, 200% stop loss - are these backed by data?
- [ ] Assignment handling: Is the stock basis calculation correct?

**Regime Adaptation**:
- [ ] Should wheel strategy be paused in crisis regime?
- [ ] Are delta/DTE adjustments by regime reasonable?
- [ ] Is position scaling by drawdown appropriate?

### 5. DATA ENGINEERING

**Data Pipeline** (`data/pipeline.py`, `data/consolidated_loader.py`):
- [ ] Is ticker normalization correct ("AAPL UW Equity" -> "AAPL")?
- [ ] Are date formats handled consistently?
- [ ] Is data validated before feature computation?

**Feature Store** (`data/feature_store.py`):
- [ ] Is Parquet partitioning efficient for access patterns?
- [ ] Is lineage tracking complete and accurate?
- [ ] Are there race conditions in concurrent writes? (Still open issue)

**Feature Pipeline** (`data/feature_pipeline.py`):
- [ ] Is the computation order (Layer 1 -> 2 -> 3) correct?
- [ ] Are dependencies between features properly handled?
- [ ] Is IV Rank now correctly using `volatility.iv_rank()` method?

### 6. CODE QUALITY

**Error Handling**:
- [ ] Division by zero edge cases (verify all guards work)
- [ ] NaN propagation (verify no silent corruption)
- [ ] Empty DataFrame handling
- [ ] Type safety

**Performance**:
- [ ] Are there remaining Python loops that should be vectorized?
- [ ] Are rolling operations using efficient pandas/numpy?
- [ ] Is memory usage reasonable for large datasets?

### 7. REMAINING KNOWN ISSUES (from prior audit)

The following issues were identified but may not be fully resolved:

1. **FeatureStore atomicity**: Writes are not atomic/locked - race condition risk
2. **VIX term structure regime**: `vix_term_structure_regime` may have denominator-zero risk
3. **calculate_optimal_contracts**: May force 1 contract when constraints imply 0
4. **RSI initialization**: May not use canonical Wilder bootstrap for first N periods

---

## Output Format

Provide your review in the following structure:

### Executive Summary
- Overall assessment (1-10 scale)
- Critical issues count (blocking deployment)
- Major issues count (should fix before production)
- Minor issues count (nice to fix)
- **Fix Verification Status**: All 6 fixes verified working? Yes/No

### Fix Verification Results
For each of the 6 claimed fixes:
- **Fix N**: [VERIFIED/FAILED]
- **Test Result**: [actual output]
- **Notes**: [any observations]

### Critical Issues (Must Fix)
For each issue:
- **Location**: File and line number
- **Issue**: Description
- **Impact**: Why this is critical
- **Evidence**: Calculation or source showing the error
- **Fix**: Recommended correction

### Major Issues (Should Fix)
Same format as above.

### Minor Issues (Nice to Fix)
Same format as above.

### Formula Verification Checklist
For each formula/calculation reviewed, state:
- [ ] Verified correct, source: [citation]
- [ ] Error found: [description]
- [ ] Unable to verify: [reason]

### Recommendations
- Architecture improvements
- Performance optimizations
- Additional features needed
- Risk considerations

---

## Files to Review

Priority order:

1. **Core Quant Logic** (VERIFY FIXES HERE):
   - `src/features/technical.py` - RSI fix
   - `src/features/regime.py` - VIX scaling fix
   - `src/features/volatility.py` - IV Rank implementation
   - `src/features/labels.py` - NaN preservation fix
   - `src/features/vol_edge.py` - Division guards
   - `engine/risk_manager.py` - VaR compounding fix
   - `engine/option_pricer.py`

2. **Feature Pipeline**:
   - `data/feature_pipeline.py` - IV Rank wiring fix
   - `data/feature_store.py`
   - `data/pipeline.py`

3. **Additional Features**:
   - `src/features/dynamics.py`
   - `src/features/options.py` - Division guards
   - `src/features/assignment.py`
   - `src/features/events.py`

4. **Trading Engine**:
   - `engine/wheel_tracker.py`
   - `engine/signals.py`

---

## Reference Sources

Use these authoritative sources to verify calculations:

**Volatility Estimators**:
- Parkinson (1980): "The Extreme Value Method for Estimating the Variance of the Rate of Return"
- Garman & Klass (1980): "On the Estimation of Security Price Volatilities from Historical Data"
- Yang & Zhang (2000): "Drift Independent Volatility Estimation Based on High, Low, Open, and Close Prices"

**Options Pricing**:
- Hull, J. (2018): "Options, Futures, and Other Derivatives" (10th ed.)
- Natenberg, S. (2014): "Option Volatility and Pricing"

**Risk Management**:
- Kelly, J.L. (1956): "A New Interpretation of Information Rate"
- Thorp, E.O. (2006): "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"

**Technical Analysis**:
- Wilder, J.W. (1978): "New Concepts in Technical Trading Systems" (RSI)

---

## Important Notes

1. **Verify Fixes First**: Before looking for new issues, confirm the 6 claimed fixes are working by running the verification tests.

2. **Use Fresh Code**: Ensure you're reviewing the latest commit (`828ca6e` or later). Run `git log -1` to verify.

3. **Be Critical**: This is a financial trading system. Errors have real monetary consequences.

4. **Show Your Work**: When verifying a formula, show the canonical form and compare.

5. **Cite Sources**: Every mathematical verification should reference an authoritative source.

6. **Test Behaviorally**: Don't just read code - run the verification tests to confirm behavior.

---

## Begin Review

1. First, verify you have the latest code:
   ```bash
   git log --oneline -1
   # Should show: 828ca6e Fix critical quant issues from external audit
   ```

2. Run all 6 verification tests to confirm fixes are working.

3. Then proceed with full review of remaining items.

Your review should be thorough enough that a quant could use it to approve the system for deploying real capital.
