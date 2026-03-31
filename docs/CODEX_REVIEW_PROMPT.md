# Smart Wheel Engine - Comprehensive Code Review & Validation Prompt

## Instructions for Reviewer

You are a senior quantitative developer and financial engineer tasked with performing an exhaustive code review of the Smart Wheel Engine. Your review must be **aggressive, thorough, and unforgiving**. Challenge every assumption, verify every formula, and stress-test every edge case.

**Your mission**: Identify bugs, logic errors, mathematical inaccuracies, security vulnerabilities, performance issues, and architectural weaknesses. Do not accept "good enough" - demand correctness.

---

## Repository Structure

```
smart-wheel-engine/
├── engine/                     # Core quantitative engine
│   ├── option_pricer.py        # Black-Scholes, American options, Greeks
│   ├── risk_manager.py         # VaR, CVaR, position sizing, stress testing
│   ├── monte_carlo.py          # Monte Carlo simulations
│   ├── volatility_surface.py   # IV surface construction
│   ├── regime_detector.py      # Market regime detection
│   ├── signals.py              # Trading signals
│   └── wheel_tracker.py        # Wheel strategy tracking
├── dashboard/                  # Professional trading dashboard
│   └── quant_dashboard.py      # Interactive CLI and API
├── utils/
│   └── security.py             # Security features
├── src/                        # Additional source modules
│   ├── features/               # Feature engineering
│   └── data/                   # Data handling
└── tests/                      # Test suite (170+ tests)
```

---

## PART 1: OPTION PRICING ENGINE (Critical Priority)

### File: `engine/option_pricer.py`

#### 1.1 Black-Scholes-Merton Implementation

**Verify the core formula**:
```
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
Call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
Put = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
```

**Test cases to verify**:
- [ ] Hull Example 15.6: S=42, K=40, T=0.5, r=0.10, σ=0.20 → Call ≈ $4.76
- [ ] Put-call parity: C - P = S·e^(-qT) - K·e^(-rT)
- [ ] At expiry (T=0): Call = max(0, S-K), Put = max(0, K-S)
- [ ] Zero volatility: Deterministic forward value
- [ ] Deep ITM call delta → e^(-qT), Deep OTM call delta → 0

#### 1.2 Greeks Verification (ALL ORDERS)

**First-Order Greeks** - Verify formulas:
```
Delta_call = e^(-qT) · N(d1)
Delta_put = e^(-qT) · (N(d1) - 1)
Gamma = e^(-qT) · n(d1) / (S · σ · √T)
Theta = -[S·e^(-qT)·n(d1)·σ/(2√T)] + terms...
Vega = S · e^(-qT) · n(d1) · √T
Rho_call = K · T · e^(-rT) · N(d2)
```

**Second-Order Greeks** - Verify:
```
Vanna = -e^(-qT) · n(d1) · d2 / σ
Charm = e^(-qT) · n(d1) · [2(r-q)T - d2·σ·√T] / (2T·σ·√T)
Volga = Vega · d1 · d2 / σ
```

**Third-Order Greeks** - Verify:
```
Speed = -Gamma · (d1/(σ√T) + 1) / S
Color = -Gamma/(2T) · [2qT + 1 + d1·(2(r-q)T - d2·σ·√T)/(σ√T)]
Ultima = -Vega · (d1·d2·(1-d1·d2) + d1² + d2²) / σ²
```

**Validation Tests**:
- [ ] Finite difference verification: (f(x+h) - f(x-h)) / 2h ≈ analytical
- [ ] Gamma should be highest ATM, decrease as you move ITM/OTM
- [ ] Vega always positive for both calls and puts
- [ ] Put delta in [-1, 0], Call delta in [0, 1]
- [ ] Sum of call delta + |put delta| ≈ e^(-qT) for same strike

#### 1.3 American Option Pricing (Barone-Adesi-Whaley)

**Verify the BAW approximation**:
```
For calls with dividends:
C_american = C_european + A2 · (S/S*)^q2  when S < S*

For puts:
P_american = P_european + A1 · (S*/S)^(-q1)  when S > S*
```

**Critical checks**:
- [ ] American call (q=0) should equal European call exactly
- [ ] American put should always be >= European put
- [ ] Deep ITM American put should approach intrinsic value
- [ ] Early exercise boundary (S*) calculation is correct
- [ ] Newton-Raphson convergence for critical price

#### 1.4 Implied Volatility Solver

**Verify Newton-Raphson + Brent fallback**:
- [ ] Recovery test: price option with σ=0.25, solve IV, get back 0.25
- [ ] Handles edge cases: below intrinsic, above maximum, expired
- [ ] Convergence within 100 iterations for reasonable inputs
- [ ] Initial guess (Brenner-Subrahmanyam) is appropriate

#### 1.5 Vectorized Operations

**Verify consistency**:
- [ ] `vectorized_bs_price()` matches `black_scholes_price()` for all cases
- [ ] Edge cases (T≤0, σ≤0) handled identically
- [ ] No division by zero, no NaN propagation
- [ ] Performance: vectorized should be faster for N>100

---

## PART 2: RISK MANAGEMENT ENGINE (Critical Priority)

### File: `engine/risk_manager.py`

#### 2.1 Value at Risk (VaR)

**Parametric (Delta-Gamma-Vega) VaR**:
```
VaR_delta = z_α · |δ_$| · σ · √T
VaR_gamma = 0.5 · |γ_$| · (z_α · σ)²
VaR_vega = |ν| · σ_vol · √T
```

**Verify**:
- [ ] z-score calculation for confidence levels (95% → 1.645, 99% → 2.326)
- [ ] Horizon scaling: √T for delta, T for gamma
- [ ] Sign handling for short gamma (adds risk) vs long gamma (reduces)

#### 2.2 Multi-Asset Covariance VaR

**Verify the formula**:
```
VaR = z_α · √(δ' Σ δ)
where δ = dollar delta vector, Σ = covariance matrix
```

**Critical checks**:
- [ ] Covariance matrix construction: Σ_ij = σ_i · σ_j · ρ_ij
- [ ] Matrix is symmetric and positive semi-definite
- [ ] Component VaR attribution sums correctly
- [ ] Diversification benefit: VaR_portfolio < Σ VaR_individual
- [ ] Marginal VaR calculation: ∂VaR/∂w_i

#### 2.3 Conditional VaR (Expected Shortfall)

**Verify**:
```
CVaR = E[X | X > VaR] = σ · φ(z_α) / (1 - α)  for normal
```

- [ ] CVaR ≥ VaR always
- [ ] Historical CVaR: mean of losses beyond VaR threshold
- [ ] Gamma scaling for tail risk (hazard rate adjustment)

#### 2.4 Historical VaR

**Verify**:
- [ ] Proper quantile interpolation (not just floor index)
- [ ] Multi-day horizon uses compounding, not scaling
- [ ] Handles missing data, outliers appropriately

#### 2.5 Kelly Criterion

**Verify the formula**:
```
f* = (p·b - q) / b
where p = win rate, q = 1-p, b = avg_win / avg_loss
```

- [ ] Returns 0 for negative expectation
- [ ] Capped at 25% (reasonable maximum)
- [ ] Half-Kelly (fraction=0.5) correctly applied

#### 2.6 Stress Testing

**Verify scenarios**:
- [ ] Crash scenarios apply correct spot AND vol shocks
- [ ] P&L calculation: δ·dS + 0.5·γ·dS² + ν·dσ + θ·dT
- [ ] Worst case combines multiple shocks correctly
- [ ] Custom scenarios work as expected

---

## PART 3: DASHBOARD & API

### File: `dashboard/quant_dashboard.py`

#### 3.1 API Correctness

- [ ] `price_european()` returns all expected keys
- [ ] `price_american()` includes early exercise premium
- [ ] `analyze_greeks()` structure is correct (first/second/third order)
- [ ] `calculate_var()` switches methods correctly (covariance vs parametric)

#### 3.2 Portfolio Management

- [ ] Positions aggregate correctly
- [ ] Correlation matrix applied properly
- [ ] Greeks sum across positions with correct signs (short = negative)

#### 3.3 Report Generation

- [ ] Reports are formatted correctly
- [ ] No sensitive data in reports
- [ ] Timestamps are accurate

---

## PART 4: SECURITY MODULE

### File: `utils/security.py`

#### 4.1 Input Validation

- [ ] Price validation: rejects negative, rejects >1M
- [ ] Quantity validation: rejects <1, rejects >10000
- [ ] Symbol sanitization: removes special characters
- [ ] Option type validation: only 'call' or 'put'

#### 4.2 Secrets Management

- [ ] .env loading works correctly
- [ ] Sensitive values never logged
- [ ] Hashing uses proper salt

#### 4.3 Rate Limiting

- [ ] Correctly limits requests per window
- [ ] Reset works properly
- [ ] Wait time calculation is accurate

---

## PART 5: EDGE CASES & BOUNDARY CONDITIONS

### 5.1 Option Pricer Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| T = 0 | Return intrinsic value |
| σ = 0 | Return deterministic forward |
| S = 0 | Should raise error or handle |
| K = 0 | Should raise error or handle |
| S = K (ATM) | Delta ≈ 0.5·e^(-qT) for call |
| Deep ITM | Delta → ±e^(-qT) |
| Deep OTM | Delta → 0 |
| Very high σ | No overflow, reasonable values |
| Negative r | Should work (negative rates exist) |

### 5.2 Risk Manager Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Empty portfolio | VaR = 0 |
| Single position | No correlation needed |
| Perfect correlation | No diversification benefit |
| Zero correlation | Maximum diversification |
| Negative portfolio value | Reject or handle |
| Missing spot price | Error or use fallback |

---

## PART 6: MATHEMATICAL CONSISTENCY TESTS

### 6.1 Put-Call Parity

For every (S, K, T, r, σ, q):
```
C - P = S·e^(-qT) - K·e^(-rT)
```
Verify tolerance < 0.0001

### 6.2 Greeks Relationships

- [ ] Delta_call - Delta_put = e^(-qT)
- [ ] Gamma_call = Gamma_put
- [ ] Vega_call = Vega_put
- [ ] Theta: C_theta - P_theta = r·K·e^(-rT) - q·S·e^(-qT)

### 6.3 Moneyness Symmetry

For ATM options (S=K):
- [ ] Call delta ≈ Put delta + e^(-qT)
- [ ] Both have same gamma, vega

### 6.4 Time Decay

As T → 0:
- [ ] ATM gamma → ∞ (correctly handled)
- [ ] ATM theta → -∞ (correctly handled)
- [ ] Vega → 0

---

## PART 7: PERFORMANCE & EFFICIENCY

### 7.1 Computational Efficiency

- [ ] No redundant CDF calls (compute N(d1), N(d2) once)
- [ ] Vectorized operations use numpy efficiently
- [ ] No unnecessary object creation in loops

### 7.2 Memory Usage

- [ ] Large arrays handled efficiently
- [ ] No memory leaks in repeated calculations

---

## PART 8: CODE QUALITY

### 8.1 Documentation

- [ ] All public functions have docstrings
- [ ] Parameter types and returns documented
- [ ] Edge cases documented

### 8.2 Error Handling

- [ ] Input validation raises clear errors
- [ ] No silent failures
- [ ] Appropriate use of Optional returns

### 8.3 Type Safety

- [ ] Type hints consistent with actual types
- [ ] Literal types used where appropriate

---

## DELIVERABLES

### 1. Issue Log

Create a table:
| ID | Severity | Location | Description | Impact | Suggested Fix |
|----|----------|----------|-------------|--------|---------------|

Severity levels:
- **P0 (Critical)**: Mathematical error, security flaw, data corruption
- **P1 (High)**: Logic error, incorrect behavior
- **P2 (Medium)**: Edge case not handled, performance issue
- **P3 (Low)**: Style, documentation, minor inconsistency

### 2. Verification Matrix

| Component | Formula Correct | Edge Cases | Tests Pass | Score |
|-----------|-----------------|------------|------------|-------|

### 3. Score Card

| Dimension | Weight | Score (1-10) | Notes |
|-----------|--------|--------------|-------|
| Mathematical Correctness | 30% | | |
| Risk Management | 20% | | |
| Edge Case Handling | 15% | | |
| Test Coverage | 15% | | |
| Code Quality | 10% | | |
| Security | 5% | | |
| Performance | 5% | | |
| **Total** | 100% | | |

### 4. Recommendations

Prioritized list of:
1. Critical fixes required
2. Improvements recommended
3. Future enhancements suggested

---

## EXECUTION INSTRUCTIONS

1. **Clone and setup**:
   ```bash
   git clone <repo>
   cd smart-wheel-engine
   pip install -e ".[dev]"
   ```

2. **Run existing tests**:
   ```bash
   pytest tests/ -v --tb=short
   ```

3. **For each component**:
   - Read the code thoroughly
   - Verify formulas against textbook references
   - Write additional test cases as needed
   - Document all findings

4. **Key files to scrutinize**:
   - `engine/option_pricer.py` - Lines 110-550 (pricing and Greeks)
   - `engine/option_pricer.py` - Lines 684-900 (American options)
   - `engine/risk_manager.py` - Lines 340-600 (VaR calculations)
   - `engine/risk_manager.py` - Lines 600-800 (covariance VaR)
   - `dashboard/quant_dashboard.py` - Full file

5. **Reference materials**:
   - Hull, "Options, Futures, and Other Derivatives" (11th Ed)
   - Barone-Adesi & Whaley (1987), Journal of Finance
   - Jorion, "Value at Risk" (3rd Ed)

---

## FINAL NOTES

Be ruthless. This is a trading system where errors cost money. Every formula must be verified against authoritative sources. Every edge case must be handled. Every assumption must be challenged.

**The goal is not to approve the code - it's to find every possible flaw.**

When in doubt, write a test case. If the test fails, you've found a bug. If it passes, document why the behavior is correct.

Good luck.
