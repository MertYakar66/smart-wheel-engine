# Model Cards - Smart Wheel Engine

This document provides model cards for all quantitative models used in the Smart Wheel Engine, following industry best practices for model governance and documentation.

---

## 1. Black-Scholes Option Pricing Model

### Model Overview
| Field | Value |
|-------|-------|
| **Model Name** | Black-Scholes-Merton |
| **Model Type** | Closed-form analytical pricing |
| **Version** | 1.0.0 |
| **Owner** | Quant Team |
| **Last Validated** | 2024-03-26 |

### Intended Use
- Pricing European-style equity options
- Computing option Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation via Newton-Raphson iteration

### Model Details

**Mathematical Foundation:**
```
C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)

where:
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

**Implementation Reference:** Hull, "Options, Futures, and Other Derivatives", 11th Edition

**Input Parameters:**
| Parameter | Description | Valid Range | Units |
|-----------|-------------|-------------|-------|
| S | Spot price | > 0 | Currency |
| K | Strike price | > 0 | Currency |
| T | Time to expiry | > 0 | Years |
| r | Risk-free rate | [-0.1, 0.5] | Decimal |
| σ | Volatility | (0, 5] | Decimal |
| q | Dividend yield | [0, 0.2] | Decimal |

**Output Bounds:**
| Greek | Call Range | Put Range |
|-------|-----------|-----------|
| Price | [0, S] | [0, K] |
| Delta | [0, 1] | [-1, 0] |
| Gamma | ≥ 0 | ≥ 0 |
| Theta | typically < 0 | typically < 0 |
| Vega | ≥ 0 | ≥ 0 |

**Second-Order Greeks (v1.1):**
| Greek | Formula | Use Case |
|-------|---------|----------|
| Vanna | -e^(-qT)·d2·n(d1)/σ | Vol-delta hedging |
| Charm | ∂Delta/∂T | Delta decay management |
| Volga | Vega·d1·d2/σ | Vega convexity hedging |

**Implied Volatility Solver (v1.1):**
- Newton-Raphson with Brenner-Subrahmanyam initial guess
- Brent's method fallback for robustness
- Arbitrage bounds checking (returns None for invalid prices)
- Precision: 1e-6 (configurable)

### Validation

**Textbook Verification (Hull Example 15.6):**
- Inputs: S=42, K=40, T=0.5, r=0.10, σ=0.20, q=0
- Expected Call Price: 4.76
- Model Output: 4.7594 ✓

**Property-Based Tests:**
- Put-call parity: |C - P - S*e^(-qT) + K*e^(-rT)| < 1e-8 ✓
- Delta bounds verified across 100+ random inputs ✓
- Gamma/Vega non-negativity verified ✓

### Limitations
1. Assumes constant volatility (no smile/skew)
2. European exercise only
3. No early exercise premium for American options
4. Assumes continuous dividend yield (not discrete)
5. Log-normal returns assumption may fail during market stress

### Risk Factors
- **Model Risk**: IV estimation errors propagate to pricing
- **Jump Risk**: Black-Scholes underprices tail events
- **Liquidity Risk**: Wide spreads in illiquid options not captured

---

## 2. Realized Volatility Estimators

### Model Overview
| Field | Value |
|-------|-------|
| **Model Name** | Multi-Estimator Volatility Suite |
| **Estimators** | Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang |
| **Version** | 1.0.0 |
| **Owner** | Quant Team |

### Intended Use
- Historical volatility estimation for IV rank calculations
- Regime detection and position sizing
- Risk management and VaR inputs

### Estimator Details

**1. Close-to-Close (Standard Deviation):**
```
RV = sqrt(252 * Σ(r_i - μ)² / (n-1))
```
- Simple, widely understood
- Inefficient: ignores intraday information

**2. Parkinson (High-Low):**
```
RV = sqrt(252 / (4 * ln(2)) * Σ ln(H/L)²)
```
- ~5x more efficient than close-to-close
- Requires accurate high/low data

**3. Garman-Klass:**
```
RV = sqrt(252 * [0.5 * ln(H/L)² - (2*ln(2)-1) * ln(C/O)²])
```
- Most efficient for continuous markets
- Biased if markets gap

**4. Yang-Zhang:**
```
Combines overnight (O-C) and intraday (OHLC) components
```
- Handles overnight gaps
- Unbiased for gap + continuous markets

### Validation
- All estimators tested against known statistical properties
- Annualization factor (√252) verified
- Non-negativity enforced across all inputs

### Limitations
1. Backward-looking only
2. Sensitive to data quality (bad ticks inflate estimates)
3. Window length trades off responsiveness vs. stability

---

## 3. Kelly Criterion Position Sizing

### Model Overview
| Field | Value |
|-------|-------|
| **Model Name** | Kelly Criterion with Risk Caps |
| **Version** | 1.0.0 |
| **Owner** | Risk Team |

### Formula
```
f* = (p * b - q) / b

where:
p = win probability [0, 1]
q = 1 - p (loss probability)
b = avg_win / avg_loss (win/loss ratio)
```

### Implementation Details
- Returns 0 for invalid inputs (p < 0, p > 1, avg_loss ≤ 0)
- Returns 0 for negative edge (p * b < q)
- Capped at 25% maximum position size
- Supports fractional Kelly (kelly_fraction parameter)

### Validation
| Test Case | Expected | Result |
|-----------|----------|--------|
| p=0.55, b=1.0 | 0.10 | ✓ |
| p=0.40, b=1.0 | 0.00 | ✓ (negative edge) |
| p=0.90, b=3.0 | 0.25 | ✓ (capped) |
| p=-0.1, b=1.0 | 0.00 | ✓ (invalid) |

### Risk Controls
1. Maximum position capped at 25% of portfolio
2. Fractional Kelly (typically 0.25-0.5x) recommended
3. Never uses full Kelly in volatile regimes

---

## 4. Value-at-Risk (VaR) Models

### Model Overview
| Field | Value |
|-------|-------|
| **Model Name** | Delta-Gamma-Vega VaR |
| **Version** | 1.0.0 |
| **Owner** | Risk Team |

### Methods

**1. Parametric VaR:**
```
VaR_Δ = Δ$ * σ_spot * z_α * √T
VaR_Γ = 0.5 * Γ$ * (σ_spot * √T)²  (expected loss component)
VaR_ν = ν * σ_vol * √T

Combined: sqrt(VaR_Δ² + VaR_Γ² + VaR_ν²)
```

**2. Historical VaR:**
- Multi-asset support with leverage effect
- Uses actual P&L distribution
- Includes regime-adjusted scaling

### Validation
- Compounded returns: (1+r1)*(1+r2)*...-1
- Back-tested against historical drawdowns
- Stress scenario coverage verified

### Limitations
1. Assumes normality in parametric method
2. Historical VaR limited by sample size
3. Correlation breakdown during crises not captured

---

## 5. Technical Indicators

### RSI (Relative Strength Index)

**Formula (Wilder's Smoothing):**
```
RS = EMA(gains, α=1/n) / EMA(losses, α=1/n)
RSI = 100 - 100/(1 + RS)
```

**Properties:**
- Range: [0, 100]
- First n values are NaN (warmup period)
- Uses exponential smoothing, not simple average

### ATR (Average True Range)

**Formula:**
```
TR = max(H-L, |H-C_prev|, |L-C_prev|)
ATR = EMA(TR, α=1/n)
```

**Properties:**
- Always non-negative
- Uses Wilder's exponential smoothing

### IV Rank

**Formula:**
```
IV_Rank = (IV_current - IV_min) / (IV_max - IV_min) * 100
```

**Properties:**
- Range: [0, 100]
- Lookback window defines min/max period
- Returns NaN when max = min (no variation)

---

## 6. Regime Detection

### Market Regime Classifier

**States:**
| Regime | Value | Conditions |
|--------|-------|------------|
| Crisis | -2 | Bear + momentum < -15% |
| Bear | -1 | Below 200 SMA + negative momentum |
| Sideways | 0 | Range-bound, mixed signals |
| Bull | 1 | Above 200 SMA + positive momentum |
| Euphoria | 2 | Bull + momentum > 20% |

### Volatility Regime Classifier

**States:**
| Regime | VIX Range |
|--------|-----------|
| Extremely Low | < 10% |
| Low | 10-15% |
| Normal | 15-25% |
| Elevated | 25-35% |
| Crisis | > 35% |

### Composite Score
- Combines trend, volatility, and liquidity regimes
- Range: [-100, +100]
- Used for position sizing and strategy adjustment

---

## Model Governance

### Change Control
1. All model changes require code review
2. Changes to pricing models require validation against textbook values
3. Material changes trigger full regression test suite

### Monitoring
1. Daily P&L attribution to detect model drift
2. IV rank vs realized vol comparison
3. Regime classifier accuracy tracking

### Escalation
1. >5% pricing deviation from market triggers review
2. Consecutive VaR breaches escalate to risk committee
3. Regime misclassification during events triggers calibration

---

## References

1. Hull, J.C. (2021). "Options, Futures, and Other Derivatives", 11th Edition
2. Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance"
3. Gatheral, J. (2006). "The Volatility Surface"
4. Yang, D. & Zhang, Q. (2000). "Drift Independent Volatility Estimation"
5. Kelly, J.L. (1956). "A New Interpretation of Information Rate"
