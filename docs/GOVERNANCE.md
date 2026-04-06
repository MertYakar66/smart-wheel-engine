# Model Governance Framework

This document establishes the governance framework for quantitative models used in the Smart Wheel Engine trading system.

---

## 1. Model Risk Management Policy

### 1.1 Scope
This policy applies to all quantitative models including:
- Option pricing models (Black-Scholes, Greeks)
- Volatility estimators (RV, IV calculations)
- Risk models (VaR, position sizing)
- Technical indicators (RSI, ATR, etc.)
- Regime detection models
- Assignment probability models

### 1.2 Model Risk Categories

| Category | Risk Level | Examples |
|----------|-----------|----------|
| **Tier 1** | High | Option pricing, VaR, position sizing |
| **Tier 2** | Medium | Volatility estimators, regime detection |
| **Tier 3** | Low | Technical indicators, data transformations |

---

## 2. Model Development Standards

### 2.1 Documentation Requirements

All models must include:
- [ ] Mathematical specification with formulas
- [ ] Input parameter definitions and valid ranges
- [ ] Output specifications and bounds
- [ ] Assumptions and limitations
- [ ] Reference to academic or industry sources

### 2.2 Code Standards

```python
# Required docstring format for all model functions:
def model_function(param1: float, param2: float) -> float:
    """
    Brief description of what the function computes.

    Mathematical basis:
        [Formula in LaTeX-style notation]

    Args:
        param1: Description with valid range [min, max]
        param2: Description with units

    Returns:
        Description with expected bounds

    Raises:
        ValueError: When inputs are invalid

    References:
        Author (Year). "Paper/Book Title"
    """
```

### 2.3 Testing Requirements

| Model Tier | Unit Tests | Property Tests | Textbook Validation |
|------------|-----------|----------------|---------------------|
| Tier 1 | Required | Required | Required |
| Tier 2 | Required | Recommended | Recommended |
| Tier 3 | Required | Optional | Optional |

**Minimum Coverage:** 70% line coverage, 80% for Tier 1 models

---

## 3. Validation Framework

### 3.1 Initial Validation

Before deployment, all models must pass:

1. **Textbook Verification**: Results match published examples
2. **Boundary Testing**: Behavior at parameter limits
3. **Property-Based Testing**: Invariants hold across random inputs
4. **Point-in-Time Testing**: No lookahead bias

### 3.2 Ongoing Validation

| Validation Type | Frequency | Owner |
|----------------|-----------|-------|
| Regression tests | Every commit | CI/CD |
| Back-test comparison | Weekly | Quant Team |
| Live performance review | Monthly | Risk Team |
| Full model review | Annual | Model Committee |

### 3.3 Back-Testing Standards

- Minimum 3 years of historical data
- Walk-forward validation with embargo periods
- Out-of-sample testing required
- Transaction costs included
- Slippage assumptions documented

---

## 4. Change Management

### 4.1 Change Classification

| Type | Description | Approval |
|------|-------------|----------|
| **Material** | Formula change, new model | Model Committee |
| **Significant** | Parameter recalibration | Quant Lead |
| **Minor** | Bug fix, code optimization | Peer Review |

### 4.2 Change Process

```
1. Create RFC (Request for Change) document
2. Develop in feature branch
3. Run full test suite
4. Peer code review
5. Quant review (for Tier 1/2)
6. Approval based on change type
7. Deploy with monitoring
8. Post-deployment validation
```

### 4.3 Emergency Changes

For critical production issues:
1. Implement fix with immediate peer review
2. Deploy with enhanced monitoring
3. Complete formal review within 48 hours
4. Document lessons learned

---

## 5. Monitoring & Alerts

### 5.1 Real-Time Monitoring

| Metric | Threshold | Action |
|--------|-----------|--------|
| Price deviation from market | > 5% | Alert + investigation |
| Greeks outside bounds | Any violation | Halt trading |
| VaR breach | > 3 consecutive | Reduce positions |
| Model latency | > 100ms | Alert + scaling |

### 5.2 Daily Reports

- P&L attribution by model component
- Model prediction vs realized outcomes
- Risk limit utilization
- Data quality metrics

### 5.3 Alert Escalation

```
Level 1 (Automated): Log + notification to on-call
Level 2 (Warning): Page quant team
Level 3 (Critical): Page risk + halt affected strategies
Level 4 (Emergency): Page all stakeholders + full halt
```

---

## 6. Data Quality

### 6.1 Input Data Standards

| Data Type | Required Checks |
|-----------|----------------|
| Prices | Non-negative, not stale (< 15 min) |
| Volumes | Non-negative integer |
| IV | Range [0.01, 5.0] |
| Greeks | Within theoretical bounds |

### 6.2 Data Lineage

All feature computations must:
- Log source data timestamp
- Track transformation steps
- Support audit replay

### 6.3 Missing Data Handling

| Scenario | Default Behavior |
|----------|-----------------|
| Missing price | Use last valid price (with staleness flag) |
| Missing IV | Flag position for manual review |
| Missing Greeks | Recompute from model |

---

## 7. Model Inventory

### 7.1 Active Models

| Model ID | Name | Tier | Status | Last Review |
|----------|------|------|--------|-------------|
| BS-001 | Black-Scholes | 1 | Active | 2026-04-02 |
| RV-001 | Realized Vol Suite | 2 | Active | 2026-04-02 |
| VR-001 | VaR Models (Parametric, Historical, Covariance) | 1 | Active | 2026-04-02 |
| MC-001 | Monte Carlo VaR | 1 | Active | 2026-04-02 |
| JD-001 | Jump-Diffusion VaR (Merton) | 1 | Active | 2026-04-02 |
| GS-001 | Greeks Stress Testing | 1 | Active | 2026-04-02 |
| KC-001 | Kelly Criterion | 1 | Active | 2026-04-02 |
| RD-001 | Regime Detector | 2 | Active | 2026-04-02 |
| TI-001 | Technical Indicators | 3 | Active | 2026-04-02 |

### 7.2 Deprecated Models

| Model ID | Name | Deprecation Date | Replacement |
|----------|------|------------------|-------------|
| (none) | - | - | - |

---

## 8. Audit & Compliance

### 8.1 Audit Trail

All model executions must log:
- Timestamp
- Input parameters
- Output values
- Model version
- User/system that triggered

### 8.2 Compliance Checkpoints

- [ ] All models documented in MODEL_CARDS.md
- [ ] Change log maintained
- [ ] Test coverage meets thresholds
- [ ] Annual review completed
- [ ] Risk limits configured and monitored

### 8.3 External Audit Support

Maintain ability to provide:
- Complete model documentation
- Historical model outputs
- Validation evidence
- Change history

---

## 9. Roles & Responsibilities

### 9.1 Quant Team
- Model development and testing
- Initial validation
- Performance optimization
- Documentation maintenance

### 9.2 Risk Team
- Model risk assessment
- Limit configuration
- Monitoring and alerts
- Escalation handling

### 9.3 Model Committee
- Material change approval
- Annual model reviews
- Policy updates
- Exception handling

---

## 10. Review Schedule

| Activity | Frequency | Next Due |
|----------|-----------|----------|
| Model performance review | Monthly | 2026-05-01 |
| Parameter recalibration | Quarterly | 2026-07-01 |
| Full model validation | Annual | 2027-04-01 |
| Governance policy review | Annual | 2027-04-01 |

---

## Appendix A: Approved Textbook References

1. Hull, J.C. - "Options, Futures, and Other Derivatives"
2. Wilmott, P. - "Paul Wilmott on Quantitative Finance"
3. Natenberg, S. - "Option Volatility and Pricing"
4. Sinclair, E. - "Volatility Trading"
5. Taleb, N.N. - "Dynamic Hedging"

## Appendix B: Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2024-03-26 | 1.0.0 | Initial governance framework | Quant Team |
| 2026-04-02 | 2.0.0 | Updated review dates, added Monte Carlo VaR | Quant Team |
