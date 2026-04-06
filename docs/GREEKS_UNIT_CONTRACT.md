# Greeks Unit Contract

**Version:** 1.0.0  
**Date:** 2026-04-06  
**Status:** Canonical Reference

---

## Purpose

This document defines the canonical unit conventions for all Greeks across the Smart Wheel Engine. All modules MUST adhere to these conventions to ensure P&L calculations, stress tests, and risk metrics are consistent.

---

## Unit Definitions

### Delta
- **Unit:** Dollar change per $1 move in underlying
- **Convention:** `delta_dollars = delta * spot * contracts * multiplier`
- **Range:** [-1, 1] for single option; aggregated for portfolios

### Gamma
- **Unit:** Dollar change in delta per $1 move in underlying
- **Convention:** `gamma_dollars = gamma * spot^2 * contracts * multiplier / 100`
- **P&L:** `gamma_pnl = 0.5 * gamma_dollars * (spot_change_pct)^2`

### Theta
- **Unit:** Dollar change per calendar day
- **Convention:** Pricer returns annual theta; convert: `daily_theta = annual_theta / 365`
- **P&L:** `theta_pnl = daily_theta * days * contracts * multiplier`

### Vega (CRITICAL - MOST COMMON SOURCE OF ERRORS)
- **Unit:** Dollar change per 1 percentage point (1 vol point) change in IV
- **Convention:** If IV moves from 25% to 26%, that's a 1 vol point change
- **Calculation in pricer:** `vega = S * exp(-qT) * N'(d1) * sqrt(T) / 100`
- **P&L formulas:**
  ```python
  # When vol_change is in decimal (e.g., 0.05 for 5% IV increase)
  vega_pnl = vega * vol_change * 100 * contracts * multiplier
  
  # When vol_change is in vol points (e.g., 5 for 5 vol point increase)
  vega_pnl = vega * vol_change * contracts * multiplier
  
  # When vol_change is relative (e.g., 0.20 for 20% increase in IV level)
  # If IV was 25%, a 20% relative increase = 25% * 0.20 = 5 vol points
  iv_change_vol_points = base_iv * relative_change * 100
  vega_pnl = vega * iv_change_vol_points * contracts * multiplier
  ```

### Rho
- **Unit:** Dollar change per 1 percentage point (100 bps) change in risk-free rate
- **Convention:** If rate moves from 5% to 6%, that's a 1 percentage point change
- **P&L:** `rho_pnl = rho * rate_change * 100 * contracts * multiplier`
  - Where `rate_change` is in decimal (0.01 = 1%)

---

## Module-Specific Implementation Notes

### engine/option_pricer.py
- `black_scholes_vega()` returns vega per 1% vol (divides by 100)
- `black_scholes_rho()` returns rho per 1% rate (divides by 100)
- These are the canonical source values

### engine/risk_manager.py
- `PortfolioGreeks.vega` aggregates vega per 1% vol
- Stress scenarios use `vega * vol_move * 100` when `vol_move` is decimal

### engine/stress_testing.py
- `greeks_stress_ladder()` uses `iv_shock` as a relative change (e.g., 0.15 = 15% increase in IV level)
- Must convert: `vega_pnl = vega * (base_iv * iv_shock) * 100 * multiplier`

---

## Validation Tests

Each module should include unit tests verifying:

1. **Finite difference check:** Vega should match `(P(sigma+0.01) - P(sigma)) / 0.01`
2. **Cross-module consistency:** Same position priced in pricer vs risk_manager should give same vega
3. **P&L decomposition:** Sum of Greek P&Ls should approximate full repricing within 1% for small moves

---

## Changelog

- 2026-04-06: Initial version establishing canonical conventions
