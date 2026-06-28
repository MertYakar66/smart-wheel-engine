# Is the real skew-edge a usable SELECTION signal? (2026-06-28)

**Question:** the rail audit (C4) confirmed `edge_vs_fair` is positive across every
VIX regime (the VRP/skew premium is real). Does sorting `market_mid` candidates by
that edge **predict realized returns** — i.e. is it a usable selection/sizing
*lever*, or just risk-compensation (neutral/trap)?

**Method:** weekly ranks over 2020-2024, 100t, rail ON; **n=3,741** `market_mid`
candidates. Forward-replay the short-put realized P&L (held to expiry), normalize
by collateral (strike×100), sort by raw `edge_vs_fair` and by **edge richness**
`(premium−fair)/premium`, stratify by VIX-at-entry. Cross-sectional predictive
(no tracker). Driver: `skew_edge_selection.py`; data: `out/candidates.csv`.

## Verdict: NOT a clean lever — the apparent signal is a REGIME CONFOUND. Don't build a selection rule on it.

**Overall it looks like a lever** — Spearman(edge, ret) **+0.31**, Spearman(richness,
ret) **+0.28**; richness quintiles rise 0.02% → 0.48% ret/collateral (Q1→Q4, Q5
0.37%). But the regime split shows that's mostly *"edge proxies crisis, and crisis
put-selling pays"*:

| VIX regime | n | mean ret/collateral | richness Q1→Q5 ret | within-regime rho |
|---|---|---|---|---|
| **calm** (≤15) | 791 | **−0.93%** | −0.51 → −0.17 → −0.94 → −1.65 → −1.38 | +0.227 |
| **elevated** (15-25) | 2071 | +0.13% | 0.11 → 0.08 → 0.02 → 0.20 → 0.25 | +0.227 |
| **crisis** (>25) | 879 | **+1.55%** | 1.43 → 1.83 → 1.49 → 1.75 → 1.26 | +0.225 |

- The overall positive rho is **Simpson-style regime mixing**: high-edge candidates
  cluster in **crisis** (mean +1.55%, fat premium + the 2020/2022 recoveries),
  low-edge in **calm** (mean **−0.93%**). The edge largely encodes *which regime*,
  not within-regime quality.
- **Within regime the lever is weak and inconsistent:** rank-rho is a mild ~+0.22
  in all three, but the *mean*-return-by-quintile (what a selector actually
  captures) is only weakly monotone in **elevated**, **flat** in crisis (all
  quintiles win), and **anti-monotone in calm** (richest quintiles have the
  *worst* mean — outlier idiosyncratic blow-ups). High win rate (0.81) with
  regime-negative means = the classic short-put payoff (many small wins, rare big
  losses), and the rare losses concentrate in the rich calm quintiles.
- **Practical consequence:** a regime-neutral "prefer high-edge" selection/sizing
  rule would **hurt in calm** (the most common regime, 791+2071 of 3741 are
  ≤25 VIX) and only "help" by concentrating into crisis exposure — the exact
  regime-dependent risk R11b grappled with. Leaning into high-edge ≈ leaning into
  crisis put exposure.

## Recommendation

**Document, build nothing** (same discipline as R11b — don't build on a confounded,
within-regime-weak signal). The skew-edge's genuine, validated value is in
**ranking/calibration** (the #435 wiring already captures it — `spearman_rho` ↑
every window in the earlier sweep), **not** as a standalone cross-sectional
selector. The VRP is earned compensation, harvested best in crisis-recovery, not a
free cross-sectional alpha. This is a **null result** that forecloses a tempting
but unsound "edge-tilt the book" idea.

*Caveat:* put-leg-only realized (no assignment→CC→recovery leg); 2020-2024 contains
no prolonged never-recovering crash. Weekly sampling → overlapping holding periods
(fine for a cross-sectional sort, inflates n for significance).
