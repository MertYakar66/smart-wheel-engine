# R11b skew-edge gate — A/B dollar-impact findings (2026-06-27/28)

**Change under test:** R11b (PR #437), a second elevated-vol size-down trigger
under R11's shared guard. Fires when `vix > 25 AND premium_source=="market_mid"
AND edge_vs_fair > 0` → downgrade proceed→review. Motivated by the with/without
real-premium sweep, where the 2020-crash OOS window (S35) showed the real
(skew-rich) premium made more candidates clear the EV bar (26→40 trades) and
dragged realized NAV −3.4%.

**Method:** reviewer-in-the-loop A/B (R11/R11b are NOT in the
`rank_candidates_by_ev` regression path — only the dossier reviewer). Both arms
apply R11a; **active** adds R11b; **suppressed** = the pre-#437 live engine. Shared
daily rank, rail ON (`SWE_OPTION_PREMIUM_DIR`), 100-ticker, realistic friction.
Driver: `r11b_skew_edge_driver.py` (adapted from the trusted D23
`r11_dollar_impact_driver.py`). §2 verified clean (0 opened puts with ev≤0) in
every arm.

## Verdict: R11b is §2-safe but **net-costly over a full cycle — do not merge as-is.**

| Window | suppressed NAV (R11a only) | active NAV (R11a+R11b) | Δ NAV | Δ return | Δ Sharpe | R11b blocked | blocked counterfactual (held-to-expiry) |
|---|---|---|---|---|---|---|---|
| **Pilot** 2020 crash Q (24t, frictionless) | $986,343 | $983,566 | −$2,777 | −0.28pp | −0.26 | 69 | **−$159/contract (losers averted)** |
| **W3** 2020-2024 (100t, full friction) | $1,389,375 | $1,240,905 | **−$148,470** | **−14.85pp** | −0.12 | 251 | **+$363/contract (winners forgone)** |
| **W4** 2021-2025 (100t, full friction) | $1,489,878 | $1,362,738 | **−$127,140** | **−12.71pp** | −0.15 | 55 | **+$367/contract (winners forgone)** |

The direction is **robust across both full-cycle windows** (−12.7 to −14.9pp,
Sharpe −0.12/−0.15). The blocked set is net **winning** in both, in *every* VIX
bucket:

- **W3** — 25-35: +$263/contract (n=178, 15% assignment); vix>35: +$608/contract (n=73, **6.8%** assignment).
- **W4** — 25-35: +$351/contract (n=54, 20%); vix>35: +$1,247 (n=1).

## Why — the premise was too narrow

The motivating S35 −3.4% was a **single acute-crash window** (Feb-Mar 2020). The
pilot reproduces that locally: in the crash quarter the blocked trades *lost*
(−$159/contract) so R11b helped. But over a full cycle the **far more common
pattern is a VIX spike that precedes a recovery**, where selling the fat-skew
premium is *profitable*. The companion rail audit (C4) independently shows
`edge_vs_fair` is **positive across every VIX bucket** — the skew/VRP premium is
compensation the seller earns on average. R11b's blunt VIX-*level* trigger fires
**post-spike** and cannot distinguish a transient spike from a sustained crash, so
it forgoes that earned premium. This is the same mechanism D23 documented for R11a
("insurance, net-neutral-to-the-book, not free crisis alpha") — but R11b is net
**negative** because its footprint is larger and it targets the profitable harvest
directly.

## Caveats (honesty)

- The whole-book Δ (−$148k/−$127k) exceeds the blocked-set counterfactual
  (+$91k/+$20k) because the active arm **backfills** freed quota/BP into other
  trades; the Δ includes those replacements + capital-path effects, so it is a
  whole-book figure, not a clean per-trade attribution. The robust, attributable
  fact is the *direction* + the blocked-set being net winners.
- n=2 windows; the magnitude is path-dependent. The *direction* is corroborated by
  both windows AND by the structural argument (VRP is positive across regimes).
- R11b is **§2-clean** (downgrade-only; never rescued a non-tradeable candidate;
  overlay-guard count unchanged at 6) and **re-baseline-free** (no-op on the
  synthetic path → CI/regression byte-identical). Those properties hold; the
  problem is purely that the intervention is net-costly.

## Persistence post-hoc — the redesign option is also rejected (W3 blocked set)

The natural rescue for R11b is a **persistence-gated trigger** (fire only when
VIX>25 has *held* N consecutive days — catch sustained crashes, skip transient
spikes; the D23 R11a research card). Tested cheaply on the W3 blocked set by VIX
persistence-at-entry (no re-run needed):

| min consecutive VIX>25 days | n blocked | counterfactual total | mean/contract | assignment % |
|---|---|---|---|---|
| ≥1 | 251 | +$91,181 | +$363 | 12.7% |
| ≥3 | 228 | +$86,015 | +$377 | 11.8% |
| ≥5 | 210 | +$82,389 | +$392 | 10.5% |
| ≥8 | 194 | +$86,447 | +$446 | 6.2% |
| ≥12 | 173 | +$80,140 | +$463 | 3.5% |
| ≥20 | 128 | +$69,557 | +$543 | 1.6% |

**The blocked set stays net-positive at every persistence threshold and gets
*more* positive (and lower-assignment) the longer the high-VIX persists.** Even
20+-day sustained crises: +$543/contract, 1.6% assignment (premium kept). So
persistence-gating shrinks R11b's footprint but **does not flip its sign** — it
would still forgo winners, just fewer. Over 2020-2024 the put-selling VRP harvest
is *more* profitable the deeper/longer the elevated vol — the opposite of R11b's
premise. *(Caveat: 2020-2024 contains no multi-month never-recovering crash beyond
the acute 2020 Feb-Mar; a 2008-style prolonged drawdown is out-of-sample.)*

## Recommendation

1. **Do not merge #437.** Validation shows R11b costs −12.7 to −14.9pp over a cycle.
2. **Close it.** The skew/VRP premium is earned compensation, and the persistence
   redesign is data-rejected above — handle crash exposure via the existing
   concentration caps (R9/R10) + position sizing + R11a, not a premium-gating rule.
3. The real-premium **wiring (#435) stays** — it's correct (rail audit) and its
   calibration win is robust; only the R11b *gate* on top of it is the misstep.

Decision deferred to the operator (verdict-moving / strategy call). PR #437 marked
draft to prevent accidental merge.

Artifacts: `w3_2020_2024_100t/summary.json`, `w4_2021_2025_100t/summary.json`,
`r11b_skew_edge_driver.py`. Companion: `../real_premium_rail_audit_2026-06-27/`.
