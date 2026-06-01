# R11 dollar-impact backtest — post-ship validation findings

**Date:** 2026-06-01 · **Driver:** `r11_dollar_impact_driver.py` (this folder) ·
**Engine:** `origin/main` @ `1c2e5d4c` · **Scale:** $1M / 100 tickers
(`UNIVERSE_100`) / full friction / 35-DTE 25-Δ short puts → wheel into CC.

## What this measures

R11 (elevated-vol top-bin size-down, `engine/candidate_dossier.py` :534-566)
shipped on #306/#307 on the heavy-verify I11 leave-one-crisis-out evidence
("2020 +$86k / 2022 +$3.5k averted-vs-forgone", per-contract). It had never
been measured in a full-scale dollar backtest. This run does that, two ways over
the same daily rank:

- **`suppressed`** — the literal pre-R11 engine: open on EV>0, top-N, BP-gated.
  Identical to S38/S43/S44's open policy.
- **`active`** — same, but a candidate is NOT opened when R11's exact gate fires
  (`prob_profit > 0.90` AND `vix(as_of) > 25` AND `ev > $10`, constants imported
  from `engine.candidate_dossier`, VIX from `get_vix_regime` — the exact source
  `wheel_runner` threads). The day's quota backfills to the next candidate.

R11 gates PUT ENTRIES only; the CC leg is identical between arms. Both arms run
`require_ev_authority=False` (the S38 baseline) so R11 is the ONLY difference.

**§2:** every candidate routes through `EVEngine.evaluate` via
`rank_candidates_by_ev`; R11 only ever REMOVES an open (downgrade-only). §2 scan
below: 0 nonpositive-EV opens in either arm, both windows. No `engine/` file was
modified — observation only.

## Headline verdict

**R11 is targeted insurance against a SUSTAINED elevated-vol grind-down (2022),
not a uniformly-positive overlay. Its whole-book dollar sign flips by window, it
lowers Sharpe in BOTH windows, and its per-contract "averted loss" framing
over-states its value to the full wheel.**

1. **R11 reliably averts large losses in a sustained grind-down bear (2022).**
   It blocked the 2022 top-bin opens whose held-to-expiry counterfactual was
   **−$268,804 (W3)** / **−$165,688 (W4)** — ~50% assignment, ≈ −$1,300 to
   −$2,100 per contract. This is R11 working exactly as designed, in both
   windows, and it is the single regime where R11 clearly earns its keep.
2. **R11 BACKFIRES in a sharp V-crash (2020) and over-fires in calm.** Its
   `VIX>25` *level* trigger fires AFTER vol has already spiked, so in 2020 it
   blocked trades that then rode the V-recovery (forwent **+$26,525**); brief
   calm-regime VIX blips forwent another **+$67,405** (W3). The level trigger
   cannot distinguish "spike that will recover" from "grind-down that won't."
3. **Whole-book sign flips by window.** W3 2020-2024 (contains the V-crash):
   **−$37,590 / −3.76pp** (R11 HURTS). W4 2021-2025 (2022 grind-down dominates,
   no 2020 offset): **+$21,733 / +2.17pp** (R11 HELPS).
4. **R11's whole-book impact is statistically indistinguishable from zero on
   this evidence.** The point-estimate Sharpe deltas (W3 ≈ 0.00 to −0.01, W4
   −0.054) and the NAV deltas above are NOT significant: the paired
   active-minus-suppressed daily-return stream has |t| = 0.62 (W3) and 0.24
   (W4) — far inside noise. So the honest claim is NOT "R11 hurts Sharpe"; it is
   that R11 does not *measurably help* the book either, while I11 claimed a
   large benefit. (See "Significance" below — this is the correction that keeps
   this doc from repeating I11's overclaim with the sign flipped.)
5. **The held-to-expiry "averted" number over-states R11's full-wheel value.**
   In W3 R11 "averted" −$173,583 on the blocked CSPs in isolation — yet the book
   ended $37,590 WORSE, because the suppressed arm gets assigned and wheels those
   names into the 2020-21 / 2023-24 recovery via covered calls, which the
   CSP-leg counterfactual never sees.

This nuances — does not contradict — I11: R11's per-contract CSP-leg protection
is real and large in 2022, but (a) its 2020 contribution is a forgone gain, not
an averted loss, at the full-wheel level, and (b) its book-level value is small,
window-dependent, and Sharpe-negative.

## Per-window headline

| Metric | W3 2020-2024 supp | W3 active | W3 Δ | W4 2021-2025 supp | W4 active | W4 Δ |
|---|---|---|---|---|---|---|
| Final NAV | $1,370,771 | $1,333,180 | **−$37,590** | $1,458,145 | $1,479,878 | **+$21,733** |
| Return | +37.08% | +33.32% | −3.76pp | +45.81% | +47.99% | +2.17pp |
| Sharpe (ann.) | 0.511 | 0.500 | −0.010 | 0.694 | 0.640 | **−0.054** |
| Puts opened | 521 | 490 | −31 | 477 | 418 | −59 |
| Put assignments | 116 | 110 | −6 | 97 | 93 | −4 |
| Executed realised (closed CSPs) | +$148,461 | +$119,974 | −$28,486 | +$162,552 | +$66,386 | −$96,167 |
| R11 blocked opens | 0 | 399 | +399 | 0 | 145 | +145 |

## Significance — the whole-book deltas are within noise

The per-window Δ NAV / Δ Sharpe are **point estimates over two overlapping
windows (W3/W4 share 2021-2024), driven by one 2020 V-crash and one 2022 bear —
not independent samples.** A paired test on the active-minus-suppressed daily
return stream (the arms share the same daily rank, so this is the right paired
quantity):

| Window | days arms differ | daily mean (act−supp) | t-stat | distinguishable from 0? |
|---|---|---|---|---|
| W3 2020-2024 | 1266/1304 | −0.21 bp/day | **−0.62** | no |
| W4 2021-2025 | 1303/1303 | +0.23 bp/day | **+0.24** | no |

R11 changes the book on nearly every day, but the **net risk-adjusted
difference washes out to noise in both windows.** Treat the headline ±$ and
Sharpe figures as directionally suggestive of the regime mechanism below, NOT
as an established net benefit or cost. What IS robust is the *qualitative*
decomposition (per-regime sign, the post-spike timing of the level trigger, and
the CSP-leg-vs-full-wheel divergence) — those are mechanisms, not noisy point
estimates.

## Blocked-set counterfactual by regime

Held-to-expiry P&L of the trades R11 removed (positive = R11 forwent profit;
negative = R11 averted loss):

**W3 (2020-2024):**

| Regime | n blocked | Counterfactual realised | mean/contract | assignment % | reading |
|---|---|---|---|---|---|
| **bear_2022** | 129 | **−$268,804** | −$2,083.75 | 51.2% | R11 AVERTED — works as designed |
| crash_2020 | 97 | +$26,525 | +$273.45 | 13.4% | R11 forwent — fired post-spike, missed V-recovery |
| calm | 168 | +$67,405 | +$401.22 | 3.0% | R11 over-fired on brief VIX blips |
| bull_2021 | 5 | +$1,290 | +$258.08 | 0.0% | negligible |
| **net** | 399 | **−$173,583** | −$435.05 | 21.1% | CSP-leg "averted"; book still −$37,590 (wheel recovery) |

**W4 (2021-2025):**

| Regime | n blocked | Counterfactual realised | mean/contract | assignment % | reading |
|---|---|---|---|---|---|
| **bear_2022** | 125 | **−$165,688** | −$1,325.51 | 49.6% | R11 AVERTED — same mechanism as W3 |
| bull_2021 | 17 | +$12,580 | +$740.02 | 11.8% | R11 forwent |
| calm | 3 | +$1,056 | +$351.96 | 0.0% | negligible |
| **net** | 145 | **−$152,052** | −$1,048.63 | 44.1% | almost all 2022; book +$21,733 (no V-crash to offset) |

By VIX-at-entry bucket (the mechanism slice):

| Window | 25-35 band (grind-down) | >35 band (spike) |
|---|---|---|
| W3 | 292 blocked · −$213,953 · 25.7% assigned | 107 blocked · +$40,370 · 8.4% assigned |
| W4 | 142 blocked · −$153,251 · 45.1% assigned | 3 blocked · +$1,199 · 0.0% assigned |

The 25-35 band (mostly 2022) is where R11 earns its keep; the >35 band (mostly
the 2020 spike) is where it backfires. W4 barely touches >35 (it starts 2021),
which is exactly why R11 nets positive there.

## §2 invariant scan

- **W3 suppressed:** 521 opened, 0 ev_dollars ≤ 0. **W3 active:** 490 opened, 0
  ev_dollars ≤ 0.
- **W4 suppressed:** 477 opened, 0 ev_dollars ≤ 0. **W4 active:** 418 opened, 0
  ev_dollars ≤ 0.
- **§2 OK** — R11 only ever removed opens; no candidate bypassed `EVEngine.evaluate`.

## Methodology + caveats

- **Gate replication, not reviewer call.** R11 lives in the dossier reviewer,
  not in `rank_candidates_by_ev` (the backtest's ranker route). Calling the full
  reviewer with no chart would make R2 (chart-missing) downgrade everything and
  swamp R11. So the active arm replicates R11's exact gate using the engine's
  own constants (`R11_TOP_BIN_PROB`, `R11_VIX_THRESHOLD`, `MIN_PROCEED_EV_DOLLARS`)
  and VIX source (`get_vix_regime`). The suppressed arm is the documented
  pre-R11 baseline.
- **Size-down → skip (upper bound).** R11 nominally "sizes down"; the harness
  has no fractional contracts, so the active arm SKIPS the blocked candidate and
  backfills. This is the maximal interpretation of R11's effect (both the
  averted loss and the forgone gain are largest under full-skip). A true
  half-size R11 would roughly halve both the averted-loss and the Sharpe cost.
- **Held-to-expiry counterfactual vs full-wheel book.** The blocked-set
  counterfactual marks each removed CSP to its 35-day expiry in isolation. The
  whole-book Δ NAV captures the second-order effects the counterfactual cannot:
  backfill into lower-ranked names, BP reallocation, and — the dominant one —
  the suppressed arm wheeling its extra assignments into covered calls through
  the recovery. The two metrics answer different questions; the sign gap between
  them (W3: −$173,583 averted vs −$37,590 book) IS finding #5.
- **In-sample engine parameters** (HMM/POT-GPD fit on overlapping data),
  Bloomberg-only universe (no SPY), `put_iv ≡ call_iv` (no skew) — all inherited
  from the S22-S44 backtest lineage.
- **Two windows, not a distribution.** W3/W4 overlap 2021-2024; the divergent
  sign is driven by W3's inclusion of the 2020 V-crash, not independent samples.
  Don't read the ±$ as a forecast — read the regime decomposition.
- Same engine SHA across both windows; no main-advance mid-run.

## Bottom line for the R11 ship decision

R11 is **correctly motivated and works in the regime it was built for** (the
2022-style sustained grind-down: ~50% assignment, large per-contract bleed,
reliably averted in both windows). But this backtest does NOT support reading
it as a free or uniformly-positive overlay:

- Its whole-book NAV/Sharpe effect is **not statistically distinguishable from
  zero** over these two windows (paired daily-return |t| < 0.7 both) — so the
  evidence does NOT show R11 helping the book, and does NOT show it hurting
  either. The directional point estimates (W3 −$37,590, W4 +$21,733) track the
  regime mechanism but are within noise.
- Its per-contract "averted loss" headline is a CSP-leg-in-isolation metric;
  at the full-wheel book level that benefit largely **disappears** (the
  suppressed arm wheels its assignments into the recovery), which is why a real
  per-contract protection produces no measurable book benefit.

If R11 is kept (it is downgrade-only and §2-safe, so it cannot create risk), the
honest framing is "insurance premium against sustained-bear assignment, paid in
Sharpe and in forgone V-recovery premium" — not "free crisis alpha." A
follow-up worth its own study: a trigger that distinguishes spike-with-recovery
from grind-down (e.g. VIX *level* AND term-structure/realised-vol slope) might
keep the 2022 averted loss while dropping the 2020 forgone gain.

## Reproduce

```
python docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_dollar_impact_driver.py \
    --universe 100 --start 2020-01-02 --end 2024-12-31 --out-dir <tmp>/r11_w3
python .../r11_dollar_impact_driver.py --universe 100 --start 2021-01-04 --end 2025-12-31 --out-dir <tmp>/r11_w4
python .../r11_dollar_impact_driver.py --analyze --out-dir <tmp>/r11_w3   # re-emits the tables
```

Captured companions in this folder: `r11_w3_analysis_RAW.txt`,
`r11_w3_summary.json`, `r11_w4_analysis_RAW.txt`, `r11_w4_summary.json`. The
large per-row `rank_log.csv` stays in the scratch out-dir (not committed).
