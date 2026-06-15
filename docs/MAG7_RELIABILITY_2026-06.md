# MAG7 Wheel Reliability Study — $200k, eight regime-diverse one-year windows (2026-06-14)

**Author:** autonomous quant session (brain + executor). **Engine:** `origin/main` @ `83eacdd`
(worktree `swe-mag7`, branch `claude/mag7-reliability`). **Provider:** `MarketDataConnector`
(Bloomberg CSVs). **Driver:** `backtests/regression/mag7_reliability.py`. **Analysis:**
`scripts/mag7_analysis.py`. **Artifacts:** `%TEMP%/mag7_backtest/<window>/{none,bid_ask,full}/`
(deterministic, regenerable; not committed per the S43 convention).

---

## 1. Executive summary

Eight one-year wheel campaigns (short cash-secured puts → covered calls), each starting at
**$200,000** and trading **only the Magnificent Seven** (AAPL, MSFT, GOOGL, AMZN, META, NVDA,
TSLA), entered at eight regime-diverse past dates. The question: *is the engine reliable and
realistic when the universe is collapsed to seven highly-correlated mega-caps?*

**Headline (full-friction): 7/8 windows positive, mean +7.6%, median +8.3%, worst −17.1%
(the 2022 bear — which still beat Mag7 buy-and-hold by +30pp).** Three findings dominate, all
verified against source (§9):

1. **The wheel trades upside for a downside cushion.** It *lagged* Mag7 buy-and-hold in every
   bull window (−44 to −92pp) and *beat* it only in the 2022 bear (+30pp). On high-momentum
   growth names the upside-capping cost of covered calls is severe.
2. **The engine's selection edge holds, but only where it acts.** Across the *traded* region
   (ev>0 — the only candidates it opens), EV→realized rank correlation is positive and
   significant in all 8 windows (rho +0.19…+0.66). The naive full-sample rho is negative in
   5/8, but that is an artifact of the *non-traded* negative-EV region (see §8) — not a
   decision defect.
3. **A $200k Mag7-only book is structurally undeployable under the engine's own caps.** It
   routinely held all 7 names at once with single-name exposure peaking 18–35% of NAV — far
   above R10 (10% single-name) and R9 (25% sector; all 7 are one tech cluster). Caps-off was
   required to run it at all.

The engine produced **no catastrophic year** and behaved like a textbook wheel. But on Mag7 it
is a concentrated, idiosyncratic-risk bet that **underperforms simply holding the names** in
anything but a bear.

---

## 2. Methodology

- **Universe:** the seven Mag7 names (Alphabet via GOOGL; one share class to avoid
  double-counting). All present in the connector, OHLCV 2018-01-02→2026-06-04.
- **Capital:** $200,000. **Knobs (canonical, = S38/S43/SIM-200K):** `top_n=15` (non-binding —
  only 7 names), `max_new_per_day=3`, `dte_target=35`, `delta_target=0.25`, `contracts=1`,
  `seed=42`, `require_ev_authority=False` (caps-off).
- **Harness:** `run_backtest_multi_friction` (S38/S43 lineage), three friction levels
  (none / bid_ask / full); headline = full. Every tradeable path routes through
  `WheelRunner.rank_candidates_by_ev` → `EVEngine.evaluate` (no §2 bypass). Only positive-EV
  candidates are opened (`_tracker_try_opens`: `ev_dollars <= 0` is skipped).
- **Windows (same eight as the UNIVERSE_100 study, for comparability):**

  | id | window | regime |
  |---|---|---|
  | w1 | 2020-02-03 → 2021-01-29 | crash entry (COVID) |
  | w2 | 2020-06-01 → 2021-05-28 | recovery, rich IV |
  | w3 | 2021-01-04 → 2021-12-31 | calm bull |
  | w4 | 2022-01-03 → 2022-12-30 | rate-shock bear |
  | w5 | 2022-10-03 → 2023-09-29 | bottom entry |
  | w6 | 2023-07-03 → 2024-06-28 | chop into bull |
  | w7 | 2024-06-03 → 2025-05-30 | late-cycle bull |
  | w8 | 2025-03-03 → 2026-02-27 | most recent full year |

---

## 3. No-lookahead / point-in-time discipline

Identical to the UNIVERSE_100 study (same harness, Opus-reviewed PIT-clean there):

- Every rank call passes `as_of=today.isoformat()`; OHLCV / IV / HMM / forward-distribution are
  all sliced `<= as_of`. `EVEngine.evaluate` is a pure function of the as-of snapshot.
- Settlement uses the expiry-day close **only once the day-stepping loop reaches it**
  (forward-replay). No window extends past the 2026-06-04 OHLCV frontier.
- **Independently re-verified here (§9):** 107/108 closed legs settle consistently with the
  connector's *real forward close* (OTM⇔spot>strike for puts, ITM⇔spot>strike for assigned
  calls). The engine is not fabricating outcomes or peeking.

---

## 4. Honest limitations register (read before trusting any number)

- **Mag7 survivorship is the whole point — and a flatter.** All seven survived to 2026; the
  study trades the *known* winners. In 2020 you could not have known these seven would be "the
  Mag7." This inflates levels (less so the cross-regime *shape*).
- **Caps-off is unavoidable but unrealistic.** Armed R9/R10 would block almost every open on a
  7-name tech book (§7). Returns here are for the un-deployable caps-off config.
- **Zero-skew IV ⇒ conservative put premiums.** Bloomberg `vol_iv` is ATM-only
  (`put_iv == call_iv`), so the short-put premiums the engine prices are conservative vs a real
  skewed market (the wheel sells the put side). Realized returns are, if anything, understated
  on the premium-harvest side.
- **Synthetic / model premiums, not live NBBO.** Friction is modeled (none/bid_ask/full), not a
  real fill book.
- **Window overlap.** The eight windows overlap; they are regime samples, not independent draws.
- **$200k buying-power saturation.** Cash-secured collateral (strike×100) on $100–700 names
  limits the book to a handful of positions — which *is* the concentration finding (§7).

---

## 5. Results (full-friction)

| window | ret % | final NAV | maxDD % | trades | put-assign | call-assign | hit-rate |
|---|---|---|---|---|---|---|---|
| w1 2020 crash-entry | **+13.68** | $227,360 | −6.2 | 28 | 7 | — | 0.843 |
| w2 2020 recovery | +3.93 | $207,860 | −4.2 | 29 | 3 | — | 0.933 |
| w3 2021 calm bull | +7.32 | $214,645 | −5.0 | 16 | 2 | — | 0.851 |
| w4 2022 bear | **−17.06** | $165,880 | −24.4 | 10 | 8 | — | 0.635 |
| w5 2022 bottom | +9.22 | $218,440 | −3.6 | 17 | 3 | — | 0.871 |
| w6 2023 chop | +2.92 | $205,840 | −0.6 | 18 | 1 | — | 0.914 |
| w7 2024 late-cycle | +13.22 | $226,440 | −16.9 | 21 | 7 | — | 0.781 |
| w8 2025 recent | **+27.78** | $255,560 | −11.6 | 7 | 7 | — | 0.841 |

(Closed-leg exit mix across all windows: **100 puts expired OTM + 8 covered calls assigned**.)

**Rollup (full):** n=8, mean **+7.63%**, median **+8.27%**, σ 12.66pp, **positive 7/8**,
worst −17.06%, best +27.78%. Friction sensitivity is modest (none ≥ bid_ask ≥ full as expected);
the full-friction headline is the conservative cut.

---

## 6. Wheel vs. just holding Mag7 (the central trade-off)

| window | wheel ret % | equal-weight Mag7 B&H % | alpha (pp) |
|---|---|---|---|
| w1 2020 crash-entry | +13.68 | +105.52 | **−91.84** |
| w2 2020 recovery | +3.93 | +80.06 | −76.13 |
| w3 2021 calm bull | +7.32 | +51.19 | −43.87 |
| w4 2022 bear | −17.06 | −47.42 | **+30.36** |
| w5 2022 bottom | +9.22 | +65.88 | −56.66 |
| w6 2023 chop | +2.92 | +54.33 | −51.41 |
| w7 2024 late-cycle | +13.22 | +25.52 | −12.30 |
| w8 2025 recent | +27.78 | +28.13 | −0.35 |

The benchmark uses the connector's *true* close (not the rotated raw CSV column — the trap from
the UNIVERSE_100 study). The pattern is unambiguous and realistic: **the wheel sheds 44–92pp of
upside in bull markets** (covered calls cap exactly the explosive moves Mag7 is bought for) and
**adds +30pp only in the bear** (premium + assignment-at-lower-basis cushions the drawdown). w8
(+27.8% vs +28.1%) is the one window the wheel kept pace — the recent year's repeated
assignments (7) rode the up-move through stock rather than calls. **Conclusion: a Mag7 wheel is a
risk-reducer, not a return-maximizer; if you are bullish Mag7, the wheel is the wrong vehicle.**

---

## 7. Concentration — the structural finding (verified, recon 91–99% vs tracker)

Daily book reconstructed from open + closed positions (collateral = strike×100 per name), then
**validated against the tracker's own per-day `num_positions`** (match 91.7–99.2% across
windows; residual = entry-day timing).

| window | peak single-name % NAV | peak name (date) | peak # positions | mean # positions |
|---|---|---|---|---|
| w1 | 17.9 | MSFT (2020-09-01) | 7 | — |
| w2 | 24.2 | — | 7 | — |
| w3 | 29.4 | — | 7 | — |
| w4 | 18.0 | MSFT (2022-12-28) | 7 | — |
| w5 | 22.7 | — | 7 | — |
| w6 | 34.9 | — | 7 | — |
| w7 | 33.2 | — | 7 | — |
| w8 | 32.3 | — | 7 | — |

The book **routinely held all seven names at once**, with single-name exposure peaking
**18–35% of NAV** — every window breaches R10 (10% single-name) and R9 (25% sector; all seven
sit in Info-Tech / Comm-Services / Consumer-Disc). **A $200k Mag7-only wheel cannot be run with
the engine's production concentration gates armed.** This is not a tuning issue; it is the
arithmetic of seven high-priced, same-sector names against $200k of cash-secured collateral.

---

## 8. Calibration (pooled 7,436 candidate-rows)

**prob_profit reliability** (win = hold-to-expiry realized_pnl > 0):

| prob_profit bin | n | predicted | realized | gap (pp) |
|---|---|---|---|---|
| 0.5–0.7 | 212 | 0.671 | 0.802 | +13.0 |
| 0.7–0.8 | 2,134 | 0.757 | 0.858 | +10.0 |
| 0.8–0.9 | 4,116 | 0.842 | 0.808 | −3.4 |
| 0.9–0.95 | 885 | 0.917 | 0.908 | −0.9 |
| **0.95+** | 89 | 0.965 | 0.764 | **−20.1** |

The top bin is **over-confident by −20pp** — the same engine-wide defect the Block-B
recalibration targets (now confirmed on Mag7). Lower bins are under-confident.

**EV-ranking edge — the decision-relevant cut.** Naive full-sample `spearman(ev_dollars,
realized)` is slightly negative (−0.065) and turns significantly negative in w5/w6/w7/w8. **This
is a measurement artifact, not a defect:** the EV-decile table is monotonic at the *top*
(decile 8/9/10 realized +109/+113/+229) and inverted only at the *bottom* (decile 1, EV −492,
realized +85.9) — i.e. among candidates the engine **declines** (ev≤0), its tail-pricing
conservatism anti-correlates with the benign bull-era outcomes that didn't realize the priced
tails. Restricting to the **traded region (ev>0)**:

| window | n (ev>0) | rho on ev>0 | p | mean realized (ev>0) |
|---|---|---|---|---|
| w1 | 561 | +0.550 | <1e-3 | +84.93 |
| w2 | 464 | +0.580 | <1e-3 | +229.50 |
| w3 | 292 | +0.580 | <1e-3 | +129.10 |
| w4 | 536 | +0.485 | <1e-3 | +17.08 |
| w5 | 255 | +0.661 | <1e-3 | +164.72 |
| w6 | 143 | +0.191 | 0.022 | +218.60 |
| w7 | 311 | +0.188 | <1e-3 | +24.71 |
| w8 | 345 | +0.193 | <1e-3 | +134.02 |
| **pooled** | **2,907** | **+0.466** | ~1e-156 | **+112.89** |

EV ranking is **positive and significant in all 8 windows** on the region the engine acts on;
mean realized for ev>0 (+$112.89) is ~7× that of the declined ev≤0 set (+$16.05). The edge is in
P&L *magnitude*, not win-rate (both ~0.83). **The selection edge holds for Mag7.**

---

## 9. Verification (verify, don't hallucinate)

| check | method | result |
|---|---|---|
| **Data integrity** | split-continuity scan at all 7 Mag7 split dates (AAPL/TSLA 2020, AMZN/GOOGL/TSLA 2022, NVDA 2021/2024) + overnight-ratio seam scan | **CLEAN** — every split shows ratio ≈ 1.0 (adjusted); zero BKNG/CVNA-style unadjusted seams |
| **No-cheat settlement** | re-derive every closed leg's OTM/assigned outcome vs the connector's real forward close (per-leg put/call logic) | **107/108 consistent** (the 1 "miss" is a verification-side notes parse of META's stale put strike; real call strike <spot, consistent) |
| **EV-edge realism** | rho restricted to traded ev>0 region, per-window + pooled, with p-values | **+0.19…+0.66, all significant** (§8) — selection edge real |
| **Calibration honesty** | pooled prob_profit reliability + EV-decile | top-bin over-confidence (−20pp) confirmed; not hidden |
| **Determinism** | re-run w4 to a fresh dir, compare per-friction final_nav / executed_trades / put_assignments / row_count / spearman_rho | **PASS** — fresh re-run reproduces campaign w4 **exactly**: final_nav identical to 11 dp across all 3 frictions (e.g. full $165,879.01755105302), trades/assignments/row_count/rho bit-identical |
| **Concentration trust** | reconstruction vs tracker `num_positions` | 91.7–99.2% match across windows |

No connector mutation; pull untouched; decision trio untouched; §2 only asserted.

---

## 10. Realism check

The results pass a professional smell test. A Mag7 wheel **should** (a) make modest single-digit
to low-double-digit returns in normal years harvesting rich tech IV, (b) get hurt but cushioned
in a bear (w4 −17% vs −47% B&H), (c) badly trail a melt-up (w1 +14% vs +105%), and (d)
occasionally keep pace when assignments ride the up-move (w8). All four appear. Premiums are
conservative (zero-skew IV), so live put-side results would likely be *richer*, not poorer.
Drawdowns (max −24% in 2022) are realistic for an assigned Mag7 book in that year.

---

## 11. Verdict

For the question asked — *is the engine reliable and realistic on Mag7?* — **yes, with two
honest qualifications.** The engine produced positive outcomes in 7/8 regimes, never a
catastrophic year, with a real and significant EV-selection edge on the trades it actually takes,
correct PIT settlement, clean split-adjusted data, and deterministic, reproducible results.

But **a $200k Mag7-only wheel is the wrong product**: it is structurally undeployable under the
engine's own concentration caps (§7), and as a *strategy* it sheds most of Mag7's upside to buy a
downside cushion (§6) — rational only for an investor who explicitly wants lower variance than
holding the names, and who accepts single-name concentration the risk gates are designed to
forbid. The engine is reliable; the Mag7-only *mandate* is the risk.

Recommended next probes (not run): (i) armed-caps Mag7 to quantify how few trades survive R9/R10;
(ii) a diversified-vs-Mag7 A/B at equal capital; (iii) Block-B recalibration's effect on the
0.95+ prob_profit bin.
