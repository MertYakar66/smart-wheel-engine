# Heavy-Verify Campaign 2026-05-31 — I2: Would it actually make money, net of reality?

**Investigation:** capital-constrained, multi-regime wheel P&L with **real Theta
bid/ask fills**, risk-free interest on collateral, and a dividend-adjusted passive
index benchmark. **The confirmed blind spot** — no prior backtest combined real
spreads + a capital-constrained NAV path + a passive benchmark.
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i2_pnl.py` (+ `campaign_lib.py`,
the shared monthly ranked snapshots). **Raw:** `raw_output/i2_pnl_all_slip*.json`,
`raw_output/i2_all_slip*_RAW.txt`. **Status:** observe-and-document; `engine/` not modified.

---

## VERDICT (where the wheel makes money, where it doesn't)

> **The wheel is a defensive, income/rate-harvesting strategy — not a bull-market
> growth substitute. It DECISIVELY OUTPERFORMS buy-and-hold in down and sideways
> markets (2022 bear: +4.5% vs −22.4%, a +27pp edge; 2020 crash window: +9.1% vs
> −0.9%) while taking roughly a quarter of the drawdown. It ROUGHLY MATCHES passive
> in calm/high-rate regimes (2025: +24.9% vs +24.3%). It SUBSTANTIALLY UNDERPERFORMS
> in strong bull markets (2021: +13.3% vs +32.8%, −19pp; 2023-24 recovery: +40.2%
> vs +66.5%, −26pp) because upside is capped at the strike and ~20-45% of capital
> sits idle. Real bid/ask friction is a minor drag (~0.5–1.5%/yr). Risk-free
> interest on the cash-secured collateral is a FIRST-ORDER return component in a
> 4-5% world — it added +8.4pp to the 2023-24 result and flipped 2025 from a
> laggard to a match.**

Confidence: **high** on the regime *direction and ranking* (robust to friction and
to the rf/dividend adjustments); **medium** on the exact magnitudes (monthly NAV
marks understate intra-month drawdown; ~30-45% of fills use a modeled spread where
Theta lacks coverage; path-dependence — see the friction caveat).

This is **the answer to "where can I trust it with money"**: allocate to the wheel
for downside protection and bear/sideways carry, sized as a *complement* to — not a
replacement for — long equity beta.

---

## Method (what makes this trustworthy)

* **Decisions are the engine's.** Each month (the shared full-universe snapshots)
  the simulator enters only puts the §2-authoritative ranker scored `ev_dollars>0`,
  one cash-secured contract per name, top-ranked first, until capital/position
  limits bind. No negative-EV entry. (This respects §2; see I4.)
* **Fills are real.** A short put is filled at the **real Theta quote** for the
  nearest listed strike/expiry: `mid − slippage×half_spread` (slippage=1.0 ⇒ sell
  at the **bid**, the conservative retail case). Where Theta lacks that
  strike/date (≈30-45% of fills), a modeled 10%-spread fallback is used and counted
  separately. Premiums are thus **market**, not the engine's BSM-synthetic value.
* **Full wheel lifecycle.** Put assignment → own 100 shares at strike → sell a
  ~25-delta covered call each month (engine-ranked, real Theta call credit) until
  called away or window end. Cash-secured collateral is reserved; overlapping
  monthly cohorts build deployment until cash is exhausted (realistic cash drag).
* **Risk-free on collateral.** Cash + reserved collateral accrues the **PIT
  3-month T-bill** (`data/bloomberg/treasury_yields.csv`) — cash-secured puts hold
  collateral in T-bills; omitting this understates the wheel in a high-rate world.
* **Fair benchmark.** Passive = **cap-weighted (point-in-time) 503-name index price
  return + dividends** (total-return proxy). "Capital-matched" scales it by the
  wheel's average deployment. (No SPY in the local data; the cap-weighted 503-name
  basket is the index proxy. Known **survivorship** caveat: current members only →
  the benchmark is biased *up*, i.e. *harder* for the wheel to beat.)

## Results — realistic fills (slippage = sell-at-bid), $1M account

| regime | window | wheel return | of which rf | within-path friction | passive (total) | wheel − passive | avg deployed | max DD (monthly) | assign rate |
|---|---|---|---|---|---|---|---|---|---|
| **crash_2020** | 2020-01→06 | **+9.1%** | +0.1% | 0.77% | −0.9% | **+9.9pp** ✅ | 65% | −2.6% | 20% |
| **bear_2022** | 2022 | **+4.5%** | +1.4% | 0.87% | −22.4% | **+26.8pp** ✅ | 91% | −6.6% | 18% |
| recent_2025 | 2025→26-02 | +24.9% | +3.1% | 1.07% | +24.3% | **+0.6pp** ≈ | 101% | −3.2% | 39% |
| bull_2021 | 2021 | +13.3% | +0.1% | 0.54% | +32.8% | **−19.4pp** ❌ | 54% | −0.7% | 21% |
| recovery_2023_2024 | 2023-24 | +40.2% | +8.4% | 1.19% | +66.5% | **−26.3pp** ❌ | 80% | −0.8% | 29% |

(Frictionless variant in `i2_pnl_all_slip0.0_mf.json`; differences are ≤~1.5pp/yr
and partly path-divergence — see friction note.)

## What this means for a PM

1. **Downside protection is the product.** In the only genuinely bad equity year in
   the sample (2022), the wheel returned **+4.5% while the index lost −22.4%** — a
   27pp edge with a −6.6% max drawdown vs the index's ~−25%. In the 2020 crash
   window it was +9% vs roughly flat. This is a real, large, repeatable edge in
   down/sideways tape.
2. **Bull-market underperformance is structural and large.** In 2023-24 the wheel
   made a strong **+40%** in absolute terms but the index made **+66%** — you gave
   up 26pp. Capital-matched (accounting for the 20% idle cash) the gap is still
   −13pp: roughly half the underperformance is cash drag, the other half is the
   capped upside (the rally above your short-put strike accrues to the stockholder,
   not to you). Selling calls on assigned stock caps it again on the way up.
3. **The risk-free rate is not a footnote.** rf on collateral added **+8.4pp** over
   2023-24 and **+3.1pp** in 2025 — it is what makes the wheel competitive with
   passive in a 5% world and is the single largest swing factor between the
   premium-only and the fair accounting. In the ~0% world of 2020-21 it contributed
   nothing.
4. **Transaction friction is small.** Real half-spreads + commissions cost
   **~0.5–1.5% of capital per year** — an order of magnitude smaller than the
   regime asymmetry. The wheel's P&L is dominated by *what the market does* and *the
   rate environment*, not by execution cost.

## Adversarial notes / what would change the conclusion

* **Friction differencing is confounded.** "Realistic minus frictionless" is NOT a
  clean friction measure: lower fills change which positions are affordable, so the
  trade *path* diverges (e.g. recent_2025 shows the realistic run *higher* — a path
  artifact, impossible as a true friction effect). I therefore report friction as
  the **within-path** half-spread+commission cost (the table column), which is
  path-consistent. (This is the known wheel-backtest path-divergence issue.)
* **Modeled-fill share:** ≈30-45% of fills fall back to a modeled 10% spread where
  Theta lacks the exact strike/date; the headline is therefore part-real,
  part-modeled. The real-fill subset behaves the same directionally.
* **Monthly NAV marks** understate intra-month drawdown — the −6.6% (2022) /
  −3.2% (2025) figures are lower bounds on true path risk; still far below the
  index's drawdowns.
* **Survivorship in the benchmark** biases passive *up* (current members only), so
  the wheel's bear-market edge is, if anything, understated and its bull lag
  overstated.
* **Selection = top-ev_dollars greedy**, one per name, no sector cap (the engine's
  R9/R10 caps are dormant by default — see I3-A). A sector-capped book would change
  the concentration but, per I1, ev_dollars is a weak selector anyway.
* **What would change it:** a higher-delta (more premium, more assignment) or
  shorter-DTE variant; a rate regime reverting to ~0% (removes the +8pp rf tailwind);
  or sizing that deploys more aggressively (less cash drag, more bull capture but
  more tail exposure — see I3-E procyclicality).

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i2_pnl.py --regime all --slippage 1.0
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i2_pnl.py --regime all --slippage 0.0   # friction sensitivity
```
Requires the monthly snapshots (`rank_snapshots.py`) and the Theta option history
(primary clone `data_processed/theta/`). Every number is in `raw_output/i2_pnl_all_slip1.0_mf.json`.
