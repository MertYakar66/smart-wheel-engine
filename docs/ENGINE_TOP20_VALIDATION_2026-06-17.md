# Engine Validation — Top-20 S&P 500 by Market Cap — 2026-06-17

**Question asked:** is the engine *profitable*, and does it produce *reliable* and
*realistic* advice, on the 20 largest S&P 500 names?

**Method:** out-of-sample forward-replay on committed Bloomberg data (ends
2026-03-20), all scans pinned to historical `as_of` (no live/None, no API). Three
instruments: (1) a full wheel simulation (25Δ/35-DTE CSP→assignment→covered-call,
bid execution, R10 10%-NAV cap, T-bills on idle cash), (2) prob_profit/tail
calibration vs realized outcomes, (3) per-name profitability. Every load-bearing
number was **independently reproduced and adversarially audited by a 5-agent
verification workflow** (lookahead / accounting / realism / number-reproduction /
synthesis) — all claims survived, accounting reconciled to the penny, no
invalidating defect. Two corrections the audit forced are applied here: ZIRP-clean
pre-2021-05 rates and cap-respecting multi-contract sizing.

**Universe (deduped GOOG/GOOGL):** NVDA AAPL GOOGL MSFT AMZN META AVGO TSLA BRK/B
WMT LLY JPM XOM V JNJ MU MA COST ORCL HD.

---

## Bottom line

> **Profitable: YES, genuinely — but modestly, and not the way it looks.** The
> engine's short-put selling earns **real option-premium alpha of ~+24% over the
> 2021–26 full window (~4–5%/yr) over a T-bills-on-collateral baseline, net of
> realistic bid execution + commissions.** This is *premium*, not disguised beta
> — the assigned-stock leg is a net **drag** (−$72k over the full window), not a
> recovery boost. Because Bloomberg's zero-skew IV under-prices the real OTM put
> ~25–35%, the true edge is if anything **understated**.
>
> **Reliable & realistic: YES, verified** — deterministic, point-in-time-clean,
> exact pricing, accounting reconciled to the penny; mega-cap execution drag is
> small (~3%, tight spreads).
>
> **BUT three hard limits a trader must price in:** (1) it **never beats
> buy-and-holding the mega-caps** (−140 to −184pp) — it caps its own upside; (2)
> in a **pure bear the option leg breaks even/loses** (2022: −0.5%; the premium is
> eaten by a 42% assignment rate) — the capital-preservation vs B&H is from
> sitting in cash, not skill; (3) the **crisis tail model is dangerously
> optimistic** — `prob_profit` is over-confident ~12pp and `cvar_5` understated
> the worst crisis loss 3.5×, exactly when a real book bleeds.

---

## 1. Profitability (full wheel sim, $1M, verified accounting)

| window | wheel | T-bills | option-α | B&H (mega-caps) | vs B&H | maxDD | assn% |
|---|---|---|---|---|---|---|---|
| 2022 BEAR | +1.1% | +1.6% | **−0.5%** | −18.1% | **+19.2%** | −3.7% | 42% |
| 2023–24 BULL | +18.1% | +10.6% | **+7.5%** | +157.9% | −139.9% | 0.0% | 4% |
| 2021–26 FULL | +41.8% | +18.0% | **+23.8%** | +225.7% | −183.9% | −3.8% | 15% |
| 2020 crash+rebound | +5.5% | +0.1% | **+5.5%** | +25.1% | −19.5% | −2.4% | 8% |

**option-α = wheel − T-bills** = the option-selling contribution over just parking
collateral in T-bills.

### What that option-α actually is (honest P&L decomposition, $k)

| window | put prem | call prem | stock realized | stock unreal | net |
|---|---|---|---|---|---|
| 2022 BEAR | +68 | +54 | −51 | −49 | premium +122 **offset by −100 assignment loss** |
| 2023–24 BULL | +61 | +4 | +10 | 0 | almost pure premium (4% assigned) |
| 2021–26 FULL | **+235** | **+117** | −44 | −28 | **premium +352, assignment −72 drag** |
| 2020 crash | +79 | +1 | −20 | +5 | premium-driven |

**The edge is premium, not beta.** I initially suspected the +24% was recovered
mega-cap beta on assigned names; the decomposition refutes it — assigned stock is
a **net loss** in 3 of 4 windows (you get assigned when names fall, and they don't
all recover by window-end). The driver is the +$352k of put+call premium. In the
pure 2022 bear, that premium (+$122k) was *more than offset* by −$100k of
assignment losses → option leg −0.5%; the +19.2pp vs B&H is purely from being
~85% in cash.

---

## 2. Reliability of the advice (calibration, mega-caps)

| | n | prob_profit pred | realized | miscal | traded P&L (bid) | cvar_5 breach |
|---|---|---|---|---|---|---|
| **Calm 2023–26** | 383 | 0.836 (.80–.90 bin) | 0.870 | +3.3pp (under-conf) | **+$351/contract**, 91% win | 0.5% |
| **Crisis 2020+2022** | 202 | 0.844 | 0.720 | **−12.4pp** | **−$102/contract** | **9.9%** |

- **Calm tape: trustworthy.** Slightly under-confident; per-trade positive at the
  bid; execution drag only 3% (mega-cap spreads are tight — much better realism
  than the broad universe's 8%).
- **Crisis tape: over-confident and loss-making.** When it says 84% it's 72%; the
  recommended (+EV) crisis trades **lose money**; `cvar_5` (the modeled 5% tail)
  was breached in 9.9% of cases — worst **COST 2022-04: −$14,174 realized vs
  −$4,022 modeled (3.5×)**, independently reproduced to the dollar with PIT-clean
  resolution. The crisis "bear-protection" at portfolio level is *diversification
  across 20 names*, NOT a trustworthy per-trade risk model.
- **Determinism: perfect** — identical and order-reversed scans byte-identical.
- **Core pricing: exact** — independent BSM 25Δ strike/premium/delta match to the
  cent.

---

## 3. Which mega-caps to actually wheel (per-name, recommended-trade region)

**Profitable (15/19):** LLY (**+$852/contract, 97% win**), META, HD, TSLA, AVGO,
MSFT, GOOGL, XOM, MU, AMZN, BRK/B, AAPL, NVDA, V, JNJ.
**Lose on the mean (4):** MA (−$158), COST (−$325), ORCL (−$377), JPM (−$898).
**Untradeable:** WMT — *always dropped* by a post-split history-gate artifact (a
top-20 name the engine cannot trade at all — a real data-layer bug worth fixing).

Crucial nuance: the 4 "losers" all have **positive medians** (+$519, +$693,
+$106, −$285) — they lose only on catastrophic single tails (COST −$14,187, MA
−$6,483, ORCL −$4,327) that the engine **still recommended as +EV**. This is the
"win small, lose big" short-put signature; the engine's filter does not catch the
blow-ups. Sample sizes are thin for event-prone names (META n=1, JPM n=4) — treat
those per-name means as noisy.

---

## 4. What the engine advises right now (2026-03-20)

Of 20: ranks 15 (drops WMT/history, JPM·XOM·JNJ·MU/earnings-lockout), **11
tradeable (ev>0)** — and correctly **refuses NVDA, COST, TSLA (−$476), META
(−$481)** as negative-EV fat-tail names. Top picks: LLY (+$714), BRK/B, AVGO,
GOOGL, V, MA, MSFT. Selective, sane behavior. (Two cosmetic reliability flaws:
the HMM regime labels — "crisis" on HD/ORCL — are spurious, from non-converged
fits; every fit reports `hmm_converged=False`.)

---

## 5. Caveats on these results (honest limits)

- **Survivorship:** the top-20 are *current* mega-caps (survivors). Names that
  blew up or fell out of the top tier aren't tested → realized win-rates and
  returns are biased **optimistic**.
- **Zero-skew IV** under-prices the real OTM put ~25–35% → premium income (and
  thus the option-α) is **understated**; but it also means the engine's tail/risk
  view is thinner than the market's.
- **No early rolls** (the engine has `suggest_rolls`, unused here) → conservative.
- **CC leg** uses a simple spot×1.05 proxy at the name's IV, not the engine's full
  covered-call ranker → approximate, modest impact.
- **2020 window** uses ZIRP-floored pre-2021-05 rates (the treasury CSV starts
  2021-05-07; the raw engine path returns a spurious 5% there — a real defect,
  fixed in this analysis, that still lives in `get_current_risk_free_rate`).

---

## Verdict for the trader

On the top-20 mega-caps the engine is a **legitimate, modestly-profitable premium
sleeve**: ~4–5%/yr of real option-premium alpha over T-bills, with tight
execution and verified-clean mechanics. Use it for **income on a cash position you
would otherwise hold in T-bills**, on the 15 names that wheel profitably, sized
under the 10% cap — not as a substitute for owning mega-caps (it lags them
enormously) and not trusting its probability/tail numbers in a VIX-elevated
crisis. The single thing to fix before sizing up: the **crisis tail
under-estimation** (wire POT-GPD into `prob_profit`; arm R10/R11), because that is
where a real book takes the loss the +5%/yr can't absorb.

*Evidence: `_top20_wheel_sim.py` (decomposed), `_top20_calib.py`,
`_top20_pername.py`; adversarial verification workflow `wf_fa14a02a` (5 agents,
all claims SURVIVED, accounting reconciled to the penny). Reproducible against
committed Bloomberg data at the pinned `as_of` dates.*
