# Soundness review — engine reliability second-pass (2026-05-26)

**Reviewer:** Terminal B, fresh review of own session's earlier work.
**Question (from user):** *"review the results again to make sure our
options engine is sound."*
**Origin/main SHA at review:** `2da76ff`.

This file is a **second-pass critical re-verification** of the
session's earlier findings. It does NOT replace
`docs/LAUNCH_READINESS_ANALYSIS_2026-05-26.md` or
`docs/PRODUCTION_READINESS.md` — it audits them. **Specifically: are
the conclusions defensible? Are the numbers as advertised? Are there
hidden risks the earlier analysis missed?**

---

## Headline verdict — engine IS mechanically sound, with important
## framing correction

**Mechanically: ✅ sound.** §2 invariant intact, launch-blocker tests
green, P&L formulas internally consistent across all backtests,
5-ticker EV smoke clean, refusal mechanism works correctly in
adverse periods.

**Statistically: ✅ sound.** Spearman ρ ranges 0.19–0.50 across all
measurements, statistically overwhelming in every case
(p < 1e-48). Signal is robust to (capital × universe × window).

**Critical framing correction discovered in this review.** Earlier
session docs framed S34's +35.6% NAV at $1M as primarily driven by
the engine's put-selection edge. **The re-analysis shows ~92% of
the NAV gain in S34 (and ~100% in S27) comes from equity beta on
ASSIGNED stocks, not from put-premium selection skill.** This is a
property of the wheel strategy (capture equity beta via
assignments) — not a defect — but it must be honestly framed for
any deployment conversation.

**One concentration risk surfaced.** In S34, **BKNG alone
contributed $31,576 of $28,571 total executed realized P&L** —
without BKNG the engine's realized executions are slightly negative
($-3,004). The signal (ρ) is robust to removing BKNG (0.327 → 0.324)
so the ranking quality holds; but the **dollar alpha is highly
sensitive to a few high-priced ticker outliers**.

---

## 1. What I re-verified

### 1.1. §2 invariant (launch-blocker subset)

`pytest tests/test_audit_invariants.py tests/test_dossier_invariant.py
tests/test_authority_hardening.py tests/test_audit_viii_unit_invariants.py
tests/test_audit_viii_e2e.py tests/test_audit_viii_real_data_smoke.py
tests/test_launch_blockers.py`

→ **93 passed, 0 failed in 16.59s.** The 2 warnings are intentional
(`reliable=False` Greeks-residual gate firing as designed). **Pillar
green.**

### 1.2. 5-ticker EV smoke (CLAUDE.md §6)

```
connector: MarketDataConnector  (Bloomberg)
rows: 5
  ticker  ev_dollars      iv  premium  prob_profit  hmm_regime
3    XOM      137.57  0.3216    2.472       0.8857  bull_quiet
2    JPM      124.90  0.3255    4.559       0.8571        bear
1   MSFT       90.97  0.3383    6.303       0.8286        bear
4    UNH       62.62  0.4347    5.956       0.8857        bear
0   AAPL       20.45  0.3079    3.700       0.8571        bear
NaN counts: 0 / 0 / 0
```

All EVs positive, IV in plausible 0.31-0.43 range, prob_profit in
plausible 0.82-0.89 range. **No anomalies.**

Note: AAPL/MSFT moved from earlier-session negative EVs to positive.
Several engine PRs landed on `main` between sessions (#220, #222,
#227 in particular). The engine state I tested for S22/S27/S32/S34/S35
backtests is slightly behind current `main`; that's expected.

### 1.3. P&L formula consistency across executed rows

Independent recomputation per harness formulas:

| Backtest | Executed rows | Mismatches (|delta|>$0.50) | Status |
|---|---|---|---|
| S27 | 77 | 0 | ✅ |
| S32 | 135 | 0 | ✅ (using `premium_adj`) |
| S34 | 277 | 0 | ✅ (using `premium_adj`) |
| S35 | 43 | 0 | ✅ (using `premium_adj`) |

**Zero P&L formula mismatches across 532 executed rows in four
backtests.** The mechanical accounting is internally consistent.

### 1.4. Signal consistency (Spearman ρ across backtests)

| Backtest | Config | N | ρ (full friction, puts) | p |
|---|---|---|---|---|
| S27 | $100k / 24t / 2022-2024 | 6,163 | 0.2183 | 2.3e-67 |
| S32 | $1M / 24t / 2022-2024 | 5,743 | 0.1918 | 9.8e-49 |
| **S34** | **$1M / 100t / 2022-2024** | **10,315** | **0.3273** | **4.8e-256** |
| S35 | $100k / 24t / 2018-2020 | 1,946 | 0.4970 | 6.6e-122 |

**The signal is real and consistent.** ρ is positive and statistically
overwhelming in every measurement. The variation (0.19 to 0.50) is
driven by window characteristics, not by engine soundness.

**Spearman without BKNG (S34):** 0.3244 (vs 0.3273 with). **The
ranking signal does NOT depend on BKNG.** Removing the largest
P&L contributor barely moves ρ.

---

## 2. NEW critical findings from this review

### 2.1. Equity beta dominates the dollar alpha

S27 and S34 NAVs grew substantially, but the realized P&L from
executed trades tells a different story:

| Backtest | NAV gain | Realized P&L (executed) | Equity-beta residual |
|---|---|---|---|
| S27 ($100k 24t) | +$51,444 | **−$3,421** | +$54,865 |
| S32 ($1M 24t) | +$18,514 | +$15,290 | +$3,224 |
| **S34 ($1M 100t)** | **+$356,128** | **+$28,571** | **+$327,557 (92% of gain)** |
| S35 ($100k 24t 2018-2020) | +$3,566 | −$48,326 | +$51,892 (gain from assignments) |

**S27 and S35 had NEGATIVE realized P&L from executed trades** —
the engine's put-selection alpha was actually negative in both
windows. **All the NAV growth came from STOCK APPRECIATION on
positions that got assigned and were then held.**

**For S34, the largest "win"**, only 8% of the +$356k NAV gain came
from realized executions; 92% came from being long stocks during a
bull market via the wheel-strategy assignment mechanism.

**Interpretation:**

This is a property of the wheel strategy by design, not a defect.
But it should reframe the engine's value proposition:
- The engine's **put-selection EDGE** is small per trade (positive
  ρ statistically; near-zero dollar contribution per executed
  trade in aggregate).
- The engine's **stock-selection** (which stocks to ACCEPT
  ASSIGNMENT on, via writing puts on them) is the primary alpha
  driver in bull markets.
- The engine's **refusal mechanism** is the primary value in bear
  markets (S35 / 2022 in S34).

**The "+27pp over SPY" headline from S22/S27** is therefore largely
a claim about **selecting the right stocks to be assigned**, not
about premium-selling skill per se. The two are correlated in
practice (engine ranks high-EV puts on stocks it implicitly likes)
but the framing matters.

### 2.2. Concentration risk in S34

| Tier | S27 | S32 | S34 |
|---|---|---|---|
| Total executed realized | −$3,421 | +$15,290 | **+$28,571** |
| Top-1 contributor | MSFT +$2,568 | MSFT +$3,650 | **BKNG +$31,576** |
| Top-1 share | n/a (total neg) | 23.9% | **110.5%** |
| Top-3 share | n/a | 60.5% | 129.8% |
| Sum of positive contributors | +$6,916 | +$16,632 | +$55,059 |
| Sum of negative contributors | −$10,338 | −$1,342 | **−$26,488** |

**S34's positive alpha is dominated by BKNG.** A single ticker
contributed more than the entire net total. Removing BKNG flips
the engine's realized P&L from +$28,571 to **−$3,004**.

**Why this happened (BKNG deep-dive):**

- BKNG's stock rose from ~$2,336 in Jan 2022 to ~$5,000+ by 2024
  (massive bull run).
- BKNG's high price means each put has premium scaled by spot (~$30–
  $50 per contract = $3k–$5k per trade).
- The engine wrote 9 BKNG puts over the window. 7 expired OTM
  (premium captured). 2 assigned (small loss + held stock).
- Premium income per executed BKNG trade averaged $3,508.
- The engine's BKNG **within-ticker ρ = 0.156** — modest. The engine
  didn't have unusual skill at TIMING BKNG entries; it just kept
  selling puts that BKNG kept making OTM as the stock rose.

This is **a high-priced-ticker leverage effect**, not pure
put-selection alpha. The engine systematically ranks BKNG high
(because of its IV and price level), the strategy collects scaled
premiums, and BKNG's bull trajectory turned all those premiums into
realized gains.

**Implication for deployment:** in a market where high-priced
mega-cap tickers (BKNG, AVGO, ORCL, etc.) underperform, the engine's
dollar alpha would be much lower. The S34 +35.6% NAV is partly a
**bet on high-priced ticker outperformance**, not engine skill alone.

### 2.3. Engine refusal behavior IS strong

| Period | Candidates | Executed | Refusal rate | Mean realized of all candidates |
|---|---|---|---|---|
| S35 COVID (2020-02-15 → 2020-05-15) | 482 | 1 | **99.8%** | −$370 |
| S34 2022 bear (Jan–Oct) | 2,760 | 59 | **97.9%** | −$193 |

**The engine refused ~98% of candidates in adverse periods.** And
the refusals were correct in aggregate: if the engine had executed
blindly on every candidate in S35's COVID window, it would have
averaged −$370 per trade × 482 trades = −$178,000 of losses on
$100k capital.

**The refusal mechanism is the engine's most defensible property
for any production deployment.** It's the layer that genuinely
prevents catastrophic losses.

---

## 3. Re-revised honest framing for the user's question

The earlier session docs framed the engine as:
> *"Engine signal is real, scale-invariant, AND window-invariant.
> Dollar-alpha is window-dependent."*

That's mechanically correct. But a **more complete framing** after
this re-review:

1. **The ranker IS good.** ρ = 0.22–0.50 across all measured
   configurations. Statistically overwhelming. Robust to removing
   the largest contributor. The signal is genuine.

2. **The refusal mechanism IS strong.** 98–99.8% refusal in adverse
   periods. Refusals are correct in aggregate (candidates the
   engine refused averaged $-200 to $-400 per trade if blindly
   entered).

3. **Per-trade dollar edge from put selection is small.** S34's
   executed mean realized of $179 per put × 180 puts = $32k total
   — modest. S27's was actually NEGATIVE.

4. **Most of the NAV growth comes from equity beta on assignments.**
   In 2022-2024, the assigned stocks (megacaps) outperformed SPY.
   That's the engine's real "alpha source" at the dollar level.

5. **Concentration risk is real.** S34's positive alpha depends
   heavily on BKNG. A 2020-style market where BKNG underperforms
   would produce a different outcome.

6. **Window sensitivity (from S35) compounds with the above.**
   2018-2020's −41pp under SPY isn't just bad luck — it's a regime
   where the equity-beta path didn't work (COVID crash + recovery
   was bad for the wheel strategy's assignment dynamics).

### Bottom-line deployment framing

**The engine is mechanically sound, statistically defensible as a
ranker, and operationally strong as a refusal mechanism.** As a
research signal or supervised decision-aid, it has real value.

**As a deployed dollar-alpha generator, its claims need careful
qualification:**
- The "+27pp over SPY" (S22/S27) and "+11.6pp over SPY" (S34)
  headlines are largely **levered equity beta on
  bull-market-favored single names**, not pure put-selection
  alpha.
- A market regime where the engine's preferred assignments DO
  NOT outperform SPY (e.g., 2018-2020 per S35, or a future
  regime where megacaps lag) would likely show the engine
  underperforming a simple SPY hold by 10-40pp.

**This isn't a "the engine is bad" finding.** It's a
"the source of the engine's apparent alpha is partly market
structure, not pure put-selling skill" finding. For honest
deployment, this needs to be in the disclosure.

---

## 4. What about my earlier session's documents?

The earlier docs (LAUNCH_READINESS_ANALYSIS_2026-05-26.md,
PRODUCTION_READINESS.md, ENGINE_BACKTEST_S34_UNIVERSE.md) are
**mechanically correct but interpretation-incomplete.** They frame
the engine's alpha as primarily from ranking skill. This re-review
shows the alpha is primarily from equity-beta-on-assignments, with
ranking skill being a smaller (but real) contributor.

**The earlier docs' deployment matrix conclusions still stand**:
- ✅ Research signal / supervised at ≤ $100k: still defensible
- ⚠ Supervised at $1M with 100-ticker universe: still conditional
- ❌ Autonomous at any scale: still no

**But the FRAMING of WHY** needs the equity-beta clarification.
Specifically:
- `PRODUCTION_READINESS.md` §1 "Headline status" should add a row:
  "Where does the alpha come from? Mostly equity beta on assigned
  stocks (2023-2024 bull) + refusal mechanism in bears + modest
  per-trade ranking edge."
- `LAUNCH_READINESS_ANALYSIS_2026-05-26.md` §1.7 (S34 deep-dive)
  should note that $327k of $356k NAV gain is equity beta, not
  put-premium alpha.
- The deployment matrix's reasoning column should mention
  concentration risk (BKNG-style dependence).

This soundness review doc records these clarifications. The
earlier docs are still useful but should be read alongside this
one.

---

## 5. Mechanical checks summary

| Check | Result |
|---|---|
| Launch-blocker test subset | ✅ 93/93 passed |
| 5-ticker EV smoke (CLAUDE.md §6) | ✅ 5 rows, 0 NaN |
| P&L formula consistency (532 executed rows across 4 backtests) | ✅ 0 mismatches |
| Spearman ρ statistically overwhelming | ✅ p < 1e-48 in all measurements |
| Engine refusal behavior in adverse periods | ✅ 98-99.8% refusal rate, correct in aggregate |
| Concentration risk (S34) | ⚠ BKNG drove 110% of executed alpha |
| Equity-beta vs put-selection alpha decomposition | ⚠ 92% beta / 8% selection in S34 |
| §2 invariant | ✅ Intact across all paths |
| F4 tail-risk gap | ⚠ Still open (PR #221 diagnostic shipped) |
| D17 live-wire | ⚠ Still open |
| Window sensitivity | ⚠ ±70pp swing across configurations |

**Net soundness verdict: ✅ engine is mechanically sound and
statistically defensible. ⚠ Three honest-framing items must
accompany any deployment conversation:**

1. **Equity beta dominates the dollar alpha** at scale (S34: 92%).
2. **Concentration risk** (one ticker can drive aggregate result).
3. **Window sensitivity** (S22/S27/S34 +20-30pp vs SPY in 2022-2024;
   S35 -41pp in 2018-2020).

The engine is **sound for the purposes it claims** (ranking,
refusal mechanism). The "beats SPY" framing is **conditionally
true** but **not for the reasons most readers would assume**.

---

## 6. Updates needed to prior session docs (recommended)

The earlier session docs (PR #218 PRODUCTION_READINESS, PR #225
LAUNCH_READINESS_ANALYSIS) are mechanically correct but
interpretation-incomplete. They should be augmented with the
equity-beta-vs-ranking-alpha decomposition from this review.

Three small follow-up PRs would tighten the picture:

1. **PR amending `LAUNCH_READINESS_ANALYSIS_2026-05-26.md`** —
   add a §1.9 "Where does the alpha come from?" section pointing
   out that 92% of S34's NAV gain is equity beta.
2. **PR amending `PRODUCTION_READINESS.md`** — add a "Soft caveat —
   alpha decomposition" entry in §4 noting that most dollar-alpha
   is equity beta on assignments.
3. **PR amending `ENGINE_BACKTEST_S34_UNIVERSE.md`** — add a §
   "Alpha decomposition" with the BKNG concentration finding.

These can be batched into a single docs PR. **The headlines in the
deployment matrix don't change** — the conditional verdicts
remain accurate. What changes is the explanation of WHY they hold.

---

## 7. Sources

- `docs/LAUNCH_READINESS_ANALYSIS_2026-05-26.md` (PR #225) — the
  earlier session's full analysis
- `docs/PRODUCTION_READINESS.md` (PR #218) — the gate doc
- `docs/ENGINE_BACKTEST_S34_UNIVERSE.md` (PR #226) — S34 backtest
- `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` (PR #224) — S35
  backtest
- `docs/ENGINE_BACKTEST_S32_FRICTION.md` (merged) — S32 backtest
- `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` (merged) — S27
  backtest
- `docs/F4_TAIL_RISK_DIAGNOSTIC.md` (PR #221 merged) — F4
  mechanical diagnostic
- Throwaway scripts used in this review:
  `_soundness_check.py`, `_pnl_recheck.py`, `_concentration_check.py`
  (deleted before commit per the throwaway convention).

---

## 8. Update protocol

This soundness review represents the engine's reliability picture at
SHA `2da76ff` on 2026-05-26. Re-run the soundness checks (especially
the concentration analysis) when:

- A new backtest in a different market regime completes.
- The F4 tail-risk fix lands (re-verify the engine's caution in
  bear regimes).
- The D17 live-wire ships (verify it doesn't introduce new
  bypass paths).
- The first real-money deployment is contemplated (this doc is the
  go/no-go).
